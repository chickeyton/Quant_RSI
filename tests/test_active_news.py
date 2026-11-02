"""Offline discovery orchestration, real study integration, and provider transport checks."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_rsi.cli import main
from quant_rsi.research import active_news, llm


def inputs(prompt):
    return json.loads(prompt.split("INPUT_JSON:\n", 1)[1])


def article(url):
    return {
        "url": url,
        "final_url": url,
        "received_at": "2026-10-30T06:00:00Z",
        "status": "available",
        "text": "An offline article says DEMO supplies the target, whose demand is uncertain.",
        "text_truncated": False,
        "content_sha256": "a" * 64,
        "content_extent": "offline_fixture_text",
    }


class ResearchFixture:
    mode = "codex"

    def __init__(self):
        self.search_inputs, self.review_inputs = [], []
        self.review_schemas = []

    def search(self, prompt, schema):
        data = inputs(prompt)
        self.search_inputs.append(data)
        assert "rationale" in schema["required"]
        query = data["followup_query"] or "KO earnings supplier"
        return {
            "text": json.dumps(
                {
                    "query": query,
                    "rationale": "Check demand uncertainty and supplier exposure.",
                    "results": [
                        {
                            "title": "<script>untrusted page</script>",
                            "url": f"https://example.com/article-{len(self.search_inputs)}",
                            "summary": "Search snippet is not article evidence.",
                        }
                    ],
                }
            ),
            "search_actions": [{"type": "search", "query": query}],
        }

    def structured(self, prompt, schema):
        data = inputs(prompt)
        self.review_inputs.append(data)
        self.review_schemas.append(schema)
        ids = [a["id"] for a in data["articles"] if a["status"] == "available"]
        assert "untrusted data" in prompt
        return json.dumps(
            {
                "findings": [{"text": "Demand uncertainty.", "evidence_ids": ids}] if ids else [],
                "related_companies": [
                    {
                        "symbol": "DEMO",
                        "name": "Fixture supplier",
                        "relationship": "Supplier candidate, not verified ticker identity.",
                        "evidence_ids": ids,
                    }
                ]
                if ids
                else [],
                "followup_query": "KO demand details" if data["followup_allowed"] else None,
            }
        )


def test_search_read_followup_review_binds_sources_and_preserves_configured_relationships():
    fixture, events = ResearchFixture(), []
    related = {"provider": "instrument_relationship_profile", "stocks": ["CCEP"]}
    result = active_news.research_news(
        "KO", related, fixture, on_stage=events.append, reader=article
    )
    assert len(fixture.search_inputs) == len(fixture.review_inputs) == 2
    assert fixture.search_inputs[1]["followup_query"] == "KO demand details"
    assert fixture.review_inputs[1]["followup_allowed"] is False
    for data, schema in zip(fixture.review_inputs, fixture.review_schemas, strict=True):
        ids = sorted(a["id"] for a in data["articles"] if a["status"] == "available")
        for definition in ("Finding", "RelatedCompany"):
            assert schema["$defs"][definition]["properties"]["evidence_ids"]["items"]["enum"] == ids
    assert len(events) == 6 and len(result["articles"]) == 2
    assert result["evidence_news_ids"] == result["related_candidates"][0]["evidence_ids"]
    news, enriched = active_news.attach_research({"items": []}, related, result, "KO")
    assert related == {"provider": "instrument_relationship_profile", "stocks": ["CCEP"]}
    assert enriched["stocks"] == ["CCEP"] and enriched["research_candidates"]
    assert all(item["published_at"] is None for item in news["items"])
    assert all(item["received_at"] == "2026-10-30T06:00:00Z" for item in news["items"])
    assert all("text" not in item for item in news["research"]["articles"])


def test_unread_articles_remain_missing_without_fabricated_findings():
    fixture = ResearchFixture()

    def unavailable(url):
        return article(url) | {"status": "unavailable", "reason": "http_403", "text": ""}

    result = active_news.research_news(
        "KO", {}, fixture, on_stage=lambda _: None, reader=unavailable
    )
    assert result["status"] == "unavailable" and not result["evidence_news_ids"]
    assert not result["findings"] and not result["related_candidates"]
    assert result["articles"][0]["reason"] == "http_403"
    assert result["articles"][1]["reason"] == "skipped_denied_host"
    assert result["articles"][1]["host_failure"] == "http_403"
    assert result["articles"][1]["fetch_attempted"] is False
    for schema in fixture.review_schemas:
        assert schema["properties"]["findings"]["maxItems"] == 0
        assert schema["properties"]["related_companies"]["maxItems"] == 0
    assert fixture.search_inputs[1]["followup_query"] == "KO demand details"


def test_review_schema_excludes_failed_reads_and_does_not_leak_ids_between_calls():
    fixture = ResearchFixture()
    available = article("https://example.com/read") | {"id": "read-id"}
    denied = article("https://example.com/denied") | {
        "id": "denied-id",
        "status": "unavailable",
        "reason": "http_403",
        "text": "",
    }
    active_news._review("KO", {}, [available, denied], fixture, True)
    active_news._review("KO", {}, [], fixture, True)
    active_news._review("KO", {}, [available | {"id": "next-id"}], fixture, False)
    for schema, expected in zip(
        (fixture.review_schemas[0], fixture.review_schemas[2]),
        (["read-id"], ["next-id"]),
        strict=True,
    ):
        for definition in ("Finding", "RelatedCompany"):
            assert (
                schema["$defs"][definition]["properties"]["evidence_ids"]["items"]["enum"]
                == expected
            )
        assert schema["properties"]["findings"]["maxItems"] == 3
    assert fixture.review_schemas[1]["properties"]["findings"]["maxItems"] == 0
    assert fixture.review_schemas[1]["properties"]["related_companies"]["maxItems"] == 0


@pytest.mark.parametrize("field", ["findings", "related_companies"])
@pytest.mark.parametrize("readable", [True, False])
def test_review_still_rejects_unread_citations_when_provider_ignores_schema(field, readable):
    available = article("https://example.com/read") | {"id": "read-id"}
    denied = available | {"id": "denied-id", "status": "unavailable", "text": ""}
    cited = {"evidence_ids": ["denied-id"]}
    cited |= (
        {"text": "Claim"}
        if field == "findings"
        else {"symbol": "DEMO", "name": "Supplier", "relationship": "Supplies the target"}
    )
    fixture = SimpleNamespace(
        structured=lambda prompt, schema: json.dumps(
            {"findings": [], "related_companies": [], "followup_query": None, field: [cited]}
        )
    )
    with pytest.raises(ValueError, match="cited an unavailable or unknown article"):
        active_news._review("KO", {}, [available, denied] if readable else [denied], fixture, True)


@pytest.mark.parametrize("failure", ["unknown_citation", "self_relation", "extra_followup"])
def test_review_cannot_invent_evidence_or_extend_the_search_loop(failure):
    fixture = ResearchFixture()
    original = fixture.structured

    def invalid(prompt, schema):
        result = json.loads(original(prompt, schema))
        if failure == "unknown_citation":
            result["findings"][0]["evidence_ids"] = ["invented"]
        elif failure == "self_relation":
            result["related_companies"][0]["symbol"] = "KO"
        else:
            result["followup_query"] = "a third query"
        return json.dumps(result)

    fixture.structured = invalid
    with pytest.raises(ValueError, match="cited|exclude the target|one follow-up"):
        active_news.research_news("KO", {}, fixture, on_stage=lambda _: None, reader=article)


def command(root, mode):
    return [
        "research-study",
        "KO",
        "--output",
        str(root),
        "--llm",
        mode,
        "--related",
        "simulated",
        "--news",
        "simulated",
        "--prices",
        "synthetic",
        "--news-research",
    ]


def test_active_offline_study_executes_volatility_and_keeps_articles_out_of_training_archive(
    tmp_path,
):
    root = tmp_path / "study"
    assert main(command(root, "simulated")) == 0
    result = json.loads((root / "result.json").read_text())
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    discovery = json.loads((root / "news_research.json").read_text())
    decisions = [event for event in result["trace"] if event["stage"].endswith("_decision")]
    assert [event["stage"] for event in decisions] == ["round_1_decision", "round_2_decision"]
    assert [event["output"]["completed_fits"] for event in decisions] == [1, 2]
    assert result["trace"][0]["stage"] == "extract_guidance"
    assert steps["extract_guidance"]["status"] == "simulated_historical_extractions"
    assert len(result["trace"]) - len(decisions) == 22
    assert result["trace"][-2]["stage"] == "terminal_evaluation"
    assert [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()] == result[
        "trace"
    ]
    assert "research.log_return_volatility_10" in steps["second_prediction"]["factor_ids"]
    assert steps["second_prediction"]["training_feature_columns"] == 3
    for key in ("initial_hypothesis", "revised_hypothesis"):
        assert set(steps[key]["evidence_news_ids"]) <= set(discovery["evidence_news_ids"])
    history = (root / "news_history.json").read_text()
    assert "research-news-" not in history and "/article-" not in history
    assert len(json.loads(history)) == 128
    for filename in ("report.html", "report.md"):
        report = (root / filename).read_text()
        assert "Suggested related companies" in report and "SIMPEER" in report
        assert discovery["evidence_news_ids"][0] in report
        assert "Active news research" in report and "Sharpe (ann.)" in report
    assert "2 search requests, 2/2 articles read" in result["study"]["conclusion"]
    assert "Sharpe None" not in result["study"]["conclusion"]


def test_codex_study_passes_read_evidence_and_relationships_into_both_hypotheses(
    tmp_path, monkeypatch
):
    fixture, hypotheses = ResearchFixture(), []
    monkeypatch.setattr(llm.LLMHypothesizer, "search", lambda _, *a: fixture.search(*a))
    monkeypatch.setattr(llm.LLMHypothesizer, "structured", lambda _, *a: fixture.structured(*a))
    monkeypatch.setattr(active_news, "read_article", article)

    def hypothesis(self, prompt, schema):
        data = inputs(prompt)
        hypotheses.append(data)
        assert data["context"]["related"]["research_candidates"][0]["symbol"] == "DEMO"
        ids = data["context"]["news"]["research_evidence_ids"]
        op = "log_return_volatility" if data["feedback"] else "momentum"
        return json.dumps(
            {
                "hypothesis": "Read demand uncertainty motivates this bounded price recipe.",
                "factor_ids": [f"research.{op}_10"],
                "evidence_news_ids": [ids[-1]],
                "event_signal": 0.0,
                "proposed_factors": [{"op": op, "window": 10, "description": "Demand uncertainty"}],
                "proposed_news_factors": [
                    {
                        "name": "demand",
                        "keywords": ["demand"],
                        "window_days": 7,
                        "description": "Retained headlines only",
                    }
                ],
            }
        )

    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", hypothesis)
    root = tmp_path / "connected"
    assert main(command(root, "codex")) == 0
    assert len(hypotheses) == 2
    assert hypotheses[1]["feedback"]["strategy_diagnostics"]["strategy"]["risk_statistics"]
    html = (root / "report.html").read_text()
    assert "<script>untrusted page</script>" not in html
    assert "&lt;script&gt;untrusted page&lt;/script&gt;" in html
    assert "https://example.com/article-2" in html


def test_discovery_failure_preserves_completed_search_and_read_stages(tmp_path, monkeypatch):
    fixture = ResearchFixture()
    monkeypatch.setattr(llm.LLMHypothesizer, "search", lambda _, *a: fixture.search(*a))
    monkeypatch.setattr(llm.LLMHypothesizer, "structured", lambda *_: "invalid JSON")
    monkeypatch.setattr(active_news, "read_article", article)
    root = tmp_path / "failed"
    assert main(command(root, "codex")) == 1
    failure = json.loads((root / "failure.json").read_text())
    assert failure["last_completed_stage"] == "article_reads_1"
    completed = [
        json.loads(line)["stage"] for line in (root / "trace.jsonl").read_text().splitlines()
    ]
    assert completed == ["extract_guidance", "news_search_1", "article_reads_1"]
    assert not (root / "report.html").exists()


def test_hypothesis_must_use_a_retrieved_article_when_discovery_succeeds(tmp_path, monkeypatch):
    fixture = ResearchFixture()
    monkeypatch.setattr(llm.LLMHypothesizer, "search", lambda _, *a: fixture.search(*a))
    monkeypatch.setattr(llm.LLMHypothesizer, "structured", lambda _, *a: fixture.structured(*a))
    monkeypatch.setattr(active_news, "read_article", article)

    def ignore_discovery(self, prompt, schema):
        data = inputs(prompt)
        return json.dumps(
            {
                "hypothesis": "Only considering original news, ignoring discovered articles.",
                "factor_ids": ["research.momentum_10"],
                "evidence_news_ids": [data["context"]["news"]["items"][0]["id"]],
                "event_signal": 0.0,
                "proposed_factors": [{"op": "momentum", "window": 10, "description": "Momentum"}],
                "proposed_news_factors": [
                    {
                        "name": "demand",
                        "keywords": ["demand"],
                        "window_days": 7,
                        "description": "Retained headlines",
                    }
                ],
            }
        )

    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", ignore_discovery)
    root = tmp_path / "ignored"
    assert main(command(root, "codex")) == 1
    failure = json.loads((root / "failure.json").read_text())
    assert "must cite a retrieved article" in failure["error"]
    assert not (root / "report.html").exists()


def test_codex_discovery_enables_only_search_and_retains_only_tool_metadata(monkeypatch):
    def run(command, **kwargs):
        assert 'web_search="live"' in command and "--json" in command
        assert "shell_tool" in command and "apps" in command and "multi_agent" in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        Path(command[command.index("--output-last-message") + 1]).write_text('{"test":true}')
        for item in (
            {"type": "reasoning", "text": "must not enter research evidence"},
            {"type": "web_search", "action": {"type": "search", "query": "KO earnings"}},
        ):
            kwargs["stdout"].write(json.dumps({"type": "item.completed", "item": item}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(llm, "run_provider", run)
    result = llm.LLMHypothesizer("codex").search("research", {})
    assert result["text"] == '{"test":true}'
    assert result["search_actions"] == [
        {"type": "search", "query": "KO earnings", "source_urls": []}
    ]


@pytest.mark.parametrize("searched", [True, False, "open_only"])
def test_api_discovery_requires_recorded_search_and_uses_existing_key(monkeypatch, searched):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")

    def urlopen(request, timeout):
        body = json.loads(request.data)
        assert body["tools"] == [{"type": "web_search"}]
        assert body["max_tool_calls"] == 2 and body["tool_choice"] == "required"
        assert body["store"] is False and body["text"]["format"]["strict"] is True
        assert body["include"] == ["web_search_call.action.sources"]
        items = [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]
        if searched:
            items.append(
                {
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "open_page" if searched == "open_only" else "search",
                        "queries": ["KO earnings"],
                        "sources": [{"url": "https://example.com/earnings"}],
                    },
                }
            )
        return io.BytesIO(json.dumps({"status": "completed", "output": items}).encode())

    monkeypatch.setattr(llm, "urlopen", urlopen)
    provider = llm.LLMHypothesizer("api", "test-model")
    if searched is True:
        result = provider.search("research", {})
        assert result["search_actions"][0]["source_urls"] == ["https://example.com/earnings"]
        assert "test-secret" not in json.dumps(result)
    else:
        with pytest.raises(ValueError, match="without a recorded web search"):
            provider.search("research", {})
