"""Real RSS parsing with offline HTTP/LLM fixtures; no claims about signal quality."""

import json
from html import escape
from pathlib import Path

import pytest

from quant_rsi.cli import main
from quant_rsi.market import news_feed
from quant_rsi.research import demo, demo_news, llm

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"
RECEIVED = "2026-10-29T12:00:00+00:00"


def item(identity, published, summary="<p>Revenue &amp; demand rose.</p>"):
    return (
        f"<item><guid>{identity}</guid><title>{identity} release</title>"
        f"<link>https://investors.coca-colacompany.com/{identity}</link>"
        f"<pubDate>{published}</pubDate><description>{escape(summary)}</description></item>"
    )


def feed(*items):
    return ("<rss><channel>" + "".join(items) + "</channel></rss>").encode()


@pytest.fixture
def transport(monkeypatch):
    calls = []
    body = feed(
        item("unknown", ""),
        item("old", "Tue, 20 Oct 2026 12:00:00 GMT"),
        item("latest", "Wed, 28 Oct 2026 12:00:00 GMT"),
        item("day-only", "2026-10-27"),
    )

    def fetch(source, limit):
        calls.append((source.instrument, limit))
        return body, {"received_at": RECEIVED, "http_status": 200}

    monkeypatch.setattr(news_feed, "fetch", fetch)
    return calls


def test_registered_feed_selection_preserves_origin_and_timestamp_precision(transport):
    news = demo_news.issuer_news({"symbol": "KO"}, limit=2)
    assert transport == [("KO", 262144)]
    assert news["provider"] == "issuer_rss" and news["scope"] == "target_issuer_feed_only"
    assert news["available_items"] == 4 and news["omitted_items"] == 2
    first, second = news["items"]
    assert first["title"] == "latest release"
    assert first["summary"] == "Revenue & demand rose."
    assert first["text"] == "latest release\nRevenue & demand rose."
    assert first["published_at"] == "2026-10-28T12:00:00+00:00"
    assert first["received_at"] == RECEIVED and first["url"].endswith("/latest")
    assert second["published_at"] is None and second["published_date"] == "2026-10-27"
    assert second["publication_precision"] == "day"
    all_news = demo_news.issuer_news({"symbol": "KO"}, limit=20)
    assert all_news["items"][-1]["publication_precision"] == "unknown"
    assert all_news["items"][-1]["published_at"] is None


def test_cli_passes_actual_news_and_feedback_into_both_llm_calls(transport, monkeypatch, capsys):
    calls = []

    def codex(self, prompt, schema):
        data = json.loads(prompt.split("\nINPUT_JSON:\n")[1])
        calls.append(data)
        news = data["context"]["news"]
        assert news["provider"] == "issuer_rss" and len(news["items"]) == 2
        assert news["items"][0]["title"] == "latest release"
        assert "timestamps" in prompt and "invented placeholders" in prompt
        ids = [news["items"][0]["id"]]
        assert ids[0] in schema["properties"]["evidence_news_ids"]["items"]["enum"]
        return json.dumps(
            {
                "hypothesis": "Explore the supplied revenue news, then revise after feedback.",
                "factor_ids": ["equity.price_mean_return_5", "equity.announced_events"]
                if data["feedback"]
                else ["equity.price_momentum_5"],
                "evidence_news_ids": ids,
                "event_signal": 0.5,
            }
        )

    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", codex)
    assert (
        main(
            [
                "research-demo",
                "ko",
                "--catalog",
                str(CATALOG),
                "--news",
                "issuer-rss",
                "--news-limit",
                "2",
                "--llm",
                "codex",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert len(transport) == 1 and len(calls) == 2
    assert calls[0]["context"]["news"] == calls[1]["context"]["news"]
    assert calls[1]["feedback"]["error"] == 4
    assert result["mode"] == "mixed"
    assert "real issuer RSS" in result["notice"] and "real LLM" in result["notice"]
    assert result["summary"]["factors_changed"]
    assert result["summary"]["improvement"] == "not_evaluated"


def test_real_news_with_simulated_llm_is_also_labeled_mixed(transport):
    result = demo.run_demo(
        "KO", catalog=lambda: demo.load_catalog(CATALOG), news=demo_news.issuer_news
    )
    assert result["mode"] == "mixed"
    assert "simulated LLM" in result["notice"] and "real issuer RSS" in result["notice"]


def test_summary_context_is_bounded_and_truncation_disclosed(monkeypatch):
    monkeypatch.setattr(
        news_feed,
        "fetch",
        lambda *args: (
            feed(item("long", "", "<p>" + "x" * 3000 + "</p>")),
            {"received_at": RECEIVED},
        ),
    )
    record = demo_news.issuer_news({"symbol": "KO"})["items"][0]
    assert len(record["summary"]) == 2000 and record["text_truncated"]


@pytest.mark.parametrize(
    "body, receipt, message",
    [
        (b"", {"error": "http_error"}, "retrieval failed"),
        (feed(), {}, "no news"),
        (b"<rss>", {}, "could not be parsed"),
    ],
)
def test_bad_news_stops_before_llm_without_backfill(monkeypatch, capsys, body, receipt, message):
    monkeypatch.setattr(news_feed, "fetch", lambda *args: (body, receipt))

    def unexpected(*args, **kwargs):
        pytest.fail("LLM should not run after a news failure")

    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", unexpected)
    assert main(["research-demo", "--news", "issuer-rss", "--llm", "codex"]) == 1
    output = capsys.readouterr()
    assert message in output.err and not output.out


def test_unknown_stock_and_invalid_limit_fail_before_network(transport):
    with pytest.raises(ValueError, match="No registered issuer RSS feed for CALM"):
        demo_news.issuer_news({"symbol": "CALM"})
    with pytest.raises(ValueError, match="between 1 and 20"):
        demo_news.issuer_news({"symbol": "KO"}, limit=21)
    assert not transport
