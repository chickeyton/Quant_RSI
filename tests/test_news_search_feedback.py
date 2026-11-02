"""Failure-driven discovery uses alternate sources without pretending skipped pages were read."""

import json

import pytest

from quant_rsi.research.active_news import research_news


class SearchFixture:
    mode = "codex"

    def __init__(self, urls):
        self.urls = urls
        self.search_inputs = []
        self.review_inputs = []

    def search(self, prompt, schema):
        data = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        self.search_inputs.append(data)
        assert "alternative accessible official or primary sources" in prompt
        return {
            "text": json.dumps(
                {
                    "query": data["followup_query"] or "KO initial news",
                    "rationale": "Find readable primary evidence.",
                    "results": [
                        {"url": url, "title": "Fixture news", "summary": "Search summary only"}
                        for url in self.urls[len(self.search_inputs) - 1]
                    ],
                }
            ),
            "search_actions": [{"type": "search"}],
        }

    def structured(self, prompt, schema):
        data = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        self.review_inputs.append(data)
        assert "failed reads and their reasons" in prompt
        return json.dumps(
            {
                "findings": [],
                "related_companies": [],
                "followup_query": "KO alternative primary source"
                if data["followup_allowed"]
                else None,
            }
        )


def article(url, reason=None):
    return {
        "url": url,
        "final_url": url if reason is None else None,
        "status": "available" if reason is None else "unavailable",
        "received_at": "2026-10-30T10:00:00Z",
        "text": "Actual fixture page text." if reason is None else "",
        "text_truncated": False,
        "content_sha256": "a" * 64 if reason is None else None,
        "content_extent": "fixture_page",
        **({"reason": reason} if reason else {}),
    }


@pytest.mark.parametrize("reason", ["http_401", "http_403", "http_429"])
def test_denied_hosts_guide_followup_and_are_skipped_before_network(reason):
    fixture = SearchFixture(
        [
            ["https://NEWS.example./denied", "https://primary.example/first"],
            ["https://news.example/another-page", "https://alternative.example/release"],
        ]
    )
    fetched, events = [], []

    def read(url):
        fetched.append(url)
        return article(url, reason if "denied" in url else None)

    result = research_news(
        "KO",
        {},
        fixture,
        reader=read,
        on_start=lambda stage: events.append(("start", stage)),
        on_stage=lambda event: events.append(("complete", event["stage"])),
    )
    followup = fixture.search_inputs[1]
    assert followup["avoid_hosts"] == ["news.example"]
    assert followup["previous_articles"][0]["reason"] == reason
    assert fixture.review_inputs[0]["avoid_hosts"] == ["news.example"]
    assert result["searches"][1]["avoid_hosts"] == ["news.example"]
    assert "https://news.example/another-page" not in fetched
    assert len(fetched) == 3
    skipped = result["articles"][2]
    assert skipped["status"] == "unavailable" and skipped["reason"] == "skipped_denied_host"
    assert skipped["host_failure"] == reason
    assert skipped["fetch_attempted"] is False and skipped["received_at"] is None
    assert skipped["text"] == "" and skipped["content_sha256"] is None
    assert sum(a["fetch_attempted"] for a in result["articles"]) == 3
    assert skipped["id"] not in result["evidence_news_ids"]
    assert result["status"] == "partial"
    stages = [
        "news_search_1",
        "article_reads_1",
        "news_review_1",
        "news_search_2",
        "article_reads_2",
        "news_review_2",
    ]
    assert events == [(status, stage) for stage in stages for status in ("start", "complete")]


@pytest.mark.parametrize("reason", ["http_404", "request_failed", "no_readable_text"])
def test_page_specific_or_transient_failures_do_not_suppress_other_pages_on_host(reason):
    fixture = SearchFixture([["https://primary.example/bad"], ["https://primary.example/good"]])
    fetched = []

    def read(url):
        fetched.append(url)
        return article(url, reason if url.endswith("/bad") else None)

    result = research_news("KO", {}, fixture, reader=read, on_stage=lambda _: None)
    assert len(fetched) == 2
    assert fixture.search_inputs[1]["avoid_hosts"] == []
    assert fixture.search_inputs[1]["previous_articles"][0]["reason"] == reason
    assert result["articles"][1]["status"] == "available"
    assert result["articles"][1]["fetch_attempted"] is True


def test_known_denied_redirect_host_and_same_batch_results_are_not_refetched():
    fixture = SearchFixture(
        [
            ["https://gateway.example/redirect", "https://publisher.example/second"],
            ["https://publisher.example/third", "https://primary.example/alternative"],
        ]
    )
    fetched = []

    def read(url):
        fetched.append(url)
        if "gateway" in url:
            return article(url, "http_403") | {"final_url": "https://publisher.example/denied"}
        return article(url)

    result = research_news("KO", {}, fixture, reader=read, on_stage=lambda _: None)
    assert fetched == ["https://gateway.example/redirect", "https://primary.example/alternative"]
    assert fixture.search_inputs[1]["avoid_hosts"] == ["gateway.example", "publisher.example"]
    assert [a["fetch_attempted"] for a in result["articles"]] == [True, False, False, True]


def test_start_hook_reports_failing_operation_without_a_completed_event():
    fixture = SearchFixture([["https://primary.example/story"]])
    events = []

    def read(url):
        raise ValueError("Controlled reader failure")

    with pytest.raises(ValueError, match="Controlled reader failure"):
        research_news(
            "KO",
            {},
            fixture,
            reader=read,
            on_start=lambda stage: events.append(("start", stage)),
            on_stage=lambda event: events.append(("complete", event["stage"])),
        )
    assert events == [
        ("start", "news_search_1"),
        ("complete", "news_search_1"),
        ("start", "article_reads_1"),
    ]


@pytest.mark.parametrize("initial", [[], ["https://denied.example/story"]])
def test_missing_sources_trigger_bounded_recovery_when_llm_declines_followup(initial):
    fixture = SearchFixture([initial, ["https://primary.example/release"]])
    fixture.structured = lambda *_: json.dumps(
        {"findings": [], "related_companies": [], "followup_query": None}
    )
    events = []
    result = research_news(
        "KO",
        {},
        fixture,
        reader=lambda url: article(url, "http_403" if "denied" in url else None),
        on_stage=events.append,
    )
    assert len(result["searches"]) == 2
    assert result["searches"][1]["query_origin"] == "unavailable_source_recovery"
    assert result["articles"][-1]["status"] == "available"
    review = next(e["output"] for e in events if e["stage"] == "news_review_1")
    assert review["followup_query"] is None  # Preserve what the LLM actually requested.
    assert fixture.search_inputs[1]["followup_query"] == review["recovery_query"]
    assert len(result["articles"]) <= 6


def test_readable_sources_do_not_trigger_unrequested_second_search():
    fixture = SearchFixture([["https://primary.example/release"]])
    fixture.structured = lambda *_: json.dumps(
        {"findings": [], "related_companies": [], "followup_query": None}
    )
    result = research_news("KO", {}, fixture, reader=article, on_stage=lambda _: None)
    assert len(result["searches"]) == 1
