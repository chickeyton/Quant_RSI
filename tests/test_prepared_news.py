"""Prepared RSS contexts are reused exactly, including when a study resumes."""

import json
from copy import deepcopy

import pytest
from test_research_study import command
from test_study_business import proposal as business_proposal
from test_study_news import multi
from test_study_news import snapshot as archive_snapshot

from quant_rsi.cli import main
from quant_rsi.core.artifacts import canonical
from quant_rsi.research import study, study_news


def snapshot(**kwargs):
    payload = archive_snapshot(**kwargs)
    for item in payload["items"]:
        item["text"] = item["title"] + "\n" + item["summary"]
    return payload


def test_snapshot_loader_keeps_target_peer_selection_and_original_receipts(tmp_path):
    payload = multi(snapshot(), snapshot(symbol="PEP", received="2026-10-29T15:01:00+00:00"))
    path = tmp_path / "news.json"
    path.write_bytes(canonical(payload))
    assert study_news.load_snapshot(path, "KO", limit=1) == payload
    with pytest.raises(ValueError, match="target differs"):
        study_news.load_snapshot(path, "PEP")
    with pytest.raises(ValueError, match="target differs"):
        study_news.load_snapshot(path, "AAPL")


@pytest.mark.parametrize(
    "invalid", ["receipt", "provider", "oversized", "limit", "symlink", "text", "empty_target"]
)
def test_invalid_snapshot_is_rejected_without_repair_or_truncation(tmp_path, invalid):
    payload = snapshot()
    if invalid == "receipt":
        payload["items"][0]["received_at"] = "2026-10-30T15:00:00+00:00"
    elif invalid == "provider":
        payload["provider"] = "simulated_news_archive"
    elif invalid == "limit":
        item = deepcopy(payload["items"][0])
        item.update(id="KO.2", url="https://example.com/KO/2")
        payload["items"].append(item)
    elif invalid == "text":
        payload["items"][0].pop("text")
    elif invalid == "empty_target":
        payload = multi(payload, snapshot(symbol="PEP"))
        payload["items"] = payload["items"][1:]
    path = tmp_path / "news.json"
    path.write_bytes(canonical(payload))
    if invalid == "oversized":
        path.write_bytes(b" " * (study_news.MAX_FILE_BYTES + 1))
    elif invalid == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError):
        study_news.load_snapshot(path, "KO", limit=1)


def test_prepared_news_reaches_both_rounds_and_resume_without_another_fetch(tmp_path, monkeypatch):
    path = tmp_path / "news.json"
    payload = snapshot()
    path.write_bytes(canonical(payload))
    original = study.simulated_hypothesis
    contexts, failed = [], False

    def hypothesis(context, *args, **kwargs):
        nonlocal failed
        contexts.append(deepcopy(context["news"]))
        if kwargs.get("previous") and not failed:
            failed = True
            raise ValueError("Temporary hypothesis outage")
        result = original(context, *args, **kwargs)
        business = business_proposal(revised=kwargs.get("previous") is not None)
        result["proposed_business_factors"] = [business]
        result["factor_ids"].append(f"business.{business['name']}")
        return result

    def no_fetch(*args, **kwargs):
        pytest.fail("A prepared-news study must not refetch issuer RSS")

    monkeypatch.setattr(study, "issuer_news", no_fetch)
    monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    root = tmp_path / "study"
    assert (
        main(
            command(
                root,
                "--llm",
                "simulated",
                "--financials",
                "--news",
                "issuer-rss",
                "--news-snapshot",
                str(path),
            )
        )
        == 1
    )
    first_fit = (root / "checkpoints/fit_1.json").read_bytes()
    news_checkpoint = (root / "checkpoints/issuer_news.json").read_bytes()
    path.unlink()  # Completed acquisition is retained in the checkpoint, like prices.
    assert main(["research-resume", str(root)]) == 0
    assert len(contexts) == 3 and all(value == payload for value in contexts)
    assert (root / "checkpoints/fit_1.json").read_bytes() == first_fit
    assert (root / "checkpoints/issuer_news.json").read_bytes() == news_checkpoint
    result = json.loads((root / "result.json").read_bytes())
    collected = next(
        e["output"] for e in result["runtime"]["stages"] if e["stage"] == "collect_news_context"
    )
    assert collected["news"] == payload and collected["acquisition"] == "retained_snapshot"
    assert json.loads((root / "news_history.json").read_bytes()) == [payload]
    assert result["runtime"]["counts"]["model_stages_completed"] == 1
    assert result["runtime"]["previous_attempts"][0]["counts"]["model_stages_completed"] == 1
    attempts = result["iterations"]["attempts"]
    assert "business.inventory_gap" in attempts[0]["prediction"]["factor_ids"]
    assert "business.margin_gap" in attempts[1]["prediction"]["factor_ids"]
    assert result["study"]["financial_comparison"]["status"] == "completed"


def test_snapshot_option_cannot_silently_override_simulated_provider(tmp_path, monkeypatch):
    def no_call(*args, **kwargs):
        pytest.fail("Invalid provider configuration must fail before acquisition or hypotheses")

    monkeypatch.setattr(study, "simulated_hypothesis", no_call)
    assert main(command(tmp_path / "study", "--news-snapshot", str(tmp_path / "news.json"))) == 1
    assert not (tmp_path / "study/checkpoints").exists()
