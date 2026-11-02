"""Recovery evidence: genuine model fits, injected provider failure, fresh runner invocation."""

import json

import lightgbm as lgb
import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study
from quant_rsi.research.study_estimators import StandardizedRidge
from quant_rsi.research.study_resume import StudyCache, run_lock


@pytest.mark.parametrize("estimator", ["lightgbm", "ridge"])
@pytest.mark.parametrize("horizon", [5, 60])
def test_resume_keeps_first_fit_and_hypothesis_and_terminal_results(
    tmp_path, monkeypatch, estimator, horizon
):
    root = tmp_path / "study"
    calls, fits = [], []
    original = study.simulated_hypothesis
    fail = True

    def hypothesis(*args, **kwargs):
        calls.append(bool(kwargs.get("previous")))
        if kwargs.get("previous") and fail:
            raise ValueError("Temporary LLM outage")
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    if estimator == "lightgbm":
        train = lgb.train

        def counted(*args, **kwargs):
            fits.append(1)
            return train(*args, **kwargs)

        monkeypatch.setattr(lgb, "train", counted)
    else:
        init = StandardizedRidge.__init__

        def counted(self, *args, **kwargs):
            fits.append(1)
            init(self, *args, **kwargs)

        monkeypatch.setattr(StandardizedRidge, "__init__", counted)
    assert (
        main(
            command(
                root,
                "--llm",
                "simulated",
                "--model",
                estimator,
                "--news-research",
                "--horizon-sessions",
                str(horizon),
            )
        )
        == 1
    )
    first = (root / "checkpoints/fit_1.json").read_bytes()
    failure = (root / "failure.json").read_bytes()
    assert calls == [False, True] and len(fits) == 1
    fail = False
    assert main(["research-resume", str(root)]) == 0
    assert calls == [False, True, True] and len(fits) == 2
    assert (root / "checkpoints/fit_1.json").read_bytes() == first
    assert (root / "attempts/1/failure.json").read_bytes() == failure
    result = json.loads((root / "result.json").read_bytes())
    assert result["terminal_evaluation"]["status"] == "completed"
    assert result["runtime"]["counts"]["model_stages_completed"] == 1
    assert result["runtime"]["previous_attempts"][0]["counts"]["model_stages_completed"] == 1
    assert result["runtime"]["reused_operations"] >= 6
    assert "Earlier attempt" in (root / "report.html").read_text()
    result_bytes = (root / "result.json").read_bytes()
    assert main(["research-resume", str(root)]) == 1
    assert (root / "result.json").read_bytes() == result_bytes


def test_corrupt_checkpoint_and_changed_configuration_fail_without_llm(tmp_path, monkeypatch):
    root = tmp_path / "study"
    monkeypatch.setattr(
        study, "simulated_hypothesis", lambda *a, **k: (_ for _ in ()).throw(ValueError("offline"))
    )
    assert main(command(root, "--llm", "simulated")) == 1
    config = json.loads((root / "config.json").read_bytes())
    original = (root / "config.json").read_bytes()
    config["rounds"] = 3
    (root / "config.json").write_text(json.dumps(config))
    assert main(["research-resume", str(root)]) == 1
    assert not (root / "attempts").exists()
    (root / "config.json").write_bytes(original)
    checkpoint = root / "checkpoints/issuer_news.json"
    value = json.loads(checkpoint.read_bytes())
    value["output"]["provider"] = "tampered"
    checkpoint.write_text(json.dumps(value))
    assert main(["research-resume", str(root)]) == 1
    assert "Checkpoint output changed" in json.loads((root / "failure.json").read_bytes())["error"]


def test_run_lock_prevents_concurrent_resume(tmp_path):
    with run_lock(tmp_path), pytest.raises(ValueError, match="already running"):
        with run_lock(tmp_path):
            pass


def test_cache_failed_operation_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_rsi.research.study_resume.binding", lambda config: config)
    cache = StudyCache(tmp_path, {})
    with pytest.raises(ValueError):
        cache.call("news", {}, lambda: (_ for _ in ()).throw(ValueError("timeout")))
    assert not (tmp_path / "checkpoints/news.json").exists()
    assert cache.call("news", {}, lambda: {"articles": []}) == {"articles": []}


def test_active_news_resume_reuses_completed_searches_and_reads(tmp_path, monkeypatch):
    from quant_rsi.research import active_news

    monkeypatch.setattr("quant_rsi.research.study_resume.binding", lambda config: config)
    cache = StudyCache(tmp_path, {})
    calls = {"search": 0, "read": 0, "review": 0}
    fail = True

    def search(symbol, related, llm, query, articles, question=""):
        assert question == "Inventory demand risk"
        calls["search"] += 1
        return active_news._simulated_search(symbol, query, 1 if query is None else 2)

    def reader(url):
        calls["read"] += 1
        return active_news._simulated_read(url)

    def review(symbol, related, articles, llm, followup_allowed, question=""):
        assert question == "Inventory demand risk"
        calls["review"] += 1
        if not followup_allowed and fail:
            raise ValueError("Review service unavailable")
        return active_news._simulated_review(symbol, articles, 1 if followup_allowed else 2)

    monkeypatch.setattr(active_news, "_search", search)
    monkeypatch.setattr(active_news, "_review", review)
    from types import SimpleNamespace

    def run():
        return active_news.research_news(
            "KO",
            {},
            SimpleNamespace(mode="codex"),
            on_stage=lambda e: None,
            reader=reader,
            cache=cache,
            question="Inventory demand risk",
        )

    with pytest.raises(ValueError, match="Review service"):
        run()
    before = dict(calls)
    fail = False
    result = run()
    assert result["articles"]
    assert calls["search"] == before["search"] == 2
    assert calls["read"] == before["read"]
    assert calls["review"] == before["review"] + 1


def test_failure_after_history_registration_can_finish_without_refitting(tmp_path, monkeypatch):
    from quant_rsi.research import study_report

    root = tmp_path / "study"
    original = study_report.render_study
    monkeypatch.setattr(
        study_report,
        "render_study",
        lambda r: (_ for _ in ()).throw(ValueError("disk interrupted")),
    )
    assert (
        main(command(root, "--llm", "simulated", "--history-dir", str(tmp_path / "history"))) == 1
    )
    first = (root / "checkpoints/fit_1.json").stat().st_mtime_ns
    monkeypatch.setattr(study_report, "render_study", original)
    assert main(["research-resume", str(root)]) == 0
    assert (root / "checkpoints/fit_1.json").stat().st_mtime_ns == first
    assert (root / "report.html").is_file()
    assert [
        json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()
    ] == json.loads((root / "result.json").read_bytes())["trace"]


def test_failed_managed_collection_retries_fresh_and_preserves_receipt(tmp_path, monkeypatch):
    from test_daily_data import chart, install

    from quant_rsi.research import demo_prices

    root = tmp_path / "study"
    raw = chart()
    raw["chart"]["result"][0]["meta"].update(
        symbol="KO", instrumentType="EQUITY", exchangeName="NYQ"
    )
    raw["chart"]["result"][0]["events"] = {}
    install(monkeypatch, data=raw)
    collect = demo_prices.collect_daily
    calls = []

    def flaky(symbol, directory):
        calls.append(directory)
        if len(calls) == 1:
            directory.mkdir()
            (directory / "collection_receipt.json").write_text('{"status":"failed"}')
            raise ValueError("Calendar transport unavailable")
        return collect(symbol, directory)

    monkeypatch.setattr(demo_prices, "collect_daily", flaky)
    assert (
        main(command(root, "--llm", "simulated", "--prices", "daily", "--no-related-prices")) == 1
    )
    original = (root / "prices/collection_receipt.json").read_bytes()
    assert main(["research-resume", str(root)]) == 0
    assert len(calls) == 2 and calls[1].parent == root / "price_retries"
    assert (root / "prices/collection_receipt.json").read_bytes() == original
