"""Offline integration of the versioned news recipe with the original worker ledger."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_forecast_comparison import dummy_worker  # noqa: F401
from test_forecast_features import prepare_inputs  # noqa: F401

from quant_rsi.core.artifacts import canonical
from quant_rsi.core.journal import Journal
from quant_rsi.forecast.forecast_experiment import (
    experiment_root,
    inspect_experiment,
    open_experiment,
    prepare_experiment,
    resume_experiment,
)
from quant_rsi.forecast.forecast_features import ordered_feature_values
from quant_rsi.forecast.forecast_inputs import ForecastRecipe
from quant_rsi.forecast.forecast_protocol import comparison_recipe


@pytest.fixture
def news_inputs(request, monkeypatch):
    inputs, envelope = request.getfixturevalue("prepare_inputs")(future_calendar=5)
    calls = []

    def scalar(inputs, origin, as_of):
        calls.append(origin)
        missing = origin == inputs.target[0].dates[85]
        return dict(
            value=None if missing else 2,
            reasons=["news_coverage_partial"] if missing else [],
            source_refs=[],
        )

    # The extractor has its own evidence tests. This fixture isolates the worker interface.
    monkeypatch.setitem(
        sys.modules,
        "quant_rsi.factors.news",
        SimpleNamespace(
            NEWS_FEATURE_DEFINITION=dict(
                name="issuer_earnings_release_count_7d",
                unit="event_count",
                window_days=7,
                transformation="retained_title_rule_v1",
            ),
            news_scalar=scalar,
            authorize_news_comparison=lambda *args, **kwargs: {"isolated_protocol_fixture": True},
        ),
    )
    prepared = prepare_experiment(
        envelope, "news-pair", fold_origins=((80, 85, 90),), news_factor=True
    )
    return inputs, envelope, prepared, calls


def test_news_protocol_freezes_ordered_scalar_shared_exclusions_and_equal_allowances(news_inputs):
    _, _, prepared, _ = news_inputs
    journal, store, _, plan, inputs = open_experiment(experiment_root("news-pair"))
    try:
        assert plan.schema_version == 2
        assert [p.recipe for p in plan.proposals] == ["B0", "N1", None]
        assert prepared["budget"] == dict(
            max_attempts=3, worker_timeout_seconds=180, max_worker_seconds=540
        )
        protocol = store.json(plan.protocol_ref)
        assert protocol["schema_version"] == 2
        mask = store.json(protocol["folds"][0]["evaluation_mask_ref"])
        assert mask["eligible_indices"] == [80, 90]
        assert mask["rows"][1]["reasons"] == ["news_coverage_partial"]
        recipe = comparison_recipe("N1")
        view = store.json(mask["rows"][0]["feature_view_refs"][recipe.ref])
        assert ordered_feature_values(view, recipe, input_ref=inputs.ref)[-1] == 2
        assert view["feature_units"] == ["dimensionless_return"] * 3 + ["event_count"]
        changed = deepcopy(view)
        changed["feature_units"][-1] = "dimensionless_return"
        with pytest.raises(ValueError, match="binding mismatch"):
            ordered_feature_values(changed, recipe, input_ref=inputs.ref)
        changed_plan = plan.model_dump(mode="json")
        changed_plan["schema_version"] = 1
        with pytest.raises(ValueError):
            type(plan).model_validate_json(canonical(changed_plan))
    finally:
        journal.close()


def test_news_pair_uses_original_worker_and_terminal_replay_never_refits(news_inputs, request):
    worker_state = request.getfixturevalue("dummy_worker")
    _, envelope, prepared, calls = news_inputs
    count = len(calls)
    result = resume_experiment("news-pair", protocol_ref=prepared["protocol_ref"], max_fits=3)
    assert worker_state["fits"] == 3
    assert result["reserved_worker_seconds"] == 540
    assert set(result["selection"]["candidates"]) == {"B0", "N1"}
    assert result["selection"]["selected_recipe"] == "B0"
    assert result == inspect_experiment("news-pair", replay=True)
    assert result == resume_experiment(
        "news-pair", protocol_ref=prepared["protocol_ref"], max_fits=3
    )
    assert result == prepare_experiment(
        envelope, "news-pair", fold_origins=((80, 85, 90),), news_factor=True
    )
    assert len(calls) == count and worker_state["fits"] == 3
    with pytest.raises(ValueError, match="different inputs/protocol"):
        prepare_experiment(envelope, "news-pair", fold_origins=((80, 85, 90),))


def test_news_interrupted_completion_recovers_under_original_charge(
    news_inputs, request, monkeypatch
):
    worker_state = request.getfixturevalue("dummy_worker")
    _, _, prepared, _ = news_inputs
    finish = Journal.finish
    once = [True]

    def crash(self, ordinal, status, **kwargs):
        if once[0] and status == "completed":
            once[0] = False
            raise RuntimeError("before commit")
        return finish(self, ordinal, status, **kwargs)

    monkeypatch.setattr(Journal, "finish", crash)
    with pytest.raises(RuntimeError, match="before commit"):
        resume_experiment("news-pair", protocol_ref=prepared["protocol_ref"])
    result = resume_experiment("news-pair", protocol_ref=prepared["protocol_ref"], max_fits=3)
    assert worker_state["launches"] == [0, 1, 2]
    assert result["reserved_worker_seconds"] == 540
    assert result["attempts"][0]["status"] == "completed"


def test_legacy_recipe_serialization_unchanged_and_news_version_is_explicit():
    legacy = comparison_recipe("B0")
    assert legacy.model_dump(mode="json") == dict(
        schema_version=1, kind="forecast_feature_recipe", group="B0"
    )
    assert "news_factor" not in legacy.definition
    with pytest.raises(ValueError, match="news recipe requires"):
        ForecastRecipe(schema_version=1, kind="forecast_feature_recipe", group="N1")


@pytest.mark.parametrize("real_model", [False, True])
def test_retained_search_extraction_features_and_original_worker_end_to_end(
    request, tmp_path, monkeypatch, real_model
):
    """Invented prospective receipts test plumbing, not historical predictive quality."""
    from datetime import timedelta

    from quant_rsi.core.artifacts import ArtifactStore, digest
    from quant_rsi.factors import news
    from quant_rsi.forecast.forecast_inputs import ForecastInputs, SignalSnapshot
    from quant_rsi.market.market_contracts import utc_text

    if not real_model:
        worker_state = request.getfixturevalue("dummy_worker")
    inputs, envelope = request.getfixturevalue("prepare_inputs")(
        future_calendar=5, target_instrument="KO"
    )
    store = ArtifactStore(tmp_path / "news_artifacts")
    from quant_rsi.market import market_contracts

    trusted_clock = market_contracts.trusted_now
    for index in range(60, 100):
        boundary = inputs.target[0].decision_at(index)
        cutoff = boundary - timedelta(minutes=1)
        start = cutoff - timedelta(days=7)
        published = cutoff - timedelta(minutes=1)
        title = f"The Coca-Cola Company Reports First Quarter {2000 + index} Results"
        evidence = {}

        def put(value, evidence=evidence):
            raw = canonical(value)
            evidence[digest(raw)] = raw
            return digest(raw)

        query_ref = put(dict(kind="invented_prospective_retained_corpus"))
        receipt_ref = put(dict(received_at=utc_text(boundary), scope="fixture_corpus_only"))
        payload_ref = put(dict(title=title))
        snapshot = SignalSnapshot.model_validate_json(
            canonical(
                dict(
                    schema_version=1,
                    kind="forecast_signal_snapshot",
                    events=[
                        dict(
                            schema_version=1,
                            kind="news",
                            source_id=news.SOURCE,
                            entity="KO",
                            event_id=f"fixture_release_{index}",
                            revision=0,
                            query_ref=query_ref,
                            published_at=utc_text(published),
                            published_date=None,
                            publication_precision="second",
                            source_available_at=None,
                            first_seen_at=utc_text(published),
                            report_period_end=None,
                            payload_ref=payload_ref,
                            receipt_ref=receipt_ref,
                            extractor_version="fixture_v1",
                            units=dict(title="text"),
                            fields=dict(title=title),
                        )
                    ],
                    coverage=[
                        dict(
                            schema_version=1,
                            source_id=news.SOURCE,
                            entity="KO",
                            query_ref=query_ref,
                            start_at=utc_text(start),
                            end_at=utc_text(cutoff),
                            first_seen_at=utc_text(cutoff),
                            status="complete",
                            reason=None,
                            receipt_ref=receipt_ref,
                        )
                    ],
                )
            )
        )
        req = news.NewsRequest(
            request_id=f"observation_{index}",
            campaign_id="prospective_fixture",
            start_at=start,
            end_at=cutoff,
            as_of=cutoff,
        )
        clock = [cutoff + timedelta(seconds=10)]

        def advancing_clock(clock=clock):
            clock[0] += timedelta(milliseconds=100)
            return utc_text(clock[0])

        monkeypatch.setattr(market_contracts, "trusted_now", advancing_clock)
        result = news.run_news(req, snapshot, evidence, store)
        assert result["usage"]["network_requests"] == 0
        extraction_ref = result["snapshot"]["events"][0]["fields"]["extraction_ref"]
        extraction = news.Extraction.model_validate_json(
            bytes.fromhex(result["evidence_hex"][extraction_ref])
        )
        assert cutoff < extraction.completed_at < boundary
        envelope = news.attach_news_result(envelope, result)
    monkeypatch.setattr(market_contracts, "trusted_now", trusted_clock)
    from quant_rsi.forecast import forecast_experiment

    # A crash after the campaign claim cannot mint a fresh allowance under a new ID.
    freeze = forecast_experiment.freeze_protocol

    def interrupted_freeze(*args, **kwargs):
        raise RuntimeError("interrupted after campaign claim")

    monkeypatch.setattr(forecast_experiment, "freeze_protocol", interrupted_freeze)
    with pytest.raises(RuntimeError, match="after campaign claim"):
        prepare_experiment(
            envelope, "real-news-pair", fold_origins=((80, 85, 90),), news_factor=True
        )
    monkeypatch.setattr(forecast_experiment, "freeze_protocol", freeze)
    with pytest.raises(ValueError, match="single comparison allowance"):
        prepare_experiment(
            envelope, "renamed-news-pair", fold_origins=((80, 85, 90),), news_factor=True
        )
    with pytest.raises(ValueError, match="single comparison allowance"):
        prepare_experiment(envelope, "real-news-pair", fold_origins=((80, 85),), news_factor=True)
    changed = deepcopy(envelope)
    extra = b"A changed input must not renew an allowance"
    changed["evidence_hex"][digest(extra)] = extra.hex()
    with pytest.raises(ValueError, match="single comparison allowance"):
        prepare_experiment(
            changed, "real-news-pair", fold_origins=((80, 85, 90),), news_factor=True
        )
    prepared = prepare_experiment(
        envelope, "real-news-pair", fold_origins=((80, 85, 90),), news_factor=True
    )
    assert prepared["news_accounting"]["fit_allowance_policy"] == "single_paired_comparison"
    assert prepared["news_accounting"]["cumulative_usage"]["extractions"] == 40
    # A later search is charged to the campaign but cannot invalidate the frozen
    # evidence set during prepare, resume, or terminal verification/replay.
    later_request = req.model_copy(update={"request_id": "after_comparison_freeze"})
    later = news.run_news(later_request, snapshot, evidence, store)
    assert later["query_ref"] not in prepared["news_accounting"]["searches"]
    assert later["cumulative_usage"]["extractions"] == 41
    assert prepared == prepare_experiment(
        envelope, "real-news-pair", fold_origins=((80, 85, 90),), news_factor=True
    )
    result = resume_experiment("real-news-pair", protocol_ref=prepared["protocol_ref"], max_fits=3)
    if not real_model:
        assert worker_state["fits"] == 3
    assert result["reserved_worker_seconds"] == 540
    assert set(result["selection"]["candidates"]) == {"B0", "N1"}
    journal, artifacts, _, plan, loaded = open_experiment(experiment_root("real-news-pair"))
    try:
        assert loaded.ref == ForecastInputs(envelope).ref
        news_fit = artifacts.json(result["attempts"][1]["result_ref"])
        metadata = artifacts.json(news_fit["worker_metadata_ref"])
        assert metadata["sidecar_inputs_used_in_training"] == ["news"]
        assert metadata["feature_names"][-1] == news.NEWS_FEATURE_DEFINITION["name"]
        assert len(metadata["feature_names"]) == 4
        baseline_fit = artifacts.json(result["attempts"][0]["result_ref"])
        assert len(artifacts.json(baseline_fit["worker_metadata_ref"])["feature_names"]) == 3
        report = artifacts.json(news_fit["report_ref"])
        assert report["sample_coverage"]["evaluation"]["eligible"] == 3
        assert set(report["naive_baselines"]) == {"unchanged_price", "training_mature_label_mean"}
        protocol = artifacts.json(plan.protocol_ref)
        mask = artifacts.json(protocol["folds"][0]["evaluation_mask_ref"])
        view = artifacts.json(mask["rows"][0]["feature_view_refs"][comparison_recipe("N1").ref])
        assert view["values"][-1] > 0 and view["source_refs"]
    finally:
        journal.close()
    assert result == inspect_experiment("real-news-pair", replay=True)
    if not real_model:
        from quant_rsi.core.artifacts import atomic_write, decode
        from quant_rsi.evaluation.evidence import authority_root

        campaign_path = authority_root() / "news_campaigns/prospective_fixture/news_campaign.json"
        saved = campaign_path.read_bytes()
        missing_claim = decode(saved)
        missing_claim.pop("comparison")
        missing_claim.pop("comparison_receipt")
        atomic_write(campaign_path, canonical(missing_claim))
        try:
            with pytest.raises(ValueError, match="allowance has not been registered"):
                resume_experiment("real-news-pair", protocol_ref=prepared["protocol_ref"])
            # Read-only resume validation must not silently reclaim the missing allowance.
            assert "comparison" not in decode(campaign_path.read_bytes())
        finally:
            atomic_write(campaign_path, saved)
    assert result == resume_experiment(
        "real-news-pair", protocol_ref=prepared["protocol_ref"], max_fits=3
    )
    if not real_model:
        assert worker_state["fits"] == 3


def test_newly_observed_corpus_seals_insufficient_evidence_without_retrying(
    request, tmp_path, monkeypatch
):
    from datetime import timedelta

    from quant_rsi.core.artifacts import ArtifactStore, digest
    from quant_rsi.factors import news
    from quant_rsi.forecast import forecast_experiment
    from quant_rsi.forecast.forecast_inputs import SignalSnapshot
    from quant_rsi.market import market_contracts
    from quant_rsi.market.market_contracts import timestamp, utc_text

    inputs, envelope = request.getfixturevalue("prepare_inputs")(
        future_calendar=5, target_instrument="KO"
    )
    now = timestamp(market_contracts.trusted_now())
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: utc_text(now))
    start = inputs.target[0].decision_at(0) - timedelta(days=7)
    raw = canonical(dict(kind="invented_empty_corpus_observed_today"))
    ref = digest(raw)
    snapshot = SignalSnapshot.model_validate_json(
        canonical(
            dict(
                schema_version=1,
                kind="forecast_signal_snapshot",
                events=[],
                coverage=[
                    dict(
                        schema_version=1,
                        source_id=news.SOURCE,
                        entity="KO",
                        query_ref=ref,
                        start_at=utc_text(start),
                        end_at=utc_text(now),
                        first_seen_at=utc_text(now),
                        status="complete",
                        reason=None,
                        receipt_ref=ref,
                    )
                ],
            )
        )
    )
    query = news.NewsRequest(
        request_id="recent_corpus",
        campaign_id="insufficient_fixture",
        start_at=start,
        end_at=now,
        as_of=now,
    )
    result = news.run_news(query, snapshot, {ref: raw}, ArtifactStore(tmp_path / "empty_news"))
    envelope = news.attach_news_result(envelope, result)
    outcome = prepare_experiment(
        envelope, "insufficient-news", fold_origins=((80, 85, 90),), news_factor=True
    )
    assert outcome["status"] == "insufficient_evidence"
    assert outcome["reason"] == "insufficient common mature training rows"
    assert outcome["fit_reservations"] == outcome["reserved_worker_seconds"] == 0
    assert outcome["protocol_ref"] is None
    assert outcome["sample_coverage"]["training"]["eligible"] == 0
    assert outcome["sample_coverage"]["training"]["exclusions"]

    def must_not_prepare_again(*args, **kwargs):
        pytest.fail("terminal insufficient evidence was recomputed")

    monkeypatch.setattr(forecast_experiment, "freeze_protocol", must_not_prepare_again)
    assert outcome == prepare_experiment(
        envelope, "insufficient-news", fold_origins=((80, 85, 90),), news_factor=True
    )
    assert outcome == inspect_experiment("insufficient-news", replay=True)
    with pytest.raises(ValueError, match="no fitted protocol"):
        resume_experiment("insufficient-news", protocol_ref="0" * 64)
    with pytest.raises(ValueError, match="single comparison allowance"):
        prepare_experiment(envelope, "retry-news", fold_origins=((80, 85, 90),), news_factor=True)
    assert not (experiment_root("insufficient-news") / "journal.sqlite").exists()


def test_whole_query_omission_rejected_and_authoritative_results_frozen(
    request, tmp_path, monkeypatch
):
    from test_news_factor import END, corpus, snapshot
    from test_news_factor import request as news_request

    from quant_rsi.core.artifacts import ArtifactStore
    from quant_rsi.factors import news
    from quant_rsi.forecast.forecast_inputs import ForecastInputs
    from quant_rsi.market import market_contracts

    _, envelope = request.getfixturevalue("prepare_inputs")(
        future_calendar=5, target_instrument="KO"
    )
    clock = [END]
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: clock[0])
    store = ArtifactStore(tmp_path / "omission_artifacts")
    value, evidence = corpus()
    first = news.run_news(news_request(), snapshot(value), evidence, store)
    first_only = news.attach_news_result(envelope, first)
    clock[0] = "2025-07-08T00:01:00Z"
    partial, partial_evidence = corpus(coverage="partial")
    second = news.run_news(
        news_request(request_id="partial"), snapshot(partial), partial_evidence, store
    )
    complete = news.attach_news_result(first_only, second)
    assert news.news_scalar(ForecastInputs(first_only), None, clock[0])["value"] == 1.0
    assert news.news_scalar(ForecastInputs(complete), None, clock[0])["reasons"] == [
        "news_coverage_partial"
    ]
    with pytest.raises(ValueError, match="coverage differs"):
        news.authorize_news_comparison(ForecastInputs(first_only), "omitted", ((80, 85, 90),))
    receipt = news.authorize_news_comparison(ForecastInputs(complete), "frozen", ((80, 85, 90),))
    assert set(receipt["searches"]) == {first["query_ref"], second["query_ref"]}
    assert all(store.json(ref)["query_ref"] == query for query, ref in receipt["searches"].items())
    later = news.run_news(news_request(request_id="later"), snapshot(value), evidence, store)
    assert later["query_ref"] not in receipt["searches"]
    assert receipt == news.authorize_news_comparison(
        ForecastInputs(complete), "frozen", ((80, 85, 90),), verify_only=True
    )
    assert receipt == news.authorize_news_comparison(
        ForecastInputs(complete), "frozen", ((80, 85, 90),)
    )

    # Even an empty later result must not silently expand the frozen state while
    # the receipt retained by the experiment still names the original searches.
    from quant_rsi.core.artifacts import atomic_write, decode
    from quant_rsi.evaluation.evidence import authority_root

    empty = deepcopy(value)
    empty.update(events=[], coverage=[])
    extra = news.run_news(news_request(request_id="later_empty"), snapshot(empty), evidence, store)
    path = authority_root() / "news_campaigns/news_campaign/news_campaign.json"
    saved = path.read_bytes()
    state = decode(saved)
    state["comparison_searches"][extra["query_ref"]] = state["runs"]["later_empty"]["result_ref"]
    atomic_write(path, canonical(state))
    with pytest.raises(ValueError, match="receipt differs"):
        news.authorize_news_comparison(
            ForecastInputs(complete), "frozen", ((80, 85, 90),), verify_only=True
        )
    state = decode(saved)
    state["comparison_receipt"]["comparison"]["experiment_id"] = "changed"
    atomic_write(path, canonical(state))
    with pytest.raises(ValueError, match="receipt differs"):
        news.authorize_news_comparison(
            ForecastInputs(complete), "frozen", ((80, 85, 90),), verify_only=True
        )
    state = decode(saved)
    state.pop("comparison_searches")
    atomic_write(path, canonical(state))
    with pytest.raises(ValueError, match="frozen searches are missing"):
        news.authorize_news_comparison(
            ForecastInputs(complete), "frozen", ((80, 85, 90),), verify_only=True
        )
    atomic_write(path, saved)


def test_sec_sidecar_preserved_while_missing_and_forged_news_rejected(
    request, tmp_path, monkeypatch
):
    from test_news_factor import END, corpus, snapshot
    from test_news_factor import request as news_request

    from quant_rsi.core.artifacts import ArtifactStore, digest
    from quant_rsi.factors import news
    from quant_rsi.forecast.forecast_inputs import ForecastInputs
    from quant_rsi.market import market_contracts

    _, envelope = request.getfixturevalue("prepare_inputs")(
        future_calendar=5, target_instrument="KO"
    )
    sec, sec_evidence = corpus()
    sec["events"][0].update(
        kind="fundamental",
        source_id="sec_companyfacts",
        report_period_end="2025-06-30",
        fields={"revenue": "100"},
        units={"revenue": "USD"},
    )
    sec["coverage"][0]["source_id"] = "sec_companyfacts"
    envelope["signals"] = sec
    envelope["evidence_hex"].update({ref: raw.hex() for ref, raw in sec_evidence.items()})
    prior = ForecastInputs(envelope).signals.model_dump(mode="json")
    value, evidence = corpus()
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: END)
    result = news.run_news(
        news_request(), snapshot(value), evidence, ArtifactStore(tmp_path / "sidecar_artifacts")
    )
    attached = news.attach_news_result(envelope, result)
    assert attached["signals"]["events"][0] == prior["events"][0]
    assert attached["signals"]["coverage"][0] == prior["coverage"][0]
    for mutation in ("missing", "forged"):
        changed = deepcopy(attached)
        if mutation == "missing":
            changed["signals"]["events"] = [changed["signals"]["events"][0]]
        else:
            changed["signals"]["events"][1]["fields"]["title"] = "Forged title"
        with pytest.raises(ValueError, match="evidence differs"):
            news.authorize_news_comparison(ForecastInputs(changed), "bad", ((80, 85, 90),))
    forged = deepcopy(attached)
    raw = canonical({"campaign_id": "news_campaign"})
    forged["evidence_hex"][digest(raw)] = raw.hex()
    forged["signals"]["events"][1]["query_ref"] = digest(raw)
    with pytest.raises(ValueError, match="not a completed registered search"):
        news.authorize_news_comparison(ForecastInputs(forged), "bad", ((80, 85, 90),))
    prepared = prepare_experiment(
        attached, "with-sec-sidecar", fold_origins=((80, 85, 90),), news_factor=True
    )
    assert prepared["news_accounting"]["fit_allowance_policy"] == "single_paired_comparison"
    assert prepared == inspect_experiment("with-sec-sidecar", replay=True)
