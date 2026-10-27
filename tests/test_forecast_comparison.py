import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_forecast_features import prepare_inputs  # noqa: F401

from quant_rsi import controller, worker
from quant_rsi.artifacts import canonical, decode, digest
from quant_rsi.forecast_experiment import (
    experiment_root,
    inspect_experiment,
    open_experiment,
    prepare_experiment,
    resume_experiment,
)
from quant_rsi.forecast_protocol import DEFAULT_FOLDS, ForecastComparisonPlan
from quant_rsi.forecast_publication import confirmed_report
from quant_rsi.journal import Journal
from quant_rsi.market_contracts import utc_text


@pytest.fixture
def experiment(request):
    context_factory = request.getfixturevalue("prepare_inputs")
    inputs, envelope = context_factory(future_calendar=5)
    result = prepare_experiment(envelope, "engineering-context", fold_origins=((80, 85, 90),))
    return inputs, envelope, result


@pytest.fixture
def dummy_worker(monkeypatch):
    """Exercise actual worker/controller dispatch without fitting a real model."""
    state = {"fits": 0, "launches": [], "fail": set()}

    class Dataset:
        def __init__(self, x, label, feature_name):
            self.x, self.y, self.names = x, label, feature_name

    class Booster:
        def __init__(self, *, model_str):
            self.value = json.loads(model_str)

        def num_feature(self):
            return len(self.value["names"])

        def feature_name(self):
            return self.value["names"]

        def predict(self, x, *, num_threads):
            assert num_threads == 1 and x.shape[1] == self.num_feature()
            return np.full(len(x), self.value["mean"])

        def model_to_string(self):
            return json.dumps(self.value, sort_keys=True)

    def train(params, dataset, *, num_boost_round):
        state["fits"] += 1
        assert num_boost_round == 24 and params["num_threads"] == 1
        return Booster(
            model_str=json.dumps(
                dict(
                    names=dataset.names,
                    mean=float(np.mean(dataset.y)),
                    rows=len(dataset.y),
                )
            )
        )

    monkeypatch.setitem(
        sys.modules, "lightgbm", SimpleNamespace(Dataset=Dataset, Booster=Booster, train=train)
    )
    monkeypatch.setattr(worker.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker.signal, "setitimer", lambda *args: None)

    def run(command, **kwargs):
        job = decode(__import__("pathlib").Path(command[-1]).read_bytes())
        ordinal = job["binding"]["ordinal"]
        state["launches"].append(ordinal)
        if ordinal in state["fail"]:
            return SimpleNamespace(returncode=1)
        monkeypatch.setattr(sys, "argv", ["worker", command[-1]])
        worker.main()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(controller.subprocess, "run", run)
    return state


def test_protocol_freezes_common_rows_and_separate_publication_boundary(experiment):
    inputs, _, prepared = experiment
    assert [len(values) for values in DEFAULT_FOLDS] == [25, 25, 25]
    assert [values[0] for values in DEFAULT_FOLDS] == [378, 504, 630]
    assert [values[-1] for values in DEFAULT_FOLDS] == [498, 624, 750]
    assert prepared["fit_reservations"] == 0
    journal, store, _, plan, _ = open_experiment(experiment_root("engineering-context"))
    try:
        protocol = store.json(plan.protocol_ref)
        training = store.json(protocol["folds"][0]["training_mask_ref"])
        publication = store.json(protocol["publication"]["training_mask_ref"])
        assert training["eligible_indices"] == list(range(60, 76))
        assert training["label_asof"] == utc_text(
            inputs.target[0].contract.calendar.sessions[80].close_at
        )
        assert publication["eligible_indices"] == list(range(60, 95))
        assert publication["label_knowledge_basis"] == "observed_receipts"
        assert protocol["sidecar_inputs_used_in_training"] == []
        assert prepared["budget"] == dict(
            max_attempts=4, worker_timeout_seconds=180, max_worker_seconds=720
        )
        changed = plan.model_dump(mode="json")
        changed["budget"]["max_attempts"] = 10
        with pytest.raises(ValueError, match="reservation"):
            ForecastComparisonPlan.model_validate_json(canonical(changed))
    finally:
        journal.close()


def test_worker_selection_publication_replay_and_repeat_do_not_refit(experiment, dummy_worker):
    _, envelope, prepared = experiment
    result = resume_experiment(
        "engineering-context", protocol_ref=prepared["protocol_ref"], max_fits=4
    )
    assert dummy_worker["launches"] == [0, 1, 2, 3]
    assert dummy_worker["fits"] == result["fit_reservations"] == 4
    assert result["reserved_worker_seconds"] == 720
    assert result["selection"]["selected_recipe"] == "B0"
    assert set(result["selection"]["candidates"]) == {"B0", "B1", "B2"}
    forecast = result["forecast"]
    assert forecast["training_rows"] == 35
    assert forecast["publication_model_validation_metrics"] is None
    assert forecast["publication_model_inherits_evaluation_scores"] is False
    assert forecast["sidecar_inputs_used_in_training"] == []
    assert forecast["model_inputs"] == ["target_prices"]
    assert forecast["forward"] is False
    assert result == inspect_experiment("engineering-context", replay=True)
    assert result == resume_experiment(
        "engineering-context", protocol_ref=prepared["protocol_ref"], max_fits=10
    )
    assert result == prepare_experiment(
        envelope, "engineering-context", fold_origins=((80, 85, 90),)
    )
    assert dummy_worker["fits"] == 4
    journal, store, _, _, _ = open_experiment(experiment_root("engineering-context"))
    try:
        eval_model = store.json(result["attempts"][0]["result_ref"])["model_ref"]
        assert eval_model != forecast["model_ref"]
    finally:
        journal.close()


def test_failed_fit_remains_charged_and_incomplete_candidate_cannot_win(experiment, dummy_worker):
    _, _, prepared = experiment
    dummy_worker["fail"].add(1)
    result = resume_experiment(
        "engineering-context", protocol_ref=prepared["protocol_ref"], max_fits=4
    )
    assert result["attempts"][1]["status"] == "failed"
    assert result["fit_reservations"] == 4 and result["reserved_worker_seconds"] == 720
    assert dummy_worker["fits"] == 3
    assert "B1" not in result["selection"]["candidates"]
    inspect_experiment("engineering-context", replay=True)
    resume_experiment("engineering-context", protocol_ref=prepared["protocol_ref"], max_fits=10)
    assert dummy_worker["launches"] == [0, 1, 2, 3]


def test_complete_worker_output_recovers_without_training_again(
    experiment, dummy_worker, monkeypatch
):
    _, _, prepared = experiment
    finish = Journal.finish
    once = [True]

    def crash(self, ordinal, status, **kwargs):
        if once[0] and status == "completed":
            once[0] = False
            raise RuntimeError("crash before controller commit")
        return finish(self, ordinal, status, **kwargs)

    monkeypatch.setattr(Journal, "finish", crash)
    with pytest.raises(RuntimeError, match="before controller commit"):
        resume_experiment("engineering-context", protocol_ref=prepared["protocol_ref"])
    assert dummy_worker["fits"] == 1
    result = resume_experiment("engineering-context", protocol_ref=prepared["protocol_ref"])
    assert dummy_worker["launches"] == [0, 1]
    assert result["fit_reservations"] == 2
    assert result["attempts"][0]["status"] == "completed"


def test_unknown_running_fit_is_charged_and_never_redispatched(experiment, dummy_worker):
    _, _, prepared = experiment
    journal, store, _, plan, _ = open_experiment(experiment_root("engineering-context"))
    try:
        proposal = plan.proposals[0]
        journal.reserve(0, proposal.name, store.put_json(proposal.model_dump(mode="json")), 4)
        journal.start(0, 180, 720)
    finally:
        journal.close()
    result = resume_experiment("engineering-context", protocol_ref=prepared["protocol_ref"])
    assert dummy_worker["launches"] == [1]
    assert result["attempts"][0]["status"] == "interrupted"
    assert result["fit_reservations"] == 2
    assert result["reserved_worker_seconds"] == 360


def test_exact_protocol_reference_required_before_reservation(experiment, dummy_worker):
    with pytest.raises(ValueError, match="exact frozen protocol"):
        resume_experiment("engineering-context", protocol_ref="0" * 64)
    assert dummy_worker["launches"] == []
    assert inspect_experiment("engineering-context")["fit_reservations"] == 0


def test_post_commit_receipt_after_deadline_cannot_claim_forward():
    report = dict(
        status="forecast",
        origin_close_at="2026-10-26T20:00:00Z",
        issued_at="2026-10-27T13:29:59Z",
        forward_deadline="2026-10-27T13:30:00Z",
    )
    confirmation = dict(forecast_ref=digest(canonical(report)), confirmed_at="2026-10-27T13:30:01Z")
    assert confirmed_report(report, confirmation)["forward"] is False
    confirmation["confirmed_at"] = "2026-10-27T13:29:59Z"
    assert confirmed_report(report, confirmation)["forward"] is True


def test_selection_uses_pooled_error_and_requires_every_fold(experiment):
    from math import log1p

    from quant_rsi.forecast_comparison import select_recipe

    _, _, prepared = experiment
    journal, _, _, original, _ = open_experiment(experiment_root("engineering-context"))
    journal.close()
    value = original.model_dump(mode="json")
    value["proposals"] = [
        dict(name=f"fold_{fold}_{group.lower()}", purpose="evaluation", recipe=group, fold=fold)
        for fold in range(2)
        for group in ("B0", "B1", "B2")
    ] + [dict(name="publication", purpose="publication", recipe=None, fold=None)]
    value["budget"] = dict(max_attempts=7, worker_timeout_seconds=180, max_worker_seconds=1260)
    plan = ForecastComparisonPlan.model_validate_json(canonical(value))
    reports = {}
    for group in ("B0", "B1", "B2"):
        for fold, count in enumerate((1, 9)):
            if group == "B2" and fold == 1:
                continue
            error = (0.10 if fold == 0 else 0.0) if group == "B0" else 0.03
            report = dict(
                recipe=group,
                fold=fold,
                eligible=True,
                input_ref=prepared["input_ref"],
                protocol_ref=prepared["protocol_ref"],
                metrics={"mae_pp": error * 100},
                rows=[
                    dict(origin_session=f"s{fold}_{i}", predicted=log1p(error), actual=0.0)
                    for i in range(count)
                ],
            )
            reports[f"fold_{fold}_{group.lower()}"] = (digest(canonical(report)), report)
    selection = select_recipe(plan, reports)
    assert selection["selected_recipe"] == "B0"
    assert selection["candidates"]["B0"]["pooled"]["mae_pp"] == pytest.approx(1.0)
    assert "B2" not in selection["candidates"]


def test_tampered_slot_charge_cannot_reset_remaining_budget(experiment, dummy_worker):
    _, _, prepared = experiment
    resume_experiment("engineering-context", protocol_ref=prepared["protocol_ref"])
    journal = Journal(experiment_root("engineering-context"))
    try:
        with journal.connection:
            journal.connection.execute("UPDATE attempts SET worker_seconds=0 WHERE ordinal=0")
    finally:
        journal.close()
    with pytest.raises(ValueError, match="charge differs"):
        resume_experiment("engineering-context", protocol_ref=prepared["protocol_ref"])
    assert dummy_worker["launches"] == [0]
