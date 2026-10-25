from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_rsi import controller, research, researcher
from quant_rsi.artifacts import ArtifactStore, canonical, decode
from quant_rsi.journal import Journal
from quant_rsi.research_journal import ResearchJournal
from quant_rsi.researcher import ResearchPlan


@pytest.fixture
def research_plan():
    path = Path(__file__).parents[1] / "examples/research_development.json"
    return ResearchPlan.model_validate_json(path.read_bytes())


def limited(plan, attempts=8, workers=160, no_progress=3):
    document = plan.model_dump(mode="json")
    document["development"]["budget"].update(max_attempts=attempts, max_worker_seconds=workers)
    document["policy"]["max_no_progress"] = no_progress
    return ResearchPlan.model_validate_json(canonical(document))


def ledger(root):
    journal = ResearchJournal(root)
    try:
        return journal.requests(), journal.attempts(), journal.events(), journal.state()
    finally:
        journal.close()


def forbidden(*args, **kwargs):
    pytest.fail("unexpected new proposal, worker, or artifact publication")


def test_finite_pause_shared_budget_and_readonly_replay(tmp_path, research_plan, monkeypatch):
    root = tmp_path / "campaign"
    first = research.create_research(research_plan, root, steps=1)
    assert first["execution_state"] == "paused" and first["stop_reason"] == "step_limit"
    assert first["attempts_reserved"] == first["kernel_attempts"] == 1
    assert first["worker_seconds_charged"] == 20
    before = ledger(root)
    with monkeypatch.context() as locked:
        locked.setattr(researcher, "propose", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        locked.setattr(ArtifactStore, "put", forbidden)
        assert research.inspect_research(root)["checkpoint_ref"] == first["checkpoint_ref"]
        replay = research.inspect_research(root, replay=True)
        assert replay["verified_experiments"] == 1
        assert ledger(root) == before
    second = research.resume_research(root, steps=1)
    assert second["attempts_reserved"] == 2
    assert second["worker_seconds_charged"] == 40
    assert second["results"][1]["parent_ref"] == first["checkpoint_ref"]
    assert second["results"][1]["decision"]["proposal"]["factor"] == {"op": "momentum", "window": 6}
    assert second["package_ref"] == first["package_ref"]


def test_committed_feedback_changes_next_proposal_with_provenance(
    tmp_path, research_plan, monkeypatch
):
    original_accept = controller.accept_worker_result
    histories = []
    for improvement in (True, False):

        def controlled_score(work, job, store, plan, data, *, improves=improvement):
            bundle_ref = original_accept(work, job, store, plan, data)
            bundle = store.json(bundle_ref)
            report = store.json(bundle["report_ref"])
            ordinal = job["binding"]["ordinal"]
            # Controlled trusted-evaluator fixture for policy behavior, not financial evidence.
            report["metrics"][plan.primary_metric] = (
                0.1 if ordinal == 0 else (0.2 if improves and ordinal == 1 else 0.05)
            )
            report["eligible"] = True
            report["eligibility_reasons"] = []
            bundle["report_ref"] = store.put_json(report)
            return store.put_json(bundle)

        with monkeypatch.context() as injected:
            injected.setattr(controller, "accept_worker_result", controlled_score)
            root = tmp_path / str(improvement)
            report = research.create_research(research_plan, root, steps=3)
        rows = ledger(root)[0]
        store = ArtifactStore(root / "artifacts")
        request = store.json(rows[2]["request_ref"])
        assert rows[1]["observation_ref"] in request["evidence_refs"]
        assert rows[1]["observation_event_seq"] < rows[2]["request_event_seq"]
        assert request["feedback_watermark"] == rows[1]["observation_event_seq"]
        assert request["checkpoint_ref"] == rows[1]["checkpoint_ref"]
        histories.append([r["decision"]["proposal"]["factor"]["window"] for r in report["results"]])
    assert histories == [[5, 6, 7], [5, 6, 4]]


@pytest.mark.parametrize("fault", ["duplicate", "invalid", "outside_scope", "proposal_failure"])
def test_bad_proposals_keep_global_reservations(tmp_path, research_plan, monkeypatch, fault):
    root = tmp_path / "campaign"
    plan = limited(research_plan, attempts=2)
    research.create_research(plan, root, steps=1)

    def bad_proposal(plan, checkpoint):
        if fault == "proposal_failure":
            raise ValueError("controlled proposer failure")
        factor = {"op": "momentum", "window": 5}
        if fault == "invalid":
            factor["op"] = "lead"
        elif fault == "outside_scope":
            factor["window"] = 11
        return {
            "name": "research_0001",
            "hypothesis": "Controlled rejection fixture.",
            "factor": factor,
        }

    monkeypatch.setattr(researcher, "propose", bad_proposal)
    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    report = research.resume_research(root, steps=1)
    assert report["attempts_reserved"] == 2
    assert report["worker_seconds_charged"] == 20
    assert report["results"][1]["observation"]["score"] is None
    assert report["kernel_attempts"] + report["unmapped_proposal_reservations"] == 2
    assert report["unmapped_proposal_reservations"] == (1 if fault == "proposal_failure" else 0)
    assert report["no_progress"] == 1
    assert report["execution_state"] == "completed"
    before = ledger(root)
    assert research.resume_research(root, steps=5) == report
    assert ledger(root) == before


@pytest.mark.parametrize(
    "stage",
    [
        "request",
        "proposal",
        "decision",
        "worker_result",
        "kernel_commit",
        "observation_before",
        "observation_transaction",
        "observation_after",
    ],
)
def test_committed_recovery_never_repeats_cost_or_learning(
    tmp_path, research_plan, monkeypatch, stage
):
    root = tmp_path / "campaign"
    plan = limited(research_plan, attempts=1, workers=20)
    targets = {
        "request": (ResearchJournal, "reserve_request"),
        "proposal": (ResearchJournal, "commit_decision"),
        "decision": (ResearchJournal, "commit_decision"),
        "worker_result": (controller, "accept_worker_result"),
        "kernel_commit": (Journal, "finish"),
        "observation_before": (ResearchJournal, "commit_observation"),
        "observation_transaction": (ResearchJournal, "event"),
        "observation_after": (ResearchJournal, "commit_observation"),
    }
    owner, method = targets[stage]
    original = getattr(owner, method)

    def crash(*args, **kwargs):
        if stage == "observation_transaction":
            result = original(*args, **kwargs)
            if args[2] == "research_step_committed":
                raise RuntimeError("injected observation transaction crash")
            return result
        if stage in {"proposal", "worker_result", "observation_before"}:
            raise RuntimeError("injected before commit")
        result = original(*args, **kwargs)
        raise RuntimeError("injected after commit")

    with monkeypatch.context() as injected:
        injected.setattr(owner, method, crash)
        with pytest.raises(RuntimeError, match="injected"):
            research.create_research(plan, root, steps=1)
    before = ledger(root)
    committed_refs = [
        (r["request_ref"], r["decision_ref"], r["observation_ref"], r["checkpoint_ref"])
        for r in before[0]
    ]
    if stage != "request":
        monkeypatch.setattr(researcher, "propose", forbidden)
    if stage in {
        "worker_result",
        "kernel_commit",
        "observation_before",
        "observation_transaction",
        "observation_after",
    }:
        monkeypatch.setattr(controller.subprocess, "run", forbidden)
    report = research.resume_research(root, steps=1)
    after = ledger(root)
    assert report["execution_state"] == "completed"
    assert report["attempts_reserved"] == report["kernel_attempts"] == 1
    assert report["worker_seconds_charged"] == 20
    assert report["results"][0]["observation"]["status"] == "completed"
    assert sum(e["kind"] == "research_request_reserved" for e in after[2]) == 1
    assert sum(e["kind"] == "worker_slot_reserved" for e in after[2]) == 1
    assert sum(e["kind"] == "research_step_committed" for e in after[2]) == 1
    for old, row in zip(committed_refs, after[0], strict=True):
        new = (
            row["request_ref"],
            row["decision_ref"],
            row["observation_ref"],
            row["checkpoint_ref"],
        )
        assert all(a is None or a == b for a, b in zip(old, new, strict=True))
    assert research.resume_research(root, steps=1) == report
    assert ledger(root) == after


def test_charged_worker_without_output_is_real_terminal_failure(
    tmp_path, research_plan, monkeypatch
):
    root = tmp_path / "campaign"
    original = controller.atomic_write

    def crash_before_launch(path, data):
        if path.name == "job.json":
            raise RuntimeError("injected after worker reservation")
        return original(path, data)

    with monkeypatch.context() as injected:
        injected.setattr(controller, "atomic_write", crash_before_launch)
        with pytest.raises(RuntimeError):
            research.create_research(research_plan, root)
    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    monkeypatch.setattr(researcher, "propose", forbidden)
    report = research.resume_research(root, steps=10)
    assert report["stop_reason"] == "baseline_unavailable"
    assert report["worker_seconds_charged"] == 20
    assert report["results"][0]["observation"]["status"] == "interrupted"
    assert report["results"][0]["observation"]["score"] is None


def test_full_worker_slot_limit_stops_before_next_proposal(tmp_path, research_plan, monkeypatch):
    root = tmp_path / "campaign"
    report = research.create_research(limited(research_plan, workers=39), root, steps=10)
    assert report["stop_reason"] == "worker_budget_exhausted"
    assert report["attempts_reserved"] == 1 and report["worker_seconds_remaining"] == 19
    monkeypatch.setattr(researcher, "propose", forbidden)
    assert research.resume_research(root, steps=5) == report


def test_kernel_prediction_and_accounting_match_legacy_run(tmp_path, research_plan):
    legacy = tmp_path / "legacy"
    campaign = tmp_path / "research"
    controller.create_run(research_plan.development, legacy)
    research.create_research(research_plan, campaign, steps=1)
    stores = [ArtifactStore(p / "artifacts") for p in (legacy, campaign)]
    journal = Journal(legacy)
    try:
        references = [journal.attempts()[0]["result_ref"], ledger(campaign)[1][0]["result_ref"]]
    finally:
        journal.close()
    bundles = [store.json(ref) for store, ref in zip(stores, references, strict=True)]
    for field in ("model_ref", "predictions_ref", "report_ref"):
        assert bundles[0][field] == bundles[1][field]


def test_campaign_lock_and_runtime_boundary(tmp_path, research_plan, monkeypatch):
    root = tmp_path / "campaign"
    research.create_research(research_plan, root, steps=1)
    with controller.controller_lock(root):
        with pytest.raises(ValueError, match="still owns"):
            research.resume_research(root)
    monkeypatch.setattr(controller, "runtime_identity", lambda: {"changed": True})
    assert research.inspect_research(root)["execution_state"] == "paused"
    with pytest.raises(ValueError, match="source/runtime"):
        research.resume_research(root)


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_improvement_threshold_must_be_finite_nonnegative(research_plan, value):
    document = research_plan.model_dump(mode="json")
    document["policy"]["min_improvement"] = value
    with pytest.raises((ValueError, ValidationError)):
        ResearchPlan.model_validate_json(canonical(document))


def test_missing_replay_artifact_is_not_recreated(tmp_path, research_plan):
    root = tmp_path / "campaign"
    research.create_research(research_plan, root, steps=1)
    store = ArtifactStore(root / "artifacts")
    experiment = store.json(ledger(root)[0][0]["experiment_ref"])
    path = root / "artifacts" / experiment["research_context_ref"]
    path.unlink()
    with pytest.raises(OSError):
        research.inspect_research(root, replay=True)
    assert not path.exists()


def test_cached_legal_wrong_candidate_is_rejected_without_dispatch(
    tmp_path, research_plan, monkeypatch
):
    root = tmp_path / "campaign"
    plan = limited(research_plan, attempts=2)
    research.create_research(plan, root, steps=1)

    def crash(*args, **kwargs):
        raise RuntimeError("injected after response publication")

    with monkeypatch.context() as injected:
        injected.setattr(ResearchJournal, "commit_decision", crash)
        with pytest.raises(RuntimeError):
            research.resume_research(root, steps=1)
    response_path = root / "proposals/0001.json"
    response = decode(response_path.read_bytes())
    assert response["proposal"]["factor"] == {"op": "momentum", "window": 6}
    response["proposal"]["factor"]["window"] = 4
    corrupted = canonical(response)
    response_path.write_bytes(corrupted)
    monkeypatch.setattr(researcher, "propose", forbidden)
    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    report = research.resume_research(root, steps=1)
    assert report["attempts_reserved"] == 2 and report["worker_seconds_charged"] == 20
    assert report["results"][1]["observation"]["status"] == "rejected"
    assert report["results"][1]["observation"]["error"] == "proposal_policy_mismatch"
    assert report["results"][1]["observation"]["score"] is None
    assert response_path.read_bytes() == corrupted
    response_ref = report["results"][1]["decision"]["response_ref"]
    assert ArtifactStore(root / "artifacts").get(response_ref) == corrupted
    assert research.inspect_research(root, replay=True)["verified_experiments"] == 1


def test_semantic_replay_rejects_internally_consistent_wrong_policy_success(
    tmp_path, research_plan, monkeypatch
):
    root = tmp_path / "campaign"
    research.create_research(limited(research_plan, attempts=2), root, steps=1)
    original_validate = research.validate_response

    def faulty_adapter(plan, checkpoint):
        proposal = researcher.expected_proposal(plan, checkpoint)
        proposal["factor"]["window"] = 4
        return proposal

    def faulty_acceptance(*args):
        decision = original_validate(*args)
        return decision | {"error": None}

    # Simulate a self-consistent historical bad success, bypassing the new ingress guard.
    with monkeypatch.context() as injected:
        injected.setattr(researcher, "propose", faulty_adapter)
        injected.setattr(research, "validate_response", faulty_acceptance)
        bad_report = research.resume_research(root, steps=1)
    assert bad_report["results"][1]["observation"]["status"] == "completed"
    with pytest.raises(ValueError, match="frozen deterministic policy"):
        research.inspect_research(root, replay=True)


@pytest.mark.parametrize("stage", ["before_journal", "initial_commit"])
def test_research_initialization_recovery_preserves_one_identity(
    tmp_path, research_plan, monkeypatch, stage
):
    root = tmp_path / "campaign"
    original_event = ResearchJournal.event

    def crash_before_journal(*args, **kwargs):
        raise RuntimeError("injected initialization")

    def crash_at_event(self, ordinal, kind, payload):
        original_event(self, ordinal, kind, payload)
        if kind == "run_created":
            raise RuntimeError("injected initialization")

    with monkeypatch.context() as injected:
        if stage == "before_journal":
            injected.setattr(research, "ResearchJournal", crash_before_journal)
        else:
            injected.setattr(ResearchJournal, "event", crash_at_event)
        with pytest.raises(RuntimeError):
            research.create_research(research_plan, root)
    campaign_id = decode((root / "research_initialization.json").read_bytes())["campaign_id"]
    report = research.resume_research(root, steps=1)
    assert report["campaign_id"] == campaign_id and report["attempts_reserved"] == 1
    assert sum(e["kind"] == "run_created" for e in ledger(root)[2]) == 1


def test_no_progress_and_observation_cas(tmp_path, research_plan, monkeypatch):
    root = tmp_path / "campaign"
    original_accept = controller.accept_worker_result

    def equal_score(work, job, store, plan, data):
        bundle = store.json(original_accept(work, job, store, plan, data))
        report = store.json(bundle["report_ref"])
        report["metrics"][plan.primary_metric] = 0.1
        report["eligible"] = True
        report["eligibility_reasons"] = []
        bundle["report_ref"] = store.put_json(report)
        return store.put_json(bundle)

    monkeypatch.setattr(controller, "accept_worker_result", equal_score)
    report = research.create_research(limited(research_plan, no_progress=1), root, steps=10)
    assert report["stop_reason"] == "no_progress_limit" and report["attempts_reserved"] == 2
    assert report["incumbent"]["ordinal"] == 0 and report["no_progress"] == 1
    before = ledger(root)
    journal = ResearchJournal(root)
    try:
        row = journal.requests()[1]
        initial = decode((root / "manifest.json").read_bytes())["initial_checkpoint_ref"]
        journal.commit_observation(
            1, row["parent_ref"], initial, row["observation_ref"], row["checkpoint_ref"]
        )
        with pytest.raises(ValueError, match="conflicting observation"):
            journal.commit_observation(
                1, row["parent_ref"], initial, "0" * 64, row["checkpoint_ref"]
            )
        with pytest.raises(ValueError, match="compare-and-swap"):
            journal.reserve_request("0" * 64, row["parent_ref"], initial, 8)
    finally:
        journal.close()
    assert ledger(root) == before


@pytest.mark.parametrize(
    "field,value", [("visibility", "protected"), ("feedback_watermark", 999999)]
)
def test_pending_request_cannot_gain_uncommitted_feedback(
    tmp_path, research_plan, monkeypatch, field, value
):
    root = tmp_path / "campaign"
    research.create_research(research_plan, root, steps=1)
    original = ResearchJournal.reserve_request

    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected pending request")

    with monkeypatch.context() as injected:
        injected.setattr(ResearchJournal, "reserve_request", crash)
        with pytest.raises(RuntimeError):
            research.resume_research(root)
    store = ArtifactStore(root / "artifacts")
    row = ledger(root)[0][1]
    altered = store.json(row["request_ref"]) | {field: value}
    journal = ResearchJournal(root)
    try:
        with journal.connection:
            journal.connection.execute(
                "UPDATE research_requests SET request_ref=? WHERE ordinal=1",
                (store.put_json(altered),),
            )
    finally:
        journal.close()
    monkeypatch.setattr(researcher, "propose", forbidden)
    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="evidence availability mismatch"):
        research.resume_research(root)
    assert len(ledger(root)[0]) == 2 and len(ledger(root)[1]) == 1
