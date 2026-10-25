"""C4 mechanism, provenance, resource, and failure-window regressions."""

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from quant_rsi import changes, controller, research
from quant_rsi import package_experiment as packages
from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.package_journal import PackageJournal
from quant_rsi.research_journal import ResearchJournal
from quant_rsi.researcher import ResearchPlan


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("c4_evidence") / "source"
    plan = ResearchPlan.model_validate_json(Path("examples/research_development.json").read_bytes())
    report = research.create_research(plan, root, steps=100)
    assert report["attempts_reserved"] == 6
    return root


def snapshot(root):
    return {
        str(p.relative_to(root)): digest(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name != ".controller.lock"
    }


def ledger(root):
    journal = PackageJournal(root)
    try:
        return journal.state(), journal.grants(), journal.rounds(), journal.events()
    finally:
        journal.close()


def forbidden(*args, **kwargs):
    pytest.fail("unexpected proposal, worker, or artifact mutation")


def test_real_matched_packages_and_readonly_replay(tmp_path, evidence, monkeypatch):
    original = snapshot(evidence)
    root = tmp_path / "package"
    report = packages.create_package(evidence, root, state_policy="reset")
    accepted = report["accepted_pointer"]
    assert report["decision"]["outcome"] == "patch"
    assert report["research_budget"]["attempts_granted"] == 16
    assert report["research_budget"]["attempts_consumed"] == 2
    assert report["historical_evidence_cost"]["historical_attempts"] == 6
    for arm in report["arms"].values():
        assert arm["committed_observations"] == 1
        assert arm["results"][0]["input_evidence_refs"] == []
    report = packages.resume_package(root)
    assert [
        report["arms"][arm]["results"][1]["observation"]["factor"]
        for arm in ("parent", "candidate")
    ] == [{"op": "momentum", "window": 6}, {"op": "mean_return", "window": 5}]
    assert report["arms"]["parent"]["package_ref"] != report["arms"]["candidate"]["package_ref"]
    report = packages.resume_package(root, steps=100)
    assert report["execution_state"] == "completed"
    assert report["research_budget"]["attempts_consumed"] == 14
    assert report["research_budget"]["worker_slots_consumed"] == 280
    assert report["comparison"]["candidate_minus_parent_development_score"] == 0
    before = snapshot(root)
    with monkeypatch.context() as locked:
        locked.setattr(changes, "propose_change", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        locked.setattr(ArtifactStore, "put", forbidden)
        assert packages.inspect_package(root)["accepted_pointer"] == accepted
        replay = packages.inspect_package(root, replay=True)
        assert sum(r["verified_experiments"] for r in replay["arms"].values()) == 14
        assert packages.resume_package(root, steps=100)["accepted_pointer"] == accepted
    assert snapshot(root) == before
    assert snapshot(evidence) == original
    with pytest.raises(ValueError, match="managed research arm"):
        research.resume_research(root / "arms/candidate")
    with pytest.raises(FileExistsError):
        packages.create_package(evidence, root, state_policy="reset")


def test_two_committed_feedback_histories_control_patch(tmp_path, monkeypatch):
    plan = ResearchPlan.model_validate_json(Path("examples/research_development.json").read_bytes())
    outcomes = []
    real_evaluate = controller.evaluate
    for supports in (True, False):
        scores = {}
        sequence = iter((1.0, 0.9, 0.8, 0.85 if supports else 0.7))

        def controlled_evaluator(
            data, cutoff, predictions, policy, annualization, *, scores=scores, sequence=sequence
        ):
            report = real_evaluate(data, cutoff, predictions, policy, annualization)
            key = digest(canonical(predictions))
            if key not in scores:
                scores[key] = next(sequence)
            # Synthetic trusted-evaluator fixture, not an economic performance claim.
            report["metrics"][plan.development.primary_metric] = scores[key]
            return report

        with monkeypatch.context() as injected:
            injected.setattr(controller, "evaluate", controlled_evaluator)
            source = tmp_path / f"source_{supports}"
            research.create_research(plan, source, steps=100)
            root = tmp_path / f"package_{supports}"
            report = packages.create_package(source, root, state_policy="reset")
            outcomes.append(report["decision"]["outcome"])
            request = ArtifactStore(root / "artifacts").json(report["request_ref"])
            assert len(request["evidence_refs"]) == 4
            assert report["change_requests_reserved"] == 1
            if not supports:
                assert report["arms"] == {}
                assert report["research_budget"]["attempts_granted"] == 0
                before = snapshot(root)
                packages.resume_package(root)
                packages.inspect_package(root, replay=True)
                assert snapshot(root) == before
    assert outcomes == ["patch", "no_patch"]


def reserve_then_crash(monkeypatch, evidence, root):
    original = PackageJournal.reserve

    def crash(self, request_ref):
        original(self, request_ref)
        raise SystemExit("after request reservation")

    with monkeypatch.context() as broken:
        broken.setattr(PackageJournal, "reserve", crash)
        with pytest.raises(SystemExit):
            packages.create_package(evidence, root, state_policy="reset")
    assert ledger(root)[0]["request_ref"] is not None


@pytest.mark.parametrize(
    "fault",
    [
        "base",
        "field",
        "value",
        "evidence",
        "watermark",
        "scope",
        "migration",
        "interface",
        "legal_but_wrong_no_patch",
    ],
)
def test_bad_cached_patch_keeps_response_and_reservation(tmp_path, evidence, monkeypatch, fault):
    root = tmp_path / "package"
    reserve_then_crash(monkeypatch, evidence, root)
    store = ArtifactStore(root / "artifacts")
    manifest = decode((root / "manifest.json").read_bytes())
    request = store.json(manifest["request_ref"])
    response = changes.expected_change(
        manifest["request_ref"], request, store.json(manifest["evidence_ref"])
    )
    patch = response["patch"]
    if fault == "base":
        patch["base_package_ref"] = "0" * 64
    elif fault == "field":
        patch["budget"] = {"max_attempts": 100}
    elif fault == "value":
        patch["value"] = "window_first"
    elif fault == "evidence":
        patch["evidence_refs"].append("f" * 64)
    elif fault == "watermark":
        patch["feedback_watermark"] += 1
    elif fault == "scope":
        patch["visibility"] = "protected"
    elif fault == "migration":
        patch["state_policy"] = "migrate"
    elif fault == "interface":
        patch["state_schema_version"] = 2
    else:
        response.update(outcome="no_patch", patch=None, reason="operator_not_better")
    path = root / "change_response.json"
    atomic_write(path, canonical(response))
    raw = path.read_bytes()
    with monkeypatch.context() as locked:
        locked.setattr(changes, "propose_change", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        report = packages.resume_package(root)
        replay = packages.inspect_package(root, replay=True)
    assert report["decision"]["error"] == "patch_policy_mismatch"
    assert report["change_requests_reserved"] == 1
    assert report["research_budget"]["attempts_granted"] == 0
    assert replay["arms"] == {}
    assert path.read_bytes() == raw
    assert store.json(report["decision"]["response_ref"]) == response


@pytest.mark.parametrize(
    "stage",
    [
        "reserve",
        "response",
        "candidate",
        "decision_before",
        "decision_transaction",
        "decision_after",
        "child_initialized",
        "child_receipt",
        "child_result",
        "round_transaction",
        "round_after",
    ],
)
def test_failure_windows_reuse_grants_ids_and_committed_work(
    tmp_path, evidence, monkeypatch, stage
):
    root = tmp_path / "package"
    fired = False

    def stop():
        nonlocal fired
        if not fired:
            fired = True
            raise SystemExit(stage)

    with monkeypatch.context() as broken:
        if stage == "reserve":
            original = PackageJournal.reserve

            def invoke(self, ref):
                original(self, ref)
                stop()

            broken.setattr(PackageJournal, "reserve", invoke)
        elif stage == "response":
            original = packages.atomic_write

            def invoke(path, data):
                original(path, data)
                if path.name == "change_response.json":
                    stop()

            broken.setattr(packages, "atomic_write", invoke)
        elif stage == "candidate":
            original = ArtifactStore.put_json

            def invoke(self, value):
                ref = original(self, value)
                if (
                    isinstance(value, dict)
                    and value.get("kind") == "ResearcherPackage"
                    and value.get("lineage")
                ):
                    stop()
                return ref

            broken.setattr(ArtifactStore, "put_json", invoke)
        elif stage in {"decision_before", "decision_after"}:
            original = PackageJournal.decide

            def invoke(self, ref, grants):
                if stage == "decision_before":
                    stop()
                original(self, ref, grants)
                stop()

            broken.setattr(PackageJournal, "decide", invoke)
        elif stage in {"decision_transaction", "round_transaction"}:
            original = PackageJournal.event

            def invoke(self, ordinal, kind, payload):
                original(self, ordinal, kind, payload)
                if kind == (
                    "change_decided"
                    if stage == "decision_transaction"
                    else "package_round_committed"
                ):
                    stop()

            broken.setattr(PackageJournal, "event", invoke)
        elif stage in {"child_initialized", "child_receipt"}:
            original = PackageJournal.mark_initialized

            def invoke(self, arm):
                if stage == "child_initialized":
                    stop()
                original(self, arm)
                stop()

            broken.setattr(PackageJournal, "mark_initialized", invoke)
        elif stage == "child_result":
            original = research.resume_research

            def invoke(*args, **kwargs):
                result = original(*args, **kwargs)
                stop()
                return result

            broken.setattr(research, "resume_research", invoke)
        else:
            original = PackageJournal.finish_round

            def invoke(self, ordinal, ref):
                original(self, ordinal, ref)
                stop()

            broken.setattr(PackageJournal, "finish_round", invoke)
        with pytest.raises(SystemExit):
            packages.create_package(evidence, root, state_policy="reset")
    assert fired
    before_state, before_grants, _, _ = ledger(root)
    before_children = {}
    for path in (root / "arms").glob("*") if (root / "arms").exists() else []:
        if (path / "manifest.json").exists():
            before_children[path.name] = decode((path / "manifest.json").read_bytes())
    original_response = (
        (root / "change_response.json").read_bytes()
        if (root / "change_response.json").exists()
        else None
    )
    with monkeypatch.context() as locked:
        if original_response is not None:
            locked.setattr(changes, "propose_change", forbidden)
        result = packages.resume_package(root)
    state, grants, rounds, events = ledger(root)
    assert state["request_ref"] == before_state["request_ref"]
    assert len(grants) == 2 and len({g["grant_ref"] for g in grants}) == 2
    if before_grants:
        assert [g["grant_ref"] for g in grants] == [g["grant_ref"] for g in before_grants]
    assert sum(e["kind"] == "change_reserved" for e in events) == 1
    assert sum(e["kind"] == "change_decided" for e in events) == 1
    assert sum(e["kind"] == "arm_initialized" for e in events) == 2
    expected = 2 if stage == "round_after" else 1
    assert all(r["committed_observations"] == expected for r in result["arms"].values())
    assert result["research_budget"]["attempts_consumed"] == 2 * expected
    assert result["research_budget"]["attempts_granted"] == 16
    assert all(row["receipt_ref"] for row in rounds)
    for arm, manifest in before_children.items():
        assert decode((root / "arms" / arm / "manifest.json").read_bytes()) == manifest
    if original_response is not None:
        assert (root / "change_response.json").read_bytes() == original_response
    packages.inspect_package(root, replay=True)


def test_live_orphan_keeps_both_locks_and_failure_cost(tmp_path, evidence, monkeypatch):
    root = tmp_path / "package"
    processes = []

    def orphan(*args, **kwargs):
        assert len(kwargs["pass_fds"]) == 2
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=kwargs["pass_fds"],
        )
        processes.append(process)
        raise SystemExit("controller died while child owned both locks")

    try:
        with monkeypatch.context() as broken:
            broken.setattr(controller.subprocess, "run", orphan)
            with pytest.raises(SystemExit):
                packages.create_package(evidence, root, state_policy="reset")
        with pytest.raises(ValueError, match="still owns"):
            packages.resume_package(root)
        with pytest.raises(ValueError, match="still owns"):
            with controller.controller_lock(root / "arms/parent"):
                pass
        processes[0].communicate(b"x", timeout=5)
        result = packages.resume_package(root)
        parent = result["arms"]["parent"]
        assert parent["stop_reason"] == "baseline_unavailable"
        assert parent["results"][0]["observation"]["status"] == "interrupted"
        assert parent["incumbent"] is None
        assert result["comparison"]["candidate_minus_parent_development_score"] is None
        assert result["research_budget"]["worker_slots_consumed"] == 40
        assert result["research_budget"]["attempts_granted"] == 16
        packages.inspect_package(root, replay=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


@pytest.mark.parametrize(
    "damage", ["delete_child", "bad_database", "cross_checkpoint", "missing_artifact"]
)
def test_no_rebuilding_or_cross_package_state(tmp_path, evidence, monkeypatch, damage):
    root = tmp_path / "package"
    packages.create_package(evidence, root, state_policy="reset")
    before = ledger(root)
    child = root / "arms/candidate"
    if damage == "delete_child":
        shutil.rmtree(child)
    elif damage == "bad_database":
        (child / "journal.sqlite").write_bytes(b"damaged")
    elif damage == "cross_checkpoint":
        source = ResearchJournal(root / "arms/parent")
        target = ResearchJournal(child)
        try:
            with target.connection:
                target.connection.execute(
                    "UPDATE research_state SET head_ref=?", (source.state()["head_ref"],)
                )
        finally:
            source.close()
            target.close()
    else:
        store = ArtifactStore(root / "artifacts")
        decision = store.json(before[0]["decision_ref"])
        (store.root / decision["candidate_package_ref"]).unlink()
    corrupted = snapshot(root)
    with monkeypatch.context() as locked:
        locked.setattr(changes, "propose_change", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        locked.setattr(ArtifactStore, "put", forbidden)
        with pytest.raises((ValueError, FileNotFoundError, sqlite3.DatabaseError)):
            packages.inspect_package(root, replay=True)
        with pytest.raises((ValueError, FileNotFoundError, sqlite3.DatabaseError)):
            packages.resume_package(root)
    assert ledger(root) == before
    assert snapshot(root) == corrupted


def test_reject_migration_and_foreign_source_before_request(tmp_path, evidence, monkeypatch):
    with pytest.raises(ValueError, match="reset"):
        packages.create_package(evidence, tmp_path / "migration", state_policy="migrate")
    clone = tmp_path / "source"
    shutil.copytree(evidence, clone)
    journal = ResearchJournal(clone)
    store = ArtifactStore(clone / "artifacts")
    try:
        row = journal.requests()[-1]
        observation = store.json(row["observation_ref"])
        observation["visibility"] = "protected"
        ref = store.put_json(observation)
        with journal.connection:
            journal.connection.execute(
                "UPDATE research_requests SET observation_ref=? WHERE ordinal=?",
                (ref, row["ordinal"]),
            )
    finally:
        journal.close()
    with monkeypatch.context() as locked:
        locked.setattr(changes, "propose_change", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        with pytest.raises(ValueError):
            packages.create_package(clone, tmp_path / "invalid_source", state_policy="reset")
    assert not (tmp_path / "invalid_source" / "journal.sqlite").exists()


@pytest.mark.parametrize("stage", ["before_journal", "schema_transaction"])
def test_package_initialization_is_recoverable(tmp_path, evidence, monkeypatch, stage):
    root = tmp_path / "package"
    with monkeypatch.context() as broken:
        if stage == "before_journal":

            def crash(*args, **kwargs):
                raise SystemExit("before package journal")

            broken.setattr(PackageJournal, "__init__", crash)
        else:
            original = PackageJournal.initialize_extra

            def crash(self):
                original(self)
                raise SystemExit("inside initialization transaction")

            broken.setattr(PackageJournal, "initialize_extra", crash)
        with pytest.raises(SystemExit):
            packages.create_package(evidence, root, state_policy="reset")
    marker = decode((root / "package_initialization.json").read_bytes())
    report = packages.resume_package(root)
    assert report["campaign_id"] == marker["campaign_id"]
    assert report["change_requests_reserved"] == 1
    assert report["research_budget"]["attempts_consumed"] == 2
    assert not (root / "package_initialization.json").exists()
    assert sum(e["kind"] == "run_created" for e in ledger(root)[3]) == 1


@pytest.mark.parametrize("stage", ["before", "transaction", "after"])
def test_final_outcome_recovery_does_not_start_another_round(
    tmp_path, evidence, monkeypatch, stage
):
    root = tmp_path / "package"
    with monkeypatch.context() as broken:
        if stage == "transaction":
            original = PackageJournal.event

            def crash(self, ordinal, kind, payload):
                original(self, ordinal, kind, payload)
                if kind == "package_completed":
                    raise SystemExit("inside outcome transaction")

            broken.setattr(PackageJournal, "event", crash)
        else:
            original = PackageJournal.complete

            def crash(self, ref):
                if stage == "after":
                    original(self, ref)
                raise SystemExit("outcome commit boundary")

            broken.setattr(PackageJournal, "complete", crash)
        with pytest.raises(SystemExit):
            packages.create_package(evidence, root, state_policy="reset", steps=100)
    before = ledger(root)
    seen = packages.inspect_package(root, replay=True)
    assert seen["research_budget"]["attempts_consumed"] == 14
    assert ledger(root) == before  # Inspection must not publish the missing outer outcome.
    with monkeypatch.context() as locked:
        locked.setattr(changes, "propose_change", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        result = packages.resume_package(root, steps=100)
    assert result["execution_state"] == "completed"
    assert len(ledger(root)[2]) == 1
    assert sum(e["kind"] == "package_completed" for e in ledger(root)[3]) == 1
    assert result["accepted_pointer"] == seen["accepted_pointer"]
    assert result["research_budget"] == seen["research_budget"]


def test_change_generation_failure_is_terminal_and_charged(tmp_path, evidence, monkeypatch):
    root = tmp_path / "package"

    def unavailable(*args):
        raise RuntimeError("controlled proposer failure")

    with monkeypatch.context() as broken:
        broken.setattr(changes, "propose_change", unavailable)
        result = packages.create_package(evidence, root, state_policy="reset")
    assert result["decision"]["outcome"] == "failed"
    assert result["change_requests_reserved"] == 1
    assert result["research_budget"]["attempts_consumed"] == 0
    before = snapshot(root)
    with monkeypatch.context() as locked:
        locked.setattr(changes, "propose_change", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        packages.resume_package(root)
        packages.inspect_package(root, replay=True)
    assert snapshot(root) == before


@pytest.mark.parametrize("invalid", ["malformed_json", "boolean_interface", "numeric_overflow"])
def test_invalid_response_bytes_are_preserved_and_rejected(
    tmp_path, evidence, monkeypatch, invalid
):
    root = tmp_path / "package"
    reserve_then_crash(monkeypatch, evidence, root)
    store = ArtifactStore(root / "artifacts")
    manifest = decode((root / "manifest.json").read_bytes())
    raw = b"not-valid-json: \xff"
    if invalid == "numeric_overflow":
        raw = b'{"extra":1e309}'
    if invalid == "boolean_interface":
        response = changes.expected_change(
            manifest["request_ref"],
            store.json(manifest["request_ref"]),
            store.json(manifest["evidence_ref"]),
        )
        response["patch"]["interface_version"] = True
        raw = canonical(response)
    atomic_write(root / "change_response.json", raw)
    result = packages.resume_package(root)
    assert result["decision"]["outcome"] == "rejected"
    assert result["execution_state"] == "completed"
    assert result["change_requests_reserved"] == 1
    assert result["research_budget"]["attempts_granted"] == 0
    assert result["research_budget"]["worker_slots_consumed"] == 0
    assert (root / "change_response.json").read_bytes() == raw
    wrapper = store.json(result["decision"]["response_ref"])
    if invalid in {"malformed_json", "numeric_overflow"}:
        assert store.get(wrapper["raw_ref"]) == raw
    before = snapshot(root)
    with monkeypatch.context() as locked:
        locked.setattr(changes, "propose_change", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        locked.setattr(ArtifactStore, "put", forbidden)
        packages.inspect_package(root, replay=True)
        packages.resume_package(root)
    assert snapshot(root) == before
