"""Actual selected-model C5 integration and authority protocol failure windows."""

import os
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from quant_rsi import comparison, evidence, package_experiment, research, researcher
from quant_rsi.artifacts import ArtifactStore, decode, digest
from quant_rsi.evidence import ComparisonInput, ProgramInput
from quant_rsi.researcher import ResearchPlan


@pytest.fixture(scope="module")
def c5_snapshots(tmp_path_factory):
    root = tmp_path_factory.mktemp("c5_actual")
    plan = ResearchPlan.model_validate_json(Path("examples/research_development.json").read_bytes())
    research.create_research(plan, root / "source", steps=100)
    package_experiment.create_package(
        root / "source", root / "package", state_policy="reset", steps=100
    )
    workspace = root / "workspace"
    workspace.mkdir()
    previous = os.environ.get("QUANT_RSI_WORKSPACE")
    os.environ["QUANT_RSI_WORKSPACE"] = str(workspace)
    try:
        spec = ProgramInput(
            name="integration", package_campaign=str(root / "package"), task_count=1
        )
        with patch("quant_rsi.evidence.secrets.randbits", return_value=12345):
            evidence.init_evidence(spec)
        saved = root / "available"
        shutil.copytree(workspace / ".quant_rsi", saved)
        output = root / "handle"
        comparison.register_comparison(ComparisonInput(name="one", mode="engineering_only"), output)
        shutil.copytree(workspace / ".quant_rsi", root / "reserved")
        while True:
            status = comparison.resume_comparison(output, steps=1)
            if status["selection_coverage_ref"] and not status["exposed"]:
                shutil.copytree(workspace / ".quant_rsi", root / "sealed")
            if status["exposed"]:
                shutil.copytree(workspace / ".quant_rsi", root / "opened")
                break
        comparison.resume_comparison(output, steps=100)
        report = comparison.inspect_comparison(output, replay=True)
        completed = root / "completed"
        shutil.copytree(workspace / ".quant_rsi", completed)
        return {
            "root": root,
            "available": saved,
            "completed": completed,
            "output": output,
            "spec": spec,
            "report": report,
        }
    finally:
        if previous is None:
            os.environ.pop("QUANT_RSI_WORKSPACE", None)
        else:
            os.environ["QUANT_RSI_WORKSPACE"] = previous


@pytest.fixture
def authority(c5_snapshots, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(c5_snapshots["available"], workspace / ".quant_rsi")
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    return workspace, tmp_path / "comparison"


def test_actual_selected_models_and_retain(c5_snapshots, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(c5_snapshots["completed"], workspace / ".quant_rsi")
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    output = c5_snapshots["output"]
    result = comparison.inspect_comparison(output, replay=True)
    assert result["batch_state"] == "RETIRED"
    assert result["development_attempts_granted"] == 16
    assert result["development_seconds_granted"] == 320
    assert result["scoring_seconds_charged"] == 40
    assert result["report"]["promotion_eligible"] is False
    assert result["report"]["inference"]["formal_tests_executed"] == 0
    assert result["report"]["inference"]["p_value"] is None
    decision = comparison.decide_comparison(output, "same")
    assert decision["decision"] == "retain"
    assert comparison.decide_comparison(output, "same") == decision
    with pytest.raises(ValueError, match="CAS"):
        comparison.decide_comparison(output, "another")
    assert comparison.resume_comparison(output, steps=1)["scoring_seconds_charged"] == 40


def test_confirmatory_preflight_and_cancel(authority, c5_snapshots):
    workspace, output = authority
    with pytest.raises(ValueError, match="blocked_uncalibrated"):
        comparison.register_comparison(ComparisonInput(name="blocked", mode="confirmatory"), output)
    assert not output.exists()
    with evidence.Authority(workspace / ".quant_rsi") as a:
        assert a.state["batch"] is None
    first = comparison.register_comparison(
        ComparisonInput(name="one", mode="engineering_only"), output
    )
    cancelled = comparison.cancel_comparison(output)
    assert cancelled["batch_state"] == "RETIRED"
    assert cancelled["development_seconds_granted"] == first["development_seconds_granted"]
    renamed = comparison.register_comparison(
        ComparisonInput(name="renamed", mode="engineering_only"), output.parent / "renamed"
    )
    assert renamed["batch_state"] == "RETIRED"
    assert renamed["development_seconds_granted"] == 320
    assert (
        evidence.init_evidence(c5_snapshots["spec"].model_copy(update={"name": "changed"}))[
            "batch_state"
        ]
        == "RETIRED"
    )


@pytest.fixture
def snapshot(c5_snapshots, tmp_path, monkeypatch):
    def use(name):
        workspace = tmp_path / name
        workspace.mkdir()
        source = c5_snapshots.get(name, c5_snapshots["root"] / name)
        shutil.copytree(source, workspace / ".quant_rsi")
        monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
        return workspace / ".quant_rsi", c5_snapshots["output"]

    return use


def test_readonly_replay_never_fits_or_proposes(snapshot, monkeypatch):
    root, output = snapshot("completed")

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden side effect during replay")

    monkeypatch.setattr(researcher, "propose", forbidden)
    import lightgbm

    monkeypatch.setattr(lightgbm, "train", forbidden)
    monkeypatch.setattr(ArtifactStore, "put", forbidden)
    before = {
        str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()
    }
    comparison.inspect_comparison(output, replay=True)
    comparison.resume_comparison(output, steps=1)
    after = {
        str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()
    }
    assert before == after


def test_no_protected_read_before_open(snapshot, monkeypatch):
    root, output = snapshot("reserved")
    original = ArtifactStore.get

    def guarded(store, key):
        assert store.root.name != "protected", "protected source read before OPENED"
        return original(store, key)

    monkeypatch.setattr(ArtifactStore, "get", guarded)
    while not comparison.resume_comparison(output, steps=1)["exposed"]:
        pass
    with evidence.Authority(root) as authority:
        assert authority.state["scoring_seconds_charged"] == 0


@pytest.mark.parametrize(
    "kind,after,expected_seconds",
    [
        ("score_attempt_debit_and_outbox", False, 40),
        ("score_attempt_debit_and_outbox", True, 40),
        ("score_dispatch_may_have_started", True, 60),
        ("score_attempt_finished", False, 40),
        ("score_attempt_finished", True, 40),
        ("comparison_completed", False, 40),
        ("comparison_completed", True, 40),
        ("exposed_batch_retired", False, 40),
    ],
)
def test_scoring_crash_windows(snapshot, monkeypatch, kind, after, expected_seconds):
    root, output = snapshot("opened")
    original = evidence.Authority.update
    fired = False

    def crash(authority, event, change):
        nonlocal fired
        selected = event == kind and not fired
        if selected and not after:
            fired = True
            raise RuntimeError("injected crash")
        result = original(authority, event, change)
        if selected and after:
            fired = True
            raise RuntimeError("injected crash")
        return result

    monkeypatch.setattr(evidence.Authority, "update", crash)
    with pytest.raises(RuntimeError, match="injected"):
        comparison.resume_comparison(output, steps=100)
    assert fired
    with pytest.raises(ValueError, match="fixed steps"):
        comparison.resume_comparison(output, steps=1)
    monkeypatch.setattr(evidence.Authority, "update", original)
    result = comparison.resume_comparison(output, steps=100)
    assert result["batch_state"] == "RETIRED"
    assert result["scoring_seconds_charged"] == expected_seconds
    comparison.inspect_comparison(output, replay=True)


def test_score_debit_outbox_rollback_is_atomic(snapshot, monkeypatch):
    root, output = snapshot("opened")
    original = evidence.Authority.update

    def trigger(authority, event, change):
        if event == "score_attempt_debit_and_outbox":
            with authority.connection:
                authority.connection.execute(
                    "CREATE TRIGGER fail_event BEFORE INSERT ON events "
                    "BEGIN SELECT RAISE(ABORT, 'injected transaction'); END"
                )
        return original(authority, event, change)

    monkeypatch.setattr(evidence.Authority, "update", trigger)
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        comparison.resume_comparison(output, steps=100)
    with evidence.Authority(root) as authority:
        assert authority.state["scoring_seconds_charged"] == 0
        assert authority.state["attempts"] == {}
        with authority.connection:
            authority.connection.execute("DROP TRIGGER fail_event")
    monkeypatch.setattr(evidence.Authority, "update", original)
    assert comparison.resume_comparison(output, steps=100)["scoring_seconds_charged"] == 40


def test_unknown_dispatch_has_one_charged_retry(snapshot, monkeypatch):
    root, output = snapshot("opened")
    monkeypatch.setattr(comparison.subprocess, "run", lambda *a, **k: None)
    result = comparison.resume_comparison(output, steps=100)
    assert result["scoring_seconds_charged"] == 80
    assert set(result["score_statuses"].values()) == {"infrastructure_failed"}
    assert result["report"]["evidence_status"] == "insufficient_evidence"
    assert result["report"]["failure_classification"] == "reference_unavailable"
    assert comparison.resume_comparison(output, steps=100) == result
    with evidence.Authority(root) as authority:
        assert len(authority.state["attempts"]) == 4


@pytest.mark.parametrize("bad", [b"broken", b'{"extra":1e309}'])
def test_bad_cached_output_is_not_retried_or_cash(snapshot, monkeypatch, bad):
    root, output = snapshot("opened")

    def malformed(args, **kwargs):
        Path(args[-2]).write_bytes(bad)

    monkeypatch.setattr(comparison.subprocess, "run", malformed)
    result = comparison.resume_comparison(output, steps=100)
    assert result["scoring_seconds_charged"] == 40
    assert set(result["score_statuses"].values()) == {"invalid"}
    assert result["report"]["weighted_delta"] is None
    with evidence.Authority(root) as authority:
        for attempt in authority.state["attempts"].values():
            assert authority.store.get(attempt["raw_ref"]) == bad


@pytest.mark.parametrize(
    "mutation",
    ["drop_key", "alias", "model", "train_cutoff", "positive", "mode", "registry", "loss_event"],
)
def test_registered_coverage_and_authority_reject_substitution(snapshot, mutation):
    root, output = snapshot("completed")
    with evidence.Authority(root) as authority:

        def mutate(state):
            batch = state["batch"]
            if mutation == "drop_key":
                batch["runs"].pop(next(iter(batch["runs"])))
            elif mutation in {"alias", "model"}:
                coverage = authority.store.json(batch["selections_ref"])
                if mutation == "alias":
                    coverage["logical"]["task_0:initial"] = coverage["logical"]["task_0:candidate"]
                else:
                    selection = authority.store.json(coverage["unique"]["task_0:parent"])
                    selection["model_ref"] = authority.store.put(b"substituted model")
                    coverage["unique"]["task_0:parent"] = authority.store.put_json(selection)
                batch["selections_ref"] = authority.store.put_json(coverage)
            elif mutation in {"train_cutoff", "mode", "loss_event"}:
                plan = authority.store.json(batch["plan_ref"])
                if mutation == "train_cutoff":
                    plan["training"]["cutoff"] = 39
                elif mutation == "mode":
                    plan["mode"] = "confirmatory"
                else:
                    plan["utility"]["registered_total_loss_events"] = [82]
                batch["plan_ref"] = authority.store.put_json(plan)
            elif mutation == "positive":
                report = authority.store.json(batch["report_ref"])
                report.update(
                    origin="production",
                    promotion_eligible=True,
                    promotion_decision="promote",
                    evidence_status="supported",
                )
                batch["report_ref"] = authority.store.put_json(report)
            else:
                state["registry"]["package_ref"] = authority.program["packages"]["candidate"]

        authority.update("injected_bad_binding", mutate)
        with pytest.raises(ValueError):
            comparison.verify_comparison(authority, replay=True)


@pytest.mark.parametrize(
    "missing", ["authority.json", "authority/control.sqlite", "model", "child_db"]
)
def test_initialized_missing_state_never_rebuilt(snapshot, missing, c5_snapshots):
    root, output = snapshot("completed")
    if missing in {"model", "child_db"}:
        with evidence.Authority(root) as authority:
            if missing == "child_db":
                path = comparison.child_root(authority, "task_0:parent") / "journal.sqlite"
            else:
                coverage = authority.store.json(authority.state["batch"]["selections_ref"])
                selection = authority.store.json(coverage["unique"]["task_0:parent"])
                assert selection["role"] == "active"
                path = authority.store.root / selection["model_ref"]
    else:
        path = root / missing
    path.unlink()
    with pytest.raises((ValueError, OSError, sqlite3.DatabaseError)):
        comparison.resume_comparison(output, steps=100)
    with pytest.raises((ValueError, OSError, sqlite3.DatabaseError)):
        if missing.startswith("authority"):
            evidence.init_evidence(c5_snapshots["spec"])
        else:
            comparison.inspect_comparison(output)
    assert not path.exists()


def test_cash_fallback_requires_authoritative_development_failure(snapshot, monkeypatch):
    root, output = snapshot("reserved")

    def failed(*args):
        raise RuntimeError("registered development proposer failure fixture")

    monkeypatch.setattr(researcher, "propose", failed)
    result = comparison.resume_comparison(output, steps=100)
    assert result["report"]["weighted_delta"] == 0.0
    with evidence.Authority(root) as authority:
        coverage = authority.store.json(authority.state["batch"]["selections_ref"])
        for selection_ref in coverage["unique"].values():
            selected = authority.store.json(selection_ref)
            assert selected["role"] == "cash_fallback"
            assert selected["model_ref"] is None
        assert result["scoring_seconds_charged"] == 40
    comparison.inspect_comparison(output, replay=True)


@pytest.mark.parametrize("stage", ["inside_schema", "before_anchor", "after_anchor"])
def test_bootstrap_recovers_same_identity(c5_snapshots, tmp_path, monkeypatch, stage):
    workspace = tmp_path / "bootstrap"
    workspace.mkdir()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    original_write = evidence.atomic_write
    original_connect = sqlite3.connect

    class CrashConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "INSERT INTO state VALUES (1,?)":
                raise RuntimeError("bootstrap crash")
            return super().execute(sql, *args, **kwargs)

    def write(path, raw):
        if path.name == "authority.json" and stage == "before_anchor":
            raise RuntimeError("bootstrap crash")
        original_write(path, raw)
        if path.name == "authority.json" and stage == "after_anchor":
            raise RuntimeError("bootstrap crash")

    if stage == "inside_schema":
        monkeypatch.setattr(
            sqlite3,
            "connect",
            lambda *a, **k: original_connect(*a, **(k | {"factory": CrashConnection})),
        )
    else:
        monkeypatch.setattr(evidence, "atomic_write", write)
    with pytest.raises(RuntimeError, match="bootstrap crash"):
        evidence.init_evidence(c5_snapshots["spec"])
    bootstrap = workspace / ".quant_rsi" / "bootstrap.json"
    identity = decode(bootstrap.read_bytes())["id"]
    monkeypatch.setattr(sqlite3, "connect", original_connect)
    monkeypatch.setattr(evidence, "atomic_write", original_write)
    result = evidence.init_evidence(c5_snapshots["spec"])
    assert result["authority_id"] == identity
    assert not bootstrap.exists()
    assert result["batch_state"] == "AVAILABLE"


def test_partial_child_manifest_bootstrap_recovers_same_grant(snapshot, monkeypatch):
    root, output = snapshot("reserved")
    original = research.ResearchJournal

    def crash(path, manifest_digest=None):
        if manifest_digest is not None:
            raise RuntimeError("child manifest before journal")
        return original(path)

    monkeypatch.setattr(research, "ResearchJournal", crash)
    with pytest.raises(RuntimeError, match="before journal"):
        comparison.resume_comparison(output, steps=1)
    with evidence.Authority(root) as authority:
        pending = authority.state["batch"]["pending"]
        child = comparison.child_root(authority, pending["key"])
        grant = authority.store.json(authority.state["batch"]["runs"][pending["key"]]["grant_ref"])
        assert (child / "manifest.json").exists()
        assert (child / "research_initialization.json").exists()
        assert not (child / "journal.sqlite").exists()
    monkeypatch.setattr(research, "ResearchJournal", original)
    comparison.resume_comparison(output, steps=1)
    with evidence.Authority(root) as authority:
        report = comparison.inspect_child(authority, pending["key"])
        assert report["campaign_id"] == grant["campaign_id"]
        assert len(report["results"]) == 1
        assert not (child / "research_initialization.json").exists()


def test_child_result_before_receipt_does_not_run_extra_observation(snapshot, monkeypatch):
    root, output = snapshot("reserved")
    original = evidence.Authority.update

    def crash(authority, kind, change):
        if kind == "operation_committed":
            raise RuntimeError("child result already committed")
        return original(authority, kind, change)

    monkeypatch.setattr(evidence.Authority, "update", crash)
    with pytest.raises(RuntimeError, match="already committed"):
        comparison.resume_comparison(output, steps=1)
    with evidence.Authority(root) as authority:
        key = authority.state["batch"]["pending"]["key"]
        before = comparison.inspect_child(authority, key)
        assert len(before["results"]) == 1
    monkeypatch.setattr(evidence.Authority, "update", original)
    comparison.resume_comparison(output, steps=1)
    with evidence.Authority(root) as authority:
        assert comparison.inspect_child(authority, key) == before


def test_scorer_orphan_keeps_authority_lock(snapshot, monkeypatch):
    import subprocess
    import sys

    root, output = snapshot("opened")
    original = subprocess.run
    children = []

    def orphan(args, **kwargs):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            pass_fds=kwargs["pass_fds"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
        )
        children.append(child)
        raise RuntimeError("controller lost after dispatch")

    monkeypatch.setattr(comparison.subprocess, "run", orphan)
    try:
        with pytest.raises(RuntimeError, match="controller lost"):
            comparison.resume_comparison(output, steps=100)
        with pytest.raises(ValueError, match="another controller"):
            comparison.resume_comparison(output, steps=100)
    finally:
        for child in children:
            child.wait(timeout=5)
    monkeypatch.setattr(comparison.subprocess, "run", original)
    assert comparison.resume_comparison(output, steps=100)["scoring_seconds_charged"] == 60


def test_configured_authority_survives_cwd_and_label_changes(authority, c5_snapshots, monkeypatch):
    workspace, output = authority
    before = evidence.init_evidence(c5_snapshots["spec"])
    monkeypatch.chdir(workspace)
    after = evidence.init_evidence(c5_snapshots["spec"].model_copy(update={"name": "a new label"}))
    assert before == after
    with pytest.raises(ValueError, match="cannot reseed"):
        evidence.init_evidence(c5_snapshots["spec"].model_copy(update={"task_count": 2}))
    monkeypatch.delenv("QUANT_RSI_WORKSPACE")
    with pytest.raises(ValueError, match="absolute trusted workspace"):
        evidence.authority_root()


def test_four_task_grants_and_nonoverlapping_grid(c5_snapshots, tmp_path, monkeypatch):
    workspace = tmp_path / "four_tasks"
    workspace.mkdir()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    evidence.init_evidence(c5_snapshots["spec"].model_copy(update={"task_count": 4}))
    result = comparison.register_comparison(
        ComparisonInput(name="four", mode="engineering_only"), tmp_path / "handle"
    )
    assert result["development_attempts_granted"] == 64
    assert result["development_seconds_granted"] == 1280
    with evidence.Authority(workspace / ".quant_rsi") as authority:
        plan = comparison.compiled_plan(authority)
        coverage = plan["expected_coverage"]
        assert len(coverage["unique"]) == 8
        assert len(coverage["logical"]) == 12
        all_dates = [day for task in plan["tasks"] for day in task["sessions"]]
        assert len(set(all_dates)) == 4 * 122
        assert sum(task["weight"] for task in plan["tasks"]) == 1.0
        assert authority.program["allowance"]["scoring_seconds"] == 320
        for task in plan["tasks"]:
            view = authority.store.json(task["view_ref"])
            assert len(view["data"]["dates"]) == 80
            assert len(view["data"]["opens"]) == 80
            assert "seed" not in view
            assert coverage["logical"][task["task"] + ":initial"] == task["task"] + ":parent"
