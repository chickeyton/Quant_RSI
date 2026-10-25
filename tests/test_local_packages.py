"""Local CSV coverage fixtures, not market or statistical improvement evidence."""

import json
import shutil
import subprocess
from pathlib import Path

import lightgbm as lgb
import pytest
from market_helpers import account_rules, contract_and_events, write_events

from quant_rsi import (
    changes,
    controller,
    evidence,
    local_research,
    market_data,
    research,
    researcher,
)
from quant_rsi import package_experiment as packages
from quant_rsi.artifacts import ArtifactStore, canonical, decode, digest
from quant_rsi.baseline import infer_at, seal_baseline
from quant_rsi.data import synthetic_data
from quant_rsi.package_journal import PackageJournal
from quant_rsi.research_journal import ResearchJournal
from quant_rsi.researcher import ResearchPlan
from quant_rsi.schemas import DataSpec


def forbidden(*args, **kwargs):
    pytest.fail("unexpected fit, proposal, publication or live data export")


def files(root):
    return {
        str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()
    }


@pytest.fixture(scope="module")
def local_c4_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("local_c4_coverage")
    value = json.loads(Path("examples/research_development.json").read_text())
    value["development"]["data"]["seed"] = 0
    # Explicitly invented CSV: this exercises the local ingestion path, not a provider claim.
    prices = synthetic_data(DataSpec.model_validate_json(canonical(value["development"]["data"])))
    symbols = tuple("ABCD")
    contract, events, availability = contract_and_events(instruments=symbols, count=80)
    for event in events:
        if event["session_id"] is not None:
            i, j = int(event["session_id"][1:]), symbols.index(event["instrument"])
            quotes = prices.opens if event["kind"] == "OPEN_QUOTE" else prices.closes
            event["price"] = f"{quotes[i, j]:.10f}"
    availability_path = root / "availability.txt"
    availability_path.write_bytes(availability)
    csv_path = write_events(root / "invented.csv", contract, events)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QUANT_RSI_WORKSPACE", str(root))
        imported = market_data.ingest(
            contract, csv_path, request_id="local-c4-fixture", evidence_file=availability_path
        )
        value["development"]["data"].update(
            source="local_event_snapshot",
            snapshot_ref=imported["snapshot_ref"],
            decision_delay_seconds=60,
        )
        costs = dict(commission_bps="2", tax_bps="0", slippage_bps="3")
        value["development"]["portfolio"] = account_rules(
            initial_cash="10000",
            allocation="0.9",
            max_positions=2,
            buy_costs=costs,
            sell_costs=costs,
            fill_cap_units={symbol: 100000 for symbol in symbols},
            max_drawdown="0.25",
        ).model_dump(mode="json")
        plan = ResearchPlan.model_validate_json(canonical(value))
        source = research.create_research(plan, root / "source", steps=100)
        assert source["attempts_reserved"] == 4
        assert source["worker_seconds_charged"] == 80

        def interrupt(*args, **kwargs):
            raise SystemExit("after actual paired grants, before first child initialization")

        patch.setattr(research, "create_research", interrupt)
        with pytest.raises(SystemExit):
            packages.create_package(
                root / "source", root / "reserved", state_policy="reset", steps=100
            )
    return dict(root=root, plan=plan, source=source, contract=contract, events=events)


def clone_reserved(fixture, tmp_path):
    root = tmp_path / "package"
    shutil.copytree(fixture["root"] / "reserved", root)
    return root


def local_workspace(fixture, tmp_path, monkeypatch):
    shutil.copytree(fixture["root"] / ".quant_rsi", tmp_path / ".quant_rsi")
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))


def test_frozen_pair_after_original_inputs_disappear_and_actual_descendant_seal(
    local_c4_fixture, tmp_path, monkeypatch
):
    fixture = local_c4_fixture
    local_workspace(fixture, tmp_path, monkeypatch)
    root = clone_reserved(fixture, tmp_path)
    # The complete evidence copy is authoritative; the original C2 can disappear.
    original = fixture["root"] / "source"
    unavailable = fixture["root"] / "source-unavailable"
    original.rename(unavailable)
    csv_path = fixture["root"] / "invented.csv"
    csv_path.rename(fixture["root"] / "invented-unavailable.csv")
    correction = next(e for e in fixture["events"] if e["kind"] == "CLOSE_QUOTE") | {
        "revision": 1,
        "price": "999",
        "source_available_at": "2021-02-01T16:00:00Z",
    }
    market_data.ingest(
        fixture["contract"],
        write_events(tmp_path / "revision.csv", fixture["contract"], [correction]),
        request_id="later-revision",
        evidence_file=fixture["root"] / "availability.txt",
    )
    market_root = market_data.market_root()
    offline = market_root.with_name("market-offline")
    market_root.rename(offline)
    try:
        with monkeypatch.context() as blocked:
            blocked.setattr(local_research, "export_local_data", forbidden)
            report = packages.resume_package(root, steps=100)
    finally:
        offline.rename(market_root)
        unavailable.rename(original)
        (fixture["root"] / "invented-unavailable.csv").rename(csv_path)
    assert report["execution_state"] == "completed"
    assert report["decision"]["outcome"] == "patch"
    assert report["historical_evidence_cost"]["historical_attempts"] == 4
    assert report["historical_evidence_cost"]["historical_worker_slots"] == 80
    source_manifest = decode((root / "evidence/manifest.json").read_bytes())
    source_store = ArtifactStore(root / "evidence/artifacts")
    source_first = source_store.json(fixture["source"]["results"][0]["observation"]["result_ref"])
    for arm, arm_report in report["arms"].items():
        path = root / "arms" / arm
        manifest = decode((path / "manifest.json").read_bytes())
        assert manifest["data_ref"] == source_manifest["data_ref"]
        assert manifest["task_ref"] == source_manifest["task_ref"]
        store = ArtifactStore(path / "artifacts")
        arm_plan = store.json(manifest["research_plan_ref"])
        assert arm_plan["development"] == fixture["plan"].development.model_dump(mode="json")
        first = store.json(arm_report["results"][0]["observation"]["result_ref"])
        assert first["model_ref"] == source_first["model_ref"]
        assert first["predictions_ref"] == source_first["predictions_ref"]
    budget = report["research_budget"]
    assert (
        budget["attempts_granted"]
        == budget["attempts_consumed"] + budget["attempts_outstanding"]
        == 16
    )
    assert (
        budget["worker_slots_granted"]
        == budget["worker_slots_consumed"] + budget["worker_slots_outstanding"]
        == 320
    )
    assert budget["worker_slots_consumed"] == 20 * budget["attempts_consumed"]
    candidate = report["arms"]["candidate"]["incumbent"]
    sealed = seal_baseline(root / "arms/candidate")
    assert (
        sealed["model_ref"]
        == ArtifactStore(root / "arms/candidate/artifacts").json(candidate["result_ref"])[
            "model_ref"
        ]
    )
    assert sealed["training_information_cutoff"] < sealed["actual_fit_completed_at"]
    before = files(tmp_path)
    with monkeypatch.context() as blocked:
        for target, name in (
            (lgb, "train"),
            (changes, "propose_change"),
            (researcher, "propose"),
            (controller.subprocess, "run"),
            (ArtifactStore, "put"),
        ):
            blocked.setattr(target, name, forbidden)
        replay = packages.inspect_package(root, replay=True)
        assert replay["comparison"] == report["comparison"]
        for arm in ("parent", "candidate"):
            research.inspect_research(root / "arms" / arm, replay=True)
        predicted = infer_at(
            sealed["bundle_ref"],
            fixture["plan"].development.data.snapshot_ref,
            "s79",
            as_of="2021-01-06T16:01:00Z",
        )
        assert predicted["usage"] == "post_fit_historical_exploration"
        assert len(predicted["prediction"]["rows"]) == 4
    assert files(tmp_path) == before
    # Exact legacy dimensions/budget cannot send local input into a synthetic C5 program.
    with monkeypatch.context() as blocked:
        blocked.setattr(evidence, "synthetic_data", forbidden)
        with pytest.raises(ValueError, match="explicit local C5 task program"):
            evidence.import_packages(root)
        clean_workspace = tmp_path / "clean-c5-authority"
        clean_workspace.mkdir()
        blocked.setenv("QUANT_RSI_WORKSPACE", str(clean_workspace))
        with pytest.raises(ValueError, match="explicit local C5 task program"):
            evidence.init_evidence(
                evidence.ProgramInput(
                    name="local-refusal", package_campaign=str(root), task_count=1
                )
            )
        authority = evidence.authority_root()
    assert not (authority / "bootstrap.json").exists()
    assert not (authority / "authority.json").exists()


@pytest.mark.parametrize(
    "mutation", ["campaign", "grant", "arm", "path", "rules", "budget", "foreign_fd", "no_fd"]
)
def test_local_grant_requires_actual_owner_identity_and_inputs(
    local_c4_fixture, tmp_path, mutation
):
    root = clone_reserved(local_c4_fixture, tmp_path)
    with controller.controller_lock(root) as owner:
        journal, store, manifest, _ = packages.open_package(root)
        try:
            row = next(r for r in journal.grants() if r["arm"] == "parent")
            grant = store.json(row["grant_ref"])
            context = packages.launch_context(manifest, "parent", row["grant_ref"])
            plan = ResearchPlan.model_validate_json(store.get(grant["research_plan_ref"]))
            path, campaign, fd = root / grant["directory"], grant["campaign_id"], owner
            with (root / ".controller.lock").open("r+") as foreign:
                if mutation == "campaign":
                    campaign = "0" * 32
                elif mutation == "grant":
                    context["grant_ref"] = "0" * 64
                elif mutation == "arm":
                    context["arm"] = "candidate"
                elif mutation == "path":
                    path = root / "arms/unallocated"
                elif mutation in {"rules", "budget"}:
                    value = plan.model_dump(mode="json")
                    if mutation == "rules":
                        value["development"]["portfolio"]["allocation"] = "0.8"
                    else:
                        value["development"]["budget"].update(
                            max_attempts=7, max_worker_seconds=140
                        )
                    plan = ResearchPlan.model_validate_json(canonical(value))
                elif mutation == "foreign_fd":
                    fd = foreign.fileno()
                else:
                    fd = None
                before = files(root)
                with pytest.raises(ValueError, match="local C4"):
                    research.create_research(
                        plan,
                        path,
                        campaign_id=campaign,
                        launch_context=context,
                        owner_fd=fd,
                        initialize_only=True,
                    )
                assert files(root) == before
                assert not path.exists()
        finally:
            journal.close()


@pytest.mark.parametrize("protected", [False, True])
def test_local_never_accepts_arbitrary_or_protected_context(local_c4_fixture, tmp_path, protected):
    plan = local_c4_fixture["plan"]
    kwargs = {"launch_context": {"kind": "ProtectedComparisonDevelopmentGrant"}}
    if protected:
        kwargs["bound_view"] = {"kind": "BoundDevelopmentView", "origin_ref": "0" * 64, "data": {}}
    with pytest.raises(
        ValueError, match="protected bound data|actual C4 grant|distinct actual authority grant"
    ):
        research.create_research(plan, tmp_path / "forged", **kwargs)
    assert not (tmp_path / "forged").exists()


@pytest.mark.parametrize("stage", ["child_manifest", "child_observation", "completion_response"])
def test_local_faults_preserve_grants_inputs_and_committed_cost(
    local_c4_fixture, tmp_path, monkeypatch, stage
):
    root = clone_reserved(local_c4_fixture, tmp_path)
    with monkeypatch.context() as broken:
        if stage == "child_manifest":
            original = ResearchJournal.__init__

            def interrupt(self, path, *args, **kwargs):
                if Path(path) == root / "arms/parent":
                    assert (Path(path) / "manifest.json").exists()
                    raise SystemExit("child manifest published")
                original(self, path, *args, **kwargs)

            broken.setattr(ResearchJournal, "__init__", interrupt)
        elif stage == "child_observation":
            original = research.resume_research

            def interrupt(*args, **kwargs):
                kwargs["steps"] = 1
                original(*args, **kwargs)
                raise SystemExit("one actual local observation committed")

            broken.setattr(research, "resume_research", interrupt)
        else:
            original = PackageJournal.complete

            def interrupt(self, ref):
                original(self, ref)
                raise SystemExit("completion response lost")

            broken.setattr(PackageJournal, "complete", interrupt)
        with pytest.raises(SystemExit):
            packages.resume_package(root, steps=100)
    journal = PackageJournal(root)
    grants = journal.grants()
    rounds_before = journal.rounds()
    journal.close()
    before_attempts = {}
    for arm in ("parent", "candidate"):
        path = root / "arms" / arm
        if (path / "journal.sqlite").exists():
            child = ResearchJournal(path)
            before_attempts[arm] = child.attempts()
            child.close()
    with monkeypatch.context() as blocked:
        blocked.setattr(changes, "propose_change", forbidden)
        blocked.setattr(local_research, "export_local_data", forbidden)
        if stage == "completion_response":
            blocked.setattr(controller.subprocess, "run", forbidden)
        result = packages.resume_package(root, steps=100)
    assert result["execution_state"] == "completed"
    journal = PackageJournal(root)
    assert [(r["arm"], r["grant_ref"]) for r in journal.grants()] == [
        (r["arm"], r["grant_ref"]) for r in grants
    ]
    assert len(journal.rounds()) == len(rounds_before) == 1
    journal.close()
    for arm, attempts in before_attempts.items():
        child = ResearchJournal(root / "arms" / arm)
        assert child.attempts()[: len(attempts)] == attempts
        child.close()
    budget = result["research_budget"]
    assert budget["attempts_consumed"] == sum(r["kernel_attempts"] for r in result["arms"].values())
    assert budget["worker_slots_consumed"] == 20 * budget["attempts_consumed"]
    assert budget["attempts_granted"] == 16


def test_local_dispatch_failure_is_charged_once_and_has_no_score(
    local_c4_fixture, tmp_path, monkeypatch
):
    root = clone_reserved(local_c4_fixture, tmp_path)
    with monkeypatch.context() as broken:
        broken.setattr(
            controller.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 7)
        )
        result = packages.resume_package(root, steps=100)
    assert result["execution_state"] == "completed"
    assert result["research_budget"]["attempts_consumed"] == 2
    assert result["research_budget"]["worker_slots_consumed"] == 40
    assert result["research_budget"]["attempts_outstanding"] == 14
    for arm in result["arms"].values():
        observation = arm["results"][0]["observation"]
        assert observation["status"] == "failed" and observation["score"] is None
        assert "worker_exit_7" in observation["error"]
        assert arm["incumbent"] is None
    before = files(root)
    with monkeypatch.context() as blocked:
        blocked.setattr(controller.subprocess, "run", forbidden)
        replay = packages.resume_package(root, steps=100)
        assert replay["research_budget"] == result["research_budget"]
        packages.inspect_package(root, replay=True)
    assert files(root) == before


def test_local_round_target_rejects_extra_work_before_dispatch(
    local_c4_fixture, tmp_path, monkeypatch
):
    root = tmp_path / "package"
    original = research.resume_research
    calls = []

    def check(path, **kwargs):
        with pytest.raises(ValueError, match="round target"):
            original(path, **(kwargs | {"steps": 2}))
        calls.append(Path(path).name)
        return original(path, **kwargs)

    monkeypatch.setattr(research, "resume_research", check)
    result = packages.create_package(
        local_c4_fixture["root"] / "source", root, state_policy="reset", steps=1
    )
    assert calls == ["parent", "candidate"]
    assert result["research_budget"]["attempts_consumed"] == 2
    assert result["research_budget"]["worker_slots_consumed"] == 40
    assert all(arm["committed_observations"] == 1 for arm in result["arms"].values())


def test_copied_local_source_tampering_fails_closed(local_c4_fixture, tmp_path):
    root = clone_reserved(local_c4_fixture, tmp_path)
    manifest = decode((root / "evidence/manifest.json").read_bytes())
    store = ArtifactStore(root / "evidence/artifacts")
    path = store.root / manifest["data_ref"]
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="snapshot content changed"):
        packages.resume_package(root)
    assert not (root / "arms").exists()
