"""Local C5 protocol fixtures are invented CSV engineering coverage, not alpha evidence."""

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from market_helpers import account_rules, contract_and_events, write_events

from quant_rsi.core.artifacts import ArtifactStore, canonical, digest
from quant_rsi.core.data import synthetic_data
from quant_rsi.core.schemas import DataSpec
from quant_rsi.evaluation import comparison, evidence, local_research
from quant_rsi.evaluation.evidence import ComparisonInput
from quant_rsi.evaluation.local_evidence import LocalProgramInput
from quant_rsi.market import market_data
from quant_rsi.research import package_experiment, research
from quant_rsi.research.researcher import ResearchPlan


@pytest.fixture(scope="module")
def local_c5(tmp_path_factory):
    root = tmp_path_factory.mktemp("local_c5_actual")
    value = json.loads(Path("examples/research_development.json").read_text())
    value["development"]["data"]["seed"] = 0
    prices = synthetic_data(
        DataSpec.model_validate_json(canonical(value["development"]["data"] | {"sessions": 104}))
    )
    symbols = tuple("ABCD")
    contract, events, availability = contract_and_events(instruments=symbols, count=104)
    for event in events:
        if event["session_id"] is not None:
            i, j = int(event["session_id"][1:]), symbols.index(event["instrument"])
            quotes = prices.opens if event["kind"] == "OPEN_QUOTE" else prices.closes
            event["price"] = f"{quotes[i, j]:.10f}"
    availability_path = root / "availability.txt"
    availability_path.write_bytes(availability)
    csv_path = write_events(root / "invented.csv", contract, events)
    workspace = root / "workspace"
    workspace.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
        imported = market_data.ingest(
            contract, csv_path, request_id="local-c5-fixture", evidence_file=availability_path
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
            termination="preserve_inventory",
        ).model_dump(mode="json")
        plan = ResearchPlan.model_validate_json(canonical(value))
        source = research.create_research(plan, root / "source", steps=100)
        assert source["attempts_reserved"] == 4
        package = package_experiment.create_package(
            root / "source", root / "package", state_policy="reset", steps=100
        )
        assert package["decision"]["outcome"] == "patch"
        spec = LocalProgramInput.model_validate_json(
            canonical(
                dict(
                    name="local engineering fixture",
                    source="local_event_snapshot",
                    package_campaign=str(root / "package"),
                    tasks=[
                        dict(
                            task_id="task_0",
                            data=value["development"]["data"] | {"sessions": 104},
                            development_sessions=80,
                            score_session_ids=[
                                s.session_id for s in contract.calendar.sessions[80:]
                            ],
                            terminal_session_id=contract.calendar.sessions[-1].session_id,
                            weight="1",
                            economic_group="invented_market",
                            dependence_cluster="unverified_common_cluster",
                            prior_exposure="public_or_previously_exposed_engineering",
                        )
                    ],
                )
            )
        )
        evidence.init_evidence(spec)
        shutil.copytree(workspace / ".quant_rsi", root / "available")
        output = root / "handle"
        comparison.register_comparison(
            ComparisonInput(name="local", mode="engineering_only"), output
        )
        shutil.copytree(workspace / ".quant_rsi", root / "reserved")
        while True:
            status = comparison.resume_comparison(output, steps=1)
            if status["selection_coverage_ref"] and not status["exposed"]:
                shutil.copytree(workspace / ".quant_rsi", root / "sealed")
            if status["exposed"]:
                shutil.copytree(workspace / ".quant_rsi", root / "opened")
                break
        report = comparison.resume_comparison(output, steps=100)
        shutil.copytree(workspace / ".quant_rsi", root / "completed")
    return dict(
        root=root,
        workspace=workspace,
        spec=spec,
        report=report,
        output=output,
        contract=contract,
        events=events,
    )


def restored(local_c5, tmp_path, monkeypatch, state="completed"):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(local_c5["root"] / state, workspace / ".quant_rsi")
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    return workspace


def test_local_actual_development_selection_scoring_and_retain(local_c5, tmp_path, monkeypatch):
    restored(local_c5, tmp_path, monkeypatch)
    output = local_c5["output"]
    result = comparison.inspect_comparison(output, replay=True)
    assert result["batch_state"] == "RETIRED"
    assert result["development_attempts_granted"] == 16
    assert result["scoring_seconds_charged"] == 40
    assert result["report"]["promotion_eligible"] is False
    assert result["report"]["synthetic_only"] is False
    assert result["report"]["local_engineering_only"] is True
    assert (
        result["report"]["origin"]
        == "actual_local_development_selection_and_registered_decimal_account"
    )
    assert (
        result["report"]["source_qualifications"]["task_0"]["scientific_independence"]
        == "unqualified"
    )
    with evidence.Authority(evidence.authority_root()) as authority:
        plan = authority.store.json(authority.state["batch"]["plan_ref"])
        assert plan["utility"]["return_count_by_task"] == {"task_0": 24}
        assert plan["training"]["maturity"] == "actual_per_row_knowledge_and_availability_mask"
        for key, score in authority.state["batch"]["scores"].items():
            scored = authority.store.json(score["result_ref"])
            assert scored["status"] == "completed", scored["error"]
            assert scored["return_count"] == 24
            assert len(scored["accounting"]["account"]["valuations"]) == 24
            assert len(scored["predictions"]) == 23
            assert scored["consumed_evidence"]["rows"][-1]["feature_view"] is None
            assert Decimal(scored["accounting"]["net_returns"][0]) == 0
            selection = authority.store.json(
                authority.store.json(authority.state["batch"]["selections_ref"])["unique"][key]
            )
            training = authority.store.json(selection["training_policy_ref"])
            mask = authority.store.json(training["actual_mask_ref"])
            assert sum(r["training_eligible"] for r in mask["rows"]) == training["training_rows"]
            child = comparison.child_root(authority, key)
            store = ArtifactStore(child / "artifacts")
            view = authority.store.json(plan["tasks"][0]["view_ref"])
            assert view["data"]["kind"] == "DerivedLocalResearchData"
            assert {
                r["event"]["session_id"]
                for r in view["data"]["snapshot"]["events"]
                if r["event"]["kind"] in {"OPEN_QUOTE", "CLOSE_QUOTE"}
            } <= set(plan["tasks"][0]["sessions"][:80])
            for source_ref in [
                plan["tasks"][0]["source_ref"],
                *plan["tasks"][0]["original_raw_refs"],
            ]:
                assert not (store.root / source_ref).exists()
    decision = comparison.decide_comparison(output, "retain-local")
    assert decision["decision"] == "retain"
    assert comparison.decide_comparison(output, "retain-local") == decision


def test_local_readonly_replay_uses_only_sealed_consumption(local_c5, tmp_path, monkeypatch):
    workspace = restored(local_c5, tmp_path, monkeypatch)
    before = {
        str(p.relative_to(workspace)): digest(p.read_bytes())
        for p in workspace.rglob("*")
        if p.is_file()
    }
    original = ArtifactStore.get

    def no_protected(self, ref):
        assert self.root.name != "protected", "read-only replay reopened protected originals"
        return original(self, ref)

    monkeypatch.setattr(ArtifactStore, "get", no_protected)

    def forbidden(*args, **kwargs):
        pytest.fail("readonly local C5 attempted fitting, creation, scoring or publication")

    monkeypatch.setattr(local_research, "fit_predict_local", forbidden)
    monkeypatch.setattr(research, "create_research", forbidden)
    monkeypatch.setattr(comparison, "compute_score", forbidden)
    monkeypatch.setattr(ArtifactStore, "put", forbidden)
    result = comparison.inspect_comparison(local_c5["output"], replay=True)
    assert result["batch_state"] == "RETIRED"
    after = {
        str(p.relative_to(workspace)): digest(p.read_bytes())
        for p in workspace.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_local_confirmatory_rejected_before_grant_or_protected_read(
    local_c5, tmp_path, monkeypatch
):
    workspace = restored(local_c5, tmp_path, monkeypatch, "available")
    before = {
        str(p.relative_to(workspace)): digest(p.read_bytes())
        for p in workspace.rglob("*")
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("confirmatory preflight accessed authority artifacts")

    monkeypatch.setattr(ArtifactStore, "get", forbidden)
    with pytest.raises(ValueError, match="no protected access or grant"):
        comparison.register_comparison(
            ComparisonInput(name="forbidden", mode="confirmatory"), tmp_path / "output"
        )
    after = {
        str(p.relative_to(workspace)): digest(p.read_bytes())
        for p in workspace.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize("field", ["revision", "development_cost"])
def test_local_registration_refuses_corrupt_registry_or_existing_unallocated_cost(
    local_c5, tmp_path, monkeypatch, field
):
    restored(local_c5, tmp_path, monkeypatch, "available")
    with evidence.locked_authority() as (root, _), evidence.Authority(root) as authority:
        if field == "revision":
            authority.update("test_only_corruption", lambda s: s["registry"].update(revision=1))
        else:
            authority.update(
                "test_only_corruption",
                lambda s: s.update(development_attempts_granted=1, development_seconds_granted=20),
            )
        before = authority.state
    with pytest.raises(ValueError, match="registry|cannot be reset"):
        comparison.register_comparison(
            ComparisonInput(name="cannot reset", mode="engineering_only"), tmp_path / "new"
        )
    with evidence.Authority(root) as authority:
        assert authority.state == before
        assert authority.state["batch"] is None


def test_local_grant_is_distinct_and_requires_actual_pending_owner(local_c5, tmp_path, monkeypatch):
    restored(local_c5, tmp_path, monkeypatch, "reserved")
    original = research.create_research

    def interrupt(*args, **kwargs):
        raise RuntimeError("unit interruption before child initialization")

    monkeypatch.setattr(research, "create_research", interrupt)
    with pytest.raises(RuntimeError, match="unit interruption"):
        comparison.resume_comparison(local_c5["output"], steps=1)
    monkeypatch.setattr(research, "create_research", original)
    from quant_rsi.evaluation.local_evidence import authorized_view

    with evidence.locked_authority() as (root, fd), evidence.Authority(root) as authority:
        key = authority.state["batch"]["pending"]["key"]
        grant = authority.store.json(authority.state["batch"]["runs"][key]["grant_ref"])
        plan = ResearchPlan.model_validate_json(authority.store.get(grant["research_plan_ref"]))
        context = comparison.launch_context(authority, key)
        path = comparison.child_root(authority, key)
        view = authorized_view(path, plan, grant["campaign_id"], context, fd)
        assert digest(canonical(view)) == grant["view_ref"]
        for field in ("kind", "authority_id", "batch_id", "key", "grant_ref"):
            with pytest.raises(ValueError):
                original(
                    plan,
                    path,
                    campaign_id=grant["campaign_id"],
                    launch_context=context | {field: "wrong"},
                    owner_fd=fd,
                    bound_view=view,
                    initialize_only=True,
                )
            assert not path.exists()
        for bad_fd in (None, -1):
            with pytest.raises(ValueError, match="owning authority lock"):
                authorized_view(path, plan, grant["campaign_id"], context, bad_fd)
        with pytest.raises(ValueError, match="distinct actual authority grant"):
            authorized_view(
                path,
                plan,
                grant["campaign_id"],
                dict(
                    package_campaign_id="C4",
                    arm="parent",
                    grant_ref=context["grant_ref"],
                    state_policy="reset",
                ),
                fd,
            )
        document = plan.model_dump(mode="json")
        document["development"]["budget"]["max_attempts"] = 7
        with pytest.raises(ValueError, match="child/package/input"):
            authorized_view(
                path,
                ResearchPlan.model_validate_json(canonical(document)),
                grant["campaign_id"],
                context,
                fd,
            )


@pytest.mark.parametrize("case", ["unknown", "atomic_rollback"])
def test_local_score_protocol_mock_failure_accounting(local_c5, tmp_path, monkeypatch, case):
    # These are transaction/mocked-dispatch tests, not real process-crash evidence.
    import test_comparison as protocol

    def snapshot(state):
        workspace = restored(local_c5, tmp_path, monkeypatch, state)
        return workspace / ".quant_rsi", local_c5["output"]

    if case == "unknown":
        protocol.test_unknown_dispatch_has_one_charged_retry(snapshot, monkeypatch)
    else:
        protocol.test_score_debit_outbox_rollback_is_atomic(snapshot, monkeypatch)


def test_local_source_validation_receipt_substitution_fails_readonly(
    local_c5, tmp_path, monkeypatch
):
    restored(local_c5, tmp_path, monkeypatch)
    with evidence.locked_authority() as (root, _), evidence.Authority(root) as authority:
        attempt_id = next(iter(authority.state["attempts"]))
        attempt = authority.state["attempts"][attempt_id]
        receipt = authority.store.json(attempt["source_validation_receipt_ref"])
        wrong = authority.store.put_json(receipt | {"source_ref": "0" * 64})
        authority.update(
            "test_only_receipt_substitution",
            lambda s: s["attempts"][attempt_id].update(source_validation_receipt_ref=wrong),
        )
    with pytest.raises(ValueError, match="validation receipt binding mismatch"):
        comparison.inspect_comparison(local_c5["output"], replay=True)


def test_direct_local_replay_entrypoints_cannot_reopen_protected_originals(
    local_c5, tmp_path, monkeypatch
):
    from quant_rsi.evaluation.local_scoring import local_score
    from quant_rsi.evaluation.protected_scoring import compute_score, read_episode

    workspace = restored(local_c5, tmp_path, monkeypatch)
    before = {
        str(p.relative_to(workspace)): digest(p.read_bytes())
        for p in workspace.rglob("*")
        if p.is_file()
    }
    original = ArtifactStore.get

    def no_protected(self, ref):
        assert self.root.name != "protected", "direct replay reopened protected originals"
        return original(self, ref)

    monkeypatch.setattr(ArtifactStore, "get", no_protected)
    with evidence.Authority(evidence.authority_root()) as authority:
        attempt_id = next(iter(authority.state["attempts"]))
        expected = authority.store.json(authority.state["attempts"][attempt_id]["result_ref"])
        assert compute_score(authority, attempt_id, replay=True) == expected
        assert local_score(authority, attempt_id, replay=True) == expected
        with pytest.raises(ValueError, match="sealed evidence"):
            read_episode(authority, attempt_id, replay=True)
        with pytest.raises(ValueError, match="not running"):
            read_episode(authority, attempt_id)
    after = {
        str(p.relative_to(workspace)): digest(p.read_bytes())
        for p in workspace.rglob("*")
        if p.is_file()
    }
    assert before == after
