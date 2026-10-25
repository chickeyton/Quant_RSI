from pathlib import Path

import lightgbm as lgb
import pytest
from market_helpers import account_rules, contract_and_events, write_events

from quant_rsi import controller, market_data, research, researcher
from quant_rsi.artifacts import ArtifactStore, canonical, digest
from quant_rsi.local_research import LocalResearchData, export_local_data, training_view
from quant_rsi.researcher import ResearchPlan
from quant_rsi.schemas import FactorSpec


@pytest.fixture
def local_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    contract, events, evidence = contract_and_events(instruments=("A", "B"), count=32)
    evidence_path = tmp_path / "availability.txt"
    evidence_path.write_bytes(evidence)
    report = market_data.ingest(
        contract,
        write_events(tmp_path / "market.csv", contract, events),
        request_id="local-research",
        evidence_file=evidence_path,
    )
    value = ResearchPlan.model_validate_json(
        Path("examples/research_development.json").read_bytes()
    ).model_dump(mode="json")
    value["development"]["data"] = dict(
        source="local_event_snapshot",
        sessions=32,
        instruments=2,
        train_cutoff=20,
        seed=7,
        snapshot_ref=report["snapshot_ref"],
        decision_delay_seconds=60,
    )
    value["development"]["portfolio"] = account_rules().model_dump(mode="json")
    value["development"]["proposals"][0]["factor"] = dict(op="momentum", window=2)
    value["development"]["budget"].update(max_attempts=2, max_worker_seconds=40)
    value["policy"].update(window_min=1, window_max=3, max_no_progress=2)
    return ResearchPlan.model_validate_json(canonical(value))


def test_real_inputs_use_same_finite_c2_kernel_and_readonly_replay(
    local_plan, tmp_path, monkeypatch
):
    root = tmp_path / "research"
    first = research.create_research(local_plan, root, steps=1)
    assert first["data_source"] == "local_event_snapshot"
    assert first["kernel_attempts"] == 1 and first["worker_seconds_charged"] == 20
    assert first["results"][0]["observation"]["status"] == "completed", first["results"]
    assert first["incumbent"] is not None
    with monkeypatch.context() as locked:

        def forbidden(*args, **kwargs):
            pytest.fail("read-only local replay attempted fit/proposal/publication")

        locked.setattr(lgb, "train", forbidden)
        locked.setattr(researcher, "propose", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        locked.setattr(ArtifactStore, "put", forbidden)
        before = {
            str(path): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()
        }
        replay = research.inspect_research(root, replay=True)
        assert replay["verified_experiments"] == 1
        assert before == {
            str(path): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()
        }
    second = research.resume_research(root, steps=1)
    assert second["kernel_attempts"] == 2 and second["worker_seconds_charged"] == 40
    assert second["results"][1]["parent_ref"] == first["checkpoint_ref"]


def test_local_training_information_age_is_not_refreshed_by_import(local_plan):
    value = export_local_data(local_plan.development.data)
    data = LocalResearchData.from_json(value)
    _, _, metadata, view = training_view(
        data, local_plan.development, FactorSpec(op="momentum", window=2)
    )
    assert metadata["training_rows"] == 34
    assert metadata["training_information_cutoff"] == "2020-11-08T09:00:00.000000+00:00"
    assert metadata["training_view_asof"] == "2020-11-08T16:01:00.000000+00:00"
    assert sum(row["training_eligible"] for row in view["rows"]) == 34
    tampered = value | {
        "raw_inputs": {
            key: text.replace("100", "999", 1) for key, text in value["raw_inputs"].items()
        }
    }
    with pytest.raises(ValueError, match="raw input hash"):
        LocalResearchData.from_json(tampered)


def test_synthetic_generator_refuses_local_input(local_plan):
    from quant_rsi.data import synthetic_data

    with pytest.raises(ValueError, match="cannot be replaced"):
        synthetic_data(local_plan.development.data)


def test_actual_incumbent_seals_and_infers_last_session_without_fit(
    local_plan, tmp_path, monkeypatch
):
    from quant_rsi.baseline import infer_at, inspect_baseline, seal_baseline

    root = tmp_path / "research"
    outcome = research.create_research(local_plan, root, steps=1)
    sealed = seal_baseline(root)
    assert sealed["source_checkpoint_ref"] == outcome["checkpoint_ref"]
    assert seal_baseline(root) == sealed
    assert sealed["training_information_cutoff"].startswith("2020-")
    assert sealed["actual_fit_completed_at"] > sealed["training_information_cutoff"]
    with monkeypatch.context() as locked:

        def forbidden(*args, **kwargs):
            pytest.fail("baseline inference/replay attempted fit, proposal or write")

        locked.setattr(lgb, "train", forbidden)
        locked.setattr(researcher, "propose", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        locked.setattr(ArtifactStore, "put", forbidden)
        market_root = market_data.market_root()
        before = {
            str(path): digest(path.read_bytes())
            for path in market_root.rglob("*")
            if path.is_file()
        }
        assert inspect_baseline(sealed["bundle_ref"]) == sealed
        output = infer_at(
            sealed["bundle_ref"],
            local_plan.development.data.snapshot_ref,
            "s31",
            as_of="2020-11-19T16:01:00Z",
        )
        assert len(output["prediction"]["rows"]) == 2
        assert all(row["score"] is not None for row in output["prediction"]["rows"])
        assert output["usage"] == "post_fit_historical_exploration"
        assert output["model_effective_for_requested_time"] is False
        assert before == {
            str(path): digest(path.read_bytes())
            for path in market_root.rglob("*")
            if path.is_file()
        }
    with pytest.raises(ValueError, match="future knowledge"):
        infer_at(
            sealed["bundle_ref"],
            local_plan.development.data.snapshot_ref,
            "s31",
            as_of="2099-01-01T00:00:00Z",
        )
