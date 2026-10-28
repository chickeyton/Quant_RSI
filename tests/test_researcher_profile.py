"""Pure researcher profile regression: no ingestion, model fitting or execution."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_task_definition import EXAMPLE, forecast_value, recursive_value

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import researcher
from quant_rsi.research.researcher import ResearchPlan
from quant_rsi.research.task_definition import domain_ref, forecast_profile_ref


def checkpoint(window=4):
    factor = {"op": "momentum", "window": window}
    return {
        "next_position": 1,
        "incumbent": {"factor": factor},
        "visited": [researcher.factor_identity(factor)],
    }


@pytest.mark.parametrize("forecast", [False, True])
def test_v1_serialization_keeps_original_keys_and_defaults(forecast):
    value = forecast_value() if forecast else json.loads(EXAMPLE.read_bytes())
    expected = json.loads(canonical(value))
    expected["package_lineage"] = None
    expected["policy"].setdefault("neighbor_order", "window_first")
    plan = ResearchPlan.model_validate_json(canonical(value))
    assert plan.model_dump(mode="json") == expected
    assert json.loads(plan.model_dump_json()) == expected
    assert set(plan.model_dump()) == {
        "schema_version",
        "plan_type",
        "development",
        "policy",
        "package_lineage",
    }
    assert plan.model_dump(mode="json", exclude_unset=True) == value


@pytest.mark.parametrize("field", ["neighbor_domain", "forecast_profile"])
def test_v1_rejects_even_explicit_null_profile_fields(field):
    value = forecast_value()
    value[field] = None
    with pytest.raises(ValidationError, match="legacy research cannot acquire"):
        ResearchPlan.model_validate_json(canonical(value))


def test_archived_v2_is_not_reinterpreted():
    value = recursive_value()
    value["schema_version"] = 2
    with pytest.raises(ValidationError, match="1 or 3"):
        ResearchPlan.model_validate_json(canonical(value))


@pytest.mark.parametrize("field", ["neighbor_domain", "forecast_profile"])
def test_v3_requires_both_explicit_objects(field):
    value = recursive_value()
    value.pop(field)
    with pytest.raises(ValidationError, match="explicit domain and forecast profile"):
        ResearchPlan.model_validate_json(canonical(value))


def test_v3_rejects_account_plan_and_mismatched_forecast_binding():
    value = recursive_value()
    value["development"] = json.loads(EXAMPLE.read_bytes())["development"]
    with pytest.raises(ValidationError, match="requires forecast development"):
        ResearchPlan.model_validate_json(canonical(value))
    value = recursive_value()
    value["development"]["objective"]["calendar_evidence_ref"] = digest(b"changed calendar")
    with pytest.raises(ValidationError, match="profile differs"):
        ResearchPlan.model_validate_json(canonical(value))


@pytest.mark.parametrize(
    "change",
    [
        {"neighbor_order": "operator_first"},
        {"window_min": 2},
        {"window_max": 4},
        {"window_max": 6},
        {"window_max": 8},
    ],
)
def test_v3_fresh_root_policy_must_match_initial_registered_domain(change):
    value = recursive_value()
    value["policy"].update(change)
    with pytest.raises(ValidationError, match="registered cap/domain"):
        ResearchPlan.model_validate_json(canonical(value))


def test_lineaged_type_can_carry_later_cap_without_authorizing_execution():
    value = recursive_value()
    value["policy"]["window_max"] = 6
    value["package_lineage"] = {
        **{
            name: digest(name.encode())
            for name in (
                "base_package_ref",
                "patch_ref",
                "proposer_ref",
                "request_ref",
                "evidence_ref",
            )
        },
        "state_policy": "reset",
    }
    plan = ResearchPlan.model_validate_json(canonical(value))
    assert plan.policy.window_max == 6
    # This validates representation only; managed admission must authenticate the real closure.
    value["policy"]["window_max"] = 8
    with pytest.raises(ValidationError, match="registered cap/domain"):
        ResearchPlan.model_validate_json(canonical(value))


def test_v3_maturity_fix_does_not_reinterpret_v1_forecast_neighbor():
    legacy = ResearchPlan.model_validate_json(canonical(forecast_value()))
    current = ResearchPlan.model_validate_json(canonical(recursive_value()))
    # At cutoff20, window5 has only11 mature five-session rows, not the old estimate14.
    assert researcher.next_factor(legacy, checkpoint()) == {"op": "momentum", "window": 5}
    assert researcher.next_factor(current, checkpoint()) == {"op": "momentum", "window": 3}
    old_proposal = researcher.expected_proposal(legacy, checkpoint())
    assert old_proposal["hypothesis"] == (
        "Test changing {'op': 'momentum', 'window': 4} to "
        "{'op': 'momentum', 'window': 5} for net development return; "
        "unsupported if improvement or feasibility fails."
    )
    assert (
        "simple-return MAE in percentage points"
        in researcher.expected_proposal(current, checkpoint())["hypothesis"]
    )


def test_v1_account_neighbor_keeps_one_session_tail():
    value = json.loads(EXAMPLE.read_bytes())
    value["development"]["data"].update(train_cutoff=20, instruments=1)
    value["development"]["portfolio"]["max_positions"] = 1
    plan = ResearchPlan.model_validate_json(canonical(value))
    assert researcher.next_factor(plan, checkpoint(window=5)) == {"op": "momentum", "window": 6}


@pytest.mark.parametrize("version", [1, 3])
def test_baseline_still_requires_twelve_mature_forecast_rows(version):
    value = forecast_value(window=5) if version == 1 else recursive_value(window=5)
    with pytest.raises(ValidationError, match="baseline lacks mature"):
        ResearchPlan.model_validate_json(canonical(value))


def test_package_version_binds_helper_and_profiles_without_changing_state_contract():
    legacy = ResearchPlan.model_validate_json(canonical(forecast_value()))
    current = ResearchPlan.model_validate_json(canonical(recursive_value()))
    old = researcher.package_definition(legacy)
    new = researcher.package_definition(current)
    assert old["engine_generation"] == "five_session_price_forecast_v1"
    assert "package_version" not in old
    assert set(old["implementation_files"]) == {
        "researcher.py",
        "research.py",
        "research_journal.py",
    }
    assert new["package_version"] == 3
    assert new["engine_generation"] == "five_session_recursive_forecast_v3"
    assert new["registered_neighbor_domain_ref"] == domain_ref(current)
    assert new["forecast_recursive_profile_ref"] == forecast_profile_ref(current)
    helper = Path(researcher.__file__).with_name("task_definition.py")
    assert new["implementation_files"]["task_definition.py"] == digest(helper.read_bytes())
    assert new["interface_version"] == old["interface_version"] == 1
    assert new["state_schema_version"] == old["state_schema_version"] == 1
    assert "neighbor_domain" not in old and "forecast_recursive_profile_ref" not in old
