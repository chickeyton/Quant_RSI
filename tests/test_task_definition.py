"""Pure task/profile contracts; hashes here name type fixtures, not imported market data."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_rsi.core.artifacts import ArtifactStore, canonical, digest
from quant_rsi.forecast.forecast_contracts import ForecastDevelopmentPlan
from quant_rsi.research.researcher import ResearchPlan
from quant_rsi.research.task_definition import (
    ForecastRecursiveProfile,
    RegisteredNeighborDomain,
    domain_ref,
    forecast_profile_ref,
    task_definition,
    training_row_capacity,
)

EXAMPLE = Path(__file__).parents[1] / "examples" / "research_development.json"


def domain_value(*, lower=1, upper=7, initial=5):
    return {
        "kind": "RegisteredNeighborDomain",
        "schema_version": 1,
        "operators": ["momentum", "mean_return"],
        "window_min": lower,
        "window_max": upper,
        "initial_window_max": initial,
        "cap_step": 1,
        "max_cap_advances": 2,
    }


def forecast_value(*, cutoff=20, window=4, instruments=1):
    value = json.loads(EXAMPLE.read_bytes())
    development = value["development"]
    development.pop("portfolio")
    development.pop("annualization")
    development.update(
        plan_type="forecast_development",
        primary_metric="negative_mae_pp",
        evaluation_stride=5,
        min_development_sessions=3,
    )
    development["data"].update(
        source="local_event_snapshot",
        sessions=80,
        train_cutoff=cutoff,
        instruments=instruments,
        snapshot_ref=digest(b"unimported type fixture"),
        decision_delay_seconds=60,
    )
    development["objective"] = {
        "schema_version": 1,
        "kind": "five_session_raw_close",
        "horizon_sessions": 5,
        "target": "log_close_ratio",
        "currency": "TEST",
        "price_unit": "raw_currency_per_share",
        "price_unit_claim": "explicit_assumption",
        "price_unit_evidence_ref": digest(b"unit type fixture"),
        "calendar_evidence_ref": digest(b"calendar type fixture"),
        "notices": [],
        "unknown_action_policy": "affected_or_unknown_interval_abstain",
        "prior_exposure": "previously_exposed_historical_exploration",
        "source_qualifications": [],
    }
    development["proposals"][0]["factor"]["window"] = window
    value["policy"].update(window_min=1, window_max=5, neighbor_order="window_first")
    return value


def recursive_value(**kwargs):
    value = forecast_value(**kwargs)
    development = ForecastDevelopmentPlan.model_validate_json(canonical(value["development"]))
    value.update(
        schema_version=3,
        neighbor_domain=domain_value(),
        forecast_profile=ForecastRecursiveProfile.from_development(development).model_dump(
            mode="json"
        ),
    )
    return value


@pytest.mark.parametrize(
    "upper,expected", [(5, (5,)), (6, (5, 6)), (7, (5, 6, 7)), (30, (5, 6, 7))]
)
def test_domain_preserves_archived_clipped_cap_meaning(upper, expected):
    value = domain_value(upper=upper)
    domain = RegisteredNeighborDomain.model_validate_json(canonical(value))
    assert domain.caps == expected
    assert domain.model_dump(mode="json") == value


@pytest.mark.parametrize(
    "change",
    [
        {"window_min": 6},
        {"window_max": 4},
        {"operators": ["mean_return", "momentum"]},
        {"cap_step": 2},
        {"max_cap_advances": 3},
        {"authority_ref": "unexpected"},
    ],
)
def test_domain_rejects_changed_grammar_or_authority(change):
    with pytest.raises(ValidationError):
        RegisteredNeighborDomain.model_validate_json(canonical(domain_value() | change))


def test_profile_binds_full_existing_forecast_rules_without_authority_or_caps():
    value = forecast_value()
    development = ForecastDevelopmentPlan.model_validate_json(canonical(value["development"]))
    profile = ForecastRecursiveProfile.from_development(development)
    assert profile.objective_ref == digest(canonical(development.objective.model_dump(mode="json")))
    assert profile.development_template_ref == digest(
        canonical(development.model_dump(mode="json"))
    )
    assert profile.model_dump(mode="json") == {
        "kind": "ForecastRecursiveProfile",
        "schema_version": 3,
        "training_mode": "M0",
        "horizon_sessions": 5,
        "target": "log_close_ratio",
        "primary_metric": "negative_mae_pp",
        "score_units": "simple_return_percentage_points",
        "state_policy": "reset",
        "objective_ref": profile.objective_ref,
        "development_template_ref": profile.development_template_ref,
    }
    changed = development.model_copy(update={"min_development_sessions": 4})
    with pytest.raises(ValueError, match="profile differs"):
        profile.validate_development(changed)
    account = ResearchPlan.model_validate_json(EXAMPLE.read_bytes()).development
    with pytest.raises(ValueError, match="requires forecast"):
        ForecastRecursiveProfile.from_development(account)


def test_one_builder_keeps_legacy_task_exact_and_binds_new_cas_objects(tmp_path):
    legacy = ResearchPlan.model_validate_json(EXAMPLE.read_bytes())
    assert task_definition(legacy, "legacy-data", "legacy-template") == {
        "data_ref": "legacy-data",
        "development_template_ref": "legacy-template",
    }
    assert domain_ref(legacy) is None and forecast_profile_ref(legacy) is None
    plan = ResearchPlan.model_validate_json(canonical(recursive_value()))
    store = ArtifactStore(tmp_path / "artifacts")
    data_ref = store.put_json({"kind": "unimported_type_fixture"})
    template_ref = store.put_json(plan.development.model_dump(mode="json"))
    dref = store.put_json(plan.neighbor_domain.model_dump(mode="json"))
    pref = store.put_json(plan.forecast_profile.model_dump(mode="json"))
    expected = {
        "data_ref": data_ref,
        "development_template_ref": template_ref,
        "schema_version": 3,
        "registered_neighbor_domain_ref": dref,
        "forecast_recursive_profile_ref": pref,
    }
    assert task_definition(plan, data_ref, template_ref) == expected
    restored = ResearchPlan.model_validate_json(canonical(plan.model_dump(mode="json")))
    assert canonical(task_definition(restored, data_ref, template_ref)) == canonical(expected)
    assert store.json(dref) == plan.neighbor_domain.model_dump(mode="json")
    assert store.json(pref) == plan.forecast_profile.model_dump(mode="json")
    with pytest.raises(ValueError, match="task development template"):
        task_definition(plan, data_ref, digest(b"wrong template"))


def test_domain_or_objective_changes_task_identity():
    value = recursive_value()
    first = ResearchPlan.model_validate_json(canonical(value))
    value["neighbor_domain"]["window_max"] = 6
    second = ResearchPlan.model_validate_json(canonical(value))
    data_ref = digest(b"same data")
    a = task_definition(first, data_ref, first.forecast_profile.development_template_ref)
    b = task_definition(second, data_ref, second.forecast_profile.development_template_ref)
    assert a["registered_neighbor_domain_ref"] != b["registered_neighbor_domain_ref"]
    assert a["forecast_recursive_profile_ref"] == b["forecast_recursive_profile_ref"]
    value["development"]["objective"]["price_unit_evidence_ref"] = digest(b"changed units")
    dev = ForecastDevelopmentPlan.model_validate_json(canonical(value["development"]))
    value["forecast_profile"] = ForecastRecursiveProfile.from_development(dev).model_dump(
        mode="json"
    )
    third = ResearchPlan.model_validate_json(canonical(value))
    c = task_definition(third, data_ref, third.forecast_profile.development_template_ref)
    assert c["forecast_recursive_profile_ref"] != b["forecast_recursive_profile_ref"]


def test_capacity_is_structural_and_horizon_aware():
    forecast = ForecastDevelopmentPlan.model_validate_json(
        canonical(forecast_value()["development"])
    )
    account = ResearchPlan.model_validate_json(EXAMPLE.read_bytes()).development
    account = account.model_copy(
        update={"data": account.data.model_copy(update={"train_cutoff": 20, "instruments": 1})}
    )
    assert training_row_capacity(forecast, 4) == 12
    assert training_row_capacity(forecast, 5) == 11
    assert training_row_capacity(account, 5) == 14
    assert training_row_capacity(forecast, 30) == 0
    multiple = forecast.model_copy(
        update={"data": forecast.data.model_copy(update={"instruments": 3})}
    )
    assert training_row_capacity(multiple, 5) == 33


@pytest.mark.parametrize("window", [True, 0, 31, 4.0])
def test_capacity_requires_supported_integer_window(window):
    dev = ForecastDevelopmentPlan.model_validate_json(canonical(forecast_value()["development"]))
    with pytest.raises(ValueError, match="integer factor window"):
        training_row_capacity(dev, window)
