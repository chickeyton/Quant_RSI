"""Registered C2 exploration preserves incumbents, shared costs and deterministic recovery."""

from copy import deepcopy

import pytest
from pydantic import ValidationError
from test_research import forbidden, ledger, limited
from test_research import research_plan as research_plan

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, canonical
from quant_rsi.research import research, researcher
from quant_rsi.research.research_journal import ResearchJournal
from quant_rsi.research.researcher import ResearchPlan


def exploring(plan, **limits):
    value = limited(plan, **limits).model_dump(mode="json")
    value["policy"]["algorithm"] = "neighbor_exploration_v1"
    return ResearchPlan.model_validate_json(canonical(value))


def observe(plan, state, factor, score=0.1, status="completed"):
    return researcher.advance(
        plan,
        state,
        {
            "campaign_id": state["campaign_id"],
            "ordinal": state["next_position"],
            "visibility": "development",
            "status": status,
            "score": score,
            "eligible": status == "completed",
            "factor": factor,
            "result_ref": "result",
            "proposal_ref": "proposal",
        },
        "memory",
        "parent",
    )


def test_plateau_exploration_is_registered_and_does_not_accept_equal_or_failed_trials(
    research_plan,
):
    plan = exploring(research_plan, no_progress=1)
    state = researcher.initial_state("campaign", "package", "memory")
    state = observe(plan, state, researcher.next_factor(plan, state))
    baseline = deepcopy(state["incumbent"])
    state = observe(plan, state, researcher.next_factor(plan, state))
    assert state["incumbent"] == baseline
    assert researcher.next_factor(plan, state) == {"op": "momentum", "window": 1}
    state = observe(plan, state, researcher.next_factor(plan, state), None, "failed")
    assert state["incumbent"] == baseline
    assert researcher.next_factor(plan, state) == {"op": "mean_return", "window": 1}
    state = observe(plan, state, researcher.next_factor(plan, state), 0.2)
    assert state["incumbent"]["factor"] == {"op": "mean_return", "window": 1}
    assert state["no_progress"] == 0
    assert researcher.next_factor(plan, state) == {"op": "mean_return", "window": 2}
    assert state["rng_state"] is None
    package = researcher.package_definition(plan)
    assert package["algorithm"] == package["policy"]["algorithm"] == "neighbor_exploration_v1"
    assert "task_definition.py" in package["implementation_files"]
    assert researcher.package_definition(research_plan)["algorithm"] == "neighbor_search_v1"


def test_exhausted_neighbors_fall_back_without_waiting_for_plateau(research_plan):
    plan = exploring(research_plan, no_progress=100)
    state = researcher.initial_state("campaign", "package", "memory")
    factors = []
    while (factor := researcher.next_factor(plan, state)) is not None:
        assert factor not in factors
        factors.append(factor)
        state = observe(plan, state, factor)
    assert factors[:4] == [
        {"op": "momentum", "window": 5},
        {"op": "momentum", "window": 6},
        {"op": "momentum", "window": 4},
        {"op": "mean_return", "window": 5},
    ]
    assert factors[4] == {"op": "momentum", "window": 1}
    assert len(factors) == 20
    assert state["incumbent"]["ordinal"] == 0
    assert len(state["visited"]) == 20


def test_exploration_filters_immature_windows_before_reservation(research_plan):
    value = exploring(research_plan).model_dump(mode="json")
    value["development"]["data"].update(train_cutoff=18, instruments=1)
    value["development"]["portfolio"]["max_positions"] = 1
    value["policy"]["window_max"] = 30
    plan = ResearchPlan.model_validate_json(canonical(value))
    state = researcher.initial_state("campaign", "package", "memory")
    factors = []
    while (factor := researcher.next_factor(plan, state)) is not None:
        factors.append(factor)
        state = observe(plan, state, factor)
    assert len(factors) == 10 and max(f["window"] for f in factors) == 5


def test_real_campaign_explores_after_pause_with_one_budget_and_readonly_replay(
    tmp_path, research_plan, monkeypatch
):
    value = exploring(research_plan, no_progress=1, attempts=4, workers=80).model_dump(mode="json")
    # Predeclare an unreachable threshold to exercise a plateau with authentic accounting.
    value["policy"]["min_improvement"] = 1e12
    plan = ResearchPlan.model_validate_json(canonical(value))
    root = tmp_path / "exploration"
    first = research.create_research(plan, root, steps=2)
    assert first["stop_reason"] == "step_limit" and first["attempts_reserved"] == 2
    original_commit = ResearchJournal.commit_observation

    def crash(self, ordinal, *args, **kwargs):
        if ordinal == 2:
            raise RuntimeError("controlled exploration observation interruption")
        return original_commit(self, ordinal, *args, **kwargs)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(ResearchJournal, "commit_observation", crash)
        with pytest.raises(RuntimeError, match="exploration observation"):
            research.resume_research(root, steps=1)
    with monkeypatch.context() as recovering:
        recovering.setattr(researcher, "propose", forbidden)
        recovering.setattr(controller.subprocess, "run", forbidden)
        recovered = research.resume_research(root, steps=1)
    assert recovered["attempts_reserved"] == 3 and recovered["worker_seconds_charged"] == 60
    second = research.resume_research(root, steps=10)
    assert second["stop_reason"] == "attempt_budget_exhausted"
    assert second["attempts_reserved"] == second["kernel_attempts"] == 4
    assert second["worker_seconds_charged"] == 80
    assert second["incumbent"] == first["incumbent"]
    factors = [r["decision"]["proposal"]["factor"] for r in second["results"]]
    assert factors[2:] == [{"op": "momentum", "window": 1}, {"op": "mean_return", "window": 1}]
    rows = ledger(root)[0]
    request = ArtifactStore(root / "artifacts").json(rows[2]["request_ref"])
    assert request["checkpoint_ref"] == rows[1]["checkpoint_ref"]
    assert rows[1]["observation_ref"] in request["evidence_refs"]
    before = ledger(root)
    with monkeypatch.context() as locked:
        locked.setattr(researcher, "propose", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        assert research.resume_research(root, steps=10) == second
        assert research.inspect_research(root, replay=True)["verified_experiments"] == 4
    assert ledger(root) == before


@pytest.mark.parametrize(
    "attempts,workers,reason",
    [(2, 100, "attempt_budget_exhausted"), (10, 40, "worker_budget_exhausted")],
)
def test_plateau_cannot_bypass_either_budget(research_plan, attempts, workers, reason):
    plan = exploring(research_plan, attempts=attempts, workers=workers, no_progress=1)
    state = researcher.initial_state("campaign", "package", "memory")
    for _ in range(2):
        state = observe(plan, state, researcher.next_factor(plan, state))

    class Journal:
        def requests(self):
            return [None, None]

        def attempts(self):
            return [{"worker_seconds": 20}, {"worker_seconds": 20}]

    assert research.stop_reason(plan, state, Journal()) == reason


def test_registered_recursive_cap_policy_cannot_silently_acquire_exploration():
    from test_forecast_recursive_changes import plan_value

    value = plan_value().model_dump(mode="json")
    value["policy"]["algorithm"] = "neighbor_exploration_v1"
    with pytest.raises(ValidationError, match="registered cap/domain"):
        ResearchPlan.model_validate_json(canonical(value))


def test_forecast_exploration_respects_five_session_label_maturity():
    from test_task_definition import forecast_value

    value = forecast_value(cutoff=20, window=4)
    value["policy"].update(algorithm="neighbor_exploration_v1", window_max=30)
    plan = ResearchPlan.model_validate_json(canonical(value))
    state = researcher.initial_state("campaign", "package", "memory")
    factors = []
    while (factor := researcher.next_factor(plan, state)) is not None:
        factors.append(factor)
        state = observe(plan, state, factor)
    assert len(factors) == 8 and max(f["window"] for f in factors) == 4


def test_exhausted_space_and_failed_baseline_are_explicit_terminal_states(research_plan):
    plan = exploring(research_plan)
    state = researcher.initial_state("campaign", "package", "memory")

    class EmptyJournal:
        def requests(self):
            return []

        def attempts(self):
            return []

    failed = observe(plan, state, researcher.next_factor(plan, state), None, "failed")
    assert research.stop_reason(plan, failed, EmptyJournal()) == "baseline_unavailable"
    while (factor := researcher.next_factor(plan, state)) is not None:
        state = observe(plan, state, factor)
    assert research.stop_reason(plan, state, EmptyJournal()) == "registered_search_space_exhausted"
