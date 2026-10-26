"""Frozen deterministic search; experience changes state, never this program."""

import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.artifacts import canonical, digest
from quant_rsi.forecast_contracts import ForecastDevelopmentPlan
from quant_rsi.schemas import DevelopmentPlan, FactorSpec, StrictModel

C2_FINGERPRINT = "16b85204c89093fef074a58f2f445216d17ef81875b4042352babd42dd64230f"


class SearchPolicy(StrictModel):
    algorithm: Literal["neighbor_search_v1"]
    window_min: int = Field(ge=1, le=30)
    window_max: int = Field(ge=1, le=30)
    min_improvement: float = Field(ge=0)
    max_no_progress: int = Field(ge=1, le=100)
    neighbor_order: Literal["window_first", "operator_first"] = "window_first"


class PackageLineage(StrictModel):
    base_package_ref: str
    patch_ref: str
    proposer_ref: str
    request_ref: str
    evidence_ref: str
    state_policy: Literal["reset"]


class ResearchPlan(StrictModel):
    schema_version: Literal[1]
    plan_type: Literal["development_research"]
    development: DevelopmentPlan | ForecastDevelopmentPlan
    policy: SearchPolicy
    package_lineage: PackageLineage | None = None

    @model_validator(mode="after")
    def valid_start(self) -> Self:
        if len(self.development.proposals) != 1:
            raise ValueError("research template must contain exactly the initial baseline")
        factor = FactorSpec.model_validate_json(canonical(self.development.proposals[0].factor))
        if not self.policy.window_min <= factor.window <= self.policy.window_max:
            raise ValueError("baseline must belong to the declared search range")
        label_tail = 1
        if isinstance(self.development, ForecastDevelopmentPlan):
            label_tail = self.development.objective.horizon_sessions - 1
        if (
            self.development.data.train_cutoff - label_tail - factor.window
        ) * self.development.data.instruments < 12:
            raise ValueError("baseline lacks mature training observations")
        return self


def package_definition(plan: ResearchPlan) -> dict:
    return {
        "kind": "ResearcherPackage",
        "interface_version": 1,
        "state_schema_version": 1,
        "backend": "trusted_deterministic_no_llm",
        "engine_generation": (
            "five_session_price_forecast_v1"
            if isinstance(plan.development, ForecastDevelopmentPlan)
            else "local_asof_development_v1"
            if plan.development.data.source == "local_event_snapshot"
            else "c5_bound_development_v1"
        ),
        "engineering_ancestor": {
            "implementation_fingerprint": C2_FINGERPRINT,
            "relationship": "engineering_upgrade_not_promotion",
        },
        "lineage": plan.package_lineage.model_dump(mode="json") if plan.package_lineage else None,
        "algorithm": "neighbor_search_v1",
        "source_digest": digest(Path(__file__).read_bytes()),
        "implementation_files": {
            name: digest(Path(__file__).with_name(name).read_bytes())
            for name in ("researcher.py", "research.py", "research_journal.py")
        },
        "policy": plan.policy.model_dump(mode="json"),
        "rng": None,
    }


def factor_identity(factor: dict) -> str:
    return digest(canonical(factor))


def initial_state(campaign_id: str, package_ref: str, memory_ref: str) -> dict:
    return {
        "kind": "ResearchStateCheckpoint",
        "schema_version": 1,
        "parent_ref": None,
        "campaign_id": campaign_id,
        "package_ref": package_ref,
        "memory_ref": memory_ref,
        "incumbent": None,
        "baseline": None,
        "visited": [],
        "no_progress": 0,
        "next_position": 0,
        "rng_state": None,
        "budget_ledger_id": campaign_id,
    }


def next_factor(plan: ResearchPlan, checkpoint: dict) -> dict | None:
    """Finite grammar preflight; filtered neighbors consume no proposal reservation."""
    if checkpoint["next_position"] == 0:
        return plan.development.proposals[0].factor
    incumbent = checkpoint["incumbent"]
    if incumbent is None:
        return None
    op, window = incumbent["factor"]["op"], incumbent["factor"]["window"]
    neighbors = [
        (op, window + 1),
        (op, window - 1),
        ("mean_return" if op == "momentum" else "momentum", window),
    ]
    if plan.policy.neighbor_order == "operator_first":
        neighbors = [neighbors[2], *neighbors[:2]]
    for next_op, next_window in neighbors:
        if not plan.policy.window_min <= next_window <= plan.policy.window_max:
            continue
        if (
            plan.development.data.train_cutoff - 1 - next_window
        ) * plan.development.data.instruments < 12:
            continue
        factor = {"op": next_op, "window": next_window}
        if factor_identity(factor) in checkpoint["visited"]:
            continue
        return factor
    return None


def expected_proposal(plan: ResearchPlan, checkpoint: dict) -> dict:
    """Pure policy content, also used to verify a recorded response without making a request."""
    factor = next_factor(plan, checkpoint)
    if factor is None:
        raise ValueError("no eligible neighbor")
    position = checkpoint["next_position"]
    previous = checkpoint["incumbent"]
    rationale = (
        plan.development.proposals[0].hypothesis
        if previous is None
        else (
            f"Test changing {previous['factor']} to {factor} for net development return; "
            "unsupported if improvement or feasibility fails."
        )
    )
    return {"name": f"research_{position:04d}", "hypothesis": rationale, "factor": factor}


def propose(plan: ResearchPlan, checkpoint: dict) -> dict:
    """Request adapter; the caller has already durably reserved this step."""
    return expected_proposal(plan, checkpoint)


def advance(
    plan: ResearchPlan, previous: dict, observation: dict, memory_ref: str, parent_ref: str
) -> dict:
    if (
        observation["campaign_id"] != previous["campaign_id"]
        or observation["ordinal"] != previous["next_position"]
        or observation["visibility"] != "development"
    ):
        raise ValueError("observation is not available to this research state")
    if observation["status"] == "completed":
        local_unresolved = (
            plan.development.data.source == "local_event_snapshot"
            and observation["score"] is None
            and observation["eligible"] is False
        )
        if (
            not local_unresolved
            and (
                type(observation["score"]) not in (float, int)
                or not math.isfinite(observation["score"])
            )
        ) or type(observation["eligible"]) is not bool:
            raise ValueError("completed observation requires a finite trusted score")
    elif observation["score"] is not None or observation["eligible"]:
        raise ValueError("failed observation cannot supply performance")
    state = previous | {
        "parent_ref": parent_ref,
        "memory_ref": memory_ref,
        "next_position": previous["next_position"] + 1,
        "no_progress": previous["no_progress"] + 1,
        "visited": list(previous["visited"]),
    }
    factor = observation["factor"]
    if factor is not None:
        identity = factor_identity(factor)
        if identity not in state["visited"]:
            state["visited"].append(identity)
    if observation["status"] == "completed" and observation["eligible"]:
        selection = {
            "factor": factor,
            "score": observation["score"],
            "result_ref": observation["result_ref"],
            "proposal_ref": observation["proposal_ref"],
            "ordinal": observation["ordinal"],
        }
        if state["baseline"] is None and observation["ordinal"] == 0:
            state["baseline"] = selection
        incumbent = previous["incumbent"]
        if (
            incumbent is None
            or selection["score"] - incumbent["score"] > plan.policy.min_improvement
        ):
            state["incumbent"] = selection
            state["no_progress"] = 0
    return state
