"""Fixed-model development contracts for synthetic and explicitly bound local inputs."""

import json
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.market.market_contracts import AccountRules, Hash


class DataSpec(StrictModel):
    source: Literal["synthetic"]
    sessions: int = Field(ge=24, le=1000)
    instruments: int = Field(ge=1, le=20)
    train_cutoff: int = Field(ge=12)
    seed: int = Field(ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def check_split(self) -> Self:
        if self.train_cutoff > self.sessions - 5:
            raise ValueError(
                "need a training split and at least five subsequent registered sessions"
            )
        return self


class PortfolioSpec(StrictModel):
    kind: Literal["long_only_topk"]
    initial_cash: float = Field(gt=0, le=1e9)
    max_positions: int = Field(ge=1, le=20)
    allocation: float = Field(gt=0, le=1)
    fee_bps: float = Field(ge=0, le=100)
    slippage_bps: float = Field(ge=0, le=100)
    max_drawdown: float = Field(gt=0, le=1)
    benchmark: Literal["cash_zero_interest"]
    terminal_policy: Literal["liquidate_at_final_open"]


class LocalDataSpec(DataSpec):
    source: Literal["local_event_snapshot"]
    snapshot_ref: Hash
    decision_delay_seconds: int = Field(ge=0, le=86400)


class BudgetSpec(StrictModel):
    max_attempts: int = Field(ge=1, le=100)
    worker_timeout_seconds: int = Field(ge=1, le=300)
    max_worker_seconds: int = Field(ge=1, le=3600)


class Proposal(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    hypothesis: str = Field(min_length=1, max_length=1000)
    # Keep malformed *factor* proposals representable so rejection enters the ledger.
    factor: dict[str, JsonValue]


class FactorSpec(StrictModel):
    op: Literal["momentum", "mean_return"]
    window: int = Field(ge=1, le=30)


class DevelopmentPlan(StrictModel):
    schema_version: Literal[1]
    plan_type: Literal["development"]
    training_mode: Literal["M0"]
    data: DataSpec | LocalDataSpec
    portfolio: PortfolioSpec | AccountRules
    budget: BudgetSpec
    annualization: int = Field(ge=1, le=366)
    primary_metric: Literal["annualized_mean_net_active_return"]
    min_development_sessions: int = Field(ge=3)
    baseline: str
    proposals: tuple[Proposal, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def check_plan(self) -> Self:
        if isinstance(self.data, LocalDataSpec) != isinstance(self.portfolio, AccountRules):
            raise ValueError("local snapshots require explicit finite account rules")
        names = [p.name for p in self.proposals]
        if len(names) != len(set(names)) or self.baseline != names[0]:
            raise ValueError("unique proposal names required; the first proposal is the baseline")
        if self.portfolio.max_positions > self.data.instruments:
            raise ValueError("max_positions exceeds the declared development universe")
        if self.budget.max_worker_seconds < self.budget.worker_timeout_seconds:
            raise ValueError("worker budget cannot fund even one reservation")
        if self.data.sessions - 2 - self.data.train_cutoff < self.min_development_sessions:
            raise ValueError("insufficient development sessions")
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self
