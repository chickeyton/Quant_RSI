"""Canonical task identity for legacy research and explicit recursive forecasts."""

from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from quant_rsi.artifacts import canonical, digest
from quant_rsi.contracts_base import StrictModel
from quant_rsi.forecast_contracts import ForecastDevelopmentPlan
from quant_rsi.market_contracts import Hash
from quant_rsi.schemas import DevelopmentPlan

if TYPE_CHECKING:
    from quant_rsi.researcher import ResearchPlan


class RegisteredNeighborDomain(StrictModel):
    """The archived finite domain; a clipped cap set may allow fewer than two advances."""

    kind: Literal["RegisteredNeighborDomain"]
    schema_version: Literal[1]
    operators: tuple[Literal["momentum"], Literal["mean_return"]]
    window_min: int = Field(strict=True, ge=1, le=30)
    window_max: int = Field(strict=True, ge=1, le=30)
    initial_window_max: int = Field(strict=True, ge=1, le=30)
    cap_step: Literal[1]
    max_cap_advances: Literal[2]

    @model_validator(mode="after")
    def valid_bounds(self) -> Self:
        if not self.window_min <= self.initial_window_max <= self.window_max:
            raise ValueError("initial cap must belong to the registered global domain")
        return self

    @property
    def caps(self) -> tuple[int, ...]:
        return tuple(
            range(self.initial_window_max, min(self.initial_window_max + 2, self.window_max) + 1)
        )


class ForecastRecursiveProfile(StrictModel):
    """Versioned forecast semantics; authority, caps and policy remain separate."""

    kind: Literal["ForecastRecursiveProfile"]
    schema_version: Literal[3]
    training_mode: Literal["M0"]
    horizon_sessions: Literal[5]
    target: Literal["log_close_ratio"]
    primary_metric: Literal["negative_mae_pp"]
    score_units: Literal["simple_return_percentage_points"]
    state_policy: Literal["reset"]
    objective_ref: Hash
    development_template_ref: Hash

    @classmethod
    def from_development(cls, development: ForecastDevelopmentPlan) -> Self:
        if not isinstance(development, ForecastDevelopmentPlan):
            raise ValueError("recursive forecast profile requires forecast development")
        return cls(
            kind="ForecastRecursiveProfile",
            schema_version=3,
            training_mode="M0",
            horizon_sessions=5,
            target="log_close_ratio",
            primary_metric="negative_mae_pp",
            score_units="simple_return_percentage_points",
            state_policy="reset",
            objective_ref=digest(canonical(development.objective.model_dump(mode="json"))),
            development_template_ref=digest(canonical(development.model_dump(mode="json"))),
        )

    def validate_development(self, development: ForecastDevelopmentPlan) -> None:
        if self != self.from_development(development):
            raise ValueError(
                "forecast profile differs from the bound objective/development template"
            )


def training_row_capacity(
    development: DevelopmentPlan | ForecastDevelopmentPlan, window: int
) -> int:
    """Structural upper bound only; knowledge/actions/missing quotes can reduce actual rows."""
    if type(window) is not int or not 1 <= window <= 30:
        raise ValueError("training capacity requires a supported integer factor window")
    if not isinstance(development, (DevelopmentPlan, ForecastDevelopmentPlan)):
        raise ValueError("training capacity requires a supported development plan")
    tail = (
        development.objective.horizon_sessions - 1
        if isinstance(development, ForecastDevelopmentPlan)
        else 1
    )
    return max(0, development.data.train_cutoff - tail - window) * development.data.instruments


def domain_ref(plan: "ResearchPlan") -> str | None:
    if plan.neighbor_domain is None:
        return None
    return digest(canonical(plan.neighbor_domain.model_dump(mode="json")))


def forecast_profile_ref(plan: "ResearchPlan") -> str | None:
    if plan.forecast_profile is None:
        return None
    return digest(canonical(plan.forecast_profile.model_dump(mode="json")))


def task_definition(plan: "ResearchPlan", data_ref: str, template_ref: str) -> dict:
    """Pure live/frozen builder; legacy tasks retain their exact two-field shape."""
    value = dict(data_ref=data_ref, development_template_ref=template_ref)
    if plan.schema_version == 1:
        return value
    if plan.schema_version != 3 or plan.neighbor_domain is None or plan.forecast_profile is None:
        raise ValueError("task requires an explicit supported research profile")
    plan.forecast_profile.validate_development(plan.development)
    if template_ref != plan.forecast_profile.development_template_ref:
        raise ValueError("task development template differs from the forecast profile")
    value.update(
        schema_version=3,
        registered_neighbor_domain_ref=domain_ref(plan),
        forecast_recursive_profile_ref=forecast_profile_ref(plan),
    )
    return value


def verify_domain(plan: "ResearchPlan", store) -> None:
    """Validate normalized v3 domain/profile objects without reading mutable authority."""
    if plan.schema_version == 1:
        return
    plan.forecast_profile.validate_development(plan.development)
    for ref, value in (
        (domain_ref(plan), plan.neighbor_domain.model_dump(mode="json")),
        (forecast_profile_ref(plan), plan.forecast_profile.model_dump(mode="json")),
    ):
        if store.json(ref) != value:
            raise ValueError("registered forecast domain/profile differs from frozen task")
