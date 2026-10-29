"""Paper risk rules. Any reason blocks new entries; exits are never blocked.

The paper-only guard is not a rule here: configuration must never be able to remove it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import Field, field_validator

from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.market.market_contracts import decimal_text


class RiskLimits(StrictModel):
    max_gross_notional: str
    max_position_notional: str
    max_orders_per_minute: int = Field(ge=1, le=600)
    max_daily_loss_fraction: str
    max_consecutive_rejects: int = Field(ge=1, le=100)
    max_clock_skew_seconds: float = Field(gt=0, le=60)

    @field_validator("max_gross_notional", "max_position_notional", "max_daily_loss_fraction")
    @classmethod
    def amount(cls, value):
        return decimal_text(value, positive=True)


@dataclass
class RiskState:
    gross_notional: Decimal = Decimal(0)
    largest_position_notional: Decimal = Decimal(0)
    orders_last_minute: int = 0
    session_pnl: Decimal = Decimal(0)
    capital: Decimal = Decimal(1)
    consecutive_rejects: int = 0
    clock_skew_seconds: float = 0.0
    stale_feed: bool = False
    model_expired: bool = False
    in_restart_window: bool = False
    extra_reasons: list[str] = field(default_factory=list)


Rule = Callable[[RiskState, RiskLimits], str | None]


def _gross(s, lim):
    return "max_gross_notional" if s.gross_notional > Decimal(lim.max_gross_notional) else None


def _position(s, lim):
    limit = Decimal(lim.max_position_notional)
    return "max_position_notional" if s.largest_position_notional > limit else None


def _rate(s, lim):
    return "max_orders_per_minute" if s.orders_last_minute >= lim.max_orders_per_minute else None


def _loss(s, lim):
    limit = -Decimal(lim.max_daily_loss_fraction) * s.capital
    return "daily_loss_stop" if s.session_pnl <= limit else None


def _rejects(s, lim):
    return "consecutive_rejects" if s.consecutive_rejects >= lim.max_consecutive_rejects else None


def _skew(s, lim):
    return "clock_skew" if s.clock_skew_seconds > lim.max_clock_skew_seconds else None


RULES: tuple[Rule, ...] = (
    _gross,
    _position,
    _rate,
    _loss,
    _rejects,
    _skew,
    lambda s, lim: "stale_feed" if s.stale_feed else None,
    lambda s, lim: "model_expired" if s.model_expired else None,
    lambda s, lim: "planned_restart_window" if s.in_restart_window else None,
)


def evaluate(state: RiskState, limits: RiskLimits) -> list[str]:
    """All triggered reasons, so every block is recorded, not only the first."""
    reasons = [reason for rule in RULES if (reason := rule(state, limits))]
    return reasons + list(state.extra_reasons)
