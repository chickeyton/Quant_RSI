"""Forward session purposes and the temporal ordering rule (design section 7).

Every future session gets a purpose by a calendar-only rule fixed before paper activation.
Researcher-facing views of reserved sessions carry counts and integrity only. Protected or
release scoring may use only sessions strictly after everything the compared subjects
trained, developed or were selected on, plus a registered gap.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.core.contracts_base import StrictModel

Purpose = Literal["development_observation", "reserved_protected", "reserved_release"]
RESEARCHER_VISIBLE_FIELDS = frozenset({"session_id", "purpose", "integrity_status", "bars"})


class ReleaseBlock(StrictModel):
    first_session: date
    last_session: date

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.last_session < self.first_session:
            raise ValueError("release block must be chronological")
        return self


class ForwardSessionSchedule(StrictModel):
    schema_version: Literal[1]
    rule: Literal["block_4w_dev_4w_reserved_v1"]
    anchor_monday: date
    development_weeks: Literal[4]
    reserved_weeks: Literal[4]
    embargo_sessions: int = Field(ge=0, le=20)
    release_block: ReleaseBlock | None = None

    @model_validator(mode="after")
    def monday(self) -> Self:
        if self.anchor_monday.weekday() != 0:
            raise ValueError("anchor_monday must be a Monday")
        return self

    def purpose(self, session_day: date) -> Purpose:
        if session_day < self.anchor_monday:
            raise ValueError("session precedes the registered schedule")
        block = self.release_block
        if block is not None and block.first_session <= session_day <= block.last_session:
            return "reserved_release"
        week = (session_day - self.anchor_monday).days // 7
        epoch = self.development_weeks + self.reserved_weeks
        return (
            "development_observation"
            if week % epoch < self.development_weeks
            else ("reserved_protected")
        )

    def reserved_blocks(self, session_days: Sequence[date]) -> list[tuple[date, ...]]:
        """Contiguous runs of reserved_protected sessions, in order."""
        blocks, current = [], []
        for day in sorted(session_days):
            if self.purpose(day) == "reserved_protected":
                if current and (day - current[-1]) > timedelta(days=7):
                    blocks.append(tuple(current))
                    current = []
                current.append(day)
            elif current:
                blocks.append(tuple(current))
                current = []
        if current:
            blocks.append(tuple(current))
        return blocks


def anchor_for(activation_day: date) -> date:
    return activation_day - timedelta(days=activation_day.weekday())


class TemporalOrderViolation(ValueError):
    pass


def validate_temporal_order(
    calendar_days: Sequence[date],
    scored: Iterable[date],
    used: Iterable[date],
    embargo_sessions: int,
) -> None:
    """Reject scoring sessions that are not strictly after all used sessions plus the gap."""
    order = {day: index for index, day in enumerate(sorted(calendar_days))}
    scored, used = sorted(set(scored)), sorted(set(used))
    if not scored:
        raise TemporalOrderViolation("no sessions to score")
    unknown = [d for d in (*scored, *used) if d not in order]
    if unknown:
        raise TemporalOrderViolation(f"sessions outside the registered calendar: {unknown[:3]}")
    if not used:
        return
    first_allowed = order[used[-1]] + embargo_sessions + 1
    if order[scored[0]] < first_allowed:
        raise TemporalOrderViolation(
            f"scored session {scored[0]} is not after the last used session {used[-1]} "
            f"plus {embargo_sessions} embargo sessions"
        )


def development_cutoff(
    schedule: ForwardSessionSchedule, calendar_days: Sequence[date], block: Sequence[date]
) -> list[date]:
    """Development sessions a cycle may use when it will score reserved `block`."""
    order = sorted(calendar_days)
    first = order.index(min(block))
    allowed_last = first - schedule.embargo_sessions - 1
    return [
        day
        for index, day in enumerate(order)
        if index <= allowed_last and schedule.purpose(day) == "development_observation"
    ]


def researcher_view(
    schedule: ForwardSessionSchedule, records: Iterable[Mapping[str, object]]
) -> list[dict]:
    """Purpose-filtered outcomes: reserved sessions keep only counts and integrity status."""
    view = []
    for record in records:
        day = date.fromisoformat(str(record["session_id"]))
        purpose = schedule.purpose(day)
        if purpose == "development_observation":
            view.append({**record, "purpose": purpose})
        else:
            visible = {k: v for k, v in record.items() if k in RESEARCHER_VISIBLE_FIELDS}
            view.append({**visible, "purpose": purpose})
    return view
