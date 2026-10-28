"""Versioned, frozen inputs for forecast features; no collection, fitting, or publication."""

from datetime import date
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from quant_rsi.core.artifacts import canonical, decode, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.evaluation.local_research import LocalResearchData
from quant_rsi.forecast.forecast_contracts import ForecastObjective
from quant_rsi.forecast.forecast_kernel import cash_interpretation_matches
from quant_rsi.market.market_contracts import Hash, Identifier, timestamp


class ForecastRecipe(StrictModel):
    schema_version: Literal[1, 2]
    kind: Literal["forecast_feature_recipe"]
    group: Literal["B0", "B1", "B2", "N1"]

    @model_validator(mode="after")
    def versioned_group(self) -> Self:
        if (self.schema_version == 2) != (self.group == "N1"):
            raise ValueError("news recipe requires version 2; legacy recipes require version 1")
        return self

    @property
    def names(self):
        names = ("momentum_5", "return_1", "intraday_return")
        if self.group in {"B1", "B2"}:
            names += ("return_20", "return_60", "volatility_20", "volatility_60")
        if self.group == "B2":
            names += tuple(
                f"{role}_{quantity}_{window}"
                for role in ("industry", "market")
                for quantity in ("return", "relative_return")
                for window in (5, 20)
            )
        if self.group == "N1":
            from quant_rsi.factors.news import NEWS_FEATURE_DEFINITION

            names += (NEWS_FEATURE_DEFINITION["name"],)
        return names

    @property
    def definition(self):
        value = dict(
            **self.model_dump(mode="json"),
            feature_names=list(self.names),
            feature_units=["dimensionless_return"] * len(self.names),
            close_windows=[5] if self.group == "B0" else [5, 20, 60],
            volatility="population_std_of_simple_daily_returns_not_annualized",
            relative_return="target_simple_return_minus_context_simple_return",
            context_alignment="exact_registered_close_timestamp_no_fill",
        )
        if self.group == "N1":
            from quant_rsi.factors.news import NEWS_FEATURE_DEFINITION

            value.update(
                feature_units=["dimensionless_return"] * 3 + [NEWS_FEATURE_DEFINITION["unit"]],
                close_windows=[5],
                news_factor=NEWS_FEATURE_DEFINITION,
            )
        return value

    @property
    def ref(self):
        return digest(canonical(self.definition))


class SignalEvent(StrictModel):
    schema_version: Literal[1]
    kind: Literal["news", "fundamental"]
    source_id: Identifier
    entity: Identifier
    event_id: Identifier
    revision: int = Field(ge=0)
    query_ref: Hash
    published_at: AwareDatetime | None
    published_date: date | None
    publication_precision: Literal["second", "day", "unknown"]
    source_available_at: AwareDatetime | None
    first_seen_at: AwareDatetime
    report_period_end: date | None
    payload_ref: Hash
    receipt_ref: Hash
    extractor_version: Identifier
    units: dict[str, str]
    fields: dict[str, str]

    @model_validator(mode="after")
    def valid_fundamental(self) -> Self:
        if self.publication_precision == "second":
            if self.published_at is None:
                raise ValueError("second precision requires an actual publication timestamp")
        elif self.publication_precision == "day":
            if self.published_at is not None or self.published_date is None:
                raise ValueError("date-only publication cannot invent a timestamp")
        elif self.published_at is not None or self.published_date is not None:
            raise ValueError("unknown publication time must remain unknown")
        if self.kind == "fundamental" and self.report_period_end is None:
            raise ValueError("fundamentals require an explicit report period")
        if set(self.units) != set(self.fields):
            raise ValueError("each extracted signal field requires an explicit unit")
        return self

    @property
    def available_at(self):
        return max(
            value
            for value in (
                self.published_at,
                self.first_seen_at,
                self.source_available_at,
            )
            if value is not None
        )


class SignalCoverage(StrictModel):
    schema_version: Literal[1]
    source_id: Identifier
    entity: Identifier
    query_ref: Hash
    start_at: AwareDatetime
    end_at: AwareDatetime
    first_seen_at: AwareDatetime
    status: Literal["complete", "partial", "unknown"]
    reason: str | None
    receipt_ref: Hash

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.start_at >= self.end_at:
            raise ValueError("coverage interval must be nonempty")
        if self.status == "complete" and self.first_seen_at < self.end_at:
            raise ValueError("a complete interval cannot be received before its end")
        if self.status != "complete" and not self.reason:
            raise ValueError("partial or unknown coverage requires a reason")
        return self


class SignalSnapshot(StrictModel):
    schema_version: Literal[1]
    kind: Literal["forecast_signal_snapshot"]
    events: tuple[SignalEvent, ...]
    coverage: tuple[SignalCoverage, ...]

    @model_validator(mode="after")
    def unique_revisions(self) -> Self:
        keys = [(e.source_id, e.entity, e.event_id, e.revision) for e in self.events]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate signal revision")
        groups = {}
        for event in self.events:
            groups.setdefault((event.source_id, event.entity, event.event_id), []).append(event)
        for events in groups.values():
            ordered = sorted(events, key=lambda e: e.revision)
            if any(
                a.first_seen_at > b.first_seen_at
                for a, b in zip(ordered, ordered[1:], strict=False)
            ):
                raise ValueError("signal revisions have decreasing receipt times")
        coverage_keys = [
            (c.source_id, c.entity, c.query_ref, c.start_at, c.end_at, c.first_seen_at)
            for c in self.coverage
        ]
        if len(coverage_keys) != len(set(coverage_keys)):
            raise ValueError("ambiguous coverage receipts")
        return self


class ForecastInputs:
    """A new envelope wrapping unchanged LocalResearchData for each independent source."""

    def __init__(self, value):
        value = decode(canonical(value))
        if set(value) != {
            "kind",
            "schema_version",
            "target",
            "contexts",
            "signals",
            "evidence_hex",
        }:
            raise ValueError("invalid forecast input envelope")
        if value["kind"] != "ForecastInputs" or value["schema_version"] != 1:
            raise ValueError("unsupported forecast input version")
        if not set(value["contexts"]) <= {"industry", "market"}:
            raise ValueError("unsupported context role")
        self.value = value
        self.ref = digest(canonical(value))
        self.evidence = {}
        for ref, encoded in value["evidence_hex"].items():
            raw = bytes.fromhex(encoded)
            if digest(raw) != ref:
                raise ValueError("forecast evidence hash mismatch")
            self.evidence[ref] = raw
        self.target = self._market(value["target"])
        self.contexts = {role: self._market(item) for role, item in value["contexts"].items()}
        currency = self.target[0].contract.currency
        target_instrument = self.target[0].instruments[0]
        seen = {target_instrument}
        for data, _ in self.contexts.values():
            if data.contract.currency != currency:
                raise ValueError(
                    "context currency differs from target; no FX conversion is declared"
                )
            if data.instruments[0] in seen:
                raise ValueError("target and context roles require distinct instruments")
            seen.add(data.instruments[0])
        self.signals = (
            SignalSnapshot.model_validate_json(canonical(value["signals"]))
            if value["signals"] is not None
            else None
        )
        if self.signals:
            for event in self.signals.events:
                self._require(event.payload_ref)
                self._require(event.receipt_ref)
                self._require(event.query_ref)
            for coverage in self.signals.coverage:
                self._require(coverage.receipt_ref)
                self._require(coverage.query_ref)

    def _require(self, ref):
        if ref not in self.evidence:
            raise ValueError("forecast input is missing referenced evidence")

    def _market(self, value):
        if set(value) != {"data", "objective"}:
            raise ValueError("market input requires data and its own action/unit interpretation")
        data = LocalResearchData.from_json(value["data"])
        objective = ForecastObjective.model_validate_json(canonical(value["objective"]))
        if len(data.instruments) != 1:
            raise ValueError("each input role must contain exactly one instrument")
        if data.contract.currency != objective.currency or data.contract.price_basis != "raw":
            raise ValueError("forecast input unit/currency mismatch")
        self._require(objective.price_unit_evidence_ref)
        self._require(objective.calendar_evidence_ref)
        actions = {
            item["payload_ref"]: item
            for item in data.snapshot["events"]
            if item["event"]["kind"] == "CORPORATE_ACTION_NOTICE"
        }
        for notice in objective.notices:
            self._require(notice.evidence_ref)
            if notice.notice_ref not in actions:
                raise ValueError("action interpretation is not bound to this source")
            if notice.action_kind == "cash_dividend" and not cash_interpretation_matches(
                actions[notice.notice_ref], notice
            ):
                raise ValueError("cash interpretation differs from source payload")
        return data, objective


def signal_view(snapshot, *, source_id, entity, query_ref, start_at, end_at, as_of):
    """Exact-query coverage on [start, end); receipt-gated news never treats unknown as zero."""
    start, end, boundary = timestamp(start_at), timestamp(end_at), timestamp(as_of)
    if start >= end or end > boundary:
        raise ValueError("signal window must end by the knowledge boundary")
    key = (source_id, entity, query_ref)
    coverage = (
        []
        if snapshot is None
        else [
            row
            for row in snapshot.coverage
            if (row.source_id, row.entity, row.query_ref) == key
            and row.start_at == start
            and row.end_at == end
            and row.first_seen_at <= boundary
        ]
    )
    latest_coverage = max(coverage, key=lambda row: row.first_seen_at) if coverage else None
    latest = {}
    for event in () if snapshot is None else snapshot.events:
        if (event.source_id, event.entity, event.query_ref) != key or event.available_at > boundary:
            continue
        old = latest.get(event.event_id)
        if old is None or old.revision < event.revision:
            latest[event.event_id] = event
    uncertain = [event for event in latest.values() if event.published_at is None]
    events = [
        event
        for event in latest.values()
        if event.published_at is not None and start <= event.published_at < end
    ]
    events.sort(key=lambda row: (row.published_at, row.event_id))
    status = latest_coverage.status if latest_coverage else "unknown"
    return dict(
        coverage=status,
        coverage_ref=digest(canonical(latest_coverage.model_dump(mode="json")))
        if latest_coverage
        else None,
        news_count=sum(event.kind == "news" for event in events)
        if status == "complete" and not any(event.kind == "news" for event in uncertain)
        else None,
        events=[event.model_dump(mode="json") for event in events],
        publication_time_unresolved=[event.model_dump(mode="json") for event in uncertain],
        reason=latest_coverage.reason if latest_coverage else "no_visible_exact_query_coverage",
        knowledge_basis="observed_receipts",
        as_of=boundary.isoformat(),
    )
