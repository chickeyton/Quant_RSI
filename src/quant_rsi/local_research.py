"""Local as-of inputs through the existing C2 Experiment Kernel, never a second trainer."""

from datetime import timedelta
from decimal import Decimal

import numpy as np

from quant_rsi import execution_account as account
from quant_rsi import market_contracts
from quant_rsi.artifacts import canonical, digest
from quant_rsi.asof import make_feature_view, mature_label, predict_view, qualifications
from quant_rsi.market_contracts import DataContract, timestamp, utc_text
from quant_rsi.market_data import economic_identity, market_context, parse_csv, visible_revisions
from quant_rsi.schemas import LocalDataSpec


def export_local_data(spec):
    with market_context() as (registry, _):
        snapshot, contract = registry.snapshot(spec.snapshot_ref)
        value = dict(
            kind="LocalResearchData",
            specification=spec.model_dump(mode="json"),
            contract=contract.model_dump(mode="json"),
            snapshot=snapshot,
            raw_inputs={
                receipt["raw_ref"]: registry.store.get(receipt["raw_ref"]).decode("utf-8")
                for receipt in snapshot["receipts"]
            },
            availability_evidence_hex=None
            if contract.availability_evidence_ref is None
            else registry.store.get(contract.availability_evidence_ref).hex(),
        )
    LocalResearchData.from_json(value)
    return value


class LocalResearchData:
    def __init__(self, value):
        projected = value.get("kind") == "DerivedLocalResearchData"
        if set(value) != {
            "kind",
            "specification",
            "contract",
            "snapshot",
            "raw_inputs",
            "availability_evidence_hex",
        } | ({"projection_lineage"} if projected else set()) or value["kind"] not in {
            "LocalResearchData",
            "DerivedLocalResearchData",
        }:
            raise ValueError("invalid local research data envelope")
        self.value = value
        self.spec = LocalDataSpec.model_validate_json(canonical(value["specification"]))
        self.contract = DataContract.model_validate_json(canonical(value["contract"]))
        self.snapshot = value["snapshot"]
        if (
            digest(canonical(self.snapshot)) != self.spec.snapshot_ref
            or digest(canonical(value["contract"])) != self.snapshot["contract_ref"]
        ):
            raise ValueError("local research contract/snapshot binding mismatch")
        if self.snapshot["source_id"] != self.contract.source_id:
            raise ValueError("local research source binding mismatch")
        if projected:
            from quant_rsi.local_projection import verify_projection

            verify_projection(value, self.contract)
        elif self.contract.availability_evidence_ref is not None:
            evidence = bytes.fromhex(value["availability_evidence_hex"])
            if digest(evidence) != self.contract.availability_evidence_ref:
                raise ValueError("local historical availability evidence mismatch")
        elif value["availability_evidence_hex"] is not None:
            raise ValueError("unregistered availability evidence")
        if (
            len(self.contract.calendar.sessions) < self.spec.sessions
            or len(self.contract.instruments) != self.spec.instruments
        ):
            raise ValueError(
                "local research dimensions differ from the registered calendar/universe"
            )
        self.dates = tuple(
            session.session_id for session in self.contract.calendar.sessions[: self.spec.sessions]
        )
        self.instruments = self.contract.instruments
        for index in range(self.spec.sessions - 1):
            if self.decision_at(index) >= self.contract.calendar.sessions[index + 1].open_at:
                raise ValueError(
                    "declared historical decision delay reaches the next registered open"
                )
        self._verify_receipts()

    def _verify_receipts(self):
        raw_inputs, snapshot = self.value["raw_inputs"], self.snapshot
        if set(raw_inputs) != {receipt["raw_ref"] for receipt in snapshot["receipts"]}:
            raise ValueError("local raw receipt coverage mismatch")
        reconstructed, identities, event_identities = {}, {}, {}
        last_seq, last_at = 0, None
        for receipt in snapshot["receipts"]:
            if (
                type(receipt["seq"]) is not int
                or not last_seq < receipt["seq"] <= snapshot["receipt_watermark"]
                or last_at is not None
                and timestamp(receipt["first_seen_at"]) < last_at
            ):
                raise ValueError("local receipt order/clock mismatch")
            last_seq, last_at = receipt["seq"], timestamp(receipt["first_seen_at"])
            raw = raw_inputs[receipt["raw_ref"]].encode("utf-8")
            if digest(raw) != receipt["raw_ref"]:
                raise ValueError("local raw input hash mismatch")
            for parsed in parse_csv(self.contract, raw, receipt["first_seen_at"]):
                event = parsed.model_dump(mode="json")
                event.update(
                    event_at=utc_text(parsed.event_at),
                    source_available_at=None
                    if parsed.source_available_at is None
                    else utc_text(parsed.source_available_at),
                )
                key = (event["event_id"], event["revision"])
                identity = digest(canonical(economic_identity(event)))
                if (
                    identities.setdefault(identity, event["event_id"]) != event["event_id"]
                    or event_identities.setdefault(event["event_id"], identity) != identity
                ):
                    raise ValueError("local event identity was relabeled")
                row = dict(
                    event=event,
                    payload_ref=digest(canonical(event)),
                    first_seen_at=receipt["first_seen_at"],
                    receipt_seq=receipt["seq"],
                )
                if key in reconstructed and reconstructed[key]["payload_ref"] != row["payload_ref"]:
                    raise ValueError("local raw revision conflict")
                reconstructed.setdefault(key, row)
        if [reconstructed[key] for key in sorted(reconstructed)] != snapshot["events"]:
            raise ValueError("local first-seen/revision snapshot differs from original receipts")

    @classmethod
    def from_json(cls, value):
        return cls(value)

    def to_json(self):
        return self.value

    def timestamp(self, index, phase):
        return utc_text(getattr(self.contract.calendar.sessions[index], phase + "_at"))

    def decision_at(self, index):
        return self.contract.calendar.sessions[index].close_at + timedelta(
            seconds=self.spec.decision_delay_seconds
        )


def training_view(data, plan, factor):
    if data.spec != plan.data:
        raise ValueError("local worker data specification differs from the sealed plan")
    cutoff = data.decision_at(plan.data.train_cutoff)
    rows, x, y, used = [], [], [], set()
    available = []
    for index in range(factor.window, plan.data.train_cutoff):
        view = make_feature_view(
            data.snapshot, data.contract, data.dates[index], data.decision_at(index), factor
        )
        for feature in view["rows"]:
            label = mature_label(
                data.snapshot, data.contract, data.dates[index], feature["instrument"], cutoff
            )
            eligible = feature["status"] == "eligible" and label["status"] == "mature"
            rows.append(
                dict(
                    session_id=data.dates[index],
                    instrument=feature["instrument"],
                    feature_view_ref=digest(canonical(view)),
                    feature=feature,
                    label=label,
                    training_eligible=eligible,
                )
            )
            if eligible:
                x.append(feature["features"])
                y.append(label["value"])
                used.update(feature["source_refs"] + label["source_refs"])
                available.append(label["available_at"])
    if len(x) < 12:
        raise ValueError("insufficient mature, eligible local training observations")
    market_times = [
        timestamp(row["event"]["event_at"])
        for row in data.snapshot["events"]
        if row["payload_ref"] in used
    ]
    view = dict(
        kind="LocalTrainingView",
        knowledge_basis=data.contract.knowledge_basis,
        snapshot_ref=data.spec.snapshot_ref,
        training_view_asof=utc_text(cutoff),
        rows=rows,
    )
    metadata = dict(
        training_view_ref=digest(canonical(view)),
        training_view_asof=utc_text(cutoff),
        training_information_cutoff=utc_text(max(market_times)),
        last_training_label_available=utc_text(max(timestamp(value) for value in available)),
        training_rows=len(x),
        knowledge_basis=data.contract.knowledge_basis,
        qualifications=qualifications(data.contract),
    )
    return np.asarray(x), np.asarray(y), metadata, view


def development_predictions(data, plan, factor, model_text):
    return [
        predict_view(
            model_text,
            make_feature_view(
                data.snapshot, data.contract, data.dates[index], data.decision_at(index), factor
            ),
        )
        for index in range(plan.data.train_cutoff, plan.data.sessions)
    ]


def fit_predict_local(data, plan, factor):
    import lightgbm as lgb

    x, y, metadata, _ = training_view(data, plan, factor)
    started = market_contracts.trusted_now()
    if timestamp(started) < timestamp(metadata["training_view_asof"]):
        raise ValueError("historical research training cutoff is still in the future")
    params = dict(
        objective="regression",
        verbosity=-1,
        num_threads=1,
        deterministic=True,
        force_col_wise=True,
        seed=plan.data.seed,
        num_leaves=7,
        min_data_in_leaf=5,
        learning_rate=0.05,
        feature_pre_filter=False,
    )
    model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=24)
    completed = market_contracts.trusted_now()
    if timestamp(completed) < timestamp(started):
        raise ValueError("trusted fit clock moved backwards")
    text = model.model_to_string()
    return dict(
        model_text=text,
        model_digest=digest(text.encode()),
        **metadata,
        actual_fit_started_at=started,
        actual_fit_completed_at=completed,
        model_effective_at=completed,
        predictions=development_predictions(data, plan, factor, text),
    )


def verify_local_result(data, plan, factor, result, model_text):
    _, _, expected, _ = training_view(data, plan, factor)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "local worker mature-label/preprocessing metadata differs from bound inputs"
        )
    if (
        not (
            timestamp(result["training_view_asof"])
            <= timestamp(result["actual_fit_started_at"])
            <= timestamp(result["actual_fit_completed_at"])
        )
        or result["model_effective_at"] != result["actual_fit_completed_at"]
    ):
        raise ValueError("local fit/effective time binding mismatch")
    predictions = development_predictions(data, plan, factor, model_text)
    if result["predictions"] != predictions:
        raise ValueError(
            "local predictions differ from the actual serialized model and causal inputs"
        )


def quotes_asof(snapshot, contract, session_id, phase, as_of, *, basis=None):
    visible = visible_revisions(snapshot, contract, as_of, basis=basis)
    return {
        row["event"]["instrument"]: dict(
            price=row["event"]["price"], payload_ref=row["payload_ref"]
        )
        for row in visible
        if row["event"]["kind"] == phase + "_QUOTE"
        and row["event"]["session_id"] == session_id
        and row["event"]["quality"] == "valid"
    }


def known_notices(snapshot, contract, as_of, *, basis=None):
    return [
        row
        for row in visible_revisions(snapshot, contract, as_of, basis=basis)
        if row["event"]["kind"] == "CORPORATE_ACTION_NOTICE"
    ]


def evaluate_local(data, plan, predictions):
    rules = plan.portfolio
    if rules.currency != data.contract.currency or set(rules.fill_cap_units) != set(
        data.instruments
    ):
        raise ValueError("account currency/fill caps differ from the registered instrument grid")
    expected = list(data.dates[plan.data.train_cutoff :])
    if [prediction["session_id"] for prediction in predictions] != expected:
        raise ValueError("local development predictions omit a controller-bound session")
    state = account.new_account(rules)
    for index, prediction in zip(
        range(plan.data.train_cutoff, plan.data.sessions), predictions, strict=True
    ):
        as_of = utc_text(data.decision_at(index))
        if [row["instrument"] for row in prediction["rows"]] != list(
            data.instruments
        ) or prediction["as_of"] != as_of:
            raise ValueError("local development prediction grid/time mismatch")
        for notice in known_notices(data.snapshot, data.contract, as_of):
            state = account.apply_action_notice(
                state, notice["event"]["instrument"], notice["payload_ref"]
            )
        pending = sorted(
            {
                order["target_session"]
                for order in state["orders"]
                if order["status"] == "pending_resolution"
            }
        )
        for session_id in pending:
            quotes = quotes_asof(data.snapshot, data.contract, session_id, "OPEN", as_of)
            state = account.resolve_open(
                state, quotes, rules, session_id=session_id, resolution_at=as_of
            )
        quotes = quotes_asof(data.snapshot, data.contract, data.dates[index], "CLOSE", as_of)
        state = account.value_close(
            state,
            quotes,
            rules,
            session_id=data.dates[index],
            close_at=data.timestamp(index, "close"),
        )
        if index + 1 < plan.data.sessions and not any(
            order["status"] in account.ACTIVE for order in state["orders"]
        ):
            instructions = account.rebalance_instructions(state, prediction, quotes, rules)
            state, ids = account.prepare_orders(
                state,
                instructions,
                rules,
                decision_ref=digest(canonical(prediction)),
                target_session=data.dates[index + 1],
                target_at=data.timestamp(index + 1, "open"),
                prediction_finished_at=as_of,
            )
            state = account.confirm_submission(state, ids, as_of)
    complete = all(
        row["status"] == "complete" and row["return_value"] is not None
        for row in state["valuations"]
    )
    returns = [
        float(row["return_value"]) for row in state["valuations"] if row["return_value"] is not None
    ]
    drawdowns = [
        float(row["drawdown"]) for row in state["valuations"] if row["drawdown"] is not None
    ]
    reasons = list(state["suspended_reasons"])
    if not complete:
        reasons.append("incomplete_fresh_valuation_path")
    if len(returns) < plan.min_development_sessions:
        reasons.append("insufficient_complete_development_returns")
    return dict(
        mode="local_historical_exploration",
        submission_clock_basis="declared_historical_decision_time",
        knowledge_basis=data.contract.knowledge_basis,
        qualifications=qualifications(data.contract),
        eligible=not reasons,
        eligibility_reasons=reasons,
        metrics=dict(
            annualized_mean_net_active_return=float(np.mean(returns) * plan.annualization)
            if complete
            else None,
            total_return=float(Decimal(state["last_nav"]) / Decimal(rules.initial_cash) - 1)
            if complete
            else None,
            max_drawdown=min(drawdowns) if complete else None,
            final_cash=state["cash"],
            costs=amount_costs(state),
        ),
        account=state,
        forward=False,
        terminal_policy="finite_report_preserves_inventory",
    )


def amount_costs(state):
    return format(
        sum(
            (Decimal(state["cumulative_" + key]) for key in ("commission", "tax", "slippage")),
            Decimal(0),
        ),
        "f",
    )
