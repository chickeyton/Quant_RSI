"""Finite, durable baseline observation. No broker, background service or promotion."""

import re
from copy import deepcopy

from quant_rsi.core import controller
from quant_rsi.core.artifacts import canonical
from quant_rsi.execution import execution_account
from quant_rsi.execution import observer_engine as engine
from quant_rsi.execution.baseline import read_bundle
from quant_rsi.execution.observer_journal import ObserverJournal, initialize_schema
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import ObservationPlan, parse_observation_plan, timestamp
from quant_rsi.market.market_data import market_context


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ValueError("episode/request identity must be 1..128 plain identifier characters")


def control_transition(previous, kind, payload, plan):
    schemas = {
        "ACTIVATE": {"at"},
        "WINDOW_START": {"at", "request_id", "steps"},
        "WINDOW_END": {"at", "request_id", "reason"},
        "STOP": {"at", "request_id"},
        "CONFIRM": {"at"},
        "RECOVER_UNCONFIRMED": {"at"},
    }
    if kind not in schemas or set(payload) != schemas[kind]:
        raise ValueError("unrecognized trusted observer control envelope")
    engine.clock_after(previous, payload["at"])
    if kind in {"CONFIRM", "RECOVER_UNCONFIRMED"}:
        return engine.confirm_submission(
            previous, payload["at"], recover=kind == "RECOVER_UNCONFIRMED"
        )
    state = deepcopy(previous)
    if kind == "ACTIVATE":
        if state["active_at"] is not None:
            raise ValueError("episode was already activated")
        state["active_at"] = payload["at"]
    elif kind == "WINDOW_START":
        identifier(payload["request_id"])
        if (
            state["active_at"] is None
            or state["window"] is not None
            or payload["request_id"] in state["requests"]
        ):
            raise ValueError("observer request is not a new finite window")
        if type(payload["steps"]) is not int or not 1 <= payload["steps"] <= 1000:
            raise ValueError("observer steps must be 1..1000")
        window = dict(
            request_id=payload["request_id"],
            steps=payload["steps"],
            start=state["processed_steps"],
            target=state["processed_steps"] + payload["steps"],
            status="running",
        )
        state["window"] = window
        state["requests"][payload["request_id"]] = deepcopy(window)
    elif kind == "WINDOW_END":
        window = state["window"]
        if (
            window is None
            or window["request_id"] != payload["request_id"]
            or state["pending_submission"] is not None
        ):
            raise ValueError(
                "finite window cannot finish with a different request or pending submission"
            )
        if payload["reason"] not in {"step_limit", "awaiting_input"}:
            raise ValueError("unsupported finite window completion reason")
        if (
            state["processed_steps"] > window["target"]
            or payload["reason"] == "step_limit"
            and state["processed_steps"] != window["target"]
        ):
            raise ValueError("finite observer window exceeded/differs from its fixed target")
        state["requests"][window["request_id"]].update(
            status="completed",
            reason=payload["reason"],
            completed_at=payload["at"],
            actual_steps=state["processed_steps"] - window["start"],
        )
        state["window"] = None
    else:
        identifier(payload["request_id"])
        if state["stop"] is not None:
            raise ValueError("a durable stop receipt cannot be replaced")
        if state["pending_submission"] is not None:
            state = engine.confirm_submission(state, payload["at"], recover=True)
        state["stop"] = dict(request_id=payload["request_id"], received_at=payload["at"])
        if state["account"] is not None:
            state["account"] = execution_account.stop_account(state["account"], payload["at"])
    state["last_control_at"] = payload["at"]
    return state


def register_observer(plan, episode_id):
    identifier(episode_id)
    with market_context() as (registry, _):
        bundle, _, contract, _ = read_bundle(registry, plan.model_bundle_ref)
        if (
            bundle["contract_ref"] != plan.contract_ref
            or registry.contract(plan.contract_ref) != contract
        ):
            raise ValueError("observer plan/model/data interface mismatch")
        if isinstance(plan, ObservationPlan) and (
            plan.currency != contract.currency
            or set(plan.fill_cap_units) != set(contract.instruments)
        ):
            raise ValueError(
                "observer currency and fill caps must cover the complete registered grid"
            )
        initialize_schema(registry)
        journal = ObserverJournal(registry)
        plan_ref = registry.store.put_json(plan.model_dump(mode="json"))
        now = market_contracts.trusted_now()
        if timestamp(now) < timestamp(bundle["model_effective_at"]):
            raise ValueError("observer cannot be activated before actual model registration")
        watermark = registry.connection.execute(
            "SELECT COALESCE(MAX(seq),0) FROM receipts"
        ).fetchone()[0]
        runtime_ref = registry.store.put_json(controller.runtime_identity())
        initial = engine.initial_state(episode_id, plan, runtime_ref, now, watermark)
        episode = journal.register(episode_id, plan_ref, initial)
        state = registry.store.json(episode["head_ref"])
        if state["active_at"] is None:
            # This timestamp is taken after the initial episode transaction committed.
            payload = dict(at=market_contracts.trusted_now())
            state = control_transition(state, "ACTIVATE", payload, plan)
            episode = journal.commit(episode, state, "ACTIVATE", payload)
        return summary(episode, state, plan, bundle)


def load_episode(registry, episode_id, *, replay):
    journal = ObserverJournal(registry)
    episode, initial, head, events = journal.trace(episode_id)
    plan = parse_observation_plan(registry.store.get(episode["plan_ref"]))
    if registry.store.json(initial["runtime_ref"]) != controller.runtime_identity():
        raise ValueError("observer source/runtime changed; use its recorded implementation")
    if initial != engine.initial_state(
        episode_id, plan, initial["runtime_ref"], initial["created_at"], initial["start_watermark"]
    ):
        raise ValueError("observer initial account/receipt cursor differs from its registered plan")
    bundle, model_text, contract, _ = read_bundle(registry, plan.model_bundle_ref)
    if (
        bundle["contract_ref"] != plan.contract_ref
        or registry.contract(plan.contract_ref) != contract
    ):
        raise ValueError("observer registered model/contract identity mismatch")
    for row in head["observations"]:
        registry.store.get(row["observation_ref"])
    if replay:
        state = initial
        for event in events:
            if event["kind"] == "STEP":
                payload = event["payload"]
                snapshot, input_contract = registry.snapshot(payload["snapshot_ref"])
                if input_contract != contract:
                    raise ValueError("observer step consumed another input contract")
                recorded_clocks = iter(payload["finish_clocks"])
                try:
                    computed = engine.transition_step(
                        state,
                        plan,
                        bundle,
                        model_text,
                        contract,
                        snapshot,
                        payload["started_at"],
                        lambda readings=recorded_clocks: next(readings),
                    )
                except StopIteration as error:
                    raise ValueError(
                        "observer replay lacks a committed trusted clock receipt"
                    ) from error
                if computed is None or next(recorded_clocks, None) is not None:
                    raise ValueError("observer replay clock/step count mismatch")
                next_state, expected_payload, objects = computed
                if expected_payload != payload:
                    raise ValueError("observer replay differs from committed input/timing bindings")
                for ref, value in objects.items():
                    if canonical(value) != registry.store.get(ref):
                        raise ValueError("observer replay differs from an immutable derived output")
            else:
                next_state = control_transition(state, event["kind"], event["payload"], plan)
            if canonical(next_state) != registry.store.get(event["state_ref"]):
                raise ValueError(
                    "observer replay disagrees with committed account/checkpoint state"
                )
            state = next_state
        if state != head:
            raise ValueError("observer replay did not reach the actual durable head")
    return journal, episode, head, plan, bundle, model_text, contract


def advance_observer(episode_id, *, request_id, steps=None):
    identifier(episode_id)
    identifier(request_id)
    with market_context() as (registry, _):
        journal, episode, state, plan, bundle, model_text, contract = load_episode(
            registry, episode_id, replay=True
        )
        if state["active_at"] is None:
            raise ValueError("episode registration is unfinished; repeat observer register")
        existing = state["requests"].get(request_id)
        if existing is not None:
            if steps is not None and steps != existing["steps"]:
                raise ValueError("request already has a fixed step grant")
            if existing["status"] == "completed":
                return summary(episode, state, plan, bundle, request_id=request_id)
        else:
            if steps is None:
                raise ValueError("unknown resume request; register a finite run window first")
            payload = dict(at=market_contracts.trusted_now(), request_id=request_id, steps=steps)
            state = control_transition(state, "WINDOW_START", payload, plan)
            episode = journal.commit(episode, state, "WINDOW_START", payload)
        if state["window"]["request_id"] != request_id:
            raise ValueError("a different finite request is pending; its target cannot be replaced")
        if state["pending_submission"] is not None:
            payload = dict(at=market_contracts.trusted_now())
            state = control_transition(state, "RECOVER_UNCONFIRMED", payload, plan)
            episode = journal.commit(episode, state, "RECOVER_UNCONFIRMED", payload)
        reason = "step_limit"
        while state["processed_steps"] < state["window"]["target"]:
            snapshot_ref = registry.latest_snapshot(plan.contract_ref)
            snapshot, _ = registry.snapshot(snapshot_ref)
            now = market_contracts.trusted_now()
            computed = engine.transition_step(
                state,
                plan,
                bundle,
                model_text,
                contract,
                snapshot,
                now,
                market_contracts.trusted_now,
            )
            if computed is None:
                reason = "awaiting_input"
                break
            state, payload, objects = computed
            episode = journal.commit(episode, state, "STEP", payload, objects=objects)
            if state["pending_submission"] is not None:
                # Never move this trusted witness inside the preceding SQLite transaction.
                payload = dict(at=market_contracts.trusted_now())
                state = control_transition(state, "CONFIRM", payload, plan)
                episode = journal.commit(episode, state, "CONFIRM", payload)
        payload = dict(at=market_contracts.trusted_now(), request_id=request_id, reason=reason)
        state = control_transition(state, "WINDOW_END", payload, plan)
        episode = journal.commit(episode, state, "WINDOW_END", payload)
        return summary(episode, state, plan, bundle, request_id=request_id)


def stop_observer(episode_id, *, request_id):
    identifier(episode_id)
    identifier(request_id)
    with market_context() as (registry, _):
        journal, episode, state, plan, bundle, _, _ = load_episode(
            registry, episode_id, replay=True
        )
        if state["stop"] is None:
            payload = dict(at=market_contracts.trusted_now(), request_id=request_id)
            state = control_transition(state, "STOP", payload, plan)
            episode = journal.commit(episode, state, "STOP", payload)
        return summary(episode, state, plan, bundle)


def inspect_observer(episode_id, *, replay=False):
    identifier(episode_id)
    with market_context() as (registry, _):
        _, episode, state, plan, bundle, _, _ = load_episode(registry, episode_id, replay=replay)
        result = summary(episode, state, plan, bundle)
        result["replay"] = (
            "recorded_receipts_clocks_inference_and_accounting_no_fit_or_writes" if replay else None
        )
        return result


def summary(episode, state, plan, bundle, *, request_id=None):
    account_state = state["account"]
    latest = (
        None
        if account_state is None or not account_state["valuations"]
        else account_state["valuations"][-1]
    )
    return dict(
        episode_id=episode["episode_id"],
        checkpoint_ref=episode["head_ref"],
        sequence=episode["sequence"],
        mode=plan.account_mode,
        purpose="developer_visible_observation",
        promotion_status="none",
        active_at=state["active_at"],
        start_watermark=state["start_watermark"],
        receipt_cursor=state["receipt_cursor"],
        market_cursor=state["market_cursor"],
        processed_steps=state["processed_steps"],
        observation_count=len(state["observations"]),
        observations=state["observations"][-5:],
        correction_count=len(state["correction_flags"]),
        window=state["window"],
        request=None if request_id is None else state["requests"].get(request_id),
        pending_submission=state["pending_submission"],
        stop=state["stop"],
        cash=None if account_state is None else account_state["cash"],
        holdings=None if account_state is None else account_state["holdings"],
        reserved_cash=None
        if account_state is None
        else str(execution_account.reserved_cash(account_state)),
        pending_order_count=0
        if account_state is None
        else sum(order["status"] in execution_account.ACTIVE for order in account_state["orders"]),
        latest_valuation=latest,
        model_blocked_reasons=engine.model_reasons(plan, bundle, state["last_control_at"]),
        as_of_last_committed_control=state["last_control_at"],
        training_qualifications=bundle["qualifications"],
        qualifications=[
            "baseline_observation_only",
            "real_pilot_not_established_by_engineering_fixtures",
        ],
    )
