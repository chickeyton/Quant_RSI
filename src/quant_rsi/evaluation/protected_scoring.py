"""Trusted fixed-model scorer. No training or researcher API is imported here."""

import os
import sys
from pathlib import Path

import numpy as np

from quant_rsi.core.accounting import evaluate, returns_from_equities
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical
from quant_rsi.core.data import MarketData
from quant_rsi.core.model import predict_fixed
from quant_rsi.core.schemas import FactorSpec, PortfolioSpec
from quant_rsi.evaluation.evidence import Authority, authority_root, ref


def score_binding(authority, key):
    batch = authority.state["batch"]
    if batch is None or batch["selections_ref"] is None:
        raise ValueError("selected artifacts are not sealed")
    plan = authority.store.json(batch["plan_ref"])
    selections = authority.store.json(batch["selections_ref"])
    selection_ref = selections["unique"][key]
    task = next(t for t in plan["tasks"] if t["task"] == key.split(":")[0])
    return {
        "kind": "ProtectedScoreBinding",
        "authority_id": authority.anchor["authority_id"],
        "batch_id": batch["id"],
        "plan_ref": batch["plan_ref"],
        "key": key,
        "selection_ref": selection_ref,
        "source_ref": task["source_ref"],
        "episode_ref": task["episode_ref"],
        "task_ref": ref(task),
        "runtime_ref": authority.program["runtime_ref"],
        "execution_seed": 0,
        "reset_policy": "fresh_cash_empty_positions_orders_no_accrual",
        "grid": task["grid"],
    }


def authorize_score(authority, attempt_id, *, replay=False):
    state = authority.state
    batch = state["batch"]
    attempt = state["attempts"].get(attempt_id)
    if (
        batch is None
        or not batch["exposed"]
        or batch["status"] not in {"OPENED", "COMPLETED", "RETIRED"}
    ):
        raise ValueError("protected access requires a committed OPENED transition")
    if attempt is None or attempt["outbox"] != "DISPATCHED" or attempt["seconds"] != 20:
        raise ValueError("protected access requires a charged dispatched attempt")
    if not replay and attempt["status"] != "running":
        raise ValueError("attempt is not running")
    binding = score_binding(authority, attempt["key"])
    plan = authority.store.json(binding["plan_ref"])
    if plan.get("source_kind") == "local_event_snapshot" and not replay:
        pending = batch["pending"]
        if (
            pending is None
            or pending["kind"] != "score"
            or pending["key"] != attempt["key"]
            or pending["attempt_id"] != attempt_id
            or batch["scores"][attempt["key"]]["attempt_ids"][-1] != attempt_id
        ):
            raise ValueError("local scorer requires its actual pending dispatched operation")
    if plan["mode"] != "engineering_only" or plan["utility"]["registered_total_loss_events"]:
        raise ValueError("production scoring rejects unregistered mode/total-loss events")
    if ref(binding) != attempt["binding_ref"]:
        raise ValueError("attempt computation binding changed")
    return attempt, binding


def read_episode(authority, attempt_id, *, replay=False):
    if replay and authority.program.get("source_kind") == "local_event_snapshot":
        raise ValueError(
            "local replay uses sealed evidence; protected original reads are forbidden"
        )
    _, binding = authorize_score(authority, attempt_id, replay=replay)
    plan = authority.store.json(binding["plan_ref"])
    task = next(t for t in plan["tasks"] if ref(t) == binding["task_ref"])
    value = ArtifactStore(authority.home / "protected").json(binding["source_ref"])
    full = MarketData.from_json(value)
    if plan.get("source_kind") == "local_event_snapshot":
        if (
            value["kind"] != "LocalResearchData"
            or ref(value) != task["episode_ref"]
            or list(full.dates) != task["sessions"]
            or list(full.instruments) != task["instruments"]
            or full.spec.snapshot_ref != task["original_snapshot_ref"]
        ):
            raise ValueError("local protected episode differs from its registered original source")
        return full
    start = task["offset"]
    data = MarketData(
        full.dates[start : start + 122],
        full.instruments,
        full.opens[start : start + 122],
        full.closes[start : start + 122],
    )
    if ref(data.to_json()) != task["episode_ref"] or list(data.dates) != task["sessions"]:
        raise ValueError("protected episode differs from the registered source/grid")
    return data


def compute_score(authority, attempt_id, *, replay=False):
    if authority.program.get("source_kind") == "local_event_snapshot":
        from quant_rsi.evaluation.local_scoring import local_score

        return local_score(authority, attempt_id, replay=replay)
    _, binding = authorize_score(authority, attempt_id, replay=replay)
    envelope = {
        "kind": "ProtectedScoreResult",
        "attempt_id": attempt_id,
        "binding_ref": ref(binding),
    }
    # A semantic failure is terminal and cannot be retried or converted to cash.
    try:
        data = read_episode(authority, attempt_id, replay=replay)
        selection = authority.store.json(binding["selection_ref"])
        portfolio = PortfolioSpec.model_validate_json(canonical(selection["portfolio"]))
        if selection["role"] == "cash_fallback":
            returns = [0.0] * 40
            accounting = {
                "orders": [],
                "fills": [],
                "net_returns": returns,
                "snapshots": [
                    {
                        "valuation_time": data.timestamp(t, "open"),
                        "equity": portfolio.initial_cash,
                        "cash": portfolio.initial_cash,
                        "holdings": [0] * len(data.instruments),
                    }
                    for t in range(82, 122)
                ],
                "metrics": {
                    "annualized_mean_net_active_return": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "fees": 0.0,
                    "filled_trades": 0,
                    "final_cash": portfolio.initial_cash,
                },
                "benchmark": "cash_zero_interest",
            }
            predictions = []
        else:
            model_text = authority.store.get(selection["model_ref"]).decode()
            factor = FactorSpec.model_validate_json(canonical(selection["factor"]))
            predictions = predict_fixed(data, factor, model_text, 80)
            accounting = evaluate(data, 80, predictions, portfolio, 252)
        returns = returns_from_equities(
            portfolio.initial_cash,
            [s["equity"] for s in accounting["snapshots"]],
            registered_zero_equity=True,
        )
        if len(returns) != 40 or returns != accounting["net_returns"]:
            raise ValueError("scored returns differ from registered valuation grid")
        eligible = accounting["metrics"]["max_drawdown"] >= -portfolio.max_drawdown
        return envelope | {
            "status": "completed" if eligible else "risk_failure",
            "error": None,
            "predictions": predictions,
            "accounting": accounting,
            "return_count": 40,
            "utility": 252 * float(np.mean(returns)),
            "eligible": eligible,
        }
    except (ValueError, OSError) as error:
        return envelope | {
            "status": "invalid",
            "error": f"{type(error).__name__}: {error}"[:500],
            "predictions": None,
            "accounting": None,
            "return_count": 0,
            "utility": None,
            "eligible": False,
        }


def main():
    attempt_id, output, fd = sys.argv[1:]
    os.fstat(int(fd))  # The live child keeps the authority lock even if its controller dies.
    with Authority(authority_root()) as authority:
        if authority.program.get("source_kind") == "local_event_snapshot":
            from quant_rsi.evaluation.local_scoring import require_score_owner

            require_score_owner(authority.root, int(fd))
        result = compute_score(authority, attempt_id)
    atomic_write(Path(output), canonical(result))


if __name__ == "__main__":
    main()
