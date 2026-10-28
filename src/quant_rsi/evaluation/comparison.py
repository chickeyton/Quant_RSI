"""Finite C5 engineering comparison over a single protected evidence authority."""

import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

from quant_rsi.core import controller
from quant_rsi.core.artifacts import atomic_write, canonical, decode
from quant_rsi.core.data import features
from quant_rsi.core.model import predict_fixed
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.evaluation import promotion
from quant_rsi.evaluation.evidence import Authority, ComparisonInput, locked_authority, ref
from quant_rsi.evaluation.protected_scoring import compute_score, score_binding
from quant_rsi.research import research
from quant_rsi.research.researcher import ResearchPlan


def compiled_plan(authority):
    p, store = authority.program, authority.store
    unique, logical = {}, {}
    for task in p["tasks"]:
        for role in ("parent", "candidate"):
            key = task["task"] + ":" + role
            task_plans = task.get("plans", p["plans"])
            plan = store.json(task_plans[role])
            unique[key] = {
                "key": key,
                "task_ref": ref(task),
                "role": role,
                "campaign_id": uuid.uuid5(uuid.UUID(p["authority_id"]), key).hex,
                "research_plan_ref": task_plans[role],
                "development_plan_ref": ref(plan["development"]),
                "package_ref": p["packages"][role],
                "view_ref": task["view_ref"],
                "budget": plan["development"]["budget"],
                "state_policy": "reset",
                "ranking": {
                    "metric": plan["development"]["primary_metric"],
                    "strict_min_improvement": plan["policy"]["min_improvement"],
                    "tie": "keep_incumbent",
                    "finite_accounting_required": True,
                },
                "eligibility": {
                    "min_development_sessions": plan["development"]["min_development_sessions"],
                    "min_protected_sessions": task["grid"]["return_count"],
                    "max_drawdown": plan["development"]["portfolio"]["max_drawdown"],
                },
            }
            logical[key] = key
        logical[task["task"] + ":initial"] = task["task"] + ":parent"
    result = {
        "kind": "RegisteredComparisonPlan",
        "version": 1,
        "program_ref": authority.anchor["program_ref"],
        "authority_id": p["authority_id"],
        "mode": "engineering_only",
        "tasks": p["tasks"],
        "expected_coverage": {"unique": unique, "logical": logical},
        "parent_package_ref": p["packages"]["parent"],
        "initial_package_ref": p["packages"]["parent"],
        "candidate_package_ref": p["packages"]["candidate"],
        "expected_registry_revision": 0,
        "training": {
            "mode": "M0",
            "cutoff": 40,
            "label": "open_t_plus_2/open_t_plus_1-1",
            "maturity": "last_training_label_at_open40",
            "warmup": "factor.window",
            "preprocessing": "factor_daily_intraday_no_fitted_scaler",
            "execution_seed": 0,
        },
        "fallback": {
            "only_when": "authoritative_terminal_no_valid_incumbent",
            "strategy": "cash_zero_interest",
            "currency": "synthetic_quote_currency",
            "external_flows": 0,
            "accrual": 0,
            "initial_cash": store.json(p["plans"]["parent"])["development"]["portfolio"][
                "initial_cash"
            ],
            "IR_null_is_eligible": True,
            "corrupt_or_missing_artifacts": "fail_closed",
        },
        "utility": {
            "metric": "annualized_arithmetic_net_active_return",
            "annualization": 252,
            "return_count": 40,
            "benchmark": "cash_zero_interest",
            "terminal_cost": "included_in_return_39",
            "min_effect": 0.01,
            "initial_margin": 0.0,
            "registered_total_loss_events": [],
            "source_limitation": "positive_quotes_cash_constrained_long_only_no_bankruptcy_events",
        },
        "inference": {
            "method": "blocked_uncalibrated",
            "calibration_ref": None,
            "feasibility_ref": None,
            "formal_tests_executed": 0,
            "alpha": [0.025, 0.025],
            "reason": "No calibrated inference method; one registered dependence cluster",
        },
        "retry": {
            "max_attempts_per_unique_key": 2,
            "seconds_per_attempt": 20,
            "only_after": "possibly_dispatched_without_valid_output",
            "refund": False,
        },
        "allowed_decisions": ["retain"],
        "runtime_ref": p["runtime_ref"],
    }
    if p.get("source_kind") == "local_event_snapshot":
        from quant_rsi.evaluation.local_evidence import local_comparison_metadata

        local_comparison_metadata(result, store.json(p["plans"]["parent"]))
    return result


def pointer(authority):
    batch = authority.state["batch"]
    return {
        "kind": "ComparisonPointer",
        "authority_id": authority.anchor["authority_id"],
        "program_ref": authority.anchor["program_ref"],
        "batch_id": batch["id"],
        "plan_ref": batch["plan_ref"],
    }


def verify_pointer(authority, output):
    if decode((Path(output) / "comparison.json").read_bytes()) != pointer(authority):
        raise ValueError("comparison belongs to another authority/plan/batch")


def register_comparison(spec: ComparisonInput, output: Path):
    if spec.mode == "confirmatory":
        raise ValueError(
            "confirmatory preflight blocked_uncalibrated: no calibrated method/feasibility; "
            "no protected access or grant"
        )
    with locked_authority() as (root, _), Authority(root) as authority:
        if authority.program.get("source_kind") == "local_event_snapshot":
            verify_registry(authority)
            if authority.state["batch"] is None and (
                authority.state["development_attempts_granted"]
                or authority.state["development_seconds_granted"]
                or authority.state["attempts"]
                or authority.state["scoring_seconds_charged"]
            ):
                raise ValueError("unallocated local C5 cost ledger cannot be reset by registration")
        plan = compiled_plan(authority)
        plan_ref = authority.store.put_json(plan)
        output = output.resolve()
        path = output / "comparison.json"
        if output.exists() and any(output.iterdir()) and not path.exists():
            raise ValueError("output directory is not an empty comparison handle")
        if authority.state["batch"] is None:

            def reserve(state):
                count = len(plan["expected_coverage"]["unique"])
                state["development_attempts_granted"] = count * 8
                state["development_seconds_granted"] = count * 160
                state["batch"] = {
                    "id": uuid.uuid5(uuid.UUID(plan["authority_id"]), "only_batch").hex,
                    "plan_ref": plan_ref,
                    "status": "RESERVED",
                    "exposed": False,
                    "cancelled": False,
                    "cursor": 0,
                    "window": None,
                    "pending": None,
                    "selections_ref": None,
                    "report_ref": None,
                    "runs": {
                        key: {"grant_ref": authority.store.put_json(grant), "initialized": False}
                        for key, grant in plan["expected_coverage"]["unique"].items()
                    },
                    "scores": {
                        key: {"attempt_ids": [], "result_ref": None, "status": "pending"}
                        for key in plan["expected_coverage"]["unique"]
                    },
                }

            authority.update("batch_reserved_with_irrevocable_development_grants", reserve)
        elif authority.state["batch"]["plan_ref"] != plan_ref:
            raise ValueError("this authority already reserved its only comparison")
        expected = canonical(pointer(authority))
        if path.exists() and path.read_bytes() != expected:
            raise ValueError("output pointer conflict")
        if not path.exists():
            atomic_write(path, expected)
        verify_comparison(authority)
        return summarize(authority)


def child_root(authority, key):
    return authority.home / "development" / key.replace(":", "__")


def launch_context(authority, key):
    batch = authority.state["batch"]
    return {
        "kind": "ProtectedComparisonDevelopmentGrant",
        "authority_id": authority.anchor["authority_id"],
        "batch_id": batch["id"],
        "key": key,
        "grant_ref": batch["runs"][key]["grant_ref"],
    }


def inspect_child(authority, key, *, selection=False, publish=False):
    batch = authority.state["batch"]
    grant = authority.store.json(batch["runs"][key]["grant_ref"])
    path = child_root(authority, key)
    marker = path / "research_initialization.json"
    if marker.exists():
        if batch["runs"][key]["initialized"]:
            raise ValueError("initialized child unexpectedly has a bootstrap marker")
        saved = decode(marker.read_bytes())
        view = authority.store.json(grant["view_ref"])
        refs = [*authority.program["package_artifacts"], view["origin_ref"]]
        expected = {
            "kind": "research_initialization_v1",
            "campaign_id": grant["campaign_id"],
            "plan": authority.store.json(grant["research_plan_ref"]),
            "runtime": controller.runtime_identity(),
            "package_artifacts": {r: authority.store.json(r) for r in refs},
            "launch_context": launch_context(authority, key),
            "bound_view": view,
        }
        if saved != expected:
            raise ValueError("child bootstrap differs from fixed grant/UUID/view")
        # Status remains read-only. The owning develop operation alone finishes
        # initialization, even when manifest exists before the SQLite commit.
        return None
    if not (path / "manifest.json").exists():
        if batch["runs"][key]["initialized"]:
            raise ValueError("initialized development run missing; cannot recreate")
        return None
    with controller.controller_lock(path):
        journal, store, manifest, plan, data = research.open_research(path)
        try:
            if (
                manifest["campaign_id"] != grant["campaign_id"]
                or manifest["research_plan_ref"] != grant["research_plan_ref"]
                or manifest["package_ref"] != grant["package_ref"]
                or manifest.get("bound_view_ref") != grant["view_ref"]
                or store.json(manifest["launch_context_ref"]) != launch_context(authority, key)
            ):
                raise ValueError("development run/plan/view/grant binding mismatch")
            trace = research.verify_trace(journal, store, manifest, plan, data, replay=True)
            report = research.summarize(journal, store, manifest, plan, trace)
            if not selection:
                return report
            if report["execution_state"] != "completed" or report["pending_requests"]:
                raise ValueError("selection requires a terminal committed development run")

            def save(value):
                result = ref(value)
                if publish:
                    authority.store.put_json(value)
                elif authority.store.json(result) != value:
                    raise ValueError("sealed provenance differs from actual development")
                return result

            incumbent = report["incumbent"]
            model_ref, factor, result_ref, training_ref, preprocessing_ref = (
                None,
                None,
                None,
                None,
                None,
            )
            if incumbent is not None:
                result_ref, factor = incumbent["result_ref"], incumbent["factor"]
                bundle = store.json(result_ref)
                model_ref = bundle["model_ref"]
                metadata = store.json(bundle["worker_metadata_ref"])
                if plan.development.data.source == "local_event_snapshot":
                    from quant_rsi.evaluation.local_evidence import seal_training

                    training_ref = seal_training(
                        data,
                        plan.development,
                        factor,
                        metadata,
                        store.get(model_ref).decode(),
                        store.json(bundle["predictions_ref"]),
                        result_ref,
                        grant,
                        save,
                    )
                else:
                    rows = list(range(factor["window"], plan.development.data.train_cutoff - 1))
                    if (
                        metadata["training_rows"] != len(rows) * len(data.instruments)
                        or metadata["training_information_cutoff"] != data.timestamp(40, "close")
                        or metadata["last_training_label_available"] != data.timestamp(40, "open")
                    ):
                        raise ValueError("selected model mature-label provenance mismatch")
                    model_text = store.get(model_ref).decode()
                    parsed_factor = FactorSpec.model_validate_json(canonical(factor))
                    if predict_fixed(data, parsed_factor, model_text, 40) != store.json(
                        bundle["predictions_ref"]
                    ):
                        raise ValueError("selected model bytes disagree with committed predictions")
                    train_x = features(data, parsed_factor)[rows]
                    labels = data.opens[np.asarray(rows) + 2] / data.opens[np.asarray(rows) + 1] - 1
                    training_ref = save(
                        {
                            "kind": "ReconstructedTrainingPolicy",
                            "view_ref": grant["view_ref"],
                            "development_plan_ref": grant["development_plan_ref"],
                            "rows": rows,
                            "feature_values_ref": save(train_x.tolist()),
                            "mature_labels_ref": save(labels.tolist()),
                            "information_cutoff": data.timestamp(40, "close"),
                            "last_label_available": data.timestamp(40, "open"),
                            "training_rows": len(rows) * len(data.instruments),
                            "result_ref": result_ref,
                        }
                    )
                preprocessing_ref = save(
                    {
                        "kind": "FixedPreprocessing",
                        "factor": factor,
                        "features": ["factor", "daily_return", "intraday_return"],
                        "fitted_scaler": None,
                        "runtime_ref": authority.program["runtime_ref"],
                    }
                )
                for artifact_ref in [result_ref, *bundle.values()]:
                    raw = store.get(artifact_ref)
                    if publish:
                        if authority.store.put(raw) != artifact_ref:
                            raise ValueError("selected artifact copy mismatch")
                    elif authority.store.get(artifact_ref) != raw:
                        raise ValueError("sealed artifact differs from selected development result")
            elif report["research_outcome"] != "no_valid_incumbent":
                raise ValueError("fallback lacks an authoritative no-valid-candidate outcome")
            execution_ref = save(
                {
                    "kind": "ExecutionStateManifest",
                    "initial_cash": plan.development.portfolio.initial_cash,
                    "positions": [],
                    "pending_orders": [],
                    "accrual": 0,
                    "cursor": plan.development.data.sessions,
                    "task_ref": grant["task_ref"],
                    "account_reset": True,
                    "model_reset": "sealed_model_no_fit",
                    "warmup_view_ref": grant["view_ref"],
                }
            )
            return {
                "kind": "SelectedStrategyBundle",
                "key": key,
                "campaign_id": grant["campaign_id"],
                "grant_ref": batch["runs"][key]["grant_ref"],
                "package_ref": grant["package_ref"],
                "research_plan_ref": grant["research_plan_ref"],
                "task_ref": grant["task_ref"],
                "checkpoint_ref": save(store.json(report["checkpoint_ref"])),
                "source_manifest_ref": save(manifest),
                "result_ref": result_ref,
                "role": "active" if incumbent is not None else "cash_fallback",
                "fallback_reason": None
                if incumbent is not None
                else "authoritative_terminal_no_valid_incumbent",
                "factor": factor,
                "model_ref": model_ref,
                "preprocessing_ref": preprocessing_ref,
                "training_policy_ref": training_ref,
                "execution_state_ref": execution_ref,
                "portfolio": plan.development.portfolio.model_dump(mode="json"),
            }
        finally:
            journal.close()


def selection_coverage(authority, *, publish=False):
    batch = authority.state["batch"]
    plan = authority.store.json(batch["plan_ref"])
    unique = {}
    for key in plan["expected_coverage"]["unique"]:
        selection = inspect_child(authority, key, selection=True, publish=publish)
        if selection is None:
            raise ValueError("expected development run is missing")
        unique[key] = ref(selection)
        if publish:
            authority.store.put_json(selection)
        elif authority.store.json(unique[key]) != selection:
            raise ValueError("sealed selection mismatch")
    return {
        "kind": "SelectionCoverage",
        "plan_ref": batch["plan_ref"],
        "unique": unique,
        "logical": {
            key: {"unique_key": target, "selection_ref": unique[target]}
            for key, target in plan["expected_coverage"]["logical"].items()
        },
    }


def verify_comparison(authority, *, replay=False):
    authority.verify_chain()
    state = authority.state
    batch = state["batch"]
    if batch is None:
        return
    plan = compiled_plan(authority)
    if authority.store.json(batch["plan_ref"]) != plan:
        raise ValueError("registered comparison plan differs from frozen program")
    expected = plan["expected_coverage"]["unique"]
    if set(batch["runs"]) != set(expected) or set(batch["scores"]) != set(expected):
        raise ValueError("expected unique coverage mismatch")
    if (state["development_attempts_granted"], state["development_seconds_granted"]) != (
        8 * len(expected),
        160 * len(expected),
    ):
        raise ValueError("development grant accounting mismatch")
    for key, grant in expected.items():
        if authority.store.json(batch["runs"][key]["grant_ref"]) != grant:
            raise ValueError("development grant differs from predeclared plan")
        if batch["runs"][key]["initialized"]:
            inspect_child(authority, key)
        score = batch["scores"][key]
        ids = score["attempt_ids"]
        if len(ids) > 2 or len(set(ids)) != len(ids):
            raise ValueError("scoring retry limit or attempt identity violation")
        for index, attempt_id in enumerate(ids):
            attempt = state["attempts"].get(attempt_id)
            if attempt is None or attempt["key"] != key or attempt["seconds"] != 20:
                raise ValueError("scoring grant coverage mismatch")
            if attempt["binding_ref"] != ref(score_binding(authority, key)):
                raise ValueError("score attempt binding mismatch")
            if index and state["attempts"][ids[index - 1]]["status"] != "interrupted":
                raise ValueError("retry does not follow an infrastructure interruption")
            if attempt["result_ref"]:
                result = authority.store.json(attempt["result_ref"])
                if (
                    result["attempt_id"] != attempt_id
                    or result["binding_ref"] != attempt["binding_ref"]
                ):
                    raise ValueError("score output substitution")
                if authority.program.get("source_kind") == "local_event_snapshot":
                    from quant_rsi.evaluation.local_scoring import validate_accepted_score

                    validate_accepted_score(authority, attempt_id, result)
                elif replay and result != compute_score(authority, attempt_id, replay=True):
                    raise ValueError("fixed-model scoring replay mismatch")
        if score["result_ref"] and score["result_ref"] != state["attempts"][ids[-1]]["result_ref"]:
            raise ValueError("score selected another attempt result")
    if set(state["attempts"]) != {
        a for score in batch["scores"].values() for a in score["attempt_ids"]
    }:
        raise ValueError("unallocated scoring attempts")
    if batch["selections_ref"] is not None:
        if authority.store.json(batch["selections_ref"]) != selection_coverage(authority):
            raise ValueError("selection logical/unique coverage mismatch")
    if batch["exposed"] and batch["selections_ref"] is None:
        raise ValueError("exposure preceded selection sealing")
    if batch["report_ref"] is not None:
        if authority.store.json(batch["report_ref"]) != build_report(authority):
            raise ValueError("report differs from registered actual score results")
    verify_registry(authority)


def decision_binding(authority):
    batch = authority.state["batch"]
    plan = authority.store.json(batch["plan_ref"])
    return {
        "plan_ref": batch["plan_ref"],
        "report_ref": batch["report_ref"],
        "expected_package_ref": plan["parent_package_ref"],
        "expected_revision": plan["expected_registry_revision"],
        "candidate_package_ref": plan["candidate_package_ref"],
        "mode": plan["mode"],
    }


def verify_registry(authority):
    state = authority.state
    reconstructed = {
        "registry": {
            "package_ref": authority.program["packages"]["parent"],
            "revision": 0,
            "last_report_ref": None,
            "last_decision_ref": None,
        },
        "decisions": {},
    }
    for request_id, saved in sorted(
        state["decisions"].items(), key=lambda item: item[1]["decision"]["old_revision"]
    ):
        batch = state["batch"]
        if batch["status"] != "RETIRED" or batch["report_ref"] is None:
            raise ValueError("registry decision precedes terminal evidence")
        decision = promotion.transition(
            reconstructed, request_id, decision_binding(authority), "retain"
        )
        if (
            saved != reconstructed["decisions"][request_id]
            or authority.store.json(ref(decision)) != decision
        ):
            raise ValueError("registry decision differs from registered production authorization")
    if reconstructed != {"registry": state["registry"], "decisions": state["decisions"]}:
        raise ValueError("registry pointer/revision/report history mismatch")


def next_operation(authority):
    batch = authority.state["batch"]
    for key in batch["runs"]:
        report = inspect_child(authority, key)
        if report is None or report["execution_state"] != "completed":
            return {
                "kind": "develop",
                "key": key,
                "target_observations": 1
                if report is None
                else len([r for r in report["results"] if "observation" in r]) + 1,
            }
    if batch["selections_ref"] is None:
        return {"kind": "seal"}
    if not batch["exposed"]:
        return {"kind": "open"}
    for key, score in batch["scores"].items():
        if score["status"] == "pending":
            ordinal = len(score["attempt_ids"])
            return {
                "kind": "score",
                "key": key,
                "attempt_id": uuid.uuid5(uuid.UUID(batch["id"]), f"{key}:score:{ordinal}").hex,
            }
    return {"kind": "finalize"}


def develop(authority, operation, owner_fd):
    key = operation["key"]
    batch = authority.state["batch"]
    grant = authority.store.json(batch["runs"][key]["grant_ref"])
    path = child_root(authority, key)
    report = inspect_child(authority, key)
    if report is None:
        plan = ResearchPlan.model_validate_json(authority.store.get(grant["research_plan_ref"]))
        view = authority.store.json(grant["view_ref"])
        refs = [*authority.program["package_artifacts"], view["origin_ref"]]
        research.create_research(
            plan,
            path,
            campaign_id=grant["campaign_id"],
            package_artifacts={r: authority.store.json(r) for r in refs},
            launch_context=launch_context(authority, key),
            bound_view=view,
            owner_fd=owner_fd,
            initialize_only=True,
        )
        report = inspect_child(authority, key)
    if not batch["runs"][key]["initialized"]:
        authority.update(
            "development_initialized", lambda s: s["batch"]["runs"][key].update(initialized=True)
        )
    count = sum("observation" in r for r in report["results"])
    if count > operation["target_observations"]:
        raise ValueError("development advanced beyond its pending operation target")
    if count < operation["target_observations"] and report["execution_state"] != "completed":
        research.resume_research(path, steps=1, owner_fd=owner_fd)
    report = inspect_child(authority, key)
    count = sum("observation" in r for r in report["results"])
    if count != operation["target_observations"] and report["execution_state"] != "completed":
        raise ValueError("development operation has not reached its committed target")


def process_score(authority, operation, owner_fd):
    key, attempt_id = operation["key"], operation["attempt_id"]
    if attempt_id not in authority.state["attempts"]:
        binding_ref = authority.store.put_json(score_binding(authority, key))

        def reserve(state):
            batch = state["batch"]
            score = batch["scores"][key]
            if batch["status"] != "OPENED" or not batch["exposed"]:
                raise ValueError("score cannot be charged before OPENED")
            if len(score["attempt_ids"]) >= 2 or score["status"] != "pending":
                raise ValueError("score retry budget exhausted")
            if (
                score["attempt_ids"]
                and state["attempts"][score["attempt_ids"][-1]]["status"] != "interrupted"
            ):
                raise ValueError("semantic score failures cannot be retried")
            if (
                state["scoring_seconds_charged"] + 20
                > authority.program["allowance"]["scoring_seconds"]
            ):
                raise ValueError("program scoring budget exhausted")
            state["scoring_seconds_charged"] += 20
            state["attempts"][attempt_id] = {
                "key": key,
                "binding_ref": binding_ref,
                "seconds": 20,
                "outbox": "READY",
                "status": "reserved",
                "result_ref": None,
                "raw_ref": None,
                "error": None,
            }
            score["attempt_ids"].append(attempt_id)

        authority.update("score_attempt_debit_and_outbox", reserve)
    attempt = authority.state["attempts"][attempt_id]
    if attempt["status"] in {"completed", "risk_failure", "invalid", "interrupted"}:
        return
    path = authority.home / "score_work" / attempt_id
    path.mkdir(parents=True, exist_ok=True)
    result_path = path / "result.json"
    if attempt["outbox"] == "READY":
        if result_path.exists():
            raise ValueError("unissued attempt has an unauthorized output")
        authority.update(
            "score_dispatch_may_have_started",
            lambda s: s["attempts"][attempt_id].update(outbox="DISPATCHED", status="running"),
        )
        with (path / "worker.log").open("ab") as log:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "quant_rsi.evaluation.protected_scoring",
                        attempt_id,
                        str(result_path),
                        str(owner_fd),
                    ],
                    stdout=log,
                    stderr=log,
                    timeout=20,
                    pass_fds=(owner_fd,),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                pass
    result, error, raw_ref, source_validation_ref = None, None, None, None
    if result_path.exists():
        raw = result_path.read_bytes()
        raw_ref = authority.store.put(raw)
        try:
            result = decode(raw)
            canonical(result)
            if authority.program.get("source_kind") == "local_event_snapshot":
                from quant_rsi.evaluation.local_scoring import validate_accepted_score

                source_validation_ref = authority.store.put_json(
                    validate_accepted_score(
                        authority, attempt_id, result, first_acceptance=True, raw_ref=raw_ref
                    )
                )
            elif result != compute_score(authority, attempt_id):
                raise ValueError("cached scoring result differs from its bound computation")
        except (ValueError, TypeError, KeyError) as exc:
            error = f"invalid score output: {exc}"[:500]
            result = None
    status = "invalid" if error else ("interrupted" if result is None else result["status"])
    result_ref = None if result is None else authority.store.put_json(result)

    def finish(state):
        state["attempts"][attempt_id].update(
            status=status, result_ref=result_ref, raw_ref=raw_ref, error=error
        )
        if authority.program.get("source_kind") == "local_event_snapshot":
            state["attempts"][attempt_id]["source_validation_receipt_ref"] = source_validation_ref
        score = state["batch"]["scores"][key]
        if status != "interrupted" or len(score["attempt_ids"]) == 2:
            score.update(
                status="infrastructure_failed" if status == "interrupted" else status,
                result_ref=result_ref,
            )

    authority.update("score_attempt_finished", finish)


def build_report(authority):
    state = authority.state
    batch = state["batch"]
    plan = authority.store.json(batch["plan_ref"])
    development_costs = {}
    for key in batch["runs"]:
        child = inspect_child(authority, key)
        development_costs[key] = {
            "attempts_consumed": child["attempts_reserved"],
            "worker_seconds_consumed": child["worker_seconds_charged"],
            "attempts_granted": 8,
            "worker_seconds_granted": 160,
            "attempts_outstanding_not_refunded": 8 - child["attempts_reserved"],
            "worker_seconds_outstanding_not_refunded": 160 - child["worker_seconds_charged"],
        }
    results = {}
    for key, score in batch["scores"].items():
        if score["status"] == "pending":
            raise ValueError("cannot report incomplete score coverage")
        results[key] = (
            None if score["result_ref"] is None else authority.store.json(score["result_ref"])
        )
    tasks = []
    for task in plan["tasks"]:
        parent, candidate = [results[task["task"] + ":" + role] for role in ("parent", "candidate")]
        valid = all(
            result is not None and result["status"] == "completed" for result in (parent, candidate)
        )
        delta = (
            None
            if parent is None
            or candidate is None
            or parent["utility"] is None
            or candidate["utility"] is None
            else candidate["utility"] - parent["utility"]
        )
        tasks.append(
            {
                "task": task["task"],
                "weight": task["weight"],
                "valid": valid,
                "parent_utility": None if parent is None else parent["utility"],
                "candidate_utility": None if candidate is None else candidate["utility"],
                "candidate_minus_parent": delta,
                "candidate_minus_initial": delta,
                "initial_alias": task["task"] + ":parent",
                "economic_group": task["economic_group"],
            }
        )
    valid = all(t["valid"] for t in tasks)
    coverage = authority.store.json(batch["selections_ref"])
    integrity_failure = any(score["status"] == "invalid" for score in batch["scores"].values())
    reference_unavailable = any(
        score["status"] != "completed"
        and (
            key.endswith(":parent")
            or authority.store.json(coverage["unique"][key])["role"] == "cash_fallback"
        )
        for key, score in batch["scores"].items()
    )
    candidate_failed = any(
        key.endswith(":candidate") and score["status"] != "completed"
        for key, score in batch["scores"].items()
    )
    failure_classification = (
        "integrity_failure"
        if integrity_failure
        else "reference_unavailable"
        if reference_unavailable
        else "candidate_failed"
        if candidate_failed
        else "none"
    )
    semantic_candidate_failure = any(
        key.endswith(":candidate") and score["status"] == "risk_failure"
        for key, score in batch["scores"].items()
    )
    evidence_status = (
        "invalid"
        if integrity_failure
        else "insufficient_evidence"
        if reference_unavailable
        else "not_supported"
        if semantic_candidate_failure
        else "insufficient_evidence"
    )
    return {
        "kind": "ComparisonReport",
        "authority_id": authority.anchor["authority_id"],
        "program_ref": authority.anchor["program_ref"],
        "batch_id": batch["id"],
        "plan_ref": batch["plan_ref"],
        "mode": plan["mode"],
        "origin": "actual_local_development_selection_and_registered_decimal_account"
        if plan.get("source_kind") == "local_event_snapshot"
        else "actual_synthetic_selected_models",
        "tasks": tasks,
        "selection_coverage_ref": batch["selections_ref"],
        "scores": batch["scores"],
        "weighted_delta": sum(t["weight"] * t["candidate_minus_parent"] for t in tasks)
        if valid
        else None,
        "execution_status": "completed" if valid else "failed_consumed",
        "evidence_status": evidence_status,
        "failure_classification": failure_classification,
        "promotion_decision": "retain",
        "capability_status": "engineering_only_confirmatory_blocked",
        "promotion_eligible": False,
        "inference": plan["inference"]
        | {"confidence_interval": None, "p_value": None, "power": None},
        "error_allowance": {
            "allocated": 0.05,
            "formal_tests_executed": 0,
            "retired_after_engineering_exposure": 0.05,
            "reusable": 0.0,
        },
        "development_attempts_granted": state["development_attempts_granted"],
        "development_seconds_granted": state["development_seconds_granted"],
        "scoring_seconds_charged": state["scoring_seconds_charged"],
        "development_costs": development_costs,
        "historical_source_costs": authority.program["historical_costs"],
        "economic_mass": 1.0,
        "dependence_clusters": 1,
        "synthetic_only": plan.get("source_kind") != "local_event_snapshot",
        **(
            {
                "source_kind": "local_event_snapshot",
                "local_engineering_only": True,
                "source_qualifications": {
                    task["task"]: {
                        key: task[key]
                        for key in (
                            "knowledge_basis",
                            "qualifications",
                            "prior_exposure",
                            "scientific_independence",
                            "package_selection_role",
                        )
                    }
                    for task in plan["tasks"]
                },
            }
            if plan.get("source_kind") == "local_event_snapshot"
            else {}
        ),
    }


def run_operation(authority, operation, owner_fd):
    kind = operation["kind"]
    if kind == "develop":
        develop(authority, operation, owner_fd)
    elif kind == "seal":
        coverage_ref = authority.store.put_json(selection_coverage(authority, publish=True))
        if authority.state["batch"]["selections_ref"] is None:
            authority.update(
                "all_selections_sealed", lambda s: s["batch"].update(selections_ref=coverage_ref)
            )
        elif authority.state["batch"]["selections_ref"] != coverage_ref:
            raise ValueError("selection sealing conflict")
    elif kind == "open":
        verify_comparison(authority)
        if not authority.state["batch"]["exposed"]:
            authority.update(
                "protected_batch_OPENED", lambda s: s["batch"].update(exposed=True, status="OPENED")
            )
    elif kind == "score":
        process_score(authority, operation, owner_fd)
    elif kind == "finalize":
        report_ref = authority.store.put_json(build_report(authority))
        if authority.state["batch"]["report_ref"] is None:
            authority.update(
                "comparison_completed",
                lambda s: s["batch"].update(report_ref=report_ref, status="COMPLETED"),
            )
        elif authority.state["batch"]["report_ref"] != report_ref:
            raise ValueError("final report conflict")
    else:
        raise ValueError("unknown pending controller operation")


def resume_comparison(output: Path, *, steps=1):
    research.validate_steps(steps)
    with locked_authority() as (root, owner_fd), Authority(root) as authority:
        verify_pointer(authority, output)
        verify_comparison(authority)
        batch = authority.state["batch"]
        if batch["status"] == "RETIRED":
            return summarize(authority)
        if batch["window"] is None:
            authority.update(
                "finite_window_reserved",
                lambda s: s["batch"].update(
                    window={"steps": steps, "target": s["batch"]["cursor"] + steps}
                ),
            )
        elif batch["window"]["steps"] != steps:
            raise ValueError(
                "pending invocation has a fixed steps/target; resume with its original --steps"
            )
        while True:
            batch = authority.state["batch"]
            if batch["pending"] is None:
                if batch["status"] == "COMPLETED":
                    authority.update(
                        "exposed_batch_retired",
                        lambda s: s["batch"].update(status="RETIRED", window=None),
                    )
                    break
                if batch["cursor"] >= batch["window"]["target"]:
                    authority.update(
                        "finite_window_paused", lambda s: s["batch"].update(window=None)
                    )
                    break
                operation = next_operation(authority) | {"cursor": batch["cursor"]}
                authority.update(
                    "operation_reserved", lambda s, op=operation: s["batch"].update(pending=op)
                )
            operation = authority.state["batch"]["pending"]
            run_operation(authority, operation, owner_fd)

            def complete(state, operation=operation):
                current = state["batch"]
                if current["pending"] != operation or current["cursor"] != operation["cursor"]:
                    raise ValueError("operation receipt conflict")
                current.update(cursor=current["cursor"] + 1, pending=None)

            authority.update("operation_committed", complete)
        verify_comparison(authority)
        return summarize(authority)


def summarize(authority):
    summary = authority.summary()
    batch = authority.state["batch"]
    if batch is not None:
        summary.update(
            cursor=batch["cursor"],
            pending=batch["pending"],
            window=batch["window"],
            exposed=batch["exposed"],
            cancelled=batch["cancelled"],
            plan_ref=batch["plan_ref"],
            execution_status="completed"
            if batch["report_ref"]
            else ("cancelled" if batch["cancelled"] else "paused"),
            report_ref=batch["report_ref"],
            selection_coverage_ref=batch["selections_ref"],
            score_statuses={key: score["status"] for key, score in batch["scores"].items()},
            capability_status="engineering_only_confirmatory_blocked",
        )
        if batch["report_ref"]:
            summary["report"] = authority.store.json(batch["report_ref"])
            for key in ("execution_status", "evidence_status", "promotion_decision"):
                summary[key] = summary["report"][key]
    return summary


def inspect_comparison(output, *, replay=False):
    with locked_authority() as (root, _), Authority(root) as authority:
        verify_pointer(authority, output)
        verify_comparison(authority, replay=replay)
        return summarize(authority)


def cancel_comparison(output):
    with locked_authority() as (root, _), Authority(root) as authority:
        verify_pointer(authority, output)
        verify_comparison(authority)
        batch = authority.state["batch"]
        if batch["cancelled"]:
            return summarize(authority)
        if batch["status"] != "RESERVED" or batch["exposed"]:
            raise ValueError("only an unexposed RESERVED batch can be cancelled; no refunds")
        authority.update(
            "reserved_batch_cancelled_and_retired",
            lambda s: s["batch"].update(
                status="RETIRED", cancelled=True, window=None, pending=None
            ),
        )
        return summarize(authority)


def decide_comparison(output, request_id):
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("decision request ID must contain 1..128 characters")
    with locked_authority() as (root, _), Authority(root) as authority:
        verify_pointer(authority, output)
        verify_comparison(authority)
        batch = authority.state["batch"]
        if batch["status"] != "RETIRED" or batch["report_ref"] is None:
            raise ValueError("decision requires a completed retired comparison")
        plan = authority.store.json(batch["plan_ref"])
        report = authority.store.json(batch["report_ref"])
        if (
            plan["mode"] != "engineering_only"
            or plan["allowed_decisions"] != ["retain"]
            or report != build_report(authority)
        ):
            raise ValueError("registered authority/report eligibility mismatch")
        binding = decision_binding(authority)
        existing = authority.state["decisions"].get(request_id)
        if existing is not None:
            if existing["binding_ref"] != ref(binding):
                raise ValueError("decision request binding conflict")
            return existing["decision"]
        result = {}

        def decide(state):
            result.update(promotion.transition(state, request_id, binding, "retain"))
            authority.store.put_json(result)

        authority.update("registry_retain_pointer_report_decision_atomic", decide)
        return result
