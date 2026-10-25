"""Finite local C5 registration and independent development-view authority."""

import fcntl
import os
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.artifacts import canonical, digest
from quant_rsi.asof import qualifications
from quant_rsi.contracts_base import StrictModel
from quant_rsi.local_projection import development_projection
from quant_rsi.local_research import LocalResearchData, export_local_data
from quant_rsi.market_contracts import Identifier, trusted_now, utc_text
from quant_rsi.researcher import ResearchPlan, package_definition
from quant_rsi.schemas import LocalDataSpec


class LocalTaskInput(StrictModel):
    task_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    data: LocalDataSpec
    development_sessions: int = Field(ge=24, le=999)
    score_session_ids: tuple[Identifier, ...] = Field(min_length=3, max_length=976)
    terminal_session_id: Identifier
    weight: str = Field(pattern=r"^(0\.[0-9]+|1(?:\.0+)?)$")
    economic_group: Identifier
    dependence_cluster: Literal["unverified_common_cluster"]
    prior_exposure: Literal[
        "public_or_previously_exposed_engineering", "unverified_engineering_only"
    ]


class LocalProgramInput(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    source: Literal["local_event_snapshot"]
    package_campaign: str
    tasks: tuple[LocalTaskInput, ...] = Field(min_length=1, max_length=4)

    @property
    def task_count(self):
        return len(self.tasks)

    @model_validator(mode="after")
    def coverage(self) -> Self:
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate local task identity")
        if (
            any(Decimal(t.weight) <= 0 for t in self.tasks)
            or sum((Decimal(t.weight) for t in self.tasks), Decimal(0)) != 1
        ):
            raise ValueError("positive local task weights must sum exactly to one")
        return self


def freeze_sources(spec):
    return {
        "created_at": trusted_now(),
        "sources": [export_local_data(task.data) for task in spec.tasks],
    }


def validate_existing_market(root):
    """A market identity may precede C5; it never substitutes for a missing C5 anchor."""
    from quant_rsi.market_data import DataRegistry

    if (root / "market").is_dir():
        with DataRegistry(root / "market"):
            pass


def economic_session_keys(data, start, stop):
    """Registered asset/currency and UTC instants survive file/source/session relabeling."""
    return sorted(
        {
            digest(
                canonical(
                    dict(
                        instrument=instrument,
                        currency=data.contract.currency,
                        open_at=utc_text(session.open_at),
                        close_at=utc_text(session.close_at),
                    )
                )
            )
            for session in data.contract.calendar.sessions[start:stop]
            for instrument in data.instruments
        }
    )


def register_tasks(spec, imported, frozen, store, private):
    parent = ResearchPlan.model_validate_json(canonical(imported["plans"]["parent"]))
    template = parent.development
    tasks, source_refs, originals = [], [], []
    for request, value in zip(spec.tasks, frozen["sources"], strict=True):
        data = LocalResearchData(value)
        if data.value["kind"] != "LocalResearchData" or data.spec != request.data:
            raise ValueError("local task differs from its frozen original source")
        a, b = data.spec.model_dump(), template.data.model_dump()
        for key in ("snapshot_ref", "sessions"):
            a.pop(key)
            b.pop(key)
        if a != b:
            raise ValueError("local task must preserve the frozen M0 training specification")
        if (
            list(data.dates[request.development_sessions :]) != list(request.score_session_ids)
            or request.terminal_session_id != request.score_session_ids[-1]
            or template.portfolio.currency != data.contract.currency
            or set(template.portfolio.fill_cap_units) != set(data.instruments)
        ):
            raise ValueError("local task has an unregistered session/universe/account grid")
        projection = development_projection(
            data, request.development_sessions, created_at=frozen["created_at"]
        )
        source_ref = private.put_json(value)
        source_refs.append(source_ref)
        view_ref = store.put_json(
            dict(
                kind="BoundDevelopmentView",
                origin_ref=store.put_json(projection["projection_lineage"]),
                data=projection,
            )
        )
        plans = {}
        for role, original_plan in imported["plans"].items():
            adjusted = deepcopy(original_plan)
            adjusted["development"]["data"] = projection["specification"]
            plan = ResearchPlan.model_validate_json(canonical(adjusted))
            if digest(canonical(package_definition(plan))) != imported["packages"][role]:
                raise ValueError("task adaptation changed the frozen researcher package")
            plans[role] = store.put_json(plan.model_dump(mode="json"))
        start, end = request.development_sessions, data.spec.sessions - 1
        task = dict(
            task=request.task_id,
            source_ref=source_ref,
            episode_ref=source_ref,
            source_kind="local_event_snapshot",
            original_snapshot_ref=data.spec.snapshot_ref,
            contract_ref=data.snapshot["contract_ref"],
            source_id=data.contract.source_id,
            registry_id=data.snapshot["registry_id"],
            original_raw_refs=sorted(data.value["raw_inputs"]),
            receipt_watermark=data.snapshot["receipt_watermark"],
            knowledge_basis=data.contract.knowledge_basis,
            qualifications=qualifications(data.contract),
            prior_exposure=request.prior_exposure,
            scientific_independence="unqualified",
            economic_score_keys=economic_session_keys(data, start, end + 1),
            economic_development_keys=economic_session_keys(data, 0, start),
            package_selection_source_snapshot_ref=template.data.snapshot_ref,
            package_selection_role="already_exposed_development_history",
            view_ref=view_ref,
            plans=plans,
            sessions=list(data.dates),
            instruments=list(data.instruments),
            weight=float(Decimal(request.weight)),
            registered_weight=request.weight,
            economic_group=request.economic_group,
            dependence_cluster=request.dependence_cluster,
            grid=dict(
                development=[0, start - 1],
                training_cutoff=data.spec.train_cutoff,
                development_decisions=[data.spec.train_cutoff, start - 1],
                decisions=[start, end - 1],
                fills=[start + 1, end],
                valuations=[start, end],
                return_count=end - start + 1,
                terminal_session=request.terminal_session_id,
                terminal_policy=template.portfolio.termination,
                terminal_exit_decision=end - 1
                if template.portfolio.termination == "queue_exit_at_next_eligible_open"
                else None,
                terminal_cost="actual_fill_cost_in_registered_terminal_close_return",
                nav0=template.portfolio.initial_cash,
                first_return="cash_before_first_next_open_fill",
                unresolved_terminal="preserve_state_and_fail_no_free_liquidation",
            ),
        )
        # A different label is not another sample or another permission to expose rows.
        for previous in tasks:
            p_score = set(previous["economic_score_keys"])
            score = set(task["economic_score_keys"])
            if p_score & score:
                raise ValueError("aliased/overlapping source score rows cannot gain task weight")
            if p_score & set(task["economic_development_keys"]) or score & set(
                previous["economic_development_keys"]
            ):
                raise ValueError("a task development view would reveal another task score rows")
        originals.append(
            dict(
                task=request.task_id,
                source_ref=source_ref,
                original_snapshot_ref=data.spec.snapshot_ref,
            )
        )
        tasks.append(task)
    return (
        private.put_json(dict(kind="LocalProtectedSourceSet", sources=source_refs)),
        tasks,
        {
            "source_kind": "local_event_snapshot",
            "registrar_exposure": {
                "kind": "LocalProgramImportExposure",
                "created_at": frozen["created_at"],
                "sources": originals,
                "purpose": "trusted_registration_and_projection_not_scoring",
                "scientific_independence": "unqualified",
            },
        },
    )


def authorized_view(root, plan, campaign_id, context, owner_fd, *, steps=None, committed=0):
    from quant_rsi.comparison import child_root, compiled_plan, launch_context
    from quant_rsi.evidence import Authority, authority_root

    if (
        not isinstance(context, dict)
        or set(context) != {"kind", "authority_id", "batch_id", "key", "grant_ref"}
        or context["kind"] != "ProtectedComparisonDevelopmentGrant"
    ):
        raise ValueError("local C5 view requires its distinct actual authority grant")
    outer = authority_root()
    lock_path = outer / ".controller.lock"
    try:
        if owner_fd is None or not os.path.samestat(os.fstat(owner_fd), lock_path.stat()):
            raise ValueError("local C5 development requires its owning authority lock")
        with lock_path.open("r+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("local C5 authority lock is not held")
        fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, TypeError) as error:
        raise ValueError("local C5 development requires its owning authority lock") from error
    with Authority(outer) as authority:
        batch, key = authority.state["batch"], context["key"]
        expected = compiled_plan(authority)
        if (
            authority.program.get("source_kind", "synthetic") != plan.development.data.source
            or batch is None
            or key not in batch["runs"]
            or batch["plan_ref"] != digest(canonical(expected))
            or batch["status"] != "RESERVED"
            or batch["exposed"]
            or context != launch_context(authority, key)
        ):
            raise ValueError("local C5 authority/task/role/plan binding mismatch")
        grant = expected["expected_coverage"]["unique"][key]
        pending = batch["pending"]
        if (
            Path(root).resolve() != child_root(authority, key).resolve()
            or campaign_id != grant["campaign_id"]
            or digest(canonical(grant)) != context["grant_ref"]
            or plan.model_dump(mode="json") != authority.store.json(grant["research_plan_ref"])
            or digest(canonical(package_definition(plan))) != grant["package_ref"]
            or pending is None
            or pending["kind"] != "develop"
            or pending["key"] != key
        ):
            raise ValueError("local C5 child/package/input/pending operation mismatch")
        initialized = batch["runs"][key]["initialized"]
        if steps is None:
            if initialized:
                raise ValueError("initialized local C5 child cannot be recreated")
        elif not initialized or steps > pending["target_observations"] - committed:
            raise ValueError("local C5 work exceeds its committed operation target")
        view = authority.store.json(grant["view_ref"])
        if plan.development.data.source == "local_event_snapshot":
            data = LocalResearchData(view["data"])
            if (
                data.value["kind"] != "DerivedLocalResearchData"
                or data.spec != plan.development.data
                or authority.store.json(view["origin_ref"]) != data.value["projection_lineage"]
            ):
                raise ValueError("local C5 development projection binding mismatch")
        return view


def local_comparison_metadata(result, parent):
    development = parent["development"]
    result["version"] = 2
    result["source_kind"] = "local_event_snapshot"
    result["training"] = dict(
        mode="M0",
        cutoff="per_task_registered_information_cutoff",
        label="actual_asof_mature_next_two_open_label",
        maturity="actual_per_row_knowledge_and_availability_mask",
        warmup="factor.window_registered_calendar",
        preprocessing="actual_asof_factor_daily_intraday_no_fitted_scaler",
        execution_seed=development["data"]["seed"],
    )
    result["fallback"]["currency"] = development["portfolio"]["currency"]
    result["utility"] = dict(
        metric="annualized_arithmetic_net_active_return",
        annualization=development["annualization"],
        return_count_by_task={t["task"]: t["grid"]["return_count"] for t in result["tasks"]},
        benchmark="cash_zero_interest",
        terminal_cost="registered_per_task_actual_fill_costs_no_assumed_liquidation",
        min_effect=0.01,
        initial_margin=0.0,
        registered_total_loss_events=[],
        source_limitation="local_engineering_only_no_independence_or_market_qualification",
    )


def seal_training(data, plan, factor, metadata, model_text, predictions, result_ref, grant, save):
    from quant_rsi.local_research import training_view, verify_local_result
    from quant_rsi.schemas import FactorSpec

    parsed = FactorSpec.model_validate_json(canonical(factor))
    verify_local_result(data, plan, parsed, metadata | {"predictions": predictions}, model_text)
    x, y, expected, mask = training_view(data, plan, parsed)
    return save(
        dict(
            kind="ReconstructedLocalTrainingPolicy",
            view_ref=grant["view_ref"],
            development_plan_ref=grant["development_plan_ref"],
            actual_mask_ref=save(mask),
            feature_values_ref=save(x.tolist()),
            mature_labels_ref=save(y.tolist()),
            **expected,
            result_ref=result_ref,
        )
    )
