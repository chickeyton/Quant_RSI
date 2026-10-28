"""Bounded discovery campaigns using the existing forecast worker and Journal.

The parent preallocates a finite roster. Child identities cannot mint fit grants;
every child verifies its immutable allocation before prepare, dispatch and replay.
"""

import re
from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.core.journal import Journal
from quant_rsi.evaluation.evidence import authority_root
from quant_rsi.factors.discovery_graph import FactorLeaf, grammar, propose
from quant_rsi.forecast.forecast_experiment import (
    commit_event,
    event_payload,
    experiment_root,
    inspect_experiment,
    prepare_experiment,
    resume_experiment,
)
from quant_rsi.forecast.forecast_inputs import ForecastInputs
from quant_rsi.forecast.forecast_protocol import InsufficientComparisonEvidence
from quant_rsi.market import market_contracts


class DiscoveryPlan(StrictModel):
    schema_version: Literal[1] = 1
    campaign_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    instrument: str = Field(pattern=r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
    leaves: tuple[FactorLeaf, ...] = Field(min_length=2, max_length=4)
    fold_origins: tuple[tuple[int, ...], ...] = Field(min_length=2, max_length=3)
    max_rounds: int = Field(default=3, ge=1, le=6)
    min_improvement_pp: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    exposure: Literal["engineering_fixture", "historical_market_exploration"]
    policy: Literal["bounded_feedback_frontier_v1"] = "bounded_feedback_frontier_v1"

    @model_validator(mode="after")
    def registered(self) -> Self:
        if len({leaf.name for leaf in self.leaves}) != len(self.leaves):
            raise ValueError("duplicate discovery leaf name")
        if len(
            {canonical(leaf.model_dump(mode="json") | {"name": "leaf"}) for leaf in self.leaves}
        ) != len(self.leaves):
            raise ValueError("duplicate discovery leaf definition/binding")
        previous = -1
        for origins in self.fold_origins:
            if (
                not origins
                or list(origins) != list(range(origins[0], origins[-1] + 1, 5))
                or origins[0] < 77
                or origins[0] <= previous
            ):
                raise ValueError("discovery requires disjoint chronological step-five folds")
            previous = origins[-1] + 5
        if self.max_rounds > len(grammar(self.leaves)):
            raise ValueError("round allowance exceeds finite candidate grammar")
        return self

    @property
    def fits_per_round(self):
        return 2 * len(self.fold_origins) + 1


def campaign_root(campaign_id):
    if not isinstance(campaign_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", campaign_id):
        raise ValueError("invalid discovery campaign ID")
    return authority_root() / "factor_discovery" / campaign_id


def _news_owner(plan):
    return dict(
        kind="FactorDiscoveryCampaign",
        campaign_id=plan.campaign_id,
        plan_ref=digest(canonical(plan.model_dump(mode="json"))),
        allocated_fits=plan.max_rounds * plan.fits_per_round,
    )


def _load(campaign_id, *, verify_runtime=True):
    root = campaign_root(campaign_id)
    raw = (root / "manifest.json").read_bytes()
    manifest = decode(raw)
    if manifest["root"] != str(root.resolve()) or manifest["campaign_id"] != campaign_id:
        raise ValueError("discovery authority moved or identity changed")
    store = ArtifactStore(root / "artifacts")
    journal = Journal(root)
    try:
        metadata = dict(journal.connection.execute("SELECT key, value FROM metadata"))
        if metadata != {"manifest_digest": digest(raw)}:
            raise ValueError("discovery manifest/journal mismatch")
        plan = DiscoveryPlan.model_validate_json(store.get(manifest["plan_ref"]))
        inputs = ForecastInputs(store.json(manifest["input_ref"]))
        if plan.campaign_id != campaign_id or inputs.ref != manifest["input_ref"]:
            raise ValueError("discovery input identity mismatch")
        if verify_runtime and store.json(manifest["runtime_ref"]) != controller.runtime_identity():
            raise ValueError("discovery requires its frozen source/runtime")
        if manifest["news_accounting"] is not None:
            from quant_rsi.factors.news import authorize_news_comparison

            receipt = authorize_news_comparison(
                inputs,
                "discovery-" + campaign_id,
                plan.fold_origins,
                verify_only=True,
                comparison_owner=_news_owner(plan),
            )
            if receipt != manifest["news_accounting"]:
                raise ValueError("discovery news evidence/accounting changed")
        _verify_instances(inputs, plan, latest=False)
        return journal, store, manifest, plan, inputs
    except Exception:
        journal.close()
        raise


def _verify_instances(inputs, plan, *, latest):
    from quant_rsi.factors.lifecycle import verify_snapshot

    for leaf in plan.leaves:
        if leaf.binding_ref is not None:
            binding = decode(inputs.evidence[leaf.binding_ref])
            source = inputs.target if leaf.role == "target" else inputs.contexts.get(leaf.role)
            if (
                source is None
                or leaf.definition is None
                or binding["definition"] != leaf.definition.model_dump(mode="json")
                or binding["source"]["entity"] != source[0].instruments[0]
                or binding["source"]["field"] != "close"
            ):
                raise ValueError("catalog definition/source differs from executable leaf")
        if leaf.role != "event":
            continue
        if leaf.instance_ref not in inputs.evidence:
            raise ValueError("event instance snapshot evidence missing")
        value = decode(inputs.evidence[leaf.instance_ref])
        if value["manifest"]["binding"]["instrument"] != plan.instrument:
            raise ValueError("event instance belongs to another instrument")
        verify_snapshot(value, latest=latest)


def bind_catalog(envelope, config_root, market, instrument, roles):
    """Explicitly select numeric catalog bindings and retain their ownership evidence.

    roles maps binding IDs to registered ForecastInputs roles. Text specifications
    without an executable task are rejected, never silently reinterpreted as news.
    """
    from quant_rsi.factors.catalog import resolve_instrument_factors

    resolved = resolve_instrument_factors(config_root, market, instrument)
    by_name = {item["binding_id"]: item for item in resolved["factors"]}
    value = decode(canonical(envelope))
    leaves = []
    for name, role in roles.items():
        if name not in by_name or by_name[name]["definition"]["kind"] != "numeric":
            raise ValueError("selected catalog binding has no numeric executor")
        binding = by_name[name]
        raw = canonical(binding)
        ref = digest(raw)
        value["evidence_hex"][ref] = raw.hex()
        leaves.append(
            FactorLeaf.model_validate_json(
                canonical(
                    dict(name=name, role=role, definition=binding["definition"], binding_ref=ref)
                )
            )
        )
    return ForecastInputs(value).value, tuple(leaves)


def initialize(value, plan):
    inputs = ForecastInputs(value)
    if inputs.target[0].instruments[0] != plan.instrument:
        raise ValueError("discovery instrument differs from target data")
    if plan.fold_origins[-1][-1] + 5 >= inputs.target[0].spec.sessions:
        raise ValueError("discovery folds extend beyond mature input labels")
    _verify_instances(
        inputs, plan, latest=not (campaign_root(plan.campaign_id) / "manifest.json").exists()
    )
    root = campaign_root(plan.campaign_id)
    root.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(root):
        if (root / "manifest.json").exists():
            journal, store, manifest, saved, _ = _load(plan.campaign_id)
            try:
                if saved != plan or manifest["input_ref"] != inputs.ref:
                    raise ValueError("campaign ID already binds different discovery inputs")
                return _report(journal, store, manifest, plan)
            finally:
                journal.close()
        marker = root / "initializing.json"
        identity = dict(
            campaign_id=plan.campaign_id,
            root=str(root.resolve()),
            input_ref=inputs.ref,
            plan_ref=digest(canonical(plan.model_dump(mode="json"))),
        )
        if marker.exists():
            intent = decode(marker.read_bytes())
            if intent["identity"] != identity:
                raise ValueError("unfinished discovery initialization has different identity")
        else:
            if any(p.name != ".controller.lock" for p in root.iterdir()):
                raise ValueError("discovery authority is missing or unbound")
            intent = dict(
                identity=identity,
                as_of=market_contracts.trusted_now(),
                runtime=controller.runtime_identity(),
            )
            atomic_write(marker, canonical(intent))
        if intent["runtime"] != controller.runtime_identity():
            raise ValueError("initialization runtime changed")
        store = ArtifactStore(root / "artifacts")
        store.put_json(inputs.value)
        store.put_json(plan.model_dump(mode="json"))
        accounting = None
        if any(leaf.role == "news" for leaf in plan.leaves):
            from quant_rsi.factors.news import authorize_news_comparison

            accounting = authorize_news_comparison(
                inputs,
                "discovery-" + plan.campaign_id,
                plan.fold_origins,
                comparison_owner=_news_owner(plan),
            )
        manifest = identity | dict(
            kind="FactorDiscoveryCampaign",
            schema_version=1,
            as_of=intent["as_of"],
            runtime_ref=store.put_json(intent["runtime"]),
            news_accounting=accounting,
            roster=[f"discovery-{plan.campaign_id}-{i}" for i in range(plan.max_rounds)],
            allocated_fits=plan.max_rounds * plan.fits_per_round,
            allocated_worker_seconds=plan.max_rounds * plan.fits_per_round * 180,
        )
        journal = Journal(root, digest(canonical(manifest)))
        try:
            atomic_write(root / "manifest.json", canonical(manifest))
            marker.unlink()
            return _report(journal, store, manifest, plan)
        finally:
            journal.close()


def _history(journal, store):
    result = []
    for row in journal.events():
        if row["kind"] == "discovery_round_completed":
            item = store.json(row["payload"]["result_ref"])
            index = len(result)
            allocation = event_payload(journal, f"discovery_allocation_{index}")
            if item["round"] != index or allocation != {"binding_ref": item["binding_ref"]}:
                raise ValueError("discovery history lost its allocation or ordering")
            binding = store.json(item["binding_ref"])
            parent = digest(canonical(result[-1])) if result else None
            if binding["parent_result_ref"] != parent:
                raise ValueError("discovery parent lineage differs")
            report = store.json(item["report_ref"])
            scores = report["selection"]["candidates"]
            delta = (
                scores["B0"]["pooled"]["mae_pp"] - scores["D1"]["pooled"]["mae_pp"]
                if "B0" in scores and "D1" in scores
                else None
            )
            if item["improvement_pp"] != delta:
                raise ValueError("discovery feedback differs from executed comparison")
            result.append(item)
    return result


def verify_allocation(binding, experiment_id, input_ref, fold_origins, *, verify_runtime=True):
    """Read-only child guard; no direct experiment call can allocate a new round."""
    journal, store, manifest, plan, _ = _load(binding["campaign_id"], verify_runtime=verify_runtime)
    try:
        index = binding["round"]
        if type(index) is not int or not 0 <= index < plan.max_rounds:
            raise ValueError("discovery round outside registered allowance")
        allocated = event_payload(journal, f"discovery_allocation_{index}")
        if (
            allocated is None
            or store.json(allocated["binding_ref"]) != binding
            or experiment_id != manifest["roster"][index]
            or input_ref != manifest["input_ref"]
            or [list(f) for f in fold_origins] != [list(f) for f in plan.fold_origins]
        ):
            raise ValueError("discovery child differs from immutable allocation")
    finally:
        journal.close()


def _terminal(report):
    if report["selection"] is None:
        return False
    return report["selection"]["selected_recipe"] is None or (
        len(report["attempts"]) == report["budget"]["max_attempts"]
        and report["attempts"][-1]["status"] not in {"reserved", "running"}
    )


def resume(campaign_id, *, max_fits=1):
    if type(max_fits) is not int or not 0 <= max_fits <= 50:
        raise ValueError("max_fits must be an integer from 0 to 50")
    root = campaign_root(campaign_id)
    with controller.controller_lock(root) as owner:
        journal, store, manifest, plan, inputs = _load(campaign_id)
        try:
            if event_payload(journal, "discovery_terminal") is not None:
                return _report(journal, store, manifest, plan)
            remaining = max_fits
            while True:
                history = _history(journal, store)
                index = len(history)
                if index == plan.max_rounds:
                    commit_event(
                        journal, "discovery_terminal", {"reason": "round_allowance_exhausted"}
                    )
                    break
                allocated = event_payload(journal, f"discovery_allocation_{index}")
                if allocated is None:
                    if remaining == 0:
                        break
                    graph = propose(grammar(plan.leaves), history)
                    if graph is None:
                        commit_event(journal, "discovery_terminal", {"reason": "grammar_exhausted"})
                        break
                    binding = dict(
                        campaign_id=campaign_id,
                        round=index,
                        as_of=manifest["as_of"],
                        expression=graph.model_dump(mode="json"),
                        leaves=[leaf.model_dump(mode="json") for leaf in plan.leaves],
                        parent_result_ref=store.put_json(history[-1]) if history else None,
                        policy=plan.policy,
                    )
                    allocated = {"binding_ref": store.put_json(binding)}
                    commit_event(journal, f"discovery_allocation_{index}", allocated)
                    # Binding persistence is the proposal checkpoint; retry never reproposes.
                binding = store.json(allocated["binding_ref"])
                experiment_id = manifest["roster"][index]
                try:
                    report = prepare_experiment(
                        inputs.value,
                        experiment_id,
                        fold_origins=plan.fold_origins,
                        discovery=binding,
                    )
                except InsufficientComparisonEvidence as exc:
                    child_store = ArtifactStore(experiment_root(experiment_id) / "artifacts")
                    for mask in exc.evidence.values():
                        store.put(child_store.get(mask["mask_ref"]))
                    commit_event(
                        journal,
                        "discovery_terminal",
                        dict(
                            reason="insufficient_evidence",
                            detail=str(exc),
                            sample_coverage=exc.evidence,
                        ),
                    )
                    break
                if not _terminal(report) and remaining:
                    before = report["reserved_worker_seconds"]
                    report = resume_experiment(
                        experiment_id,
                        protocol_ref=report["protocol_ref"],
                        max_fits=min(remaining, 10),
                        parent_lock_fd=owner,
                    )
                    remaining -= (report["reserved_worker_seconds"] - before) // 180
                if not _terminal(report):
                    break
                candidates = report["selection"]["candidates"]
                delta = None
                if "B0" in candidates and "D1" in candidates:
                    delta = (
                        candidates["B0"]["pooled"]["mae_pp"] - candidates["D1"]["pooled"]["mae_pp"]
                    )
                from quant_rsi.factors.discovery_graph import recipe_from_binding

                graph = recipe_from_binding(binding).graph
                outcome = dict(
                    round=index,
                    expression_ref=graph.ref,
                    binding_ref=allocated["binding_ref"],
                    experiment_id=experiment_id,
                    report_ref=store.put_json(report),
                    improvement_pp=delta,
                    novelty="composed_definition" if len(graph.nodes) > 1 else "library_selection",
                    disposition=(
                        "selected_for_research"
                        if delta is not None and delta > plan.min_improvement_pp
                        else "retain_baseline"
                    ),
                    record_status="archived",
                )
                with journal.connection:
                    journal.event(
                        None, "discovery_round_completed", {"result_ref": store.put_json(outcome)}
                    )
            return _report(journal, store, manifest, plan)
        finally:
            journal.close()


def _report(journal, store, manifest, plan, *, replay=False):
    history = _history(journal, store)
    charged, attempts = 0, 0
    for index in range(plan.max_rounds):
        if event_payload(journal, f"discovery_allocation_{index}") is None:
            continue
        root = experiment_root(manifest["roster"][index])
        if (root / "manifest.json").exists():
            child = inspect_experiment(manifest["roster"][index], replay=replay)
            charged += child["reserved_worker_seconds"]
            attempts += len(child["attempts"])
            if index < len(history) and child != store.json(history[index]["report_ref"]):
                raise ValueError("completed discovery child changed")
        elif index < len(history):
            raise ValueError("completed discovery child authority missing")
    if charged > manifest["allocated_worker_seconds"] or attempts > manifest["allocated_fits"]:
        raise ValueError("discovery allocation exceeded")
    positive = [
        h
        for h in history
        if h["improvement_pp"] is not None and h["improvement_pp"] > plan.min_improvement_pp
    ]
    selected = max(positive, key=lambda h: h["improvement_pp"]) if positive else None
    terminal = event_payload(journal, "discovery_terminal")
    if (
        terminal
        and terminal["reason"] == "round_allowance_exhausted"
        and len(history) != plan.max_rounds
    ):
        raise ValueError("terminal discovery history is incomplete")
    if terminal:
        for mask in terminal.get("sample_coverage", {}).values():
            store.get(mask["mask_ref"])
    return dict(
        kind="FactorDiscoveryReport",
        campaign_id=plan.campaign_id,
        status=(
            "insufficient_evidence"
            if terminal and terminal["reason"] == "insufficient_evidence"
            else "completed"
            if terminal
            else "ready_or_running"
        ),
        terminal=terminal,
        input_ref=manifest["input_ref"],
        exposure=plan.exposure,
        allocated_fits=manifest["allocated_fits"],
        allocated_worker_seconds=manifest["allocated_worker_seconds"],
        fits_attempted=attempts,
        reserved_worker_seconds=charged,
        proposals_created=sum(
            event_payload(journal, f"discovery_allocation_{i}") is not None
            for i in range(plan.max_rounds)
        ),
        history=history,
        selected_expression_ref=selected["expression_ref"] if selected else None,
        selection="discovered_factor" if selected else "retain_baseline",
        evidence_status="development_only",
        promotion_decision="not_requested",
        diagnostics=dict(
            independent_confirmation=None,
            pbo=None,
            deflated_sharpe=None,
            trading_cost_stress=None,
            reason="not_calibrated_or_not_a_trading_objective",
        ),
        limitations=[
            "finite_expression_grammar",
            "fixed_five_session_objective",
            "no_live_source_discovery",
            "no_learned_intermediate_predictors",
            "no_independent_confirmation_or_production_promotion",
        ],
    )


def inspect(campaign_id, *, replay=False):
    root = campaign_root(campaign_id)
    with controller.controller_lock(root):
        journal, store, manifest, plan, _ = _load(campaign_id, verify_runtime=replay)
        try:
            return _report(journal, store, manifest, plan, replay=replay)
        finally:
            journal.close()
