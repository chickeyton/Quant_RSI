"""Named market research profile and frozen evidence closure for recursive grants."""

from pathlib import Path

from quant_rsi.artifacts import canonical, decode, digest
from quant_rsi.forecast_contracts import ForecastDevelopmentPlan
from quant_rsi.forecast_recursive_market import (
    ForecastRecursiveMarketEvidence,
    project_market_target,
)
from quant_rsi.researcher import ResearchPlan
from quant_rsi.task_definition import ForecastRecursiveProfile


def market_research_plan(projection):
    """The initial market profile is fixed before observations or worker execution."""
    if projection["quoted_sessions"] != 756 or projection["training_boundary"] != 721:
        raise ValueError(
            "initial recursive market profile requires exactly 756 historical sessions"
        )
    if projection["development_origins"] != [721, 726, 731, 736, 741, 746]:
        raise ValueError("recursive market profile requires its exact six development origins")
    inputs = projection["inputs"]
    if set(inputs["contexts"]) != {"industry", "market"}:
        raise ValueError("recursive market preparation retains both explicit context roles")
    development = ForecastDevelopmentPlan.model_validate_json(
        canonical(
            dict(
                schema_version=1,
                plan_type="forecast_development",
                training_mode="M0",
                data=projection["target_data"]["specification"],
                objective=inputs["target"]["objective"],
                budget=dict(max_attempts=8, worker_timeout_seconds=180, max_worker_seconds=1440),
                primary_metric="negative_mae_pp",
                evaluation_stride=5,
                min_development_sessions=6,
                baseline="momentum_2",
                proposals=[
                    dict(
                        name="momentum_2",
                        factor=dict(op="momentum", window=2),
                        hypothesis="Frozen initial target-only five-session forecast baseline.",
                    )
                ],
            )
        )
    )
    return ResearchPlan.model_validate_json(
        canonical(
            dict(
                schema_version=3,
                plan_type="development_research",
                development=development.model_dump(mode="json"),
                policy=dict(
                    algorithm="neighbor_search_v1",
                    neighbor_order="window_first",
                    window_min=2,
                    window_max=3,
                    min_improvement=0.0,
                    max_no_progress=3,
                ),
                neighbor_domain=dict(
                    kind="RegisteredNeighborDomain",
                    schema_version=1,
                    operators=["momentum", "mean_return"],
                    window_min=2,
                    window_max=5,
                    initial_window_max=3,
                    cap_step=1,
                    max_cap_advances=2,
                ),
                forecast_profile=ForecastRecursiveProfile.from_development(development).model_dump(
                    mode="json"
                ),
            )
        )
    )


def read_prepared_projection(preparation):
    """Verify the original preparation/registry before projecting its preserved bytes."""
    from quant_rsi.forecast_context_data import verify_preparation

    prepared = verify_preparation(Path(preparation))
    projection = project_market_target(
        prepared["inputs"], Path(prepared["receipt_path"]).read_bytes()
    )
    market_research_plan(projection)
    return prepared, projection


def store_market_projection(store, projection):
    """Copy the complete self-contained evidence, including unused contexts and signals."""
    for key in ("inputs", "target_data", "market_evidence"):
        if store.put_json(projection[key]) != projection[key + "_ref"]:
            raise ValueError("market projection content reference differs")
    if store.put(projection["preparation_receipt"]) != projection["preparation_receipt_ref"]:
        raise ValueError("original market preparation receipt bytes differ")


def verify_market_registration(grant, store, *, data_ref=None):
    """Verify frozen closure only; no mutable preparation path or registry is consulted."""
    if grant.get("exposure_mode") != "historical_market_exploration":
        return None
    evidence_ref = grant["market_evidence_ref"]
    evidence = ForecastRecursiveMarketEvidence.model_validate_json(store.get(evidence_ref))
    projection = project_market_target(
        store.json(evidence.inputs_ref), store.get(evidence.preparation_receipt_ref)
    )
    receipt = decode(projection["preparation_receipt"])
    if (
        receipt["workspace"] != grant["workspace"]
        or receipt["registry_path"] != str(Path(grant["workspace"]) / ".quant_rsi" / "market")
        or projection["market_evidence_ref"] != evidence_ref
        or store.json(evidence.target_data_ref) != projection["target_data"]
        or grant["research_plan"] != market_research_plan(projection).model_dump(mode="json")
        or (data_ref is not None and data_ref != evidence.target_data_ref)
    ):
        raise ValueError("market grant, original evidence, profile or actual source data differs")
    return projection


def verify_actual_export(projection):
    """Use the ordinary source exporter; a serialized plan alone is not parity evidence."""
    from quant_rsi.local_research import export_local_data
    from quant_rsi.schemas import LocalDataSpec

    spec = LocalDataSpec.model_validate_json(canonical(projection["target_data"]["specification"]))
    exported = export_local_data(spec)
    if digest(canonical(exported)) != projection["target_data_ref"]:
        raise ValueError(
            "actual market source export differs from the registered target projection"
        )
    return exported


def prepare_market_campaign(profile, output, *, campaign_id, source_campaign_id):
    """Collect/prepare an explicit profile and write a reviewable plan; never register or fit."""
    import re

    from quant_rsi.artifacts import atomic_write, decode
    from quant_rsi.forecast_campaign import _id
    from quant_rsi.forecast_context_collection import collect_context
    from quant_rsi.forecast_context_data import prepare_context_data
    from quant_rsi.forecast_frontier_policy import POLICY, proposer_definition
    from quant_rsi.forecast_workflow_config import ForecastWorkflowConfig

    config = ForecastWorkflowConfig.model_validate_json(canonical(profile))
    if config.history_sessions != 756:
        raise ValueError(
            "initial recursive market profile requires exactly 756 historical sessions"
        )
    _id(campaign_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", source_campaign_id):
        raise ValueError("invalid recursive source campaign ID")
    output = Path(output)
    if output.is_symlink():
        raise ValueError("recursive preparation output cannot be a symlink")
    output = output.resolve()
    intent = dict(
        kind="RecursiveMarketPreparation",
        schema_version=1,
        profile=config.model_dump(mode="json"),
        campaign_id=campaign_id,
        source_campaign_id=source_campaign_id,
        output=str(output),
        proposer_policy=POLICY,
        proposer_definition=proposer_definition(),
    )
    intent_path = output / "recursive_preparation.json"
    if intent_path.exists():
        if decode(intent_path.read_bytes()) != intent:
            raise ValueError("recursive preparation already binds another configuration or policy")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("recursive preparation requires a fresh output directory")
        atomic_write(intent_path, canonical(intent))
    audit, preparation = output / "collection", output / "preparation"
    started = output / "collection_started.json"
    if started.exists() and not audit.is_dir():
        raise ValueError("original recursive collection is missing; no replacement collection")
    audit.mkdir(exist_ok=True)
    atomic_write(started, canonical({"intent_ref": digest(canonical(intent))}))
    collected = collect_context(audit, roles=config.roles)
    if Path(collected).resolve() != audit:
        raise ValueError("recursive collector returned another audit directory")
    prepared = prepare_context_data(
        audit, preparation, history_sessions=756, roles=config.roles, resume=True
    )
    projection = project_market_target(
        prepared["inputs"], Path(prepared["receipt_path"]).read_bytes()
    )
    research_plan = market_research_plan(projection)
    workspace = Path(prepared["meta"]["workspace"])
    campaign_plan = dict(
        kind="ForecastRecursiveCampaign",
        schema_version=4,
        campaign_id=campaign_id,
        source_directory=str(workspace / "source"),
        source_campaign_id=source_campaign_id,
        research_plan=research_plan.model_dump(mode="json"),
        source_allowance=dict(max_attempts=4, max_worker_seconds=720),
        budget=dict(max_attempts=36, max_worker_seconds=6480),
        max_proposal_stages=2,
        max_generations=2,
        proposer_policy=POLICY,
        proposer_definition=proposer_definition(),
        exposure_mode="historical_market_exploration",
        market_evidence_ref=projection["market_evidence_ref"],
    )
    from quant_rsi.forecast_recursive_campaign import _normalize

    campaign_plan = _normalize(campaign_plan)
    for name, value in (
        ("research_plan.json", research_plan.model_dump(mode="json")),
        ("campaign_plan.json", campaign_plan),
        ("market_evidence.json", projection["market_evidence"]),
    ):
        path = output / name
        body = canonical(value)
        if path.exists() and path.read_bytes() != body:
            raise ValueError("saved recursive preparation plan differs from its original inputs")
        if not path.exists():
            atomic_write(path, body)
    return dict(
        kind="RecursiveMarketPreparationReport",
        workspace=str(workspace),
        preparation=str(preparation),
        research_plan=str(output / "research_plan.json"),
        campaign_plan=str(output / "campaign_plan.json"),
        market_evidence_ref=projection["market_evidence_ref"],
        exposure_mode="historical_market_exploration",
        target=config.target,
        retained_unused_contexts={"industry": config.industry, "market": config.market},
        historical_sessions=756,
        training_boundary=721,
        development_origins=projection["development_origins"],
        planned_allowance=campaign_plan["budget"],
        grant_registered=False,
        model_training_performed=False,
    )
