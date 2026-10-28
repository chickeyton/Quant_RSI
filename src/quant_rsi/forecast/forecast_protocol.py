"""Frozen paired forecast comparison using the existing artifact and worker contracts."""

from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.core.schemas import BudgetSpec, LocalDataSpec
from quant_rsi.forecast.forecast_features import build_common_mask, build_feature_view
from quant_rsi.forecast.forecast_inputs import ForecastRecipe
from quant_rsi.market.market_contracts import Hash, Identifier, timestamp, utc_text

MODEL_PARAMETERS = dict(
    objective="regression",
    verbosity=-1,
    num_threads=1,
    deterministic=True,
    force_col_wise=True,
    num_leaves=7,
    min_data_in_leaf=5,
    learning_rate=0.05,
    feature_pre_filter=False,
)
BOOST_ROUNDS = 24

GROUPS = ("B0", "B1", "B2")
NEWS_GROUPS = ("B0", "N1")


def comparison_groups(plan):
    return (
        ("B0", "D1")
        if plan.schema_version == 3
        else (NEWS_GROUPS if plan.schema_version == 2 else GROUPS)
    )


def comparison_recipe(group, discovery=None):
    if group == "D1":
        from quant_rsi.factors.discovery_graph import recipe_from_binding

        if discovery is None:
            raise ValueError("discovered recipe requires its frozen definition")
        return recipe_from_binding(discovery)
    return ForecastRecipe(
        schema_version=2 if group == "N1" else 1, kind="forecast_feature_recipe", group=group
    )


DEFAULT_FOLDS = tuple(tuple(range(start, start + 121, 5)) for start in (378, 504, 630))


class ForecastFitProposal(StrictModel):
    name: Identifier
    purpose: Literal["evaluation", "publication"]
    recipe: Literal["B0", "B1", "B2", "N1", "D1"] | None
    fold: int | None = Field(ge=0, le=2)


class ForecastComparisonPlan(StrictModel):
    schema_version: Literal[1, 2, 3]
    plan_type: Literal["forecast_comparison"]
    data: LocalDataSpec
    input_ref: Hash
    protocol_ref: Hash
    budget: BudgetSpec
    proposals: tuple[ForecastFitProposal, ...]

    @model_validator(mode="after")
    def fixed_jobs(self) -> Self:
        evaluations = self.proposals[:-1]
        groups = comparison_groups(self)
        if not evaluations or len(evaluations) % len(groups) or len(evaluations) > 3 * len(groups):
            raise ValueError("comparison requires one to three complete registered paired folds")
        expected = [
            dict(name=f"fold_{fold}_{group.lower()}", purpose="evaluation", recipe=group, fold=fold)
            for fold in range(len(evaluations) // len(groups))
            for group in groups
        ] + [dict(name="publication", purpose="publication", recipe=None, fold=None)]
        if [p.model_dump(mode="json") for p in self.proposals] != expected:
            raise ValueError("comparison job order differs from the fixed paired protocol")
        if (
            self.budget.max_attempts != len(expected)
            or self.budget.worker_timeout_seconds != 180
            or self.budget.max_worker_seconds != 180 * len(expected)
        ):
            raise ValueError("each comparison fit requires its own bounded reservation")
        return self


class InsufficientComparisonEvidence(ValueError):
    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.evidence = evidence


def freeze_protocol(
    inputs,
    store,
    *,
    publication_asof,
    fold_origins=DEFAULT_FOLDS,
    news_factor=False,
    news_accounting=None,
    discovery=None,
):
    """Freeze all masks and feature views before the first fit; perform no fitting."""
    data, _ = inputs.target
    if news_factor and discovery is not None:
        raise ValueError("news-only and discovery protocols are distinct")
    groups = ("B0", "D1") if discovery is not None else (NEWS_GROUPS if news_factor else GROUPS)
    recipes = [comparison_recipe(group, discovery) for group in groups]
    if not 1 <= len(fold_origins) <= 3:
        raise ValueError("one to three folds are required")
    last_close = data.contract.calendar.sessions[data.spec.sessions - 1].close_at
    if timestamp(publication_asof) < last_close:
        raise ValueError("publication boundary precedes the last completed session")
    if len(data.contract.calendar.sessions) < data.spec.sessions + 5:
        raise ValueError("publication requires five future registered sessions")
    cache, folds, previous_end = {}, [], -1

    def insufficient(reason, **masks):
        if not news_factor and discovery is None:
            raise ValueError(reason)
        for view in cache.values():
            store.put_json(view)
        evidence = {}
        for purpose, item in masks.items():
            mask = item["mask"]
            evidence[purpose] = dict(
                mask_ref=store.put_json(mask),
                requested=len(mask["rows"]),
                eligible=len(mask["eligible_indices"]),
                exclusions=[
                    dict(origin_session=row["origin_session"], reasons=row["reasons"])
                    for row in mask["rows"]
                    if not row["eligible"]
                ],
            )
        raise InsufficientComparisonEvidence(reason, evidence)

    for number, values in enumerate(fold_origins):
        origins = list(values)
        if (
            not origins
            or origins != list(range(origins[0], origins[-1] + 1, 5))
            or origins[0] < 77
            or origins[0] <= previous_end
            or origins[-1] + 5 >= data.spec.sessions
        ):
            raise ValueError("invalid disjoint step-five evaluation grid")
        previous_end = origins[-1] + 5
        boundary = data.contract.calendar.sessions[origins[0]].close_at
        training = build_common_mask(
            inputs,
            recipes,
            range(60, origins[0] - 4),
            boundary,
            purpose="training",
            _feature_cache=cache,
        )
        evaluation = build_common_mask(
            inputs,
            recipes,
            origins,
            publication_asof,
            purpose="evaluation",
            _feature_cache=cache,
        )
        if len(training["mask"]["eligible_indices"]) < 12:
            insufficient(
                "insufficient common mature training rows", training=training, evaluation=evaluation
            )
        if not evaluation["mask"]["eligible_indices"]:
            insufficient(
                "fold has no eligible common evaluation origins",
                training=training,
                evaluation=evaluation,
            )
        folds.append(
            dict(
                fold=number,
                training_boundary=utc_text(boundary),
                training_mask_ref=store.put_json(training["mask"]),
                evaluation_mask_ref=store.put_json(evaluation["mask"]),
                training_rows=len(training["mask"]["eligible_indices"]),
                evaluation_rows=len(evaluation["mask"]["eligible_indices"]),
                evaluation_origins=origins,
            )
        )
    publication_mask = build_common_mask(
        inputs,
        recipes,
        range(60, data.spec.sessions - 5),
        publication_asof,
        purpose="training",
        label_basis="observed_receipts",
        _feature_cache=cache,
    )
    if len(publication_mask["mask"]["eligible_indices"]) < 12:
        insufficient(
            "insufficient mature publication training rows", publication_training=publication_mask
        )
    for view in cache.values():
        store.put_json(view)
    publication_views = {
        recipe.group: store.put_json(
            build_feature_view(
                inputs, recipe, data.dates[-1], publication_asof, basis="observed_receipts"
            )
        )
        for recipe in recipes
    }
    protocol = dict(
        kind="FrozenForecastComparison",
        schema_version=3 if discovery is not None else (2 if news_factor else 1),
        input_ref=inputs.ref,
        data_specification=data.spec.model_dump(mode="json"),
        target_instrument=data.instruments[0],
        recipes={r.group: r.definition for r in recipes},
        recipe_refs={r.group: r.ref for r in recipes},
        folds=folds,
        publication=dict(
            as_of=utc_text(publication_asof),
            origin_session=data.dates[-1],
            training_mask_ref=store.put_json(publication_mask["mask"]),
            training_rows=len(publication_mask["mask"]["eligible_indices"]),
            feature_view_refs=publication_views,
        ),
        selection=dict(
            metric="pooled_mae_pp",
            tie_order=list(groups),
            requires_all_folds=True,
            independent_confirmation=False,
        ),
        model_parameters=MODEL_PARAMETERS | {"seed": data.spec.seed},
        boost_rounds=BOOST_ROUNDS,
        model_inputs=["target_prices", "industry_prices", "market_prices"],
        sidecar_inputs_used_in_training=[],
        sidecar_snapshot_ref=(
            digest(canonical(inputs.signals.model_dump(mode="json"))) if inputs.signals else None
        ),
        evaluation_knowledge_basis={
            "target": data.contract.knowledge_basis,
            **{
                role: source[0].contract.knowledge_basis for role, source in inputs.contexts.items()
            },
        },
        qualifications=[
            "development_comparison_not_independent_statistical_confirmation",
            "price_and_context_features_only_news_and_fundamentals_not_model_inputs",
        ],
    )
    if news_factor:
        if news_accounting is None:
            raise ValueError("news comparison requires campaign accounting")
        protocol.update(
            news_accounting=news_accounting,
            model_inputs=["target_prices", "typed_news_factor"],
            sidecar_inputs_used_in_training=["news"],
            qualifications=[
                "development_comparison_not_independent_statistical_confirmation",
                "retained_corpus_coverage_not_complete_historical_news",
                "news_features_require_observed_evidence_and_extraction_receipts",
            ],
        )
    if discovery is not None:
        protocol.update(
            discovery=discovery,
            model_inputs=["target_prices", "discovered_factor"],
            sidecar_inputs_used_in_training=["news"]
            if any(x["role"] == "news" for x in discovery["leaves"])
            else [],
            qualifications=[
                "development_only_bounded_factor_discovery",
                "common_registered_leaf_universe",
                "no_independent_confirmation",
            ],
        )
    protocol_ref = store.put_json(protocol)
    proposals = [
        dict(name=f"fold_{fold}_{group.lower()}", purpose="evaluation", recipe=group, fold=fold)
        for fold in range(len(folds))
        for group in groups
    ] + [dict(name="publication", purpose="publication", recipe=None, fold=None)]
    return ForecastComparisonPlan.model_validate_json(
        canonical(
            dict(
                schema_version=3 if discovery is not None else (2 if news_factor else 1),
                plan_type="forecast_comparison",
                data=data.spec.model_dump(mode="json"),
                input_ref=inputs.ref,
                protocol_ref=protocol_ref,
                budget=dict(
                    max_attempts=len(proposals),
                    worker_timeout_seconds=180,
                    max_worker_seconds=180 * len(proposals),
                ),
                proposals=proposals,
            )
        )
    )
