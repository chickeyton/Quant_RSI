"""One fixed LightGBM fit per charged comparison job, plus deterministic replay."""

import math

import numpy as np

from quant_rsi import market_contracts
from quant_rsi.artifacts import digest
from quant_rsi.forecast_features import ordered_feature_values
from quant_rsi.forecast_inputs import ForecastRecipe
from quant_rsi.forecast_kernel import forecast_errors
from quant_rsi.forecast_protocol import BOOST_ROUNDS, GROUPS, MODEL_PARAMETERS
from quant_rsi.market_contracts import timestamp, utc_text


def job_inputs(inputs, plan, proposal, store, manifest):
    protocol = store.json(plan.protocol_ref)
    if (
        inputs.ref != plan.input_ref
        or protocol["input_ref"] != inputs.ref
        or inputs.target[0].spec != plan.data
        or protocol["data_specification"] != plan.data.model_dump(mode="json")
        or manifest["data_ref"] != inputs.ref
        or protocol["model_parameters"] != MODEL_PARAMETERS | {"seed": plan.data.seed}
        or protocol["boost_rounds"] != BOOST_ROUNDS
    ):
        raise ValueError("comparison input/plan/protocol binding mismatch")
    if proposal not in plan.proposals:
        raise ValueError("fit job is outside the frozen comparison")
    if proposal.purpose == "evaluation":
        group = proposal.recipe
        section = protocol["folds"][proposal.fold]
    else:
        if "selection_ref" not in manifest:
            raise ValueError("publication requires committed evaluation selection")
        selection = store.json(manifest["selection_ref"])
        if (
            selection["protocol_ref"] != plan.protocol_ref
            or selection["input_ref"] != inputs.ref
            or selection["selected_recipe"] not in GROUPS
        ):
            raise ValueError("publication selection does not bind this experiment")
        group = selection["selected_recipe"]
        section = protocol["publication"]
    recipe = ForecastRecipe(schema_version=1, kind="forecast_feature_recipe", group=group)
    if (
        protocol["recipe_refs"][group] != recipe.ref
        or protocol["recipes"][group] != recipe.definition
    ):
        raise ValueError("comparison preprocessing recipe mismatch")
    mask = store.json(section["training_mask_ref"])
    x, y, rows = matrix(inputs, recipe, mask, store, purpose="training")
    if len(rows) < 12 or len(rows) != section["training_rows"]:
        raise ValueError("training mask dimension differs from frozen protocol")
    if proposal.purpose == "evaluation":
        evaluation = store.json(section["evaluation_mask_ref"])
        predict_x, _, predict_rows = matrix(inputs, recipe, evaluation, store, purpose="evaluation")
        if len(predict_rows) != section["evaluation_rows"]:
            raise ValueError("evaluation mask dimension differs from frozen protocol")
        boundary = section["training_boundary"]
    else:
        view = store.json(section["feature_view_refs"][group])
        predict_x = (
            np.asarray([ordered_feature_values(view, recipe, input_ref=inputs.ref)])
            if view["status"] == "eligible"
            else np.empty((0, len(recipe.names)))
        )
        predict_rows = [
            dict(
                origin_session=section["origin_session"],
                feature_view_ref=section["feature_view_refs"][group],
                status=view["status"],
                reasons=view["reasons"],
            )
        ]
        boundary = section["as_of"]
    if mask["label_asof"] != boundary:
        raise ValueError("training mask differs from the frozen fit boundary")
    if any(
        timestamp(row["label"]["interval"][1]) > timestamp(boundary)
        or timestamp(row["label"]["available_at"]) > timestamp(boundary)
        for row in rows
    ):
        raise ValueError("fit includes an immature or unavailable training label")
    metadata = dict(
        kind="ForecastComparisonFit",
        schema_version=1,
        purpose=proposal.purpose,
        job_name=proposal.name,
        input_ref=inputs.ref,
        protocol_ref=plan.protocol_ref,
        recipe_ref=recipe.ref,
        recipe_group=group,
        feature_names=list(recipe.names),
        feature_units=recipe.definition["feature_units"],
        target_instrument=inputs.target[0].instruments[0],
        training_mask_ref=section["training_mask_ref"],
        training_rows=len(rows),
        training_information_cutoff=boundary,
        last_training_label_available=utc_text(
            max(timestamp(r["label"]["available_at"]) for r in rows)
        ),
        last_training_target_close=utc_text(
            max(timestamp(r["label"]["interval"][1]) for r in rows)
        ),
        model_parameters=MODEL_PARAMETERS | {"seed": plan.data.seed},
        boost_rounds=BOOST_ROUNDS,
        selection_ref=manifest.get("selection_ref") if proposal.purpose == "publication" else None,
        sidecar_inputs_used_in_training=[],
    )
    return x, y, predict_x, predict_rows, metadata


def matrix(inputs, recipe, mask, store, *, purpose):
    if (
        mask["input_ref"] != inputs.ref
        or mask["purpose"] != purpose
        or recipe.ref not in mask["recipe_refs"]
        or mask["selection_uses_prediction_errors"] is not False
    ):
        raise ValueError("invalid shared mask binding")
    rows = [row for row in mask["rows"] if row["eligible"]]
    if [row["origin_index"] for row in rows] != mask["eligible_indices"]:
        raise ValueError("shared mask eligibility order differs")
    x, y = [], []
    for row in rows:
        view = store.json(row["feature_view_refs"][recipe.ref])
        if (
            view["origin_session"] != row["origin_session"]
            or row["instrument"] != inputs.target[0].instruments[0]
            or row["label"]["status"] != "mature"
            or not math.isfinite(row["label"]["value"])
        ):
            raise ValueError("shared mask feature/label binding mismatch")
        x.append(ordered_feature_values(view, recipe, input_ref=inputs.ref))
        y.append(row["label"]["value"])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), rows


def predict(model_text, x, rows, purpose, feature_names):
    import lightgbm as lgb

    model = lgb.Booster(model_str=model_text)
    if model.num_feature() != len(feature_names) or model.feature_name() != feature_names:
        raise ValueError("serialized model preprocessing columns differ")
    if purpose == "publication" and not len(x):
        return [rows[0] | {"score": None}]
    values = model.predict(x, num_threads=1)
    if len(values) != len(rows) or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("model did not produce a finite prediction for every frozen row")
    return [
        dict(origin_session=row["origin_session"], score=float(value))
        for row, value in zip(rows, values, strict=True)
    ]


def fit_comparison(inputs, plan, proposal, store, manifest):
    import lightgbm as lgb

    x, y, predict_x, rows, metadata = job_inputs(inputs, plan, proposal, store, manifest)
    started = market_contracts.trusted_now()
    if timestamp(started) < timestamp(metadata["training_information_cutoff"]):
        raise ValueError("fit boundary is in the future")
    model = lgb.train(
        metadata["model_parameters"],
        lgb.Dataset(x, label=y, feature_name=metadata["feature_names"]),
        num_boost_round=BOOST_ROUNDS,
    )
    completed = market_contracts.trusted_now()
    if timestamp(completed) < timestamp(started):
        raise ValueError("fit clock moved backwards")
    model_text = model.model_to_string()
    return dict(
        **metadata,
        model_text=model_text,
        model_digest=digest(model_text.encode()),
        actual_fit_started_at=started,
        actual_fit_completed_at=completed,
        model_effective_at=completed,
        predictions=predict(
            model_text, predict_x, rows, proposal.purpose, metadata["feature_names"]
        ),
    )


def verify_comparison_result(inputs, plan, proposal, store, manifest, result, model_text):
    _, y, predict_x, rows, metadata = job_inputs(inputs, plan, proposal, store, manifest)
    if any(result.get(key) != value for key, value in metadata.items()):
        raise ValueError("worker fit metadata differs from the frozen inputs")
    if (
        timestamp(metadata["training_information_cutoff"])
        > timestamp(result["actual_fit_started_at"])
        or timestamp(result["actual_fit_started_at"]) > timestamp(result["actual_fit_completed_at"])
        or result["model_effective_at"] != result["actual_fit_completed_at"]
    ):
        raise ValueError("worker actual fit timing is inconsistent")
    expected = predict(model_text, predict_x, rows, proposal.purpose, metadata["feature_names"])
    if expected != result["predictions"]:
        raise ValueError("predictions differ from the actual serialized model")
    if proposal.purpose == "publication":
        return dict(
            kind="PublicationFitReport",
            eligible=True,
            metrics=None,
            evaluation_scores_inherited=False,
            selection_ref=metadata["selection_ref"],
            predictions=expected,
            training_rows=len(y),
        )
    actual = [row["label"]["value"] for row in rows]
    predictions = [row["score"] for row in expected]
    mean = float(np.mean(y))
    return dict(
        kind="ForecastFoldReport",
        eligible=True,
        fold=proposal.fold,
        recipe=proposal.recipe,
        metrics=forecast_errors(predictions, actual),
        rows=[
            dict(origin_session=row["origin_session"], predicted=pred, actual=truth)
            for row, pred, truth in zip(rows, predictions, actual, strict=True)
        ],
        naive_baselines=dict(
            unchanged_price=forecast_errors([0.0] * len(actual), actual),
            training_mature_label_mean=forecast_errors([mean] * len(actual), actual),
        ),
        training_mean_log_return=mean,
        training_mask_ref=metadata["training_mask_ref"],
        protocol_ref=plan.protocol_ref,
        input_ref=inputs.ref,
    )


def select_recipe(plan, reports):
    """Use all frozen folds, pooled MAE, and the predeclared B0/B1/B2 tie order."""
    protocol_folds = len(plan.proposals[:-1]) // 3
    by_group, evidence = {group: [] for group in GROUPS}, {}
    for name, (report_ref, report) in reports.items():
        expected = next((p for p in plan.proposals[:-1] if p.name == name), None)
        if (
            expected is None
            or report["recipe"] != expected.recipe
            or report["fold"] != expected.fold
        ):
            raise ValueError("selection report is outside the frozen grid")
        if report["protocol_ref"] != plan.protocol_ref or report["input_ref"] != plan.input_ref:
            raise ValueError("selection report belongs to different inputs")
        if report["eligible"]:
            by_group[expected.recipe].append(report)
            evidence[name] = report_ref
    candidates = {}
    for group in GROUPS:
        folds = sorted(by_group[group], key=lambda r: r["fold"])
        if [r["fold"] for r in folds] != list(range(protocol_folds)):
            continue
        rows = [row for report in folds for row in report["rows"]]
        candidates[group] = dict(
            pooled=forecast_errors([r["predicted"] for r in rows], [r["actual"] for r in rows]),
            per_fold=[dict(fold=r["fold"], metrics=r["metrics"]) for r in folds],
        )
    selected = (
        min(
            candidates,
            key=lambda group: (candidates[group]["pooled"]["mae_pp"], GROUPS.index(group)),
        )
        if candidates
        else None
    )
    return dict(
        kind="ForecastRecipeSelection",
        schema_version=1,
        input_ref=plan.input_ref,
        protocol_ref=plan.protocol_ref,
        selected_recipe=selected,
        candidates=candidates,
        report_refs=evidence,
        selection_metric="pooled_mae_pp",
        tie_order=list(GROUPS),
        independent_confirmation=False,
    )
