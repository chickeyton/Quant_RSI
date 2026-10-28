"""Pure ordered features and shared input-eligibility masks for bounded forecast recipes."""

import math

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.forecast.forecast_kernel import (
    action_assessment,
    forecast_feature_view,
    mature_forecast_label,
)
from quant_rsi.market.asof import session_index
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.market.market_data import visible_revisions


def _history(source, closes, as_of, basis):
    data, objective = source
    instrument = data.instruments[0]
    visible = visible_revisions(data.snapshot, data.contract, as_of, basis=basis)
    calendar = {session.close_at: session for session in data.contract.calendar.sessions}
    reasons, refs, prices = [], set(), []
    actions, action_evidence = action_assessment(
        visible,
        instrument,
        closes[0],
        closes[-1],
        objective,
    )
    reasons.extend(actions)
    refs.update(item["notice_ref"] for item in action_evidence)
    for close in closes:
        session = calendar.get(close)
        if session is None:
            reasons.append("context_calendar_close_missing")
            continue
        memberships = [
            item
            for item in visible
            if item["event"]["kind"] == "UNIVERSE_STATUS"
            and item["event"]["instrument"] == instrument
            and timestamp(item["event"]["event_at"]) <= close
        ]
        latest_at = max(
            (timestamp(item["event"]["event_at"]) for item in memberships),
            default=None,
        )
        latest = [item for item in memberships if timestamp(item["event"]["event_at"]) == latest_at]
        refs.update(item["payload_ref"] for item in latest)
        if not latest or {item["event"]["status"] for item in latest} != {"active"}:
            reasons.append("history_membership_unknown_or_inactive")
        quotes = [
            item
            for item in visible
            if item["event"]["kind"] == "CLOSE_QUOTE"
            and item["event"]["instrument"] == instrument
            and item["event"]["session_id"] == session.session_id
        ]
        refs.update(item["payload_ref"] for item in quotes)
        if len(quotes) != 1 or quotes[0]["event"]["quality"] != "valid":
            reasons.append("history_quote_missing_invalid_or_conflicting")
            continue
        value = float(quotes[0]["event"]["price"])
        if not math.isfinite(value) or value <= 0:
            reasons.append("history_quote_nonfinite_or_nonpositive")
            continue
        prices.append(value)
    return prices if not reasons else None, sorted(set(reasons)), sorted(refs)


def _return(prices, window):
    return prices[-1] / prices[-window - 1] - 1


def _volatility(prices, window):
    recent = prices[-window - 1 :]
    daily = [b / a - 1 for a, b in zip(recent, recent[1:], strict=False)]
    mean = math.fsum(daily) / window
    return math.sqrt(math.fsum((value - mean) ** 2 for value in daily) / window)


def build_feature_view(inputs, recipe, origin, as_of, *, basis=None):
    target, objective = inputs.target
    index = session_index(target.contract, origin)
    legacy = forecast_feature_view(
        target.snapshot,
        target.contract,
        origin,
        as_of,
        FactorSpec(op="momentum", window=5),
        objective,
        basis=basis,
    )
    row = legacy["rows"][0]
    reasons = list(row["reasons"])
    refs = set(row["source_refs"])
    values = list(row["features"]) if row["features"] is not None else None
    if recipe.group in {"B1", "B2"}:
        if index < 60:
            reasons.append("insufficient_60_session_history")
        else:
            closes = [
                session.close_at
                for session in target.contract.calendar.sessions[index - 60 : index + 1]
            ]
            history, problems, history_refs = _history(inputs.target, closes, as_of, basis)
            refs.update(history_refs)
            reasons.extend(problems)
            if history is not None and values is not None:
                values.extend(
                    [
                        _return(history, 20),
                        _return(history, 60),
                        _volatility(history, 20),
                        _volatility(history, 60),
                    ]
                )
            if recipe.group == "B2":
                for role in ("industry", "market"):
                    source = inputs.contexts.get(role)
                    if source is None:
                        reasons.append(role + "_input_missing")
                        continue
                    context, problems, context_refs = _history(source, closes[-21:], as_of, basis)
                    refs.update(context_refs)
                    reasons.extend(role + ":" + problem for problem in problems)
                    if context is not None and history is not None and values is not None:
                        values.extend([_return(context, window) for window in (5, 20)])
                        values.extend(
                            [
                                _return(history, window) - _return(context, window)
                                for window in (5, 20)
                            ]
                        )
    if recipe.group == "N1":
        from quant_rsi.factors.news import news_scalar

        news = news_scalar(inputs, origin, as_of)
        reasons.extend(news["reasons"])
        refs.update(news["source_refs"])
        if values is not None and not news["reasons"]:
            values.append(news["value"])
    if not reasons and (
        values is None or len(values) != len(recipe.names) or not all(map(math.isfinite, values))
    ):
        reasons.append("invalid_feature_dimension_or_nonfinite_value")
    return dict(
        kind="ForecastRecipeFeatureView",
        schema_version=1,
        input_ref=inputs.ref,
        recipe_ref=recipe.ref,
        feature_names=list(recipe.names),
        feature_units=recipe.definition["feature_units"],
        target_instrument=target.instruments[0],
        origin_session=origin,
        decision_at=legacy["decision_at"],
        as_of=utc_text(as_of),
        knowledge_bases={
            "target": basis or target.contract.knowledge_basis,
            **{
                role: basis or source[0].contract.knowledge_basis
                for role, source in inputs.contexts.items()
            },
        },
        status="eligible" if not reasons else "ineligible",
        reasons=sorted(set(reasons)),
        values=values if not reasons else None,
        source_refs=sorted(refs),
        legacy_feature_view_ref=digest(canonical(legacy)),
        qualifications=legacy["qualifications"],
    )


def build_common_mask(
    inputs,
    recipes,
    origin_indices,
    label_asof,
    *,
    purpose,
    basis=None,
    label_basis=None,
    _feature_cache=None,
):
    """Freeze one shared mask before fitting, with explicit exclusions and no score selection."""
    if purpose not in {"training", "evaluation"}:
        raise ValueError("mask purpose must be training or evaluation")
    if not recipes or len({recipe.ref for recipe in recipes}) != len(recipes):
        raise ValueError("a mask requires distinct registered recipes")
    origins = list(origin_indices)
    if not origins or origins != sorted(set(origins)):
        raise ValueError("mask origins must be ordered, unique, and nonempty")
    data, objective = inputs.target
    if any(type(index) is not int or not 0 <= index < len(data.dates) for index in origins):
        raise ValueError("mask origin is outside the target data")
    if any(data.decision_at(index) > timestamp(label_asof) for index in origins):
        raise ValueError("mask contains an origin beyond its label knowledge boundary")
    rows = []
    for index in origins:
        session = data.dates[index]
        views = []
        for recipe in recipes:
            key = (inputs.ref, recipe.ref, session, basis)
            view = _feature_cache.get(key) if _feature_cache is not None else None
            if view is None:
                view = build_feature_view(
                    inputs, recipe, session, data.decision_at(index), basis=basis
                )
                if _feature_cache is not None:
                    _feature_cache[key] = view
            views.append(view)
        label = mature_forecast_label(
            data.snapshot,
            data.contract,
            session,
            data.instruments[0],
            label_asof,
            objective,
            basis=label_basis or basis,
        )
        reasons = sorted(
            {reason for view in views for reason in view["reasons"]}
            | (set() if label["status"] == "mature" else {"label_" + label["status"]})
        )
        rows.append(
            dict(
                origin_index=index,
                origin_session=session,
                instrument=data.instruments[0],
                eligible=not reasons,
                reasons=reasons,
                feature_view_refs={
                    recipe.ref: digest(canonical(view))
                    for recipe, view in zip(recipes, views, strict=True)
                },
                label=label,
            )
        )
    value = dict(
        kind="ForecastCommonInputMask",
        schema_version=1,
        input_ref=inputs.ref,
        purpose=purpose,
        recipe_refs=[recipe.ref for recipe in recipes],
        label_asof=utc_text(label_asof),
        label_knowledge_basis=label_basis or basis or data.contract.knowledge_basis,
        origin_indices=origins,
        eligible_indices=[row["origin_index"] for row in rows if row["eligible"]],
        rows=rows,
        selection_uses_prediction_errors=False,
    )
    return dict(mask_ref=digest(canonical(value)), mask=value)


def ordered_feature_values(view, recipe, *, input_ref):
    """Model consumers must verify the entire ordered preprocessing interface."""
    if (
        view["kind"] != "ForecastRecipeFeatureView"
        or view["schema_version"] != 1
        or view["input_ref"] != input_ref
        or view["recipe_ref"] != recipe.ref
        or view["feature_names"] != list(recipe.names)
        or view["feature_units"] != recipe.definition["feature_units"]
    ):
        raise ValueError("feature input/recipe/order/unit binding mismatch")
    values = view["values"]
    if (
        view["status"] != "eligible"
        or values is None
        or len(values) != len(recipe.names)
        or not all(type(value) in (int, float) and math.isfinite(value) for value in values)
    ):
        raise ValueError("model input features are ineligible or have invalid dimensions/values")
    return list(values)
