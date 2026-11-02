"""Finite development iterations using the existing stage callbacks and selection rule."""

from copy import deepcopy

from quant_rsi.research.demo_proposals import extend_library


def validate_limits(rounds, fit_budget, patience):
    for name, value in (("rounds", rounds), ("fit_budget", fit_budget)):
        if type(value) is not int or not 2 <= value <= 10:
            raise ValueError(f"{name} must be an integer from 2 through 10")
    if patience is not None and (type(patience) is not int or not 1 <= patience <= 10):
        raise ValueError("patience must be an integer from 1 through 10")


def iterate(
    context,
    library,
    *,
    hypothesize,
    predict,
    observe,
    resolve_factors,
    call,
    rounds,
    fit_budget,
    patience,
    select,
):
    validate_limits(rounds, fit_budget, patience)
    attempts = []
    incumbent = None
    no_progress = 0
    feedback = None
    limit = min(rounds, fit_budget)
    stop_reason = "fit_budget_exhausted" if fit_budget < rounds else "round_limit"
    for number in range(1, limit + 1):
        prefix = {1: "initial", 2: "revised"}.get(number, f"round_{number}")
        prediction_stage = {1: "first_prediction", 2: "second_prediction"}.get(
            number, f"round_{number}_prediction"
        )
        if incumbent is not None:

            def collect_feedback(incumbent=incumbent):
                prediction = incumbent["prediction"]
                observed = observe(deepcopy(context), deepcopy(prediction))
                value = {
                    **observed,
                    "predicted_price": prediction["predicted_price"],
                    "error": round(prediction["predicted_price"] - observed["observed_price"], 4),
                    "error_definition": "predicted_price_minus_observed_price",
                }
                if select is not None:
                    value["incumbent_round"] = incumbent["round"]
                    value["research_history"] = [
                        {
                            key: deepcopy(item[key])
                            for key in (
                                "round",
                                "hypothesis",
                                "factor_ids",
                                "decision",
                                "development_metrics",
                            )
                        }
                        for item in attempts
                    ]
                return value

            feedback = call(
                "feedback" if number == 2 else f"round_{number}_feedback", collect_feedback
            )
        hypothesis = call(
            f"{prefix}_hypothesis",
            hypothesize,
            deepcopy(context),
            deepcopy(library),
            **(
                {"previous": deepcopy(incumbent["hypothesis"]), "feedback": deepcopy(feedback)}
                if incumbent is not None
                else {}
            ),
        )
        if context.get("allow_factor_proposals"):
            library = call(f"{prefix}_factor_library", extend_library, library, hypothesis)
        factors = call(f"{prefix}_factors", resolve_factors, library, hypothesis)
        prediction = call(
            prediction_stage, predict, deepcopy(context), factors, deepcopy(hypothesis)
        )
        decision = select(incumbent["prediction"], prediction) if incumbent and select else None
        accepted = incumbent is None or select is None or decision["revision_accepted"]
        prior_ids = set(incumbent["factor_ids"]) if incumbent else set()
        ids = list(prediction["factor_ids"])
        attempt = {
            "round": number,
            "hypothesis": deepcopy(hypothesis),
            "factor_ids": ids,
            "prediction": deepcopy(prediction),
            "prediction_stage": prediction_stage,
            "development_metrics": deepcopy(prediction.get("development", {}).get("metrics", {})),
            "decision": deepcopy(decision),
            "accepted": accepted,
            "factor_changes": {
                "activated": sorted(set(ids) - prior_ids),
                "deactivated": sorted(prior_ids - set(ids)),
                "retained": sorted(prior_ids & set(ids)),
            },
        }
        attempts.append(attempt)
        if accepted:
            incumbent, no_progress = attempt, 0
        else:
            no_progress += 1
        if select is not None:
            output = {
                "round": number,
                "accepted": accepted,
                "incumbent_round": incumbent["round"],
                "decision": decision,
                "factor_changes": attempt["factor_changes"],
                "completed_fits": number,
                "fit_budget": fit_budget,
                "development_metrics": attempt["development_metrics"],
                "strategy_metrics": prediction.get("strategy", {})
                .get("strategy", {})
                .get("metrics"),
            }
            call(f"round_{number}_decision", lambda output=output: output)
        if patience is not None and no_progress >= patience:
            stop_reason = "no_progress_limit"
            break
    return {
        "attempts": attempts,
        "selected_round": incumbent["round"],
        "completed_fits": len(attempts),
        "max_rounds": rounds,
        "fit_budget": fit_budget,
        "patience": patience,
        "stop_reason": stop_reason,
        "feedback": feedback,
        "scope": "development_selection_only; no independent validation",
    }
