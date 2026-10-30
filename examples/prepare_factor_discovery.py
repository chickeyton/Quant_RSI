"""Freeze shared KO factors and an existing input envelope; no collection or fits."""

import argparse
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical, decode
from quant_rsi.factors.discovery import DiscoveryPlan, bind_catalog
from quant_rsi.forecast.forecast_protocol import DEFAULT_FOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--folds", default=canonical(DEFAULT_FOLDS).decode())
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--factor-set", choices=("legacy", "candidate-prices"), default="legacy")
    parser.add_argument(
        "--exposure",
        choices=("engineering_fixture", "historical_market_exploration"),
        required=True,
    )
    args = parser.parse_args()
    envelope, leaves = bind_catalog(
        decode(args.inputs.read_bytes()),
        Path(__file__).resolve().parents[1] / "instruments",
        "US",
        "KO",
        (
            {"price_momentum": "target", "price_mean_return": "target"}
            if args.factor_set == "legacy"
            else {"qlib_roc": "target", "qlib_ma": "target"}
        ),
    )
    plan = DiscoveryPlan.model_validate_json(
        canonical(
            dict(
                campaign_id=args.campaign_id,
                instrument="KO",
                exposure=args.exposure,
                leaves=[leaf.model_dump(mode="json") for leaf in leaves],
                fold_origins=decode(args.folds),
                max_rounds=args.max_rounds,
            )
        )
    )
    outputs = {
        "inputs.json": canonical(envelope),
        "plan.json": canonical(plan.model_dump(mode="json")),
    }
    for name, raw in outputs.items():
        path = args.output / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"existing preparation differs: {path}")
    for name, raw in outputs.items():
        atomic_write(args.output / name, raw)
    print(args.output / "plan.json")


if __name__ == "__main__":
    main()
