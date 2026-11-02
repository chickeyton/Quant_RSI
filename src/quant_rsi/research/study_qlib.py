"""Optional upstream Qlib workflow over the selected study's frozen numerical inputs."""

import importlib.metadata
from copy import deepcopy

import numpy as np

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.research.demo_features import price_metrics
from quant_rsi.research.demo_strategy import evaluate_strategy

QLIB_VERSION = "0.9.7"
NOTICE = (
    "Actual Qlib DatasetH/DataHandlerLP/StaticDataLoader/LGBModel and local workflow recorder. "
    "Both engines use the selected native study's features, training labels and evaluation "
    "sessions, with the same LightGBM parameters. One additional comparison fit; no search, "
    "early stopping or feedback to selection. Strategy results use the shared Quant RSI "
    "account adapter, not Qlib's portfolio/backtest engine. This compares workflow semantics, "
    "not independent alpha, broader Qlib research capabilities or research-policy quality."
)


def require_qlib():
    """Fail before news/LLM work when the explicitly requested optional runtime is absent."""
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in ("pyqlib", "mlflow", "pandas", "numpy", "lightgbm")
        }
        if versions["pyqlib"] != QLIB_VERSION:
            raise ValueError(f"--compare-qlib requires pyqlib=={QLIB_VERSION}")
        from qlib.contrib.model.gbdt import LGBModel  # noqa: F401
        from qlib.data.dataset import DatasetH  # noqa: F401
        from qlib.workflow import R  # noqa: F401
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ValueError(
            "--compare-qlib needs the optional Qlib workflow environment; "
            "see docs/qlib_workflow_comparison.md"
        ) from exc
    return versions


def compare_qlib(bundle, data, selected, terminal, root, *, policy):
    """Execute upstream fit/predict, then evaluate with the existing common account rules."""
    versions = require_qlib()
    import pandas as pd
    import qlib
    from mlflow.tracking import MlflowClient
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader
    from qlib.workflow import R

    root.mkdir(exist_ok=False)
    inputs = deepcopy(bundle) | {
        "price_provider": selected["price_provider"],
        "native_prediction_sha256": digest(canonical(selected)),
        "price_inputs_sha256": digest(
            canonical(
                {"dates": data.dates, "opens": data.opens.tolist(), "closes": data.closes.tolist()}
            )
        ),
        "account_policy": policy,
        "notice": NOTICE,
    }
    raw = canonical(inputs)
    (root / "inputs.json").write_bytes(raw)
    segments, frames = {}, []
    for name, segment in inputs["segments"].items():
        if not segment["available"]:
            if name in {"train", "development"}:
                raise ValueError("Comparison training/development inputs must be complete")
            continue
        sessions = pd.to_datetime(segment["sessions"])
        index = pd.MultiIndex.from_product(
            [sessions, [inputs["symbol"]]], names=["datetime", "instrument"]
        )
        features = pd.DataFrame(segment["features"], index=index, columns=inputs["factor_ids"])
        labels = pd.DataFrame({"log_return": segment.get("labels", np.nan)}, index=index)
        frames.append(pd.concat({"feature": features, "label": labels}, axis=1))
        segments[name] = (sessions[0], sessions[-1])
    frame = pd.concat(frames).sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Comparison train/evaluation sessions must not overlap")
    if max(inputs["segments"]["train"]["label_sessions"]) > min(
        inputs["segments"]["development"]["sessions"]
    ):
        raise ValueError("Comparison requires mature training labels")
    tracking_uri = "sqlite:///" + str(root / "mlflow.db")
    experiment = "quant-rsi-common-inputs"
    MlflowClient(tracking_uri=tracking_uri).create_experiment(
        experiment, artifact_location=(root / "artifacts").as_uri()
    )
    qlib.init(
        provider_uri=str(root),
        region="us",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": tracking_uri, "default_exp_name": experiment},
        },
    )
    dataset = DatasetH(
        handler=DataHandlerLP(
            data_loader=StaticDataLoader(frame), infer_processors=[], learn_processors=[]
        ),
        segments=segments,
    )
    for name in segments:
        actual = dataset.prepare(name, col_set="feature")
        expected = inputs["segments"][name]
        if (
            actual.index.get_level_values("datetime").strftime("%Y-%m-%d").tolist()
            != expected["sessions"]
            or list(actual.columns) != inputs["factor_ids"]
            or not np.array_equal(actual.to_numpy(), expected["features"])
        ):
            raise ValueError("Qlib changed the common feature rows or order")
    labels = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
    if not np.array_equal(labels.to_numpy().ravel(), inputs["segments"]["train"]["labels"]):
        raise ValueError("Qlib changed the mature training labels")
    model = LGBModel(
        early_stopping_rounds=0,
        num_boost_round=BOOST_ROUNDS,
        **(MODEL_PARAMETERS | {"seed": 7}),
    )
    result = {
        "status": "completed",
        "versions": versions,
        "notice": NOTICE,
        "selected_round": inputs["selected_round"],
        "factor_ids": inputs["factor_ids"],
        "inputs_sha256": digest(raw),
        "comparison_fits": 1,
        "segments": {},
        "tolerances": {"log_return_atol": 1e-10, "price_atol": 1e-8},
    }
    with R.start(experiment_name=experiment, uri=tracking_uri):
        R.log_params(inputs_sha256=result["inputs_sha256"], selected_round=inputs["selected_round"])
        model.fit(dataset, verbose_eval=0)
        model.model.save_model(str(root / "model.txt"))
        for name in ("development", "terminal"):
            if name not in segments:
                result["segments"][name] = {
                    "status": "unavailable",
                    "reason": "disabled_or_missing_selected_features; no rows dropped",
                }
                continue
            native = selected["development"] if name == "development" else terminal["evaluation"]
            native_account = selected["strategy"] if name == "development" else terminal["strategy"]
            values = model.predict(dataset, segment=name)
            rows = deepcopy(native["rows"])
            if values.index.get_level_values("datetime").strftime("%Y-%m-%d").tolist() != [
                row["origin_session"] for row in rows
            ]:
                raise ValueError("Qlib predictions differ from the frozen evaluation population")
            for row, value in zip(rows, values, strict=True):
                row["predicted_log_return"] = float(value)
                row["predicted_price"] = float(row["reference_price"] * np.exp(value))
                if not np.isfinite(row["predicted_price"]) or row["predicted_price"] <= 0:
                    raise ValueError("Qlib produced an invalid price forecast")
            differences = {
                key: max(abs(a[key] - b[key]) for a, b in zip(rows, native["rows"], strict=True))
                for key in ("predicted_log_return", "predicted_price")
            }
            qlib_account = evaluate_strategy(data, inputs["symbol"], rows, **policy)
            if name == "terminal":
                qlib_account["evidence_mode"] = "historical_terminal_hypothetical_account"
            result["segments"][name] = {
                "status": "completed",
                "sample_count": len(rows),
                "native": {"metrics": native["metrics"], "strategy": native_account},
                "qlib": {
                    "metrics": price_metrics(rows),
                    "rows": rows,
                    "strategy": qlib_account,
                },
                "max_absolute_difference": differences,
                "prediction_parity": differences["predicted_log_return"] <= 1e-10
                and differences["predicted_price"] <= 1e-8,
            }
        result["recorder_id"] = R.get_recorder().id
    (root / "comparison.json").write_bytes(canonical(result))
    return result
