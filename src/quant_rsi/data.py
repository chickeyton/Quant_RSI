from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from quant_rsi.schemas import DataSpec, FactorSpec


@dataclass(frozen=True)
class MarketData:
    dates: tuple[str, ...]
    instruments: tuple[str, ...]
    opens: np.ndarray
    closes: np.ndarray

    def __post_init__(self):
        shape = (len(self.dates), len(self.instruments))
        if len(set(self.instruments)) != shape[1] or not shape[1]:
            raise ValueError("invalid universe")
        parsed = tuple(date.fromisoformat(d) for d in self.dates)
        if tuple(sorted(set(parsed))) != parsed or len(parsed) < 3:
            raise ValueError("dates must be unique, chronological ISO dates")
        for values in (self.opens, self.closes):
            if values.shape != shape or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("complete positive finite quotes required")

    def timestamp(self, index: int, phase: str) -> str:
        clock = {"open": "09:00:00", "close": "16:00:00"}[phase]
        return f"{self.dates[index]}T{clock}+00:00"

    def to_json(self) -> dict:
        return {
            "kind": "synthetic_weekday_quotes_not_market_data",
            "dates": list(self.dates),
            "instruments": list(self.instruments),
            "opens": self.opens.tolist(),
            "closes": self.closes.tolist(),
        }

    @classmethod
    def from_json(cls, value: dict) -> "MarketData":
        if value["kind"] in {"LocalResearchData", "DerivedLocalResearchData"}:
            from quant_rsi.local_research import LocalResearchData

            return LocalResearchData.from_json(value)
        if value["kind"] != "synthetic_weekday_quotes_not_market_data":
            raise ValueError("only synthetic data is supported")
        return cls(
            tuple(value["dates"]),
            tuple(value["instruments"]),
            np.asarray(value["opens"], dtype=float),
            np.asarray(value["closes"], dtype=float),
        )


def synthetic_data(spec: DataSpec, *, rng=None) -> MarketData:
    if spec.source != "synthetic":
        raise ValueError("local inputs cannot be replaced with generated synthetic data")
    rng = np.random.default_rng(spec.seed) if rng is None else rng
    days = []
    day = date(2020, 1, 1)
    while len(days) < spec.sessions:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    opens = np.empty((spec.sessions, spec.instruments))
    closes = np.empty_like(opens)
    previous = np.linspace(50, 100, spec.instruments)
    for t in range(spec.sessions):
        opens[t] = previous * np.exp(rng.normal(0, 0.006, spec.instruments))
        closes[t] = opens[t] * np.exp(rng.normal(0.0002, 0.008, spec.instruments))
        previous = closes[t]
    return MarketData(
        tuple(days), tuple(f"SYN{i:02d}" for i in range(spec.instruments)), opens, closes
    )


def features(data: MarketData, factor: FactorSpec) -> np.ndarray:
    close = data.closes
    daily = np.full_like(close, np.nan)
    daily[1:] = close[1:] / close[:-1] - 1
    values = np.full_like(close, np.nan)
    w = factor.window
    if factor.op == "momentum":
        values[w:] = close[w:] / close[:-w] - 1
    else:
        for t in range(w, len(close)):
            values[t] = daily[t - w + 1 : t + 1].mean(axis=0)
    return np.stack([values, daily, close / data.opens - 1], axis=2)


def decision_indices(data: MarketData, cutoff: int) -> range:
    # A decision at close t enters at open t+1 and is valued/exited at open t+2.
    if cutoff < 1 or cutoff > len(data.dates) - 3:
        raise ValueError("invalid decision interval or missing terminal open")
    return range(cutoff, len(data.dates) - 2)
