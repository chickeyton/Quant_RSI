"""Daily (interday) feature leaves: values, causality, missing data and live parity."""

import math

import numpy as np
import pytest
from intraday_helpers import build_snapshot, sessions

from quant_rsi.intraday.features import (
    DAILY_LEAVES,
    build_features,
    daily_bar,
    session_features,
    validate_leaves,
)
from quant_rsi.intraday.paper import LivePanel
from quant_rsi.intraday.storage import Block

# Leaves that read only earlier sessions, so they are constant within a session.
PRIOR_ONLY = ("d_ret_1", "d_ret_5", "d_ret_20", "d_rv_20", "d_atr_14", "d_prev_range")


class Changed:
    """A bar source that rewrites chosen blocks; everything else is the snapshot's."""

    def __init__(self, snapshot, change):
        self.contract, self.snapshot, self.change = snapshot.contract, snapshot, change

    def block(self, instrument, grid, session_id):
        block = self.snapshot.block(instrument, grid, session_id)
        return None if block is None else self.change(block)


def scaled(block: Block, factor: float, after: int = -1) -> Block:
    columns = {k: np.array(v, copy=True) for k, v in block.columns.items()}
    for name in ("open", "high", "low", "close"):
        columns[name][after + 1 :] *= factor
    return Block(block.instrument, block.grid, block.session_id, columns)


def leaves_at(source, index, t, leaves=DAILY_LEAVES):
    sessions = source.contract.calendar.sessions
    _, values = session_features(source, sessions, index, "KO", leaves)
    return dict(zip(leaves, values[t], strict=True))


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    store, ref = build_snapshot(tmp_path_factory.mktemp("daily"), count=56)
    return store.load(ref)


def test_daily_leaves_match_a_direct_computation(snapshot):
    sessions = snapshot.contract.calendar.sessions
    index, t = 55, 10
    got = leaves_at(snapshot, index, t)
    bars = [
        daily_bar(snapshot.block("KO", "decision", s.session_id))
        for s in sessions[index - 50 : index]
    ]
    closes = np.array([bar.close for bar in bars])
    now = snapshot.block("KO", "decision", sessions[index].session_id).columns["close"][t]
    prev = bars[-1]
    true_ranges = [
        max(bar.high, before.close) - min(bar.low, before.close)
        for before, bar in zip(bars[-15:-1], bars[-14:], strict=True)
    ]
    atr = np.mean(true_ranges)
    expected = {
        "d_ret_1": math.log(closes[-1] / closes[-2]),
        "d_ret_5": math.log(closes[-1] / closes[-6]),
        "d_ret_20": math.log(closes[-1] / closes[-21]),
        "d_rv_20": np.std(np.diff(np.log(closes[-21:]))),
        "d_atr_14": atr / prev.close,
        "d_move_atr": (now - prev.close) / atr,
        "d_ma_20_dist": now / closes[-20:].mean() - 1,
        "d_ma_50_dist": now / closes[-50:].mean() - 1,
        "d_prev_range": (prev.high - prev.low) / prev.close,
        "d_prev_close_pos": (prev.close - prev.low) / (prev.high - prev.low),
        "d_prev_high_dist": now / prev.high - 1,
        "d_prev_low_dist": now / prev.low - 1,
    }
    assert got == pytest.approx(expected, rel=1e-12)
    # A daily bar is the session's regular-hours decision bars.
    block = snapshot.block("KO", "decision", sessions[index - 1].session_id)
    assert prev.open == block.columns["open"][0] and prev.close == block.columns["close"][-1]
    assert prev.high == block.columns["high"].max() and prev.low == block.columns["low"].min()


def test_prior_session_leaves_are_constant_within_a_session(snapshot):
    sessions = snapshot.contract.calendar.sessions
    _, values = session_features(snapshot, sessions, 30, "KO", PRIOR_ONLY)
    assert np.isfinite(values).all()
    assert (values == values[0]).all()


def test_daily_leaves_never_read_today_after_t_or_later_sessions(snapshot):
    sessions = snapshot.contract.calendar.sessions
    index, t = 52, 20
    today = sessions[index].session_id
    later = {s.session_id for s in sessions[index + 1 :]}

    def future(block):
        if block.session_id == today:
            return scaled(block, 1.5, after=t)
        return scaled(block, 2.0) if block.session_id in later else block

    assert leaves_at(Changed(snapshot, future), index, t) == leaves_at(snapshot, index, t)

    def yesterday(block):
        return scaled(block, 1.1) if block.session_id == sessions[index - 1].session_id else block

    moved = leaves_at(Changed(snapshot, yesterday), index, t)
    assert moved["d_ret_1"] != leaves_at(snapshot, index, t)["d_ret_1"]


def test_warm_up_and_a_missing_bar_make_only_the_affected_leaves_missing(tmp_path):
    gap = sessions(56)[36].session_id  # in the 20-session window of session 55, not ATR's 15
    store, ref = build_snapshot(tmp_path, count=56, missing={("KO", gap): range(25, 30)})
    snapshot = store.load(ref)
    got = leaves_at(snapshot, 55, 10)
    assert math.isnan(got["d_ma_20_dist"]) and math.isnan(got["d_ma_50_dist"])
    assert math.isnan(got["d_ret_20"]) and math.isnan(got["d_rv_20"])
    assert np.isfinite([got[n] for n in ("d_ret_5", "d_atr_14", "d_prev_range")]).all()
    early = leaves_at(snapshot, 30, 10)  # fewer than 50 earlier sessions
    assert math.isnan(early["d_ma_50_dist"]) and np.isfinite(early["d_ma_20_dist"])


def test_a_calendar_hole_makes_windows_across_it_missing(snapshot):
    sessions = snapshot.contract.calendar.sessions
    # Dropping a full week leaves a gap no holiday weekend explains.
    holed = (*sessions[:20], *sessions[25:])
    _, values = session_features(snapshot, holed, 22, "KO", ("d_ret_1", "d_ret_5", "d_ret_20"))
    d_ret_1, d_ret_5, d_ret_20 = values[10]
    assert np.isfinite(d_ret_1) and math.isnan(d_ret_5) and math.isnan(d_ret_20)


def test_live_panel_gives_the_replay_values(snapshot):
    sessions = snapshot.contract.calendar.sessions
    today = sessions[-1]
    panel = LivePanel(snapshot, today)
    recorded = snapshot.block("KO", "decision", today.session_id).columns
    for name, column in recorded.items():
        panel.live[("KO", "decision")][name] = np.array(column, copy=True)
    _, live = session_features(
        panel, panel.contract.calendar.sessions, panel.index, "KO", DAILY_LEAVES
    )
    replay = build_features(snapshot, DAILY_LEAVES, session_indices=[len(sessions) - 1])
    assert np.isfinite(live).all()
    np.testing.assert_array_equal(live, replay.values)


def test_daily_leaves_are_registered_target_leaves(snapshot):
    validate_leaves(snapshot.contract, DAILY_LEAVES)
    for unknown in ("d_ret_2", "d_ma_10_dist", "ctx:SPY:d_ret_1"):
        with pytest.raises(ValueError, match="unsupported"):
            validate_leaves(snapshot.contract, (unknown,))
