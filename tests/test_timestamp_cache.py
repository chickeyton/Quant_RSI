"""A bounded parse cache must not cache mutable clocks or evidence visibility."""

from datetime import datetime, timedelta, timezone, tzinfo

import pytest

from quant_rsi.market import market_contracts


def original_timestamp(value):
    result = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("an explicit timezone is required")
    return result.astimezone(timezone.utc)


@pytest.mark.parametrize(
    "value",
    [
        "2025-01-02T03:04:05Z",
        "2025-01-02T04:04:05+01:00",
        "2025-01-01T22:04:05-05:00",
        "2025-01-02T03:04:05.123456+00:00",
        "2025-01-02T03:04:05." + "0" * 200 + "Z",
        "2025-01-02T03:04:05",
        "not-a-timestamp",
        datetime(2025, 1, 2),
        datetime(2025, 1, 2, tzinfo=timezone(timedelta(hours=7))),
    ],
)
def test_timestamp_matches_original_parser_cold_and_warm(value):
    market_contracts._timestamp_text.cache_clear()
    try:
        expected = original_timestamp(value)
    except ValueError as error:
        for _ in range(2):
            with pytest.raises(ValueError) as actual:
                market_contracts.timestamp(value)
            assert str(actual.value) == str(error)
    else:
        for _ in range(2):
            actual = market_contracts.timestamp(value)
            assert actual == expected
            assert actual.tzinfo is timezone.utc


def test_text_cache_is_bounded_and_bypasses_large_or_subclass_strings():
    cached = market_contracts._timestamp_text
    cached.cache_clear()
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for second in range(4097):
        market_contracts.timestamp((start + timedelta(seconds=second)).isoformat())
    assert cached.cache_info().currsize == cached.cache_info().maxsize == 4096
    before = cached.cache_info()

    class UnhashableText(str):
        __hash__ = None

    for value in (UnhashableText(start.isoformat()), "2025-01-02T03:04:05." + "0" * 200 + "Z"):
        assert market_contracts.timestamp(value) == original_timestamp(value)
    assert cached.cache_info() == before
    cached.cache_clear()


def test_datetime_custom_timezone_and_trusted_clock_remain_live(monkeypatch):
    class MutableZone(tzinfo):
        offset = timedelta(0)

        def utcoffset(self, dt):
            return self.offset

        def dst(self, dt):
            return timedelta(0)

    zone = MutableZone()
    value = datetime(2025, 1, 1, tzinfo=zone)
    first = market_contracts.timestamp(value)
    zone.offset = timedelta(hours=1)
    assert market_contracts.timestamp(value) == first - timedelta(hours=1)
    zone.offset = None
    with pytest.raises(ValueError, match="an explicit timezone is required"):
        market_contracts.timestamp(value)
    ticks = iter((first, first + timedelta(seconds=1)))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return next(ticks)

    monkeypatch.setattr(market_contracts, "datetime", Clock)
    assert market_contracts.trusted_now() == first.isoformat(timespec="microseconds")
    assert market_contracts.trusted_now() == (first + timedelta(seconds=1)).isoformat(
        timespec="microseconds"
    )
