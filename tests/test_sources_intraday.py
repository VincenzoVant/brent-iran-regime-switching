import numpy as np
import pandas as pd
import pytest

from brent import sources as S

NONE = {"kind": "none"}


def bars(ts, close, volume=1.0, tz="UTC"):
    idx = pd.DatetimeIndex(pd.to_datetime(ts))
    idx = idx.tz_localize(tz) if tz else idx
    c = np.asarray(close, float)
    return pd.DataFrame({"timestamp_utc": idx, "open": c, "high": c, "low": c, "close": c, "volume": volume})


def test_weekend_gap_lands_on_monday_once():
    h = S.normalize_hourly(bars(["2025-01-10 20:00", "2025-01-10 21:00", "2025-01-12 23:00", "2025-01-13 00:00"],
                                [100, 101, 103, 102]))
    daily, removed = S.intraday_daily(h, NONE)
    d = daily.set_index("date")
    fri, mon = pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-13")
    assert list(d.index) == [fri, mon]                          # no Saturday or Sunday row
    assert d.loc[fri, "n_hours"] == 1 and d.loc[mon, "n_hours"] == 2
    gap, last = 100 * np.log(103 / 101), 100 * np.log(102 / 103)
    assert d.loc[fri, "rv_intraday"] == pytest.approx((100 * np.log(101 / 100)) ** 2)
    assert d.loc[mon, "rv_intraday"] == pytest.approx(gap**2 + last**2)
    assert (d.loc[mon, "open"], d.loc[mon, "high"], d.loc[mon, "low"], d.loc[mon, "close"]) == (103, 103, 102, 102)
    assert S.hourly_returns(h).sum() == pytest.approx(100 * np.log(102 / 100))
    assert removed.empty


def test_day_boundary_is_22_utc_in_summer_and_winter():
    ts = pd.DatetimeIndex(["2025-07-01 22:00", "2025-07-01 23:00",
                           "2025-01-14 21:00", "2025-01-14 22:00"]).tz_localize("Europe/London")
    h = S.normalize_hourly(bars(ts, [1, 1, 1, 1], tz=None))
    assert str(h["timestamp_utc"].dt.tz) == "UTC"
    assert list(S.intraday_day(h["timestamp_utc"])) == [
        pd.Timestamp("2025-01-14"), pd.Timestamp("2025-01-15"),    # 21:00 UTC stays, 22:00 UTC moves
        pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-02")]    # 22:00 BST = 21:00 UTC stays, 23:00 BST moves


def test_naive_timestamps_are_utc():
    h = S.normalize_hourly(bars(["2025-01-14 22:00"], [1.0], tz=None))
    assert h["timestamp_utc"].iloc[0] == pd.Timestamp("2025-01-14 22:00", tz="UTC")


def test_friday_22utc_bar_goes_to_saturday():
    h = S.normalize_hourly(bars(["2025-01-10 21:00", "2025-01-10 22:00"], [100, 101]))
    daily, _ = S.intraday_daily(h, NONE)
    assert daily["date"].tolist()[-1] == pd.Timestamp("2025-01-11")


def test_roll_jump_found_and_removed():
    ts = pd.date_range("2025-03-03 00:00", periods=400, freq="h", tz="UTC")
    ts = ts[ts.day_of_week < 5][:200]                           # trading days only (weekend bars are dropped)
    r = np.where(np.arange(199) % 2 == 0, 0.001, -0.001)
    r[150] = 0.05                                               # the return of bar ts[151]
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    h = S.normalize_hourly(pd.DataFrame({"timestamp_utc": ts, "open": close, "high": close,
                                         "low": close, "close": close, "volume": 1.0}))
    cand = S.roll_candidates(h, k=8.0, window=120)
    assert len(cand) == 1 and cand["timestamp_utc"].iloc[0] == ts[151]
    assert cand["ratio"].iloc[0] == pytest.approx(50.0)
    d0, rm0 = S.intraday_daily(h, NONE)
    assert rm0.empty
    for rule in ({"kind": "threshold", "k": 8.0, "window": 120},
                 {"kind": "timestamps", "timestamps": [ts[151].isoformat()]}):
        d1, rm1 = S.intraday_daily(h, rule)
        assert len(rm1) == 1 and rm1["timestamp_utc"].iloc[0] == ts[151] and rm1["reason"].iloc[0] == "roll"
        assert d0["n_hours"].sum() - d1["n_hours"].sum() == 1
        assert d0["rv_intraday"].sum() - d1["rv_intraday"].sum() == pytest.approx(25.0)
    with pytest.raises(ValueError):
        S.intraday_daily(h, None)
    with pytest.raises(ValueError):
        S.roll_mask(h, {"kind": "bogus"})


def test_zero_volume_and_freeze_cut():
    h = S.normalize_hourly(bars(["2026-11-06 20:00", "2026-11-06 21:00", "2026-11-06 22:00"],
                                [100, 100, 101], volume=[1, 0, 1]))
    assert h["timestamp_utc"].tolist() == [pd.Timestamp("2026-11-06 20:00", tz="UTC")]


def test_dukascopy_frame_to_hourly_schema():
    # dukascopy_python.to_dataframe output: tz-aware 'timestamp' index plus extra columns
    idx = pd.DatetimeIndex(["2025-01-10 01:00", "2025-01-10 02:00"], tz="UTC", name="timestamp")
    raw = pd.DataFrame({"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0],
                        "volume": [5.0, 6.0], "time_unit": "hour", "interval": 1}, index=idx)
    out = S.dukascopy_frame_to_hourly(raw)
    assert list(out.columns) == ["timestamp_utc", "open", "high", "low", "close", "volume"]
    assert out["timestamp_utc"].iloc[0] == pd.Timestamp("2025-01-10 01:00", tz="UTC")


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_gap_hours_independent_of_time_resolution(unit):
    # pandas 3 often stores timestamps in microseconds; gap_hours must still be in hours.
    ts = pd.DatetimeIndex(["2025-01-10 20:00", "2025-01-10 21:00", "2025-01-13 01:00"], tz="UTC").as_unit(unit)
    h = S.normalize_hourly(pd.DataFrame({"timestamp_utc": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                                         "close": [100.0, 101.0, 150.0], "volume": 1.0}))
    cand = S.roll_candidates(h, k=0.0, window=1)
    assert cand.set_index("timestamp_utc").loc[ts[2], "gap_hours"] == pytest.approx(52.0)


def test_gap_rule_removes_only_returns_across_data_holes():
    # Decided 2026-11-23: no roll removal; only returns spanning a data hole (> 84 h) are removed.
    # A normal weekend (52 h) and an Easter/Christmas closure (76 h) are kept.
    ts = ["2025-01-09 20:00", "2025-01-09 21:00",      # Thu
          "2025-01-13 01:00",                          # Mon after weekend: 76 h gap -> kept
          "2025-01-13 02:00",
          "2025-01-17 03:00"]                          # 97 h later: data hole -> removed
    h = S.normalize_hourly(bars(ts, [100, 101, 102, 103, 110]))
    rule = {"kind": "gap", "max_gap_hours": 84}
    assert S.roll_mask(h, rule).tolist() == [False, False, False, False, True]
    daily, removed = S.intraday_daily(h, rule)
    assert len(removed) == 1 and removed["reason"].iloc[0] == "data_gap"
    assert removed["timestamp_utc"].iloc[0] == pd.Timestamp("2025-01-17 03:00", tz="UTC")
    assert daily.set_index("date").loc[pd.Timestamp("2025-01-17"), "n_hours"] == 0
    # threshold/timestamps rules still label their removals as rolls
    _, rm = S.intraday_daily(h, {"kind": "timestamps", "timestamps": ["2025-01-13T02:00:00+00:00"]})
    assert rm["reason"].tolist() == ["roll"]


def test_closed_market_weekend_bars_dropped():
    # Found 2026-11-23: 2012-13 Dukascopy has stale flat bars on Sat/Sun (market closed). Bars starting on
    # Saturday, or on Sunday before 22:00 UTC, are dropped; Fri 22:00 and the Sun 22:00+ reopening are kept.
    ts = ["2025-01-10 21:00",            # Fri, kept
          "2025-01-10 22:00",            # Fri 22:00 (goes to Saturday's date / next week), kept
          "2025-01-11 10:00",            # Sat, dropped
          "2025-01-12 12:00",            # Sun before reopening, dropped
          "2025-01-12 22:00",            # Sun reopening, kept (counts for Monday)
          "2025-01-13 01:00"]            # Mon, kept
    h = S.normalize_hourly(bars(ts, [100, 101, 111.71, 111.71, 102, 103]))
    kept = [t.strftime("%a %H:%M") for t in h["timestamp_utc"]]
    assert kept == ["Fri 21:00", "Fri 22:00", "Sun 22:00", "Mon 01:00"]
    daily, _ = S.intraday_daily(h, NONE)
    assert (daily["date"].dt.day_of_week != 6).all()
    # Monday's first return is measured against the last real (Friday) close, not the stale weekend print
    assert S.hourly_returns(h).iloc[2] == pytest.approx(100 * np.log(102 / 101))
