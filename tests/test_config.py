import pandas as pd
from brent import config as C


def n_fridays(a, b):
    return len(pd.date_range(a, b, freq="W-FRI"))


def test_window_counts():
    assert n_fridays(*C.TRAIN) == 417
    assert n_fridays(*C.VALID) == 313
    assert n_fridays(*C.TEST) == 365
    assert n_fridays(*C.HOLDOUT) == 45
    assert n_fridays(C.SAMPLE_START, C.FREEZE_DATE) == 1140
    # Dukascopy hourly Brent starts Thu 2010-12-02 (verified 2026-11-23 with both tools), but Jan 2012 -
    # May 2013 is unreliable; decided 2026-11-23: hourly sample starts Fri 2013-06-07, and the hourly
    # variables are scaled over its first two years (105 weeks).
    assert C.INTRADAY_START == "2010-12-02" and C.HOURLY_SAMPLE_START == "2013-06-07"
    assert C.STANDARDIZE_WINDOW_HOURLY == (C.HOURLY_SAMPLE_START, "2015-06-05")
    assert n_fridays(*C.STANDARDIZE_WINDOW_HOURLY) == 105


def test_window_bounds_are_fridays():
    for a, b in (C.TRAIN, C.VALID, C.TEST, C.HOLDOUT, C.STANDARDIZE_WINDOW_HOURLY):
        assert pd.Timestamp(a).weekday() == 4 and pd.Timestamp(b).weekday() == 4
    assert pd.Timestamp(C.HOURLY_SAMPLE_START).weekday() == 4


def test_events_frozen():
    assert len(C.EVENTS) == 9 and len(C.NON_IRAN_EVENTS) == 3
    assert all(len(e) == 4 for e in C.EVENTS)


def test_intraday_and_extra_model_names():
    assert C.INTRADAY_DAY_END_UTC == 22 and C.MAIN_RV == "rv_hourly"
    assert C.RV_MEASURES == ("rv_hourly", "rv", "rv_gk")
    assert C.BENCHMARKS_MAIN == ("GARCH_t", "HAR", "RW", "HAR_h", "HARX_C", "HARX_CI", "GARCHX_C", "GARCHX_CI")
    assert C.IRAN_TONE_VARS == ("iran_conflict", "iran_coop")


def test_gdelt_v2_patch_window():
    # GDELT 1.0 has no daily-feed rows Jun 14 - Jul 1, 2025 (Israel-Iran war); patched from GDELT 2.0.
    assert C.GDELT_V2_PATCH_DAYS == ("2025-06-14", "2025-07-01")
    assert C.GDELT_V2_PATCH_QUERY == (20250101, 20251231)     # calibration year for the level shift
    assert C.GDELT_SPLICE_MIN_CORR == 0.8
    assert C.GDELT_STALE_POLICY == "merge"                    # stale rows are counted as daily feed


def test_gdelt_empty_week_policy():
    # GDELT 1.0 AND 2.0 are both down Jun 15 - Jul 1, 2025: the week ending 2025-06-27 has no GDELT day.
    # Decided 2026-11-23: carry the previous week's value forward (no look-ahead), flagged in the panel.
    assert C.GDELT_EMPTY_WEEK_POLICY == "locf"
    assert C.GDELT_MAX_EMPTY_WEEKS == 1      # more consecutive empty weeks than this -> stop, don't fill


def test_intraday_files_not_committed():
    # Decided 2026-11-23 by the user: the Dukascopy-derived daily files stay local (git-ignored).
    from brent import paths
    assert C.INTRADAY_REDISTRIBUTE is False
    ignored = (paths.ROOT / ".gitignore").read_text().splitlines()
    for f in ("data/raw/brent_intraday_daily.csv", "data/raw/brent_intraday_removed.csv"):
        assert f in ignored


def test_intraday_rule_decided():
    # Decided 2026-11-23: no contract rolls found; remove only returns across data holes (> 84 h).
    assert C.INTRADAY_ROLL_RULE == {"kind": "gap", "max_gap_hours": 84}
