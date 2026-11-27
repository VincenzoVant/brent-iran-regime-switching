import numpy as np
import pandas as pd
import pytest

from brent import config as C
from brent import weekly as W


def T(s):
    return pd.Timestamp(s)


SENT = {"articles_iran_conflict": 1, "articles_iran_coop": 1, "tone_x_articles_iran": -1.0,
        "articles_iran_tone_n": 1, "goldstein_x_articles_iran": 1.0, "articles_iran_goldstein_n": 1}


# ------------------------------------------------------------------ week mapping and Brent

def test_week_ending_and_weekend_rule():
    assert W.week_ending(T("2019-09-14")) == T("2019-09-20")   # Abqaiq, Saturday
    assert W.week_ending(T("2026-02-28")) == T("2026-03-06")   # Saturday
    assert W.week_ending(T("2026-03-01")) == T("2026-03-06")   # Sunday
    assert W.week_ending(T("2026-03-06")) == T("2026-03-06")   # Friday stays


def test_holiday_friday_week():
    days = pd.to_datetime(["2025-04-11", "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-21"])
    brent = pd.DataFrame({"date": days, "brent": [60.0, 61, 62, 61, 63, 64]})
    out = W.weekly_brent(brent, pd.DatetimeIndex([T("2025-04-18"), T("2025-04-25")]))
    wk = out.loc[T("2025-04-18")]
    lr = 100 * np.diff(np.log([60.0, 61, 62, 61, 63]))
    assert wk["price"] == 63 and wk["n_days"] == 4
    assert wk["r"] == pytest.approx(lr.sum()) and wk["rv"] == pytest.approx((lr**2).sum())
    assert out.loc[T("2025-04-25"), "r"] == pytest.approx(100 * np.log(64 / 63))


def test_week_without_trading_raises(synthetic_raw):
    raw = dict(synthetic_raw)
    b = raw["brent"]
    raw["brent"] = b[~b["date"].between("2010-03-06", "2010-03-12")]
    with pytest.raises(ValueError, match="2010-03-12"):
        W.build_weekly_panel(raw)


# ------------------------------------------------------------------ GDELT

def test_gdelt_share_split_stale_merge_and_drop():
    gd = pd.DataFrame({
        "feed": ["backfile", "backfile", "daily", "daily", "stale"],
        "date": pd.to_datetime(["2013-03-29", "2013-04-02", "2013-04-01", "2013-04-02", "2013-04-02"]),
        "n_events": [1, 1, 1, 1, 50], "articles_all": [100, 999, 300, 100, 5000],
        "articles_iran_actor": [1, 999, 3, 4, 0], "articles_iran_actor_geo": [2, 999, 4, 5, 0], "events_iran_actor": 1,
        **SENT})
    out, counts = W.gdelt_weekly(gd, pd.DatetimeIndex([T("2013-03-29"), T("2013-04-05")]))
    assert counts == {"wrong_side": 1, "stale_rows": 1, "stale_events": 50, "patched_days": 0, "filled_weeks": 0}
    # the backfile row dated after the switch is dropped; the stale row is MERGED (counted as daily feed)
    assert out.loc[T("2013-04-05"), "iran_share"] == pytest.approx(7 / 5400)   # sum/sum, not mean of shares
    assert out.loc[T("2013-04-05"), "gdelt_days"] == 2
    assert out.loc[T("2013-03-29"), "iran_share"] == pytest.approx(1 / 100)


def test_gdelt_conflict_coop_tone():
    gd = pd.DataFrame({
        "feed": "daily", "date": pd.to_datetime(["2015-06-08", "2015-06-09"]),
        "n_events": 1, "articles_all": [100, 300], "articles_iran_actor": [10, 20],
        "articles_iran_actor_geo": [12, 22], "events_iran_actor": 1,
        "articles_iran_conflict": [3, 5], "articles_iran_coop": [6, 12],
        "tone_x_articles_iran": [-20.0, 10.0], "articles_iran_tone_n": [4, 6],
        "goldstein_x_articles_iran": [5.0, -15.0], "articles_iran_goldstein_n": [10, 10]})
    out, _ = W.gdelt_weekly(gd, pd.DatetimeIndex([T("2015-06-12")]))
    wk = out.loc[T("2015-06-12")]
    assert wk["iran_gdelt"] == pytest.approx(np.log(30 / 400))
    assert wk["iran_conflict"] == pytest.approx(np.log(8 / 400))
    assert wk["iran_coop"] == pytest.approx(np.log(18 / 400))
    assert wk["iran_tone"] == pytest.approx(-1.0)       # (-20 + 10) / (4 + 6)
    assert wk["iran_goldstein"] == pytest.approx(-0.5)  # (5 - 15) / 20
    assert wk["gdelt_filled"] == 0


def _gdelt_days(days, all_=1000, iran=10, tone_mean=-2.0, gold_mean=1.0, feed="daily"):
    n = len(days)
    iran = np.broadcast_to(np.asarray(iran, float), (n,))
    tm = np.broadcast_to(np.asarray(tone_mean, float), (n,))
    gm = np.broadcast_to(np.asarray(gold_mean, float), (n,))
    return pd.DataFrame({"feed": feed, "date": days, "n_events": 100, "articles_all": float(all_),
                         "articles_iran_actor": iran, "articles_iran_actor_geo": 1.5 * iran, "events_iran_actor": 5,
                         "articles_iran_conflict": 0.4 * iran, "articles_iran_coop": 0.5 * iran,
                         "tone_x_articles_iran": tm * 20, "articles_iran_tone_n": 20.0,
                         "goldstein_x_articles_iran": gm * 20, "articles_iran_goldstein_n": 20.0})


def test_empty_week_carries_previous_week_forward():
    weeks = pd.DatetimeIndex([T("2025-06-13"), T("2025-06-20"), T("2025-06-27")])
    days = pd.date_range("2025-06-07", "2025-06-27")
    gd = _gdelt_days(days, iran=np.where(days <= "2025-06-13", 10, 30))
    gd = gd[~gd["date"].between("2025-06-14", "2025-06-20")]           # middle week: no GDELT day
    out, counts = W.gdelt_weekly(gd, weeks)
    cols = ["iran_share", "iran_gdelt", "iran_gdelt_geo", "iran_conflict", "iran_coop", "iran_tone", "iran_goldstein"]
    assert out.loc[T("2025-06-20"), cols].equals(out.loc[T("2025-06-13"), cols])
    assert out.loc[T("2025-06-20"), "gdelt_filled"] == 1 and out.loc[T("2025-06-20"), "gdelt_days"] == 0
    assert out.loc[T("2025-06-27"), "gdelt_filled"] == 0 and counts["filled_weeks"] == 1
    # two empty weeks in a row, or an empty first week, are never filled
    with pytest.raises(ValueError, match="2025-06-20"):
        W.gdelt_weekly(gd[gd["date"] <= "2025-06-13"], weeks)
    with pytest.raises(ValueError, match="2025-06-13"):
        W.gdelt_weekly(gd[gd["date"] >= "2025-06-21"], weeks)


def _v1_v2_2025():
    """GDELT 1.0 = 2 x the volume of 2.0, half its Iran share, tone +0.5 and Goldstein +1.0 higher."""
    days = pd.date_range("2025-01-01", "2025-12-31")
    f = 1 + 0.5 * np.sin(np.arange(len(days)) / 9.0)             # common news cycle in both versions
    g = np.cos(np.arange(len(days)) / 5.0)
    v2 = _gdelt_days(days, all_=1000, iran=20 * f, tone_mean=-1.5 + 0.3 * g, gold_mean=1.0 + 0.2 * g, feed="v2")
    v1_full = _gdelt_days(days, all_=2000, iran=20 * f, tone_mean=-1.0 + 0.3 * g, gold_mean=2.0 + 0.2 * g)
    return v1_full, v2


def test_v2_patch_fills_outage_days_at_v1_level():
    v1_full, v2 = _v1_v2_2025()
    p0, p1 = C.GDELT_V2_PATCH_DAYS
    v1 = v1_full[~v1_full["date"].between(p0, p1)]                    # the 18-day outage
    weeks = W.fridays("2025-01-03", "2025-12-26")
    truth, _ = W.gdelt_weekly(v1_full, weeks)
    out, counts = W.gdelt_weekly(v1, weeks, v2=v2)
    assert counts["patched_days"] == 18 and counts["filled_weeks"] == 0
    for wk in ("2025-06-20", "2025-06-27", "2025-07-04"):                # weeks touching the outage
        for col in ("iran_gdelt", "iran_gdelt_geo", "iran_conflict", "iran_coop", "iran_tone", "iran_goldstein"):
            assert out.loc[T(wk), col] == pytest.approx(truth.loc[T(wk), col], abs=1e-9), (wk, col)
    info = W.v2_patch_info(v1, v2)
    assert info.loc["articles_iran_actor", "shift"] == pytest.approx(np.log(0.5))
    assert info.loc["articles_iran_actor", "corr"] == pytest.approx(1.0)
    assert info.loc["iran_tone", "shift"] == pytest.approx(0.5)
    assert bool(info.loc["articles_iran_actor", "pass"])


def test_v2_patch_refuses_uncorrelated_versions():
    v1_full, v2 = _v1_v2_2025()
    p0, p1 = C.GDELT_V2_PATCH_DAYS
    v1 = v1_full[~v1_full["date"].between(p0, p1)]
    rng = np.random.default_rng(7)
    v2 = v2.assign(articles_iran_actor=rng.uniform(5, 40, len(v2)))   # 2.0 no longer tracks 1.0
    with pytest.raises(ValueError, match="correlation"):
        W.gdelt_weekly(v1, W.fridays("2025-01-03", "2025-12-26"), v2=v2)


@pytest.mark.parametrize("col", ["articles_iran_actor", "articles_iran_conflict"])
def test_zero_iran_week_raises(synthetic_raw, col):
    raw = dict(synthetic_raw)
    g = raw["gdelt"].copy()
    g.loc[g["date"].between("2015-06-06", "2015-06-12"), [col]] = 0
    raw["gdelt"] = g
    with pytest.raises(ValueError, match="2015-06-12"):
        W.build_weekly_panel(raw)


# ------------------------------------------------------------------ other sources

def test_aigpr_publication_lag():
    m = pd.DataFrame({"month": pd.to_datetime(["2020-01-01", "2020-02-01"]), "aigpr_iran": [10.0, 20.0]})
    s = W.aigpr_weekly(m, pd.DatetimeIndex([T("2020-02-28"), T("2020-03-06"), T("2020-04-03")]))
    assert np.isnan(s.loc[T("2020-02-28")])
    assert s.loc[T("2020-03-06")] == 10.0 and s.loc[T("2020-04-03")] == 20.0


def test_missing_vix_stays_nan():
    vix = pd.DataFrame({"date": pd.to_datetime(["2020-01-06"]), "vix": [15.0]})
    s = W.weekly_last(vix, "vix", pd.DatetimeIndex([T("2020-01-10"), T("2020-01-17")]))
    assert s.loc[T("2020-01-10")] == 15.0 and np.isnan(s.loc[T("2020-01-17")])


def test_weekly_mean_uses_all_seven_days():
    gpr = pd.DataFrame({"date": pd.date_range("2020-01-04", "2020-01-10"), "gprd": [1.0, 2, 3, 4, 5, 6, 7]})
    assert W.weekly_mean(gpr, "gprd", pd.DatetimeIndex([T("2020-01-10")])).iloc[0] == 4.0


# ------------------------------------------------------------------ intraday

def test_intraday_weekly_sums_and_saturday_rule():
    days = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
                           "2025-01-11", "2025-01-13"])           # Sat Jan 11 = Friday 22:00 UTC bar
    daily = pd.DataFrame({"date": days, "rv_intraday": [1.0, 2, 3, 4, 5, 6, 7],
                          "n_hours": [22, 23, 23, 23, 21, 1, 23],
                          "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    weeks = pd.DatetimeIndex([T("2025-01-03"), T("2025-01-10"), T("2025-01-17")])
    out = W.intraday_weekly(daily, weeks, first_week="2025-01-10")
    assert out.loc[T("2025-01-03")].isna().all()
    assert out.loc[T("2025-01-10"), "rv_hourly"] == 15 and out.loc[T("2025-01-10"), "n_hours_week"] == 112
    assert out.loc[T("2025-01-17"), "rv_hourly"] == 13 and out.loc[T("2025-01-17"), "n_hours_week"] == 24
    assert out.loc[T("2025-01-17"), "log_rv_hourly"] == pytest.approx(np.log(13))


def test_rv_gk_with_opening_gap():
    daily = pd.DataFrame({"date": pd.to_datetime(["2025-01-03", "2025-01-10"]), "rv_intraday": 1.0, "n_hours": 22,
                          "open": [98.0, 100.0], "high": [100.0, 104.0], "low": [97.0, 98.0], "close": [99.0, 102.0]})
    out = W.intraday_weekly(daily, pd.DatetimeIndex([T("2025-01-10")]), first_week="2025-01-10")
    gk = 1e4 * (np.log(100 / 99) ** 2 + 0.5 * np.log(104 / 98) ** 2 - (2 * np.log(2) - 1) * np.log(102 / 100) ** 2)
    assert out.loc[T("2025-01-10"), "rv_gk"] == pytest.approx(gk)


def test_zero_hour_week_raises():
    daily = pd.DataFrame({"date": pd.to_datetime(["2025-01-06", "2025-01-20"]), "rv_intraday": 1.0, "n_hours": 22,
                          "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    weeks = pd.DatetimeIndex([T("2025-01-10"), T("2025-01-17"), T("2025-01-24")])
    with pytest.raises(ValueError, match="2025-01-17"):
        W.intraday_weekly(daily, weeks, first_week="2025-01-10")


# ------------------------------------------------------------------ full panel

PANEL_COLUMNS = ["date", "price", "r", "rv", "log_rv", "n_days", "iran_share", "iran_gdelt", "iran_gdelt_geo",
                 "gdelt_days", "gdelt_filled", "gpr", "vix", "ovx", "aigpr_iran", "iran_conflict", "iran_coop",
                 "iran_tone", "iran_goldstein", "rv_hourly", "log_rv_hourly", "n_hours_week", "rv_gk"]


def test_full_panel_shape(synthetic_raw):
    p = W.build_weekly_panel(synthetic_raw)
    assert len(p) == 1140 and p["date"].iloc[0] == T("2005-01-07") and p["date"].iloc[-1] == T("2026-11-06")
    assert (p["date"].dt.day_of_week == 4).all() and (p["rv"] > 0).all()
    assert list(p.columns) == PANEL_COLUMNS
    assert p.loc[p["date"] < "2007-05-04", "ovx"].isna().all()      # synthetic OVX starts 2007-05-01
    assert p.loc[p["date"] < C.HOURLY_SAMPLE_START, "rv_hourly"].isna().all()
    assert (p.loc[p["date"] >= C.HOURLY_SAMPLE_START, "rv_hourly"] > 0).all()
    assert p["iran_tone"].eq(-2.0).all()                             # fixture: -20 / 10 every day
    assert p["gdelt_filled"].eq(0).all()


def test_event_weeks():
    wk = {e[0]: W.week_ending(T(e[1])) for e in C.EVENTS}
    assert wk["abqaiq_2019"] == T("2019-09-20") and wk["hormuz_2026"] == T("2026-03-06")
