"""Daily raw data -> one row per Friday week (spec §8.12, §9.1). Used by notebook 02.

The big picture
---------------
Every model works on WEEKS. A week runs Saturday -> Friday and is labelled by its Friday
(`week_ending`). Each daily source is mapped to its week and aggregated:

    Brent spot (FRED)     -> Friday close `price`, weekly return `r`, daily-based RV `rv`, `n_days`
    GDELT (Iran news)     -> weekly Iran share (sum of Iran articles / sum of all articles), conflict and
                             cooperation shares, article-weighted tone and Goldstein, `gdelt_days`
    GPR / VIX / OVX       -> weekly mean (GPR) or last close (VIX, OVX)
    AI-GPR Iran (monthly) -> the last month that was already PUBLISHED by that Friday
    Dukascopy daily RV    -> `rv_hourly` (sum of the days' hourly RV), `rv_gk` (Garman-Klass), `n_hours_week`

Values are NOT lagged here: each row holds what was known at that Friday's close. The one-week lag
the models use is applied later, in design.py (the only other file allowed to shift data).

Nothing is filled silently. A week with no Brent trading, zero Iran articles, or (from the hourly start)
no hourly return raises an error listing the weeks. The one exception, decided with the user, is GDELT's
Jun 2025 outage: see `gdelt_weekly`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brent import config as C

# GDELT count columns that are summed per day and per week (everything except `feed` and `date`).
GDELT_COUNT_COLUMNS = ["n_events", "articles_all", "articles_iran_actor", "articles_iran_actor_geo",
                       "events_iran_actor", "articles_iran_conflict", "articles_iran_coop",
                       "tone_x_articles_iran", "articles_iran_tone_n",
                       "goldstein_x_articles_iran", "articles_iran_goldstein_n"]
# Panel share column -> the GDELT numerator it is built from (denominator is always articles_all).
SHARE_NUMERATORS = {"iran_gdelt": "articles_iran_actor", "iran_gdelt_geo": "articles_iran_actor_geo",
                    "iran_conflict": "articles_iran_conflict", "iran_coop": "articles_iran_coop"}
# Panel mean column -> (weighted sum, weight) for the article-weighted tone and Goldstein means.
MEAN_PARTS = {"iran_tone": ("tone_x_articles_iran", "articles_iran_tone_n"),
              "iran_goldstein": ("goldstein_x_articles_iran", "articles_iran_goldstein_n")}
GDELT_VALUE_COLUMNS = ["iran_share", "iran_gdelt", "iran_gdelt_geo", "iran_conflict", "iran_coop",
                       "iran_tone", "iran_goldstein"]
PANEL_COLUMNS = ["date", "price", "r", "rv", "log_rv", "n_days", "iran_share", "iran_gdelt", "iran_gdelt_geo",
                 "gdelt_days", "gdelt_filled", "gpr", "vix", "ovx", "aigpr_iran", "iran_conflict", "iran_coop",
                 "iran_tone", "iran_goldstein", "rv_hourly", "log_rv_hourly", "n_hours_week", "rv_gk"]


# =====================================================================================
# Week mapping
# =====================================================================================

def week_ending(d):
    """The Friday that ends d's Saturday->Friday week: d + ((4 - weekday) mod 7) days.

    Saturday and Sunday go to the NEXT Friday. That is the plan's weekend rule: news from a
    weekend (e.g. Abqaiq, Sat Sep 14, 2019) belongs to the week in which markets can react.
    Returns the same type it was given (Timestamp, DatetimeIndex or Series).
    """
    if isinstance(d, pd.Timestamp):
        return d + pd.Timedelta(days=(4 - d.weekday()) % 7)
    if isinstance(d, pd.Series):
        return d + pd.to_timedelta((4 - d.dt.day_of_week) % 7, unit="D")
    idx = pd.DatetimeIndex(d)
    return idx + pd.to_timedelta((4 - idx.day_of_week) % 7, unit="D")


def fridays(start: str = C.SAMPLE_START, end: str = C.FREEZE_DATE) -> pd.DatetimeIndex:
    """All Fridays from start to end (inclusive): the panel's rows (1,140 for the full sample)."""
    return pd.date_range(start, end, freq="W-FRI")


def _list(weeks) -> str:
    """Week dates as 'YYYY-MM-DD, ...' for error messages."""
    return ", ".join(pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks)


# =====================================================================================
# Brent, GPR, VIX, OVX, AI-GPR
# =====================================================================================

def weekly_brent(brent: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Brent per week: Friday close, weekly return, daily-based realized variance, trading days.

    price  = close on the week's last trading day (Friday, or earlier if Friday is a holiday)
    r      = 100 x ln(price / previous week's price)
    rv     = sum over the week's trading days of (100 x daily log return)^2. The first daily return of a
             week uses the previous trading day's close, so the week's returns add up to r exactly.
    n_days = number of trading days in the week (0 if none: build_weekly_panel then raises)
    """
    b = brent.sort_values("date")
    p = b["brent"].to_numpy(float)
    lr = np.concatenate([[np.nan], 100 * np.diff(np.log(p))])          # daily log returns, in %
    g = pd.DataFrame({"week": week_ending(b["date"]).to_numpy(), "price": p, "r2": lr**2}).groupby("week")
    price = g["price"].last()
    # Weekly return over CONSECUTIVE calendar weeks (a week without trading gives NaN, never a 2-week return).
    full = fridays(price.index.min(), price.index.max())
    lp = np.log(price.reindex(full))
    r = 100 * lp.diff()
    out = pd.DataFrame({"price": price.reindex(weeks), "r": r.reindex(weeks),
                        "rv": g["r2"].sum(min_count=1).reindex(weeks),
                        "n_days": g.size().reindex(weeks).fillna(0).astype(int)})
    out.index.name = "date"
    return out


def weekly_last(daily: pd.DataFrame, col: str, weeks) -> pd.Series:
    """Last observation of `col` within each Sat->Fri week (e.g. VIX at the Friday close); NaN if none."""
    d = daily.sort_values("date")
    return d.groupby(week_ending(d["date"]).to_numpy())[col].last().reindex(weeks)


def weekly_mean(daily: pd.DataFrame, col: str, weeks) -> pd.Series:
    """Mean of `col` over the week's observations (GPR: all 7 days, Sat->Fri); NaN if none."""
    return daily.groupby(week_ending(daily["date"]).to_numpy())[col].mean().reindex(weeks)


def aigpr_weekly(monthly: pd.DataFrame, weeks, lag_months: int = C.AIGPR_PUBLICATION_LAG_MONTHS) -> pd.Series:
    """AI-GPR Iran known at each Friday: month m counts as published from the first Friday on or after the
    1st of month m + 1 + lag_months (so January's value is usable from the first Friday of March).

    Carrying the last published month forward IS the definition here (it is what a forecaster had),
    not a fill of missing data.
    """
    m = monthly.sort_values("month")
    avail = week_ending(m["month"] + pd.DateOffset(months=1 + lag_months))   # first Friday it is known
    s = pd.Series(m["aigpr_iran"].to_numpy(float), index=pd.DatetimeIndex(avail))
    weeks = pd.DatetimeIndex(weeks)
    return s.reindex(s.index.union(weeks)).ffill().reindex(weeks)


# =====================================================================================
# GDELT
# =====================================================================================

def _gdelt_daily_totals(gd: pd.DataFrame, switch: str = C.GDELT_SWITCH_DATE) -> tuple[pd.DataFrame, dict]:
    """GDELT 1.0 rows -> one row of summed counts per calendar day, plus what was dropped or merged.

    - 'stale' rows (event date > 1 year before it was added) are counted as daily feed
      (GDELT_STALE_POLICY = "merge": in Jan 2020 they are current news with a broken event date).
    - 'backfile' rows are kept only before the Apr 1, 2013 switch, 'daily' rows only from it on.
      Anything on the wrong side is dropped and counted (`wrong_side`).
    """
    stale = gd["feed"] == "stale"
    feed = gd["feed"].where(~stale, "daily") if C.GDELT_STALE_POLICY == "merge" else gd["feed"]
    sw = pd.Timestamp(switch)
    keep = ((feed == "backfile") & (gd["date"] < sw)) | ((feed == "daily") & (gd["date"] >= sw))
    counts = {"wrong_side": int((~keep & ~(stale & (C.GDELT_STALE_POLICY != "merge"))).sum()),
              "stale_rows": int(stale.sum()), "stale_events": int(gd.loc[stale, "n_events"].sum())}
    daily = gd.loc[keep].groupby("date")[GDELT_COUNT_COLUMNS].sum()
    return daily, counts


def _weekly_sums(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-week sums of the daily GDELT counts, plus `days` = number of days with data in that week."""
    wk = week_ending(pd.DatetimeIndex(daily.index))
    g = daily.groupby(wk)
    out = g[GDELT_COUNT_COLUMNS].sum()
    out["days"] = g.size()
    return out


def _v2_calibration(d1: pd.DataFrame, d2: pd.DataFrame, days=C.GDELT_V2_PATCH_DAYS) -> tuple[pd.DataFrame, float]:
    """How GDELT 2.0 relates to 1.0, measured on the weeks both cover fully (spec §6.4 splice rule).

    Overlap weeks: 7 days of data in BOTH versions and not touching the outage days.
    Returns a table (one row per series: shift, corr, n_weeks, pass) and `vol`, the ratio of
    1.0's to 2.0's total article volume over the overlap days.
      - shares (Iran, geo, conflict, coop): shift = mean(weekly log share 1.0 - weekly log share 2.0)
      - tone, Goldstein: shift = mean(weekly mean 1.0 - weekly mean 2.0)   (levels)
      - corr = correlation of the two weekly series; pass = corr >= GDELT_SPLICE_MIN_CORR
      - weight_ratio (tone, Goldstein only) = 1.0 / 2.0 ratio of their article weights over the overlap
    """
    p0, p1 = pd.Timestamp(days[0]), pd.Timestamp(days[1])
    w1, w2 = _weekly_sums(d1), _weekly_sums(d2)
    weeks = w1.index.intersection(w2.index)
    touches = (weeks >= p0) & (weeks - pd.Timedelta(days=6) <= p1)
    full = (w1.loc[weeks, "days"] == 7).to_numpy() & (w2.loc[weeks, "days"] == 7).to_numpy()
    ov = weeks[full & ~touches]
    a, b = w1.loc[ov], w2.loc[ov]
    rows = {}
    for num in SHARE_NUMERATORS.values():
        x, y = np.log(a[num] / a["articles_all"]), np.log(b[num] / b["articles_all"])
        rows[num] = (float((x - y).mean()), float(np.corrcoef(x, y)[0, 1]) if len(ov) > 2 else np.nan)
    for name, (sx, sn) in MEAN_PARTS.items():
        x, y = a[sx] / a[sn], b[sx] / b[sn]
        rows[name] = (float((x - y).mean()), float(np.corrcoef(x, y)[0, 1]) if len(ov) > 2 else np.nan)
    info = pd.DataFrame(rows, index=["shift", "corr"]).T
    # Weight ratio for each mean: tone/Goldstein are weighted by IRAN articles, whose 1.0/2.0 ratio differs
    # from the all-articles ratio. Using it keeps a patched day's weight 1.0-sized inside a mixed week.
    info["weight_ratio"] = np.nan
    for name, (_, sn) in MEAN_PARTS.items():
        info.loc[name, "weight_ratio"] = float(a[sn].sum() / b[sn].sum())
    info["n_weeks"] = len(ov)
    info["pass"] = info["corr"] >= C.GDELT_SPLICE_MIN_CORR
    d_ov = week_ending(pd.DatetimeIndex(d1.index)).isin(ov)
    vol = float(d1.loc[d_ov, "articles_all"].sum() / d2.loc[week_ending(pd.DatetimeIndex(d2.index)).isin(ov),
                                                            "articles_all"].sum())
    return info, vol


def v2_patch_info(gd: pd.DataFrame, v2: pd.DataFrame, days=C.GDELT_V2_PATCH_DAYS) -> pd.DataFrame:
    """The calibration table for notebook 02: shift and overlap correlation per series (raw frames in)."""
    d1, _ = _gdelt_daily_totals(gd)
    d2 = v2.groupby("date")[GDELT_COUNT_COLUMNS].sum()              # 'v2' and 'stale' rows both count
    info, vol = _v2_calibration(d1, d2, days)
    info.attrs["vol"] = vol
    return info


def _v2_patch_rows(d1: pd.DataFrame, d2: pd.DataFrame, days=C.GDELT_V2_PATCH_DAYS) -> pd.DataFrame:
    """GDELT 2.0 days that fill 1.0's outage, rescaled so they look like 1.0 days.

    Only outage days where 1.0 has NO row and 2.0 has one are used (in the real data: Jun 14, 2025).
    Each patched day: articles_all = 2.0 articles_all x vol (1.0-sized weight in its week);
    share numerators = 2.0 numerator x exp(shift) x vol (so the share = 2.0 share x exp(shift));
    tone/Goldstein: weight x its own weight_ratio, and the day's mean is moved by the level shift.
    Raises if the main Iran share of the two versions correlates below GDELT_SPLICE_MIN_CORR.
    """
    info, vol = _v2_calibration(d1, d2, days)
    main = info.loc["articles_iran_actor"]
    if not main["corr"] >= C.GDELT_SPLICE_MIN_CORR:
        raise ValueError(f"GDELT 2.0 patch refused: weekly Iran-share correlation {main['corr']:.3f} with 1.0 "
                         f"is below {C.GDELT_SPLICE_MIN_CORR} ({int(main['n_weeks'])} overlap weeks)")
    span = pd.date_range(days[0], days[1])
    todo = span[~span.isin(d1.index) & span.isin(d2.index)]
    src = d2.loc[todo]
    rows = pd.DataFrame(index=todo, columns=GDELT_COUNT_COLUMNS, dtype=float)
    for col in ("n_events", "articles_all", "events_iran_actor"):
        rows[col] = src[col] * vol
    for num in SHARE_NUMERATORS.values():
        rows[num] = src[num] * np.exp(info.loc[num, "shift"]) * vol
    for name, (sx, sn) in MEAN_PARTS.items():
        rows[sn] = src[sn] * info.loc[name, "weight_ratio"]
        rows[sx] = (src[sx] / src[sn] + info.loc[name, "shift"]) * rows[sn]
    rows.index.name = "date"
    return rows


def gdelt_weekly(gd: pd.DataFrame, weeks, switch: str = C.GDELT_SWITCH_DATE,
                 v2: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Weekly Iran-news measures from the daily GDELT rows.

    Steps:
      1. Pick the right feed per day (backfile before Apr 2013, daily after; stale rows merged).
      2. If `v2` (GDELT 2.0 rows) is given: fill 1.0's outage days from 2.0, level-shifted (decided 2026-11-23).
      3. Sum numerators and denominators per week, THEN divide: share = sum(Iran articles) / sum(all
         articles). This weights busy news days more than an average of daily shares would.
      4. A week with no GDELT day at all (only the week ending 2025-06-27) takes the previous week's values
         (GDELT_EMPTY_WEEK_POLICY = "locf": what a forecaster had while GDELT was down; no look-ahead) and
         gets gdelt_filled = 1. More than GDELT_MAX_EMPTY_WEEKS empty weeks in a row, or an empty first
         week, raise instead.
      5. A non-empty week with zero Iran / conflict / cooperation articles, or zero toned articles,
         raises (a log of zero is impossible and nothing is offset).

    Columns: iran_share (level), iran_gdelt / iran_gdelt_geo / iran_conflict / iran_coop (logs of shares),
    iran_tone / iran_goldstein (article-weighted means, levels), gdelt_days (true count), gdelt_filled (0/1).
    The dict counts wrong_side, stale_rows, stale_events, patched_days, filled_weeks.
    """
    weeks = pd.DatetimeIndex(weeks)
    daily, counts = _gdelt_daily_totals(gd, switch)
    patched = 0
    if v2 is not None:
        d2 = v2.groupby("date")[GDELT_COUNT_COLUMNS].sum()
        rows = _v2_patch_rows(daily, d2)
        patched = len(rows)
        daily = pd.concat([daily, rows]).sort_index()
    sums = _weekly_sums(daily).reindex(weeks)
    days = sums["days"].fillna(0).astype(int)

    out = pd.DataFrame(index=weeks)
    out["iran_share"] = sums["articles_iran_actor"] / sums["articles_all"]
    with np.errstate(divide="ignore"):                     # zero numerators are caught below, not logged
        for col, num in SHARE_NUMERATORS.items():
            out[col] = np.log(sums[num] / sums["articles_all"])
    for col, (sx, sn) in MEAN_PARTS.items():
        out[col] = sums[sx] / sums[sn]

    # Step 4: empty weeks (no GDELT day at all)
    empty = days == 0
    if empty.iloc[0]:
        raise ValueError(f"GDELT: the first week has no data ({_list(weeks[:1])}); nothing to carry forward")
    streak_id = (~empty).cumsum()                                  # consecutive empty weeks share an id
    streak_len = empty.astype(int).groupby(streak_id).transform("sum")
    too_long = weeks[(empty & (streak_len > C.GDELT_MAX_EMPTY_WEEKS)).to_numpy()]
    if len(too_long):
        raise ValueError(f"GDELT: more than {C.GDELT_MAX_EMPTY_WEEKS} empty week(s) in a row: "
                         f"{_list(too_long)}; not filled")
    if C.GDELT_EMPTY_WEEK_POLICY != "locf" and empty.any():
        raise ValueError(f"GDELT: weeks with no data: {_list(weeks[empty.to_numpy()])}")
    out.loc[empty, GDELT_VALUE_COLUMNS] = np.nan
    out.loc[empty, GDELT_VALUE_COLUMNS] = out[GDELT_VALUE_COLUMNS].ffill().loc[empty]
    out["gdelt_days"] = days
    out["gdelt_filled"] = empty.astype(int)

    # Step 5: zero numerators or weights in weeks that do have data
    zero = pd.Series(False, index=weeks)
    for num in list(SHARE_NUMERATORS.values()) + [sn for _, sn in MEAN_PARTS.values()]:
        zero |= (sums[num] == 0) & ~empty
    if zero.any():
        raise ValueError(f"GDELT: weeks with zero Iran articles in at least one measure: {_list(weeks[zero.to_numpy()])}")

    counts.update(patched_days=patched, filled_weeks=int(empty.sum()))
    out.index.name = "date"
    return out, counts


# =====================================================================================
# Intraday (Dukascopy) -> weekly hourly RV
# =====================================================================================

def intraday_weekly(daily: pd.DataFrame, weeks, first_week: str = C.HOURLY_SAMPLE_START) -> pd.DataFrame:
    """Weekly hourly-RV measures from the daily Dukascopy file.

    rv_hourly    = sum of the days' rv_intraday (sum of squared hourly returns); a Saturday row (Friday
                   22:00 UTC bar) goes to the NEXT week via week_ending, so a week runs Fri 22:00 -> Fri 22:00 UTC.
    n_hours_week = number of hourly returns in the week
    rv_gk        = sum over days of the Garman-Klass variance plus the opening gap, in %^2:
                   1e4 x [(ln O_d - ln C_{d-1})^2 + 0.5 (ln H_d - ln L_d)^2 - (2 ln 2 - 1)(ln C_d - ln O_d)^2]
                   with C_{d-1} the previous daily row's close.
    All four are NaN before `first_week` (decided: 2013-06-07, the start of clean data). From it on, a week
    with no hourly return raises: its overnight/weekend gap return would otherwise land in the next week.
    """
    d = daily.sort_values("date").reset_index(drop=True)
    lo, lh, ll, lc = (np.log(d[c].to_numpy(float)) for c in ("open", "high", "low", "close"))
    prev_lc = np.log(d["close"].shift().to_numpy(float))            # previous day's close (shift allowed here)
    gk = 1e4 * ((lo - prev_lc) ** 2 + 0.5 * (lh - ll) ** 2 - (2 * np.log(2) - 1) * (lc - lo) ** 2)
    f = pd.DataFrame({"week": week_ending(d["date"]).to_numpy(), "rv": d["rv_intraday"].to_numpy(float),
                      "n": d["n_hours"].to_numpy(float), "gk": gk})
    g = f.groupby("week")
    weeks = pd.DatetimeIndex(weeks)
    rv = g["rv"].sum().reindex(weeks)
    n = g["n"].sum().reindex(weeks)
    gk_w = g["gk"].sum().where(g["gk"].count() == g.size()).reindex(weeks)   # NaN if any day lacks C_{d-1}
    before = weeks < pd.Timestamp(first_week)
    # Mask the weeks before the hourly start FIRST (the broken 2012 data has zero-RV weeks; no log of 0).
    rv, gk_w = rv.mask(before), gk_w.mask(before)
    out = pd.DataFrame({"rv_hourly": rv, "log_rv_hourly": np.log(rv), "n_hours_week": n.mask(before),
                        "rv_gk": gk_w}, index=weeks)
    bad = weeks[~before & (n.isna() | (n == 0)).to_numpy()]
    if len(bad):
        raise ValueError(f"intraday: weeks on/after {first_week} with no hourly return: {_list(bad)}")
    out.index.name = "date"
    return out


# =====================================================================================
# The panel
# =====================================================================================

def build_weekly_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All sources -> data/processed/weekly_panel.csv (one row per Friday, 2005-01-07 -> 2026-11-06).

    raw keys: brent, vix, ovx, gpr, aigpr, gdelt, intraday, and optionally gdelt_v2 (the 2025 patch file).
    Logs are taken for iran_gdelt(_geo), iran_conflict, iran_coop, gpr, vix, ovx, aigpr_iran and rv
    (models use logs); iran_tone and iran_goldstein stay levels. NaNs (e.g. OVX before May 2007, hourly RV
    before Jun 2013, a missing VIX day) stay NaN; nothing is filled except GDELT's one empty week.
    GDELT counts (stale merged, patched days, filled weeks) are kept in panel.attrs["gdelt_counts"].
    """
    weeks = fridays()
    b = weekly_brent(raw["brent"], weeks)
    if (b["n_days"] == 0).any():
        raise ValueError(f"Brent: weeks with no trading day: {_list(weeks[(b['n_days'] == 0).to_numpy()])}")
    gd, counts = gdelt_weekly(raw["gdelt"], weeks, v2=raw.get("gdelt_v2"))
    aig = aigpr_weekly(raw["aigpr"], weeks)
    if (aig <= 0).any():
        raise ValueError(f"AI-GPR: values <= 0 in weeks: {_list(weeks[(aig <= 0).to_numpy()])}")
    intr = intraday_weekly(raw["intraday"], weeks)

    p = pd.DataFrame(index=weeks)
    p["price"], p["r"], p["rv"] = b["price"], b["r"], b["rv"]
    p["log_rv"] = np.log(b["rv"])
    p["n_days"] = b["n_days"]
    for col in ("iran_share", "iran_gdelt", "iran_gdelt_geo", "gdelt_days", "gdelt_filled"):
        p[col] = gd[col]
    p["gpr"] = np.log(weekly_mean(raw["gpr"], "gprd", weeks))
    p["vix"] = np.log(weekly_last(raw["vix"], "vix", weeks))
    p["ovx"] = np.log(weekly_last(raw["ovx"], "ovx", weeks))
    p["aigpr_iran"] = np.log(aig)
    for col in ("iran_conflict", "iran_coop", "iran_tone", "iran_goldstein"):
        p[col] = gd[col]
    for col in ("rv_hourly", "log_rv_hourly", "n_hours_week", "rv_gk"):
        p[col] = intr[col]
    p.index.name = "date"
    p = p.reset_index()[PANEL_COLUMNS]
    p.attrs["gdelt_counts"] = counts
    return p
