"""Design matrices (spec §8.5, §9.3). The only place (with weekly.py) where lags and z-scores happen.

Row t of X and Z holds values dated t-1; x_next / z_next hold values dated at the origin T,
for the forecast of week T+1. Rows dated after `end` are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from brent import config as C
from brent import paths

if TYPE_CHECKING:
    from brent.ms import MSSpec

WEEK = pd.Timedelta(days=7)


@dataclass
class Design:
    dates: pd.DatetimeIndex
    y: pd.Series
    X: pd.DataFrame
    Z: pd.DataFrame | None
    x_next: pd.Series
    z_next: pd.Series | None
    origin: pd.Timestamp
    std_stats: pd.DataFrame


def _indexed(panel: pd.DataFrame, end) -> pd.DataFrame:
    """Panel indexed by Friday date, cut at `end` (later rows are dropped, never read)."""
    P = panel.copy()
    P.index = pd.DatetimeIndex(pd.to_datetime(P["date"]))
    P = P.sort_index()
    if pd.Timestamp(end) not in P.index:
        raise ValueError(f"end {pd.Timestamp(end).date()} is not a date in the panel")
    P = P.loc[:pd.Timestamp(end)]
    if len(P) > 1 and not (P.index.to_series().diff().dropna() == WEEK).all():
        raise ValueError("panel dates are not consecutive weeks")
    return P


def _check_nan(P: pd.DataFrame, cols, start) -> None:
    """Raise if any used column has a NaN on [start, end]."""
    cols = list(dict.fromkeys(cols))
    sub = P.loc[pd.Timestamp(start):, cols]
    for c in cols:
        bad = sub.index[sub[c].isna()]
        if len(bad):
            raise ValueError(f"NaN in '{c}' on {len(bad)} weeks, first {bad[0].date()}, last {bad[-1].date()}")


def standardization_stats(panel: pd.DataFrame, vars, window) -> pd.DataFrame:
    """Mean and sd (ddof=1) of each var over `window`; `dummy_*` columns pass through (mean 0, sd 1)."""
    P = panel.copy()
    P.index = pd.DatetimeIndex(pd.to_datetime(P["date"]))
    w = P.loc[pd.Timestamp(window[0]):pd.Timestamp(window[1])]
    rows = {}
    for v in vars:
        if v.startswith("dummy_"):
            rows[v] = (0.0, 1.0)
            continue
        if len(w) < 2 or w[v].isna().any():
            raise ValueError(f"'{v}' has NaN or fewer than 2 rows in window {tuple(window)}")
        rows[v] = (w[v].mean(), w[v].std(ddof=1))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["mean", "sd"])


def _z(P: pd.DataFrame, vars, stats: pd.DataFrame) -> pd.DataFrame:
    """Standardize the columns with fixed stats."""
    out = pd.DataFrame(index=P.index)
    for v in vars:
        out[v] = (P[v] - stats.loc[v, "mean"]) / stats.loc[v, "sd"]
    return out


def build_design(panel: pd.DataFrame, spec: MSSpec, end) -> Design:
    """Design for an MS-AR spec with data up to the origin `end`."""
    if spec.kind == "mshar":
        raise NotImplementedError("E1 is built in Plan 3")
    P = _indexed(panel, end)
    tv, mx = tuple(spec.tvtp_vars), tuple(spec.mean_exog)
    _check_nan(P, ["r", *tv, *mx], spec.start)
    P = P.loc[pd.Timestamp(spec.start):]
    stat_vars = list(dict.fromkeys([*tv, *mx]))
    stats = standardization_stats(panel, stat_vars, spec.std_window)
    zs = _z(P, stat_vars, stats)
    lagged = pd.DataFrame({"r_lag": P["r"], **{v: zs[v] for v in mx}}).shift(1)
    lagged_z = zs[list(tv)].shift(1)
    dates = P.index[1:]
    X = lagged.loc[dates]
    Z = None
    z_next = None
    if tv:
        Z = lagged_z.loc[dates].copy()
        Z.insert(0, "const", 1.0)
        z_next = pd.concat([pd.Series({"const": 1.0}), zs[list(tv)].iloc[-1]])
    x_next = pd.Series({"r_lag": P["r"].iloc[-1], **{v: zs[v].iloc[-1] for v in mx}})
    return Design(dates=dates, y=P["r"].loc[dates], X=X, Z=Z, x_next=x_next, z_next=z_next,
                  origin=P.index[-1], std_stats=stats)


def har_design(panel: pd.DataFrame, end, rv_col: str = "rv", x_vars: tuple[str, ...] = (),
               start: str | None = None, std_window=None):
    """HAR(-X) rows indexed by target week s: const, rv_1, rv_4, rv_13 (+ z-scored x_vars, lag 1).

    start / std_window default to the config values at call time.
    """
    start = C.SAMPLE_START if start is None else start
    std_window = C.STANDARDIZE_WINDOW if std_window is None else std_window
    x_vars = tuple(x_vars)
    P = _indexed(panel, end)
    _check_nan(P, [rv_col, *x_vars], start)
    P = P.loc[pd.Timestamp(start):]
    rv = P[rv_col]
    # regressors dated s: value at s, mean of s-3..s, mean of s-12..s (they enter row s+1)
    reg = pd.DataFrame({"rv_1": rv, "rv_4": rv.rolling(4).mean(), "rv_13": rv.rolling(13).mean()})
    if x_vars:
        stats = standardization_stats(panel, x_vars, std_window)
        reg = pd.concat([reg, _z(P, x_vars, stats)], axis=1)
    rows = reg.shift(1).iloc[13:]
    rows.insert(0, "const", 1.0)
    x_next = pd.concat([pd.Series({"const": 1.0}), reg.iloc[-1]])
    return rows, rv.loc[rows.index], x_next


def garchx_design(panel: pd.DataFrame, end, x_vars: tuple[str, ...], rv_col: str | None = None,
                  start: str | None = None, scale_window=None):
    """GARCH-X inputs: returns r, nonnegative regressors dated t-1 (rv_col raw, exp(x)/scale mean), x_next.

    rv_col / start / scale_window default to the config values at call time.
    """
    rv_col = C.X_RV_COL if rv_col is None else rv_col
    start = C.X_SAMPLE_START if start is None else start
    scale_window = C.GARCHX_SCALE_WINDOW if scale_window is None else scale_window
    x_vars = tuple(x_vars)
    P = _indexed(panel, end)
    _check_nan(P, ["r", rv_col, *x_vars], start)
    full = _indexed(panel, panel["date"].max())
    w = full.loc[pd.Timestamp(scale_window[0]):pd.Timestamp(scale_window[1])]
    P = P.loc[pd.Timestamp(start):]
    lev = pd.DataFrame({rv_col: P[rv_col]})
    for v in x_vars:
        lev[v] = np.exp(P[v]) / np.exp(w[v]).mean()
    rows = lev.shift(1).iloc[1:]
    return P["r"].loc[rows.index], rows, lev.iloc[-1]


def freeze_std_stats(panel: pd.DataFrame) -> Path:
    """Write results/frozen/std_stats.csv: TRAIN, OVX, HOURLY z-score stats and GARCH-X scale means."""
    tv = list(dict.fromkeys([*C.TVTP_CONTROLS, C.IRAN_VAR, *C.IRAN_TONE_VARS, "iran_tone", "iran_goldstein",
                             "aigpr_iran"]))
    hv = ["log_rv_hourly", "gpr", "vix", "iran_gdelt"]
    jobs = [("TRAIN", C.TRAIN, tv), ("HOURLY", C.STANDARDIZE_WINDOW_HOURLY, hv)]
    if "ovx" in panel.columns:
        jobs.insert(1, ("OVX", C.OVX_STD_WINDOW, ["ovx"]))
    frames = []
    for label, win, vs in jobs:
        s = standardization_stats(panel, vs, win)
        frames.append(s.rename_axis("var").reset_index().assign(window=label, window_start=win[0],
                                                                  window_end=win[1]))
    lv = panel.copy()
    for v in ("gpr", "vix", "iran_gdelt"):
        lv[v] = np.exp(lv[v])
    s = standardization_stats(lv, ["gpr", "vix", "iran_gdelt"], C.GARCHX_SCALE_WINDOW)
    w = C.GARCHX_SCALE_WINDOW
    frames.append(s.rename_axis("var").reset_index().assign(window="GARCHX_SCALE", window_start=w[0],
                                                              window_end=w[1]))
    out = pd.concat(frames, ignore_index=True)[["var", "window", "window_start", "window_end", "mean", "sd"]]
    paths.FROZEN.mkdir(parents=True, exist_ok=True)
    path = paths.FROZEN / "std_stats.csv"
    out.to_csv(path, index=False)
    return path
