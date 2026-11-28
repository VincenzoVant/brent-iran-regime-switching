"""Expanding-window out-of-sample loop with checkpoints, resume, assembly and guards (spec §8.8, §9.2, §9.4).

Checkpoints: results/checkpoints/<run>_{forecasts,params,fitlog}.csv, appended after every origin (in that
order); the fitlog rows mark an origin as complete. Every row carries `settings_hash`.

Benchmarks that cannot run yet at an origin (see `first_origin`) write no forecast or params rows; their
fitlog row has `converged = False`, the other fit columns empty and
`restart_reason = "skipped (first origin YYYY-MM-DD)"`, so every origin still has one fitlog row per model.
Count non-converged fits with the skip rows excluded.

A "degenerate" flag triggers a restart only if the model's fit at the previous origin was not degenerate
(or on a scheduled restart origin); a degeneracy that persists is logged, not restarted every week.

A fit that raises a numerical error does not stop the run. MS model: the restart path is tried; if it fails
too, last week's parameters are carried forward (converged = False, degenerate = True,
restart_reason = "fit_failed"). Benchmark: last week's forecast and params are reused (converged = False,
restart_reason = "fit_failed (previous forecast reused)"), or no row is written if there is none yet
("fit_failed (no forecast)"). Data errors (NaN regressor, missing date) still stop the run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from brent import benchmarks as bench
from brent import config as C
from brent import design
from brent import forecast
from brent import ms
from brent import paths

FIRST_ORIGIN = pd.Timestamp("2012-12-28")
WEEK = pd.Timedelta(days=7)
FC_COLS = ["date", "origin", "split", "run", "model", "target", "forecast"]
PARAM_COLS = ["origin", "model", "name", "value", "settings_hash"]
FITLOG_COLS = ["origin", "model", "settings_hash", "loglike", "converged", "n_restarts_used", "relabeled",
               "refit_from_scratch", "degenerate", "nesting_ok", "seconds", "sigma2_N", "sigma2_H", "dur_N",
               "dur_H", "xi_H_origin", "har_floored", "restart_reason", "ll_gain_from_restart", "n_warnings",
               "at_cap"]
HAR_HOURLY = ("HAR_h", "HARX_C", "HARX_CI")
GARCHX = ("GARCHX_C", "GARCHX_CI")
VAR_ONLY = ("GARCH_t", "HAR", "HAR_h", "HARX_C", "HARX_CI", "GARCHX_C", "GARCHX_CI")


def origins(start_target: str, end_target: str) -> list[pd.Timestamp]:
    """Origins (target Friday minus 7 days) for every target Friday in [start_target, end_target]."""
    return list(pd.date_range(start_target, end_target, freq="W-FRI") - WEEK)


_origins = origins                                   # run_oos has an argument named `origins`


def origin_index(origin) -> int:
    """Position of the origin in the full origin list (2012-12-28 = 0); the seed is C.SEED + this."""
    return (pd.Timestamp(origin) - FIRST_ORIGIN).days // 7


def first_origin(model: str) -> pd.Timestamp | None:
    """First origin at which a benchmark can be fitted without lookahead; None = no restriction.

    HAR_h / HARX_*: har_design must give at least C.HAR_MIN_ROWS rows (the first row comes 13 weeks after
    C.X_SAMPLE_START) and the floor window must be non-empty. GARCHX_*: the scale window C.GARCHX_SCALE_WINDOW must lie before
    the origin. Config is read at call time.
    """
    if model in HAR_HOURLY:
        start = pd.Timestamp(C.X_SAMPLE_START)
        win = C.TRAIN if C.X_RV_COL == "rv" else C.STANDARDIZE_WINDOW_HOURLY
        return max(start + (13 + C.HAR_MIN_ROWS - 1) * WEEK, pd.Timestamp(win[0]))
    if model in GARCHX:
        return pd.Timestamp(C.GARCHX_SCALE_WINDOW[1])
    return None


def settings_hash(ms_specs, benchmarks, iran_var) -> str:
    """First 16 hex chars of the SHA-256 of the frozen settings (not the origins, SPEC_FROZEN or run_id)."""
    d = {k: getattr(C, k) for k in ("SEARCH_REPS_FIRST", "N_RESTARTS", "RESTART_EVERY", "SIGMA2_RATIO_MIN",
                                    "VAR_FLOOR_FRAC", "MIN_DURATION", "MAX_TVTP_COEF", "CONTROLS_ONLY_NH",
                                    "M3_VARIANT", "PHI_SWITCHING", "INTRADAY_ROLL_RULE",
                                    "INTRADAY_REDISTRIBUTE", "X_RV_COL", "X_SAMPLE_START",
                                    "HOURLY_SAMPLE_START", "STANDARDIZE_WINDOW_HOURLY", "TRAIN",
                                    "GARCHX_SCALE_WINDOW", "X_CONTROLS", "SEED", "HIGHVOL_PCTL",
                                    "RV_FLOOR_FRAC", "VALID", "TEST", "SAMPLE_START", "STANDARDIZE_WINDOW",
                                    "FREEZE_DATE", "HAR_MIN_ROWS")}
    d["iran_var"] = iran_var
    d["tvtp_slopes"] = "bounded_at_MAX_TVTP_COEF"         # bounded MLE (2026-11-27); invalidates older checkpoints
    d["specs"] = sorted((dataclasses.asdict(s) for s in ms_specs), key=lambda a: a["name"])
    d["benchmarks"] = list(benchmarks)
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sanity_flags(fit, ll_prev_on_current: float | None, ll_parent: float | None, ratio_min: float) -> list[str]:
    """Reasons to restart a warm-start fit (empty list = the fit looks fine)."""
    flags = []
    if not fit.converged:
        flags.append("not_converged")
    if ll_prev_on_current is not None and fit.loglike < ll_prev_on_current - 1e-8:
        flags.append("ll_below_start")
    if not fit.params["sigma2[1]"] / fit.params["sigma2[0]"] >= ratio_min:
        flags.append("sigma2_ratio")
    if fit.degenerate:
        flags.append("degenerate")
    if ll_parent is not None and fit.loglike < ll_parent - 1e-6:
        flags.append("nesting")
    return flags


def _split(target: pd.Timestamp) -> str:
    """validation / test / holdout by target date."""
    if target <= pd.Timestamp(C.VALID[1]):
        return "validation"
    return "test" if target <= pd.Timestamp(C.TEST[1]) else "holdout"


def _ckpt(run_id: str) -> dict:
    """Checkpoint file paths."""
    return {k: paths.CHECKPOINTS / f"{run_id}_{k}.csv" for k in ("forecasts", "params", "fitlog")}


def _append(path, df: pd.DataFrame) -> None:
    """Append rows (header only when the file is new), then flush and fsync."""
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        df.to_csv(fh, header=new, index=False)
        fh.flush()
        os.fsync(fh.fileno())


def _rewrite(path, df: pd.DataFrame) -> None:
    """Replace a checkpoint atomically (write a temp file, fsync, os.replace)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        df.to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_text(path) -> pd.DataFrame:
    """Read a checkpoint as text, so a rewrite leaves every value byte-identical.

    A last line without a newline is a torn write (crash mid-append): it is dropped and attrs["torn"] is set.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    with open(path, "rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        if size:
            fh.seek(-1, os.SEEK_END)
        torn = bool(size) and fh.read(1) != b"\n"
    if torn and len(df):
        df = df.iloc[:-1].copy()
    df.attrs["torn"] = bool(torn)
    return df


def _check_finite(model: str, T: pd.Timestamp, values: dict) -> None:
    """A non-finite forecast stops the run (e.g. a missing regressor at the origin)."""
    bad = [k for k, v in values.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite forecast {bad} for {model} at origin {T.date()}")


FIT_ERRORS = (RuntimeError, ValueError, ArithmeticError, np.linalg.LinAlgError)
_DATA_ERROR_TEXT = ("NaN", "no finite", "not a date", "consecutive")


def _is_data_error(e: Exception) -> bool:
    """Data problems (NaN regressor, missing date) must stop the run; numerical fit failures must not."""
    return isinstance(e, ValueError) and any(t in str(e) for t in _DATA_ERROR_TEXT)


def _carry_forward(dsn, spec, start, n_warn):
    """Last week's params as this week's fit after every candidate failed (flagged, never hidden)."""
    model = ms.make_model(dsn, spec)
    p = start.reindex(list(model.param_names)).astype(float)
    try:
        with np.errstate(all="ignore"):
            ll = float(model.loglike(p.values))
    except FIT_ERRORS:
        ll = np.nan
    fixed = ms.constraint_info(spec, model)["fixed"] if spec.kind == "msar" else {}
    return ms.MSFit(spec, p, ll, False, C.N_RESTARTS, False, False, True, 0.0, model, dict(fixed), n_warn)


def _fit_ms_origin(panel, spec, T, seed, start, parent_fit, idx, prev_degenerate=False):
    """Warm fit, sanity checks, restart if scheduled or flagged; returns (fit, design, log extras).

    A "degenerate" flag alone does not trigger a restart when the previous origin's fit was degenerate too.
    """
    try:
        dsn = design.build_design(panel, spec, T)
    except ValueError as e:
        raise ValueError(f"{spec.name} at origin {T.date()}: {e}") from e
    nest = ()
    if parent_fit is not None and start is None:
        nest = (ms.nest_start(parent_fit, ms.make_model(dsn, spec), spec),)
    reps0 = 0 if start is not None else C.SEARCH_REPS_FIRST
    try:
        warm = ms.fit_ms(dsn, spec, start, reps0, 0, seed, extra_starts=nest)
    except FIT_ERRORS as e:
        if _is_data_error(e):
            raise
        # warm fit failed: try the restart path (with the nesting start); else carry last week forward
        extra = nest or ((ms.nest_start(parent_fit, ms.make_model(dsn, spec), spec),)
                         if parent_fit is not None else ())
        try:
            fit = ms.fit_ms(dsn, spec, start, reps0, C.N_RESTARTS, seed, extra_starts=extra)
            reason = "fit_failed+restart"
        except FIT_ERRORS as e2:
            if _is_data_error(e2) or start is None:
                raise RuntimeError(f"{spec.name} at origin {T.date()}: every fit failed and there is no "
                                   f"previous fit to carry forward ({e2})") from e2
            fit, reason = _carry_forward(dsn, spec, start, 2), "fit_failed"
        ll_parent = parent_fit.loglike if parent_fit is not None else None
        nesting_ok = None if ll_parent is None else bool(fit.loglike >= ll_parent - 1e-6)
        return fit, dsn, dict(restart_reason=reason, ll_gain_from_restart=np.nan, n_restarts_used=C.N_RESTARTS,
                              n_warnings=fit.n_warnings + 1, nesting_ok=nesting_ok)
    ll_prev = None
    if start is not None:
        try:
            ll_prev = float(warm.model.loglike(start.reindex(list(warm.model.param_names)).values))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            ll_prev = None
        ll_prev = ll_prev if ll_prev is not None and np.isfinite(ll_prev) else None
    ll_parent = parent_fit.loglike if parent_fit is not None else None
    flags = sanity_flags(warm, ll_prev, ll_parent, C.SIGMA2_RATIO_MIN)
    fit, reason, gain, n_rest, n_warn = warm, "", np.nan, 0, warm.n_warnings
    scheduled = idx % C.RESTART_EVERY == 0
    trigger = [f for f in flags if f != "degenerate" or not prev_degenerate]
    if scheduled or trigger:
        reason = "+".join((["scheduled"] if scheduled else []) + flags)
        extra = ()
        if parent_fit is not None and ("nesting" in flags or start is None):
            extra = nest or (ms.nest_start(parent_fit, ms.make_model(dsn, spec), spec),)
        try:
            refit = ms.fit_ms(dsn, spec, start if start is not None else warm.params, 0, C.N_RESTARTS, seed,
                              extra_starts=extra)
        except FIT_ERRORS as e:
            if _is_data_error(e):
                raise
            refit = None
        if refit is None:                                # restart failed: keep the warm fit, say so
            reason, n_rest, n_warn = reason + "+refit_failed", C.N_RESTARTS, n_warn + 1
        else:
            fit = ms.pick_best([warm, refit])
            gain, n_rest, n_warn = fit.loglike - warm.loglike, C.N_RESTARTS, n_warn + refit.n_warnings
    nesting_ok = None if ll_parent is None else bool(fit.loglike >= ll_parent - 1e-6)
    return fit, dsn, dict(restart_reason=reason, ll_gain_from_restart=gain, n_restarts_used=n_rest,
                          n_warnings=n_warn, nesting_ok=nesting_ok)


def _rebuild_state(panel, ms_specs, params: pd.DataFrame, done: list) -> tuple[dict, dict]:
    """Warm starts from the last completed origin of each model (MS names mapped back to statsmodels)."""
    ms_state, bm_state = {}, {}
    sub = params[params["origin"].isin([o.strftime("%Y-%m-%d") for o in done])]
    specs = {s.name: s for s in ms_specs}
    for name, g in sub.groupby("model", sort=False):
        last = max(g["origin"])
        g = g[g["origin"] == last]
        vals = pd.Series([float(v) for v in g["value"]], index=list(g["name"]))
        if name in specs:
            model = ms.make_model(design.build_design(panel, specs[name], pd.Timestamp(last)), specs[name])
            inv = ms.readable_to_sm(model)
            ms_state[name] = vals.rename(index=inv).reindex(list(model.param_names))
        else:
            bm_state[name] = vals
    return ms_state, bm_state


def run_oos(panel, run_id, ms_specs, benchmarks=None, origins=None, iran_var=None,
            force=False, max_origins=None) -> None:
    """Run (or resume) the expanding-window loop and append every origin to the checkpoints.

    benchmarks=None / iran_var=None mean C.BENCHMARKS_MAIN / C.IRAN_VAR at call time.
    origins=None means origins(C.VALID[0], C.FREEZE_DATE). max_origins stops after that many new origins.
    """
    if C.PHI_SWITCHING is None:
        raise ValueError("C.PHI_SWITCHING is None: decide it in notebook 03 first")
    benchmarks = C.BENCHMARKS_MAIN if benchmarks is None else benchmarks
    iran_var = C.IRAN_VAR if iran_var is None else iran_var
    bms = tuple(benchmarks)
    specs = sorted(ms_specs, key=lambda s: len(s.tvtp_vars))
    h = settings_hash(specs, bms, iran_var)
    if C.SPEC_FROZEN and run_id == "main":
        f = paths.FROZEN / "spec_freeze.json"
        if not f.exists():
            raise RuntimeError(f"{f} (spec_freeze.json) is missing: the full run needs the frozen settings")
        frozen = json.loads(f.read_text()).get("settings_hash")
        if frozen != h:
            raise RuntimeError(f"settings_hash {h} differs from spec_freeze.json ({frozen})")
    all_orig = [pd.Timestamp(o) for o in (_origins(C.VALID[0], C.FREEZE_DATE) if origins is None else origins)]
    n_models = len(specs) + len(bms)
    files = _ckpt(run_id)

    # read the checkpoint (no file is changed before the guards pass)
    resume = None
    if not force and all(p.exists() for p in files.values()):
        tabs = {k: _read_text(p) for k, p in files.items()}
        if all(len(t) == 0 or (t["settings_hash"] == h).all() for t in tabs.values()) and len(tabs["fitlog"]):
            cnt = tabs["fitlog"].groupby("origin").size()
            done = sorted(pd.Timestamp(o) for o in cnt.index[cnt == n_models])
            if done:
                resume = (tabs, done)
    todo = [o for o in all_orig if resume is None or o > resume[1][-1]]
    if max_origins is not None:
        todo = todo[:max_origins]
    late = [o for o in todo if o + WEEK > pd.Timestamp(C.VALID[1])]
    if late and not C.SPEC_FROZEN:
        raise RuntimeError(f"SPEC_FROZEN is False but target {(late[0] + WEEK).date()} is after the "
                           f"validation window: freeze the spec first")

    for d in (paths.CHECKPOINTS, paths.FORECASTS, paths.FITS):
        d.mkdir(parents=True, exist_ok=True)
    ms_state, bm_state, prev_deg = {}, {}, {}
    bm_last = {}                                         # benchmark -> (last forecast values, last params)
    if resume is None:
        for p in files.values():
            p.unlink(missing_ok=True)
    else:
        tabs, done = resume
        keep = {o.strftime("%Y-%m-%d") for o in done}
        for k, p in files.items():                       # drop rows of a crashed, incomplete origin
            t = tabs[k]
            if t.attrs.get("torn") or (len(t) and not t["origin"].isin(keep).all()):
                _rewrite(p, t[t["origin"].isin(keep)])
        ms_state, bm_state = _rebuild_state(panel, specs, tabs["params"][tabs["params"]["origin"].isin(keep)],
                                            done)
        lg = tabs["fitlog"]
        last = lg[lg["origin"] == done[-1].strftime("%Y-%m-%d")]
        prev_deg = {m: v == "True" for m, v in zip(last["model"], last["degenerate"]) if m in ms_state}
        fcs = tabs["forecasts"][tabs["forecasts"]["origin"].isin(keep)]
        for name, g in fcs[fcs["model"].isin(bms)].groupby("model", sort=False):
            g = g[g["origin"] == max(g["origin"])]
            bm_last[name] = ({t: float(v) for t, v in zip(g["target"], g["forecast"])},
                             bm_state.get(name, pd.Series(dtype=float)))
    bm_state["iran_var"] = iran_var
    thr = forecast.highvol_threshold(panel)
    t_start, n_bad = time.perf_counter(), 0

    for n, T in enumerate(todo, start=1):
        idx = origin_index(T)
        seed = C.SEED + idx
        target = T + WEEK
        iso, split = T.strftime("%Y-%m-%d"), _split(target)
        fc, par, log = [], [], []
        fits = {}

        def fc_row(model, tgt, v):
            fc.append(dict(date=target.strftime("%Y-%m-%d"), origin=iso, split=split, run=run_id, model=model,
                           target=tgt, forecast=float(v), settings_hash=h))

        for spec in specs:
            t0 = time.perf_counter()
            parent = ms.nest_parent(spec, specs)
            fit, dsn, extra = _fit_ms_origin(panel, spec, T, seed, ms_state.get(spec.name),
                                             fits.get(parent.name) if parent is not None else None, idx,
                                             prev_deg.get(spec.name, False))
            fits[spec.name] = fit
            ms_state[spec.name] = fit.params
            prev_deg[spec.name] = bool(fit.degenerate)
            one = forecast.one_step(fit, dsn, thr)
            vals = {"var": one.var_fc, "r": one.mean_fc, "p_high": one.pi_H, "p_exceed": one.p_exceed}
            _check_finite(spec.name, T, vals)
            for tgt, v in vals.items():
                fc_row(spec.name, tgt, v)
            for k, v in fit.params.items():
                par.append(dict(origin=iso, model=spec.name, name=fit.model.readable[k], value=float(v),
                                settings_hash=h))
            try:
                dur = ms.expected_durations(fit.params.reindex(list(fit.model.param_names)).values, fit.model)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                dur = {"N": np.nan, "H": np.nan}
            log.append(dict(origin=iso, model=spec.name, settings_hash=h, loglike=fit.loglike,
                            converged=bool(fit.converged), relabeled=bool(fit.relabeled),
                            refit_from_scratch=bool(fit.refit_from_scratch), degenerate=bool(fit.degenerate),
                            seconds=time.perf_counter() - t0, sigma2_N=float(fit.params["sigma2[0]"]),
                            sigma2_H=float(fit.params["sigma2[1]"]), dur_N=dur["N"], dur_H=dur["H"],
                            xi_H_origin=one.xi_H_origin, at_cap=len(ms.at_cap(fit.params)), **extra))
            n_bad += not fit.converged

        for name in bms:
            fo = first_origin(name)
            if fo is not None and T < fo:
                log.append(dict(origin=iso, model=name, settings_hash=h, converged=False, at_cap=0,
                                restart_reason=f"skipped (first origin {fo.date()})"))
                continue
            if name != "RW" and name not in VAR_ONLY:
                raise ValueError(f"unknown benchmark '{name}'")
            t0 = time.perf_counter()
            try:
                out = bench.model_forecast(name, panel, T, bm_state)
            except FIT_ERRORS as e:
                if _is_data_error(e):
                    raise ValueError(f"{name} at origin {T.date()}: {e}") from e
                prev = bm_last.get(name)
                log.append(dict(origin=iso, model=name, settings_hash=h, converged=False, at_cap=0,
                                seconds=time.perf_counter() - t0, n_warnings=1,
                                restart_reason="fit_failed (previous forecast reused)" if prev
                                else "fit_failed (no forecast)"))
                n_bad += 1
                if prev:
                    for tgt, v in prev[0].items():
                        fc_row(name, tgt, v)
                    for k, v in prev[1].items():
                        par.append(dict(origin=iso, model=name, name=k, value=float(v), settings_hash=h))
                continue
            vals = {"r": out["mean_fc"]} if name == "RW" else {"var": out["var_fc"]}
            _check_finite(name, T, vals)
            for tgt, v in vals.items():
                fc_row(name, tgt, v)
            for k, v in out["params"].items():
                par.append(dict(origin=iso, model=name, name=k, value=float(v), settings_hash=h))
            bm_last[name] = (dict(vals), out["params"])
            log.append(dict(origin=iso, model=name, settings_hash=h, loglike=out["loglike"],
                            converged=bool(out["converged"]), seconds=out["seconds"],
                            har_floored=bool(out.get("floored", False)), n_warnings=out["n_warnings"], at_cap=0))
            n_bad += not out["converged"]

        _append(files["forecasts"], pd.DataFrame(fc, columns=FC_COLS + ["settings_hash"]))
        _append(files["params"], pd.DataFrame(par, columns=PARAM_COLS))
        _append(files["fitlog"], pd.DataFrame(log, columns=FITLOG_COLS))
        if n % 50 == 0:
            print(f"origin {iso}: {n} origins, {time.perf_counter() - t_start:.0f} s, {n_bad} non-converged fits")


def _read_num(path, **kw) -> pd.DataFrame:
    """Read a checkpoint (pandas 3's default float parser round-trips to_csv output exactly)."""
    return pd.read_csv(path, **kw)


def assemble(run_id) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Checkpoints -> forecasts (spec §9.2), params (wide) and fitlog; writes them to results/."""
    files = _ckpt(run_id)
    txt = {"origin": str, "settings_hash": str}
    log = _read_num(files["fitlog"], dtype={**txt, "restart_reason": str})
    cnt = log.groupby("origin")["model"].size()
    done = set(cnt.index[cnt == log["model"].nunique()])      # origins with a complete fitlog
    log = log[log["origin"].isin(done)].reset_index(drop=True)
    fc = _read_num(files["forecasts"], dtype={**txt, "date": str})
    fc = fc[fc["origin"].isin(done)].reset_index(drop=True)
    pl = _read_num(files["params"], dtype=txt)
    pl = pl[pl["origin"].isin(done)]
    fc = fc[FC_COLS].copy()
    fc["date"], fc["origin"] = pd.to_datetime(fc["date"]), pd.to_datetime(fc["origin"])
    if len(pl):
        wide = pl.set_index(["origin", "model", "name"])["value"].unstack("name")
        rows = pl[["origin", "model"]].drop_duplicates()
        wide = wide.reindex(pd.MultiIndex.from_frame(rows))[list(pl["name"].unique())]
        wide = wide.rename_axis(columns=None).reset_index()
    else:
        wide = pd.DataFrame(columns=["origin", "model"])
    wide["origin"] = pd.to_datetime(wide["origin"])
    log = log[FITLOG_COLS].copy()
    log["origin"] = pd.to_datetime(log["origin"])
    for d in (paths.FORECASTS, paths.FITS):
        d.mkdir(parents=True, exist_ok=True)
    fc.to_csv(paths.FORECASTS / f"forecasts_{run_id}.csv", index=False)
    wide.to_csv(paths.FITS / f"params_{run_id}.csv", index=False)
    log.to_csv(paths.FITS / f"fitlog_{run_id}.csv", index=False)
    return fc, wide, log


def load_forecasts(run_id, with_actuals=True) -> pd.DataFrame:
    """Read results/forecasts/forecasts_<run>.csv; optionally join the actuals from the weekly panel."""
    f = _read_num(paths.FORECASTS / f"forecasts_{run_id}.csv")
    f["date"], f["origin"] = pd.to_datetime(f["date"]), pd.to_datetime(f["origin"])
    if not with_actuals:
        return f
    panel = pd.read_csv(paths.PROCESSED / "weekly_panel.csv")
    panel["date"] = pd.to_datetime(panel["date"])
    thr = forecast.highvol_threshold(panel)
    act = pd.DataFrame({"date": panel["date"]})
    for col in ("rv", "rv_hourly", "rv_gk", "r"):          # a column missing from the panel gives NaN
        act[f"actual_{col}"] = panel[col] if col in panel.columns else np.nan
    act["actual_exceed"] = (panel["r"].abs() > thr).astype(float).where(panel["r"].notna())
    return f.merge(act, on="date", how="left")


def fake_forecasts(panel, models=None) -> pd.DataFrame:
    """Schema-correct dummy forecasts over all 723 targets (run = "fake"); the caller saves it.

    models=None means ("M1", "M2", "M3") + C.BENCHMARKS_MAIN at call time.

    var = rv_hourly x lognormal(0, 0.3) at the target (rv where rv_hourly is NaN); hourly benchmarks
    start at `first_origin`, as in a real run.
    """
    models = ("M1", "M2", "M3") + tuple(C.BENCHMARKS_MAIN) if models is None else models
    rng = np.random.default_rng(C.SEED)
    P = panel.copy()
    P.index = pd.DatetimeIndex(pd.to_datetime(P["date"]))
    level = P["rv_hourly"].fillna(P["rv"]) if "rv_hourly" in P.columns else P["rv"]
    rows = []
    for T in origins(C.VALID[0], C.FREEZE_DATE):
        target = T + WEEK
        base = dict(date=target, origin=T, split=_split(target), run="fake")
        for m in models:
            fo = first_origin(m)
            if fo is not None and T < fo:
                continue
            if m == "RW":
                vals = {"r": 0.0}
            elif m in VAR_ONLY:
                vals = {"var": float(level[target]) * rng.lognormal(0, 0.3)}
            else:
                vals = {"var": float(level[target]) * rng.lognormal(0, 0.3), "r": rng.normal(0, 0.5),
                        "p_high": rng.uniform(), "p_exceed": rng.uniform(0, 0.5)}
            rows += [dict(base, model=m, target=k, forecast=v) for k, v in vals.items()]
    return pd.DataFrame(rows, columns=FC_COLS)
