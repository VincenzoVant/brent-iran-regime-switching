"""Markov-switching specifications and fitting (spec §8.4)."""

import time
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess3
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from brent import config as C


@dataclass(frozen=True)
class MSSpec:
    """One model specification: which regressors drive the transitions and the mean."""

    name: str
    tvtp_vars: tuple[str, ...]
    iran_vars: tuple[str, ...]
    iran_only_NH: bool
    phi_switching: bool
    mean_exog: tuple[str, ...] = ()
    kind: str = "msar"
    canonical_by: str = "sigma2"
    std_window: tuple[str, str] = field(default_factory=lambda: C.STANDARDIZE_WINDOW)
    start: str = field(default_factory=lambda: C.SAMPLE_START)
    controls_only_NH: bool = field(default_factory=lambda: C.CONTROLS_ONLY_NH)


def make_specs(phi_switching: bool, variant: str | None = None,
               controls: tuple[str, ...] | None = None,
               iran_var: str | None = None, controls_only_NH: bool | None = None) -> dict[str, MSSpec]:
    """The ladder M1 (constant TP), M2 (controls), M3 (controls + Iran); None means the config value now."""
    variant = C.M3_VARIANT if variant is None else variant
    controls = tuple(C.TVTP_CONTROLS if controls is None else controls)
    iran_var = C.IRAN_VAR if iran_var is None else iran_var
    co = C.CONTROLS_ONLY_NH if controls_only_NH is None else controls_only_NH
    return {
        "M1": MSSpec("M1", (), (), False, phi_switching, controls_only_NH=co),
        "M2": MSSpec("M2", controls, (), False, phi_switching, controls_only_NH=co),
        "M3": MSSpec("M3", controls + (iran_var,), (iran_var,), variant == "constrained", phi_switching,
                     controls_only_NH=co),
    }


def with_iran(spec: MSSpec, iran_vars: tuple[str, ...], name: str) -> MSSpec:
    """Swap the restricted Iran variables inside tvtp_vars; controls and other fields are kept."""
    iran_vars = tuple(iran_vars)
    keep = tuple(v for v in spec.tvtp_vars if v not in spec.iran_vars)
    return replace(spec, name=name, tvtp_vars=keep + iran_vars, iran_vars=iran_vars)


def _readable_names(model, tvtp_names: list[str], mean_exog: tuple[str, ...]) -> dict[str, str]:
    """Map statsmodels parameter names to readable names (a bijection)."""
    names = list(model.param_names)
    out = {}
    for n in names:
        base = n.split("[")[0]
        if n.startswith("p[") and ".tvtp" in n:
            j = int(n.split(".tvtp")[1])
            pre = "p00" if n.startswith("p[0->0]") else "p10"
            out[n] = f"{pre}_{tvtp_names[j]}"
        elif n == "p[0->0]":
            out[n] = "p00"
        elif n == "p[1->0]":
            out[n] = "p10"
        elif base in ("const", "sigma2"):
            stem = "mu" if base == "const" else "sigma2"
            twin = n[:-3] + ("[1]" if n.endswith("[0]") else "[0]")
            out[n] = (stem + ("_N" if n.endswith("[0]") else "_H")) if twin in names else stem
        elif base.startswith("x") and base[1:].isdigit():
            k = int(base[1:])
            if k == 1:
                twin = n[:-3] + ("[1]" if n.endswith("[0]") else "[0]")
                out[n] = ("phi_N" if n.endswith("[0]") else "phi_H") if twin in names else "phi"
            else:
                out[n] = f"theta_{mean_exog[k - 2]}"
        else:
            out[n] = n
    return out


def make_model(design, spec: MSSpec, **overrides) -> MarkovRegression:
    """Two-regime MarkovRegression; TVTP when design.Z is given. Extra kwargs go to MarkovRegression."""
    y, X, Z = design.y, design.X, design.Z
    assert X.index.equals(y.index) and (Z is None or Z.index.equals(y.index)), "design index mismatch"
    try:
        idx = pd.DatetimeIndex(design.y.index, freq="W-FRI")
        y, X = y.set_axis(idx), X.set_axis(idx)
        Z = Z.set_axis(idx) if Z is not None else None
    except (ValueError, TypeError):
        pass
    kw = dict(k_regimes=2, trend="c", exog=X,
              switching_exog=[spec.phi_switching] + [False] * len(spec.mean_exog),
              switching_variance=True)
    if Z is not None:
        kw["exog_tvtp"] = Z
    kw.update(overrides)
    model = MarkovRegression(y, **kw)
    tv = list(design.Z.columns) if design.Z is not None else []
    model.tvtp_name_map = {v: i for i, v in enumerate(tv)}
    model.readable = _readable_names(model, tv, tuple(spec.mean_exog))
    return model


def readable_to_sm(model) -> dict[str, str]:
    """Inverse of model.readable."""
    return {v: k for k, v in model.readable.items()}


def canonicalize(params: pd.Series, model, by: str = "sigma2") -> tuple[pd.Series, bool]:
    """Relabel so regime 1 has the higher sigma2 (or const); returns (params, relabeled)."""
    key = "sigma2" if by == "sigma2" else "const"
    if not params[f"{key}[0]"] > params[f"{key}[1]"]:
        return params, False
    old = params.copy()
    new = params.copy()
    for n in old.index:
        if n.endswith("[0]") and n[:-3] + "[1]" in old.index:
            tw = n[:-3] + "[1]"
            new[n], new[tw] = old[tw], old[n]
    if any(".tvtp" in n for n in old.index):
        for n in old.index:
            if n.startswith("p[0->0]."):
                new[n] = -old[n.replace("p[0->0]", "p[1->0]")]
            elif n.startswith("p[1->0]."):
                new[n] = -old[n.replace("p[1->0]", "p[0->0]")]
    else:
        new["p[0->0]"], new["p[1->0]"] = 1 - old["p[1->0]"], 1 - old["p[0->0]"]
    return new, True


def transition_coefs_deck(params: pd.Series, tvtp_names: list[str]) -> pd.DataFrame:
    """Deck coefficients: N->H = -(p[0->0] coef), H->N = +(p[1->0] coef)."""
    out = pd.DataFrame(index=["N->H", "H->N"], columns=list(tvtp_names), dtype=float)
    for j, v in enumerate(tvtp_names):
        out.loc["N->H", v] = -params[f"p[0->0].tvtp{j}"]
        out.loc["H->N", v] = params[f"p[1->0].tvtp{j}"]
    return out


def expected_durations(params, model, exog_tvtp=None) -> dict[str, float]:
    """Expected durations 1/(1-P_kk); default TVTP regressors are the zero row [1, 0, ..., 0]."""
    if getattr(model, "tvtp_name_map", None):
        if exog_tvtp is None:
            row = np.zeros(len(model.tvtp_name_map))
            row[0] = 1.0
            exog_tvtp = row[None, :]
        P = model.regime_transition_matrix(params, exog_tvtp=np.atleast_2d(exog_tvtp))
    else:
        P = model.regime_transition_matrix(params)
    P = np.asarray(P)
    if P.ndim == 3:
        P = P[..., 0]
    with np.errstate(divide="ignore"):  # P_kk = 1 gives an infinite duration, not a warning
        return {"N": float(1.0 / (1.0 - P[0, 0])), "H": float(1.0 / (1.0 - P[1, 1]))}


def constraint_info(spec: MSSpec, model) -> dict:
    """Fixed H->N coefficients (controls if controls_only_NH, Iran if iran_only_NH) and the Iran beta names."""
    fixed, beta = {}, {}
    tmap = getattr(model, "tvtp_name_map", None) or {}
    if spec.controls_only_NH:
        for v in spec.tvtp_vars:
            if v not in spec.iran_vars:
                fixed[f"p[1->0].tvtp{tmap[v]}"] = 0.0
    for v in spec.iran_vars:
        j = tmap[v]
        beta[v] = f"p[0->0].tvtp{j}"
        if spec.iran_only_NH:
            fixed[f"p[1->0].tvtp{j}"] = 0.0
    return {"fixed": fixed, "beta": beta}


@dataclass
class MSFit:
    """A fitted model in canonical labels (regime 0 = N, regime 1 = H)."""

    spec: MSSpec
    params: pd.Series
    loglike: float
    converged: bool
    n_restarts_used: int
    relabeled: bool
    refit_from_scratch: bool
    degenerate: bool
    seconds: float
    model: MarkovRegression
    fixed: dict[str, float] = field(default_factory=dict)
    n_warnings: int = 0


def is_degenerate(params, model, var_y: float, floor_frac=None, min_duration=None, max_coef=None) -> bool:
    """True if min sigma2 < floor_frac * var_y, a regime's expected duration < min_duration,
    or any TVTP coefficient has |value| > max_coef (near-separation)."""
    floor_frac = C.VAR_FLOOR_FRAC if floor_frac is None else floor_frac
    min_duration = C.MIN_DURATION if min_duration is None else min_duration
    max_coef = C.MAX_TVTP_COEF if max_coef is None else max_coef
    p = pd.Series(np.asarray(params, dtype=float), index=list(model.param_names))
    if min(p["sigma2[0]"], p["sigma2[1]"]) < floor_frac * var_y:
        return True
    tv = p[[n for n in p.index if ".tvtp" in n]]
    if len(tv) and not (tv.abs() <= max_coef + 1e-9).all():             # at the cap is not degenerate
        return True
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            dur = expected_durations(p.values, model)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return True
    return not all(np.isfinite(d) or d > 0 for d in dur.values()) or min(dur.values()) < min_duration


def _slope_names(model) -> list[str]:
    """TVTP slope parameters (".tvtpJ", J >= 1); the logit constants ".tvtp0" are not slopes."""
    return [n for n in model.param_names if ".tvtp" in n and not n.endswith(".tvtp0")]


def at_cap(params: pd.Series, tol: float = 1e-6) -> list[str]:
    """Names of TVTP slopes at the +-C.MAX_TVTP_COEF cap (|coef| >= cap - tol)."""
    cap = C.MAX_TVTP_COEF
    return [n for n, v in params.items() if ".tvtp" in n and not n.endswith(".tvtp0") and abs(v) >= cap - tol]


def pick_best(fits: list):
    """Among non-degenerate fits (or all, if every fit is degenerate): converged first, then highest loglike."""
    ok = [f for f in fits if not f.degenerate]
    return max(ok or fits, key=lambda f: (bool(getattr(f, "converged", True)),
                                          f.loglike if np.isfinite(f.loglike) else -np.inf))


def _fit_once(model, spec, start, search_reps, seed):
    """One statsmodels fit; returns (canonical params, relabeled, converged, n_warnings) or None on failure."""
    kw = dict(disp=0, maxiter=200)
    if start is None:
        kw.update(search_reps=search_reps, rng=seed, method="bfgs")
    else:
        kw.update(start_params=np.asarray(start, dtype=float), search_reps=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = model.fit(**kw)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return None
    nw = sum(issubclass(x.category, (ConvergenceWarning, RuntimeWarning)) for x in w)
    params = pd.Series(np.asarray(res.params, dtype=float), index=list(model.param_names))
    conv = bool(res.mle_retvals.get("converged", False)) if getattr(res, "mle_retvals", None) else False
    params, relabeled = canonicalize(params, model, spec.canonical_by)
    return params, relabeled, conv, nw


def fit_model(model, spec: MSSpec, start_params, search_reps: int, n_restarts: int, seed: int,
              extra_starts=()) -> MSFit:
    """Fit a built model: main fit, extra starts, then perturbed restarts; keep the best non-degenerate.

    Specs with fixed coefficients (controls and/or Iran only in N->H) or TVTP slopes take the constrained,
    bounded path (slopes capped at +-C.MAX_TVTP_COEF). n_warnings aggregates over all candidates; a candidate
    whose fit raises counts as one warning.
    """
    info = constraint_info(spec, model) if spec.kind == "msar" else None
    if info is not None and (info["fixed"] or _slope_names(model)):
        return _fit_restricted(model, spec, info["fixed"], start_params, search_reps, n_restarts, seed,
                               extra_starts)
    return _fit_free(model, spec, start_params, search_reps, n_restarts, seed, extra_starts)


def _fit_free(model, spec: MSSpec, start_params, search_reps: int, n_restarts: int, seed: int,
              extra_starts=()) -> MSFit:
    """Unbounded statsmodels fit (no fixed coefficients); used directly for M1 and as a start elsewhere."""
    t0 = time.perf_counter()
    var_y = float(np.asarray(model.endog).var())
    names = list(model.param_names)
    cands: list[MSFit] = []
    n_warn = 0

    def add(start, reps, sd):
        nonlocal n_warn
        r = _fit_once(model, spec, start, reps, sd)
        if r is None:
            n_warn += 1
            return
        params, rel, conv, nw = r
        n_warn += nw
        try:
            ll = float(model.loglike(params.values))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            n_warn += 1
            return
        cands.append(MSFit(spec, params, ll, conv, n_restarts, rel, False,
                           is_degenerate(params, model, var_y), 0.0, model))

    start = None if start_params is None else start_params.reindex(names)
    add(start, search_reps, seed)
    for s in extra_starts:
        add(s.reindex(names), 0, seed)
    for k in range(1, n_restarts + 1):
        base = start if start is not None else (pick_best(cands).params if cands else None)
        if base is None:
            break
        u = model.untransform_params(np.asarray(base, dtype=float))
        u = u + np.random.default_rng(seed + k).normal(0, 0.5, size=len(u))
        add(pd.Series(model.transform_params(u), index=names), 0, seed + k)
    if not cands:
        raise RuntimeError("all fit candidates failed")
    best = pick_best(cands)
    best.seconds = time.perf_counter() - t0
    best.n_warnings = n_warn
    return best


def _full_params(model, u_free, free_idx, base_u, fixed_idx, fixed_vals):
    """Natural-space params from free untransformed values plus fixed values."""
    u = base_u.copy()
    u[free_idx] = u_free
    u[fixed_idx] = fixed_vals
    return model.transform_params(u)


class _Counted(tuple):
    """(params, loglike, converged) that also carries n_warnings as an attribute."""

    n_warnings = 0


def fit_constrained(model, start_params: pd.Series, fixed: dict, n_restarts: int, seed: int):
    """Maximize the loglike over the free parameters with `fixed` held exactly and every free TVTP slope
    bounded to [-C.MAX_TVTP_COEF, +C.MAX_TVTP_COEF] (bounded MLE against near-separation).

    Returns (params incl. fixed values, loglike, converged); L-BFGS-B, then bounded Nelder-Mead if it fails.
    Starts (and restart perturbations) are clipped into the bounds first.
    The tuple's `n_warnings` counts runs that hit the penalty, failed, or flipped and were dropped.
    """
    names = list(model.param_names)
    fixed_idx = np.array([names.index(n) for n in fixed], dtype=int)
    fixed_vals = np.array([fixed[n] for n in fixed], dtype=float)
    free_idx = np.array([i for i in range(len(names)) if i not in set(fixed_idx)], dtype=int)
    n_obs = len(model.endog)
    var_y = float(np.asarray(model.endog).var())
    cap = float(C.MAX_TVTP_COEF)
    slopes = set(_slope_names(model))
    sp = start_params.reindex(names).astype(float).copy()
    sp[[n for n in names if n in slopes]] = sp[[n for n in names if n in slopes]].clip(-cap, cap)
    u_start = np.asarray(model.untransform_params(np.asarray(sp, dtype=float)), float)
    # TVTP coefficients are not transformed by statsmodels, so the bounds apply to them directly
    lo = np.array([-cap if names[i] in slopes else -np.inf for i in free_idx])
    hi = np.array([cap if names[i] in slopes else np.inf for i in free_idx])
    bounds = [(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b) for a, b in zip(lo, hi)]

    def build(uf):
        return _full_params(model, uf, free_idx, u_start, fixed_idx, fixed_vals)

    hits = [0]                                   # penalty evaluations in the current run

    def neg(uf):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                v = -model.loglike(build(uf)) / n_obs
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                v = np.nan
        if not np.isfinite(v):
            hits[0] += 1
            return 1e10
        return v

    def run(uf0):
        hits[0] = 0
        uf0 = np.clip(uf0, lo, hi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(neg, uf0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
            ok = bool(r.success)
            if not ok:
                r = minimize(neg, np.clip(r.x, lo, hi), method="Nelder-Mead", bounds=bounds,
                             options={"maxiter": 4000, "maxfev": 4000, "xatol": 1e-6, "fatol": 1e-10})
                ok = bool(r.success)
        p = pd.Series(build(np.clip(r.x, lo, hi)), index=names)
        p[list(fixed)] = fixed_vals
        return p, float(model.loglike(p.values)), ok, hits[0] > 0

    starts = [u_start[free_idx]] + [u_start[free_idx] + np.random.default_rng(seed + k).normal(0, 0.5, len(free_idx))
                                    for k in range(1, n_restarts + 1)]
    res = [run(u0) for u0 in starts]
    best = max(res, key=lambda r: (r[0]["sigma2[0]"] <= r[0]["sigma2[1]"], not is_degenerate(r[0], model, var_y),
                                   r[1] if np.isfinite(r[1]) else -np.inf))
    out = _Counted(best[:3])
    out.n_warnings = sum(int(r[3]) + int(not r[2]) + int(r is not best and r[0]["sigma2[0]"] > r[0]["sigma2[1]"])
                         for r in res)
    return out


def _fit_restricted(model, spec, fixed, start_params, search_reps, n_restarts, seed, extra_starts):
    """Constrained fit: warm start if given (refit from scratch on a label flip), else first-fit procedure."""
    t0 = time.perf_counter()
    names = list(model.param_names)
    var_y = float(np.asarray(model.endog).var())

    def finish(params, ll, ok, refit, nw):
        return MSFit(spec, params, float(ll), ok, n_restarts, False, refit, is_degenerate(params, model, var_y),
                     time.perf_counter() - t0, model, dict(fixed), nw)

    nw_warm = 0
    if start_params is not None:
        # warm start plus any extra starts (e.g. the nesting start on a "nesting" restart in the OOS loop)
        warm = [fit_constrained(model, start_params.reindex(names), fixed, n_restarts, seed)]
        warm += [fit_constrained(model, s.reindex(names), fixed, 0, seed) for s in extra_starts]
        good_w = [c for c in warm if c[0]["sigma2[0]"] <= c[0]["sigma2[1]"]]
        nw_warm = sum(getattr(c, "n_warnings", 0) for c in warm) + len(warm) - len(good_w)
        if good_w:
            ok_w = [c for c in good_w if not is_degenerate(c[0], model, var_y)]
            params, ll, ok = max(ok_w or good_w, key=lambda c: c[1] if np.isfinite(c[1]) else -np.inf)
            return finish(params, ll, ok, False, nw_warm)
        # every warm result flipped: dropped, refit from scratch (flips already counted in nw_warm)
    # unbounded statsmodels search, used only as a start (fit_constrained clips its slopes into the bounds)
    base = _fit_free(model, replace(spec, iran_only_NH=False, controls_only_NH=False), None, search_reps, 0, seed)
    starts = [base.params] + [s.reindex(names) for s in extra_starts]
    cands = [fit_constrained(model, st, fixed, n_restarts, seed) for st in starts]
    good = [c for c in cands if c[0]["sigma2[0]"] <= c[0]["sigma2[1]"]]
    nw = (nw_warm + base.n_warnings + sum(getattr(c, "n_warnings", 0) for c in cands)
          + len(cands) - len(good))
    if good:
        ok_c = [c for c in good if not is_degenerate(c[0], model, var_y)]
        params, ll, ok = max(ok_c or good, key=lambda c: c[1])
        return finish(params, ll, ok, start_params is not None, nw)
    # every candidate flipped: fall back to a canonical, constraint-satisfying point, flagged as failed
    fallbacks = ([start_params.reindex(names)] if start_params is not None else []) \
        + [s.reindex(names) for s in extra_starts] + [base.params]
    sl = _slope_names(model)
    for fb in fallbacks:
        p = fb.copy()
        p[list(fixed)] = list(fixed.values())
        p[sl] = p[sl].clip(-C.MAX_TVTP_COEF, C.MAX_TVTP_COEF)
        if p["sigma2[0]"] <= p["sigma2[1]"]:
            break
    fit = finish(p, model.loglike(p.values), False, start_params is not None, nw + 1)
    fit.degenerate = True
    return fit


def param_cov(model, params: pd.Series, fixed: dict) -> pd.DataFrame:
    """Covariance of the free parameters, inv(-Hessian of the loglike); NaN where it cannot be computed.

    Slopes at the +-MAX_TVTP_COEF cap get NaN (a bound, like garchx_se); the Hessian uses the other free params.
    """
    names = [n for n in params.index if n not in fixed]
    k = len(names)
    nan = pd.DataFrame(np.full((k, k), np.nan), index=names, columns=names)
    capped = set(at_cap(params))
    out_names, names = names, [n for n in names if n not in capped]
    base = params.copy()

    def ll(theta):
        p = base.copy()
        p[names] = theta
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return float(model.loglike(p.reindex(list(model.param_names)).values))

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            H = approx_hess3(params[names].values.astype(float), ll)
            H = (H + H.T) / 2
            if not np.all(np.isfinite(H)):
                return nan
            if np.linalg.cond(H) > 1e14:  # quasi-separated fit: SEs would be meaningless
                return nan
            cov = np.linalg.inv(-H)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return nan
    cov = (cov + cov.T) / 2
    bad = ~(np.diag(cov) > 0)
    cov[bad, :] = np.nan
    cov[:, bad] = np.nan
    return pd.DataFrame(cov, index=names, columns=names).reindex(index=out_names, columns=out_names)


def fit_ms(design, spec: MSSpec, start_params, search_reps: int, n_restarts: int, seed: int,
           extra_starts=()) -> MSFit:
    """Build the model for `spec` and fit it (see fit_model)."""
    return fit_model(make_model(design, spec), spec, start_params, search_reps, n_restarts, seed, extra_starts)


def nest_start(small: MSFit, big_model, big_spec: MSSpec) -> pd.Series:
    """Start values for a bigger model from a nested fit; names are matched through `readable`."""
    s_read, b_read = small.model.readable, big_model.readable
    sm_vals = {s_read[n]: v for n, v in small.params.items()}
    logit = lambda p: float(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
    out = pd.Series(0.0, index=list(big_model.param_names))
    for n in out.index:
        r = b_read[n]
        if r in sm_vals:
            out[n] = sm_vals[r]
        elif r in ("p00_const", "p10_const") and r[:3] in sm_vals:
            out[n] = logit(sm_vals[r[:3]])
        elif r in ("phi_N", "phi_H") and "phi" in sm_vals:
            out[n] = sm_vals["phi"]
    if big_spec.kind == "msar" and getattr(big_model, "tvtp_name_map", None):
        for n, v in constraint_info(big_spec, big_model)["fixed"].items():
            out[n] = v                                   # the start satisfies the big model's constraints
    return out


def nest_parent(spec: MSSpec, specs: list) -> "MSSpec | None":
    """The spec in `specs` that `spec` nests: same kind/phi/mean_exog, fewer TVTP vars, Iran vars contained."""
    cands = [s for s in specs
             if s.kind == spec.kind and s.phi_switching == spec.phi_switching and s.mean_exog == spec.mean_exog
             and set(s.tvtp_vars) < set(spec.tvtp_vars) and set(s.iran_vars) <= set(spec.iran_vars)]
    return max(cands, key=lambda s: len(s.tvtp_vars)) if cands else None
