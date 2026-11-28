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


def make_specs(phi_switching: bool, variant: str | None = None,
               controls: tuple[str, ...] | None = None,
               iran_var: str | None = None) -> dict[str, MSSpec]:
    """The ladder M1 (constant TP), M2 (controls), M3 (controls + Iran); None means the config value now."""
    variant = C.M3_VARIANT if variant is None else variant
    controls = tuple(C.TVTP_CONTROLS if controls is None else controls)
    iran_var = C.IRAN_VAR if iran_var is None else iran_var
    return {
        "M1": MSSpec("M1", (), (), False, phi_switching),
        "M2": MSSpec("M2", controls, (), False, phi_switching),
        "M3": MSSpec("M3", controls + (iran_var,), (iran_var,), variant == "constrained", phi_switching),
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
    """Fixed parameters (Iran only moves N->H) and the Iran beta parameter names."""
    fixed, beta = {}, {}
    for v in spec.iran_vars:
        j = model.tvtp_name_map[v]
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


def is_degenerate(params, model, var_y: float, floor_frac=None, min_duration=None) -> bool:
    """True if min sigma2 < floor_frac * var_y or a regime's expected duration < min_duration."""
    floor_frac = C.VAR_FLOOR_FRAC if floor_frac is None else floor_frac
    min_duration = C.MIN_DURATION if min_duration is None else min_duration
    p = pd.Series(np.asarray(params, dtype=float), index=list(model.param_names))
    if min(p["sigma2[0]"], p["sigma2[1]"]) < floor_frac * var_y:
        return True
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            dur = expected_durations(p.values, model)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return True
    return not all(np.isfinite(d) or d > 0 for d in dur.values()) or min(dur.values()) < min_duration


def pick_best(fits: list):
    """Best loglike among non-degenerate fits; if all are degenerate, the best loglike overall."""
    ok = [f for f in fits if not f.degenerate]
    return max(ok or fits, key=lambda f: f.loglike if np.isfinite(f.loglike) else -np.inf)


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

    Restricted specs (Iran only in N->H) take the constrained path. n_warnings aggregates over all
    candidates; a candidate whose fit raises counts as one warning.
    """
    info = constraint_info(spec, model) if (spec.iran_vars and spec.iran_only_NH and spec.kind == "msar") else None
    if info is not None:
        return _fit_restricted(model, spec, info["fixed"], start_params, search_reps, n_restarts, seed,
                               extra_starts)
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


def fit_constrained(model, start_params: pd.Series, fixed: dict, n_restarts: int, seed: int):
    """Maximize the loglike over the free parameters with `fixed` held exactly.

    Returns (params incl. fixed values, loglike, converged); BFGS, then Nelder-Mead if BFGS fails.
    """
    names = list(model.param_names)
    fixed_idx = np.array([names.index(n) for n in fixed], dtype=int)
    fixed_vals = np.array([fixed[n] for n in fixed], dtype=float)
    free_idx = np.array([i for i in range(len(names)) if i not in set(fixed_idx)], dtype=int)
    n_obs = len(model.endog)
    var_y = float(np.asarray(model.endog).var())
    u_start = np.asarray(model.untransform_params(np.asarray(start_params.reindex(names), dtype=float)), float)

    def build(uf):
        return _full_params(model, uf, free_idx, u_start, fixed_idx, fixed_vals)

    def neg(uf):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                v = -model.loglike(build(uf)) / n_obs
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return 1e10
        return v if np.isfinite(v) else 1e10

    def run(uf0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(neg, uf0, method="BFGS", options={"maxiter": 200})
            ok = bool(r.success)
            if not ok:
                r = minimize(neg, r.x, method="Nelder-Mead", options={"maxiter": 4000, "maxfev": 4000,
                                                                     "xatol": 1e-6, "fatol": 1e-10})
                ok = bool(r.success)
        p = pd.Series(build(r.x), index=names)
        p[list(fixed)] = fixed_vals
        return p, float(model.loglike(p.values)), ok

    starts = [u_start[free_idx]] + [u_start[free_idx] + np.random.default_rng(seed + k).normal(0, 0.5, len(free_idx))
                                    for k in range(1, n_restarts + 1)]
    res = [run(u0) for u0 in starts]
    return max(res, key=lambda r: (r[0]["sigma2[0]"] <= r[0]["sigma2[1]"], not is_degenerate(r[0], model, var_y),
                                   r[1] if np.isfinite(r[1]) else -np.inf))


def _fit_restricted(model, spec, fixed, start_params, search_reps, n_restarts, seed, extra_starts):
    """Constrained fit: warm start if given (refit from scratch on a label flip), else first-fit procedure."""
    t0 = time.perf_counter()
    names = list(model.param_names)
    var_y = float(np.asarray(model.endog).var())

    def finish(params, ll, ok, refit, nw):
        return MSFit(spec, params, float(ll), ok, n_restarts, False, refit, is_degenerate(params, model, var_y),
                     time.perf_counter() - t0, model, dict(fixed), nw)

    if start_params is not None:
        params, ll, ok = fit_constrained(model, start_params.reindex(names), fixed, n_restarts, seed)
        if params["sigma2[0]"] <= params["sigma2[1]"]:
            return finish(params, ll, ok, False, 0)
    base = fit_model(model, replace(spec, iran_only_NH=False), None, search_reps, 0, seed)
    starts = [base.params] + [s.reindex(names) for s in extra_starts]
    cands = [fit_constrained(model, st, fixed, n_restarts, seed) for st in starts]
    good = [c for c in cands if c[0]["sigma2[0]"] <= c[0]["sigma2[1]"]]
    nw = base.n_warnings + len(cands) - len(good)
    if good:
        ok_c = [c for c in good if not is_degenerate(c[0], model, var_y)]
        params, ll, ok = max(ok_c or good, key=lambda c: c[1])
        return finish(params, ll, ok, start_params is not None, nw)
    # every candidate flipped: fall back to a canonical, constraint-satisfying point, flagged as failed
    fallbacks = ([start_params.reindex(names)] if start_params is not None else []) \
        + [s.reindex(names) for s in extra_starts] + [base.params]
    for fb in fallbacks:
        p = fb.copy()
        p[list(fixed)] = list(fixed.values())
        if p["sigma2[0]"] <= p["sigma2[1]"]:
            break
    fit = finish(p, model.loglike(p.values), False, start_params is not None, nw + 1)
    fit.degenerate = True
    return fit


def param_cov(model, params: pd.Series, fixed: dict) -> pd.DataFrame:
    """Covariance of the free parameters, inv(-Hessian of the loglike); NaN where it cannot be computed."""
    names = [n for n in params.index if n not in fixed]
    k = len(names)
    nan = pd.DataFrame(np.full((k, k), np.nan), index=names, columns=names)
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
    return pd.DataFrame(cov, index=names, columns=names)


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
    return out


def nest_parent(spec: MSSpec, specs: list) -> "MSSpec | None":
    """The spec in `specs` that `spec` nests: same kind/phi/mean_exog, fewer TVTP vars, Iran vars contained."""
    cands = [s for s in specs
             if s.kind == spec.kind and s.phi_switching == spec.phi_switching and s.mean_exog == spec.mean_exog
             and set(s.tvtp_vars) < set(spec.tvtp_vars) and set(s.iran_vars) <= set(spec.iran_vars)]
    return max(cands, key=lambda s: len(s.tvtp_vars)) if cands else None
