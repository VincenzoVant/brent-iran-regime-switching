"""Linear benchmarks (spec §8.7): HAR family, GARCH(1,1)-t, random walk. GARCH-t-X (spec §8.7)."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln
from statsmodels.tools.numdiff import approx_hess3

from brent import config as C
from brent import design as D


def _names(cols, x_vars) -> list[str]:
    """Design columns -> readable coefficient names."""
    out = {"const": "b0", "rv_1": "b_rv1", "rv_4": "b_rv4", "rv_13": "b_rv13"}
    out.update({v: f"c_{v}" for v in x_vars})
    return [out[c] for c in cols]


def harx_ols(panel, end, rv_col="rv", x_vars=(), start=None):
    """OLS of rv on the HAR(-X) design; returns the statsmodels RegressionResults."""
    start = C.SAMPLE_START if start is None else start
    X, y, _ = D.har_design(panel, end, rv_col=rv_col, x_vars=tuple(x_vars), start=start)
    return sm.OLS(y.values, X.values).fit()


def harx_forecast(panel, end, rv_col="rv", x_vars=(), start=None) -> dict:
    """One-week-ahead variance forecast, floored at RV_FLOOR_FRAC x min(rv_col) over TRAIN within [start, end]."""
    x_vars = tuple(x_vars)
    start = C.SAMPLE_START if start is None else start
    X, y, xn = D.har_design(panel, end, rv_col=rv_col, x_vars=x_vars, start=start)
    b = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    d = pd.to_datetime(panel["date"])
    win = C.TRAIN if rv_col == "rv" else C.STANDARDIZE_WINDOW_HOURLY
    lo = max(pd.Timestamp(start), pd.Timestamp(win[0]))
    hi = min(pd.Timestamp(end), pd.Timestamp(win[1]))
    m = panel.loc[(d >= lo) & (d <= hi), rv_col].min()
    if not np.isfinite(m):
        raise ValueError(f"no finite '{rv_col}' for the floor window {lo.date()}..{hi.date()}")
    floor = C.RV_FLOOR_FRAC * float(m)
    raw = float(xn[list(X.columns)].values @ b)
    return {"var_fc": max(raw, floor), "floored": raw < floor,
            "params": pd.Series(b, index=_names(X.columns, x_vars))}


def har_forecast(panel, end) -> dict:
    """Daily-RV HAR forecast."""
    return harx_forecast(panel, end, "rv")


def harx_insample(panel, end, rv_col, x_vars, start, hac_lags=4) -> pd.DataFrame:
    """Coefficients with Newey-West SEs (coef, se, t, p) plus adj_r2 and n rows."""
    x_vars = tuple(x_vars)
    start = C.SAMPLE_START if start is None else start
    X, y, _ = D.har_design(panel, end, rv_col=rv_col, x_vars=x_vars, start=start)
    res = sm.OLS(y.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    names = _names(X.columns, x_vars)
    t = pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues}, index=names)
    t.loc["adj_r2", "coef"] = res.rsquared_adj
    t.loc["n", "coef"] = float(res.nobs)
    return t


_GARCH_ORDER = ["mu", "omega", "alpha", "beta", "nu"]


def garch_t_forecast(r_hist, start_params=None) -> dict:
    """GARCH(1,1)-t with constant mean; var_fc = h(T+1) + mu^2 (second moment of the next return)."""
    r = np.asarray(r_hist, float)
    am = arch_model(r[~np.isnan(r)], mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    sv = None if start_params is None else np.asarray(start_params[_GARCH_ORDER], float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = am.fit(disp="off", show_warning=False, starting_values=sv)
        h = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
    p = res.params.values
    mu = float(p[0])
    return {"var_fc": h + mu ** 2, "mean_fc": mu, "params": pd.Series(p, index=_GARCH_ORDER),
            "converged": bool(res.convergence_flag == 0), "loglike": float(res.loglikelihood), "n_warnings": len(caught)}


@dataclass
class GarchXFit:
    """Fitted GARCH(1,1)-t-X: params (mu, omega, alpha, beta, gamma_*, nu), conditional variances h."""
    params: pd.Series
    loglike: float
    converged: bool
    seconds: float
    h: pd.Series
    x_names: list
    n_warnings: int = 0


def _garchx_h(theta, r, X, h1):
    """Conditional variance path h(1..T) via a linear filter."""
    k = X.shape[1]
    mu, om, al, be = theta[:4]
    g = np.asarray(theta[4:4 + k])
    eps = r - mu
    c = np.empty(len(r))
    c[0] = h1
    c[1:] = om + al * eps[:-1] ** 2 + (X[1:] @ g if k else 0.0)
    return lfilter([1.0], [1.0, -be], c)


def garchx_loglike(theta, r, X, h1: float) -> float:
    """Student-t (unit variance) GARCH-X log-likelihood; theta = [mu, omega, alpha, beta, gammas, nu]."""
    r = np.asarray(r, float)
    X = np.asarray(X, float).reshape(len(r), -1)
    nu = theta[-1]
    h = _garchx_h(theta, r, X, h1)
    if not np.all(h > 0) or not np.all(np.isfinite(h)):
        return -1e10
    e2 = (r - theta[0]) ** 2
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(h)
          - (nu + 1) / 2 * np.log1p(e2 / (h * (nu - 2))))
    return float(np.sum(ll))


def _garchx_names(x_names) -> list:
    return ["mu", "omega", "alpha", "beta", *[f"gamma_{c}" for c in x_names], "nu"]


def _slsqp(theta0, r, X, h1):
    """One bounded SLSQP run; returns (theta, loglike, success)."""
    k = X.shape[1]
    bounds = [(None, None), (1e-8, None), (0, 1), (0, 1), *[(0, None)] * k, (2.05, 100)]
    cons = [{"type": "ineq", "fun": lambda th: 0.999 - th[2] - th[3]}]
    res = minimize(lambda th: -garchx_loglike(th, r, X, h1), theta0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    return res.x, -float(res.fun), bool(res.success)


def garchx_fit(r: pd.Series, X: pd.DataFrame, start_params: pd.Series | None = None) -> GarchXFit:
    """MLE by SLSQP. Cold start = zero-X GARCH-t solution with gamma = 0; warm start retried from it."""
    t0 = time.perf_counter()
    rv = np.asarray(r, float)
    Xv = np.asarray(X, float).reshape(len(rv), -1)
    k = Xv.shape[1]
    names = _garchx_names(list(X.columns))
    h1 = float(np.var(rv))
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always")

        def nesting():
            v = h1
            th0 = np.array([rv.mean(), 0.05 * v, 0.05, 0.90, 8.0])
            th, _, _ = _slsqp(th0, rv, np.zeros((len(rv), 0)), h1)
            return np.concatenate([th[:4], np.zeros(k), th[4:]])

        cands = []
        if start_params is not None:
            th0 = start_params[names].values.astype(float)
            cands.append(_slsqp(th0, rv, Xv, h1))
        if not cands or not cands[0][2]:
            cands.append(_slsqp(nesting(), rv, Xv, h1))
        th, ll, ok = max(cands, key=lambda c: c[1])
        if start_params is not None:                     # never end below the warm start itself
            ll0 = garchx_loglike(start_params[names].values.astype(float), rv, Xv, h1)
            if ll0 > ll:
                th, ll = start_params[names].values.astype(float), ll0
    h = pd.Series(_garchx_h(th, rv, Xv, h1), index=r.index)
    return GarchXFit(pd.Series(th, index=names), float(ll), ok, time.perf_counter() - t0, h,
                     list(X.columns), len(caught))


def garchx_se(fit: GarchXFit, r, X) -> pd.Series:
    """Hessian SEs (approx_hess3); NaN for parameters at a bound or when the Hessian is singular."""
    rv = np.asarray(r, float)
    Xv = np.asarray(X, float).reshape(len(rv), -1)
    th = fit.params.values
    names = list(fit.params.index)
    h1 = float(np.var(rv))
    free = np.array([not (n.startswith("gamma_") and th[i] <= 1e-6) and not (n == "omega" and th[i] <= 1e-7)
                     and not (n in ("alpha", "beta") and (th[i] <= 1e-6 or th[i] >= 1 - 1e-6))
                     and not (n == "nu" and (th[i] <= 2.05 + 1e-6 or th[i] >= 100 - 1e-6)) for i, n in enumerate(names)])
    se = pd.Series(np.nan, index=names)
    idx = np.where(free)[0]

    def f(sub):
        full = th.copy()
        full[idx] = sub
        return garchx_loglike(full, rv, Xv, h1)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            H = approx_hess3(th[idx], f)
            if not np.all(np.isfinite(H)) or np.linalg.cond(H) > 1e14:
                return se
            v = np.diag(np.linalg.inv(-H))
        except np.linalg.LinAlgError:
            return se
    ok = v > 0
    se.iloc[idx[ok]] = np.sqrt(v[ok])
    return se


def garchx_forecast(fit: GarchXFit, r, x_next) -> dict:
    """h(T+1) = omega + alpha eps_T^2 + beta h_T + gamma'x_next; var_fc = h_next + mu^2."""
    p = fit.params
    eps = float(np.asarray(r, float)[-1] - p["mu"])
    g = sum(p[f"gamma_{c}"] * float(x_next[c]) for c in fit.x_names)
    h_next = float(p["omega"] + p["alpha"] * eps ** 2 + p["beta"] * fit.h.iloc[-1] + g)
    return {"h_next": h_next, "var_fc": h_next + float(p["mu"]) ** 2}


def garchx_simulate(params: pd.Series, X: pd.DataFrame, seed: int) -> pd.Series:
    """Simulate returns from the GARCH-t-X with regressor rows X (row t enters h(t))."""
    rng = np.random.default_rng(seed)
    n = len(X)
    gam = np.array([params[f"gamma_{c}"] for c in X.columns])
    Xv = np.asarray(X, float).reshape(n, -1)
    om, al, be, nu, mu = params["omega"], params["alpha"], params["beta"], params["nu"], params["mu"]
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = (om + Xv[0] @ gam) / (1 - al - be)
    r[0] = mu + np.sqrt(h[0]) * z[0]
    for t in range(1, n):
        h[t] = om + al * (r[t - 1] - mu) ** 2 + be * h[t - 1] + Xv[t] @ gam
        r[t] = mu + np.sqrt(h[t]) * z[t]
    return pd.Series(r, index=X.index)


def rw_forecast() -> float:
    """Random-walk (zero-mean) return forecast."""
    return 0.0


def model_forecast(name, panel, end, state) -> dict:
    """Dispatch a benchmark by name; reads config at call time. `state` carries warm starts and iran_var."""
    t0 = time.perf_counter()
    end = pd.Timestamp(end)
    iran = state.get("iran_var", C.IRAN_VAR)
    har = {"HAR": ("rv", (), C.SAMPLE_START),
           "HAR_h": (C.X_RV_COL, (), C.X_SAMPLE_START),
           "HARX_C": (C.X_RV_COL, tuple(C.X_CONTROLS), C.X_SAMPLE_START),
           "HARX_CI": (C.X_RV_COL, tuple(C.X_CONTROLS) + (iran,), C.X_SAMPLE_START)}
    if name in har:
        rv_col, xv, start = har[name]
        out = harx_forecast(panel, end, rv_col, xv, start)
        out.update(converged=True, loglike=np.nan, n_warnings=0)
    elif name == "GARCH_t":
        d = pd.to_datetime(panel["date"])
        out = garch_t_forecast(panel.loc[d <= end, "r"].values, state.get("GARCH_t"))
        state["GARCH_t"] = out["params"]
        out["floored"] = False
    elif name in ("GARCHX_C", "GARCHX_CI"):
        xv = tuple(C.X_CONTROLS) + ((iran,) if name == "GARCHX_CI" else ())
        r, X, x_next = D.garchx_design(panel, end, xv)
        fit = garchx_fit(r, X, state.get(name))
        state[name] = fit.params
        out = garchx_forecast(fit, r, x_next)
        out.update(params=fit.params, converged=fit.converged, loglike=fit.loglike, floored=False,
                   n_warnings=fit.n_warnings)
    elif name == "RW":
        out = {"mean_fc": rw_forecast(), "params": pd.Series(dtype=float), "converged": True,
               "loglike": np.nan, "floored": False, "n_warnings": 0}
    else:
        raise ValueError(f"unknown benchmark '{name}'")
    out["seconds"] = time.perf_counter() - t0
    return out
