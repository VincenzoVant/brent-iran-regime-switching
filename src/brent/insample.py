"""In-sample tests and fits: LR, Wald, parametric bootstrap, model ladder, coefficient tables (spec §8.9)."""

from math import comb
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from brent import config as C
from brent import design as D
from brent import benchmarks as B
from brent import ms

NAN = float("nan")


def _k_free(fit) -> int:
    """Number of free parameters."""
    return len(fit.params) - len(fit.fixed)


def lr_test(fit_restricted, fit_full) -> dict:
    """Likelihood-ratio test; a negative raw LR is flagged and clipped to zero."""
    lr_raw = 2.0 * (fit_full.loglike - fit_restricted.loglike)
    kr, kf = _k_free(fit_restricted), _k_free(fit_full)
    df = kf - kr
    stat = max(lr_raw, 0.0)
    p = float(chi2.sf(stat, df)) if df > 0 else NAN
    return {"lr_raw": lr_raw, "stat": stat, "df": df, "pvalue": p, "negative": bool(lr_raw < 0),
            "k_restricted": kr, "k_full": kf}


def chibar_pvalue(lr: float, k: int) -> float:
    """Tail probability of the chi-bar-square mixture with binomial(k, 1/2) weights."""
    tot = 0.0
    for j in range(k + 1):
        tail = (1.0 if lr <= 0 else 0.0) if j == 0 else float(chi2.sf(lr, j))
        tot += comb(k, j) / 2**k * tail
    return tot


def _wald_diff(diff: float, var: float) -> dict:
    """Wald test of a scalar difference; NaN everywhere if the variance is unusable."""
    if not np.isfinite(var) or var <= 0 or not np.isfinite(diff):
        return {"stat": NAN, "pvalue": NAN, "df": 1, "diff": diff, "se": NAN}
    stat = diff**2 / var
    return {"stat": float(stat), "pvalue": float(chi2.sf(stat, 1)), "df": 1, "diff": float(diff),
            "se": float(np.sqrt(var))}


def wald_equal(fit, a: str, b: str, _cov=None, _beta=None) -> dict:
    """Wald test of H0 beta_a = beta_b for two restricted variables (deck sign: beta = -coef of p[0->0])."""
    beta = _beta if _beta is not None else ms.constraint_info(fit.spec, fit.model)["beta"]
    cov = _cov if _cov is not None else ms.param_cov(fit.model, fit.params, fit.fixed)
    na, nb = beta[a], beta[b]
    diff = -(fit.params[na] - fit.params[nb])
    var = cov.loc[na, na] + cov.loc[nb, nb] - 2 * cov.loc[na, nb]
    return _wald_diff(float(diff), float(var))


def circular_shift(series: pd.Series, k: int, window: tuple[str, str]) -> pd.Series:
    """Roll the values inside the window by k places; values outside are unchanged."""
    out = series.copy()
    m = (series.index >= pd.Timestamp(window[0])) & (series.index <= pd.Timestamp(window[1]))
    out.iloc[np.flatnonzero(m)] = np.roll(series.values[m], k)
    return out


def _sm_name(model, readable: str) -> str:
    """statsmodels parameter name for a readable name."""
    return ms.readable_to_sm(model)[readable]


def _fit_plain(spec, model, seed):
    """Unrestricted-style fit with the standard first-fit search effort."""
    return ms.fit_model(model, spec, None, C.SEARCH_REPS_FIRST, 0, seed)


def _restricted_start(full, model_r) -> pd.Series:
    """Start for a restricted refit: the full fit with the two regimes' values replaced by their mean."""
    p = full.params
    vals = {}
    for n, v in p.items():
        vals[full.model.readable[n]] = v
    for stem, a, b in (("mu", "mu_N", "mu_H"), ("phi", "phi_N", "phi_H")):
        if a in vals and b in vals:
            vals[stem] = 0.5 * (vals[a] + vals[b])
    return pd.Series({n: vals.get(model_r.readable[n], 0.0) for n in model_r.param_names})


def equality_tests(design, spec) -> pd.DataFrame:
    """LR and Wald tests of equal intercepts (mu) and equal AR coefficients (phi) across regimes."""
    seed = C.SEED
    full = _fit_plain(spec, ms.make_model(design, spec), seed)
    cov = ms.param_cov(full.model, full.params, full.fixed)
    rows = {}
    for key, kw, (na, nb) in (("mu", {"switching_trend": False}, ("const[0]", "const[1]")),
                              ("phi", {"switching_exog": False}, ("x1[0]", "x1[1]"))):
        model_r = ms.make_model(design, spec, **kw)
        fr = _fit_plain(spec, model_r, seed)
        res = lr_test(fr, full)
        if res["negative"]:
            alt = ms.fit_model(model_r, spec, _restricted_start(full, model_r), 0, 0, seed)
            if alt.loglike > fr.loglike and (not alt.degenerate or fr.degenerate):
                fr, res = alt, lr_test(alt, full)
        w = _wald_diff(float(full.params[na] - full.params[nb]),
                       float(cov.loc[na, na] + cov.loc[nb, nb] - 2 * cov.loc[na, nb]))
        rows[key] = {"est_N": full.params[na], "est_H": full.params[nb], "lr": res["stat"], "df": res["df"],
                     "p_lr": res["pvalue"], "wald": w["stat"], "p_wald": w["pvalue"], "lr_raw": res["lr_raw"]}
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["est_N", "est_H", "lr", "df", "p_lr", "wald", "p_wald", "lr_raw"])


def _ar1_ols(y: np.ndarray, x: np.ndarray):
    """Gaussian AR(1) MLE (= OLS): intercept, phi, sigma2, loglike."""
    A = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    s2 = float(np.mean((y - A @ coef) ** 2))
    ll = -0.5 * len(y) * (np.log(2 * np.pi * s2) + 1.0)
    return float(coef[0]), float(coef[1]), s2, ll


def bootstrap_lr_1v2(y: pd.Series, X: pd.DataFrame, n_reps: int, seed: int, search_reps: int) -> dict:
    """Parametric bootstrap of the LR of one-regime AR(1) vs two-regime M1 (phi switching)."""
    spec = ms.make_specs(True)["M1"]
    n = len(y)
    c, phi, s2, ll_one = _ar1_ols(y.values, X["r_lag"].values)

    def two(yv, xv, sd):
        d = SimpleNamespace(y=pd.Series(yv, index=y.index), X=pd.DataFrame({"r_lag": xv}, index=y.index), Z=None)
        return ms.fit_model(ms.make_model(d, spec), spec, None, search_reps, 0, sd)

    f = two(y.values, X["r_lag"].values, seed)
    lr = max(2 * (f.loglike - ll_one), 0.0)
    lr_boot, n_deg = np.full(n_reps, np.nan), 0
    for b in range(n_reps):
        rng = np.random.default_rng(seed + b)
        eps = rng.normal(size=n)
        yb = np.empty(n)
        prev = float(X["r_lag"].iloc[0])
        for t in range(n):
            yb[t] = c + phi * prev + np.sqrt(s2) * eps[t]
            prev = yb[t]
        xb = np.concatenate([[X["r_lag"].iloc[0]], yb[:-1]])
        _, _, _, llb = _ar1_ols(yb, xb)
        try:
            fb = two(yb, xb, seed + b)
            lr_boot[b] = max(2 * (fb.loglike - llb), 0.0)
            n_deg += bool(fb.degenerate)
        except RuntimeError:
            pass  # failed replicate stays NaN and is excluded from the p-value
    valid = ~np.isnan(lr_boot)
    pval = (1 + int(np.sum(lr_boot[valid] >= lr))) / (int(valid.sum()) + 1)
    return {"lr": float(lr), "pvalue": float(pval), "n_reps": n_reps, "n_degenerate": n_deg, "lr_boot": lr_boot,
            "n_failed": int((~valid).sum()), "n_valid": int(valid.sum()), "degenerate_obs": bool(f.degenerate),
            "ll_one": ll_one, "ll_two": f.loglike}


def _nested_fit(panel, end, spec, parent, n_restarts):
    """Fit `spec` with a fresh search plus a nesting start from `parent`."""
    d = D.build_design(panel, spec, end)
    m = ms.make_model(d, spec)
    starts = (ms.nest_start(parent, m, spec),) if parent is not None else ()
    return ms.fit_model(m, spec, None, C.SEARCH_REPS_FIRST, n_restarts, C.SEED, starts)


def fit_ladder(panel, end, phi_switching: bool, iran_var: str | None = None, n_restarts: int = 0) -> dict:
    """M1, M2, M3 with nesting starts (M2 from M1, M3 from M2)."""
    specs = ms.make_specs(phi_switching, C.M3_VARIANT, iran_var=C.IRAN_VAR if iran_var is None else iran_var)
    out, parent = {}, None
    for k in ("M1", "M2", "M3"):
        out[k] = parent = _nested_fit(panel, end, specs[k], parent, n_restarts)
    return out


def fit_tone(panel, end, phi_switching: bool, ladder: dict, n_restarts: int = 0,
             variants=("M3tone", "M3toneL", "M3gold")) -> dict:
    """Tone variants of M3: M3tone from M2; M3toneL and M3gold from M3."""
    m3 = ladder["M3"].spec
    defs = {"M3tone": (C.IRAN_TONE_VARS, "M2"), "M3toneL": (("iran_gdelt", "iran_tone"), "M3"),
            "M3gold": (("iran_gdelt", "iran_goldstein"), "M3")}
    return {k: _nested_fit(panel, end, ms.with_iran(m3, defs[k][0], k), ladder[defs[k][1]], n_restarts)
            for k in variants}


def coef_table(fit, label: str) -> pd.DataFrame:
    """One row per transition coefficient in deck signs, with SEs from the numerical Hessian."""
    cols = ["model", "logit", "var", "coef", "se", "z", "p", "fixed"]
    tv = list(getattr(fit.model, "tvtp_name_map", {}) or {})
    if not tv:
        return pd.DataFrame(columns=cols)
    cov = ms.param_cov(fit.model, fit.params, fit.fixed)
    rows = []
    for logit, stem, sign in (("N->H", "p[0->0]", -1.0), ("H->N", "p[1->0]", 1.0)):
        for j, v in enumerate(tv):
            n = f"{stem}.tvtp{j}"
            if n in fit.fixed:
                rows.append([label, logit, v, 0.0, NAN, NAN, NAN, True])
                continue
            se = float(np.sqrt(cov.loc[n, n])) if n in cov.index else NAN
            coef = sign * float(fit.params[n])
            z = coef / se if np.isfinite(se) and se > 0 else NAN
            p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else NAN
            rows.append([label, logit, v, coef, se, z, p, False])
    return pd.DataFrame(rows, columns=cols)


def fit_summary(fit, nobs: int) -> dict:
    """Log-likelihood, information criteria, durations at standardized regressors, and regime moments."""
    k, ll = _k_free(fit), float(fit.loglike)
    dur = ms.expected_durations(fit.params.values, fit.model)
    rd = {fit.model.readable[n]: v for n, v in fit.params.items()}
    mu = lambda s: rd.get(f"mu_{s}", rd.get("mu", NAN))
    return {"loglike": ll, "k": k, "aic": 2 * k - 2 * ll, "bic": k * np.log(nobs) - 2 * ll,
            "dur_N": dur["N"], "dur_H": dur["H"], "sigma2_N": rd["sigma2_N"], "sigma2_H": rd["sigma2_H"],
            "mu_N": mu("N"), "mu_H": mu("H")}


def garchx_lr(panel, end, x_small, x_big) -> dict:
    """Chi-bar-square LR of GARCH-X: small (None = GARCH-t, zero X columns) vs big, same sample.

    Big starts from small's params with zero gamma for the added regressors, so LL(big) >= LL(small).
    """
    r, Xb, _ = D.garchx_design(panel, end, tuple(x_big))
    if x_small is None:
        Xs = Xb.iloc[:, :0]
    else:
        _, Xs, _ = D.garchx_design(panel, end, tuple(x_small))
    small = B.garchx_fit(r, Xs)
    start = pd.Series({n: (small.params[n] if n in small.params.index else 0.0)
                       for n in B._garchx_names(list(Xb.columns))})
    big = B.garchx_fit(r, Xb, start)
    added = [c for c in Xb.columns if c not in Xs.columns]
    lr = 2.0 * (big.loglike - small.loglike)
    se = B.garchx_se(big, r, Xb)
    g = [f"gamma_{c}" for c in added]
    return {"lr": lr, "k": len(added), "pvalue_chibar": chibar_pvalue(max(lr, 0.0), len(added)),
            "gammas": big.params[g], "ses": se[g], "ll_small": small.loglike, "ll_big": big.loglike}
