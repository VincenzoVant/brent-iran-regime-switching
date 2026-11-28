"""One-step-ahead Markov-switching forecaster (spec §8.6). Filtered probabilities only, never smoothed."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from brent import config as C


@dataclass
class OneStep:
    """One-week-ahead forecast: P(high regime), variance, mean, P(|r| > thr), filtered P(H) at the origin."""

    pi_H: float
    var_fc: float
    mean_fc: float
    p_exceed: float
    xi_H_origin: float


def _transitions(model, params, exog_tvtp=None) -> np.ndarray:
    """Transition matrices as shape (2, 2, n); P[i, j] = P(S(t+1)=i | S(t)=j)."""
    kw = {} if exog_tvtp is None else {"exog_tvtp": exog_tvtp}
    P = np.asarray(model.regime_transition_matrix(params, **kw), dtype=float)
    return P[:, :, None] if P.ndim == 2 else P


def predicted_probs(model, params) -> np.ndarray:
    """Predicted regime probabilities pi(t | t-1), shape (nobs, 2), column 1 = high-volatility regime."""
    filt = np.asarray(model.filter(params).filtered_marginal_probabilities, dtype=float)
    P = _transitions(model, params)
    out = np.empty_like(filt)
    out[0] = np.asarray(model.initial_probabilities(params), dtype=float).ravel()[:2]
    for t in range(1, len(filt)):
        out[t] = P[:, :, t if P.shape[2] > 1 else 0] @ filt[t - 1]
    return out


def filter_fixed(model, params) -> pd.Series:
    """Filtered P(high regime at t | data to t) with fixed params, indexed by the model's dates."""
    filt = np.asarray(model.filter(params).filtered_marginal_probabilities, dtype=float)
    return pd.Series(filt[:, 1], index=model.data.row_labels, name="xi_H")


def highvol_threshold(panel: pd.DataFrame) -> float:
    """Percentile C.HIGHVOL_PCTL of |r| over the TRAIN window."""
    d = pd.to_datetime(panel["date"])
    m = (d >= pd.Timestamp(C.TRAIN[0])) & (d <= pd.Timestamp(C.TRAIN[1]))
    return float(np.percentile(panel.loc[m.values, "r"].abs(), C.HIGHVOL_PCTL))


def _coef(params: pd.Series, k: int, i: int) -> float:
    """Coefficient of exog column x{k}: regime i if it switches, else the lone x{k}[1]."""
    if f"x{k}[0]" in params.index and f"x{k}[1]" in params.index:
        return float(params[f"x{k}[{i}]"])
    return float(params[f"x{k}[1]"])


def one_step(fit, design, thr: float) -> OneStep:
    """Forecast week T+1 from the fit's origin T (design.x_next, design.z_next)."""
    if fit.spec.kind == "mshar":
        raise NotImplementedError("E1 is built in Plan 3")
    model, pr = fit.model, fit.params
    xi = np.asarray(model.filter(pr.values).filtered_marginal_probabilities, dtype=float)[-1]
    z = None if design.z_next is None else design.z_next.values[None, :]
    pi = _transitions(model, pr.values, z)[:, :, 0] @ xi
    m = np.empty(2)
    for i in (0, 1):
        m[i] = float(pr[f"const[{i}]"]) + _coef(pr, 1, i) * float(design.x_next["r_lag"])
        for k, v in enumerate(fit.spec.mean_exog):
            m[i] += _coef(pr, k + 2, i) * float(design.x_next[v])
    sd = np.sqrt([float(pr["sigma2[0]"]), float(pr["sigma2[1]"])])
    exc = 1 - norm.cdf((thr - m) / sd) + norm.cdf((-thr - m) / sd)
    return OneStep(pi_H=float(pi[1]), var_fc=float(pi @ (sd**2 + m**2)), mean_fc=float(pi @ m),
                   p_exceed=float(pi @ exc), xi_H_origin=float(xi[1]))
