"""Shared synthetic raw data: the same shapes as data/raw/*.csv, generated with fixed seeds."""

import numpy as np
import pandas as pd
import pytest

GDELT_VALUES = {"n_events": 100, "articles_all": 1000, "articles_iran_actor": 10, "articles_iran_actor_geo": 12,
                "events_iran_actor": 5, "articles_iran_conflict": 4, "articles_iran_coop": 5,
                "tone_x_articles_iran": -20.0, "articles_iran_tone_n": 10,
                "goldstein_x_articles_iran": 10.0, "articles_iran_goldstein_n": 10}


def _walk(n, seed, start=50.0):
    """Positive random-walk price path."""
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


@pytest.fixture
def synthetic_raw():
    bdays = pd.bdate_range("2004-12-01", "2026-11-06")
    cdays = pd.date_range("2004-12-01", "2026-11-06")
    brent = pd.DataFrame({"date": bdays, "brent": _walk(len(bdays), 0)})
    vix = pd.DataFrame({"date": bdays, "vix": _walk(len(bdays), 2, 15.0)})
    ovx_days = bdays[bdays >= "2007-05-01"]
    ovx = pd.DataFrame({"date": ovx_days, "ovx": _walk(len(ovx_days), 3, 30.0)})
    gpr = pd.DataFrame({"date": cdays, "gprd": 100.0 + np.arange(len(cdays)) % 7})
    months = pd.date_range("2004-01-01", "2026-10-01", freq="MS")
    aigpr = pd.DataFrame({"month": months, "aigpr_iran": 5.0 + np.arange(len(months)) % 3})
    gdays = pd.date_range("2005-01-01", "2026-11-06")
    gdelt = pd.DataFrame({"feed": np.where(gdays < "2013-04-01", "backfile", "daily"), "date": gdays,
                          **GDELT_VALUES})
    idays = pd.bdate_range("2006-11-17", "2026-11-06")
    rng = np.random.default_rng(1)
    o = _walk(len(idays), 4, 60.0)
    c = o * np.exp(rng.normal(0, 0.01, len(idays)))
    intraday = pd.DataFrame({"date": idays, "rv_intraday": rng.uniform(0.5, 3.0, len(idays)), "n_hours": 22,
                             "open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                             "close": c})
    return {"brent": brent, "vix": vix, "ovx": ovx, "gpr": gpr, "aigpr": aigpr, "gdelt": gdelt,
            "intraday": intraday}


def _ar1(rng, n, phi=0.8, sd=0.5):
    """AR(1) path started at zero."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sd)
    return x


@pytest.fixture(scope="session")
def sim_panel():
    """Factory for a simulated weekly panel with a two-state TVTP Markov chain (state 1 = high vol)."""
    from brent import config as C

    def make(n=400, beta=1.5, phi=(0.3, 0.3), seed=405):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2005-01-07", periods=n, freq="7D")
        reg = {k: _ar1(rng, n) for k in ["gpr", "vix", "iran_gdelt", "iran_conflict", "iran_coop",
                                         "iran_tone", "iran_goldstein"]}
        lam = lambda x: 1.0 / (1.0 + np.exp(-x))
        mu, sig, phi = np.array([0.2, -0.3]), np.array([1.0, 3.0]), np.asarray(phi, float)
        s = np.zeros(n, dtype=int)
        r = np.zeros(n)
        r[0] = mu[0] + sig[0] * rng.normal()
        for t in range(1, n):
            p_nh = lam(-2.2 + beta * reg["iran_gdelt"][t - 1] + 0.3 * reg["gpr"][t - 1])
            p_hn = lam(-1.0 + 0.2 * reg["vix"][t - 1])
            u = rng.uniform()
            s[t] = (u < p_nh) if s[t - 1] == 0 else (u >= p_hn)
            r[t] = mu[s[t]] + phi[s[t]] * r[t - 1] + sig[s[t]] * rng.normal()
        rv = sig[s] ** 2 * rng.chisquare(1, n) + 0.05
        rv_h = 1.1 * rv * rng.lognormal(0, 0.2, n)
        rv_h[dates < pd.Timestamp(C.HOURLY_SAMPLE_START)] = np.nan
        return pd.DataFrame({"date": dates, "r": r, "rv": rv, "log_rv": np.log(rv), "gpr": reg["gpr"],
                             "vix": reg["vix"], "iran_gdelt": reg["iran_gdelt"],
                             "iran_conflict": reg["iran_conflict"], "iran_coop": reg["iran_coop"],
                             "iran_tone": reg["iran_tone"], "iran_goldstein": reg["iran_goldstein"],
                             "aigpr_iran": rng.normal(size=n), "rv_hourly": rv_h,
                             "log_rv_hourly": np.log(rv_h)})
    return make
