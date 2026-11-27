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
