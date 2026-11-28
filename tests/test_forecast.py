import re, pathlib
import numpy as np, pandas as pd, pytest
from dataclasses import replace
from scipy.stats import norm
from brent import ms, design as D, forecast as F, config as C

@pytest.fixture(scope="module")
def fitted(sim_panel):
    p = sim_panel(n=300, beta=2.0); out = {}
    for phi in (False, True):
        s = ms.make_specs(phi)
        for k in ("M1", "M2", "M3"):
            d = D.build_design(p, s[k], p["date"].iloc[-1])
            out[(k, phi)] = (d, s[k], ms.fit_ms(d, s[k], None, 3, 0, seed=405))
    return p, out

@pytest.mark.parametrize("key", [("M1", False), ("M2", True), ("M3", False)])
def test_predicted_probs_match_statsmodels(fitted, key):
    d, s, f = fitted[1][key]
    ours = F.predicted_probs(f.model, f.params)
    theirs = f.model.smooth(f.params).predicted_marginal_probabilities
    assert np.max(np.abs(ours - np.asarray(theirs))) < 1e-10 and np.allclose(ours.sum(axis=1), 1.0)

@pytest.mark.parametrize("key", [("M1", False), ("M2", True)])
def test_one_step_matches_hand_computation(fitted, key):
    d, s, f = fitted[1][key]; thr = 2.0
    o = F.one_step(f, d, thr); pr = f.params
    xi = np.asarray(f.model.filter(pr).filtered_marginal_probabilities)[-1]
    P = f.model.regime_transition_matrix(pr, **({} if d.z_next is None else {"exog_tvtp": d.z_next.values[None, :]}))
    P = np.asarray(P)[:, :, 0] if P.ndim == 3 else np.asarray(P)
    pi = P @ xi
    phi = [pr["x1[0]"], pr["x1[1]"]] if s.phi_switching else [pr["x1[1]"]] * 2
    m = np.array([pr[f"const[{i}]"] + phi[i] * d.x_next["r_lag"] for i in (0, 1)])
    sd = np.sqrt([pr["sigma2[0]"], pr["sigma2[1]"]])
    assert o.pi_H == pytest.approx(pi[1]) and 0 <= o.pi_H <= 1
    assert o.var_fc == pytest.approx(float(pi @ (sd**2 + m**2)))
    assert o.mean_fc == pytest.approx(float(pi @ m)) and o.xi_H_origin == pytest.approx(xi[1])
    assert o.p_exceed == pytest.approx(float(pi @ (1 - norm.cdf((thr - m) / sd) + norm.cdf((-thr - m) / sd))))

def test_filter_fixed_is_causal(fitted):
    p, out = fitted; d, s, f = out[("M2", False)]
    short_end = p["date"].iloc[200]
    d2 = D.build_design(p, s, short_end); m2 = ms.make_model(d2, s)
    a = F.filter_fixed(m2, f.params); b = F.filter_fixed(f.model, f.params)
    assert np.max(np.abs(a.values - b.loc[a.index].values)) < 1e-10        # appending data changes no past value

def test_highvol_threshold_uses_train_only(sim_panel):
    p = sim_panel(n=1140); q = p.copy(); q.loc[q["date"] > C.TRAIN[1], "r"] *= 10
    assert F.highvol_threshold(p) == F.highvol_threshold(q)
    tr = p.loc[(p["date"] >= C.TRAIN[0]) & (p["date"] <= C.TRAIN[1]), "r"].abs()
    assert F.highvol_threshold(p) == pytest.approx(np.percentile(tr, C.HIGHVOL_PCTL))

def test_forecast_module_never_uses_smoothed_probabilities():
    src = pathlib.Path(F.__file__).read_text()
    assert not re.search(r"smoothed_marginal|\.smooth\(", src)
