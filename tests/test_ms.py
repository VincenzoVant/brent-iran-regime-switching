import numpy as np, pandas as pd, pytest
from dataclasses import replace
from types import SimpleNamespace as NS

from brent import ms, design as D, config as C


@pytest.fixture(scope="module")
def dsn(sim_panel):
    p = sim_panel(n=300, beta=2.0, seed=405)
    s = ms.make_specs(False)
    end = p["date"].iloc[-1]
    return {k: (D.build_design(p, v, end), v) for k, v in s.items()}, p


def test_param_names_layout(dsn):
    (d, s) = dsn[0]["M2"]
    m = ms.make_model(d, s)
    assert list(m.param_names) == (
        [f"{a}.tvtp{j}" for j in range(4) for a in ("p[0->0]", "p[1->0]")]
        + ["const[0]", "const[1]", "x1[1]", "sigma2[0]", "sigma2[1]"])
    assert m.tvtp_name_map == {"const": 0, "gpr": 1, "vix": 2, "log_rv": 3}
    m1 = ms.make_model(*dsn[0]["M1"])
    assert list(m1.param_names) == ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "x1[1]", "sigma2[0]", "sigma2[1]"]
    m2 = ms.make_model(d, replace(s, phi_switching=True))
    assert "x1[0]" in m2.param_names and "x1[1]" in m2.param_names


def test_readable_names_are_a_bijection(dsn):
    m = ms.make_model(*dsn[0]["M3"])
    assert len(set(m.readable.values())) == len(m.readable) == len(m.param_names)
    assert m.readable["p[0->0].tvtp4"] == "p00_iran_gdelt" and m.readable["sigma2[1]"] == "sigma2_H"
    assert m.readable["x1[1]"] == "phi" and m.readable["p[0->0].tvtp0"] == "p00_const"
    assert ms.readable_to_sm(m)["p10_vix"] == "p[1->0].tvtp2"
    m2 = ms.make_model(dsn[0]["M2"][0], replace(dsn[0]["M2"][1], phi_switching=True))
    assert m2.readable["x1[0]"] == "phi_N" and m2.readable["x1[1]"] == "phi_H"
    m1 = ms.make_model(*dsn[0]["M1"])
    assert m1.readable["p[0->0]"] == "p00" and m1.readable["const[1]"] == "mu_H"


def test_mean_exog_name_is_theta(sim_panel):
    p = sim_panel(n=200, seed=3)
    s = replace(ms.make_specs(False)["M2"], mean_exog=("iran_gdelt",))
    m = ms.make_model(D.build_design(p, s, p["date"].iloc[-1]), s)
    assert "x2[1]" in m.param_names and m.readable["x2[1]"] == "theta_iran_gdelt"


def test_overrides_pass_through(dsn):
    m = ms.make_model(*dsn[0]["M1"], switching_trend=False)
    assert "const[0]" not in m.param_names and "const" in m.param_names
    assert m.readable["const"] == "mu" and m.readable["sigma2[1]"] == "sigma2_H"
    m = ms.make_model(*dsn[0]["M1"], switching_variance=False)
    assert m.readable["sigma2"] == "sigma2" and m.readable["const[0]"] == "mu_N"


def test_phi_switching_with_mean_exog(sim_panel):
    p = sim_panel(n=200, seed=3)
    s = replace(ms.make_specs(True)["M2"], mean_exog=("iran_gdelt",))
    m = ms.make_model(D.build_design(p, s, p["date"].iloc[-1]), s)
    assert {"x1[0]", "x1[1]", "x2[1]"} <= set(m.param_names)
    assert (m.readable["x1[0]"], m.readable["x1[1]"], m.readable["x2[1]"]) == ("phi_N", "phi_H", "theta_iran_gdelt")


def _arbitrary(m, seed):
    rng = np.random.default_rng(seed)
    p = pd.Series(rng.normal(0, 0.4, len(m.param_names)), index=m.param_names)
    for n in p.index:
        if n.startswith("sigma2"): p[n] = 1.0 if n.endswith("[0]") else 4.0
        if n.startswith("p[") and ".tvtp" not in n: p[n] = 0.8 if n.startswith("p[0") else 0.3
    return p


def _flip(p, m):
    q = p.copy()
    for n in p.index:
        if n.endswith("[0]"):
            twin = n[:-3] + "[1]"; q[n], q[twin] = p[twin], p[n]
    if any(".tvtp" in n for n in p.index):
        for n in p.index:
            if n.startswith("p[0->0]."): q[n] = -p[n.replace("p[0->0]", "p[1->0]")]
            if n.startswith("p[1->0]."): q[n] = -p[n.replace("p[1->0]", "p[0->0]")]
    else:
        q["p[0->0]"], q["p[1->0]"] = 1 - p["p[1->0]"], 1 - p["p[0->0]"]
    return q


@pytest.mark.parametrize("name", ["M1", "M2", "M3"])
def test_canonicalize_preserves_ll_and_is_idempotent(dsn, name):
    m = ms.make_model(*dsn[0][name]); p = _arbitrary(m, 1); f = _flip(p, m)
    assert m.loglike(f) == pytest.approx(m.loglike(p), abs=1e-8)
    c, flipped = ms.canonicalize(f, m)
    assert flipped and c["sigma2[1]"] > c["sigma2[0]"]
    pd.testing.assert_series_equal(c, p, atol=1e-12)
    c2, again = ms.canonicalize(c, m)
    assert not again and c2.equals(c)


def test_canonicalize_by_const(dsn):
    m = ms.make_model(*dsn[0]["M2"]); p = _arbitrary(m, 2)
    p["const[0]"], p["const[1]"] = 1.0, -1.0
    c, flipped = ms.canonicalize(p, m, by="const")
    assert flipped and c["const[1]"] > c["const[0]"]


def test_transition_coefs_deck_signs():
    p = pd.Series({"p[0->0].tvtp0": 2.0, "p[1->0].tvtp0": -1.0, "p[0->0].tvtp1": -0.8, "p[1->0].tvtp1": 0.5})
    out = ms.transition_coefs_deck(p, ["const", "iran_gdelt"])
    assert out.loc["N->H", "const"] == -2.0 and out.loc["N->H", "iran_gdelt"] == 0.8
    assert out.loc["H->N", "const"] == -1.0 and out.loc["H->N", "iran_gdelt"] == 0.5


def test_expected_durations(dsn):
    m1 = ms.make_model(*dsn[0]["M1"]); p = _arbitrary(m1, 3)
    p["p[0->0]"], p["p[1->0]"] = 0.9, 0.2
    d = ms.expected_durations(p, m1)
    assert d["N"] == pytest.approx(10.0) and d["H"] == pytest.approx(5.0)
    m2 = ms.make_model(*dsn[0]["M2"]); q = _arbitrary(m2, 4)
    q["p[0->0].tvtp0"], q["p[1->0].tvtp0"] = 2.0, -1.0
    lam = lambda a: 1 / (1 + np.exp(-a))
    d2 = ms.expected_durations(q, m2)
    assert d2["N"] == pytest.approx(1 / (1 - lam(2.0))) and d2["H"] == pytest.approx(1 / lam(-1.0))


def test_constraint_info(dsn):
    m = ms.make_model(*dsn[0]["M3"]); info = ms.constraint_info(dsn[0]["M3"][1], m)
    assert info["fixed"] == {"p[1->0].tvtp4": 0.0} and info["beta"] == {"iran_gdelt": "p[0->0].tvtp4"}
    both = replace(dsn[0]["M3"][1], iran_only_NH=False)
    assert ms.constraint_info(both, m)["fixed"] == {}


@pytest.fixture(scope="module")
def small(sim_panel):
    p = sim_panel(n=300, beta=2.0, seed=405)
    end = p["date"].iloc[-1]
    s = ms.make_specs(False)
    d = {k: D.build_design(p, v, end) for k, v in s.items()}
    fits = {}
    fits["M1"] = ms.fit_ms(d["M1"], s["M1"], None, 5, 0, seed=405)
    fits["M2"] = ms.fit_ms(d["M2"], s["M2"], None, 5, 0, seed=405,
                           extra_starts=(ms.nest_start(fits["M1"], ms.make_model(d["M2"], s["M2"]), s["M2"]),))
    return p, s, d, fits


def test_fit_is_canonical_and_seeded(small):
    p, s, d, fits = small
    f = fits["M1"]
    assert f.params["sigma2[1]"] > f.params["sigma2[0]"] and f.loglike == pytest.approx(f.model.loglike(f.params))
    again = ms.fit_ms(d["M1"], s["M1"], None, 5, 0, seed=405)
    pd.testing.assert_series_equal(f.params, again.params)


def test_nesting_ll_and_restarts_counted(small):
    p, s, d, fits = small
    assert fits["M2"].loglike >= fits["M1"].loglike - 1e-6
    r = ms.fit_ms(d["M1"], s["M1"], fits["M1"].params, 0, 2, seed=7)
    assert r.n_restarts_used == 2 and r.loglike >= fits["M1"].loglike - 1e-6


def test_degeneracy_rule_and_pick_best(small):
    p, s, d, fits = small
    f = fits["M1"]; m = f.model; vy = float(m.endog.var())
    assert not ms.is_degenerate(f.params, m, vy, floor_frac=0.05, min_duration=2.0)
    spike = f.params.copy(); spike["sigma2[0]"] = 1e-6 * vy
    assert ms.is_degenerate(spike, m, vy, floor_frac=0.05, min_duration=2.0)
    short = f.params.copy(); short["p[0->0]"] = 0.2
    assert ms.is_degenerate(short, m, vy, floor_frac=0.05, min_duration=2.0)
    a, b = NS(loglike=-100.0, degenerate=True), NS(loglike=-120.0, degenerate=False)
    assert ms.pick_best([a, b]) is b
    assert ms.pick_best([a, NS(loglike=-110.0, degenerate=True)]) is a and a.degenerate


def test_nest_parent():
    s = ms.make_specs(False); t = ms.with_iran(s["M3"], C.IRAN_TONE_VARS, "M3tone")
    allsp = list(s.values()) + [t]
    assert ms.nest_parent(s["M1"], allsp) is None and ms.nest_parent(s["M2"], allsp).name == "M1"
    assert ms.nest_parent(s["M3"], allsp).name == "M2" and ms.nest_parent(t, allsp).name == "M2"


def test_nest_start_maps_names(small):
    p, s, d, fits = small
    m2 = ms.make_model(d["M2"], s["M2"])
    st = ms.nest_start(fits["M1"], m2, s["M2"])
    f1 = fits["M1"].params
    assert list(st.index) == list(m2.param_names)
    assert st["x1[1]"] == f1["x1[1]"] and st["sigma2[1]"] == f1["sigma2[1]"]
    assert st["p[0->0].tvtp1"] == 0.0 and st["p[1->0].tvtp3"] == 0.0
    assert st["p[0->0].tvtp0"] == pytest.approx(np.log(f1["p[0->0]"] / (1 - f1["p[0->0]"])))
    assert st["p[1->0].tvtp0"] == pytest.approx(np.log(f1["p[1->0]"] / (1 - f1["p[1->0]"])))
    m3 = ms.make_model(d["M3"], s["M3"])
    st3 = ms.nest_start(fits["M2"], m3, s["M3"])
    assert st3["p[0->0].tvtp4"] == 0.0 and st3["p[0->0].tvtp1"] == fits["M2"].params["p[0->0].tvtp1"]


@pytest.fixture(scope="module")
def ladder(small):
    p, s, d, fits = small
    m3m = ms.make_model(d["M3"], s["M3"])
    return ms.fit_ms(d["M3"], s["M3"], None, 5, 0, seed=405,
                     extra_starts=(ms.nest_start(fits["M2"], m3m, s["M3"]),))


def test_constrained_fit_is_exact_and_nested(small, ladder):
    p, s, d, fits = small; f3 = ladder
    assert f3.params["p[1->0].tvtp4"] == 0.0 and f3.fixed == {"p[1->0].tvtp4": 0.0}
    assert f3.loglike >= fits["M2"].loglike - 1e-6
    both = replace(s["M3"], iran_only_NH=False)
    fu = ms.fit_ms(d["M3"], both, f3.params, 0, 0, seed=405)
    assert fu.fixed == {} and fu.loglike >= f3.loglike - 1e-6
    assert f3.params["sigma2[1]"] > f3.params["sigma2[0]"]


def test_beta_sign_when_iran_raises_p_NH(sim_panel):
    p = sim_panel(n=500, beta=2.5, seed=11); s = ms.make_specs(False)
    d = {k: D.build_design(p, v, p["date"].iloc[-1]) for k, v in s.items()}
    f2 = ms.fit_ms(d["M2"], s["M2"], None, 5, 0, seed=11)
    f3 = ms.fit_ms(d["M3"], s["M3"], None, 5, 0, seed=11,
                   extra_starts=(ms.nest_start(f2, ms.make_model(d["M3"], s["M3"]), s["M3"]),))
    beta = ms.transition_coefs_deck(f3.params, list(d["M3"].Z.columns)).loc["N->H", "iran_gdelt"]
    assert beta > 0


@pytest.fixture(scope="module")
def well_identified(sim_panel):
    """n=500 M3 fit; the n=300 `small` ladder sits at a quasi-separated optimum with a singular Hessian."""
    p = sim_panel(n=500, beta=2.5, seed=11); s = ms.make_specs(False)
    d = {k: D.build_design(p, v, p["date"].iloc[-1]) for k, v in s.items()}
    f2 = ms.fit_ms(d["M2"], s["M2"], None, 5, 0, seed=11)
    return ms.fit_ms(d["M3"], s["M3"], None, 5, 0, seed=11,
                     extra_starts=(ms.nest_start(f2, ms.make_model(d["M3"], s["M3"]), s["M3"]),))


def test_param_cov_free_params_only_and_nan_not_raise(well_identified, small, ladder, monkeypatch):
    f3 = well_identified
    cov = ms.param_cov(f3.model, f3.params, f3.fixed)
    assert set(cov.index) == set(f3.params.index) - set(f3.fixed) and np.allclose(cov, cov.T, equal_nan=True)
    assert np.isfinite(np.diag(cov)).all() and (np.diag(cov) > 0).all()
    flat = f3.params.copy(); flat["sigma2[0]"] = 1e-12
    assert isinstance(ms.param_cov(f3.model, flat, f3.fixed), pd.DataFrame)
    monkeypatch.setattr(np.linalg, "cond", lambda *a, **k: 1e16)    # ill-conditioned Hessian -> NaN cov
    assert ms.param_cov(f3.model, f3.params, f3.fixed).isna().all().all()
    monkeypatch.undo()
    sing = ms.param_cov(ladder.model, ladder.params, ladder.fixed)
    assert isinstance(sing, pd.DataFrame) and sing.shape == (14, 14)


def test_warm_start_stays_constrained_and_flipped_result_triggers_refit(small, ladder, monkeypatch):
    p, s, d, fits = small; f3 = ladder
    w = ms.fit_ms(d["M3"], s["M3"], f3.params, 0, 0, seed=406)
    assert w.fixed and w.params["p[1->0].tvtp4"] == 0.0 and not w.refit_from_scratch
    real, calls = ms.fit_constrained, {"n": 0}

    def fake(model, start, fixed, n_restarts, seed):
        params, ll, ok = real(model, start, fixed, n_restarts, seed); calls["n"] += 1
        if calls["n"] == 1:
            params = params.copy(); params["sigma2[0]"], params["sigma2[1]"] = params["sigma2[1]"], params["sigma2[0]"]
        return params, ll, ok
    monkeypatch.setattr(ms, "fit_constrained", fake)
    r = ms.fit_ms(d["M3"], s["M3"], f3.params, 0, 0, seed=406)
    assert r.refit_from_scratch and r.params["sigma2[1]"] > r.params["sigma2[0]"]


def test_tone_spec_fixes_both_and_nests_in_m2(small):
    p, s, d, fits = small
    t = ms.with_iran(s["M3"], C.IRAN_TONE_VARS, "M3tone")
    dt = D.build_design(p, t, p["date"].iloc[-1]); m = ms.make_model(dt, t)
    info = ms.constraint_info(t, m)
    assert set(info["fixed"]) == {"p[1->0].tvtp4", "p[1->0].tvtp5"} and set(info["beta"]) == set(C.IRAN_TONE_VARS)
    ft = ms.fit_ms(dt, t, None, 5, 0, seed=405, extra_starts=(ms.nest_start(fits["M2"], m, t),))
    assert ft.params["p[1->0].tvtp4"] == 0.0 == ft.params["p[1->0].tvtp5"]
    assert ft.loglike >= fits["M2"].loglike - 1e-6


def test_nest_start_clips_extreme_probabilities(small):
    p, s, d, fits = small
    f = replace(fits["M1"], params=fits["M1"].params.copy())
    f.params["p[0->0]"] = 1.0
    st = ms.nest_start(f, ms.make_model(d["M2"], s["M2"]), s["M2"])
    assert np.isfinite(st["p[0->0].tvtp0"])


def test_failed_candidate_counts_as_warning(small, monkeypatch):
    p, s, d, fits = small
    real = ms._fit_once
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 2 else real(*a, **k)
    monkeypatch.setattr(ms, "_fit_once", flaky)
    f = ms.fit_ms(d["M1"], s["M1"], fits["M1"].params, 0, 1, seed=3)
    assert f.n_warnings >= 1


def test_make_model_rejects_misaligned_index(small):
    p, s, d, fits = small
    bad = NS(y=d["M2"].y, X=d["M2"].X.iloc[::-1], Z=d["M2"].Z)
    with pytest.raises(AssertionError):
        ms.make_model(bad, s["M2"])


def test_all_flipped_candidates_never_raise(small, ladder, monkeypatch):
    p, s, d, fits = small; f3 = ladder
    real = ms.fit_constrained

    def flipped(model, start, fixed, n_restarts, seed):
        params, ll, ok = real(model, start, fixed, n_restarts, seed)
        params = params.copy(); params["sigma2[0]"], params["sigma2[1]"] = params["sigma2[1]"], params["sigma2[0]"]
        return params, ll, ok
    monkeypatch.setattr(ms, "fit_constrained", flipped)
    for start in (None, f3.params):
        r = ms.fit_ms(d["M3"], s["M3"], start, 0, 0, seed=406)
        assert not r.converged and r.degenerate and r.params["p[1->0].tvtp4"] == 0.0
        assert r.params["sigma2[0]"] <= r.params["sigma2[1]"] and r.n_warnings >= 1
