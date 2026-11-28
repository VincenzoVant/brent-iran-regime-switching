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
    ctrl = {f"p[1->0].tvtp{j}": 0.0 for j in (1, 2, 3)}                  # CONTROLS_ONLY_NH (default True)
    assert info["fixed"] == {**ctrl, "p[1->0].tvtp4": 0.0} and info["beta"] == {"iran_gdelt": "p[0->0].tvtp4"}
    both = replace(dsn[0]["M3"][1], iran_only_NH=False)
    assert ms.constraint_info(both, m)["fixed"] == ctrl
    old = replace(dsn[0]["M3"][1], controls_only_NH=False)
    assert ms.constraint_info(old, m)["fixed"] == {"p[1->0].tvtp4": 0.0}
    assert ms.constraint_info(replace(old, iran_only_NH=False), m)["fixed"] == {}
    m2 = ms.make_model(*dsn[0]["M2"])
    assert ms.constraint_info(dsn[0]["M2"][1], m2)["fixed"] == ctrl
    assert ms.constraint_info(dsn[0]["M1"][1], ms.make_model(*dsn[0]["M1"]))["fixed"] == {}


def test_controls_only_nh_spec_field(monkeypatch):
    assert C.CONTROLS_ONLY_NH is True
    s = ms.make_specs(False)
    assert all(v.controls_only_NH for v in s.values())
    assert ms.with_iran(s["M3"], C.IRAN_TONE_VARS, "M3tone").controls_only_NH
    monkeypatch.setattr(C, "CONTROLS_ONLY_NH", False)                     # default read at construction
    assert not ms.make_specs(False)["M2"].controls_only_NH
    assert ms.make_specs(False, controls_only_NH=True)["M2"].controls_only_NH


def test_m2_is_fitted_constrained_with_controls_fixed(small):
    p, s, d, fits = small
    f2 = fits["M2"]
    assert f2.fixed == {f"p[1->0].tvtp{j}": 0.0 for j in (1, 2, 3)}
    assert all(f2.params[n] == 0.0 for n in f2.fixed) and f2.loglike >= fits["M1"].loglike - 1e-6
    w = ms.fit_ms(d["M2"], s["M2"], f2.params, 0, 0, seed=406)            # warm path keeps them at 0
    assert all(w.params[n] == 0.0 for n in f2.fixed) and w.fixed == f2.fixed


def test_nest_start_zeroes_fixed_coefficients(small):
    p, s, d, fits = small
    old = replace(s["M2"], controls_only_NH=False)
    f_old = ms.fit_ms(d["M2"], old, None, 2, 0, seed=405)
    m3 = ms.make_model(d["M3"], s["M3"])
    st = ms.nest_start(f_old, m3, s["M3"])
    assert all(st[f"p[1->0].tvtp{j}"] == 0.0 for j in (1, 2, 3, 4))


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


def test_pick_best_prefers_converged_then_loglike():
    F = lambda ll, deg, conv: NS(loglike=ll, degenerate=deg, converged=conv)
    good_nc, good_c = F(-100.0, False, False), F(-105.0, False, True)
    assert ms.pick_best([good_nc, good_c]) is good_c                       # converged beats higher LL
    assert ms.pick_best([F(-90.0, True, True), good_nc, good_c]) is good_c  # non-degenerate first
    best_c = F(-101.0, False, True)
    assert ms.pick_best([good_c, best_c, good_nc]) is best_c               # then loglike
    d_nc, d_c = F(-100.0, True, False), F(-120.0, True, True)
    assert ms.pick_best([d_nc, d_c]) is d_c                                # all degenerate: same rule


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
    assert f3.params["p[1->0].tvtp4"] == 0.0 and f3.fixed == {f"p[1->0].tvtp{j}": 0.0 for j in (1, 2, 3, 4)}
    assert f3.loglike >= fits["M2"].loglike - 1e-6
    both = replace(s["M3"], iran_only_NH=False, controls_only_NH=False)
    fu = ms.fit_ms(d["M3"], both, f3.params, 0, 0, seed=405)
    assert fu.fixed == {} and fu.loglike >= f3.loglike - 1e-6
    assert f3.params["sigma2[1]"] > f3.params["sigma2[0]"]


def test_warm_constrained_fit_uses_extra_starts(small, ladder, monkeypatch):
    """A warm-started constrained fit also tries extra starts (the OOS loop's "nesting" restart) and keeps
    the best canonical, non-degenerate result; without extra starts it is unchanged."""
    p, s, d, fits = small
    nest = ms.nest_start(fits["M2"], ms.make_model(d["M3"], s["M3"]), s["M3"])
    poor = ladder.params.copy(); poor["sigma2[1]"] = poor["sigma2[0]"] * 1.01
    real, seen = ms.fit_constrained, []

    def spy(model, start, fixed, n_restarts, seed):
        out = real(model, start, fixed, n_restarts, seed)
        seen.append(start.copy())
        if start.equals(poor.reindex(start.index)):              # pretend the warm start got stuck low
            res = ms._Counted((out[0], out[1] - 50.0, out[2])); res.n_warnings = out.n_warnings
            return res
        return out

    monkeypatch.setattr(ms, "fit_constrained", spy)
    alone = ms.fit_ms(d["M3"], s["M3"], poor, 0, 0, seed=405)
    assert len(seen) == 1
    seen.clear()
    both = ms.fit_ms(d["M3"], s["M3"], poor, 0, 0, seed=405, extra_starts=(nest,))
    assert len(seen) == 2 and seen[1].equals(nest.reindex(seen[1].index))
    assert both.loglike > alone.loglike + 40 and not both.refit_from_scratch
    assert both.params["p[1->0].tvtp1"] == 0.0 and both.params["p[1->0].tvtp4"] == 0.0


def test_bounded_mle_caps_separated_slopes(small, ladder, monkeypatch):
    """n=300 sample is quasi-separated: unbounded slopes run off; the bounded fit stops them at +-MAX_TVTP_COEF."""
    p, s, d, fits = small; f3 = ladder
    cap = C.MAX_TVTP_COEF
    capped = ms.at_cap(f3.params)
    assert capped and all(abs(f3.params[n]) == cap for n in capped) and np.isfinite(f3.loglike)
    assert not f3.degenerate and all(".tvtp0" not in n for n in capped)
    slopes = [n for n in f3.params.index if ".tvtp" in n and not n.endswith(".tvtp0")]
    assert (f3.params[slopes].abs() <= cap).all()
    cov = ms.param_cov(f3.model, f3.params, f3.fixed)
    assert all(np.isnan(cov.loc[n, n]) for n in capped)                    # no SE at the bound
    nest = ms.nest_start(fits["M2"], ms.make_model(d["M3"], s["M3"]), s["M3"])
    monkeypatch.setattr(C, "MAX_TVTP_COEF", 1e9)                           # effectively unbounded
    fu = ms.fit_ms(d["M3"], s["M3"], None, 5, 0, seed=405, extra_starts=(nest,))
    assert fu.params[slopes].abs().max() > cap                             # the unbounded fit runs off
    monkeypatch.setattr(C, "MAX_TVTP_COEF", cap)
    assert ms.is_degenerate(fu.params, fu.model, float(np.var(fu.model.endog)))
    far = f3.params.copy(); far[slopes[0]] = 1e4                           # a start outside the bounds is clipped
    fw = ms.fit_ms(d["M3"], s["M3"], far, 0, 1, seed=405)
    assert (fw.params[slopes].abs() <= cap).all()


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
    assert isinstance(sing, pd.DataFrame) and sing.shape == (11, 11)              # 15 params, 4 fixed


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
    assert set(info["fixed"]) == {f"p[1->0].tvtp{j}" for j in range(1, 6)} and set(info["beta"]) == set(C.IRAN_TONE_VARS)
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


def test_near_separation_is_degenerate(small, monkeypatch):     # user decision 2026-11-27
    p, s, d, fits = small
    f = fits["M2"]; m = f.model; vy = float(m.endog.var())
    assert not ms.is_degenerate(f.params, m, vy, max_coef=50)
    sep = f.params.copy(); sep["p[0->0].tvtp1"] = -60.0
    assert ms.is_degenerate(sep, m, vy, max_coef=50) and not ms.is_degenerate(sep, m, vy, max_coef=100)
    big_const = f.params.copy(); big_const["const[1]"] = 1e3          # only ".tvtp" names are checked
    assert not ms.is_degenerate(big_const, m, vy, max_coef=50)
    monkeypatch.setattr(C, "MAX_TVTP_COEF", 1e9)                     # read at call time
    assert not ms.is_degenerate(sep, m, vy)
    monkeypatch.setattr(C, "MAX_TVTP_COEF", 50)
    assert ms.is_degenerate(sep, m, vy)


def test_max_tvtp_coef_config():
    assert C.MAX_TVTP_COEF == 50


def test_warm_constrained_path_counts_warnings(small, ladder, monkeypatch):
    p, s, d, fits = small; f3 = ladder
    fail = lambda fun, x0, method=None, options=None, bounds=None: NS(x=np.asarray(x0, float), success=False, fun=fun(x0))
    monkeypatch.setattr(ms, "minimize", fail)                     # every constrained run reports failure
    w = ms.fit_ms(d["M3"], s["M3"], f3.params, 0, 1, seed=406)
    assert not w.converged and w.n_warnings >= 2                   # main run + 1 restart, both failed
