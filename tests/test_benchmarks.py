import numpy as np, pandas as pd, pytest
from brent import benchmarks as B, design as D, config as C

END = "2018-12-28"


def test_harx_without_x_equals_har(sim_panel):
    p = sim_panel(n=1140)
    assert B.harx_forecast(p, END, "rv", ())["var_fc"] == B.har_forecast(p, END)["var_fc"]


def test_harx_ols_forecast_equals_lstsq(sim_panel):
    p = sim_panel(n=1140)
    X, y, xn = D.har_design(p, END, rv_col=C.X_RV_COL, x_vars=("gpr", "vix"), start=C.X_SAMPLE_START)
    b = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    out = B.harx_forecast(p, END, C.X_RV_COL, ("gpr", "vix"), C.X_SAMPLE_START)
    assert list(out["params"].index) == ["b0", "b_rv1", "b_rv4", "b_rv13", "c_gpr", "c_vix"]
    assert not out["floored"] and out["var_fc"] == pytest.approx(float(xn.values @ b))


def test_floor_applies_and_is_flagged(sim_panel, monkeypatch):
    p = sim_panel(n=1140); rng = np.random.default_rng(0)
    idx = pd.date_range("2006-01-06", periods=200, freq="W-FRI"); u = rng.uniform(0.2, 1.0, 200)
    X = pd.DataFrame({"const": 1.0, "rv_1": u, "rv_4": rng.uniform(0.2, 1, 200), "rv_13": rng.uniform(0.2, 1, 200)}, index=idx)
    y = pd.Series(1.0 - 0.5 * u, index=idx)
    xn = pd.Series({"const": 1.0, "rv_1": 10.0, "rv_4": 0.5, "rv_13": 0.5})       # OLS forecast = 1 - 5 < 0
    monkeypatch.setattr(B.D, "har_design", lambda *a, **k: (X, y, xn))
    out = B.harx_forecast(p, "2018-12-28", "rv", ())
    floor = C.RV_FLOOR_FRAC * p.loc[(p["date"] >= C.SAMPLE_START) & (p["date"] <= C.TRAIN[1]), "rv"].min()
    assert out["floored"] and out["var_fc"] == pytest.approx(floor)


def test_harx_insample_has_nw_se_and_summary_rows(sim_panel):
    p = sim_panel(n=1140)
    t = B.harx_insample(p, "2025-12-26", C.X_RV_COL, ("gpr", "vix"), C.X_SAMPLE_START)
    assert {"b0", "c_gpr", "adj_r2", "n"} <= set(t.index) and (t.loc["c_gpr", "se"] > 0)


def test_fallback_branch_runs_on_daily_rv(sim_panel, monkeypatch):    # no-intraday fallback, spec §11
    p = sim_panel(n=1140).assign(rv_hourly=np.nan)
    monkeypatch.setattr(C, "X_RV_COL", "rv"); monkeypatch.setattr(C, "X_SAMPLE_START", C.SAMPLE_START)
    monkeypatch.setattr(C, "GARCHX_SCALE_WINDOW", C.TRAIN)    # a window the simulated panel covers
    out = B.model_forecast("HARX_C", p, pd.Timestamp(END), {})
    assert np.isfinite(out["var_fc"]) and out["var_fc"] > 0
    g = B.model_forecast("GARCHX_C", p, pd.Timestamp(END), {})    # defaults must be read at call time
    assert np.isfinite(g["var_fc"]) and g["var_fc"] > 0


def test_garch_t_forecast_second_moment(sim_panel):
    p = sim_panel(n=500); r = p.set_index("date")["r"]
    out = B.garch_t_forecast(r)
    assert out["converged"] and out["var_fc"] > out["mean_fc"] ** 2 and 2.0 < out["params"]["nu"] < 200
    assert 0.3 * r.var() < out["var_fc"] < 3 * r.var() and 0 < out["params"]["alpha"] + out["params"]["beta"] < 1
    warm = B.garch_t_forecast(r, start_params=out["params"])
    assert warm["var_fc"] == pytest.approx(out["var_fc"], rel=1e-3)


def test_rw_is_zero():
    assert B.rw_forecast() == 0.0


def _neg_design(monkeypatch):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2014-01-03", periods=200, freq="W-FRI"); u = rng.uniform(0.2, 1.0, 200)
    X = pd.DataFrame({"const": 1.0, "rv_1": u, "rv_4": rng.uniform(0.2, 1, 200), "rv_13": rng.uniform(0.2, 1, 200)}, index=idx)
    xn = pd.Series({"const": 1.0, "rv_1": 10.0, "rv_4": 0.5, "rv_13": 0.5})
    monkeypatch.setattr(B.D, "har_design", lambda *a, **k: (X, pd.Series(1.0 - 0.5 * u, index=idx), xn))


def test_hourly_floor_is_finite_and_flags(sim_panel, monkeypatch):
    p = sim_panel(n=1140); _neg_design(monkeypatch)
    out = B.harx_forecast(p, END, "rv_hourly", (), C.X_SAMPLE_START)
    d = p["date"]; w = (d >= C.STANDARDIZE_WINDOW_HOURLY[0]) & (d <= C.STANDARDIZE_WINDOW_HOURLY[1])
    floor = C.RV_FLOOR_FRAC * p.loc[w, "rv_hourly"].min()
    assert np.isfinite(floor) and floor > 0
    assert out["floored"] and out["var_fc"] == pytest.approx(floor)


def test_empty_floor_window_raises(sim_panel, monkeypatch):
    p = sim_panel(n=1140); _neg_design(monkeypatch)
    with pytest.raises(ValueError, match="rv_hourly"):
        B.harx_forecast(p, END, "rv_hourly", (), "2015-07-03")    # start after the hourly floor window


def test_model_forecast_reports_n_warnings(sim_panel):
    p = sim_panel(n=500)
    for name in ("HAR", "GARCH_t", "RW"):
        assert "n_warnings" in B.model_forecast(name, p, p["date"].iloc[-1], {})


# ---- GARCH-X (Task 9) ----
from scipy.special import gammaln
from brent import insample as I


def _ar_level(n, seed, phi=0.9, sd=0.3):
    rng = np.random.default_rng(seed); z = np.zeros(n)
    for t in range(1, n): z[t] = phi * z[t - 1] + rng.normal(0, sd * np.sqrt(1 - phi**2))
    return np.exp(z)                                           # stationary sd of the log = 0.3


def test_garchx_loglike_zero_x_equals_hand_computation():
    r = np.array([0.5, -1.2, 0.3, 2.0, -0.7]); th = np.array([0.1, 0.3, 0.07, 0.8, 8.0])
    mu, om, al, be, nu = th; h1 = np.var(r)
    h = [h1]; e = r - mu
    for t in range(1, 5): h.append(om + al * e[t - 1] ** 2 + be * h[-1])
    h = np.array(h)
    ll = np.sum(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(h)
                - (nu + 1) / 2 * np.log(1 + e**2 / (h * (nu - 2))))
    assert B.garchx_loglike(th, r, np.zeros((5, 0)), h1) == pytest.approx(ll)


@pytest.fixture(scope="module")
def sim_garchx():
    n = 4000; x = _ar_level(n, 1); X = pd.DataFrame({"x": x}, index=pd.date_range("2010-01-01", periods=n, freq="W-FRI"))
    true = pd.Series({"mu": 0.1, "omega": 0.3, "alpha": 0.07, "beta": 0.80, "gamma_x": 0.5, "nu": 8.0})
    r = B.garchx_simulate(true, X, seed=405)
    return r, X, true


def test_garchx_recovery(sim_garchx):
    r, X, true = sim_garchx
    fit = B.garchx_fit(r, X)
    p = fit.params
    assert abs(p["alpha"] - 0.07) < 0.04 and abs(p["beta"] - 0.80) < 0.08 and abs(p["gamma_x"] - 0.5) < 0.25
    assert 5 <= p["nu"] <= 15 and fit.converged
    th_true = true[["mu", "omega", "alpha", "beta", "gamma_x", "nu"]].values
    assert fit.loglike >= B.garchx_loglike(th_true, r.values, X.values, float(np.var(r.values))) - 1e-6


def test_irrelevant_regressor_gamma_nonneg_and_not_significant(sim_garchx):
    r, X, _ = sim_garchx
    X2 = X.assign(noise=_ar_level(len(X), 99))
    big, small = B.garchx_fit(r, X2), B.garchx_fit(r, X)
    assert big.params["gamma_noise"] >= 0
    lr = 2 * (big.loglike - small.loglike)
    assert lr >= -1e-6 and I.chibar_pvalue(max(lr, 0.0), 1) > 0.01


def test_constant_regressor_does_not_crash(sim_garchx):
    r, X, _ = sim_garchx
    fit = B.garchx_fit(r, X.assign(const=1.0))
    assert np.isfinite(fit.loglike) and fit.params["gamma_const"] >= 0
    se = B.garchx_se(fit, r, X.assign(const=1.0))
    assert isinstance(se, pd.Series)                                  # NaN allowed, exception not


def test_garchx_forecast_formula(sim_garchx):
    r, X, _ = sim_garchx; fit = B.garchx_fit(r.iloc[:1500], X.iloc[:1500])
    xn = pd.Series({"x": 1.3}); p = fit.params
    out = B.garchx_forecast(fit, r.iloc[:1500], xn)
    eps = r.iloc[1499] - p["mu"]
    h_next = p["omega"] + p["alpha"] * eps**2 + p["beta"] * fit.h.iloc[-1] + p["gamma_x"] * 1.3
    assert out["h_next"] == pytest.approx(h_next) and out["var_fc"] == pytest.approx(h_next + p["mu"] ** 2)


def test_garchx_nesting_via_zero_x_start(sim_panel):
    p = sim_panel(n=1140)
    r, X, _ = D.garchx_design(p, "2018-12-28", ("gpr", "vix"))
    g0 = B.garchx_fit(r, X.iloc[:, :0]); g1 = B.garchx_fit(r, X)
    assert g1.loglike >= g0.loglike - 1e-6


def test_garchx_model_forecast_and_warm_state(sim_panel):
    p = sim_panel(n=1140); state = {}
    a = B.model_forecast("GARCHX_CI", p, pd.Timestamp("2018-12-28"), state)
    assert "GARCHX_CI" in state and a["var_fc"] > 0 and {"gamma_iran_gdelt", "nu"} <= set(a["params"].index)
    assert {"converged", "loglike", "floored", "seconds", "n_warnings"} <= set(a)
    b = B.model_forecast("GARCHX_CI", p, pd.Timestamp("2019-01-04"), state)
    assert np.isfinite(b["var_fc"])


def test_garchx_model_forecast_returns_mean(sim_panel):
    p = sim_panel(n=1140)
    for name in ("GARCHX_C", "GARCHX_CI"):
        out = B.model_forecast(name, p, pd.Timestamp("2018-12-28"), {})
        assert out["mean_fc"] == out["params"]["mu"]


def test_garchx_kept_warm_start_reports_warm_runs_flag(sim_garchx, monkeypatch):
    r, X, true = sim_garchx
    r, X = r.iloc[:800], X.iloc[:800]
    start = B.garchx_fit(r, X).params
    calls = {"n": 0}

    def worse(theta0, rr, XX, h1):                       # warm run fails, the cold runs "succeed" but are worse
        calls["n"] += 1
        return np.asarray(theta0, float) * 0 + np.r_[0.0, 1.0, 0.0, 0.0, [0.0] * XX.shape[1], 50.0], (-1e9 if calls["n"] == 1 else -5e8), calls["n"] > 1
    monkeypatch.setattr(B, "_slsqp", worse)
    fit = B.garchx_fit(r, X, start)
    assert np.allclose(fit.params.values, start.values) and not fit.converged
