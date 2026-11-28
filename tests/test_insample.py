from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from brent import config as C
from brent import design as D
from brent import insample as I
from brent import ms


@pytest.fixture(scope="module", autouse=True)
def fast_search():
    """Keep ladder fits cheap in tests."""
    mp = pytest.MonkeyPatch()
    mp.setattr(C, "SEARCH_REPS_FIRST", 3)
    yield
    mp.undo()


def test_chibar_values():
    assert I.chibar_pvalue(2.706, 1) == pytest.approx(0.05, abs=2e-4)
    assert I.chibar_pvalue(0.0, 3) == 1.0
    lr = 4.0
    expect = 0.375 * chi2.sf(lr, 1) + 0.375 * chi2.sf(lr, 2) + 0.125 * chi2.sf(lr, 3)
    assert I.chibar_pvalue(lr, 3) == pytest.approx(expect)


def test_lr_test_df_from_free_params_and_negative_clip():
    big = NS(loglike=-100.0, params=pd.Series(np.zeros(12)), fixed={"a": 0.0})
    small = NS(loglike=-103.0, params=pd.Series(np.zeros(10)), fixed={})
    r = I.lr_test(small, big)
    assert r["df"] == 1 and r["lr_raw"] == pytest.approx(6.0) and r["pvalue"] == pytest.approx(chi2.sf(6.0, 1))
    neg = I.lr_test(NS(loglike=-99.0, params=small.params, fixed={}), big)
    assert neg["negative"] and neg["stat"] == 0.0 and neg["pvalue"] == 1.0


def test_circular_shift_stays_in_window():
    s = pd.Series(np.arange(20.0), index=pd.date_range("2005-01-07", periods=20, freq="W-FRI"))
    w = (str(s.index[5].date()), str(s.index[14].date()))
    out = I.circular_shift(s, 3, w)
    assert (out.iloc[:5] == s.iloc[:5]).all() and (out.iloc[15:] == s.iloc[15:]).all()
    assert sorted(out.iloc[5:15]) == sorted(s.iloc[5:15]) and out.iloc[5] == s.iloc[12]


@pytest.fixture(scope="module")
def fits(fast_search, sim_panel):
    p = sim_panel(n=300, beta=2.0, seed=405)
    return p, I.fit_ladder(p, p["date"].iloc[-1], phi_switching=False)


def test_ladder_nesting_and_df(fits, monkeypatch):
    p, L = fits
    assert L["M1"].loglike <= L["M2"].loglike + 1e-6 and L["M2"].loglike <= L["M3"].loglike + 1e-6
    assert I.lr_test(L["M2"], L["M3"])["df"] == 1
    assert I.lr_test(L["M1"], L["M2"])["df"] == 6
    monkeypatch.setattr(C, "M3_VARIANT", "iran_both_logits")
    L2 = I.fit_ladder(p, p["date"].iloc[-1], phi_switching=False)
    assert I.lr_test(L2["M2"], L2["M3"])["df"] == 2


def test_tone_df_and_wald(fits):
    p, L = fits
    T = I.fit_tone(p, p["date"].iloc[-1], False, L)
    assert set(T) == {"M3tone", "M3toneL", "M3gold"}
    assert set(I.fit_tone(p, p["date"].iloc[-1], False, L, variants=("M3tone",))) == {"M3tone"}
    r = I.lr_test(L["M2"], T["M3tone"])
    assert r["df"] == 2 and T["M3tone"].loglike >= L["M2"].loglike - 1e-6
    w = I.wald_equal(T["M3tone"], "iran_conflict", "iran_coop")
    assert w["df"] == 1 and (np.isnan(w["stat"]) or w["stat"] >= 0)


def test_tone_lr_df_under_both_variants(sim_panel, monkeypatch):
    p = sim_panel(n=300, beta=2.0, seed=405)
    end = p["date"].iloc[-1]
    for variant, df in (("constrained", 2), ("iran_both_logits", 4)):
        monkeypatch.setattr(C, "M3_VARIANT", variant)
        L = I.fit_ladder(p, end, False)
        T = I.fit_tone(p, end, False, L, variants=("M3tone",))
        assert I.lr_test(L["M2"], T["M3tone"])["df"] == df


def test_wald_equal_zero_for_identical_coefficients():
    params = pd.Series({"p[0->0].tvtp1": 0.7, "p[0->0].tvtp2": 0.7, "p[1->0].tvtp1": 0.0, "p[1->0].tvtp2": 0.0})
    fit = NS(model=NS(), params=params, fixed={"p[1->0].tvtp1": 0.0, "p[1->0].tvtp2": 0.0})
    names = list(params.index[:2])
    cov = pd.DataFrame(np.array([[0.04, 0.01], [0.01, 0.09]]), index=names, columns=names)
    w = I.wald_equal(fit, "a", "b", _cov=cov, _beta={"a": "p[0->0].tvtp1", "b": "p[0->0].tvtp2"})
    assert w["stat"] == pytest.approx(0.0)
    cov_nan = cov * np.nan
    wn = I.wald_equal(fit, "a", "b", _cov=cov_nan, _beta={"a": "p[0->0].tvtp1", "b": "p[0->0].tvtp2"})
    assert np.isnan(wn["stat"]) and np.isnan(wn["pvalue"]) and np.isnan(wn["se"])


def test_coef_table_deck_signs_and_fixed_rows(fits):
    p, L = fits
    tab = I.coef_table(L["M3"], "M3")
    row = tab[(tab["var"] == "iran_gdelt") & (tab["logit"] == "N->H")].iloc[0]
    assert row["coef"] == pytest.approx(-L["M3"].params["p[0->0].tvtp4"]) and row["se"] > 0
    fx = tab[(tab["var"] == "iran_gdelt") & (tab["logit"] == "H->N")].iloc[0]
    assert fx["fixed"] and fx["coef"] == 0.0 and np.isnan(fx["se"])
    assert set(["model", "logit", "var", "coef", "se", "z", "p", "fixed"]) <= set(tab.columns)


def test_fit_summary_aic_bic(fits):
    p, L = fits
    s = I.fit_summary(L["M2"], nobs=299)
    assert s["aic"] == pytest.approx(2 * s["k"] - 2 * s["loglike"])
    assert s["bic"] == pytest.approx(s["k"] * np.log(299) - 2 * s["loglike"])
    assert s["sigma2_N"] < s["sigma2_H"] and s["dur_N"] > 0


def test_bootstrap_lr_small_and_deterministic(sim_panel):
    p = sim_panel(n=160, beta=0.0, seed=3)
    s = ms.make_specs(True)["M1"]
    d = D.build_design(p, s, p["date"].iloc[-1])
    a = I.bootstrap_lr_1v2(d.y, d.X, n_reps=4, seed=C.SEED, search_reps=2)
    b = I.bootstrap_lr_1v2(d.y, d.X, n_reps=4, seed=C.SEED, search_reps=2)
    assert a["lr"] >= 0 and 1 / 5 <= a["pvalue"] <= 1.0 and len(a["lr_boot"]) == 4
    assert a["pvalue"] == b["pvalue"] and np.allclose(a["lr_boot"], b["lr_boot"])
    assert 0 <= a["n_degenerate"] <= 4 and a["n_reps"] == 4


def test_equality_tests_shape(sim_panel):
    p = sim_panel(n=300, beta=0.0, seed=5)
    s = ms.make_specs(True)["M1"]
    d = D.build_design(p, s, p["date"].iloc[-1])
    t = I.equality_tests(d, s)
    assert list(t.index) == ["mu", "phi"] and (t["df"] == 1).all() and (t["p_lr"].between(0, 1)).all()
    assert list(t.columns) == ["est_N", "est_H", "lr", "df", "p_lr", "wald", "p_wald", "lr_raw"]


def test_mean_exog_is_one_common_coefficient(fits):
    p, L = fits
    f3, s3 = L["M3"], L["M3"].spec
    end = p["date"].iloc[-1]
    sm = replace(s3, mean_exog=("iran_gdelt",), name="M3_mean")
    dm = D.build_design(p, sm, end)
    mm = ms.make_model(dm, sm)
    assert len(mm.param_names) == len(f3.model.param_names) + 1
    assert [n for n in mm.param_names if n.startswith("x2")] == ["x2[1]"]
    assert mm.readable["x2[1]"] == "theta_iran_gdelt"
    fm = ms.fit_ms(dm, sm, None, 5, 0, seed=405, extra_starts=(ms.nest_start(f3, mm, sm),))
    assert fm.params["p[1->0].tvtp4"] == 0.0 and fm.loglike >= f3.loglike - 1e-6
    assert I.lr_test(f3, fm)["df"] == 1


def test_bootstrap_failed_replicate_excluded(sim_panel, monkeypatch):
    p = sim_panel(n=160, beta=0.0, seed=3)
    s = ms.make_specs(True)["M1"]
    d = D.build_design(p, s, p["date"].iloc[-1])
    real, calls = ms.fit_model, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:  # first replicate (call 1 is the observed fit)
            raise RuntimeError("all fit candidates failed")
        return real(*a, **k)

    monkeypatch.setattr(ms, "fit_model", flaky)
    r = I.bootstrap_lr_1v2(d.y, d.X, n_reps=3, seed=C.SEED, search_reps=2)
    assert r["n_failed"] == 1 and r["n_valid"] == 2 and np.isnan(r["lr_boot"][0])
    assert r["pvalue"] >= 1 / 3 and "degenerate_obs" in r


def test_garchx_lr_chibar_and_nesting(sim_panel):
    p = sim_panel(n=1140)
    a = I.garchx_lr(p, "2018-12-28", None, ("gpr", "vix"))              # GARCHX_C vs GARCH-t, k = 3
    assert a["k"] == 3 and a["ll_big"] >= a["ll_small"] - 1e-6 and 0 <= a["pvalue_chibar"] <= 1
    b = I.garchx_lr(p, "2018-12-28", ("gpr", "vix"), ("gpr", "vix", "iran_gdelt"))
    assert b["k"] == 1 and b["ll_big"] >= b["ll_small"] - 1e-6
