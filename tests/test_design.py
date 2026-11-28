import re, pathlib
import numpy as np, pandas as pd, pytest
from dataclasses import replace
from brent import design as D, ms, config as C

def spec(name="M3", phi=False, **kw):
    return replace(ms.make_specs(phi)[name], **kw)

def test_make_specs_and_with_iran():
    s = ms.make_specs(True)
    assert s["M1"].tvtp_vars == () and s["M1"].iran_vars == ()
    assert s["M2"].tvtp_vars == ("gpr", "vix", "log_rv") and s["M2"].iran_vars == ()
    assert s["M3"].tvtp_vars == ("gpr", "vix", "log_rv", "iran_gdelt") and s["M3"].iran_only_NH
    assert not ms.make_specs(True, variant="iran_both_logits")["M3"].iran_only_NH
    t = ms.with_iran(s["M3"], C.IRAN_TONE_VARS, "M3tone")
    assert t.tvtp_vars == ("gpr", "vix", "log_rv", "iran_conflict", "iran_coop")
    assert t.iran_vars == C.IRAN_TONE_VARS and t.name == "M3tone" and t.phi_switching

def test_rows_hold_lagged_values_and_x_next_uses_origin(sim_panel):
    p = sim_panel(n=1140); P = p.set_index("date")
    d = D.build_design(p, spec(), end="2018-12-28")
    t = d.dates[10]; prev = t - pd.Timedelta(days=7)
    tr = P.loc[C.TRAIN[0]:C.TRAIN[1], "gpr"]
    assert d.dates[0] == pd.Timestamp("2005-01-14") and d.origin == pd.Timestamp("2018-12-28")
    assert d.X.loc[t, "r_lag"] == P.loc[prev, "r"] and d.y[t] == P.loc[t, "r"]
    assert d.Z.loc[t, "const"] == 1.0
    assert d.Z.loc[t, "gpr"] == pytest.approx((P.loc[prev, "gpr"] - tr.mean()) / tr.std())
    T = d.origin
    assert d.x_next["r_lag"] == P.loc[T, "r"]
    assert d.z_next["gpr"] == pytest.approx((P.loc[T, "gpr"] - tr.mean()) / tr.std())

def test_standardization_uses_train_window_only(sim_panel):
    p = sim_panel(n=1140); q = p.copy()
    q.loc[q["date"] > C.TRAIN[1], "gpr"] += 5.0
    a = D.standardization_stats(p, ["gpr"], C.TRAIN); b = D.standardization_stats(q, ["gpr"], C.TRAIN)
    pd.testing.assert_frame_equal(a, b)

def test_rows_after_origin_do_not_matter_and_nan_inside_window_raises(sim_panel):
    p = sim_panel(n=1140); end = "2018-12-28"
    q = p.copy(); q.loc[q["date"] > end, ["r", "gpr", "vix", "iran_gdelt", "log_rv"]] = np.nan
    a, b = D.build_design(p, spec(), end), D.build_design(q, spec(), end)
    pd.testing.assert_frame_equal(a.Z, b.Z); pd.testing.assert_series_equal(a.x_next, b.x_next)
    bad = p.copy(); bad.loc[300, "vix"] = np.nan
    with pytest.raises(ValueError, match="vix") as e:
        D.build_design(bad, spec(), end)
    assert str(p.loc[300, "date"].date()) in str(e.value)

def test_standardization_stats_raises_on_nan_in_window(sim_panel):
    p = sim_panel(n=1140); p.loc[10, "gpr"] = np.nan
    with pytest.raises(ValueError, match="gpr"):
        D.standardization_stats(p, ["gpr"], C.TRAIN)

def test_end_must_be_a_panel_date(sim_panel):
    p = sim_panel(n=1140)
    with pytest.raises(ValueError, match="2018-12-29"):
        D.build_design(p, spec(), "2018-12-29")
    with pytest.raises(ValueError, match="2018-12-29"):
        D.har_design(p, "2018-12-29")
    with pytest.raises(ValueError, match="2018-12-29"):
        D.garchx_design(p, "2018-12-29", ("gpr",))

def test_mean_exog_enters_x(sim_panel):
    p = sim_panel(n=1140)
    d = D.build_design(p, spec(mean_exog=("iran_gdelt",)), "2018-12-28")
    assert list(d.X.columns) == ["r_lag", "iran_gdelt"] and "iran_gdelt" in d.x_next.index

def test_m1_has_no_z(sim_panel):
    d = D.build_design(sim_panel(n=1140), spec("M1"), "2018-12-28")
    assert d.Z is None and d.z_next is None

def test_mshar_not_built_here(sim_panel):
    with pytest.raises(NotImplementedError):
        D.build_design(sim_panel(n=1140), spec(kind="mshar"), "2018-12-28")

def test_har_design_rows_and_x_next(sim_panel):
    p = sim_panel(n=1140); P = p.set_index("date"); rv = P["rv"]
    X, y, x_next = D.har_design(p, "2018-12-28", rv_col="rv", x_vars=("gpr",))
    s = X.index[5]; i = rv.index.get_loc(s)
    assert X.index[0] == pd.Timestamp(C.SAMPLE_START) + pd.Timedelta(weeks=13) and y[s] == rv[s]
    assert X.loc[s, "rv_1"] == rv.iloc[i - 1]
    assert X.loc[s, "rv_4"] == pytest.approx(rv.iloc[i - 4:i].mean())
    assert X.loc[s, "rv_13"] == pytest.approx(rv.iloc[i - 13:i].mean())
    assert X.index[-1] == pd.Timestamp("2018-12-28")
    T = rv.index.get_loc(pd.Timestamp("2018-12-28"))
    assert x_next["rv_1"] == rv.iloc[T] and x_next["rv_4"] == pytest.approx(rv.iloc[T - 3:T + 1].mean())
    q = p.copy(); q.loc[q["date"] > "2018-12-28", ["rv", "gpr"]] = 1e6
    X2, y2, xn2 = D.har_design(q, "2018-12-28", rv_col="rv", x_vars=("gpr",))
    pd.testing.assert_frame_equal(X, X2); pd.testing.assert_series_equal(xn2, x_next)

def test_har_design_hourly_starts_after_hourly_sample(sim_panel):
    p = sim_panel(n=1140)
    X, y, _ = D.har_design(p, "2018-12-28", rv_col=C.X_RV_COL, start=C.X_SAMPLE_START)
    assert X.index[0] == pd.Timestamp(C.X_SAMPLE_START) + pd.Timedelta(weeks=13) and X.notna().all().all()

def test_garchx_design(sim_panel):
    p = sim_panel(n=1140); P = p.set_index("date")
    r, X, x_next = D.garchx_design(p, "2018-12-28", ("gpr", "vix", "iran_gdelt"))
    assert (X.values >= 0).all() and list(X.columns) == [C.X_RV_COL, "gpr", "vix", "iran_gdelt"]
    t = X.index[7]; prev = t - pd.Timedelta(days=7)
    assert X.loc[t, C.X_RV_COL] == P.loc[prev, C.X_RV_COL] and r.index[0] == X.index[0]
    assert r.index[0] == pd.Timestamp(C.X_SAMPLE_START) + pd.Timedelta(weeks=1)
    in_win = [(d - pd.Timedelta(days=7)) >= pd.Timestamp(C.GARCHX_SCALE_WINDOW[0]) and
              (d - pd.Timedelta(days=7)) <= pd.Timestamp(C.GARCHX_SCALE_WINDOW[1]) for d in X.index]
    assert X.loc[in_win, "gpr"].mean() == pytest.approx(1.0)
    assert x_next["gpr"] > 0

def test_freeze_std_stats_written(sim_panel, tmp_path, monkeypatch):
    monkeypatch.setattr(D.paths, "FROZEN", tmp_path)
    out = D.freeze_std_stats(sim_panel(n=1140))
    df = pd.read_csv(out)
    assert {"var", "window", "mean", "sd"} <= set(df.columns) and {"TRAIN", "HOURLY"} <= set(df["window"])

def test_no_shift_outside_design_and_weekly():
    bad = [f.name for f in pathlib.Path(D.__file__).parent.glob("*.py")
           if f.name not in ("design.py", "weekly.py") and re.search(r"\.shift\(", f.read_text())]
    assert bad == []
