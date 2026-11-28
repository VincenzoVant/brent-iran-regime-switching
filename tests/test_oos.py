import json, numpy as np, pandas as pd, pytest
from dataclasses import replace
from brent import oos, ms, design as D, config as C, paths

FITLOG_COLS = ["origin", "model", "settings_hash", "loglike", "converged", "n_restarts_used", "relabeled",
               "refit_from_scratch", "degenerate", "nesting_ok", "seconds", "sigma2_N", "sigma2_H", "dur_N",
               "dur_H", "xi_H_origin", "har_floored", "restart_reason", "ll_gain_from_restart", "n_warnings",
               "at_cap"]


@pytest.fixture
def env(tmp_path, monkeypatch, sim_panel):
    for name in ("CHECKPOINTS", "FORECASTS", "FITS", "FROZEN", "PROCESSED"):
        d = tmp_path / name.lower(); d.mkdir(); monkeypatch.setattr(paths, name, d)
    monkeypatch.setattr(C, "PHI_SWITCHING", False)
    monkeypatch.setattr(C, "SPEC_FROZEN", False)       # tests start unfrozen, whatever config.py says now
    monkeypatch.setattr(C, "SEARCH_REPS_FIRST", 2); monkeypatch.setattr(C, "N_RESTARTS", 1)
    p = sim_panel(n=1140, beta=1.0); p.to_csv(paths.PROCESSED / "weekly_panel.csv", index=False)
    specs = ms.make_specs(False)
    return p, [specs["M1"], specs["M2"]], ("HAR", "HARX_CI", "GARCH_t", "GARCHX_C", "RW"), tmp_path


ORIG = oos.origins("2013-01-04", "2013-01-18")                # 3 origins: 2012-12-28, 2013-01-04, 2013-01-11


def test_origins_and_seed_index():
    assert ORIG[0] == pd.Timestamp("2012-12-28") and ORIG[-1] == pd.Timestamp("2013-01-11")
    assert oos.origin_index(pd.Timestamp("2012-12-28")) == 0 and oos.origin_index(ORIG[2]) == 2
    assert len(oos.origins(C.VALID[0], C.FREEZE_DATE)) == 723


def test_settings_hash_changes_with_settings(env, monkeypatch):
    p, specs, bm, _ = env
    h = oos.settings_hash(specs, bm, C.IRAN_VAR)
    assert h == oos.settings_hash(specs, bm, C.IRAN_VAR) and len(h) == 16
    monkeypatch.setattr(C, "N_RESTARTS", 2)
    assert h != oos.settings_hash(specs, bm, C.IRAN_VAR)
    monkeypatch.setattr(C, "N_RESTARTS", 1)
    assert h != oos.settings_hash(specs, bm, "aigpr_iran")
    assert h != oos.settings_hash(specs, bm[:-1], C.IRAN_VAR)
    monkeypatch.setattr(C, "MAX_TVTP_COEF", 60)                         # decision A is part of the hash
    assert h != oos.settings_hash(specs, bm, C.IRAN_VAR)
    monkeypatch.setattr(C, "MAX_TVTP_COEF", 50)
    for name, val in (("CONTROLS_ONLY_NH", False), ("INTRADAY_REDISTRIBUTE", True),
                      ("STANDARDIZE_WINDOW_HOURLY", ("2013-06-07", "2016-06-03")), ("TRAIN", ("2005-01-07", "2013-12-27")),
                      ("SEED", 1), ("HIGHVOL_PCTL", 95), ("RV_FLOOR_FRAC", 0.02), ("VALID", ("2013-01-04", "2017-12-29")),
                      ("TEST", ("2019-01-04", "2024-12-27")), ("SAMPLE_START", "2006-01-06"),
                      ("STANDARDIZE_WINDOW", ("2005-01-07", "2011-12-30")), ("FREEZE_DATE", "2026-10-30"),
                      ("HAR_MIN_ROWS", 26)):
        old = getattr(C, name); monkeypatch.setattr(C, name, val)
        assert h != oos.settings_hash(specs, bm, C.IRAN_VAR), name
        monkeypatch.setattr(C, name, old)
    assert h == oos.settings_hash(specs, bm, C.IRAN_VAR)


def test_sanity_flags():
    ok = lambda **k: type("F", (), dict(converged=True, degenerate=False, loglike=-100.0,
                                        params=pd.Series({"sigma2[0]": 1.0, "sigma2[1]": 4.0}), **k))()
    assert oos.sanity_flags(ok(), -100.5, -101.0, 1.5) == []
    assert "ll_below_start" in oos.sanity_flags(ok(), -99.0, None, 1.5)
    assert "nesting" in oos.sanity_flags(ok(), None, -90.0, 1.5)
    assert "sigma2_ratio" in oos.sanity_flags(ok(), None, None, 5.0)
    bad = ok(); bad.converged = False; bad.degenerate = True
    assert {"not_converged", "degenerate"} <= set(oos.sanity_flags(bad, None, None, 1.5))


def test_first_origin_rules(sim_panel, monkeypatch):                     # decision B
    for m in ("M1", "M3", "HAR", "GARCH_t", "RW"):
        assert oos.first_origin(m) is None
    fo = oos.first_origin("HAR_h")
    assert fo == pd.Timestamp("2014-08-29") == oos.first_origin("HARX_C") == oos.first_origin("HARX_CI")
    p = sim_panel(n=1140)
    hd = lambda end: D.har_design(p, end, rv_col=C.X_RV_COL, start=C.X_SAMPLE_START)[0]
    assert len(hd(fo)) == C.HAR_MIN_ROWS == 52 and len(hd(fo - pd.Timedelta(days=7))) == 51   # first origin with 52 rows
    monkeypatch.setattr(C, "HAR_MIN_ROWS", 1)                                   # read at call time
    assert oos.first_origin("HAR_h") == pd.Timestamp("2013-09-06") and len(hd(pd.Timestamp("2013-09-06"))) == 1
    monkeypatch.setattr(C, "HAR_MIN_ROWS", 52)
    assert oos.first_origin("GARCHX_C") == pd.Timestamp(C.GARCHX_SCALE_WINDOW[1]) == oos.first_origin("GARCHX_CI")
    monkeypatch.setattr(C, "GARCHX_SCALE_WINDOW", ("2013-06-07", "2016-01-01"))   # read at call time
    assert oos.first_origin("GARCHX_C") == pd.Timestamp("2016-01-01")


def test_one_var_row_per_model_per_origin_and_schema(env):
    p, specs, bm, _ = env
    oos.run_oos(p, "t1", specs, benchmarks=bm, origins=ORIG)
    f, par, log = oos.assemble("t1")
    assert list(f.columns) == ["date", "origin", "split", "run", "model", "target", "forecast"]
    assert list(log.columns) == FITLOG_COLS and list(par.columns[:2]) == ["origin", "model"]
    assert (f["date"] - f["origin"] == pd.Timedelta(days=7)).all() and (f["split"] == "validation").all()
    v = f[f["target"] == "var"]
    for m in ("M1", "M2", "HAR", "GARCH_t"):
        assert (v[v["model"] == m].groupby("origin").size() == 1).all() and len(v[v["model"] == m]) == 3
    assert f[f["model"].isin(["HARX_CI", "GARCHX_C"])].empty            # decision B: no hourly history yet
    sk = log[log["model"].isin(["HARX_CI", "GARCHX_C"])]
    assert len(sk) == 6 and sk["restart_reason"].str.startswith("skipped").all()
    assert (sk["converged"] == False).all() and sk["loglike"].isna().all()   # skipped rows: converged=False
    assert len(f[(f["model"] == "RW") & (f["target"] == "r")]) == 3 and (f["forecast"] > 0)[f["target"] == "var"].all()
    assert f.loc[f["target"] == "p_high", "forecast"].between(0, 1).all()
    assert set(log["settings_hash"]) == {oos.settings_hash(specs, bm, C.IRAN_VAR)} and len(log) == 3 * 7
    assert {"sigma2_N", "sigma2_H", "phi"} <= set(par.columns) and len(par[par["model"] == "M1"]) == 3
    assert log["at_cap"].notna().all() and (log.loc[~log["model"].isin(["M1", "M2"]), "at_cap"] == 0).all()
    m2 = log[log["model"] == "M2"].reset_index(drop=True); pm2 = par[par["model"] == "M2"].reset_index(drop=True)
    slopes = [c for c in pm2.columns if c.startswith("p00_") and c != "p00_const"]
    assert list(m2["at_cap"]) == [int((pm2.loc[i, slopes].abs() >= C.MAX_TVTP_COEF - 1e-6).sum()) for i in range(3)]


def test_hourly_benchmarks_start_when_data_allows(env):                  # decision B
    p, specs, _, _ = env
    bm = ("HAR_h", "HARX_C", "HARX_CI", "GARCHX_C", "GARCHX_CI")
    orig = [pd.Timestamp(x) for x in ("2014-08-22", "2014-08-29", "2015-05-29", "2015-06-05")]
    oos.run_oos(p, "hb", [], benchmarks=bm, origins=orig)
    f, par, log = oos.assemble("hb")
    first = lambda m: f.loc[f["model"] == m, "origin"].min()
    assert all(first(m) == pd.Timestamp("2014-08-29") for m in bm[:3])
    assert all(first(m) == pd.Timestamp("2015-06-05") and (f["model"] == m).sum() == 1 for m in bm[3:])
    assert np.isfinite(f["forecast"]).all() and (f["forecast"] > 0).all() and set(f["target"]) == {"var"}
    assert len(log) == 4 * 5 and (log["restart_reason"].fillna("").str.startswith("skipped")).sum() == 3 + 2 * 3


def test_garchx_first_origin_has_no_lookahead(env):                      # decision B: scale window <= origin
    p, _, _, _ = env
    T = pd.Timestamp(C.GARCHX_SCALE_WINDOW[1])
    q = p.copy(); num = q.columns.drop("date")
    q.loc[q["date"] > T, num] = 1e6
    bm = ("HARX_CI", "GARCHX_C", "GARCHX_CI")
    oos.run_oos(p, "la1", [], benchmarks=bm, origins=[T])
    oos.run_oos(q, "la2", [], benchmarks=bm, origins=[T])
    a, b = oos.assemble("la1")[0], oos.assemble("la2")[0]
    assert len(a) == 3
    pd.testing.assert_frame_equal(a.drop(columns="run"), b.drop(columns="run"), check_exact=True)


def test_interrupted_run_equals_uninterrupted(env):
    p, specs, bm, _ = env
    oos.run_oos(p, "a", specs, benchmarks=bm, origins=ORIG)
    oos.run_oos(p, "b", specs, benchmarks=bm, origins=ORIG, max_origins=1)
    first = (paths.CHECKPOINTS / "b_fitlog.csv").read_text()
    oos.run_oos(p, "b", specs, benchmarks=bm, origins=ORIG)               # resumes after origin 1
    assert (paths.CHECKPOINTS / "b_fitlog.csv").read_text().startswith(first)   # origin 1 was not refitted
    for x, y in zip(oos.assemble("a")[:2], oos.assemble("b")[:2]):
        pd.testing.assert_frame_equal(x.drop(columns=["run", "seconds"], errors="ignore"),
                                      y.drop(columns=["run", "seconds"], errors="ignore"), check_exact=True)


def test_crash_mid_origin_does_not_duplicate_rows(env):
    p, specs, bm, tmp = env
    oos.run_oos(p, "c", specs, benchmarks=bm, origins=ORIG, max_origins=2)
    fl = paths.CHECKPOINTS / "c_fitlog.csv"; log = pd.read_csv(fl)
    log[log["origin"] != "2013-01-04"].to_csv(fl, index=False)        # origin 2 has forecasts/params but no fitlog rows
    oos.run_oos(p, "c", specs, benchmarks=bm, origins=ORIG)
    f, _, _ = oos.assemble("c")
    assert not f.duplicated(["date", "model", "target"]).any() and f["origin"].nunique() == 3


def test_changed_settings_discard_checkpoint(env, monkeypatch):
    p, specs, bm, _ = env
    oos.run_oos(p, "d", specs, benchmarks=bm, origins=ORIG, max_origins=2)
    monkeypatch.setattr(C, "N_RESTARTS", 2)
    oos.run_oos(p, "d", specs, benchmarks=bm, origins=ORIG, max_origins=1)
    log = pd.read_csv(paths.CHECKPOINTS / "d_fitlog.csv")
    assert log["origin"].nunique() == 1 and log["origin"].iloc[0] == "2012-12-28"


def test_future_rows_never_enter_a_forecast(env):                         # Review Focus 4
    p, specs, bm, _ = env
    q = p.copy(); num = q.columns.drop("date")
    q.loc[q["date"] > ORIG[0], num] = 1e6
    oos.run_oos(p, "e1", specs, benchmarks=bm, origins=ORIG[:1])
    oos.run_oos(q, "e2", specs, benchmarks=bm, origins=ORIG[:1])
    a, b = oos.assemble("e1")[0], oos.assemble("e2")[0]
    pd.testing.assert_frame_equal(a.drop(columns="run"), b.drop(columns="run"), check_exact=True)


def test_nonconverged_fit_is_logged_and_run_continues(env, monkeypatch):  # Review Focus 5
    p, specs, bm, _ = env; real = ms.fit_ms
    def bad(design, spec, *a, **k):
        f = real(design, spec, *a, **k)
        if spec.name == "M2": f.converged = False
        return f
    monkeypatch.setattr(ms, "fit_ms", bad)
    oos.run_oos(p, "f", specs, benchmarks=bm, origins=ORIG)
    f, _, log = oos.assemble("f")
    m2 = log[log["model"] == "M2"]
    assert (~m2["converged"].astype(bool)).all() and len(f[(f["model"] == "M2") & (f["target"] == "var")]) == 3
    assert m2["restart_reason"].str.contains("not_converged").all()


def test_missing_origin_regressor_raises(env):                             # Review Focus 1
    p, specs, bm, _ = env; q = p.copy()
    q.loc[q["date"] == ORIG[1], "vix"] = np.nan
    with pytest.raises(ValueError, match="vix"):
        oos.run_oos(q, "g", specs, benchmarks=bm, origins=ORIG)


def test_spec_frozen_guard_and_hash_check(env, monkeypatch):
    p, specs, bm, _ = env
    late = oos.origins("2019-01-04", "2019-01-04")
    with pytest.raises(RuntimeError, match="SPEC_FROZEN"):
        oos.run_oos(p, "h", specs, benchmarks=bm, origins=late)
    monkeypatch.setattr(C, "SPEC_FROZEN", True)
    with pytest.raises(RuntimeError, match="spec_freeze"):
        oos.run_oos(p, "main", specs, benchmarks=bm, origins=late)           # no freeze file
    (paths.FROZEN / "spec_freeze.json").write_text(json.dumps({"settings_hash": "deadbeef"}))
    with pytest.raises(RuntimeError, match="settings_hash"):
        oos.run_oos(p, "main", specs, benchmarks=bm, origins=late)


def test_phi_switching_unset_raises(env, monkeypatch):
    p, specs, bm, _ = env
    monkeypatch.setattr(C, "PHI_SWITCHING", None)
    with pytest.raises(ValueError, match="PHI_SWITCHING"):
        oos.run_oos(p, "phi", specs, benchmarks=bm, origins=ORIG[:1])


def test_output_directories_are_created(env, tmp_path, monkeypatch):
    p, specs, bm, _ = env
    for name in ("CHECKPOINTS", "FORECASTS", "FITS"):
        fresh = tmp_path / ("fresh_" + name.lower()); monkeypatch.setattr(paths, name, fresh)
    oos.run_oos(p, "dirs", specs[:1], benchmarks=("RW",), origins=ORIG[:1]); oos.assemble("dirs")
    assert (paths.CHECKPOINTS / "dirs_fitlog.csv").exists() and (paths.FORECASTS / "forecasts_dirs.csv").exists()
    assert (paths.FITS / "params_dirs.csv").exists() and (paths.FITS / "fitlog_dirs.csv").exists()


def test_fake_forecasts_schema_and_load_with_actuals(env):
    p, specs, bm, _ = env
    fk = oos.fake_forecasts(p)
    assert list(fk.columns) == ["date", "origin", "split", "run", "model", "target", "forecast"]
    assert fk["date"].nunique() == 723 and set(fk["split"]) == {"validation", "test", "holdout"} and (fk["run"] == "fake").all()
    assert set(fk.loc[fk["model"] == "RW", "target"]) == {"r"} and set(fk.loc[fk["model"] == "GARCHX_CI", "target"]) == {"var"}
    assert set(fk.loc[fk["model"] == "M3", "target"]) == {"var", "r", "p_high", "p_exceed"}
    assert fk.loc[fk["model"] == "GARCHX_CI", "origin"].min() == oos.first_origin("GARCHX_CI")   # mirrors decision B
    assert np.isfinite(fk["forecast"]).all()
    fk.to_csv(paths.FORECASTS / "forecasts_fake.csv", index=False)
    lf = oos.load_forecasts("fake")
    row = lf.iloc[0]; P = p.set_index("date")
    assert row["actual_rv_hourly"] == P.loc[row["date"], "rv_hourly"] or np.isnan(row["actual_rv_hourly"])
    assert row["actual_r"] == P.loc[row["date"], "r"] and row["actual_rv"] == P.loc[row["date"], "rv"]
    assert set(lf["actual_exceed"].dropna().unique()) <= {0, 1}
    assert len(oos.load_forecasts("fake", with_actuals=False).columns) == 7


def _always_degenerate(monkeypatch, names=("M1",)):
    real, calls = ms.fit_ms, []
    def deg(design, spec, *a, **k):
        f = real(design, spec, *a, **k)
        calls.append((spec.name, a[2] if len(a) > 2 else k.get("n_restarts")))
        if spec.name in names: f.degenerate = True
        return f
    monkeypatch.setattr(ms, "fit_ms", deg)
    return calls


def test_persistent_degenerate_flag_does_not_restart_every_week(env, monkeypatch):
    p, specs, _, _ = env
    calls = _always_degenerate(monkeypatch)
    oos.run_oos(p, "dg", specs[:1], benchmarks=(), origins=ORIG)
    log = oos.assemble("dg")[2]
    assert list(log["restart_reason"].fillna("")) == ["scheduled+degenerate", "", ""]
    assert list(log["n_restarts_used"]) == [C.N_RESTARTS, 0, 0] and log["degenerate"].all()
    assert [n for _, n in calls] == [0, C.N_RESTARTS, 0, 0]               # one restart, at the scheduled origin


def test_new_degenerate_flag_still_restarts(env, monkeypatch):
    p, specs, _, _ = env
    real = ms.fit_ms
    def deg_late(design, spec, *a, **k):                                   # degenerate only from origin 2 on
        f = real(design, spec, *a, **k)
        if design.origin >= ORIG[1]: f.degenerate = True
        return f
    monkeypatch.setattr(ms, "fit_ms", deg_late)
    oos.run_oos(p, "dg2", specs[:1], benchmarks=(), origins=ORIG)
    log = oos.assemble("dg2")[2]
    assert list(log["restart_reason"].fillna("")) == ["scheduled", "degenerate", ""]


def test_resume_reproduces_degenerate_restart_rule(env, monkeypatch):
    p, specs, _, _ = env
    _always_degenerate(monkeypatch)
    oos.run_oos(p, "ra", specs[:1], benchmarks=(), origins=ORIG)
    oos.run_oos(p, "rb", specs[:1], benchmarks=(), origins=ORIG, max_origins=1)
    oos.run_oos(p, "rb", specs[:1], benchmarks=(), origins=ORIG)
    a, b = oos.assemble("ra")[2], oos.assemble("rb")[2]
    assert list(a["restart_reason"].fillna("")) == list(b["restart_reason"].fillna(""))
    pd.testing.assert_frame_equal(oos.assemble("ra")[1], oos.assemble("rb")[1], check_exact=True)


def test_force_deletes_a_valid_checkpoint(env):
    p, specs, _, _ = env
    oos.run_oos(p, "fz", specs[:1], benchmarks=("RW",), origins=ORIG, max_origins=2)
    oos.run_oos(p, "fz", specs[:1], benchmarks=("RW",), origins=ORIG, max_origins=1, force=True)
    log = pd.read_csv(paths.CHECKPOINTS / "fz_fitlog.csv")
    assert list(log["origin"].unique()) == ["2012-12-28"]


def test_assemble_keeps_only_complete_origins(env):
    p, specs, _, _ = env
    oos.run_oos(p, "ac", specs[:1], benchmarks=("HAR", "RW"), origins=ORIG, max_origins=2)
    fl = paths.CHECKPOINTS / "ac_fitlog.csv"; txt = fl.read_text().splitlines()
    fl.write_text("\n".join(txt[:-1]) + "\n")                             # origin 2 lost its last fitlog row
    f, par, log = oos.assemble("ac")
    assert set(f["origin"]) == set(par["origin"]) == set(log["origin"]) == {pd.Timestamp("2012-12-28")}
    oos.run_oos(p, "ac", specs[:1], benchmarks=("HAR", "RW"), origins=ORIG)   # resume redoes origin 2
    f, _, log = oos.assemble("ac")
    assert f["origin"].nunique() == 3 and len(log) == 9 and not f.duplicated(["date", "model", "target"]).any()
    assert not list(paths.CHECKPOINTS.glob("*.tmp"))                       # atomic rewrite leaves no temp file


def test_settings_hash_is_read_as_text(env, monkeypatch):
    p, specs, _, _ = env
    monkeypatch.setattr(oos, "settings_hash", lambda *a: "00000000000001e5")  # would parse as a float
    oos.run_oos(p, "tx", specs[:1], benchmarks=("RW",), origins=ORIG, max_origins=1)
    first = (paths.CHECKPOINTS / "tx_fitlog.csv").read_text()
    oos.run_oos(p, "tx", specs[:1], benchmarks=("RW",), origins=ORIG)
    assert (paths.CHECKPOINTS / "tx_fitlog.csv").read_text().startswith(first)   # resumed, not discarded
    assert set(oos.assemble("tx")[2]["settings_hash"]) == {"00000000000001e5"}


def test_ms_fit_failure_carries_last_week_forward(env, monkeypatch):
    p, specs, _, _ = env
    real = ms.fit_ms
    def fail_at_2(design, spec, *a, **k):                                  # every M1 fit fails at origin 2
        if design.origin == ORIG[1]:
            raise RuntimeError("all fit candidates failed")
        return real(design, spec, *a, **k)
    monkeypatch.setattr(ms, "fit_ms", fail_at_2)
    oos.run_oos(p, "ff", specs[:1], benchmarks=("RW",), origins=ORIG)
    f, par, log = oos.assemble("ff")
    m1 = log[log["model"] == "M1"].reset_index(drop=True)
    assert m1.loc[1, "restart_reason"] == "fit_failed" and not m1.loc[1, "converged"] and m1.loc[1, "degenerate"]
    pm = par[par["model"] == "M1"].drop(columns=["origin", "model"]).dropna(axis=1).reset_index(drop=True)
    pd.testing.assert_series_equal(pm.iloc[0], pm.iloc[1], check_names=False)   # params carried forward
    assert f[(f["model"] == "M1")]["origin"].nunique() == 3 and np.isfinite(f["forecast"]).all()


def test_benchmark_failure_reuses_previous_forecast_and_resume_matches(env, monkeypatch):
    p, specs, _, _ = env
    real = oos.bench.model_forecast
    def fail(name, panel, end, state):
        if name == "GARCH_t" and pd.Timestamp(end) in (ORIG[0], ORIG[2]):
            raise np.linalg.LinAlgError("singular")
        return real(name, panel, end, state)
    monkeypatch.setattr(oos.bench, "model_forecast", fail)
    oos.run_oos(p, "bf", specs[:1], benchmarks=("GARCH_t", "HAR"), origins=ORIG)
    oos.run_oos(p, "bg", specs[:1], benchmarks=("GARCH_t", "HAR"), origins=ORIG, max_origins=2)
    oos.run_oos(p, "bg", specs[:1], benchmarks=("GARCH_t", "HAR"), origins=ORIG)      # resume before origin 3
    f, par, log = oos.assemble("bf")
    g = log[log["model"] == "GARCH_t"].reset_index(drop=True)
    assert list(g["restart_reason"].fillna("")) == ["fit_failed (no forecast)", "", "fit_failed (previous forecast reused)"]
    gv = f[f["model"] == "GARCH_t"]
    assert list(gv["origin"]) == [ORIG[1], ORIG[2]] and gv["forecast"].iloc[0] == gv["forecast"].iloc[1]
    for x, y in zip(oos.assemble("bf")[:2], oos.assemble("bg")[:2]):
        pd.testing.assert_frame_equal(x.drop(columns="run", errors="ignore"), y.drop(columns="run", errors="ignore"),
                                      check_exact=True)


def test_nan_regressor_in_benchmark_still_stops_the_run(env):
    p, _, _, _ = env
    q = p.copy(); T = pd.Timestamp(C.GARCHX_SCALE_WINDOW[1])
    q.loc[q["date"] == T, "vix"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        oos.run_oos(q, "nb", [], benchmarks=("GARCHX_C",), origins=[T])


def test_torn_last_line_is_not_a_completed_origin(env):
    p, specs, _, _ = env
    oos.run_oos(p, "tl", specs[:1], benchmarks=("RW",), origins=ORIG)
    oos.run_oos(p, "tm", specs[:1], benchmarks=("RW",), origins=ORIG, max_origins=2)
    fl = paths.CHECKPOINTS / "tm_fitlog.csv"; txt = fl.read_text()
    fl.write_text(txt[:-10])                                               # crash mid-write of the last row
    oos.run_oos(p, "tm", specs[:1], benchmarks=("RW",), origins=ORIG)
    a, b = oos.assemble("tl"), oos.assemble("tm")
    assert len(b[2]) == 6 and b[0]["origin"].nunique() == 3
    for x, y in zip(a[:2], b[:2]):
        pd.testing.assert_frame_equal(x.drop(columns="run", errors="ignore"), y.drop(columns="run", errors="ignore"),
                                      check_exact=True)
