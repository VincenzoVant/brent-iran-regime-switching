# Plan 2 — Models and Out-of-Sample Forecasts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From the frozen weekly panel to the frozen model specification and the 723-week expanding-window out-of-sample forecast file (`results/forecasts/forecasts_main.csv`), plus the in-sample RQ1 and RQ2 results (notebooks 03, 04, 05).

**Architecture:** Six modules under `src/brent/`, each imported by the notebooks and each with its own tests on synthetic data.
- `design.py` is the only place lags and standardization happen.
- `ms.py` fits and relabels the Markov-switching models (M1, M2, M3, M3-tone), including the constrained M3.
- `forecast.py` is the single one-step forecaster.
- `insample.py` holds the RQ1/RQ2 tests.
- `benchmarks.py` holds GARCH(1,1)-t, the HAR family and the hand-written GARCH-X.
- `oos.py` runs the expanding-window loop with checkpoints.

Notebook 03 freezes φ. Notebook 05 times the loop, validates, freezes the specification and then runs the full loop overnight. Notebook 04 runs while the full loop runs.

**Tech Stack:** Python 3.12, statsmodels ≥ 0.15 (`MarkovRegression`, `approx_hess3`, OLS), `arch` (GARCH-t), scipy (SLSQP, BFGS, `chi2`), numpy, pandas, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-brent-iran-implementation-design.md` (approved 2026-11-20, updated 2026-11-23). Section references (§N) point to it. The research plan is `Plans/Brent-Iran Regime Switching - Feasibility and Plan.md` ("plan §N"). Plan 1 is `docs/superpowers/plans/2026-11-20-plan1-foundation-and-data.md`.

**Scope:** Spec §7.3, §7.4, §7.5, §8.1 (decision table), §8.4–8.7, §8.8, §8.9 (except the two placebo functions, see Interfaces), §9.2–9.6, §10 rows `test_design`, `test_ms`, `test_forecast`, `test_benchmarks`, `test_insample`, `test_oos`, and the timeline rows for Nov 25–29 (§11). **Out of scope (Plan 3):** `evaluate.py`, the rest of `plots.py`, notebooks 06–08, robustness runs, the placebo functions, extension E1.

**Dates (spec §11):** Wed Nov 25 → Sun Nov 29. Task dates are on each task. **Slip order for the tight Wed Nov 25 / Thu Nov 26 days:** first defer Task 4's `param_cov` (and its test), then `insample.equality_tests` (Task 6); both are needed only by notebooks 03/04, not by the OOS loop, and can be finished while the validation stage runs. Critical path: Tasks 1–7 (φ frozen Thu Nov 26) → 8–10 → Task 11 (`timing` Fri, `validation` Fri evening, freeze Sat) → Task 12 (`full` overnight Sat → Sun). Task 13 (notebook 04) fits into the time the full run is going.

## What exists when this plan starts (from Plan 1)

- `brent.config` with every spec §8.1 constant. Decided values are in: `GDELT_FEED_RULE`, `AIGPR_URL`, `INTRADAY_ROLL_RULE`, `INTRADAY_REDISTRIBUTE`, and verified `INTRADAY_START` / `HOURLY_SAMPLE_START`. Still unset: `PHI_SWITCHING = None`, `SPEC_FROZEN = False`, `M3_VARIANT = "constrained"`.
- `brent.paths` (`ROOT, RAW, PROCESSED, RESULTS, FORECASTS, FITS, TABLES, FIGURES, FROZEN, CHECKPOINTS`, `save_table`, `check_budget`, `git_hash`).
- `brent.sources.load_raw`; `brent.weekly` (`week_ending`, `fridays`, `build_weekly_panel`).
- `brent.plots` with `setup_style()` and `save_fig(fig, name)` only.
- `tests/conftest.py` with the fixture `synthetic_raw`.
- `data/processed/weekly_panel.csv`: 1,140 Fridays, the 22 columns of spec §9.1, `rv_hourly` finite from `HOURLY_SAMPLE_START`.

## Global Constraints

- Python 3.12; statsmodels `>=0.15`. Parameter names are positional and are indexed **by name** through `tvtp_name_map`, never by position (R10).
- Lags happen only in `design.py` (and `weekly.py`). No `.shift(` anywhere else in `src/brent/`; placebo shifts use `np.roll` (grep test, Task 1). All forecasts for week t+1 use data dated ≤ t (the origin).
- Regime labels are canonical after **every** fit: regime 0 = N, regime 1 = H (higher σ²; for `canonical_by="const"` the higher intercept). Warm starts live in canonical labels.
- **Filtered, never smoothed** probabilities in forecasts and event plots (grep test, Task 5).
- Every random draw is seeded. Every `.fit(...)` with `search_reps > 0` passes `rng=seed` explicitly (statsmodels ignores `np.random.seed`). All other draws use `np.random.default_rng(seed)`. Per-origin seed = `C.SEED + origin_index`, with index 0 = origin 2012-12-28.
- TVTP regressors are z-scored with the fixed 2005–2012 window (`spec.std_window`) in every fit. Nothing computed on 2019+ chooses anything; 2026 chooses nothing (plan §5, §7).
- Degeneracy rule: σ²_min < `VAR_FLOOR_FRAC` × var(y), or either regime's expected duration at standardized regressors = 0 below `MIN_DURATION` weeks. Applies to every fit, bootstrap replicate and OOS fit; failures are logged, never hidden.
- The M3 variant (`C.M3_VARIANT`) is frozen for a whole run and never switches inside it.
- Frozen decisions are written into `config.py` **by hand**, with the matching `results/frozen/` file, and committed together. Later notebooks assert they match.
- Modules reference `paths.X` and `C.X` at call time (not copied at import), so tests can monkeypatch them.
- Checkpoints go to `results/checkpoints/` (git-ignored). Size budget as in Plan 1 (`paths.check_budget()` in every notebook's last cell). `forecasts_main.csv` ≈ 1.1 MB.
- Every notebook: first cell `%load_ext autoreload`, `%autoreload 2`, `import brent`, `from brent import config as C`, `np.random.seed(C.SEED)` and a `FORCE_*` flag block; last cells are a checks cell and `paths.check_budget()`. Committed with outputs after restart-and-run-all; inline figures at 100 dpi, saved PNGs at 150 dpi. Warnings from repeated fits go into the fitlog, not the notebook output.
- The test suite from Plan 1 plus this plan's tests should stay under about 3 minutes (the spec's 1 minute target was set before the MS fits were counted). Fits in tests use `search_reps ≤ 5`, n ≤ 500 and module-scoped fixtures.
- Commits are authored by the repo's git user, with **no co-author trailer**.
- The user's email and any GCP project id never appear in code, notebooks or commits.

## Review Focus

Failure modes the spec implies but its test list does not exercise, most likely first. Each has a test in the task named.

1. **NaN in the estimation window, or NaN after the origin.** A missing GPR/VIX week must not be silently dropped or filled (it would shift every lag), and NaNs after the origin (e.g. in 2026) must not matter. `build_design` raises, naming the column and dates, for NaN inside the window; later rows are ignored (Task 1).
2. **A flat or boundary likelihood.** Iran has no effect, a regressor is constant, or γ̂ sits at 0: the Hessian is singular. `param_cov` and `garchx_fit` return NaN standard errors instead of crashing; `lr_test` clips a negative LR (optimizer shortfall) to 0 and flags it (Tasks 4, 6, 9).
3. **A crash mid-origin.** Forecast rows written, fitlog row not: on resume those rows must be dropped, never duplicated, and the warm-start state must be rebuilt from the checkpoint so the resumed run equals an uninterrupted one (Task 10).
4. **Lookahead through the panel.** Scrambling every panel row after the origin must change nothing in that origin's forecasts, for the MS, HAR and GARCH-X models alike (Task 10).
5. **A fit that fails inside a frozen run.** A non-converged or degenerate M3 week must still write its forecast row, be logged as such, and not abort the run or switch the variant (Task 10).

---

### Task 1: Simulation fixtures, `MSSpec`, `design.py`

*Wed Nov 25.*

**Files:**
- Create: `src/brent/ms.py` (this task adds only `MSSpec`, `make_specs`, `with_iran`), `src/brent/design.py`, `tests/test_design.py`
- Modify: `tests/conftest.py` (add `sim_panel`)

**Interfaces:**
- Consumes: `config` (windows, `TVTP_CONTROLS`, `IRAN_VAR`, `X_*`, `GARCHX_SCALE_WINDOW`, `STANDARDIZE_WINDOW*`, `SAMPLE_START`), `paths.FROZEN`.
- Produces (signatures exactly as spec §8.4 and §8.5):
  - `ms.MSSpec` (frozen dataclass: `name, tvtp_vars, iran_vars, iran_only_NH, phi_switching, mean_exog=(), kind="msar", canonical_by="sigma2", std_window=C.STANDARDIZE_WINDOW, start=C.SAMPLE_START`)
  - `ms.make_specs(phi_switching, variant=C.M3_VARIANT, controls=tuple(C.TVTP_CONTROLS), iran_var=C.IRAN_VAR) -> dict[str, MSSpec]` (keys `"M1","M2","M3"`)
  - `ms.with_iran(spec, iran_vars, name) -> MSSpec`: replaces the old restricted variables inside `tvtp_vars` by the new ones, keeps the controls and every other field
  - `design.Design`, `standardization_stats(panel, vars, window) -> DataFrame` (index = vars, columns `mean, sd`, `ddof=1`), `build_design`, `har_design`, `garchx_design`
  - `design.freeze_std_stats(panel) -> Path` **[plan call]**: writes `results/frozen/std_stats.csv` (columns `var, window, window_start, window_end, mean, sd`) once, for the TVTP variables, the iran/tone variables and `ovx` on their windows; the hourly-sample window for `log_rv_hourly, gpr, vix, iran_gdelt`; and the GARCH-X scale means of the levels (spec §9.3)
  - fixture `sim_panel(n=400, beta=1.5, phi=(0.3, 0.3), seed=405) -> pd.DataFrame` in `tests/conftest.py`

**Notes that fix the choices the spec leaves open:**
- `design.py` imports `MSSpec` only under `typing.TYPE_CHECKING` (circular import otherwise; `ms.py` imports `Design`).
- `Design.X` has column `r_lag`, plus one column per `mean_exog` variable (z-scored with `spec.std_window`, dated t−1) **[plan call: the spec does not say how the Iran-in-the-mean variable is scaled]**. `Design.Z` has columns `["const", *tvtp_vars]`.
- `kind="mshar"` raises `NotImplementedError("E1 is built in Plan 3")` in `build_design` (and later in `forecast.one_step`). **[plan call]**
- `har_design` X columns: `const, rv_1, rv_4, rv_13` + `x_vars` (z-scored logs, `std_window` stats). `garchx_design` X columns: `rv_col`, then `x_vars`.

- [ ] **Step 1: Add the `sim_panel` fixture to `tests/conftest.py`**

  A factory returning `make(n=400, beta=1.5, phi=(0.3, 0.3), seed=405)`. It builds a weekly panel of `n` Fridays from 2005-01-07 with columns `date, r, rv, log_rv, gpr, vix, iran_gdelt, iran_conflict, iran_coop, iran_tone, iran_goldstein, aigpr_iran, rv_hourly, log_rv_hourly`. Simulation (all with `np.random.default_rng(seed)`):
  - Regressors: `gpr, vix, iran_gdelt` are AR(1) (φ=0.8, innovation sd 0.5); `iran_conflict`, `iran_coop`, `iran_tone`, `iran_goldstein` are the same recipe with their own draws.
  - States: P(N→H)(t) = Λ(−2.2 + beta·iran_gdelt(t−1) + 0.3·gpr(t−1)); P(H→N)(t) = Λ(−1.0 + 0.2·vix(t−1)).
  - Returns: r(t) = μ_S + φ_S·r(t−1) + σ_S·ε with μ = (0.2, −0.3), σ = (1.0, 3.0), φ from the argument.
  - `rv = σ_S² · rng.chisquare(1) + 0.05`; `log_rv = ln(rv)`.
  - `rv_hourly = 1.1 · rv · lognormal(0, 0.2)`, NaN before `C.HOURLY_SAMPLE_START`. `log_rv_hourly = ln(rv_hourly)`.

  With `n=1140` the panel spans the full sample, so window-based tests work.

- [ ] **Step 2: Write the failing tests** (`tests/test_design.py`)

  ```python
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
      with pytest.raises(ValueError, match="vix"):
          D.build_design(bad, spec(), end)

  def test_mean_exog_enters_x(sim_panel):
      p = sim_panel(n=1140)
      d = D.build_design(p, spec(mean_exog=("iran_gdelt",)), "2018-12-28")
      assert list(d.X.columns) == ["r_lag", "iran_gdelt"] and "iran_gdelt" in d.x_next.index

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
  ```

- [ ] **Step 3: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_design.py -q`
  Expected: FAIL (`brent.ms` / `brent.design` missing).

- [ ] **Step 4: Implement `MSSpec`, `make_specs`, `with_iran` in `src/brent/ms.py` and everything in `src/brent/design.py`**

  - `build_design`: estimation rows are `spec.start + 1 week … end` (Fridays). Take the used columns (`r`, `tvtp_vars`, `mean_exog`) over rows `[spec.start, end]`; if any is NaN, raise `ValueError` naming the column and the dates. Rows after `end` are never read. `std_stats` comes from `standardization_stats(panel, vars, spec.std_window)`.
  - `har_design`/`garchx_design`: same NaN rule for the columns they read on `[start, end]`. This is where hourly-sample weeks before `HOURLY_SAMPLE_START` must be excluded by `start`, never by dropping NaN.
  - In `har_design`, `x_vars` are `panel[col]` z-scored with `standardization_stats(panel, x_vars, std_window)` (the **TRAIN** stats even when `start` is the hourly start, spec §9.3).
  - In `garchx_design`, the level of `x_vars[k]` is `exp(panel[k])`, divided by its mean over `scale_window`; `rv_col` is used raw.

- [ ] **Step 5: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_design.py -q`
  Expected: all PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/brent/ms.py src/brent/design.py tests/test_design.py tests/conftest.py
  git commit -m "Add MSSpec, design builders for MS, HAR-X and GARCH-X, and simulated-panel fixture"
  ```

---

### Task 2: `ms.py` core: model construction, canonical labels, durations, deck coefficients

*Wed Nov 25.*

**Files:**
- Modify: `src/brent/ms.py`
- Create: `tests/test_ms.py`

**Interfaces:**
- Consumes: `design.Design`, `ms.MSSpec`.
- Produces:
  - `ms.make_model(design, spec, **overrides) -> MarkovRegression` **[plan call: `**overrides` pass straight to `MarkovRegression`, e.g. `switching_trend=False`, for the RQ1 equality tests]**. The returned model carries two attributes: `tvtp_name_map: dict[str, int]` (`{"const": 0, "gpr": 1, …}`) and `readable: dict[str, str]` (statsmodels name → readable name).
  - `ms.canonicalize(params: pd.Series, model, by: str = "sigma2") -> tuple[pd.Series, bool]`
  - `ms.transition_coefs_deck(params, tvtp_names: list[str]) -> pd.DataFrame` (index `["N->H", "H->N"]`, columns = `tvtp_names`)
  - `ms.expected_durations(params, model, exog_tvtp=None) -> dict[str, float]` (keys `"N"`, `"H"`; default regressors = the zero row `[1, 0, …, 0]`)
  - `ms.constraint_info(spec, model) -> dict` **[plan call]**: `{"fixed": {p[1->0].tvtpJ: 0.0 for each restricted var (empty unless iran_only_NH)}, "beta": {var: "p[0->0].tvtpJ"}}`
  - `ms.readable_to_sm(model) -> dict[str, str]` (the inverse of `model.readable`; names are unique, so it is a bijection)

**Readable names (bijective):** `p[0->0].tvtpJ` → `p00_<var>` (`p00_const` for J=0), `p[1->0].tvtpJ` → `p10_<var>`; M1: `p[0->0]` → `p00`, `p[1->0]` → `p10`; `const[0|1]` → `mu_N|mu_H`; `x1[0|1]` → `phi_N|phi_H` (`x1` → `phi` when φ does not switch); further mean exog `xK` → `theta_<var>`; `sigma2[0|1]` → `sigma2_N|sigma2_H`.

**Relabel rule (the algorithm the signature does not determine).** `canonicalize` swaps regimes 0 and 1 iff `sigma2[0] > sigma2[1]` (`by="const"`: iff `const[0] > const[1]`). Swap rule on the **statsmodels-named** params:
```text
for every name ending in [0] / [1] with a twin (const, x1.., sigma2): exchange the two values
TVTP model, for every column J:  new p[0->0].tvtpJ = -old p[1->0].tvtpJ
                                 new p[1->0].tvtpJ = -old p[0->0].tvtpJ
M1 (probabilities):              new p[0->0] = 1 - old p[1->0];  new p[1->0] = 1 - old p[0->0]
```
(P(S=0 | ·) in the new labels is 1 − P(S=0 | ·) in the old, and the logit of 1 − Λ(a) is −a.) Parameters without a twin (non-switching `x1`, `const`) stay. Returns `(params, relabeled)`.

**Deck sign mapping (spec §8.4, canonical labels):** deck coef in P(N→H) = **−** coef in `p[0->0]`; deck coef in P(H→N) = **+** coef in `p[1->0]`.

- [ ] **Step 1: Write the failing tests** (`tests/test_ms.py`, part 1)

  ```python
  import numpy as np, pandas as pd, pytest
  from dataclasses import replace
  from brent import ms, design as D, config as C

  @pytest.fixture(scope="module")
  def dsn(sim_panel):
      p = sim_panel(n=300, beta=2.0, seed=405)
      s = ms.make_specs(False)
      end = p["date"].iloc[-1]
      return {k: (D.build_design(p, v, end), v) for k, v in s.items()}, p

  def test_param_names_layout(dsn):                       # R10: names are positional in statsmodels
      (d, s) = dsn[0]["M2"]
      m = ms.make_model(d, s)
      assert list(m.param_names) == (
          [f"{a}.tvtp{j}" for j in range(4) for a in ("p[0->0]", "p[1->0]")]
          + ["const[0]", "const[1]", "x1", "sigma2[0]", "sigma2[1]"])
      assert m.tvtp_name_map == {"const": 0, "gpr": 1, "vix": 2, "log_rv": 3}
      m1 = ms.make_model(*dsn[0]["M1"])
      assert list(m1.param_names) == ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "x1", "sigma2[0]", "sigma2[1]"]
      m2 = ms.make_model(d, replace(s, phi_switching=True))
      assert "x1[0]" in m2.param_names and "x1[1]" in m2.param_names

  def test_readable_names_are_a_bijection(dsn):
      m = ms.make_model(*dsn[0]["M3"])
      assert len(set(m.readable.values())) == len(m.readable) == len(m.param_names)
      assert m.readable["p[0->0].tvtp4"] == "p00_iran_gdelt" and m.readable["sigma2[1]"] == "sigma2_H"
      assert ms.readable_to_sm(m)["p10_vix"] == "p[1->0].tvtp2"

  def _arbitrary(m, seed):
      rng = np.random.default_rng(seed)
      p = pd.Series(rng.normal(0, 0.4, len(m.param_names)), index=m.param_names)
      for n in p.index:
          if n.startswith("sigma2"): p[n] = 1.0 if n.endswith("[0]") else 4.0
          if n.startswith("p[") and ".tvtp" not in n: p[n] = 0.8 if n.startswith("p[0") else 0.3
      return p

  def _flip(p, m):                                         # independent implementation of the rule
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
      assert m.loglike(f) == pytest.approx(m.loglike(p), abs=1e-8)       # the rule itself is valid
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
  ```

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_ms.py -q`
  Expected: FAIL (`make_model` etc. not defined).

- [ ] **Step 3: Implement the functions listed under Interfaces in `src/brent/ms.py`**

  - `make_model`: `MarkovRegression(design.y, k_regimes=2, trend="c", exog=design.X, switching_exog=[spec.phi_switching] + [False] * len(spec.mean_exog), switching_variance=True, exog_tvtp=design.Z)`, so an Iran-in-the-mean variable gets **one common coefficient** (spec §7.8; its statsmodels name is `x2`, readable name `theta_<var>`); for M1 (`design.Z is None`) omit `exog_tvtp`. Never `MarkovAutoregression`. Build `tvtp_name_map` from `design.Z.columns` and `readable` from the rule above. `design` is duck-typed (`y`, `X`, `Z`), so Task 6's bootstrap can pass a light object.
  - `expected_durations`: `model.regime_transition_matrix(params, exog_tvtp=zero_row[None, :])`; N: 1/(1 − P[0,0]); H: 1/(1 − P[1,1]). For M1 call without `exog_tvtp`.
  - `transition_coefs_deck`: reads `p[0->0].tvtpJ`/`p[1->0].tvtpJ` for J = index of each name in `tvtp_names`.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_ms.py -q`
  Expected: all PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/ms.py tests/test_ms.py
  git commit -m "Add MS model construction, canonical relabeling, durations and deck sign mapping"
  ```

---

### Task 3: `ms.py` unconstrained fitting: seeding, degeneracy, restarts, nesting starts

*Wed Nov 25.*

**Files:**
- Modify: `src/brent/ms.py`, `tests/test_ms.py`

**Interfaces:**
- Consumes: Task 2 functions; `config` (`VAR_FLOOR_FRAC`, `MIN_DURATION`, `SEED`).
- Produces:
  - `ms.MSFit` as in spec §8.4, plus two last fields `fixed: dict[str, float] = field(default_factory=dict)` and `n_warnings: int = 0` **[plan call: `lr_test` needs the free-parameter count, `len(params) - len(fixed)`; the spec asks for fit warnings to go to the fitlog, so `fit_model` counts the ones it catches]**
  - `ms.fit_model(model, spec, start_params, search_reps, n_restarts, seed, extra_starts=()) -> MSFit` **[plan call: the low-level fit on an already-built model, used by `fit_ms` and by the RQ1 restricted refits]**
  - `ms.fit_ms(design, spec, start_params, search_reps, n_restarts, seed, extra_starts=()) -> MSFit` (spec signature plus the optional `extra_starts: tuple[pd.Series, ...]` **[plan call: the spec describes nesting starts in §8.4 but its signature has no way to pass one]**)
  - `ms.is_degenerate(params, model, var_y: float, floor_frac=None, min_duration=None) -> bool` (defaults read from `config` at call time)
  - `ms.pick_best(fits: list[MSFit]) -> MSFit`: best `loglike` among non-degenerate; if all are degenerate, best `loglike` with `degenerate=True`
  - `ms.nest_start(small: MSFit, big_model, big_spec) -> pd.Series` **[plan call]**: M1 → TVTP model: `p[0->0].tvtp0 = logit(p00)`, `p[1->0].tvtp0 = logit(p10)`, all other TVTP coefficients 0, means/φ/σ² copied (φ copied to both regimes if the big model switches). TVTP → TVTP with more regressors: copy every shared name, new coefficients 0. Names are matched through `readable`, never by position. `nest_start` copies every shared name and sets any new parameter to 0, including the common mean coefficient `theta_<var>` when the big spec has `mean_exog` (so an M3-with-mean-exog fit starts at M3's optimum).
  - `ms.nest_parent(spec, specs: list[MSSpec]) -> MSSpec | None` **[plan call]**: among `specs` with the same `kind`, `phi_switching` and `mean_exog` (equal, so an M3-with-mean-exog spec has no parent in the ladder; the caller, i.e. Plan 3's robustness run, supplies the M3 fit as `extra_starts` via `nest_start`), those with `set(tvtp_vars)` a proper subset of `spec`'s and `set(iran_vars) ⊆` spec's; return the one with the most `tvtp_vars`. So M2 → M1, M3 → M2, M3tone → M2, M3toneL/M3gold → M3.

**`fit_model` behavior (the candidate logic, not a body):**
1. Candidates: (a) if `start_params is None`, `model.fit(search_reps=search_reps, rng=seed, disp=0, method="bfgs", maxiter=…)`; otherwise `model.fit(start_params=start_params, search_reps=0, disp=0)`; (b) each `extra_starts` entry fitted without search; (c) `n_restarts` perturbed refits, restart k from `start_params` (or the best candidate so far if `start_params is None`) plus `np.random.default_rng(seed + k).normal(0, 0.5 · scale)` on the **untransformed** parameters, `k = 1…n_restarts`.
2. Every candidate is canonicalized, then `is_degenerate` is evaluated on the canonical params. `pick_best` chooses.
3. Warnings (`ConvergenceWarning`, `RuntimeWarning`) are caught per candidate; a candidate is converged iff statsmodels says so. `MSFit.converged` is that of the chosen candidate; `n_restarts_used = n_restarts`; `relabeled` is True if the chosen candidate was flipped. Time with `time.perf_counter`.
4. `var_y` for the degeneracy rule is `design.y.var()`; `fit_model` takes it from `model.endog`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ms.py`)

  ```python
  from types import SimpleNamespace as NS

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
      pd.testing.assert_series_equal(f.params, again.params)               # same seed, search_reps > 0

  def test_nesting_ll_and_restarts_counted(small):
      p, s, d, fits = small
      assert fits["M2"].loglike >= fits["M1"].loglike - 1e-6
      r = ms.fit_ms(d["M1"], s["M1"], fits["M1"].params, 0, 2, seed=7)
      assert r.n_restarts_used == 2 and r.loglike >= fits["M1"].loglike - 1e-6   # warm start never worse

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
  ```

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_ms.py -q`
  Expected: FAIL on the new tests.

- [ ] **Step 3: Implement `MSFit`, `fit_model`, `fit_ms` (unconstrained branch), `is_degenerate`, `pick_best`, `nest_start`, `nest_parent`**

  `fit_ms` = `make_model` + `fit_model`; the constrained branch is added in Task 4. For `statsmodels` 0.15 first confirm the keyword: `inspect.signature(MarkovRegression.fit)` must contain `rng` (spec §8.4). If it is named differently, use that name everywhere and fix the spec's wording in the commit message.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_ms.py -q`
  Expected: all PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/ms.py tests/test_ms.py
  git commit -m "Add seeded MS fitting with restarts, degeneracy rule and nesting starts"
  ```

---

### Task 4: `ms.py` constrained M3, first-fit and warm-start procedures, Hessian covariance

*Wed Nov 25 → Thu Nov 26 morning.*

**Files:**
- Modify: `src/brent/ms.py`, `tests/test_ms.py`

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces:
  - `ms.fit_constrained(model, start_params: pd.Series, fixed: dict[str, float], n_restarts: int, seed: int) -> tuple[pd.Series, float, bool]` (params in natural space incl. the fixed values, loglike, converged)
  - `ms.fit_ms` now handles restricted specs (`spec.iran_vars` non-empty and `spec.iran_only_NH`): the constrained path below. For `iran_only_NH=False` (variant `iran_both_logits`) and for M1/M2 the path of Task 3 is unchanged; `MSFit.fixed` is `{}`.
  - `ms.param_cov(model, params, fixed: dict[str, float]) -> pd.DataFrame`: covariance of the **free** parameters (index = free names); NaN-filled, not an exception, if the Hessian is singular

**Constrained-fit algorithm (spec §8.4):**
- `fit_constrained` maximizes `model.loglike(model.transform_params(full(u)))` over the free **untransformed** parameters `u` (`u0 = model.untransform_params(start)` restricted to the free names). `full()` inserts the fixed values (TVTP coefficients are identity-transformed, so fixing them in either space is the same). Use `scipy.optimize.minimize(method="BFGS")` on the negative loglike, then Nelder-Mead from its result if BFGS did not report success. Restarts: `u + np.random.default_rng(seed + k).normal(0, 0.5, len(u))`, `k = 1…n_restarts`, keep the best non-degenerate LL. The fixed coefficient of the returned params is exactly `0.0`.
- First fit (`start_params is None`): unconstrained fit with `search_reps` (Task 3 path, canonicalized) → `fit_constrained` from those params with `constraint_info(spec, model)["fixed"]`. Extra starts (a nesting start from M2) are also passed through `fit_constrained`; best LL wins.
- Warm fit (`start_params` given): `fit_constrained` directly from last week's canonical constrained params. If the result has `sigma2[0] > sigma2[1]`, discard it and run the first-fit procedure; set `refit_from_scratch = True`.
- `param_cov`: `statsmodels.tools.numdiff.approx_hess3` of the summed log-likelihood in natural space over the free parameters, `cov = inv(-H)`; on `LinAlgError` or a non-positive diagonal return NaN for the affected entries.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ms.py`)

  ```python
  @pytest.fixture(scope="module")
  def ladder(small):
      p, s, d, fits = small
      m3m = ms.make_model(d["M3"], s["M3"])
      f3 = ms.fit_ms(d["M3"], s["M3"], None, 5, 0, seed=405,
                     extra_starts=(ms.nest_start(fits["M2"], m3m, s["M3"]),))
      return f3

  def test_constrained_fit_is_exact_and_nested(small, ladder):
      p, s, d, fits = small; f3 = ladder
      assert f3.params["p[1->0].tvtp4"] == 0.0 and f3.fixed == {"p[1->0].tvtp4": 0.0}
      assert f3.loglike >= fits["M2"].loglike - 1e-6                         # nesting start
      both = replace(s["M3"], iran_only_NH=False)
      fu = ms.fit_ms(d["M3"], both, f3.params, 0, 0, seed=405)
      assert fu.fixed == {} and fu.loglike >= f3.loglike - 1e-6              # unconstrained >= constrained
      assert f3.params["sigma2[1]"] > f3.params["sigma2[0]"]

  def test_beta_sign_when_iran_raises_p_NH(sim_panel):                       # beta = 2.5 in the simulation
      p = sim_panel(n=500, beta=2.5, seed=11); s = ms.make_specs(False)
      d = {k: D.build_design(p, v, p["date"].iloc[-1]) for k, v in s.items()}
      f2 = ms.fit_ms(d["M2"], s["M2"], None, 5, 0, seed=11)
      f3 = ms.fit_ms(d["M3"], s["M3"], None, 5, 0, seed=11,
                     extra_starts=(ms.nest_start(f2, ms.make_model(d["M3"], s["M3"]), s["M3"]),))
      beta = ms.transition_coefs_deck(f3.params, list(d["M3"].Z.columns)).loc["N->H", "iran_gdelt"]
      assert beta > 0

  def test_param_cov_free_params_only_and_nan_not_raise(small, ladder):
      p, s, d, fits = small; f3 = ladder
      cov = ms.param_cov(f3.model, f3.params, f3.fixed)
      assert set(cov.index) == set(f3.params.index) - set(f3.fixed) and np.allclose(cov, cov.T, equal_nan=True)
      flat = f3.params.copy(); flat["sigma2[0]"] = 1e-12                     # a degenerate point: must not raise
      assert isinstance(ms.param_cov(f3.model, flat, f3.fixed), pd.DataFrame)

  def test_warm_start_stays_constrained_and_flipped_result_triggers_refit(small, ladder, monkeypatch):
      p, s, d, fits = small; f3 = ladder
      w = ms.fit_ms(d["M3"], s["M3"], f3.params, 0, 0, seed=406)
      assert w.fixed and w.params["p[1->0].tvtp4"] == 0.0 and not w.refit_from_scratch
      real, calls = ms.fit_constrained, {"n": 0}
      def fake(model, start, fixed, n_restarts, seed):
          params, ll, ok = real(model, start, fixed, n_restarts, seed); calls["n"] += 1
          if calls["n"] == 1:                                                # first call = the warm fit
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
  ```

  ```python
  def test_mean_exog_is_one_common_coefficient(small, ladder):             # spec §7.8: Iran in the mean
      from brent import insample as I
      p, s, d, fits = small; f3 = ladder
      sm = replace(s["M3"], mean_exog=("iran_gdelt",), name="M3_mean")
      dm = D.build_design(p, sm, p["date"].iloc[-1]); mm = ms.make_model(dm, sm)
      assert len(mm.param_names) == len(f3.model.param_names) + 1
      assert [n for n in mm.param_names if n.startswith("x2")] == ["x2"] and mm.readable["x2"] == "theta_iran_gdelt"
      fm = ms.fit_ms(dm, sm, None, 5, 0, seed=405, extra_starts=(ms.nest_start(f3, mm, sm),))
      assert fm.params["p[1->0].tvtp4"] == 0.0 and fm.loglike >= f3.loglike - 1e-6
      assert I.lr_test(f3, fm)["df"] == 1
  ```

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_ms.py -q`
  Expected: FAIL on the new tests (`fit_constrained`, `param_cov` missing).

- [ ] **Step 3: Implement `fit_constrained`, the constrained branches of `fit_ms`/`fit_model`, and `param_cov`**

  Follow the algorithm above. `fit_ms` for `kind="mshar"` is not built here (Plan 3).

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_ms.py -q`
  Expected: all PASS. If `test_beta_sign` fails because the fit stalls, raise `search_reps` in that test to 10 before touching the implementation; if it still fails, the sign mapping or the nesting start is wrong.

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/ms.py tests/test_ms.py
  git commit -m "Add constrained M3 fit, first-fit and warm-start procedures, Hessian covariance"
  ```

---

### Task 5: `forecast.py`: the one-step forecaster

*Thu Nov 26.*

**Files:**
- Create: `src/brent/forecast.py`, `tests/test_forecast.py`

**Interfaces:**
- Consumes: `ms.MSFit`, `ms.make_model`, `design.Design`.
- Produces (spec §8.6):
  - `forecast.OneStep` dataclass: `pi_H, var_fc, mean_fc, p_exceed, xi_H_origin`
  - `forecast.one_step(fit: MSFit, design: Design, thr: float) -> OneStep`
  - `forecast.predicted_probs(model, params) -> np.ndarray`, shape `(nobs, 2)`, row t = π(t | t−1) in canonical labels (column 1 = H)
  - `forecast.filter_fixed(model_for_longer_sample, params) -> pd.Series`: filtered ξ_H(t | t) with fixed `params`, indexed by the model's dates
  - `forecast.highvol_threshold(panel) -> float` **[plan call]**: `np.percentile(|r|, C.HIGHVOL_PCTL)` over `C.TRAIN`. Plan 3's `evaluate.highvol_threshold` can re-export this; it lives here so `oos.py` does not depend on `evaluate.py`.

**Formulas (spec §8.6):** ξ(T|T) = last row of `model.filter(params).filtered_marginal_probabilities` (never smoothed). P(T+1) = `model.regime_transition_matrix(params, exog_tvtp=design.z_next.values[None, :])[:, :, 0]` (M1: no exog); columns sum to 1, P[i, j] = P(S(T+1)=i | S(T)=j); π = P·ξ. mᵢ = μᵢ + φᵢ·r(T) (+ θ·x for each `mean_exog` entry of `design.x_next`); when φ does not switch, φ is the common `x1`. `var_fc` = Σ πᵢ(σᵢ² + mᵢ²); `mean_fc` = Σ πᵢ mᵢ; `p_exceed` = Σ πᵢ[1 − Φ((thr − mᵢ)/σᵢ) + Φ((−thr − mᵢ)/σᵢ)]. `kind="mshar"` raises `NotImplementedError` (Plan 3).

- [ ] **Step 1: Write the failing tests** (`tests/test_forecast.py`)

  ```python
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
      phi = [pr["x1[0]"], pr["x1[1]"]] if s.phi_switching else [pr["x1"]] * 2
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
  ```

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_forecast.py -q`
  Expected: FAIL (`brent.forecast` missing).

- [ ] **Step 3: Implement `forecast.py` per the formulas above**

  `predicted_probs`: ξ(0|−1) = the model's initial probabilities (`model.initial_probabilities(params)`), then row t = P[:, :, t] · ξ(t−1 | t−1) with filtered probabilities from `model.filter(params)`. The test compares against `model.smooth(...)`, which is allowed in tests only.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_forecast.py -q`
  Expected: all PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/forecast.py tests/test_forecast.py
  git commit -m "Add one-step Markov-switching forecaster and filtered-probability helpers"
  ```

---

### Task 6: `insample.py`: LR tests, Wald, bootstrap, ladder fits, coefficient tables

*Thu Nov 26.*

**Files:**
- Create: `src/brent/insample.py`, `tests/test_insample.py`

**Interfaces:**
- Consumes: `ms` (all), `design.build_design`, `forecast.filter_fixed`, `config`.
- Produces (spec §8.9, except where marked):
  - `lr_test(fit_restricted, fit_full) -> dict` with keys `lr_raw, stat, df, pvalue, negative, k_restricted, k_full`. `df = k_full − k_restricted` with `k = len(params) − len(fixed)`; `stat = max(lr_raw, 0)`; `negative = lr_raw < 0` (optimizer shortfall, flagged not hidden); `pvalue = chi2.sf(stat, df)`.
  - `chibar_pvalue(lr: float, k: int) -> float`: Σⱼ C(k,j)/2ᵏ · P(χ²ⱼ > lr), with P(χ²₀ > lr) = 1 if lr ≤ 0 else 0
  - `wald_equal(fit, a: str, b: str) -> dict`: H0 β_a = β_b (deck convention) for restricted variables `a`, `b`; `{stat, pvalue, df: 1, diff, se}`; uses `param_cov` and `constraint_info(...)["beta"]`
  - `circular_shift(series, k: int, window: tuple[str, str]) -> pd.Series`: `np.roll` inside the window only; values outside the window are unchanged
  - `equality_tests(design, spec) -> pd.DataFrame`: spec is an M1 spec with `phi_switching=True`; rows `mu`, `phi`; columns `est_N, est_H, lr, df, p_lr, wald, p_wald`. The restricted refits are `fit_model(make_model(design, spec, switching_trend=False))` and `…switching_exog=False`, same seed and search effort; the Wald tests use the unrestricted `param_cov` on `const[0]-const[1]` and `x1[0]-x1[1]`. If an LR comes out negative, refit the restricted model once from the unrestricted params with the two regimes' values replaced by their mean, and keep the better.
  - `bootstrap_lr_1v2(y, X, n_reps: int, seed: int, search_reps: int) -> dict` with keys `lr, pvalue, n_reps, n_degenerate, lr_boot, ll_one, ll_two`. One-regime model: Gaussian AR(1) by OLS/MLE on `[1, r_lag]` (`X` has column `r_lag`). Replicate b: simulate `len(y)` observations from the fitted one-regime model with `np.random.default_rng(seed + b)`, using the same `X` recursion (`r(t) = c + φ·r(t−1) + σ·ε`, started at the first `X` value); fit both models; M1 fit via `fit_model` with `phi_switching=True`, `search_reps`, seed `seed + b`, same degeneracy rule. `pvalue = (1 + #{lr_boot ≥ lr}) / (n_reps + 1)`; degenerate replicates are kept and counted.
  - `fit_ladder(panel, end, phi_switching, iran_var=C.IRAN_VAR, n_restarts=0) -> dict[str, MSFit]`: M1, M2, M3 with the nesting starts (M2 from M1, M3 from M2), `search_reps=C.SEARCH_REPS_FIRST`, seed `C.SEED`, `C.M3_VARIANT` (the optional `n_restarts` is a **[plan call]** so notebook 04's reported fits can use the frozen `C.N_RESTARTS`)
  - `fit_tone(panel, end, phi_switching, ladder, n_restarts=0, variants=("M3tone", "M3toneL", "M3gold")) -> dict[str, MSFit]` (**[plan call]** `variants` selects which to fit; notebook 04 passes `variants=("M3tone",)`, since `M3toneL` and `M3gold` are robustness in notebook 08): `M3tone` (`with_iran(M3, C.IRAN_TONE_VARS, "M3tone")`, nested start from `ladder["M2"]`), `M3toneL` (`("iran_gdelt", "iran_tone")`) and `M3gold` (`("iran_gdelt", "iran_goldstein")`), both nested from `ladder["M3"]`
  - `coef_table(fit, label: str) -> pd.DataFrame` **[plan call]**: one row per transition coefficient, columns `model, logit` (`N->H`/`H->N`), `var`, `coef` (deck sign), `se`, `z`, `p`, `fixed`. Restricted H→N coefficients under `constrained` appear with `coef = 0.0`, `fixed = True`, `se = NaN`. SEs come from `param_cov`; the N→H SE is the SE of the statsmodels coefficient (the sign flip does not change it).
  - `fit_summary(fit, nobs: int) -> dict` **[plan call]**: `loglike, k, aic = 2k − 2ll, bic = k·ln(nobs) − 2ll, dur_N, dur_H, sigma2_N, sigma2_H, mu_N, mu_H`
  - `garchx_lr` is added in Task 9 (it needs `benchmarks`). The two placebo functions `placebo_lr` and `placebo_harx` of spec §8.9 are **not** built here **[plan call]**: they are used only by notebook 08, and `placebo_harx` needs `evaluate.qlike` (Plan 3). `circular_shift` is the building block both need.

- [ ] **Step 1: Write the failing tests** (`tests/test_insample.py`)

  ```python
  import numpy as np, pandas as pd, pytest
  from types import SimpleNamespace as NS
  from scipy.stats import chi2
  from brent import insample as I, ms, design as D, config as C

  def test_chibar_values():
      assert I.chibar_pvalue(2.706, 1) == pytest.approx(0.05, abs=2e-4)
      assert I.chibar_pvalue(0.0, 3) == 1.0
      lr = 4.0
      expect = 0.375 * chi2.sf(lr, 1) + 0.375 * chi2.sf(lr, 2) + 0.125 * chi2.sf(lr, 3)   # weights 1/8, 3/8, 3/8, 1/8
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
  def fits(sim_panel):
      p = sim_panel(n=300, beta=2.0, seed=405)
      return p, I.fit_ladder(p, p["date"].iloc[-1], phi_switching=False)

  def test_ladder_nesting_and_df(fits, monkeypatch):
      p, L = fits
      assert L["M1"].loglike <= L["M2"].loglike + 1e-6 and L["M2"].loglike <= L["M3"].loglike + 1e-6
      assert I.lr_test(L["M2"], L["M3"])["df"] == 1                          # constrained M3 vs M2
      assert I.lr_test(L["M1"], L["M2"])["df"] == 6                          # 3 controls in 2 logits
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

  def test_tone_lr_df_under_both_variants(sim_panel, monkeypatch):          # spec §10: 2 constrained, 4 both-logits
      p = sim_panel(n=300, beta=2.0, seed=405); end = p["date"].iloc[-1]
      for variant, df in (("constrained", 2), ("iran_both_logits", 4)):
          monkeypatch.setattr(C, "M3_VARIANT", variant)
          L = I.fit_ladder(p, end, False); T = I.fit_tone(p, end, False, L, variants=("M3tone",))
          assert I.lr_test(L["M2"], T["M3tone"])["df"] == df

  def test_wald_equal_zero_for_identical_coefficients():
      m = NS(); params = pd.Series({"p[0->0].tvtp1": 0.7, "p[0->0].tvtp2": 0.7, "p[1->0].tvtp1": 0.0, "p[1->0].tvtp2": 0.0})
      fit = NS(model=m, params=params, fixed={"p[1->0].tvtp1": 0.0, "p[1->0].tvtp2": 0.0})
      cov = pd.DataFrame(np.array([[0.04, 0.01], [0.01, 0.09]]), index=list(params.index[:2]), columns=list(params.index[:2]))
      w = I.wald_equal(fit, "a", "b", _cov=cov, _beta={"a": "p[0->0].tvtp1", "b": "p[0->0].tvtp2"})
      assert w["stat"] == pytest.approx(0.0)

  def test_coef_table_deck_signs_and_fixed_rows(fits):
      p, L = fits
      tab = I.coef_table(L["M3"], "M3")
      row = tab[(tab["var"] == "iran_gdelt") & (tab["logit"] == "N->H")].iloc[0]
      assert row["coef"] == pytest.approx(-L["M3"].params["p[0->0].tvtp4"]) and row["se"] > 0
      fx = tab[(tab["var"] == "iran_gdelt") & (tab["logit"] == "H->N")].iloc[0]
      assert fx["fixed"] and fx["coef"] == 0.0 and np.isnan(fx["se"])

  def test_fit_summary_aic_bic(fits):
      p, L = fits; s = I.fit_summary(L["M2"], nobs=299)
      assert s["aic"] == pytest.approx(2 * s["k"] - 2 * s["loglike"]) and s["bic"] == pytest.approx(s["k"] * np.log(299) - 2 * s["loglike"])

  def test_bootstrap_lr_small_and_deterministic(sim_panel):
      p = sim_panel(n=160, beta=0.0, seed=3); s = ms.make_specs(True)["M1"]
      d = D.build_design(p, s, p["date"].iloc[-1])
      a = I.bootstrap_lr_1v2(d.y, d.X, n_reps=4, seed=C.SEED, search_reps=2)
      b = I.bootstrap_lr_1v2(d.y, d.X, n_reps=4, seed=C.SEED, search_reps=2)
      assert a["lr"] >= 0 and 1 / 5 <= a["pvalue"] <= 1.0 and len(a["lr_boot"]) == 4
      assert a["pvalue"] == b["pvalue"] and np.allclose(a["lr_boot"], b["lr_boot"])
      assert 0 <= a["n_degenerate"] <= 4

  def test_equality_tests_shape(sim_panel):
      p = sim_panel(n=300, beta=0.0, seed=5); s = ms.make_specs(True)["M1"]
      d = D.build_design(p, s, p["date"].iloc[-1])
      t = I.equality_tests(d, s)
      assert list(t.index) == ["mu", "phi"] and (t["df"] == 1).all() and (t["p_lr"].between(0, 1)).all()
  ```
  (`wald_equal` takes optional private `_cov`/`_beta` arguments only so the hand-made case can be tested without a fit.)

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_insample.py -q`
  Expected: FAIL (`brent.insample` missing).

- [ ] **Step 3: Implement `insample.py` per the Interfaces**

  `wald_equal`: with `β = −coef(p[0->0])`, Var(βa − βb) = Var(ca) + Var(cb) − 2 Cov(ca, cb); stat = (βa − βb)² / Var, NaN if Var ≤ 0.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_insample.py tests/test_ms.py tests/test_forecast.py tests/test_design.py -q`
  Expected: all PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/insample.py tests/test_insample.py
  git commit -m "Add in-sample LR, Wald, bootstrap and ladder-fit helpers"
  ```

---

### Task 7: Notebook 03 — RQ1 regimes and the φ freeze

*Thu Nov 26. Bootstrap ≈ 20 min runs unattended.*

**Files:**
- Create: `notebooks/03_rq1_regimes.ipynb`, `results/frozen/rq1_decision.json`, `results/frozen/std_stats.csv`, `results/tables/rq1_tests.csv`, `results/tables/rq1_params_M1.csv`, `results/figures/rq1_filtered_prob_2005_2018.png`
- Modify: `src/brent/config.py` (`PHI_SWITCHING`, by hand, after the user agrees)

**Interfaces:**
- Consumes: `data/processed/weekly_panel.csv`; `design.build_design`, `design.freeze_std_stats`; `ms.make_specs`, `ms.fit_ms`, `ms.fit_model`; `insample.equality_tests`, `bootstrap_lr_1v2`, `fit_ladder`, `lr_test`, `fit_summary`; `forecast.predicted_probs`, `one_step`; `plots.setup_style`, `plots.save_fig`.
- Produces: `rq1_decision.json` with `phi_switching` (bool), `lr_stat`, `p_value`, `df`, `sample_end` (`"2018-12-28"`), `git_hash`; `config.PHI_SWITCHING` (True/False). Every later notebook asserts `C.PHI_SWITCHING == json["phi_switching"]`. No new `plots.py` helpers: figures are drawn in the notebook with matplotlib and saved with `plots.save_fig`; the Jan-1-2026 line style belongs to Plan 3.

Flags in the first cell: `FORCE_BOOT = False` (the bootstrap result is saved to `results/tables/rq1_bootstrap_draws.csv` and reloaded when present), `B = C.BOOT_REPS` (199). **Slip rule (spec §11):** if φ is not frozen by Fri Nov 27, set `B = 99` and note it in the table and the markdown.

- [ ] **Step 1: Load, build designs, freeze the standardization stats**

  Load the panel, `design.freeze_std_stats(panel)` (writes `results/frozen/std_stats.csv`), and build the M1 design on `end = C.RQ1_END` with `spec = replace(ms.make_specs(True)["M1"], name="M1")` (**M1 for RQ1 has φ switching on, whatever φ ends up being**; spec §7.3 step 1). Print the number of estimation rows (expected 2005-01-14 → 2018-12-28 = 729).

- [ ] **Step 2: Fit M1 and review the fit with the user**

  `fit_ms(design, spec, None, search_reps=C.SEARCH_REPS_FIRST, n_restarts=0, seed=C.SEED)`. Print μ, φ, σ² by regime, expected durations, weeks with filtered P(H) > 0.5, and `degenerate`/`converged`.
  - **Branch on the fit:** if `degenerate` or not `converged`, refit with `seed = C.SEED + 1, 2, 3` and show all candidates; if none is clean, stop and ask the user.
  - Flag (do not fail) a regime with < 10 weeks of filtered P > 0.5.
  - Figure `rq1_filtered_prob_2005_2018.png`: filtered P(H) (`model.filter(...)`, never smoothed) with the nine `C.EVENTS` weeks of 2005–2018 marked (`weekly.week_ending`).

- [ ] **Step 3: 1 vs. 2 regimes (spec §7.3 step 2)**

  Plain LR (one-regime Gaussian AR(1) vs. M1), BIC of both, and `bootstrap_lr_1v2(design.y, design.X, n_reps=B, seed=C.SEED, search_reps=C.SEARCH_REPS_FIRST)`. Save the draws; print p and `n_degenerate` next to each other. Time one replicate first and print the extrapolated total; if > 30 min, tell the user and use `B = 99`.

- [ ] **Step 4: Equality tests and the φ decision (spec §7.3 steps 3–5)**

  `insample.equality_tests(design, spec)` → LR and Wald for μ(N)=μ(H) and φ(N)=φ(H).
  - **Decision rule (spec §7.3 step 5):** `phi_switching = (p_lr["phi"] < 0.05)`.
  - Write `results/frozen/rq1_decision.json` with `phi_switching`, `lr_stat`, `p_value`, `df`, `sample_end`, `git_hash = paths.git_hash()`.
  - **Interactive checkpoint:** show the user the φ LR, p-value, the Wald p-value, the μ and φ estimates by regime and the decision. The user confirms (or overrides with a stated reason, which goes in the JSON as `"override"`). Then set `PHI_SWITCHING = True|False` in `config.py` by hand, and commit config and JSON together (Step 8).

- [ ] **Step 5: Constrained M3 dry run (spec §7.3 step 6)**

  Reload config (`autoreload`), assert `C.PHI_SWITCHING` equals the JSON, then `insample.fit_ladder(panel, C.RQ1_END, C.PHI_SWITCHING)`. Print and assert:
  - `LL(M3) ≥ LL(M2) − 1e-6` and `LL(M2) ≥ LL(M1) − 1e-6`;
  - the fixed coefficient is exactly `0.0` under `constrained` (`ladder["M3"].params["p[1->0].tvtp4"]`);
  - β in the deck convention (`ms.transition_coefs_deck`), its SE (`param_cov`), and `lr_test(M2, M3)["df"] == 1`.

  If `LL(M3) < LL(M2)` after the nesting start, print the best `search_reps=C.SEARCH_REPS_FIRST` re-run with `seed + 1`; if still violated, stop and ask the user (this is the R5 signal, to be settled at validation).

- [ ] **Step 6: Forecaster check on real data (spec §7.3 step 7)**

  For each of M1, M2, M3: `max |forecast.predicted_probs(model, params) − model.smooth(params).predicted_marginal_probabilities| < 1e-10`. Also print `forecast.one_step(...)` for M3 at origin 2018-12-28 (π_H, var_fc in %²; `forecast.highvol_threshold(panel)` as `thr`).

- [ ] **Step 7: Tables and checks cell**

  Save `rq1_tests.csv` (rows: `1v2_lr`, `1v2_bootstrap`, `bic_one`, `bic_two`, `mu_equal`, `phi_equal`; columns `stat, df, pvalue, note`; the note carries `B` and `n_degenerate`) and `rq1_params_M1.csv` (params with readable names, durations, weeks per regime). Checks (asserts): H has the higher σ² in every reported fit; `D.build_design(...).dates[-1] == 2018-12-28`; JSON exists and matches config; LL nesting; forecaster diff < 1e-10; `paths.check_budget()`.

- [ ] **Step 8: Restart and run all, then commit**

  ```bash
  git add notebooks/03_rq1_regimes.ipynb src/brent/config.py results/frozen/rq1_decision.json results/frozen/std_stats.csv results/tables/rq1_tests.csv results/tables/rq1_params_M1.csv results/tables/rq1_bootstrap_draws.csv results/figures/rq1_filtered_prob_2005_2018.png
  git commit -m "Run RQ1 regime tests and freeze phi (notebook 03)"
  ```

---

### Task 8: `benchmarks.py` part 1: HAR family, GARCH(1,1)-t, random walk

*Fri Nov 27 morning.*

**Files:**
- Create: `src/brent/benchmarks.py`, `tests/test_benchmarks.py`

**Interfaces:**
- Consumes: `design.har_design`, `config` (`RV_FLOOR_FRAC`, `TRAIN`, `X_CONTROLS`, `X_RV_COL`, `X_SAMPLE_START`, `IRAN_VAR`, `SAMPLE_START`).
- Produces (spec §8.7):
  - `harx_ols(panel, end, rv_col="rv", x_vars=(), start=C.SAMPLE_START)` **[plan call]**: returns the statsmodels `RegressionResults` of y on X from `har_design` (used by `harx_forecast`, `harx_insample` and the F-test in notebook 04)
  - `harx_forecast(panel, end, rv_col="rv", x_vars=(), start=C.SAMPLE_START) -> dict` with `{"var_fc", "floored", "params"}`. `params` is a Series named `b0, b_rv1, b_rv4, b_rv13, c_<x_var>…` (`c_gpr, c_vix, c_iran_gdelt`). Floor = `C.RV_FLOOR_FRAC` × min of `rv_col` over `TRAIN ∩ [start, end]`; `var_fc = max(x_next @ b, floor)`, `floored = x_next @ b < floor`.
  - `har_forecast(panel, end) -> dict` (= `harx_forecast(panel, end, "rv")`)
  - `harx_insample(panel, end, rv_col, x_vars, start, hac_lags=4) -> pd.DataFrame`: rows = coefficient names, columns `coef, se, t, p` (Newey-West, 4 lags); extra rows `adj_r2` and `n` (value in `coef`)
  - `garch_t_forecast(r_hist, start_params=None) -> dict` with `{"var_fc": h + mu², "mean_fc": mu, "params": Series(mu, omega, alpha, beta, nu), "converged": bool}`. `arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")`; warm start with `starting_values`; `var_fc = h(T+1) + μ̂²` from `forecast(horizon=1)`.
  - `rw_forecast() -> float` (0.0)

- [ ] **Step 1: Write the failing tests** (`tests/test_benchmarks.py`, part 1)

  ```python
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
      out = B.model_forecast("HARX_C", p, pd.Timestamp(END), {})
      assert np.isfinite(out["var_fc"]) and out["var_fc"] > 0

  def test_garch_t_forecast_second_moment(sim_panel):
      p = sim_panel(n=500); r = p.set_index("date")["r"]
      out = B.garch_t_forecast(r)
      assert out["converged"] and out["var_fc"] > out["mean_fc"] ** 2 and 2.0 < out["params"]["nu"] < 200
      warm = B.garch_t_forecast(r, start_params=out["params"])
      assert warm["var_fc"] == pytest.approx(out["var_fc"], rel=1e-3)

  def test_rw_is_zero():
      assert B.rw_forecast() == 0.0
  ```
  (`benchmarks.py` must call `design.har_design` through the module attribute, `D.har_design`, so the floor test can monkeypatch it; import it as `from brent import design as D`.)

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_benchmarks.py -q`
  Expected: FAIL (`brent.benchmarks` missing; `model_forecast` is added in Step 3 for the non-GARCH-X names).

- [ ] **Step 3: Implement the functions, plus `model_forecast(name, panel, end, state) -> dict`**

  `model_forecast` reads config **at call time**:
  - `"HAR"` → `har_forecast`; `"HAR_h"` → `harx_forecast(panel, end, C.X_RV_COL, (), C.X_SAMPLE_START)`; `"HARX_C"` → same with `x_vars=C.X_CONTROLS`; `"HARX_CI"` → `C.X_CONTROLS + (C.IRAN_VAR,)`. `iran_var` for robustness runs arrives through `state["iran_var"]`.
  - `"GARCH_t"` → `garch_t_forecast(r through end, state.get("GARCH_t"))`, storing the new params back into `state["GARCH_t"]`; `"RW"` → `{"mean_fc": 0.0}`.
  - `"GARCHX_C"`, `"GARCHX_CI"` are added in Task 9.
  - Return dict always has `var_fc` (or `mean_fc` for RW), `params` (Series), `converged`, `loglike`, `floored`, `seconds`.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_benchmarks.py -q`
  Expected: all PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/benchmarks.py tests/test_benchmarks.py
  git commit -m "Add HAR family, GARCH-t and random-walk benchmarks"
  ```

---

### Task 9: `benchmarks.py` part 2: GARCH-X, `garchx_lr`

*Fri Nov 27.*

**Files:**
- Modify: `src/brent/benchmarks.py`, `src/brent/insample.py`, `tests/test_benchmarks.py`, `tests/test_insample.py`

**Interfaces:**
- Consumes: `design.garchx_design`, `insample.chibar_pvalue`, `statsmodels.tools.numdiff.approx_hess3`.
- Produces (spec §8.7, §8.9):
  - `benchmarks.GarchXFit` dataclass (`params, loglike, converged, seconds, h, x_names`)
  - `garchx_loglike(theta: np.ndarray, r, X, h1: float) -> float`: `theta = [mu, omega, alpha, beta, gamma_1…gamma_k, nu]`; X may have 0 columns
  - `garchx_fit(r: pd.Series, X: pd.DataFrame, start_params: pd.Series | None = None) -> GarchXFit`
  - `garchx_forecast(fit, r, x_next) -> dict` (`var_fc = h_next + mu²`, `h_next`)
  - `garchx_simulate(params: pd.Series, X: pd.DataFrame, seed: int) -> pd.Series`
  - `model_forecast("GARCHX_C" | "GARCHX_CI", …)`: `garchx_design(panel, end, X_CONTROLS [+ iran_var])`, warm start from `state[name]`, params stored back into `state`
  - `insample.garchx_lr(panel, end, x_small: tuple[str, ...] | None, x_big: tuple[str, ...]) -> dict`: `x_small=None` means the GARCH-t on the same sample (zero X columns). Fits small, then big **started from small's params with zero γ for the added regressors** (so LL(big) ≥ LL(small)), on the same sample. Returns `{lr, k, pvalue_chibar, gammas, ses, ll_small, ll_big}`; `k = len(added gammas)`.

**Likelihood (the recursion the signature does not determine).** With ε(t) = r(t) − μ, standardized Student-t of ν degrees of freedom (unit variance, as in `arch`), and X row t already dated t−1 by `garchx_design`:
```text
h(1) = h1 (= np.var(r), ddof=0, passed in)
h(t) = omega + alpha*eps(t-1)^2 + beta*h(t-1) + gamma' X[t]            for t >= 2
ll   = sum_t [ lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(pi*(nu-2)) - 0.5*log h(t)
               - (nu+1)/2 * log(1 + eps(t)^2 / (h(t)*(nu-2))) ]
```
The recursion is linear in h, so `scipy.signal.lfilter` evaluates it without a Python loop (optional; a loop is fine if the timing stage shows it fast enough).

**Fit.** `scipy.optimize.minimize(-loglike, method="SLSQP")` with bounds ω ≥ 1e-8, α ∈ [0, 1], β ∈ [0, 1], γₖ ≥ 0, ν ∈ [2.05, 100], μ free, and the linear constraint α + β ≤ 0.999. Starts: `start_params` if given (warm); otherwise the nesting start = the zero-X GARCH-t solution (fit first, same function) with all γ = 0. If a warm start does not converge, retry from the nesting start. Keep the best LL. `converged` = SLSQP `success`. SEs are not stored on the fit; `garchx_se(fit, r, X)` **[plan call]** returns `approx_hess3` SEs, NaN where the Hessian is singular or a γ is at its bound.
- `garchx_simulate`: `h(1) = (ω + γ'X[0]) / (1 − α − β)`, innovations `rng.standard_t(ν) · sqrt((ν − 2)/ν)` from `np.random.default_rng(seed)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_benchmarks.py`)

  ```python
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
      b = B.model_forecast("GARCHX_CI", p, pd.Timestamp("2019-01-04"), state)
      assert np.isfinite(b["var_fc"])
  ```
  and in `tests/test_insample.py`:
  ```python
  def test_garchx_lr_chibar_and_nesting(sim_panel):
      p = sim_panel(n=1140)
      a = I.garchx_lr(p, "2018-12-28", None, ("gpr", "vix"))              # GARCHX_C vs GARCH-t, k = 3 (rv_hourly + gpr + vix)
      assert a["k"] == 3 and a["ll_big"] >= a["ll_small"] - 1e-6 and 0 <= a["pvalue_chibar"] <= 1
      b = I.garchx_lr(p, "2018-12-28", ("gpr", "vix"), ("gpr", "vix", "iran_gdelt"))
      assert b["k"] == 1 and b["ll_big"] >= b["ll_small"] - 1e-6
  ```
  (For `x_small=None` the "GARCH-t" has zero X columns and `x_big=("gpr","vix")` brings in `rv_hourly` too, so k = 3, as in spec §7.4 step 8. If `X_RV_COL` is `"rv"` in the fallback branch the count is unchanged.)

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_benchmarks.py tests/test_insample.py -q`
  Expected: FAIL (`garchx_*` not defined).

- [ ] **Step 3: Implement `GarchXFit`, `garchx_loglike`, `garchx_fit`, `garchx_se`, `garchx_forecast`, `garchx_simulate`, the `model_forecast` branches, and `insample.garchx_lr`**

  `garchx_fit` returns `h` as a Series indexed like `r`. Params Series order: `mu, omega, alpha, beta, gamma_<col>…, nu`. Do not add SE logic to the hot path.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_benchmarks.py tests/test_insample.py -q`
  Expected: all PASS. If the recovery test fails on tolerance, do not widen it: check the SLSQP bounds, the h(1) value and the t-density normalization first (the true-parameter LL check separates a likelihood bug from an optimizer problem).

- [ ] **Step 5: Time one fit**

  Run once in a scratch cell or `uv run python -c` on the real panel at origin 2018-12-28: `garchx_fit` for GARCHX_CI, cold and warm. Print seconds. Spec §7.5 expects ≈ 0.1–0.5 s warm; the number goes into notebook 05's timing stage anyway.

- [ ] **Step 6: Commit**

  ```bash
  git add src/brent/benchmarks.py src/brent/insample.py tests/test_benchmarks.py tests/test_insample.py
  git commit -m "Add Student-t GARCH-X with scipy likelihood, forecast, simulation and chi-bar LR"
  ```

---

### Task 10: `oos.py`: expanding-window loop, checkpoints, assembly, guards

*Fri Nov 27 (afternoon). Notebook 05's timing stage needs it.*

**Files:**
- Create: `src/brent/oos.py`, `tests/test_oos.py`

**Interfaces:**
- Consumes: `ms` (`fit_ms`, `nest_parent`, `nest_start`, `make_model`, `readable_to_sm`, `expected_durations`, `pick_best`), `design.build_design`, `forecast.one_step`, `forecast.highvol_threshold`, `benchmarks.model_forecast`, `paths` (`CHECKPOINTS, FORECASTS, FITS, FROZEN, PROCESSED`), `config`.
- Produces (spec §8.8, plus marked helpers):
  - `origins(start_target: str, end_target: str) -> list[pd.Timestamp]`: Fridays `start_target … end_target`, each minus 7 days
  - `origin_index(origin) -> int` **[plan call]**: `(origin − 2012-12-28).days // 7`; the seed of an origin is `C.SEED + origin_index(origin)`, in every run
  - `settings_hash(ms_specs, benchmarks, iran_var) -> str` **[plan call: public so notebook 05 can write it into `spec_freeze.json`]**: first 16 hex chars of the SHA-256 of the sorted-key JSON of: `SEARCH_REPS_FIRST, N_RESTARTS, RESTART_EVERY, SIGMA2_RATIO_MIN, VAR_FLOOR_FRAC, MIN_DURATION, M3_VARIANT, PHI_SWITCHING, iran_var`, each spec's `dataclasses.asdict`, `benchmarks`, `INTRADAY_ROLL_RULE`, `X_RV_COL`, `X_SAMPLE_START`, `HOURLY_SAMPLE_START`. It does **not** include the origin list, `SPEC_FROZEN` or `run_id`.
  - `sanity_flags(fit, ll_prev_on_current: float | None, ll_parent: float | None, ratio_min: float) -> list[str]` **[plan call]**: any of `"not_converged"`, `"ll_below_start"` (`fit.loglike < ll_prev_on_current − 1e-8`), `"sigma2_ratio"` (σ²_H/σ²_N < `ratio_min`), `"degenerate"`, `"nesting"` (`fit.loglike < ll_parent − 1e-6`)
  - `run_oos(panel, run_id, ms_specs, benchmarks=C.BENCHMARKS_MAIN, origins=None, iran_var=C.IRAN_VAR, force=False, max_origins=None) -> None` (`origins=None` means the full list `origins(C.VALID[0], C.FREEZE_DATE)`; `max_origins` stops after that many **new** origins, for tests and the timing stage)
  - `assemble(run_id) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: forecasts (spec §9.2 columns exactly: `date, origin, split, run, model, target, forecast`), params (wide: `origin, model, <readable names…>`), fitlog (spec §9.4 columns plus `restart_reason, ll_gain_from_restart, n_warnings` **[plan call]**); writes `results/forecasts/forecasts_<run>.csv`, `results/fits/params_<run>.csv`, `results/fits/fitlog_<run>.csv`
  - `load_forecasts(run_id, with_actuals=True) -> pd.DataFrame`: reads the forecasts file; with actuals joins from `paths.PROCESSED/weekly_panel.csv` on `date`: `actual_rv, actual_rv_hourly, actual_rv_gk, actual_r, actual_exceed` (`1{|r| > forecast.highvol_threshold(panel)}`)
  - `fake_forecasts(panel, models=("M1","M2","M3") + C.BENCHMARKS_MAIN) -> pd.DataFrame`: schema-correct dummy over all 723 targets, `run = "fake"`, `np.random.default_rng(C.SEED)`; MS models → `var, r, p_high, p_exceed`; GARCH_t, HAR, HAR_h, HARX_*, GARCHX_* → `var`; RW → `r`; `var` = `rv_hourly × lognormal(0, 0.3)`; the caller saves it (not committed)

**Per-origin procedure (the logic the signature leaves open).** For origin T with target T + 7 days, seed = `C.SEED + origin_index(T)`:
1. `ms_specs` processed in order of `len(tvtp_vars)` so parents come first. For each spec: `design = build_design(panel, spec, T)`; `start = state.get(spec.name)` (canonical params of the previous origin, or `None`); `parent = nest_parent(spec, ms_specs)`.
2. Warm fit: `fit_ms(design, spec, start, search_reps = 0 if start is not None else C.SEARCH_REPS_FIRST, n_restarts=0, seed, extra_starts = (nest_start(fits[parent.name], model, spec),) if start is None else ())`.
3. `ll_prev_on_current = model.loglike(start)` if `start` else `None`; `flags = sanity_flags(fit, ll_prev_on_current, fits[parent].loglike or None, C.SIGMA2_RATIO_MIN)`.
4. If `origin_index % C.RESTART_EVERY == 0` or `flags`: refit with `n_restarts=C.N_RESTARTS` (from `start` or the warm fit's params; add the nesting start when `"nesting" in flags` or `start is None`), keep `pick_best([warm, refit])`. Record `restart_reason` (`"scheduled"` or `"+".join(flags)`) and `ll_gain_from_restart`.
5. `state[spec.name] = fit.params`; `forecast.one_step(fit, design, thr)` with `thr = forecast.highvol_threshold(panel)` writes the four targets (`var, r, p_high, p_exceed`). A non-finite forecast raises `ValueError` naming model and origin (a missing GPR/VIX at the origin must stop the run, not write NaN).
6. Benchmarks: `benchmarks.model_forecast(name, panel, T, state)` for each name; `state` carries GARCH-t and GARCH-X warm starts; `state["iran_var"] = iran_var`.
7. Splits by target date: `validation` ≤ `C.VALID[1]`, `test` ≤ `C.TEST[1]`, else `holdout`.
8. Exceptions inside a single candidate fit are caught by `fit_model` (that candidate is dropped); if every candidate raises, the exception propagates (a bug, not a convergence failure). A non-converged or degenerate best candidate is **kept** with `converged=False` / `degenerate=True` and logged.
9. Progress: print one line every 50 origins (origin, cumulative seconds, number of non-converged fits so far).

**Checkpoints and resume.** After each origin append its rows to `results/checkpoints/<run_id>_forecasts.csv`, `_params.csv` (long: `origin, model, name, value`) and `_fitlog.csv`, in that order, each with a `settings_hash` column, flushing and `os.fsync`ing after each. The fitlog row marks the origin as complete. On start: read the checkpoints; if the files exist and **every** row's hash equals the current `settings_hash(...)`, drop forecast/params rows of origins that have no fitlog row (a crash mid-origin) and resume after the last completed origin, rebuilding `state` from the last completed origin's params rows (MS names mapped back with `readable_to_sm` on a model built from `build_design(panel, spec, last_origin)`; benchmark params as stored). Otherwise discard the checkpoints and start at the first origin. `force=True` deletes them first. CSV round-trips floats exactly, so a resumed run reproduces an uninterrupted one.

**Directories.** `run_oos` and `assemble` create `paths.CHECKPOINTS`, `paths.FORECASTS` and `paths.FITS` with `mkdir(parents=True, exist_ok=True)` before writing (a fresh clone has no `results/checkpoints/`, which is git-ignored).

**Guards (spec §8.8, §7.5).**
- `C.PHI_SWITCHING is None` → `ValueError`.
- Any origin whose target is after `C.VALID[1]` (a test or holdout target) and `not C.SPEC_FROZEN` → `RuntimeError` (**SPEC_FROZEN guard**).
- If `C.SPEC_FROZEN` and `run_id == "main"`: `results/frozen/spec_freeze.json` must exist and its `settings_hash` must equal the current one, else `RuntimeError` **[plan call: this is how "a changed setting cannot slip into the full run" is enforced]**. Other `run_id`s use the frozen config but have their own hash.

- [ ] **Step 1: Write the failing tests** (`tests/test_oos.py`)

  ```python
  import json, numpy as np, pandas as pd, pytest
  from dataclasses import replace
  from brent import oos, ms, config as C, paths

  @pytest.fixture
  def env(tmp_path, monkeypatch, sim_panel):
      for name in ("CHECKPOINTS", "FORECASTS", "FITS", "FROZEN", "PROCESSED"):
          d = tmp_path / name.lower(); d.mkdir(); monkeypatch.setattr(paths, name, d)
      monkeypatch.setattr(C, "PHI_SWITCHING", False)
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
      assert h == oos.settings_hash(specs, bm, C.IRAN_VAR)
      monkeypatch.setattr(C, "N_RESTARTS", 2)
      assert h != oos.settings_hash(specs, bm, C.IRAN_VAR)
      monkeypatch.setattr(C, "N_RESTARTS", 1)
      assert h != oos.settings_hash(specs, bm, "aigpr_iran")
      assert h != oos.settings_hash(specs, bm[:-1], C.IRAN_VAR)

  def test_sanity_flags():
      ok = lambda **k: type("F", (), dict(converged=True, degenerate=False, loglike=-100.0,
                                          params=pd.Series({"sigma2[0]": 1.0, "sigma2[1]": 4.0}), **k))()
      assert oos.sanity_flags(ok(), -100.5, -101.0, 1.5) == []
      assert "ll_below_start" in oos.sanity_flags(ok(), -99.0, None, 1.5)
      assert "nesting" in oos.sanity_flags(ok(), None, -90.0, 1.5)
      assert "sigma2_ratio" in oos.sanity_flags(ok(), None, None, 5.0)
      bad = ok(); bad.converged = False; bad.degenerate = True
      assert {"not_converged", "degenerate"} <= set(oos.sanity_flags(bad, None, None, 1.5))

  def test_one_var_row_per_model_per_origin_and_schema(env):
      p, specs, bm, _ = env
      oos.run_oos(p, "t1", specs, benchmarks=bm, origins=ORIG)
      f, par, log = oos.assemble("t1")
      assert list(f.columns) == ["date", "origin", "split", "run", "model", "target", "forecast"]
      assert (f["date"] - f["origin"] == pd.Timedelta(days=7)).all() and (f["split"] == "validation").all()
      v = f[f["target"] == "var"]
      for m in ("M1", "M2", "HAR", "HARX_CI", "GARCH_t", "GARCHX_C"):
          assert (v[v["model"] == m].groupby("origin").size() == 1).all() and len(v[v["model"] == m]) == 3
      assert len(f[(f["model"] == "RW") & (f["target"] == "r")]) == 3 and (f["forecast"] > 0)[f["target"] == "var"].all()
      assert f.loc[f["target"] == "p_high", "forecast"].between(0, 1).all()
      assert set(log["settings_hash"]) == {oos.settings_hash(specs, bm, C.IRAN_VAR)} and len(log) == 3 * 7

  def test_interrupted_run_equals_uninterrupted(env):
      p, specs, bm, _ = env
      oos.run_oos(p, "a", specs, benchmarks=bm, origins=ORIG)
      oos.run_oos(p, "b", specs, benchmarks=bm, origins=ORIG, max_origins=1)
      oos.run_oos(p, "b", specs, benchmarks=bm, origins=ORIG)               # resumes after origin 1
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
      assert (~log[log["model"] == "M2"]["converged"].astype(bool)).all() and len(f[(f["model"] == "M2") & (f["target"] == "var")]) == 3

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

  def test_output_directories_are_created(env, tmp_path, monkeypatch):
      p, specs, bm, _ = env
      for name in ("CHECKPOINTS", "FORECASTS", "FITS"):
          fresh = tmp_path / ("fresh_" + name.lower()); monkeypatch.setattr(paths, name, fresh)
      oos.run_oos(p, "dirs", specs[:1], benchmarks=("RW",), origins=ORIG[:1]); oos.assemble("dirs")
      assert (paths.CHECKPOINTS / "dirs_fitlog.csv").exists() and (paths.FORECASTS / "forecasts_dirs.csv").exists()

  def test_fake_forecasts_schema_and_load_with_actuals(env):
      p, specs, bm, _ = env
      fk = oos.fake_forecasts(p)
      assert list(fk.columns) == ["date", "origin", "split", "run", "model", "target", "forecast"]
      assert fk["date"].nunique() == 723 and set(fk["split"]) == {"validation", "test", "holdout"} and (fk["run"] == "fake").all()
      assert set(fk.loc[fk["model"] == "RW", "target"]) == {"r"} and set(fk.loc[fk["model"] == "GARCHX_CI", "target"]) == {"var"}
      assert set(fk.loc[fk["model"] == "M3", "target"]) == {"var", "r", "p_high", "p_exceed"}
      fk.to_csv(paths.FORECASTS / "forecasts_fake.csv", index=False)
      lf = oos.load_forecasts("fake")
      row = lf.iloc[0]; P = p.set_index("date")
      assert row["actual_rv_hourly"] == P.loc[row["date"], "rv_hourly"] and row["actual_r"] == P.loc[row["date"], "r"]
      assert set(lf["actual_exceed"].dropna().unique()) <= {0, 1}
  ```
  (Every model, including `RW`, writes one fitlog row per origin: 2 MS + 5 benchmarks = 7. The checkpoint's `origin` column is ISO-formatted.)

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_oos.py -q`
  Expected: FAIL (`brent.oos` missing).

- [ ] **Step 3: Implement `oos.py`**

  Per the procedure, resume and guard rules above. Call `ms.fit_ms`, `benchmarks.model_forecast` and the other modules' functions through the module attribute (not `from … import`), so the tests can monkeypatch them. Params rows for MS models use the readable names; the long checkpoint stores the readable name, and `assemble` pivots to wide. Benchmark rows write their `params` Series (HAR family: `b0, b_rv1, b_rv4, b_rv13, c_*`; GARCH-X: `mu, omega, alpha, beta, gamma_*, nu`); fitlog rows for them fill `loglike, converged, seconds, har_floored` and leave the MS columns empty.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_oos.py -q`
  Expected: all PASS (about 1–2 min).

- [ ] **Step 5: Run the whole suite**

  Run: `uv run pytest -q`
  Expected: all PASS, including Plan 1's tests and `tests/test_budget.py`.

- [ ] **Step 6: Commit**

  ```bash
  git add src/brent/oos.py tests/test_oos.py
  git commit -m "Add expanding-window OOS loop with checkpoints, resume, guards and fake forecasts"
  ```

---

### Task 11: Notebook 05, stages `timing` and `validation`, then the spec freeze

*Fri Nov 27 evening → Sat Nov 28 morning. Validation runs ≈ 0.5–1 h unattended; this is the critical path.*

**Files:**
- Create: `notebooks/05_oos_forecasts.ipynb` (stages are cells selected by `STAGE`), `results/frozen/spec_freeze.json`
- Modify: `src/brent/config.py` (by hand: `N_RESTARTS`, `RESTART_EVERY`, `SIGMA2_RATIO_MIN`, `VAR_FLOOR_FRAC`, `MIN_DURATION`, `M3_VARIANT`, `SEARCH_REPS_*`, finally `SPEC_FROZEN = True`)

**Interfaces:**
- Consumes: `oos.run_oos`, `oos.origins`, `oos.settings_hash`, `oos.assemble`; `ms.make_specs`; `config`; `paths`.
- Produces: `results/frozen/spec_freeze.json` with keys `phi_switching, SEARCH_REPS_FIRST, N_RESTARTS, RESTART_EVERY, SIGMA2_RATIO_MIN, VAR_FLOOR_FRAC, MIN_DURATION, M3_VARIANT, INTRADAY_ROLL_RULE, INTRADAY_REDISTRIBUTE, HOURLY_SAMPLE_START, MAIN_RV, X_RV_COL, X_SAMPLE_START, BENCHMARKS_MAIN, settings_hash, git_hash`; `config.SPEC_FROZEN = True`; validation checkpoints for run `main` (313 origins).

First cell: `STAGE = os.environ.get("BRENT_STAGE", "timing")` **[plan call: lets the overnight run be started from a terminal without editing the notebook]**, `FORCE_OOS = False`; specs `S = ms.make_specs(C.PHI_SWITCHING, variant=C.M3_VARIANT)`; `MS = [S["M1"], S["M2"], S["M3"]]`; `panel = pd.read_csv(…, parse_dates=["date"])`. A guard cell asserts `C.PHI_SWITCHING` equals `rq1_decision.json`.

- [ ] **Step 1: Stage `timing`** (spec §7.5 step 1)

  `oos.run_oos(panel, "timing", MS, origins=oos.origins("2013-01-04", "2013-03-08"), force=True)`: the first 10 validation origins (2012-12-28 → 2013-03-01). It never touches a test-window origin and is never resumed.
  - Print seconds per fit by model from `results/checkpoints/timing_fitlog.csv`: the cold first fit (origin 0) separately from the warm fits (origins 1–9), and the benchmarks (GARCH-t, HAR family, GARCH-X cold vs. warm).
  - **Restart cost:** in a micro-benchmark cell, on origin 2013-03-01, time `ms.fit_ms` for each MS model with `n_restarts` = 0, 1, 3, 5 (warm start from the stored params); print seconds per restart.
  - **Runtime table** for the candidate settings `(N_RESTARTS, RESTART_EVERY)` ∈ {(2, 13), (3, 13), (3, 26), (5, 13)}:
    `total_s ≈ 723 × (Σ warm s of all models) + (723/RESTART_EVERY + 0.10 × 723) × N_RESTARTS × Σ restart s per MS model`
    (the 10% term stands for sanity-flag restarts; replace it by the fraction of timing-run origins whose `restart_reason` is not `"scheduled"`, if the timing run shows any). Print the estimate in hours.

- [ ] **Step 2: Interactive checkpoint 1: restart settings**

  Show the user the timing table and the estimates. Target: the full run ≈ 2 h (spec §7.5). The user chooses `N_RESTARTS` and `RESTART_EVERY`.
  - **Branch:** if even `(1, 26)` estimates > 3 h, or GARCH-X cold fits > 2 s, stop and ask the user (options: drop GARCH-X restarts, run overnight anyway, trim `BENCHMARKS_MAIN`).
  - Set the two values in `config.py` by hand (provisional until Step 5), commit config only: `git commit -m "Set restart schedule from the timing stage"`.

- [ ] **Step 3: Stage `validation`** (spec §7.5 step 2)

  `oos.run_oos(panel, "main", MS, origins=oos.origins(C.VALID[0], C.VALID[1]))`: origins 2012-12-28 → 2018-12-21 (313 targets, 2013-01-04 → 2018-12-28), all MS models and `C.BENCHMARKS_MAIN`. The call resumes if interrupted; print the progress line every 50 origins. Start it Fri evening. Never look at 2019+.

- [ ] **Step 4: Review on validation only (spec §7.5 step 2)**

  From `results/checkpoints/main_fitlog.csv` print, per model:
  - convergence rate; `refit_from_scratch` rate; `relabeled` rate; `degenerate` count;
  - restart weeks, the share with `restart_reason` ≠ `"scheduled"`, and the share of restarts with `ll_gain_from_restart > 1e-6`;
  - distribution (min, 5th percentile, median) of σ²_H/σ²_N, of `dur_N`, `dur_H`, and of min σ² / var(r), to judge `SIGMA2_RATIO_MIN`, `MIN_DURATION`, `VAR_FLOOR_FRAC`;
  - **constrained-M3 stability:** the share of validation weeks where M3 is not converged **or** `LL(M3) < LL(M2) − 1e-6` (`nesting_ok == False`);
  - GARCH-X: non-converged weeks; HAR family: `har_floored` weeks. At origin 2012-12-28 run `insample.garchx_lr` (GARCH-t → GARCHX_C → GARCHX_CI) and print LL ordering.

  **Variant rule (spec §7.5, [our call]).** If the constrained-M3 share is **> 5%** of weeks, switch: set `M3_VARIANT = "iran_both_logits"` in config, rebuild `S`, and rerun this stage (the hash changes, so the checkpoint is discarded and validation reruns, ≈ +0.5–1 h); review again under the new variant. Otherwise keep `"constrained"`.
  The extra models have nothing to freeze.

- [ ] **Step 5: Interactive checkpoint 2: freeze the specification**

  Show the user the review tables and the proposed values of `SEARCH_REPS_FIRST, N_RESTARTS, RESTART_EVERY, SIGMA2_RATIO_MIN, VAR_FLOOR_FRAC, MIN_DURATION, M3_VARIANT`. State the consequence: any change after validation makes the settings hash differ, so `full` discards the validation checkpoint and reruns from 2012-12-28 (≈ +40% runtime). Agree the values.

- [ ] **Step 6: Write the freeze and commit**

  Edit `config.py` by hand with the agreed values and `SPEC_FROZEN = True`. Reload (`autoreload`), then:
  - `h = oos.settings_hash(MS, C.BENCHMARKS_MAIN, C.IRAN_VAR)`; write `results/frozen/spec_freeze.json` with the keys above and `git_hash = paths.git_hash()`;
  - read the hash column of `main_fitlog.csv` and print `"full will RESUME"` if it equals `h`, else `"full will RERUN from 2012-12-28"`.

  ```bash
  git add src/brent/config.py results/frozen/spec_freeze.json notebooks/05_oos_forecasts.ipynb
  git commit -m "Freeze model specification after the validation stage"
  ```
  **Slip rule (spec §11):** if the spec is not frozen by Sun Nov 29, tell the user: OOS robustness runs are dropped (robustness becomes in-sample only) and the conditional GW test is dropped.

---

### Task 12: Notebook 05, stage `full`, assembly and checks

*Sat Nov 28 evening start → Sun Nov 29 checks. The run takes ≈ 1–2 h (≈ 2–3 h if validation reran).*

**Files:**
- Modify: `notebooks/05_oos_forecasts.ipynb`, `README.md`
- Create: `results/forecasts/forecasts_main.csv`, `results/fits/params_main.csv`, `results/fits/fitlog_main.csv`, `results/figures/oos_runtime_and_convergence.png`

**Interfaces:**
- Consumes: Task 10 and Task 11 outputs.
- Produces (Plan 3 reads these exactly as spec §9.2, §9.4, §9.5 define them): `forecasts_main.csv` (723 targets × 20 rows ≈ 14.5k rows: `date, origin, split, run, model, target, forecast`), `params_main.csv`, `fitlog_main.csv`, `spec_freeze.json`, `oos.load_forecasts("main")`.

- [ ] **Step 1: Stage `full`**

  Guard cell: `C.SPEC_FROZEN`, and `spec_freeze.json["settings_hash"] == oos.settings_hash(MS, C.BENCHMARKS_MAIN, C.IRAN_VAR)`; otherwise stop (`run_oos` raises as well). Then:
  - skip if `paths.FORECASTS / "forecasts_main.csv"` exists and not `FORCE_OOS`;
  - else `oos.run_oos(panel, "main", MS)` (full origin list, 2012-12-28 → 2026-10-30; resumes from validation only if the hash matches) followed by `oos.assemble("main")`.

  Test and holdout come out of the same pass; splits are assigned by target date. The M3 variant never changes inside the run.

- [ ] **Step 2: Interactive checkpoint 3: start the overnight run**

  Confirm with the user that the spec is frozen and committed, then start from a terminal so the notebook session is free:
  ```bash
  BRENT_STAGE=full uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_oos_forecasts.ipynb
  ```
  (or run the cell in VS Code and leave the machine awake). If interrupted, rerun the same command: the checkpoint resumes. Tell the user the expected finish and that notebook 04 (Task 13) can be done meanwhile.

- [ ] **Step 3: Checks cell** (spec §7.5 Checks, §10 "Verification beyond tests")

  Each is an `assert`, printing the offending rows on failure:
  - 723 target weeks per model and target (M1–M3: `var, r, p_high, p_exceed`; GARCH_t, HAR, HAR_h, HARX_*, GARCHX_*: `var`; RW: `r`); split counts 313 / 365 / 45;
  - `origin == date − 7 days`; no target ≤ its origin; all `var` > 0; `p_high` ∈ [0, 1];
  - every row of `fitlog_main.csv` and every checkpoint row carries the frozen `settings_hash`;
  - print: fitlog convergence rate per model, `refit_from_scratch` count, non-converged M3 weeks, degenerate fits, GARCH-X non-converged weeks, HAR-family floored weeks;
  - **nesting per week:** count weeks with `LL(M3) < LL(M2) − 1e-6` and `LL(M2) < LL(M1) − 1e-6` (from `nesting_ok`); print the GARCHX_CI vs. GARCHX_C LL ordering counts (warm-started, so printed rather than asserted) and assert the first-origin ordering GARCHX_CI ≥ GARCHX_C ≥ GARCH-t from `insample.garchx_lr`;
  - **resume identity:** `run_oos(panel, "resume_ref", MS, benchmarks=…, origins=<first 8 validation origins>, force=True)` against `"resume_chk"` run with `max_origins=4` then again to the end; `assemble` both, assert forecasts and params equal (ignore `run`, `seconds`), then delete the `forecasts_resume_*`, `params_resume_*`, `fitlog_resume_*` files and checkpoints;
  - `oos.load_forecasts("main")` has no NaN `actual_rv_hourly` for 2013–2026; `paths.check_budget()`.

- [ ] **Step 4: Runtime and convergence figure**

  `oos_runtime_and_convergence.png`: fit seconds by origin per MS model (with restart weeks marked) and a rolling convergence/degenerate indicator, saved with `plots.save_fig`.

- [ ] **Step 5: Document the run in README**

  A short section: how to run stages (`BRENT_STAGE`, `FORCE_OOS`), that the checkpoint is resumable, that the full run takes ≈ 1–2 h and `forecasts_main.csv` is committed, and that notebook 05 loads committed outputs when `FORCE_OOS` is False.

- [ ] **Step 6: Restart and run all with `FORCE_OOS = False`, then commit**

  Expected: the `full` stage prints "skip" and the checks pass.
  ```bash
  git add notebooks/05_oos_forecasts.ipynb README.md results/forecasts/forecasts_main.csv results/fits/params_main.csv results/fits/fitlog_main.csv results/figures/oos_runtime_and_convergence.png
  git commit -m "Run the expanding-window OOS forecasts and assemble outputs (notebook 05)"
  ```
  **Slip rule (spec §11):** if the full run is not finished by Mon Nov 30, tell the user; Plan 3 then works in minimum-viable mode.

---

### Task 13: Notebook 04 — RQ2 in sample, M3-tone, extra models in sample

*Sat Nov 28 → Sun Nov 29, during the full run. Run it **after** Task 11's freeze so `M3_VARIANT` is final.*

**Files:**
- Create: `notebooks/04_rq2_explain.ipynb`; `results/tables/{rq2_coefficients,rq2_lr_tests,rq2_fit_summary,rq1_tests_2005_2025,rq2_m3tone,harx_insample,harx_ftest,garchx_insample}.csv` and the `_2026` supplements of the first four; `results/fits/insample_filtered_2005_2026.csv`; `results/figures/{rq2_filtered_prob,rq2_transition_probs}.png`
- Modify: none

**Interfaces:**
- Consumes: `insample` (`fit_ladder, fit_tone, lr_test, wald_equal, coef_table, fit_summary, equality_tests, bootstrap_lr_1v2, garchx_lr`), `benchmarks` (`harx_insample, harx_ols, garchx_fit, garchx_se`), `forecast.filter_fixed`, `ms.expected_durations`, `config`.
- Produces: the tables above, with the column schemas stated in Steps 2 and 7 (Plan 3's notebooks 06–07 read `insample_filtered_2005_2026.csv` with columns `date, model, xi_H`).

First cell asserts: `C.PHI_SWITCHING` equals `rq1_decision.json`; `spec_freeze.json` exists and `C.M3_VARIANT == freeze["M3_VARIANT"]` (if the freeze file is missing the notebook stops with "run Task 11's freeze first"). Flag `RUN_BOOT_2025 = False`.

- [ ] **Step 1: Fit the ladder on 2005–2025 (spec §7.4 steps 1–3)**

  `L = insample.fit_ladder(panel, C.RQ2_END, C.PHI_SWITCHING, n_restarts=C.N_RESTARTS)`; `T = insample.fit_tone(panel, C.RQ2_END, C.PHI_SWITCHING, L, n_restarts=C.N_RESTARTS, variants=("M3tone",))` (only M3tone here; `M3toneL`/`M3gold` belong to notebook 08). Print and flag any `degenerate` or `not converged` fit (stop and ask the user if M3 or M3tone is affected).

- [ ] **Step 2: Coefficients, LR tests, fit summary**

  - `rq2_coefficients.csv` = `coef_table` of M2, M3, M3tone stacked (deck sign convention, SE from `param_cov`, z, p, `fixed`).
  - `rq2_lr_tests.csv` (columns `test, model_restricted, model_full, lr_raw, stat, df, pvalue`, i.e. the `lr_test` keys plus labels; Plan 3 reads it): rows `M2_vs_M1` (df 6), `M3_vs_M2`, `M3tone_vs_M2` via `lr_test`; the df column is **computed** (`constrained`: 1 and 2; `iran_both_logits`: 2 and 4), never typed. Add a separate row `beta_iran_gdelt_z` with β, SE, z, p. Markdown: the joint LR and β's z-test coincide only when df = 1.
  - `rq2_fit_summary.csv` = `fit_summary` for M1, M2, M3, M3tone (LL, k, AIC, BIC, durations).

- [ ] **Step 3: RQ1 reported fit on 2005–2025 (spec §7.4 step 4)**

  Same tests as notebook 03 on `C.RQ2_END` (M1 with φ switching, `equality_tests`, plain 1-vs-2 LR and BIC, durations, weeks per regime) → `rq1_tests_2005_2025.csv`. The bootstrap runs only if `RUN_BOOT_2025`. **φ stays frozen**: print what the 2005–2025 φ test says and assert `C.PHI_SWITCHING` is unchanged; nothing here feeds back.

- [ ] **Step 4: Supplement 2005–2026 (spec §7.4 step 5)**

  Refit the ladder (and M3tone) on `C.FREEZE_DATE`; save `rq2_coefficients_2026.csv`, `rq2_lr_tests_2026.csv`, `rq2_fit_summary_2026.csv`, `rq1_tests_2026.csv`. Label each markdown cell "supplement: includes the 2026 holdout".

- [ ] **Step 5: Filtered probabilities and figures (spec §7.4 step 6)**

  For M1, M2, M3: build the 2005–2026 design with `build_design(panel, spec, C.FREEZE_DATE)`, `forecast.filter_fixed(make_model(design, spec), params_from_2005_2025_fit)` → `insample_filtered_2005_2026.csv` (`date, model, xi_H`; 1,139 dates per model). `rq2_filtered_prob.png`: filtered (not smoothed) P(H) for M1/M2/M3 over 2005–2026 with the nine event weeks marked and a dashed line at 2026-01-01 (`ax.axvline`; no new `plots.py` helper). `rq2_transition_probs.png`: fitted P(N→H)(t) = `model.regime_transition_matrix(params, exog_tvtp=Z)[1, 0, :]` for M2 vs. M3 on 2005–2025.

- [ ] **Step 6: M3-tone (spec §7.4 step 7)**

  `rq2_m3tone.csv`: β_conflict, β_coop (deck sign), SEs, z, p; LR vs. M2 (`lr_test`, df computed); `wald_equal(T["M3tone"], "iran_conflict", "iran_coop")`; LL and BIC of M3tone vs. M3 (non-nested, descriptive). Markdown, in the spec's words: the Wald test asks whether the tone of coverage matters beyond its amount; equal coefficients mean only the volume of Iran news matters; a positive β_coop is not a contradiction (the 2015 nuclear deal moved supply expectations). Print the correlation of `iran_conflict` and `iran_coop` (R20): if |ρ| > 0.9 say that the individual β's are imprecise and the Wald test is the main statistic. OOS for M3tone is not run here.

- [ ] **Step 7: Extra linear models in sample (spec §7.4 step 8)**

  Sample `C.X_SAMPLE_START → C.RQ2_END`, columns from config (branch: if `"HAR_h"` is not in `C.BENCHMARKS_MAIN` — the no-intraday fallback — skip `HAR_h`; the rest run on `C.X_RV_COL`):
  - `harx_insample` for `HAR_h`, `HARX_C`, `HARX_CI` stacked with a `model` column (coefficients, Newey-West SEs with 4 lags, adjusted R²); the F-tests from `harx_ols(...).compare_f_test(...)` → `harx_insample.csv` (columns `model, term, coef, se, t, p`, with `adj_r2` and `n` as terms) and `harx_ftest.csv` (columns `test, F, df_num, df_resid, n, pvalue`; rows `HARX_CI_vs_HARX_C` and `HARX_C_vs_HAR_h`, plain nested F-tests from `compare_f_test`).
  - `garchx_lr(panel, C.RQ2_END, None, X_CONTROLS)` (GARCH-t → GARCHX_C, chi-bar-squared k = 3) and `garchx_lr(..., X_CONTROLS, X_CONTROLS + (IRAN_VAR,))` (k = 1, the 50:50 mixture) → `garchx_insample.csv` (γ's, SEs from `garchx_se`, NaN where γ̂ = 0 with a note, LL, LR, p-values). The markdown says that SEs are unreliable at the boundary and that the models' status (benchmark, replacement, appendix) is still open.

- [ ] **Step 8: Checks cell (spec §7.4 Checks)**

  - Every restricted coefficient is exactly `0.0` under `constrained` (`L["M3"]`, `T["M3tone"]`);
  - `LL(M1) ≤ LL(M2) ≤ LL(M3)` and `LL(M2) ≤ LL(M3tone)` within 1e-6; `LL(GARCH-t) ≤ LL(GARCHX_C) ≤ LL(GARCHX_CI)` within 1e-6;
  - σ²_H > σ²_N in every fit (relabel applied); no degenerate fit; `C.M3_VARIANT` equals the freeze file; `insample_filtered_2005_2026.csv` has 3 × 1,139 rows (estimation starts 2005-01-14, spec §8.5); `paths.check_budget()`.

- [ ] **Step 9: Restart and run all, then commit**

  ```bash
  git add notebooks/04_rq2_explain.ipynb results/tables results/fits/insample_filtered_2005_2026.csv results/figures/rq2_filtered_prob.png results/figures/rq2_transition_probs.png
  git commit -m "Run RQ2 in-sample tests, M3-tone and extra models in sample (notebook 04)"
  ```

---

## Done when

- `uv run pytest` passes (Plan 1 and Plan 2 tests).
- Notebooks 03, 04, 05 restart-and-run-all cleanly. 05 loads the committed forecasts (`FORCE_OOS = False`); 03 and 04 load saved bootstrap draws where their flags say so.
- Committed: `results/frozen/{rq1_decision,spec_freeze,std_stats}.csv|json`, `results/forecasts/forecasts_main.csv`, `results/fits/{params,fitlog}_main.csv`, `insample_filtered_2005_2026.csv`, the tables and figures named in Tasks 7, 11–13.
- `config.py` holds `PHI_SWITCHING`, the frozen search, restart and degeneracy settings, `M3_VARIANT` and `SPEC_FROZEN = True`.
- `check_budget()` passes.
- The user has seen and agreed: the φ decision (Task 7), the restart schedule from the timing table (Task 11 step 2), the validation review and the frozen values (Task 11 step 5), and the start of the overnight run (Task 12 step 2).

**What Plan 3 consumes (names exactly as built):**
- Forecasts and fits: `oos.load_forecasts("main")`, `oos.fake_forecasts`, `oos.run_oos` and `oos.assemble` for robustness runs (origin list from `oos.origins("2019-01-04", C.FREEZE_DATE)`, run ids per spec §9.2), `oos.settings_hash`.
- Modelling building blocks for notebook 08: `ms.make_specs`, `ms.with_iran`, `design.build_design`, `insample.fit_ladder`, `fit_tone` (incl. `M3toneL`, `M3gold`), `circular_shift`, `lr_test`, `coef_table`, `garchx_lr`, `benchmarks.harx_ols`, `harx_forecast`, `garchx_fit`.
- Shared helper: `forecast.highvol_threshold(panel)` (to be re-exported by `evaluate.highvol_threshold`).
- **Not built here, left to Plan 3:** `insample.placebo_lr` and `insample.placebo_harx` (spec §8.9; `placebo_harx` needs `evaluate.qlike`); the E1 branches (`kind="mshar"`) in `design.build_design`, `ms.fit_model` and `forecast.one_step`, which currently raise `NotImplementedError`.
