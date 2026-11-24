# Plan 3 — Evaluation, Events and Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the frozen out-of-sample forecasts (Plan 2) into every table, figure and number the report cites: forecast evaluation per RV measure (notebook 06), event case studies and deck fact checks (07), and the robustness checks #1–#9 plus optional extension E1 (08). Finish with the analysis freeze on Thu Dec 3 and the hand-off package on Fri Dec 4.

**Architecture:** Three layers.
- Pure, unit-tested statistics in `evaluate.py` (losses, DM-HLN, Clark-West, Giacomini-White, Mincer-Zarnowitz with HAC, Pesaran-Timmermann, Brier, Ljung-Box, validation-only model selection). Test functions take plain arrays; notebooks pick the `actual_<measure>` column.
- Presentation helpers in `plots.py` (palette, event shading, event windows, cumulative loss, reliability) and data builders in a new `events.py` (event table, fact checks).
- Robustness orchestration in a new `robustness.py` that composes Plan 2's `ms`, `design`, `insample`, `benchmarks` and `oos.run_oos`. A few small additive functions go into Plan 2's modules (Tasks 8, 9b, 14a). Existing Plan 2 signatures never change, and Plan 2 functions are consumed by name (`ms.nest_start`, `ms.constraint_info`, `insample.coef_table`, `insample.lr_test`, `forecast.highvol_threshold`).

Notebooks 06–08 hold the steps, tables and interpretation. Every robustness section has a `FORCE_<check>` flag and saves its own outputs, so a half-finished notebook 08 still yields finished checks.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, statsmodels ≥ 0.15 (`diebold_mariano_test`, `acorr_ljungbox`, OLS with HAC), matplotlib, pytest. `joblib` only if the OOS placebo needs parallel runs (Task 11, `uv add joblib`).

**Spec:** `docs/superpowers/specs/2026-11-20-brent-iran-implementation-design.md` (approved 2026-11-20, updated 2026-11-23). Section references (§N) point to it. Plan 1 (`2026-11-20-plan1-foundation-and-data.md`) built the scaffold, data, `weekly.py` and a minimal `plots.py` (`setup_style`, `save_fig`). Plan 2 (written in parallel) builds `design`, `ms`, `forecast`, `insample`, `benchmarks`, `oos` and notebooks 03–05. Plan 3 treats their spec §8 signatures and §9 schemas as existing.

**Scope:** Spec §7.6, §7.7, §7.8, §7.9 (rows for 06–08), §8.10, §8.11, the `test_evaluate.py` row of §10, §11 (Mon Nov 30 → Fri Dec 4, slip rules, minimum viable version, cut-first list), §13 R18, §14 items 4–9, 14, 20. **Dates:** Mon Nov 30 → Fri Dec 4. `evaluate` and `plots` may be started earlier against `oos.fake_forecasts` (spec §7.6) while the OOS run is unattended.

## Global Constraints

- Python 3.12; dependencies only those of spec §5.2. No new dependency except `joblib` if Task 14 needs it.
- Splits by **target date**: validation 2013-01-04 → 2018-12-28 (313 weeks), test 2019-01-04 → 2025-12-26 (365), holdout 2026-01-02 → 2026-11-06 (45). Test and holdout are always reported **separately**, never pooled. Validation is an appendix table.
- Test and holdout metrics are computed only when `config.SPEC_FROZEN` is True. Notebook 06 may run on `oos.fake_forecasts` before that (`DEV = True`): every dev output is prefixed `dev_`, git-ignored and never reported (Task 5).
- RV measures: `config.RV_MEASURES = ("rv_hourly", "rv", "rv_gk")`; main = `config.MAIN_RV = "rv_hourly"`. Every volatility table has a `measure` column. One variance forecast per model (`target = "var"`) is scored against `actual_<measure>`. No-intraday fallback (spec §11): `RV_MEASURES = ("rv",)`, `MAIN_RV = "rv"`; every loop reads these constants, so nothing else changes. Pairs naming a model that is not in the forecasts are skipped and listed.
- QLIKE = RV/f − ln(RV/f) − 1. DM loss: QLIKE primary, squared error secondary; `diebold_mariano_test(lags=0, harvey_adj=True, horizon=1)`. MZ: Newey-West, 4 lags. High-vol event: |r| > 90th percentile of |r| on TRAIN. π_H "hit" = π_H > 0.5.
- GW instrument: `1{iran_gdelt(origin) > 75th percentile of iran_gdelt on 2013–2018}`, dated at the **origin**. Share of 1s among test-window origins must lie in [10%, 90%]; otherwise stop and report (spec §14 item 6).
- Headline = M3 vs M2 on `rv_hourly`, test window. All other comparisons are secondary: tables carry `family` ∈ {`main`, `benchmark`, `extra`} (robustness models: `robustness`), no p-value is adjusted, and the notebook states how many tests were run (spec §7.6, R18).
- "Best linear model" is chosen on **validation only** (2013–2018) among `config.LINEAR_MODELS`.
- 2019–2026 never chooses anything. OOS robustness runs end at target 2025-12-26 and never touch the 2026 holdout **[plan call]**. Placebo shifts wrap inside 2005–2025 only (spec §7.8).
- Every MS fit in notebook 08 passes `n_restarts=C.N_RESTARTS` and the notebook 04 seed (`fit_ladder`, `fit_tone`, `fit_chain`, placebo fits), so identity checks against notebook 04 hold exactly.
- Cross-plan contract with Plan 2: notebook 04 writes `results/tables/rq2_lr_tests.csv` (`test, model_restricted, model_full, lr_raw, stat, df, pvalue`) and `results/tables/harx_ftest.csv` (`test, F, df_num, df_resid, n, pvalue`; rows `HARX_CI_vs_HARX_C`, `HARX_C_vs_HAR_h`). `insample.lr_test` returns `lr_raw, stat, df, pvalue, negative, k_restricted, k_full`: tables report `stat` (clipped at 0) and `lr_raw`; placebo comparisons use `lr_raw`. `forecast.highvol_threshold` is the single high-vol threshold implementation. `make_model` gives `mean_exog` variables a non-switching coefficient `theta_<var>`.
- A robustness run never overwrites a main-run file: `run_id` "main" and "timing" are refused by `robustness.guard_run_id`.
- Outside `design.py` and `weekly.py`, no code calls `.shift(` (the grep test of spec §10 covers the new modules too). Use `np.roll`-based `insample.circular_shift` for placebos and `.diff()` or index alignment elsewhere.
- Size budget as in Plan 1 (`paths.check_budget()` in the last cell of every notebook): any file ≤ 10 MB; `data/` + `results/` ≤ 20 MB; each notebook ≤ 5 MB; all notebooks ≤ 15 MB. Figures: PNG at 150 dpi via `plots.save_fig`; inline figures at 100 dpi.
- Every notebook: first cell `%load_ext autoreload`, `%autoreload 2`, `import brent`, `from brent import config as C`, `np.random.seed(C.SEED)`, a `FORCE_*` flag block; last cell an asserts cell and `paths.check_budget()`; committed **with outputs** after restart-and-run-all, only at milestones.
- Commits are authored by the repo's git user, with **no co-author trailer** (as in Plan 1).
- Hand-off audience: Linnéa (report and presentation). The GCP project id and the user's email never appear in any file.

## Review Focus

The five failure modes the spec implies but no spec test pins. Each has a test in the task named in brackets.

1. **Misdated joins.** A forecast scored against the wrong week, or the GW instrument dated at the target instead of the origin. Expected: `origin == date − 7 days` for every row, losses aligned on `date`, instrument indexed by origin. [Task 1: `check_alignment`, `gw_instrument`; Task 4: event offsets]
2. **Non-positive forecast or RV inside QLIKE.** A floored-to-zero or negative variance forecast would give `inf` or `nan` that silently poisons a mean. Expected: `ValueError` naming the offending count. [Task 1]
3. **Degenerate test inputs.** Two identical forecasts in DM, a constant forecast in MZ, a loss differential of all zeros in GW/CW, 45 holdout weeks against 4 HAC lags. Expected: NaN statistic and p-value with `n` reported, never an exception or `inf`. [Task 2]
4. **Sparse data in figures and tables.** Events in the training window (Dec 2011, 2008) have no OOS rows; reliability bins can be empty; an event window can run past the end of the panel. Expected: the figure and table still build, with NaN where nothing exists and a `holdout` flag on 2026 rows. [Tasks 3, 4]
5. **A robustness run clobbering or peeking.** A run named `main`, an OOS robustness run that reaches into 2026, or an OOS placebo whose shifted column leaks 2026 values. Expected: refused by `guard_run_id` and `guard_oos_targets`; shifted columns equal the original outside the window. [Task 9]

---

## File structure

| File | Responsibility |
|---|---|
| `src/brent/evaluate.py` (create, Tasks 1–2) | losses, split/family tables, validation-only model choice, GW instrument, inference tests, calibration scores, alignment and guards |
| `src/brent/plots.py` (extend Plan 1's minimal file, Task 3) | palette and styles, event shading, split line, forecast-vs-actual, cumulative loss, reliability, ACF, event window and grid |
| `src/brent/events.py` (create, Task 4) | `event_table`, deck fact-check helpers (weekly-change rank, peak, n-day moves) |
| `src/brent/robustness.py` (create, Task 9) | run-id guards, variant fits, OOS comparison tables, placebo p-values, HARX gain, mixture density helpers, status log, run-time estimates |
| `src/brent/design.py`, `forecast.py`, `ms.py` (Plan 2; additive edits, Task 8) | `dummy_*` passthrough standardization; `forecast.mixture_params`; `ms.make_mshar_specs` |
| `src/brent/insample.py` (Plan 2; additive, Task 9b) | `placebo_offsets`, `placebo_fit`, `placebo_lr`, `placebo_harx` |
| `src/brent/design.py`, `ms.py`, `forecast.py`, `oos.py` (Plan 2; E1 branches, Task 14a, gated) | `kind="mshar"` design, `canonicalize(by="const")`, `one_step`, `run_oos` targets |
| `tests/conftest.py` (extend, Task 0) | every shared test fixture used below |
| `src/brent/config.py` (modify, Task 1) | a few Plan 3 constants |
| `notebooks/06_evaluation.ipynb`, `07_events.ipynb`, `08_robustness.ipynb` | steps, tables, interpretation |
| `tests/test_evaluate.py`, `test_plots.py`, `test_events.py`, `test_robustness.py` (+ additions to `test_design.py`, `test_forecast.py`, `test_ms.py`) | spec §10 plus the review-focus tests |

---

### Task 0: Shared test fixtures (`tests/conftest.py`)

**Files:**
- Modify: `tests/conftest.py` (Plan 1 created it with synthetic raw data)
- Create: `tests/test_conftest.py`

**Interfaces:**
- Consumes: Plan 2's `sim_panel(n, beta, phi, seed)` (MS simulation used by its tests) and `ms.make_specs`, `ms.fit_ms`.
- Produces (plain functions, imported by the tests below; seeded with `np.random.default_rng(405)` unless a `seed` argument is given):
  - `synthetic_panel(seed=405) -> DataFrame`: 1,140 Fridays 2005-01-07 → 2026-11-06 with every column of spec §9.1 (`date, price, r, rv, log_rv, n_days, iran_share, iran_gdelt, iran_gdelt_geo, gdelt_days, gpr, vix, ovx, aigpr_iran, iran_conflict, iran_coop, iran_tone, iran_goldstein, rv_hourly, log_rv_hourly, n_hours_week, rv_gk`); `rv`, `rv_hourly`, `rv_gk` strictly positive; `rv_hourly` NaN before `C.HOURLY_SAMPLE_START`. Aliases `toy_panel = toy_panel_full = synthetic_panel`. `VALID_MASK(panel)` = boolean mask of rows inside `C.VALID`.
  - `toy_fc(seed=405) -> DataFrame`: spec §9.2 long format incl. `actual_rv, actual_rv_hourly, actual_rv_gk, actual_r, actual_exceed`; targets 2013-01-04 → 2026-11-06; `origin = date − 7 days`; `split` by config; `run = "main"`; models `M1, M2, M3, GARCH_t, HAR, HAR_h, HARX_C, HARX_CI, GARCHX_C, GARCHX_CI` with `target = "var"` (positive), `M1, M2, M3` also `r`, `p_high`, `p_exceed`, and `RW` with `r`. `toy_fc_full = toy_fc`.
  - `toy_fc_two_models(f_a=None, f_b=None, rv=None) -> DataFrame`: models `A`, `B`, `target = "var"`. Default (all `None`): A has the lower QLIKE in validation and the higher in test, B the reverse. With `f_a, f_b, rv` given: exactly 30 Fridays from 2019-01-04, `split = "test"`, constant forecasts and `actual_rv = actual_rv_hourly = rv`.
  - `toy_filt() -> DataFrame` (`date, model, xi_H`; models M2 and M3; every Friday 2005–2026).
  - `toy_event_tbl(event_id="hormuz_2011", oos=True)`, `toy_event_tbl_all()`: **hand-built** frames with exactly the `events.event_table` columns `event_id, iran, label, event_date, event_week, k, week, split, holdout, model, filt_P_H, oos_pi_H, qlike, measure, iran_conflict, iran_coop` (k = −4…+4; models M2, M3, HARX_CI; `oos=False` leaves `oos_pi_H` and `qlike` NaN; `_all` has the 9 Iran and 3 non-Iran events). They do not import `events`, so Task 3 does not depend on Task 4.
  - `toy_brent_weekly() -> DataFrame` (`date, brent`): business days 2019-12-23 → 2020-01-17, closes constant within a week except Fridays: 2019-12-27 = 100, 2020-01-03 = 110, 2020-01-10 = 104.5, 2020-01-17 = 106.59 (log changes +ln 1.10, ln 0.95, ln 1.02).
  - `toy_brent_move() -> DataFrame`: business days 2020-01-02 → 2020-01-13 with closes 100, 102, 101, 103, 104, 105, 106, 107 (Jan 2, 3, 6, 7, 8, 9, 10, 13).
  - `toy_brent_daily_business() -> DataFrame`: random-walk closes on business days 2019-08-01 → 2026-12-01.
  - `planted_hourly_panel(seed=405) -> DataFrame`: `synthetic_panel` with `rv_hourly` regenerated as a positive series that depends strongly on `iran_gdelt(t−1)` (so the HARX_CI gain is positive in 2006-11-24 → 2013).
  - `fit_mshar_simulated(seed=405) -> (MSFit, Design)`: a short simulated log-RV series with two intercept regimes (`kind="mshar"`), fitted with `ms.fit_ms` on `make_mshar_specs()["MSHAR_C"]` (used only by Task 14a; written when that task starts).
  - `run_two_toy_origins(specs, benchmarks) -> DataFrame`: runs `oos.run_oos` on `synthetic_panel()` for two origins in a throwaway `run_id`, with `paths` pointed at `tmp_path`, and returns `oos.load_forecasts(run_id)` (used by Task 14a; written when that task starts).
  - `fit_simulated_m1(seed=405, n=400) -> (MSFit, Design)` and `fit_m2_m3_simulated(seed=405) -> (m2: MSFit, m3: MSFit, design, panel)` (constrained M3; Iran raises P(N→H), so β > 0), built from `sim_panel(n, beta, phi, seed)`, `ms.make_specs`, `ms.fit_ms` with a small `search_reps` for speed.

- [ ] **Step 1:** Write the fixtures, and one test per fixture family in `tests/test_conftest.py` asserting its documented shape (row counts, columns, `origin == date − 7 days`, positivity, the Friday closes of `toy_brent_weekly`, `m3.loglike >= m2.loglike − 1e-6` for `fit_m2_m3_simulated`).
- [ ] **Step 2:** `uv run pytest tests/test_conftest.py -q`; expect PASS.
- [ ] **Step 3: Commit**

  ```bash
  git add tests/conftest.py tests/test_conftest.py
  git commit -m "Add shared test fixtures for evaluation, events and robustness tests"
  ```

---

### Task 1: `evaluate.py` part 1 — losses, selection, instruments, calibration

**Files:**
- Create: `src/brent/evaluate.py`, `tests/test_evaluate.py`
- Modify: `src/brent/config.py` (add constants below), `tests/test_config.py`, `.gitignore`

**Interfaces:**
- Consumes: `config` (`VALID, TRAIN, TEST, HOLDOUT, RV_MEASURES, MAIN_RV, LINEAR_MODELS, BENCHMARKS_MAIN, HIGHVOL_PCTL, GW_IRAN_PCTL, GW_WINDOW, SPEC_FROZEN`); `oos.load_forecasts` frames (spec §9.2: long format, columns `date, origin, split, run, model, target, forecast, actual_rv, actual_rv_hourly, actual_rv_gk, actual_r, actual_exceed`).
- Produces (signatures as spec §8.10, plus the additions marked **new**):
  - `qlike(rv, f) -> np.ndarray` (raises `ValueError` if any `rv <= 0`, `f <= 0` or non-finite; message contains the count); `sq_err(y, f) -> np.ndarray`
  - `loss_table(fc, measure=C.MAIN_RV, split_col="split") -> pd.DataFrame` with columns `model, family, split, measure, n, qlike, rmse`; rows only for `target == "var"`
  - **new** `family(model: str) -> str`: `main` for M1, M2, M3, M3tone; `benchmark` for GARCH_t, HAR, RW; `extra` for HAR_h, HARX_C, HARX_CI, GARCHX_C, GARCHX_CI; `robustness` for anything else
  - `best_linear_model(fc, measure=C.MAIN_RV, window=C.VALID, pool=C.LINEAR_MODELS) -> str` (pool members absent from `fc` are ignored)
  - `highvol_threshold`: a **re-export** (`from brent.forecast import highvol_threshold`), not a second implementation; `gw_instrument(panel, origins) -> pd.Series` (index = origins, 0/1)
  - **new** `check_instrument_share(instr: pd.Series, lo=0.10, hi=0.90) -> float` (raises `ValueError` with the share if outside)
  - `cumulative_loss_diff(loss_a, loss_b) -> pd.Series`; `brier(p, event) -> float`; `reliability_table(p, event, n_bins=10) -> pd.DataFrame` (columns `bin, p_mean, freq, n`; empty bins kept with NaN); `sign_hit_rate(y, f) -> float`; `ljung_box(errors, lags=(4, 8, 13)) -> pd.DataFrame` (columns `lag, stat, pvalue`)
  - **new** `check_alignment(fc: pd.DataFrame) -> None` (raises if any `origin != date − 7 days`, duplicate `(date, model, target)` rows, or `split` inconsistent with `date` and the config windows)
  - **new** `guard_splits(splits, dev: bool) -> None` (raises `PermissionError` if `test`/`holdout` requested while `not C.SPEC_FROZEN and not dev`)
  - **new** `available_pairs(pairs, models) -> tuple[list, list]` (kept, skipped)
  - **new** `DM_PAIRS`, `CW_PAIRS` constants (Task 2 uses them; defined here so the count of tests is one number)

- [ ] **Step 1: Add constants to `config.py` and `.gitignore`**

  Append to `config.py`: `ROBUST_OOS_TARGETS = ("2019-01-04", "2025-12-26")`, `OVX_START = "2007-05-11"`, `OVX_STD_WINDOW = ("2007-05-11", "2012-12-28")` (spec §8.1 comment), `DUMMY_POST2013_WEEK = "2013-04-05"`, `TAIL_PCTLS = (90, 95, 99)` (percentiles of |r| on TRAIN for the tail table, **[plan call]**), `MZ_HAC_LAGS = 4`. Add to `test_config.py` a test that `ROBUST_OOS_TARGETS[1] == C.TEST[1]`. Append to `.gitignore`: `results/tables/dev_*`, `results/figures/dev_*`, `results/forecasts/forecasts_fake.csv`.

- [ ] **Step 2: Write the failing tests** (`tests/test_evaluate.py`; fixtures `toy_fc`, `toy_fc_two_models`, `toy_panel`, `VALID_MASK` from Task 0)

  ```python
  def test_qlike_zero_at_truth_and_exact_value():
      assert np.allclose(E.qlike(np.array([2.0, 3.0]), np.array([2.0, 3.0])), 0)
      assert np.isclose(E.qlike(np.array([2.0]), np.array([1.0]))[0], 1 - np.log(2))   # 0.3068528194

  @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
  def test_qlike_rejects_nonpositive(bad):               # Review focus 2
      with pytest.raises(ValueError, match="1"):
          E.qlike(np.array([1.0, 2.0]), np.array([1.0, bad]))

  def test_loss_table_uses_measure_column_and_family():
      t = E.loss_table(toy_fc(), measure="rv")
      assert set(t["measure"]) == {"rv"} and {"model", "family", "split", "n", "qlike", "rmse"} <= set(t.columns)
      assert (t.query("model == 'HARX_CI'")["family"] == "extra").all()
      assert "RW" not in set(t["model"])                 # RW has no var forecast

  def test_best_linear_model_ignores_test_window():
      fc = toy_fc_two_models()   # A: low loss in validation, high in test; B: the reverse
      assert E.best_linear_model(fc, pool=("A", "B")) == "A"
      assert E.best_linear_model(fc, pool=("A", "B", "NOT_THERE")) == "A"

  def test_gw_instrument_is_dated_at_origin_and_threshold_from_valid():   # Review focus 1
      panel = toy_panel()        # iran_gdelt = linspace; VALID rows known
      origins = pd.DatetimeIndex(["2019-01-04", "2019-01-11"])
      s = E.gw_instrument(panel, origins)
      thr = np.percentile(panel.loc[VALID_MASK, "iran_gdelt"], 75)
      assert list(s.index) == list(origins)
      assert s.iloc[0] == float(panel.set_index("date").loc["2019-01-04", "iran_gdelt"] > thr)
      panel2 = panel.copy(); panel2.loc[panel2["date"] > "2018-12-28", "iran_gdelt"] += 5
      assert E.gw_instrument(panel2, origins).iloc[0] == 1      # threshold unchanged by post-2018 data
      with pytest.raises(ValueError, match="0.0"):
          E.check_instrument_share(pd.Series([0, 0, 0, 0]))

  def test_highvol_threshold_is_the_forecast_module_function_and_uses_train_only():
      from brent import forecast as F
      assert E.highvol_threshold is F.highvol_threshold
      p = toy_panel(); t0 = E.highvol_threshold(p)
      p.loc[p["date"] > "2012-12-28", "r"] *= 10
      assert E.highvol_threshold(p) == t0 == F.highvol_threshold(p)

  def test_cumulative_loss_diff_ends_at_sum():
      a, b = np.array([1., 2, 3]), np.array([0.5, 2.5, 1])
      assert np.isclose(E.cumulative_loss_diff(a, b).iloc[-1], (a - b).sum())

  def test_brier_exact():
      assert np.isclose(E.brier(np.array([0.9, 0.1, 0.5]), np.array([1, 0, 1])), 0.09)

  def test_reliability_table_keeps_empty_bins():           # Review focus 4
      t = E.reliability_table(np.array([0.05, 0.05, 0.95, 0.95, 1.0]), np.array([0, 1, 1, 1, 1]))
      assert len(t) == 10 and t.loc[0, "n"] == 2 and t.loc[0, "freq"] == 0.5
      assert t.loc[9, "n"] == 3 and t.loc[9, "freq"] == 1.0   # p = 1.0 falls in the last bin
      assert t.loc[4, "n"] == 0 and np.isnan(t.loc[4, "freq"])

  def test_ljung_box_white_noise_vs_ar():
      rng = np.random.default_rng(405); e = rng.standard_normal(500)
      assert (E.ljung_box(e)["pvalue"] > 0.01).all() and list(E.ljung_box(e)["lag"]) == [4, 8, 13]
      ar = np.zeros(500)
      for i in range(1, 500): ar[i] = 0.8 * ar[i - 1] + e[i]
      assert (E.ljung_box(ar)["pvalue"] < 1e-10).all()

  def test_check_alignment_flags_bad_origin_and_duplicates():
      fc = toy_fc(); E.check_alignment(fc)
      bad = fc.copy(); bad.loc[bad.index[0], "origin"] = bad.loc[bad.index[0], "date"] - pd.Timedelta(days=14)
      with pytest.raises(ValueError, match="origin"): E.check_alignment(bad)
      with pytest.raises(ValueError, match="duplicate"): E.check_alignment(pd.concat([fc, fc.iloc[:1]]))

  def test_guard_splits():
      assert not C.SPEC_FROZEN
      E.guard_splits(["validation"], dev=False)
      with pytest.raises(PermissionError): E.guard_splits(["test"], dev=False)
      E.guard_splits(["test", "holdout"], dev=True)

  def test_available_pairs_skips_missing_models():
      keep, skip = E.available_pairs([("M3", "M2"), ("M3", "HAR_h")], {"M3", "M2"})
      assert keep == [("M3", "M2")] and skip == [("M3", "HAR_h")]
  ```

- [ ] **Step 3: Run `uv run pytest tests/test_evaluate.py -q`; expect FAIL** (`brent.evaluate` missing).

- [ ] **Step 4: Implement the functions above in `evaluate.py`**

  Notes where tests leave a choice: `loss_table` groups by `model, split` and takes `actual_<measure>` from the same rows; `reliability_table` bins `[0, 0.1), …, [0.9, 1.0]`; `gw_instrument` uses `np.percentile(..., C.GW_IRAN_PCTL)` (linear interpolation) over rows of `panel` inside `C.GW_WINDOW`; `check_alignment` reads the windows from config; `ljung_box` wraps `statsmodels.stats.diagnostic.acorr_ljungbox(errors, lags=list(lags))`.

  Define in the same file, once:
  ```python
  DM_PAIRS = [("M3","M2"),("M3","M1"),("M3","GARCH_t"),("M3","HAR"),("M3","HAR_h"),("M3","HARX_CI"),("M3","GARCHX_CI"),
              ("HARX_CI","HARX_C"),("GARCHX_CI","GARCHX_C"),("HARX_C","HAR_h"),("GARCHX_C","GARCH_t")]   # (A, B); spec §7.6
  CW_PAIRS = [("M3","M2"),("HARX_CI","HARX_C"),("GARCHX_CI","GARCHX_C")]                                    # (big, small)
  ```
  In both, the comparator `B` (or the pair type) fixes the `family` label of the row: `main` for M3 vs M2, otherwise `family(B)` except nested pairs, which get `extra`. Also define `NESTED_DM = {("M3","M2"), ("M3","M1"), ("HARX_CI","HARX_C"), ("GARCHX_CI","GARCHX_C"), ("HARX_C","HAR_h"), ("GARCHX_C","GARCH_t")}`; a test asserts every `NESTED_DM` pair is in `DM_PAIRS`.

- [ ] **Step 5: Run the tests; expect PASS.**

- [ ] **Step 6: Commit**

  ```bash
  git add src/brent/evaluate.py src/brent/config.py tests/test_evaluate.py tests/test_config.py .gitignore
  git commit -m "Add evaluation losses, validation-only model selection, GW instrument and calibration scores"
  ```

---

### Task 2: `evaluate.py` part 2 — inference tests

**Files:**
- Modify: `src/brent/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: Task 1 helpers; `statsmodels.tsa.stattools.diebold_mariano_test`.
- Produces (all return dicts with `stat`, `pvalue`, `n` plus the listed keys; NaN stat and p-value, never an exception, when the input is degenerate):
  - `dm_test(y, f_a, f_b, loss="qlike") -> dict` (`loss` ∈ `qlike`, `sqerr`; keys `mean_diff` = mean(L_a − L_b), `loss`). Sign convention: **stat > 0 means A has the higher loss (B is better)**. Two-sided p from the HLN t(n−1) reference.
  - `clark_west(y, f_small, f_big, hac_lags=4) -> dict` (adjusted-MSPE series f_t = (y−f_small)² − (y−f_big)² + (f_small−f_big)²; t from OLS of f_t on a constant with Newey-West `hac_lags`; **one-sided** p = `norm.sf(stat)`)
  - `giacomini_white(loss_a, loss_b, instruments: pd.DataFrame | None) -> dict` (Δ = L_a − L_b; h_t = constant plus the instrument columns; stat = n·Z̄′Ω⁻¹Z̄ with Z_t = h_tΔ_t, Ω = mean(Z_tZ_t′); χ²(q), q = number of columns of h; keys `df`, `kind` = `unconditional` or `conditional`)
  - `mincer_zarnowitz(y, f, hac_lags=C.MZ_HAC_LAGS) -> dict` (keys `a, b, se_a, se_b`, joint Wald for (a, b) = (0, 1) as `stat`, χ²(2) `pvalue`)
  - `pesaran_timmermann(y, f) -> dict` (sign of y and f as `> 0`; right-tailed normal p)

- [ ] **Step 1: Write the failing tests**

  ```python
  D = np.array([1.]*7 + [-1.]*3)         # loss differential: mean 0.4, var (ddof 0) 0.84

  def test_dm_matches_hand_hln_on_squared_error():
      # y = 0, f_b = 1, f_a = sqrt(2) where D = +1 and 0 where D = -1  =>  sq_err_a - sq_err_b = D
      f_a = np.where(D > 0, np.sqrt(2.0), 0.0); r = E.dm_test(np.zeros(10), f_a, np.ones(10), loss="sqerr")
      assert np.isclose(r["stat"], 1.3093073414, atol=1e-8) and np.isclose(r["pvalue"], 0.2228683501, atol=1e-8)
      assert r["n"] == 10 and np.isclose(r["mean_diff"], 0.4)
      # hand version: DM = mean/sqrt(var0/n); HLN = DM*sqrt((n-1)/n); p = 2*t.sf(|HLN|, n-1)

  def test_dm_identical_forecasts_is_nan_not_error():          # Review focus 3
      f = np.array([1., 2, 3, 4]); r = E.dm_test(np.array([1., 2, 2, 5]), f, f)
      assert np.isnan(r["stat"]) and np.isnan(r["pvalue"]) and r["n"] == 4

  def test_clark_west_exact_toy_and_positive_when_big_is_right():
      y = np.ones(4); f_small = np.zeros(4); f_big = np.array([1., 2, 1, 2])   # adj series = 2*y*f_big = [2,4,2,4]
      r = E.clark_west(y, f_small, f_big, hac_lags=0)
      assert np.isclose(r["stat"], 6.0) and np.isclose(r["pvalue"], 9.865876e-10, rtol=1e-4)

  def test_gw_constant_instrument_is_unconditional():
      loss_b = np.zeros(10); u = E.giacomini_white(D, loss_b, None)
      assert np.isclose(u["stat"], 1.6) and np.isclose(u["pvalue"], 0.2059032107, atol=1e-8) and u["df"] == 1
      assert E.giacomini_white(D, loss_b, pd.DataFrame({"const": np.ones(10)}))["stat"] == pytest.approx(1.6)

  def test_gw_conditional_exact():
      d = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, 1.]); x = pd.DataFrame({"iran_high": [1]*5 + [0]*5})
      r = E.giacomini_white(d, np.zeros(10), x)
      assert np.isclose(r["stat"], 5.2) and r["df"] == 2 and np.isclose(r["pvalue"], 0.0742735782, atol=1e-8)

  def test_gw_zero_differential_is_nan():                       # Review focus 3
      assert np.isnan(E.giacomini_white(np.zeros(5), np.zeros(5), None)["stat"])

  def test_pt_exact_toy_and_noise():
      y = np.array([1., 1, -1, -1]); r = E.pesaran_timmermann(y, y)
      assert np.isclose(r["stat"], 2.3094010767, atol=1e-8) and np.isclose(r["pvalue"], 0.0104606677, atol=1e-8)
      rng = np.random.default_rng(405); assert E.pesaran_timmermann(rng.standard_normal(400), rng.standard_normal(400))["pvalue"] > 0.01

  def test_mz_recovers_identity_and_exact_line():
      rng = np.random.default_rng(405); f = rng.uniform(1, 3, 2000); y = f + 0.3 * rng.standard_normal(2000)
      r = E.mincer_zarnowitz(y, f); assert abs(r["a"]) < 0.1 and abs(r["b"] - 1) < 0.05 and r["pvalue"] > 0.001
      line = E.mincer_zarnowitz(0.5 + 2 * np.array([1., 2, 3, 4]), np.array([1., 2, 3, 4]))
      assert np.isclose(line["a"], 0.5) and np.isclose(line["b"], 2.0)

  def test_mz_constant_forecast_and_short_sample_do_not_raise():    # Review focus 3
      r = E.mincer_zarnowitz(np.array([1., 2, 3, 4, 5]), np.ones(5)); assert np.isnan(r["stat"]) and r["n"] == 5
      assert "stat" in E.mincer_zarnowitz(np.arange(6.0), np.arange(6.0) + 0.1, hac_lags=4)
  ```

- [ ] **Step 2: Run; expect FAIL.**

- [ ] **Step 3: Implement the five functions.**
  - `dm_test`: build the loss series with `qlike`/`sq_err`, hand the two loss series to `diebold_mariano_test(..., lags=0, harvey_adj=True, horizon=1)` via its custom-loss callable (spec §7.6). If the library's variance convention differs from the hand formula in the first test (variance with ddof 0, HLN factor √((n−1)/n), t(n−1)), reconcile the **wrapper** to the test, and note the library's behavior in the docstring. Return NaN stat and p when the differential has zero variance or `n < 3`.
  - `clark_west`: `statsmodels.api.OLS(f_t, ones).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})`.
  - `giacomini_white`: the matrix form in the Interfaces block; NaN when Ω is singular.
  - `mincer_zarnowitz`: OLS of `y` on `[1, f]` with Newey-West; Wald via `wald_test`; when the covariance is singular or the residuals are exactly zero (noise-free line) report `stat = inf`, `pvalue = 0.0`; when `f` is constant report NaN.
  - `pesaran_timmermann`: formulas as in Pesaran & Timmermann (1992): P = hit rate, P* = P_yP_f + (1−P_y)(1−P_f), var(P) = P*(1−P*)/n, var(P*) = (2P_y−1)²P_f(1−P_f)/n + (2P_f−1)²P_y(1−P_y)/n + 4P_yP_f(1−P_y)(1−P_f)/n²; stat = (P − P*)/√(var(P) − var(P*)); NaN if the denominator is ≤ 0.

- [ ] **Step 4: Run `uv run pytest tests/test_evaluate.py -q`; expect PASS.**

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/evaluate.py tests/test_evaluate.py
  git commit -m "Add DM-HLN, Clark-West, Giacomini-White, Mincer-Zarnowitz and Pesaran-Timmermann tests"
  ```

---

### Task 3: `plots.py` — palette, event shading, figure builders

**Files:**
- Modify: `src/brent/plots.py` (Plan 1 left `setup_style()` and `save_fig(fig, name) -> Path`)
- Create: `tests/test_plots.py`

**Interfaces:**
- Consumes: `config.EVENTS, NON_IRAN_EVENTS`, `weekly.week_ending`, `evaluate.reliability_table`, `paths.FIGURES`.
- Produces (all return a `matplotlib.figure.Figure`; none call `save_fig`; notebooks do):
  - `COLORS: dict[str, str]` and `style_for(model: str) -> dict` (keys `color`, `linestyle`, `label`). N and H have their own colors; M1, M2, M3 distinct; `M3tone` a lighter M3; `GARCH_t`, `HAR`, `HAR_h` distinct; `HARX_*` use `HAR_h`'s color and `GARCHX_*` use `GARCH_t`'s, both dashed; unknown model names fall back to grey.
  - `shade_events(ax, events, dates) -> None` (shades Saturday→Friday of each event week that lies inside `dates`, mapping with `week_ending`; events outside the range are skipped)
  - `mark_split(ax) -> None` (vertical line at 2026-01-01, drawn only if the axis range contains it)
  - `forecast_vs_actual(actual: pd.Series, forecasts: dict[str, pd.Series], events, title, ylabel) -> Figure`
  - `cumloss_plot(diff: pd.Series, events, title, ylabel) -> Figure` (calls `mark_split`; Σ(L_a − L_b), positive = A worse)
  - `reliability_plot(table: pd.DataFrame, title) -> Figure` (diagonal, bin frequencies with `n` annotated, empty bins skipped)
  - `acf_plot(series: dict[str, np.ndarray], lags=26) -> Figure`
  - `placebo_hist(panels: dict[str, tuple[np.ndarray, float]], xlabel: str) -> Figure` (one histogram per entry: placebo statistics, with a vertical line at the actual value and its placebo p-value in the title)
  - `event_window_plot(event_id: str, tbl: pd.DataFrame) -> Figure` where `tbl` is the long frame from `events.event_table` (Task 4) restricted to one event; five stacked panels (spec §7.7): (1) in-sample filtered P(H) M2/M3; (2) OOS π_H M2/M3 at the target week, x-axis = offset k = −4…+4, y-label "P(high vol in week k), forecast at end of week k−1"; (3) `rv_hourly` actual with variance forecasts of M2, M3, HARX_CI, log scale; (4) weekly QLIKE of M2, M3, HARX_CI; (5) `iran_conflict`, `iran_coop`. Panels with no data (training-window events have no OOS rows) show the actual only and a note "no OOS forecast: event in training window".
  - `events_grid(tbl: pd.DataFrame) -> Figure` (one small panel per event, 4 rows × 3 columns: filtered P(H) M2 vs M3 against k, holdout event titled "holdout, not in the test")

- [ ] **Step 1: Write the failing tests** (`tests/test_plots.py`; `matplotlib.use("Agg")`)

  ```python
  def test_palette_covers_every_model_and_extra_models_are_dashed_base_color():
      for m in ("M1","M2","M3","M3tone", *C.BENCHMARKS_MAIN):
          assert "color" in P.style_for(m)
      assert P.style_for("HARX_CI")["color"] == P.style_for("HAR_h")["color"] and P.style_for("HARX_CI")["linestyle"] == "--"
      assert P.style_for("GARCHX_C")["color"] == P.style_for("GARCH_t")["color"]
      assert P.style_for("M3tone")["color"] != P.style_for("M3")["color"]
      assert P.style_for("nonsense")["color"] == "grey"

  def test_shade_events_skips_out_of_range_and_maps_weekends():
      fig, ax = plt.subplots(); dates = pd.date_range("2019-01-04", "2019-12-27", freq="W-FRI")
      ax.plot(dates, np.zeros(len(dates))); P.shade_events(ax, C.EVENTS, dates)
      assert len(ax.patches) == 2          # tankers_2019 (Jun 13) and abqaiq_2019 (Sat Sep 14 -> week of Sep 20)

  def test_cumloss_and_reliability_and_acf_build(tmp_path):
      d = pd.Series(np.arange(5.0), index=pd.date_range("2025-12-05", periods=5, freq="W-FRI"))
      assert P.cumloss_plot(d.cumsum(), C.EVENTS, "t", "y") is not None
      assert P.reliability_plot(E.reliability_table(np.array([.1, .9, .95]), np.array([0, 1, 1])), "t") is not None
      assert P.acf_plot({"M3": np.random.default_rng(405).standard_normal(100)}) is not None
      assert P.placebo_hist({"in-sample": (np.arange(10.0), 12.0)}, "LR") is not None

  def test_event_window_plot_without_oos_rows_does_not_raise():      # Review focus 4
      fig = P.event_window_plot("hormuz_2011", toy_event_tbl(oos=False)); assert len(fig.axes) >= 5

  def test_events_grid_builds_for_twelve_events():
      assert P.events_grid(toy_event_tbl_all()) is not None
  ```

- [ ] **Step 2: Run; expect FAIL.**

- [ ] **Step 3: Implement the functions above.** One consistent style from `setup_style()`; hex values are the builder's choice but must be colorblind-distinguishable for N/H, M1–M3 and the benchmark trio. `toy_event_tbl*` come from Task 0 (hand-built with the Task 4 columns).

- [ ] **Step 4: Run `uv run pytest tests/test_plots.py -q`; expect PASS.**

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/plots.py tests/test_plots.py tests/conftest.py
  git commit -m "Add palette, event shading and figure builders for evaluation and events"
  ```

---

### Task 4: `events.py` — event table and deck fact checks

**Files:**
- Create: `src/brent/events.py`, `tests/test_events.py`

**Interfaces:**
- Consumes: `weekly.week_ending`, `weekly.weekly_brent`, `weekly.fridays`, `config.EVENTS, NON_IRAN_EVENTS, HOLDOUT, MAIN_RV`, `evaluate.qlike`; frames: `panel` (spec §9.1), `fc` (`oos.load_forecasts("main")`), `filt` (`results/fits/insample_filtered_2005_2026.csv`: `date, model, xi_H`), `brent_daily` (`data/raw/brent_daily.csv`: `date, brent`).
- Produces:
  - `event_table(events, panel, fc, filt, k=4, measure=C.MAIN_RV) -> pd.DataFrame`, long format: columns `event_id, iran (bool), label, event_date, event_week, k, week, split, holdout (bool), model, filt_P_H, oos_pi_H, qlike, measure, iran_conflict, iran_coop`. One row per event × k (−4…+4) × model in (`M2`, `M3`, `HARX_CI`). `filt_P_H` only for M2/M3. `oos_pi_H` only for M2/M3 (target `p_high` at target date `week`). `qlike` from `var` forecast vs `actual_<measure>`. Missing data is NaN; weeks beyond the panel end are dropped. `holdout = event_week > C.TEST[1]`, the same value on every row of an event; `iran = event_id not in {ids of C.NON_IRAN_EVENTS}`.
  - `weekly_change_rank(brent_daily, week_end, how="log") -> dict` (`change`, `rank` among all Friday-to-Friday changes since 1987-05-22, `n`; rank 1 = largest gain)
  - `peak_after(brent_daily, start, end) -> dict` (`peak`, `date` of the highest close in the window)
  - `reaction_day(brent_daily, date) -> pd.Timestamp` (first trading day on or after `date`); `move(brent_daily, date, after_days: int) -> float`: base = last close **strictly before** `date`; `after_days=0` is the reaction move (first close on or after `date` vs base); `after_days=5` is the close 5 trading days after the reaction day vs the same base; percentages
  - `fact_check_table(brent_daily, intraday_daily | None, events, claims: pd.DataFrame | None) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

  ```python
  def test_event_week_mapping_and_offsets():                       # Review focus 1
      t = EV.event_table(C.EVENTS, toy_panel_full(), toy_fc_full(), toy_filt(), k=4)
      a = t[(t.event_id == "abqaiq_2019") & (t.model == "M2")].sort_values("k")
      assert a["event_week"].iloc[0] == pd.Timestamp("2019-09-20") and list(a["k"]) == list(range(-4, 5))
      assert a.loc[a.k == 0, "week"].iloc[0] == pd.Timestamp("2019-09-20")
      row = a.loc[a.k == 0].iloc[0]                                 # oos_pi_H at k=0 is the forecast TARGETING that week
      assert row["oos_pi_H"] == toy_fc_full().query("model=='M2' and target=='p_high' and date=='2019-09-20'")["forecast"].iloc[0]

  def test_training_window_event_has_no_oos_but_keeps_actual_columns():       # Review focus 4
      t = EV.event_table(C.EVENTS, toy_panel_full(), toy_fc_full(), toy_filt(), k=4)
      h = t[(t.event_id == "hormuz_2011") & (t.model == "M3")]
      assert h["oos_pi_H"].isna().all() and h["qlike"].isna().all() and h["iran_conflict"].notna().all()

  def test_2026_event_is_flagged_holdout_and_window_stops_at_panel_end():
      t = EV.event_table(C.EVENTS, toy_panel_full(), toy_fc_full(), toy_filt(), k=4)
      h = t[t.event_id == "hormuz_2026"]
      assert h["holdout"].all() and h["event_week"].iloc[0] == pd.Timestamp("2026-03-06")
      assert t[t.event_id == "abqaiq_2019"]["holdout"].eq(False).all()

  def test_non_iran_events_are_included_with_iran_false():
      ev = C.EVENTS + [(i, d, i, "") for i, d in C.NON_IRAN_EVENTS]
      t = EV.event_table(ev, toy_panel_full(), toy_fc_full(), toy_filt(), k=4)
      assert set(t.loc[~t.iran, "event_id"]) == {"lehman_2008", "covid_2020", "russia_ukraine_2022"}

  def test_weekly_change_rank_and_peak():
      b = toy_brent_weekly()                                  # Friday closes 100, 110, 104.5, 106.59
      r = EV.weekly_change_rank(b, "2020-01-03"); assert r["rank"] == 1 and r["n"] == 3
      assert EV.weekly_change_rank(b, "2020-01-10")["rank"] == 3
      assert EV.peak_after(b, "2020-01-03", "2020-01-17")["date"] == pd.Timestamp("2020-01-03")

  def test_move_reaction_semantics():
      b = toy_brent_move()
      assert np.isclose(EV.move(b, "2020-01-04", 0), 100 * (101 / 102 - 1))      # Saturday: base Fri Jan 3, reaction Mon Jan 6
      assert np.isclose(EV.move(b, "2020-01-04", 5), 100 * (107 / 102 - 1))      # 5 trading days after Jan 6 = Jan 13
      assert np.isclose(EV.move(b, "2020-01-08", 0), 100 * (104 / 103 - 1))      # weekday: base Jan 7, reaction Jan 8
      assert EV.reaction_day(b, "2020-01-04") == pd.Timestamp("2020-01-06")

  @pytest.mark.parametrize("d", ["2019-09-14", "2024-04-13", "2026-02-28"])   # all Saturdays
  def test_saturday_events_react_on_monday(d):
      b = toy_brent_daily_business(); assert pd.Timestamp(d).weekday() == 5
      assert EV.reaction_day(b, d) == pd.Timestamp(d) + pd.Timedelta(days=2)
  ```
  Fixtures from Task 0.

- [ ] **Step 2: Run; expect FAIL.**

- [ ] **Step 3: Implement.** `weekly_change_rank` builds Friday closes with `weekly.weekly_brent(brent_daily, weeks)` on `weekly.fridays(start="1987-05-22", end=C.FREEZE_DATE)` so the week rules are the panel's. `fact_check_table` rows: (a) close on 2026-02-27, the 2026 peak close after it and its date, the intraday high from `intraday_daily["high"]` over the same window labelled `instrument = "Dukascopy CFD"` (never mixed with the spot rows); (b) the week ending 2026-03-06: change, rank, n; (c) for every event date in `events` the reaction move (`move(..., 0)`) and the 5-day move (`move(..., 5)`), with the reaction day printed; (d) if `claims` is given (`results/frozen/deck_claims.csv`, columns `claim_id, event_id, claimed_1d, claimed_5d`, filled by hand), add `claimed_*` and `diff_*` columns.

- [ ] **Step 4: Run `uv run pytest tests/test_events.py -q`; expect PASS.**

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/events.py tests/test_events.py tests/conftest.py
  git commit -m "Add event table and deck fact-check helpers"
  ```

---

### Task 5: Notebook 06 — build on fake forecasts (Mon Nov 30, while the OOS run finishes or before)

**Files:**
- Create: `notebooks/06_evaluation.ipynb`

**Interfaces:**
- Consumes: `oos.fake_forecasts(panel)`, `oos.load_forecasts(run_id, with_actuals=True)`, Task 1–3 functions, `paths.save_table`, `plots.save_fig`.
- Produces: the notebook skeleton with all cells working in `DEV = True` mode on `validation`-split rows, writing only `results/tables/dev_*.csv` and `results/figures/dev_*.png` (git-ignored).

- [ ] **Step 1: Mode cell.**

  `DEV = not (C.SPEC_FROZEN and (paths.FORECASTS / "forecasts_main.csv").exists())`. If `DEV`, write `oos.fake_forecasts(panel).to_csv(paths.FORECASTS / "forecasts_fake.csv", index=False)` (git-ignored) and load it with `oos.load_forecasts("fake")` (which joins the `actual_*` columns), print a banner "DEV: fake forecasts, never reported", and set `OUT = "dev_"`; else load `oos.load_forecasts("main")` and `OUT = ""`. Splits to report: `("test", "holdout")` when not `DEV`, with `("validation",)` in dev; call `evaluate.guard_splits(splits, DEV)` and `evaluate.check_alignment(fc)` first. Print the model list and which `DM_PAIRS`/`CW_PAIRS` are skipped (`available_pairs`).

- [ ] **Step 2: Volatility tables** (one loop over `C.RV_MEASURES`, each table gets `measure`, `split`, `family`; save with `OUT` prefix)
  1. `vol_losses.csv`: `evaluate.loss_table` per measure, all three splits (validation appended as the appendix rows).
  2. `vol_mz.csv`: every model × split × measure, columns `a, b, se_a, se_b, stat, pvalue, n`.
  3. `vol_dm.csv`: every pair in `DM_PAIRS` × loss (`qlike`, `sqerr`) × split × measure, columns `A, B, loss, family, stat, pvalue, mean_diff, n, nested (bool)`. `nested` is True for the pairs in `evaluate.NESTED_DM` (M3/M2, M3/M1, HARX_CI/HARX_C, GARCHX_CI/GARCHX_C, HARX_C/HAR_h, GARCHX_C/GARCH_t; caveat in the markdown, spec §14 item 7).
  4. `vol_cw_gw.csv`: Clark-West on squared error for `CW_PAIRS`; Giacomini-White unconditional and conditional (QLIKE) for M3 vs M2 only, on `test` and `holdout`. Instrument = `evaluate.gw_instrument(panel, origins)` for the origins of the rows being tested; call `evaluate.check_instrument_share` on the **test** origins and stop the notebook with the printed share if it raises. If the user decided to drop the conditional test (spec §11 slip rule: spec not frozen by Sun Nov 29), skip the conditional rows and say so in a markdown cell. Add the caveats of spec §14 items 5 and 7 as markdown.
  5. `vol_best_linear.csv`: `evaluate.best_linear_model(fc)` on validation, with the validation mean QLIKE of every pool member; print the name.
  6. `ljung_box.csv`: errors `actual − forecast` for every model × target (for `var`, per measure), lags 4, 8, 13.
  7. Markdown cell: the number of tests run (len of DM, CW, GW rows after pairs are filtered) and the sentence that M3 vs M2 on `rv_hourly`, test window, is the pre-specified headline, everything else secondary, nothing adjusted.

- [ ] **Step 3: Return, high-vol and calibration tables**
  1. `ret_eval.csv`: for M1, M2, M3 (and M3tone if present): RMSE of the `r` forecast, RMSE of RW (forecast 0), the ratio, `sign_hit_rate`, Pesaran-Timmermann `stat` and `pvalue`, per split.
  2. `phigh_eval.csv`: per MS model and split: `brier` of `p_high` vs `actual_exceed`; hit rate = share of exceed weeks with `p_high > 0.5`; base rate; Brier of `p_exceed` vs `actual_exceed`.

- [ ] **Step 4: Figures** (names as spec §7.6; `shade_events` on every time-series figure)
  - `vol_fc_vs_actual_test.png`, `vol_fc_vs_actual_holdout.png`: `rv_hourly` actual with M2, M3, HARX_CI, GARCHX_CI forecasts.
  - `cumloss_M2_minus_M3.png`: Σ(L_M2 − L_M3), QLIKE vs `MAIN_RV`, test and holdout concatenated in time order with the split line; if `"rv"` is in `C.RV_MEASURES` and is not the main measure, also `cumloss_M2_minus_M3_rv.png`.
  - `cumloss_bestlinear_minus_M3.png`: Σ(L_best − L_M3) with `best` read from `vol_best_linear.csv`.
  - `vol_error_acf.png`; `ret_fc_vs_actual.png`; `ret_error_acf.png`; `phigh_reliability.png` (10 bins, M2 and M3); `phigh_vs_bigmoves.png` (π_H over time with exceedance weeks marked).
  - A cell asserts that each cumulative-loss series ends at the sum of its per-week differences.

- [ ] **Step 5: Checks cell.** Forecasts and actuals aligned on `date`; no NaN losses for any measure in 2013–2026 (dev data included); DM, CW and GW on the toy cases from `tests/test_evaluate.py` reproduce (import and call the same toy arrays); `paths.check_budget()`; assert `OUT == ""` before the commit step (a notebook committed in dev mode is rejected).

- [ ] **Step 6: Run in dev mode end to end** (restart and run all) and confirm only `dev_*` files are written: `git status --short results/` shows no new tracked files.

- [ ] **Step 7: Commit the notebook source without outputs** (dev run is not a milestone)

  ```bash
  # in VS Code: Restart kernel and clear all outputs, save
  git add notebooks/06_evaluation.ipynb
  git commit -m "Add notebook 06 skeleton, runs on fake forecasts in dev mode"
  ```

---

### Task 6: Notebook 06 — run on the real forecasts (Mon Nov 30 → Tue Dec 1)

**Files:**
- Modify: `notebooks/06_evaluation.ipynb`
- Create (by running): `results/tables/{vol_losses,vol_mz,vol_dm,vol_cw_gw,vol_best_linear,ljung_box,ret_eval,phigh_eval}.csv`, `results/figures/{vol_fc_vs_actual_test,vol_fc_vs_actual_holdout,cumloss_M2_minus_M3,cumloss_M2_minus_M3_rv,cumloss_bestlinear_minus_M3,vol_error_acf,ret_fc_vs_actual,ret_error_acf,phigh_reliability,phigh_vs_bigmoves}.png`

**Interfaces:**
- Consumes: `results/forecasts/forecasts_main.csv`, `results/frozen/spec_freeze.json`, `config.SPEC_FROZEN` (all from Plan 2, notebook 05).

- [ ] **Step 1: Entry check (branch on what exists).**

  | Condition at Mon Nov 30 | Action |
  |---|---|
  | `SPEC_FROZEN` True and `forecasts_main.csv` has 723 target weeks per model | continue with this task |
  | `SPEC_FROZEN` True, run unfinished | minimum-viable mode (spec §11): keep `DEV = True` until it finishes; build Task 7 meanwhile; notebook 08 limited to in-sample placebo, HARX placebo and in-sample AI-GPR |
  | `SPEC_FROZEN` False | stop; report to the user. Robustness becomes in-sample only and the conditional GW test is dropped (spec §11) |

  Also assert the `settings_hash` column of the fitlog equals the one in `spec_freeze.json`.

- [ ] **Step 2: Restart and run all** with `DEV = False`. Read the printed alignment check, the GW instrument share, the best linear model name and the number of tests.

- [ ] **Step 3: Interpretation cells.** For each table a short markdown cell, leading with `rv_hourly` and test; holdout (45 weeks) is labelled as small-sample, secondary measures follow. State in words: whether M3 beats M2 on QLIKE, the DM/CW/GW p-values with the nested-model caveats, M3 against HARX_CI and GARCHX_CI ("do regimes beat a linear news effect?"), whether Iran helps the linear models (HARX_CI vs HARX_C, GARCHX_CI vs GARCHX_C), and the `rv`-vs-`rv_hourly` agreement. Do **not** decide the status of the extra models (benchmark, replacement or appendix); that is the user's call after seeing the results (checkpoint below).

- [ ] **Step 4: Verify counts.** (The cross-notebook HARX identity is checked in Task 10 Step 4.) Confirm that the number of `var` rows per model per split equals 313 / 365 / 45 and that M3's non-converged week count (from `fitlog_main.csv`) is printed in the notebook.

- [ ] **Step 5: Restart and run all again, then commit.**

  ```bash
  git add notebooks/06_evaluation.ipynb results/tables/vol_*.csv results/tables/ret_eval.csv results/tables/phigh_eval.csv results/tables/ljung_box.csv results/figures
  git commit -m "Evaluate OOS forecasts per RV measure: losses, MZ, DM, CW, GW, calibration and figures (notebook 06)"
  ```

> **CHECKPOINT A (Tue Dec 1, with the user).** Show: the headline M3 vs M2 row (QLIKE vs `rv_hourly`, test), the DM/CW/GW p-values, M3 vs HARX_CI and GARCHX_CI, the best linear model name, the instrument share, the holdout numbers, the number of tests run. Ask: (1) is the framing of the extra models ("benchmark", "replacement", "appendix") decided now or left for the report? (2) any table to drop before Linnéa sees it? Do not change any frozen setting after seeing these numbers.

---

### Task 7: Notebook 07 — events and fact checks (Tue Dec 1 → Wed Dec 2)

**Files:**
- Create: `notebooks/07_events.ipynb`, `results/frozen/deck_claims.csv` (only if the user supplies the claims)
- Output: `results/tables/{event_table,deck_fact_checks}.csv`, `results/figures/event_<id>.png` (9 Iran + 3 non-Iran = 12), `results/figures/events_grid.png`

**Interfaces:**
- Consumes: `events.event_table`, `events.fact_check_table`, `plots.event_window_plot`, `plots.events_grid`, `weekly.week_ending`, `oos.load_forecasts("main")`, `results/fits/insample_filtered_2005_2026.csv`, `data/raw/brent_daily.csv`, `data/raw/brent_intraday_daily.csv` (optional; absent if git-ignored), `config.EVENTS`, `config.NON_IRAN_EVENTS`.

- [ ] **Step 1: Event weeks.** Map every event date with `week_ending`. Assert Abqaiq (Sat 2019-09-14) → 2019-09-20 and Hormuz 2026 (Sat 2026-02-28) → 2026-03-06; assert every event week is in the panel.

- [ ] **Step 2: Build and save `event_table.csv`** with `events.event_table(C.EVENTS + [(i, d, i, "") for i, d in C.NON_IRAN_EVENTS], panel, fc, filt)`. Add a cell printing, per event, the M2 and M3 filtered P(H) and OOS π_H at k = −1, 0, +1, and the weekly QLIKE of M2, M3, HARX_CI at k = 0 (the MVP item, spec §12).

- [ ] **Step 3: Figures.** For each of the 12 events call `plots.event_window_plot(event_id, table_for_event)` and `save_fig(fig, f"event_{event_id}")`; then `events_grid` → `events_grid`. The 2026 event title and figure carry "holdout, not in the test". Dec 2011 and 2008 show only actuals (training window; markdown cell says so, spec §14 item 14).

- [ ] **Step 4: Fact checks** (spec §7.7 step 4; no network).
  1. Read the deck's slide 2 claims from the local, git-ignored deck `Plans/Brent-Iran_Pitch_Modern-Sage.pptx` **only if the user pastes them**: ask the user for the seven event moves printed on slide 2 and write them to `results/frozen/deck_claims.csv`. If the user has not supplied them by Wed noon, run without claims (computed moves for all nine Iran events) and list "deck claims not compared" in the notebook and the hand-off note.
  2. `events.fact_check_table(brent_daily, intraday_or_None, C.EVENTS, claims_or_None)` → `deck_fact_checks.csv`.
  3. Print: Brent close on Feb 27, 2026 → the 2026 peak close and date; the week to 2026-03-06 change and its rank among all Friday-to-Friday changes since 1987 (this is the deck's "largest weekly gain since 1987" claim: state TRUE or FALSE from the rank); the intraday CFD high next to the spot peak, labelled as a different instrument.

- [ ] **Step 5: Checks cell.** Every event week exists in the panel; holdout event marked in every output (`holdout` column True only for `hormuz_2026`); 12 `event_*.png` files exist; `paths.check_budget()`.

- [ ] **Step 6: Restart and run all; commit.**

  ```bash
  git add notebooks/07_events.ipynb results/tables/event_table.csv results/tables/deck_fact_checks.csv results/figures/event_*.png results/figures/events_grid.png results/frozen/deck_claims.csv
  git commit -m "Add event case studies, event table and deck fact checks (notebook 07)"
  ```

> **CHECKPOINT B (Wed Dec 2 evening): minimum viable version complete.** Verify against spec §11, line by line: (1) M1–M3 in notebooks 03–05 (Plan 2); (2) GARCH-t, HAR, HARX_CI, GARCHX_CI in the forecasts; (3) test and holdout on `rv_hourly` (Task 6); (4) QLIKE, MZ, DM incl. M3 vs HARX_CI and GARCHX_CI, Clark-West (Task 6); (5) the cumulative-loss figure (Task 6); (6) the event table with M2 vs M3 probabilities (Task 7). Tick each in the notebook's last markdown cell. If anything is missing, finish it before starting anything in notebook 08 beyond Task 10. Tell the user how many hours remain before the Thu evening freeze, and ask the user for the **OOS robustness budget in hours** (`BUDGET_HOURS`, used in Task 14).

---

### Task 8: Additive functions in Plan 2's modules (`design`, `forecast`, `ms`)

Three things are new in Plan 3; everything else robustness needs already exists in Plan 2 and is consumed by name (listed under Consumes). Nothing here changes an existing signature. Tasks 8–9b can start as soon as Plan 2's `ms.py`, `forecast.py` and `insample.py` exist (Mon Nov 30 at the latest), while the OOS loop is unattended.

**Files:**
- Modify: `src/brent/design.py` (`standardization_stats`), `src/brent/forecast.py` (add `mixture_params`), `src/brent/ms.py` (add `make_mshar_specs`)
- Test: `tests/test_design.py`, `tests/test_forecast.py`, `tests/test_ms.py` (append)

**Interfaces:**
- Consumes (Plan 2, by name): `MSSpec`, `MSFit` (with attribute `fixed`), `ms.make_specs`, `ms.make_model` (the returned model carries `readable`, the map from statsmodels parameter names to readable names of spec §9.4: `mu_N, mu_H, phi_N, phi_H, sigma2_N, sigma2_H, p00_<var>, p10_<var>`, and `theta_<var>` for mean regressors), `ms.fit_ms(design, spec, start_params, search_reps, n_restarts, seed, extra_starts=())`, `ms.nest_start(small: MSFit, big_model, big_spec) -> pd.Series`, `ms.constraint_info(spec, model)["fixed"]`, `insample.coef_table(fit, label) -> DataFrame` (columns include `var, logit, coef, se, z, p, model, fixed`; read the header once and note the exact `logit` strings), `insample.lr_test`, `design.build_design`, `forecast.predicted_probs`.
- Produces:
  - `design.standardization_stats`: any variable whose name starts with `dummy_` is **not** standardized (mean 0, sd 1 in `std_stats`, value passed through) **[plan call]**. Reason: the post-2013 dummy (check #2c) is constant 0 on the 2005–2012 window, so a z-score would divide by zero.
  - `forecast.mixture_params(fit: MSFit, design: Design) -> pd.DataFrame` indexed by `design.dates`, columns `pi_H, m_N, m_H, s2_N, s2_H`: the in-sample one-step mixture π(t|t−1) from `predicted_probs`, regime means mᵢ(t) = μᵢ + φᵢ·r(t−1) (+ θ·Iran(t−1) when `mean_exog`), σᵢ². Parameters are looked up through `fit.model.readable` (never by position). `kind="msar"` only.
  - `ms.make_mshar_specs(variant: str = C.M3_VARIANT) -> dict[str, MSSpec]`: `MSHAR_C` and `MSHAR_CI` for E1 (spec §7.8): `kind="mshar"`, `canonical_by="const"`, `phi_switching=False`, `std_window=C.STANDARDIZE_WINDOW_HOURLY`, `start=C.HOURLY_SAMPLE_START`, `tvtp_vars=("gpr","vix")` for C and `("gpr","vix","iran_gdelt")` for CI, `iran_vars=()` and `("iran_gdelt",)`, `iran_only_NH = (variant == "constrained")`.

- [ ] **Step 1: Write the failing tests** (fixtures from Task 0)

  ```python
  def test_dummy_columns_are_not_standardized():               # tests/test_design.py
      p = synthetic_panel(); p["dummy_post2013"] = (p["date"] >= C.DUMMY_POST2013_WEEK).astype(float)
      stats = D.standardization_stats(p, ["gpr", "dummy_post2013"], C.STANDARDIZE_WINDOW)
      assert stats.loc["dummy_post2013", "mean"] == 0 and stats.loc["dummy_post2013", "sd"] == 1
      spec = dataclasses.replace(M.make_specs(False)["M2"], tvtp_vars=("gpr", "dummy_post2013"))
      d = D.build_design(p, spec, "2019-12-27")
      lagged = p.set_index("date").loc[d.dates - pd.Timedelta(days=7), "dummy_post2013"].to_numpy()
      assert np.allclose(d.Z["dummy_post2013"].to_numpy(), lagged)     # raw 0/1, lagged one week

  def test_mixture_params_reproduces_loglike():                 # tests/test_forecast.py
      fit, design = fit_simulated_m1(seed=405, n=400)
      mp = F.mixture_params(fit, design)
      assert ((mp.pi_H >= 0) & (mp.pi_H <= 1)).all() and list(mp.index) == list(design.dates)
      from scipy.stats import norm
      dens = (1 - mp.pi_H) * norm.pdf(design.y, mp.m_N, np.sqrt(mp.s2_N)) + mp.pi_H * norm.pdf(design.y, mp.m_H, np.sqrt(mp.s2_H))
      assert abs(np.log(dens).sum() - fit.loglike) < 1e-6

  def test_make_mshar_specs_fields():                           # tests/test_ms.py
      s = M.make_mshar_specs("constrained")
      assert s["MSHAR_C"].kind == "mshar" and s["MSHAR_CI"].iran_vars == ("iran_gdelt",)
      assert s["MSHAR_CI"].canonical_by == "const" and s["MSHAR_CI"].start == C.HOURLY_SAMPLE_START
      assert s["MSHAR_CI"].iran_only_NH and not M.make_mshar_specs("iran_both_logits")["MSHAR_CI"].iran_only_NH
  ```

- [ ] **Step 2: Run `uv run pytest tests/test_design.py tests/test_forecast.py tests/test_ms.py -q`; expect the three new tests to FAIL.**
- [ ] **Step 3: Implement the three additions** per the Interfaces block.
- [ ] **Step 4: Run the full suite `uv run pytest -q`; expect all PASS** (including the existing `.shift(` grep test).
- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/design.py src/brent/forecast.py src/brent/ms.py tests/test_design.py tests/test_forecast.py tests/test_ms.py
  git commit -m "Add dummy passthrough, mixture params and MS-HAR specs for robustness"
  ```

---

### Task 9: `robustness.py` — run builders, variant fits, comparisons, placebo and budget helpers

**Files:**
- Create: `src/brent/robustness.py`, `tests/test_robustness.py`

**Interfaces:**
- Consumes: `ms` (`make_specs`, `with_iran`, `make_model`, `fit_ms`, `nest_start`, `make_mshar_specs`), `design.build_design`, `design.har_design`, `insample` (`lr_test`, `coef_table`, `circular_shift`, `fit_ladder`), `benchmarks.harx_forecast`, `oos` (`origins`, `run_oos`, `assemble`, `load_forecasts`), `evaluate` (`qlike`, `dm_test`, `clark_west`, `family`), `forecast.mixture_params`, `paths`.
- Produces:
  - `RESERVED_RUNS = {"main", "timing"}`; `guard_run_id(run_id: str) -> None` (raises `ValueError` for reserved ids); `guard_oos_targets(targets: pd.DatetimeIndex) -> None` (raises if any target is outside `C.ROBUST_OOS_TARGETS`)
  - `run_specs(run: str, phi_switching: bool, variant: str = C.M3_VARIANT) -> list[MSSpec]`; the **single place that defines every robustness model name** (table below)
  - `shifted_panel(panel, col: str, k: int, name: str, window=("2005-01-07", "2025-12-26")) -> pd.DataFrame` (copy with new column `name`: circularly shifted inside `window`, original values outside)
  - `fit_chain(panel, end, chain: list[tuple[MSSpec, str | None]], seed: int = C.SEED, parents: dict[str, MSFit] | None = None, n_restarts: int = C.N_RESTARTS) -> dict[str, MSFit]` (a parent name is looked up in `parents` (fits made elsewhere, e.g. `insample.fit_ladder`) and then among earlier chain entries. Each spec: `design.build_design`, `model = ms.make_model(design, spec)`, then `ms.fit_ms(design, spec, start_params=None, extra_starts=(ms.nest_start(parent, model, spec),), search_reps=C.SEARCH_REPS_FIRST, n_restarts=n_restarts, seed=seed)`; a spec with parent `None` gets no `extra_starts`)
  - `variant_row(name: str, fit: MSFit, parent: MSFit, beta_vars: tuple[str, ...]) -> dict` (`model, parent, n, loglike, df, lr_raw, stat, pvalue, negative, beta_<var>, se_<var>, z_<var>, p_<var>, converged, degenerate`; β row = `insample.coef_table(fit, name)` filtered to `var == <var>` and the N→H `logit`; `lr_raw, stat, df, pvalue, negative` copied from `insample.lr_test(parent, fit)`. Tables report `stat` and `lr_raw`)
  - `NESTED_PAIRS = {("M3_alt","M2_alt"), ("M3_aigpr","M2"), ("M3_geo","M2"), ("M3tone","M2"), ("M3_ovx","M2_ovx"), ("M3_hctrl","M2_hctrl"), ("MSHAR_CI","MSHAR_C")}` (each variant that adds Iran variables to the M2-type model on the **same sample**)
  - `oos_compare(fc: pd.DataFrame, pairs: list[tuple[str, str]], measure: str = C.MAIN_RV, targets=C.ROBUST_OOS_TARGETS) -> pd.DataFrame` (columns `model, ref, n, nested, qlike_model, qlike_ref, gain, dm_stat, dm_pvalue, cw_stat, cw_pvalue`; `gain = qlike_ref − qlike_model`, positive = model better; Clark-West on squared error (model = big, ref = small) **only for pairs in `NESTED_PAIRS`**, NaN otherwise)
  - `placebo_pvalue(actual: float, placebo: np.ndarray) -> float` = (1 + #{placebo ≥ actual}) / (B + 1)
  - `harx_gain(panel, iran_col: str, window: tuple[str, str], oos_targets: tuple[str, str], c_cache: dict | None = None) -> dict` (the optional `c_cache` is filled on the first call with the HARX_C forecasts and SSR_C and reused afterwards, so a placebo loop computes HARX_C once; `lr_insample` = n·ln(SSR_C/SSR_CI) from `design.har_design` on `window`; `qlike_gain_oos` = mean QLIKE(HARX_C) − mean QLIKE(HARX_CI) vs `rv_hourly` over `oos_targets`, using `benchmarks.harx_forecast` with `x_vars=C.X_CONTROLS` and `C.X_CONTROLS + (iran_col,)`)
  - `mixture_pdf(x, mp: pd.DataFrame) -> np.ndarray` (average predictive density: mean over rows of the mixture density at each `x`), `mixture_exceed(thr: float, mp: pd.DataFrame) -> float` (mean over rows of Σᵢ πᵢ P(|r| > thr | i))
  - `estimate_hours(fitlog_main: pd.DataFrame, models: list[str], n_origins: int = 365) -> float` (median `seconds` per origin of each model in the main fitlog, summed, × n_origins / 3600; the median understates the restart weeks, which is why Task 11 adds a 1.2 margin and the first fit is not included)
  - `CUT_ORDER = ["placebo", "tone", "ovx", "gdelt_geo", "aigpr", "hourly_ctrl"]`; `plan_oos_queue(hours: dict[str, float], budget_hours: float) -> tuple[list[str], list[str]]` (run, skipped; running order is the spec's `ar_alt, hourly_ctrl, aigpr, tone, gdelt_geo, ovx, placebo`; while the sum exceeds the budget, drop the next name in `CUT_ORDER`; `ar_alt` is never dropped) **[plan call: the spec's cut list names OOS placebo, then M3-tone OOS, then "AI-GPR / GDELT-geo / OVX / hourly control" as a group; the order inside that group is ours]**
  - `log_status(check: str, status: str, note: str, path: Path = paths.TABLES / "rob_status.csv") -> None` (upsert by `check`; `status` ∈ `done, cut, in_sample_only, failed`)

  Model names of every robustness run (the `model` column of `forecasts_<run>.csv`):

  | `run` | `run_specs` returns | Notes |
  |---|---|---|
  | `ar_alt` | `M1_alt, M2_alt, M3_alt` | `phi_switching = not frozen φ` |
  | `aigpr` | `M3_aigpr` | `with_iran(M3, ("aigpr_iran",), "M3_aigpr")` |
  | `gdelt_geo` | `M3_geo` | `("iran_gdelt_geo",)` |
  | `ovx` | `M2_ovx, M3_ovx` | `make_specs(controls=("gpr", "ovx", "log_rv"))`, `std_window=C.OVX_STD_WINDOW`, `start=C.OVX_START` |
  | `hourly_ctrl` | `M2_hctrl, M3_hctrl` | `make_specs(controls=C.TVTP_CONTROLS_HOURLY)`, `std_window=C.STANDARDIZE_WINDOW_HOURLY`, `start=C.HOURLY_SAMPLE_START` |
  | `tone` | `M3tone` | `with_iran(M3, C.IRAN_TONE_VARS, "M3tone")` |
  | `mshar` | `MSHAR_C, MSHAR_CI` | `ms.make_mshar_specs` |
  | `placebo_<k>` | `M3` | `with_iran(M3, ("iran_placebo",), "M3")`; run on `shifted_panel(..., name="iran_placebo")` |

- [ ] **Step 1: Write the failing tests** (`tests/test_robustness.py`; seed 405)

  ```python
  def test_guards():                                                   # Review focus 5
      for bad in ("main", "timing"):
          with pytest.raises(ValueError): R.guard_run_id(bad)
      R.guard_run_id("aigpr")
      R.guard_oos_targets(pd.date_range("2019-01-04", "2025-12-26", freq="W-FRI"))
      with pytest.raises(ValueError, match="2026"): R.guard_oos_targets(pd.date_range("2025-12-19", "2026-01-09", freq="W-FRI"))

  def test_shifted_panel_wraps_inside_window_only():                   # Review focus 5
      p = synthetic_panel(); q = R.shifted_panel(p, "iran_gdelt", 60, "iran_placebo")
      w = (p["date"] >= "2005-01-07") & (p["date"] <= "2025-12-26")
      assert np.allclose(q.loc[~w, "iran_placebo"], p.loc[~w, "iran_gdelt"])
      assert np.allclose(np.sort(q.loc[w, "iran_placebo"]), np.sort(p.loc[w, "iran_gdelt"]))   # a permutation
      assert (q.loc[w, "iran_placebo"].to_numpy() != p.loc[w, "iran_gdelt"].to_numpy()).any()
      assert "iran_placebo" not in p.columns                          # input not mutated

  def test_placebo_pvalue_exact_and_ties_count():
      assert R.placebo_pvalue(5.0, np.array([1., 2., 6.])) == 0.5
      assert R.placebo_pvalue(5.0, np.array([5., 5.])) == 1.0
      assert R.placebo_pvalue(5.0, np.array([1., 2., 3., 4.])) == 0.2

  def test_run_specs_names_and_fields():
      s = {r: R.run_specs(r, phi_switching=True) for r in ("ar_alt", "aigpr", "gdelt_geo", "ovx", "hourly_ctrl", "tone", "mshar")}
      assert [x.name for x in s["ar_alt"]] == ["M1_alt", "M2_alt", "M3_alt"] and all(x.phi_switching is False for x in s["ar_alt"])
      assert s["aigpr"][0].iran_vars == ("aigpr_iran",) and s["gdelt_geo"][0].iran_vars == ("iran_gdelt_geo",)
      assert s["ovx"][1].tvtp_vars[:3] == ("gpr", "ovx", "log_rv") and s["ovx"][1].start == C.OVX_START
      assert s["hourly_ctrl"][0].std_window == C.STANDARDIZE_WINDOW_HOURLY and "log_rv_hourly" in s["hourly_ctrl"][1].tvtp_vars
      assert s["tone"][0].iran_vars == C.IRAN_TONE_VARS and [x.name for x in s["mshar"]] == ["MSHAR_C", "MSHAR_CI"]

  def test_oos_compare_cw_only_for_nested_pairs():
      fc = toy_fc_two_models(f_a=2.0, f_b=1.0, rv=2.0)
      r = R.oos_compare(fc, [("A", "B")], measure="rv").iloc[0]
      assert not r["nested"] and np.isnan(r["cw_stat"]) and np.isnan(r["cw_pvalue"])
      assert ("M3_ovx", "M2") not in R.NESTED_PAIRS and ("M3_ovx", "M2_ovx") in R.NESTED_PAIRS

  def test_oos_compare_exact_gain():
      fc = toy_fc_two_models(f_a=2.0, f_b=1.0, rv=2.0)     # model "A" forecasts the truth, "B" half of it, 30 weeks in 2019
      r = R.oos_compare(fc, [("A", "B")], measure="rv").iloc[0]
      assert np.isclose(r["qlike_model"], 0) and np.isclose(r["qlike_ref"], 1 - np.log(2)) and np.isclose(r["gain"], 1 - np.log(2))
      assert r["n"] == 30

  def test_harx_gain_matches_manual_ols_and_is_positive_when_iran_matters():
      p = planted_hourly_panel(seed=405)                    # rv_hourly depends on iran_gdelt(t-1)
      g = R.harx_gain(p, "iran_gdelt", window=("2006-11-24", "2012-12-28"), oos_targets=("2013-01-04", "2013-12-27"))
      X0, y0, _ = D.har_design(p, "2012-12-28", "rv_hourly", C.X_CONTROLS, C.HOURLY_SAMPLE_START)
      X1, y1, _ = D.har_design(p, "2012-12-28", "rv_hourly", C.X_CONTROLS + ("iran_gdelt",), C.HOURLY_SAMPLE_START)
      ssr = lambda X, y: float(((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum())
      assert np.isclose(g["lr_insample"], len(y0) * np.log(ssr(X0, y0) / ssr(X1, y1)))
      assert g["lr_insample"] > 0 and g["qlike_gain_oos"] > 0

  def test_mixture_helpers_exact():
      mp = pd.DataFrame({"pi_H": [0.5], "m_N": [0.0], "m_H": [0.0], "s2_N": [1.0], "s2_H": [1.0]})
      assert np.isclose(R.mixture_pdf(np.array([0.0]), mp)[0], 0.3989422804)
      assert np.isclose(R.mixture_exceed(1.0, mp), 0.3173105079)

  def test_plan_oos_queue_cut_order_and_ar_alt_is_kept():
      h = {"ar_alt": 1.0, "hourly_ctrl": 1.0, "aigpr": 1.0, "tone": 1.0, "gdelt_geo": 1.0, "ovx": 1.0, "placebo": 8.0}
      run, skipped = R.plan_oos_queue(h, budget_hours=7.0)
      assert skipped == ["placebo"] and run == ["ar_alt", "hourly_ctrl", "aigpr", "tone", "gdelt_geo", "ovx"]
      run, skipped = R.plan_oos_queue(h, budget_hours=3.0)
      assert run == ["ar_alt", "hourly_ctrl", "aigpr"] and skipped == ["placebo", "tone", "ovx", "gdelt_geo"]
      assert R.plan_oos_queue(h, budget_hours=0.0)[0] == ["ar_alt"]

  def test_estimate_hours_and_status_log(tmp_path):
      fl = pd.DataFrame({"model": ["M3"] * 3, "seconds": [2.0, 4.0, 3.0]})
      assert np.isclose(R.estimate_hours(fl, ["M3"], n_origins=3600), 3.0)         # median 3 s x 3600 / 3600
      f = tmp_path / "s.csv"; R.log_status("aigpr", "cut", "no time", path=f); R.log_status("aigpr", "done", "ok", path=f)
      t = pd.read_csv(f); assert len(t) == 1 and t.loc[0, "status"] == "done"

  def test_variant_row_on_nested_fit():
      m2, m3, design, panel = fit_m2_m3_simulated(seed=405)
      r = R.variant_row("M3", m3, m2, ("iran_gdelt",))
      assert r["df"] == 1 and r["lr_raw"] >= -1e-6 and r["stat"] >= 0 and r["beta_iran_gdelt"] > 0

  def test_fit_chain_uses_nesting_start_and_never_loses_loglike():
      m2, m3, design, panel = fit_m2_m3_simulated(seed=405)
      out = R.fit_chain(panel, design.dates[-1], [(m3.spec, "M2")], parents={"M2": m2})
      assert out[m3.spec.name].loglike >= m2.loglike - 1e-6
  ```

  In `test_plan_oos_queue`, the running order is the spec's; the first test expects `placebo` (largest, first in `CUT_ORDER`) dropped once the sum 14 exceeds 7; the second drops placebo, tone, ovx, gdelt_geo in that order until 3.0 remains.

- [ ] **Step 2: Run; expect FAIL.**

- [ ] **Step 3: Implement.** `harx_gain`'s OOS loop runs one OLS pair per target week: for each target `s` in `oos_targets`, origin = `s − 7 days`, `benchmarks.harx_forecast(panel, origin, "rv_hourly", x_vars, start=C.HOURLY_SAMPLE_START)["var_fc"]`. `mixture_pdf` and `mixture_exceed` use `scipy.stats.norm`. `estimate_hours` takes the median of `seconds` per model. `oos_compare` filters `target == "var"`, restricts to `targets`, joins on `date`, and calls `guard_oos_targets` first. Everything else follows the Interfaces block.

- [ ] **Step 4: Run `uv run pytest tests/test_robustness.py -q` then `uv run pytest -q`; expect PASS.**

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/robustness.py tests/test_robustness.py
  git commit -m "Add robustness helpers: run builders, variant fits, OOS comparison, placebo p-values, queue planner"
  ```

---

### Task 9b: `insample.placebo_*` — placebo fits (Plan 2's module, additive)

Spec §8.9 lists `placebo_lr` and `placebo_harx` in `insample.py`, but no plan builds them. Plan 3 does, because they need `robustness.harx_gain` and the offsets rule of spec §7.8.

**Files:**
- Modify: `src/brent/insample.py`; test: `tests/test_insample.py` (append)

**Interfaces:**
- Consumes: `insample.circular_shift`, `insample.lr_test`, `insample.fit_ladder`, `ms.make_model`, `ms.fit_ms(..., extra_starts=...)`, `ms.nest_start`, `design.build_design`, `ms.with_iran`, `robustness.harx_gain` (imported **inside** `placebo_harx` to avoid an import cycle).
- Produces:
  - `placebo_offsets(n_shifts: int, seed: int, n_weeks: int, min_shift: int = C.PLACEBO_MIN_SHIFT) -> np.ndarray` (integers drawn without replacement, uniformly, from `[min_shift, n_weeks − min_shift]`, `np.random.default_rng(seed)`; so every shift is at least 52 weeks from 0 in both directions)
  - `placebo_fit(panel, m2: MSFit, k: int, window: tuple[str, str], seed: int) -> float` (log-likelihood of M3 with `iran_gdelt` replaced by `circular_shift(panel["iran_gdelt"], k, window)` inside `window`, original values outside; fitted on `end = window[1]`; start from M2 via `extra_starts=(ms.nest_start(m2, model, spec),)`, `n_restarts=C.N_RESTARTS`, `search_reps=C.SEARCH_REPS_FIRST`; `k = 0` is the unshifted fit)
  - `placebo_lr(panel, n_shifts: int, seed: int, window: tuple[str, str] = ("2005-01-07", "2025-12-26")) -> pd.DataFrame` with columns `k, ll_m3, lr_raw` (`lr_raw = 2·(ll_m3 − ll_m2)`; M2 fitted once on `window`, shifts from `placebo_offsets(n_shifts, seed, n_weeks_in_window)`)
  - `placebo_harx(panel, n_shifts, seed, window=(C.HOURLY_SAMPLE_START, "2025-12-26"), oos_targets=("2019-01-04", "2025-12-26")) -> pd.DataFrame` with columns `k, lr_insample, qlike_gain_oos`: one call to `robustness.harx_gain` per offset on a panel whose `iran_gdelt` column is the shifted series, sharing one `c_cache` so HARX_C forecasts are computed once; the same offsets serve the in-sample and OOS gains

- [ ] **Step 1: Write the failing tests**

  ```python
  def test_offsets_respect_min_shift_and_seed():
      a = I.placebo_offsets(50, 405, 1100); b = I.placebo_offsets(50, 405, 1100)
      assert (a == b).all() and len(set(a)) == 50 and a.min() >= 52 and a.max() <= 1100 - 52
      assert not (a == I.placebo_offsets(50, 406, 1100)).all()

  def test_unshifted_placebo_reproduces_actual_ll_and_lr():
      m2, m3, design, panel = fit_m2_m3_simulated(seed=405)
      ll0 = I.placebo_fit(panel, m2, 0, (str(design.dates[0].date()), str(design.dates[-1].date())), seed=405)
      assert abs(ll0 - m3.loglike) < 1e-4

  def test_placebo_lr_ignores_post_window_values_and_is_nonnegative_up_to_tol():   # review focus 5
      m2, m3, design, panel = fit_m2_m3_simulated(seed=405); w = (str(design.dates[0].date()), str(design.dates[-1].date()))
      a = I.placebo_lr(panel, 2, 405, w)
      later = panel.copy(); later.loc[later["date"] > w[1], "iran_gdelt"] += 10.0
      b = I.placebo_lr(later, 2, 405, w)
      assert np.allclose(a["lr_raw"], b["lr_raw"]) and list(a.columns) == ["k", "ll_m3", "lr_raw"] and (a["lr_raw"] > -1e-4).all()

  def test_placebo_harx_unshifted_equals_actual_and_post_window_never_moves():
      p = planted_hourly_panel(seed=405); w = (C.HOURLY_SAMPLE_START, "2012-12-28"); tg = ("2013-01-04", "2013-12-27")
      actual = R.harx_gain(p, "iran_gdelt", w, tg)
      out = I.placebo_harx(p, 3, 405, window=w, oos_targets=tg)
      assert list(out.columns) == ["k", "lr_insample", "qlike_gain_oos"] and len(out) == 3
      assert out["lr_insample"].max() < actual["lr_insample"]          # planted effect beats shifted ones
      later = p.copy(); later.loc[later["date"] > w[1], "iran_gdelt"] += 10.0
      assert np.allclose(I.placebo_harx(later, 3, 405, window=w, oos_targets=tg)["lr_insample"], out["lr_insample"])
  ```
  (`test_placebo_harx_...` uses a window that ends before the OOS targets; the OOS gain there uses the shifted column through the origins, so the "never moves" assertion is made on `lr_insample` only.)

- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** the four functions per the Interfaces block (`placebo_lr` and `placebo_fit` seeds: `seed + i` for the i-th shift, so a rerun is exact).
- [ ] **Step 4: Run `uv run pytest tests/test_insample.py tests/test_robustness.py -q`; expect PASS.**
- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/insample.py tests/test_insample.py
  git commit -m "Add placebo_offsets, placebo_fit, placebo_lr and placebo_harx"
  ```

---

### Task 10: Notebook 08, part A — setup and the first four in-sample checks (Wed Dec 2)

Order of work (spec §7.8): placebo #1 → HARX placebo #9 → hourly control #7 → AI-GPR #2. In-sample runs use 2005–2025 (`END = C.RQ2_END`) and need no frozen OOS run, only a frozen `PHI_SWITCHING`.

**Files:**
- Create: `notebooks/08_robustness.ipynb`
- Output: `results/tables/{placebo_insample,placebo_harx,rob_hourly_ctrl_insample,rob_aigpr_insample,rob_status}.csv`, `results/figures/{placebo_hist,placebo_harx_hist}.png`

**Interfaces:**
- Consumes: `insample.fit_ladder`, `insample.placebo_lr`, `insample.placebo_harx`, `insample.lr_test`, Task 9 (`robustness.*`), `insample.coef_table`, notebook 04's `results/tables/{rq2_lr_tests,harx_ftest}.csv` (columns fixed by the cross-plan contract, Global Constraints), notebook 06's `vol_losses.csv`.

- [ ] **Step 1: Setup cell.**
  - Imports as in the Global Constraints; flags `FORCE_PLACEBO, FORCE_HARX_PLACEBO, FORCE_HCTRL, FORCE_AIGPR, FORCE_TONE, FORCE_AR, FORCE_DIST, FORCE_MEAN, FORCE_OVX, FORCE_GEO, FORCE_DUMMY, FORCE_OOS, FORCE_E1`.
  - `assert C.PHI_SWITCHING is not None` and that `rq1_decision.json` agrees (same rule as notebooks 04–05).
  - `INSAMPLE_ONLY = not (C.SPEC_FROZEN and (paths.FORECASTS / "forecasts_main.csv").exists())`; `MVP_MODE` set by hand to True if the full run missed Mon Nov 30 (spec §11): then only placebo #1, HARX placebo #9 and in-sample AI-GPR run, and every other check gets `log_status(check, "cut", "minimum-viable mode")`.
  - Each section below skips when its output exists and its `FORCE_` flag is False. Each finished or cut section calls `robustness.log_status`.

- [ ] **Step 2: Main ladder once.** `ladder = insample.fit_ladder(panel, END, C.PHI_SWITCHING, n_restarts=C.N_RESTARTS)` (M1, M2, M3 on 2005–2025). Assert its M3-vs-M2 `lr_raw` equals `lr_raw` of the row with `model_restricted == "M2"` and `model_full == "M3"` in notebook 04's `rq2_lr_tests.csv` to 1e-6 (**this is the "same code path" check of spec §7.8**; if it fails, stop and tell the user: the placebo p-values would be meaningless).

- [ ] **Step 3: Placebo #1, in sample** (`FORCE_PLACEBO`).
  1. `pl = insample.placebo_lr(panel, C.PLACEBO_IN, C.SEED)` (Task 9b; 200 shifts; columns `k, ll_m3, lr_raw`).
  2. Actual = `lr_test(ladder["M2"], ladder["M3"])["lr_raw"]`; `p = robustness.placebo_pvalue(actual, pl["lr_raw"].to_numpy())` (raw, unclipped LRs on both sides).
  3. Save `placebo_insample.csv` (the 200 rows plus a final row `k = 0, actual = True`). Figure `placebo_hist.png` through `plots.placebo_hist({"in-sample LR gain": (lr_array, actual)}, "LR, M3 vs M2")`; Task 13 adds the OOS panel to the same figure.
  4. Print the count of degenerate placebo fits (Plan 2 logs them); if more than 10%, say so in the markdown.

- [ ] **Step 4: HARX placebo #9** (`FORCE_HARX_PLACEBO`).
  1. `ph = insample.placebo_harx(panel, C.PLACEBO_IN, C.SEED)` (Task 9b; columns `k, lr_insample, qlike_gain_oos`).
  2. Actual: `robustness.harx_gain(panel, "iran_gdelt", (C.HOURLY_SAMPLE_START, C.RQ2_END), ("2019-01-04", "2025-12-26"))`.
  3. Two p-values via `placebo_pvalue` (in-sample LR; OOS QLIKE gain). Save `placebo_harx.csv`, `placebo_harx_hist.png` (two panels).
  4. **Identity checks (spec §7.8):**
     - the actual `lr_insample` and notebook 04's `harx_ftest.csv` row `test == "HARX_CI_vs_HARX_C"` satisfy F = (exp(LR/n) − 1)·`df_resid`/`df_num` with n = the row's `n`, to 1e-6;
     - the actual `qlike_gain_oos` equals mean QLIKE(HARX_C) − mean QLIKE(HARX_CI) on (`test`, `rv_hourly`) from notebook 06's `vol_losses.csv` to 1e-9.

     A mismatch stops the notebook (it means a different sample or lag convention).

- [ ] **Step 5: Hourly lagged control #7** (`FORCE_HCTRL`, in sample).
  1. `specs = robustness.run_specs("hourly_ctrl", C.PHI_SWITCHING)`; fits = `robustness.fit_chain(panel, END, [(specs[0], None), (specs[1], "M2_hctrl")])`.
  2. Table `rob_hourly_ctrl_insample.csv`: `robustness.variant_row` for `M3_hctrl` vs `M2_hctrl`, plus, for the comparison, the main `M3` row (vs `M2`) from `ladder`. Add the `log_rv_hourly` N→H and H→N coefficients from `insample.coef_table(fit, name)` (filter `var == "log_rv_hourly"`).
  3. Assert the sample starts at `C.HOURLY_SAMPLE_START` + 1 week and print n.

- [ ] **Step 6: AI-GPR #2** (`FORCE_AIGPR`, in sample).
  1. `assert panel["aigpr_iran"].notna().all()` for 2005–2025 (availability rule of spec §9.1); if not, print the weeks and stop.
  2. `fits = robustness.fit_chain(panel, END, [(run_specs("aigpr", PHI)[0], "M2")], parents={"M2": ladder["M2"]})`.
  3. `rob_aigpr_insample.csv`: `variant_row("M3_aigpr", ..., ("aigpr_iran",))` and the main M3 row; markdown: sign and significance of β under both Iran measures.

- [ ] **Step 7: Status and fit health.** Each section ends with `log_status(..., "done", note)`. A cell prints, for every fit made in this part, `converged` and `degenerate`; any False/True goes into the section's markdown.

- [ ] **Step 8: Commit** (notebook outputs are committed at the Task 15 freeze; here commit tables and figures only).

  ```bash
  git add notebooks/08_robustness.ipynb results/tables/placebo_insample.csv results/tables/placebo_harx.csv results/tables/rob_hourly_ctrl_insample.csv results/tables/rob_aigpr_insample.csv results/tables/rob_status.csv results/figures/placebo_hist.png results/figures/placebo_harx_hist.png
  git commit -m "Robustness part A: in-sample placebo, HARX placebo, hourly control, AI-GPR (notebook 08)"
  ```

---

### Task 11: Queue the OOS robustness runs (Wed Dec 2, evening)

**Files:**
- Modify: `notebooks/08_robustness.ipynb` (section "OOS runs")
- Create (by running): `results/forecasts/forecasts_{ar_alt,hourly_ctrl,aigpr,tone,gdelt_geo,ovx,placebo_<k>}.csv`, `results/fits/{params,fitlog}_<run>.csv`

**Interfaces:**
- Consumes: `robustness.{run_specs, estimate_hours, plan_oos_queue, guard_run_id, guard_oos_targets, shifted_panel, log_status}`, `oos.{origins, run_oos, assemble}`, `results/fits/fitlog_main.csv`, `results/tables/placebo_insample.csv` (the shifts `k`).
- Runs reuse the frozen settings in `spec_freeze.json`; each starts at origin 2018-12-28 with a fresh first fit (full first-fit procedure, `SEARCH_REPS_FIRST`) and ends at target **2025-12-26** **[plan call: the spec gives no end date; stopping before 2026 keeps the holdout out of every robustness run]**. They never rerun validation.

- [ ] **Step 1: Gate.** If `INSAMPLE_ONLY` (spec not frozen, or `forecasts_main.csv` missing): `log_status(run, "in_sample_only", "spec not frozen")` for every OOS run and stop this task (spec §11). The notebook then states that OOS robustness is a stated limitation.

- [ ] **Step 2: Budget.** Ask the user for `BUDGET_HOURS` (from Checkpoint B). Compute:
  - `hours = {run: 1.2 * robustness.estimate_hours(fitlog_main, models_of(run), 365)}` where `models_of` maps `ar_alt → [M1, M2, M3]`, `aigpr, gdelt_geo, tone → [M3]`, `ovx, hourly_ctrl → [M2, M3]`, `placebo → [M3] × C.PLACEBO_OOS` (the main-run median seconds per origin per model; the 1.2 is a margin for the first fit **[plan call]**).
  - `run, skipped = robustness.plan_oos_queue(hours, BUDGET_HOURS)`. Print both lists and the hours; log every skipped run as `cut` with the reason "budget".
  - If the placebo is skipped only because it is serial and `joblib` would fit it (hours / available cores ≤ remaining budget), say so to the user; with their yes, `uv add joblib` and run the 20 shifts with `Parallel(n_jobs=...)`, each shift its own `run_id`.

- [ ] **Step 3: Launch the queue** in the spec's order (`ar_alt, hourly_ctrl, aigpr, tone, gdelt_geo, ovx`, then placebo). For each run in `run`:
  1. `robustness.guard_run_id(run)`; `origins = oos.origins(*C.ROBUST_OOS_TARGETS)`; `robustness.guard_oos_targets(...)` on their targets.
  2. `specs = robustness.run_specs(run, C.PHI_SWITCHING)`.
  3. Skip if `results/forecasts/forecasts_<run>.csv` exists and `FORCE_OOS` is False; else `oos.run_oos(panel, run, specs, benchmarks=(), origins=origins, force=FORCE_OOS)` then `oos.assemble(run)`.
  4. Placebo: for each of the first `C.PLACEBO_OOS` shifts `k` of `placebo_insample.csv` (rows with `actual == False`; the file carries one extra row `k = 0, actual = True`), `pp = robustness.shifted_panel(panel, "iran_gdelt", k, "iran_placebo")`, specs = `run_specs("placebo", ...)`, `run_id = f"placebo_{k}"`.
  5. After each run: `log_status(run, "queued_done", f"{seconds/3600:.2f} h")` and print the run's fitlog summary (converged share, degenerate count, restarts, `refit_from_scratch`).
- The loop is resumable: an interrupted queue restarts at the last completed origin (checkpoint + settings hash, spec §8.8). Leave it running overnight.

- [ ] **Step 4: Morning check (Thu Dec 3).** Every launched run has a forecasts file with 365 `var` rows per model (the placebo runs: M3 only); no run's `date` exceeds 2025-12-26; no row in `forecasts_main.csv` changed (`git diff --stat results/forecasts/forecasts_main.csv` is empty).

- [ ] **Step 5: Commit the forecasts and fit logs.**

  ```bash
  git add results/forecasts results/fits notebooks/08_robustness.ipynb results/tables/rob_status.csv
  git commit -m "Run OOS robustness checks from origin 2018-12-28 with frozen settings"
  ```
  If the combined size nears the budget, commit only `forecasts_*.csv` for runs that are reported and leave other `params_*.csv` out (`paths.check_budget()` decides).

---

### Task 12: Notebook 08, part B — the remaining in-sample checks (Thu Dec 3)

Order of work (spec §7.8): tone variants #8 → AR term #3 → distribution #4 → Iran in the mean #5 → OVX #6 → GDELT geo #2b and post-2013 dummy #2c. Each section is independent, skips on its own `FORCE_` flag and logs its status. Anything unfinished when the Thu Dec 3 evening freeze comes is logged as `cut` and becomes a stated limitation.

**Files:**
- Modify: `notebooks/08_robustness.ipynb`
- Output: `results/tables/{rob_tone_insample,rob_ar_insample,rob_tails,rob_iran_mean,rob_ovx_insample,rob_gdelt_geo_insample,rob_post2013}.csv`, `results/figures/rob_distribution.png`

**Interfaces:**
- Consumes: `insample.fit_tone`, `insample.fit_ladder`, `insample.coef_table`, `forecast.mixture_params`, `ms.param_cov`, `robustness.{fit_chain, variant_row, run_specs, mixture_pdf, mixture_exceed}`, `design.build_design`, `dataclasses.replace`.

- [ ] **Step 1: Tone variants #8** (`FORCE_TONE`). `tone = insample.fit_tone(panel, END, PHI, ladder, n_restarts=C.N_RESTARTS)` gives `M3tone, M3toneL, M3gold`. `rob_tone_insample.csv` contains **only** `M3toneL` and `M3gold` (M3tone is in notebook 04): β's, SEs, z (deck sign), `lr_raw`, `stat`, `pvalue` vs `M3` with df from free parameters (printed, never typed). For `M3toneL` (M3 + `iran_tone`) and `M3gold` (M3 + `iran_goldstein`) the added variable is also N→H only. Markdown: collinearity warning from notebook 02's correlations (spec R20).

- [ ] **Step 2: AR term #3** (`FORCE_AR`). `alt = insample.fit_ladder(panel, END, not C.PHI_SWITCHING, n_restarts=C.N_RESTARTS)`. `rob_ar_insample.csv`: for M1, M2, M3 under both φ settings: log-likelihood, number of free parameters, AIC, BIC, expected durations, and for M3 the β and LR vs M2. Markdown: does the Iran result survive the other φ setting?

- [ ] **Step 3: Distribution #4** (`FORCE_DIST`; Student-t is cut first, logged `cut` with the note "stated as a limitation").
  1. `mp = forecast.mixture_params(ladder["M3"], design.build_design(panel, ladder["M3"].spec, END))` and the same for `M1`.
  2. Figure `rob_distribution.png`: histogram of weekly returns on the estimation sample with the model's average one-step predictive density `robustness.mixture_pdf(grid, mp)` for M1 and M3, and a single normal with the sample mean and SD, log y-axis in a second panel for the tails.
  3. `rob_tails.csv`: for each percentile in `C.TAIL_PCTLS` (90, 95, 99 of |r| on **TRAIN**), the threshold, the empirical P(|r| > thr) on the estimation sample, and the model-implied `robustness.mixture_exceed(thr, mp)` for M1 and M3 and the single normal.
  4. Markdown: whether two normals fit the tails; Student-t errors are a limitation.

- [ ] **Step 4: Iran in the mean #5** (`FORCE_MEAN`). `spec = dataclasses.replace(ladder["M3"].spec, name="M3_mean", mean_exog=("iran_gdelt",))` (one common, non-switching coefficient). `fits = robustness.fit_chain(panel, END, [(spec, "M3")], parents={"M3": ladder["M3"]})`. `rob_iran_mean.csv`: the Iran mean coefficient θ and SE (the parameter `theta_iran_gdelt`, which Plan 2's `make_model` gives mean regressors as a non-switching coefficient), `lr_raw`/`stat`/`pvalue` vs M3 with df = 1 (assert), and β (N→H) with and without the mean term. If `theta_iran_gdelt` is missing from `fit.params.index`, stop and report it as a Plan 2 gap; do not work around it.

- [ ] **Step 5: OVX #6** (`FORCE_OVX`, in sample).
  1. Assert the first week with a non-NaN `ovx` equals `C.OVX_START`; if not, set `OVX_START` and `OVX_STD_WINDOW[0]` in `config.py` to that week, rerun, and tell the user.
  2. `specs = robustness.run_specs("ovx", PHI)`; `fit_chain(panel, END, [(specs[0], None), (specs[1], "M2_ovx")])`.
  3. `rob_ovx_insample.csv`: β and LR M3_ovx vs M2_ovx, sample size, with the main M3 row for reference (different sample: say so).

- [ ] **Step 6: GDELT geo #2b and post-2013 dummy #2c** (`FORCE_GEO`, `FORCE_DUMMY`).
  1. Geo: `fit_chain(panel, END, [(run_specs("gdelt_geo", PHI)[0], "M2")], parents={"M2": ladder["M2"]})` → `rob_gdelt_geo_insample.csv` (β, LR; main M3 row alongside).
  2. Dummy: `panel["dummy_post2013"] = (panel["date"] >= C.DUMMY_POST2013_WEEK).astype(float)`; `M2d = replace(ladder["M2"].spec, name="M2_d", tvtp_vars=ladder["M2"].spec.tvtp_vars + ("dummy_post2013",))`; `M3d` likewise from `ladder["M3"].spec`, dummy in **both** logits (it is an unrestricted control, so only the Iran variable is restricted). Chain: `M2_d` parent `"M2"`, `M3_d` parent `"M2_d"`. `rob_post2013.csv`: β (Iran) with and without the dummy, the dummy's coefficients in both logits (`insample.coef_table`, filter `var == "dummy_post2013"`), `lr_raw`/`stat`/`pvalue` M3_d vs M2_d. Print the descriptive OLS and 52-week before/after means from notebook 02 next to it.

- [ ] **Step 7: Markdown synthesis cell** for part B: one paragraph per check in plain words (what changed in β and LR compared with the main M3). Append to `rob_status.csv` via `log_status`.

- [ ] **Step 8: Commit.**

  ```bash
  git add notebooks/08_robustness.ipynb results/tables/rob_*.csv results/tables/rob_status.csv results/figures/rob_distribution.png
  git add src/brent/config.py   # only if Step 5.1 changed OVX_START
  git commit -m "Robustness part B: tone, AR term, distribution, Iran in mean, OVX, geo, post-2013 dummy"
  ```

---

### Task 13: Evaluate the OOS robustness runs and the OOS placebos (Thu Dec 3)

**Files:**
- Modify: `notebooks/08_robustness.ipynb`
- Output: `results/tables/{rob_ar_oos,rob_aigpr_oos,rob_gdelt_geo_oos,rob_ovx_oos,rob_hourly_ctrl_oos,rob_tone_oos,placebo_oos,rob_fit_health}.csv`; `results/figures/placebo_hist.png` (rewritten with both panels)

**Interfaces:**
- Consumes: `oos.load_forecasts(run, with_actuals=True)`, `robustness.{oos_compare, placebo_pvalue, log_status}`, `evaluate.*`, `plots.placebo_hist`.

- [ ] **Step 1: Build the comparison frame** per run: `fc = pd.concat([main_fc, run_fc])` (model names are distinct across runs except the placebo runs, whose model is named `M3`; placebo runs are loaded one at a time and the main M3 is renamed `M3_main` before any concat). Use `measure = C.MAIN_RV`, targets 2019-01-04 → 2025-12-26 (the test window; the holdout is not used by any robustness run).

- [ ] **Step 2: Tables** via `robustness.oos_compare` (columns as in Task 9; `gain` positive = the variant is better):

  | Run | Pairs `(model, ref)` | File |
  |---|---|---|
  | `ar_alt` | `(M1_alt, M1)`, `(M2_alt, M2)`, `(M3_alt, M3)`, `(M3_alt, M2_alt)` | `rob_ar_oos.csv` |
  | `aigpr` | `(M3_aigpr, M2)`, `(M3_aigpr, M3)` | `rob_aigpr_oos.csv` |
  | `gdelt_geo` | `(M3_geo, M2)`, `(M3_geo, M3)` | `rob_gdelt_geo_oos.csv` |
  | `ovx` | `(M3_ovx, M2_ovx)`, `(M3_ovx, M2)`, `(M3_ovx, M3)` | `rob_ovx_oos.csv` |
  | `hourly_ctrl` | `(M3_hctrl, M2_hctrl)`, `(M3_hctrl, M3)`, `(M2_hctrl, M2)` | `rob_hourly_ctrl_oos.csv` |
  | `tone` | `(M3tone, M2)`, `(M3tone, M3)` | `rob_tone_oos.csv` |

  Add `family = "robustness"`. Markdown: whether the main result (M3 better than M2 or not) survives each variant; all secondary, nothing adjusted.

- [ ] **Step 3: OOS placebo** (if the `placebo_<k>` runs exist). For each `k`: gain_k = QLIKE(M2 main) − QLIKE(M3 of run `placebo_<k>`) vs `rv_hourly`, targets 2019–2025. Actual gain = the same difference with main M3; assert it equals `vol_losses.csv` (test, `rv_hourly`, M2 minus M3) to 1e-9. `p = placebo_pvalue(actual_gain, gains)` (B = number of runs found; print it). Save `placebo_oos.csv`; rewrite `placebo_hist.png` with both panels (in-sample LR and OOS QLIKE gain).

- [ ] **Step 4: Fit health table** `rob_fit_health.csv`: one row per run and model: weeks, converged share, non-converged count, degenerate count, `refit_from_scratch` count, median seconds. Printed and cited in the markdown ("each robustness run's fitlog convergence printed", spec §7.8 checks).

- [ ] **Step 5: Status.** `log_status(run, "done", ...)` for each evaluated run; runs whose files are missing or that did not fit in the budget stay `cut` or `in_sample_only` (already logged in Task 11).

- [ ] **Step 6: Commit.**

  ```bash
  git add notebooks/08_robustness.ipynb results/tables/rob_*_oos.csv results/tables/placebo_oos.csv results/tables/rob_fit_health.csv results/tables/rob_status.csv results/figures/placebo_hist.png
  git commit -m "Evaluate OOS robustness runs and OOS placebos"
  ```

---

### Task 14a: E1 module branches in Plan 2's modules (gated, first to cut)

Plan 2 raises `NotImplementedError` for `kind="mshar"` (design, canonicalize, forecast, oos). E1 therefore needs these branches. **Gate:** the same gate as Task 14 (below). Before writing anything, run the smoke test of Task 14 Step 1; if Plan 2 has by then implemented any of the branches, skip that branch and keep its test. All steps are "add a branch": no existing signature changes.

**Files:**
- Modify: `src/brent/design.py` (`build_design`), `src/brent/ms.py` (`canonicalize`), `src/brent/forecast.py` (`one_step`; new pure helper `mshar_var_fc`), `src/brent/oos.py` (`run_oos`)
- Test: `tests/test_design.py`, `tests/test_ms.py`, `tests/test_forecast.py`, `tests/test_oos.py` (append)

**Interfaces:**
- Consumes: spec §8.4–8.6 (`MSSpec.kind = "mshar"`, `canonical_by = "const"`), `ms.make_mshar_specs` (Task 8).
- Produces:
  - `design.build_design(panel, spec, end)` for `spec.kind == "mshar"`: `y = log_rv_hourly`; `X` columns `lrv1 = ln rv_hourly(t−1)`, `lrv4 = ln mean(rv_hourly(t−4..t−1))`, `lrv13 = ln mean(rv_hourly(t−13..t−1))` (means of levels, then log); `Z` as for MS-AR (const + z-scored lagged TVTP variables, `spec.std_window`); the first estimation row is the **14th week from `spec.start`**; `x_next` = the same three regressors at T = `end`.
  - `ms.canonicalize(params, model, by="const")`: regime 1 = the one with the higher intercept `const`; log-likelihood unchanged; idempotent. (For `by="sigma2"` nothing changes.)
  - `forecast.mshar_var_fc(pi: np.ndarray, m: np.ndarray, s2: np.ndarray) -> float` = Σᵢ πᵢ·exp(mᵢ + σᵢ²/2); `forecast.one_step(fit, design, thr)` for `kind="mshar"` returns `OneStep` with `var_fc = mshar_var_fc(...)`, `mᵢ = cᵢ + b′x_next`, `pi_H` as usual, `mean_fc = p_exceed = NaN`.
  - `oos.run_oos` for MS-HAR specs writes only `target ∈ {var, p_high}` rows (no `r`, no `p_exceed`).

- [ ] **Step 1: Write the failing tests**

  ```python
  def test_mshar_design_first_row_and_regressors():                 # tests/test_design.py
      p = synthetic_panel(); p = p[p["date"] >= C.HOURLY_SAMPLE_START].reset_index(drop=True)
      p["rv_hourly"] = np.arange(1, len(p) + 1, dtype=float); p["log_rv_hourly"] = np.log(p["rv_hourly"])
      spec = M.make_mshar_specs()["MSHAR_C"]; d = D.build_design(p, spec, p["date"].iloc[40])
      assert d.dates[0] == p["date"].iloc[13]                       # 14th week from spec.start
      assert np.isclose(d.y.iloc[0], np.log(14))                    # rv_hourly = index + 1
      assert np.allclose(d.X.iloc[0].to_numpy(), [np.log(13), np.log(11.5), np.log(7.0)])   # weeks 13; mean(10..13); mean(1..13)
      assert np.allclose(d.x_next.to_numpy(), [np.log(41), np.log(39.5), np.log(35.0)])   # T = row 40: rv 41; mean(38..41); mean(29..41)

  def test_canonicalize_by_const():                                 # tests/test_ms.py
      fit, design = fit_mshar_simulated(seed=405)
      params, flipped = M.canonicalize(fit.params, fit.model, by="const")
      inv = {v: k for k, v in fit.model.readable.items()}
      assert params[inv["mu_H"]] > params[inv["mu_N"]] and abs(fit.model.loglike(params) - fit.loglike) < 1e-6
      p2, f2 = M.canonicalize(params, fit.model, by="const"); assert not f2 and p2.equals(params)

  def test_mshar_var_fc_hand_computed():                            # tests/test_forecast.py
      v = F.mshar_var_fc(np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.5, 0.2]))
      assert np.isclose(v, 0.3 * np.exp(0.25) + 0.7 * np.exp(1.1)) and np.isclose(v, 2.4881238418)

  def test_one_step_mshar_matches_helper_and_nans():
      fit, design = fit_mshar_simulated(seed=405)
      o = F.one_step(fit, design, thr=1.0)
      assert o.var_fc > 0 and np.isnan(o.mean_fc) and np.isnan(o.p_exceed) and 0 <= o.pi_H <= 1

  def test_run_oos_writes_only_var_and_p_high_for_mshar(tmp_path):  # tests/test_oos.py
      rows = run_two_toy_origins(specs=[M.make_mshar_specs()["MSHAR_C"]], benchmarks=())
      assert set(rows["target"]) == {"var", "p_high"} and (rows.groupby("date")["target"].nunique() == 2).all()
  ```
  `fit_mshar_simulated` and `run_two_toy_origins` are defined in Task 0.

- [ ] **Step 2: Run; expect FAIL** (`NotImplementedError` or missing function).
- [ ] **Step 3: Implement the branches.** `canonicalize(by="const")`: reuse the permutation of spec §8.4 with `const` as the ordering key. In `oos.run_oos`, filter the rows emitted for `kind == "mshar"` specs to `var` and `p_high` before writing the checkpoint.
- [ ] **Step 4: Run `uv run pytest -q`; expect PASS.**
- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/design.py src/brent/ms.py src/brent/forecast.py src/brent/oos.py tests
  git commit -m "Add MS-HAR branches to design, canonicalize, one_step and run_oos"
  ```

---

### Task 14: Optional extension E1 — Markov-switching HAR notebook section (Thu Dec 3, only if everything else is done)

**Gate (read, do not decide):** run only if `rob_status.csv` has a row for every check in {placebo, harx_placebo, hourly_ctrl, aigpr, tone, ar, distribution, iran_mean, ovx, gdelt_geo, post2013} with status other than `queued`, **and** it is before Thu Dec 3, 15:00 (spec §11: "nothing new is started" after the freeze). Otherwise `log_status("E1", "cut", "time")`, skip Tasks 14a and 14.

**Files:**
- Modify: `notebooks/08_robustness.ipynb`
- Output: `results/tables/{ext_mshar_insample,ext_mshar_oos}.csv`; `results/forecasts/forecasts_mshar.csv`

**Interfaces:**
- Consumes: `ms.make_mshar_specs` (Task 8), Task 14a branches, `robustness.{fit_chain, variant_row, oos_compare}`, `oos.run_oos`.

- [ ] **Step 1: Smoke test the plumbing.** Try `design.build_design` for `MSHAR_C` on 2006-11-24 → 2025-12-26 and `canonicalize(by="const")`. If either raises `NotImplementedError`, do Task 14a first (time-box: if 14a is not done by Thu 17:00, `log_status("E1", "cut", "module branches not finished")` and stop). Then assert `y = log_rv_hourly`, the first estimation row is the 14th week from `HOURLY_SAMPLE_START`, and `forecast.one_step(...).var_fc > 0`.

- [ ] **Step 2: In sample.** `specs = ms.make_mshar_specs(C.M3_VARIANT)`; `fits = robustness.fit_chain(panel, C.RQ2_END, [(specs["MSHAR_C"], None), (specs["MSHAR_CI"], "MSHAR_C")])`. Assert the H regime has the higher intercept. `ext_mshar_insample.csv`: `variant_row("MSHAR_CI", ..., ("iran_gdelt",))` (β, `lr_raw`, `stat`, `pvalue`, df from free parameters) plus regime intercepts and durations.

- [ ] **Step 3: OOS** only if the remaining time before the freeze exceeds the estimate (`estimate_hours` is not available for a model absent from the main fitlog: use 1 h from the spec). `oos.run_oos(panel, "mshar", list(specs.values()), benchmarks=(), origins=oos.origins(*C.ROBUST_OOS_TARGETS))`, `assemble("mshar")`; `ext_mshar_oos.csv` with `oos_compare` pairs `(MSHAR_CI, MSHAR_C)` (nested), `(MSHAR_CI, HARX_CI)`, `(MSHAR_CI, M3)`, `(MSHAR_C, HAR_h)`. Markdown: E1 is an extension; its result is not part of any headline.

- [ ] **Step 4: Status and commit.**

  ```bash
  git add notebooks/08_robustness.ipynb results/tables/ext_mshar_*.csv results/forecasts/forecasts_mshar.csv results/tables/rob_status.csv
  git commit -m "Add optional Markov-switching HAR extension (E1)"
  ```

---

### Task 15: Analysis freeze, reproducibility check and hand-off (Thu Dec 3 evening → Fri Dec 4)

**Files:**
- Modify: `notebooks/06_evaluation.ipynb`, `07_events.ipynb`, `08_robustness.ipynb` (final runs), `README.md` (results guide)
- Create: `results/tables/handoff_numbers.csv` (written by the last cell of notebook 08)

**Interfaces:**
- Consumes: every table listed in Tasks 6, 7, 10–14 (14a has no output); notebook 04's `rq2_lr_tests.csv`, `rq2_coefficients.csv` (read headers first); `rob_status.csv`.

- [ ] **Step 1: Freeze gate (Thu Dec 3, evening).** Nothing new is started. For every check without a finished output, `log_status(check, "cut", "not finished before freeze")`. The last markdown cell of notebook 08 reads `rob_status.csv` and prints it as a "Limitations and cut checks" table, so the report author sees what was not done.

- [ ] **Step 2: Hand-off numbers cell** (last cell of notebook 08). Build `handoff_numbers.csv` with columns `name, value, source_file, note`, by reading the tables (no number typed by hand):
  - headline, test and holdout: mean QLIKE of M2 and M3 and their difference (`rv_hourly`), DM (QLIKE) p, Clark-West p, GW unconditional and conditional p (from `vol_losses.csv`, `vol_dm.csv`, `vol_cw_gw.csv`);
  - M3 vs HARX_CI and vs GARCHX_CI DM p; HARX_CI vs HARX_C and GARCHX_CI vs GARCHX_C DM and CW p;
  - the best linear model name (`vol_best_linear.csv`) and its validation QLIKE;
  - number of tests run (rows of `vol_dm.csv`, `vol_cw_gw.csv`);
  - non-converged M3 weeks and degenerate fits (`fitlog_main.csv`);
  - from notebook 04: β (Iran, N→H) and its z, LR M3 vs M2 and p (read the headers of its tables);
  - placebo p-values: in-sample, OOS (if run), HARX in-sample and OOS;
  - the deck fact checks (peak, weekly change rank) from `deck_fact_checks.csv`;
  - the count of robustness checks `done` / `cut` / `in_sample_only`.

- [ ] **Step 3: README results guide.** Add a section to `README.md`: a table of every file in `results/tables/` and `results/figures/` produced by notebooks 06–08 (name, notebook, one-line meaning), the note that `rv_hourly` is the main measure and test and holdout are separate, and which items are secondary. List explicitly the figures for slides: `cumloss_M2_minus_M3.png`, `cumloss_bestlinear_minus_M3.png`, `rq2_filtered_prob.png` (notebook 04), `events_grid.png`, `event_<id>.png`, `phigh_reliability.png`, `placebo_hist.png`, `rob_distribution.png`.

- [ ] **Step 4: Final runs.** `uv run pytest -q` passes. Restart-and-run-all notebooks 06, 07, 08 with **all FORCE flags False** (05 and the OOS runs load committed outputs); `DEV` is False in 06; `paths.check_budget()` passes in every notebook; no `dev_*` file is tracked (`git ls-files | grep dev_` is empty).

- [ ] **Step 5: Commit the freeze.**

  ```bash
  git add notebooks results README.md
  git commit -m "Analysis freeze: evaluation, events and robustness outputs, results guide"
  git tag analysis-freeze
  ```

> **CHECKPOINT C (Thu Dec 3, evening): analysis freeze.** Tell the user: the headline numbers, which robustness checks are done and which are cut (from `rob_status.csv`), runtime problems, and anything that contradicts the research plan. After this commit no table changes except for bug fixes found in Step 6.

- [ ] **Step 6: Reproducibility check (Fri Dec 4, spec §10).**
  1. Fresh clone into a temporary folder outside the repo; `uv sync`; `uv run pytest -q`.
  2. Branch on `config.INTRADAY_REDISTRIBUTE`: if `False`, run `uv sync --group intraday` and notebook 01's Dukascopy cell first (the intraday files are git-ignored); if `True`, nothing.
  3. Restart-and-run-all notebooks 02 → 08 with FORCE flags off.
  4. `git diff --stat` shows **no change** to committed tables (`results/tables`). If something differs, find the nondeterministic step (unseeded draw, ordering) before handing off.
  5. Optional, if the user wants it and time allows: rerun notebook 05 in full overnight and confirm identical forecasts (seeds make this exact).

- [ ] **Step 7: Hand-off to Linnéa (report and presentation), Fri Dec 4.** Give her:
  - the repo at the `analysis-freeze` tag (or the last commit if bug fixes followed);
  - `results/tables/handoff_numbers.csv` and the README results guide;
  - the executed notebooks 06, 07, 08 (readable on GitHub with outputs) and `rob_status.csv` (limitations);
  - the list of open framing decisions from Checkpoint A (the status of HAR-X and GARCH-X as benchmark, replacement or appendix).

  Write the hand-off message to the user as a short list (what is in the package, what is cut, what the user still decides). Do not start the write-up.

---

## Done when

- `uv run pytest` passes (new tests in `test_evaluate.py`, `test_plots.py`, `test_events.py`, `test_robustness.py` and additions to `test_design.py`, `test_forecast.py`, `test_ms.py`).
- Notebooks 06, 07, 08 restart-and-run-all cleanly offline from committed files, with `DEV = False` in 06 and all FORCE flags off.
- Committed: the spec §7.6 tables and figures per RV measure, `event_table.csv`, `deck_fact_checks.csv`, 12 event figures and the grid, every robustness output the status log marks `done`, `rob_status.csv`, `handoff_numbers.csv`.
- No table, figure or notebook under `dev_*` is tracked; `forecasts_main.csv` is unchanged; no robustness file contains a 2026 target.
- The reproducibility check shows an empty `git diff` for `results/tables`.
- The user has seen and approved: the Checkpoint A numbers, the OOS budget (Checkpoint B), the freeze summary (Checkpoint C).

## Slip rules (spec §11), as they bind this plan

| Situation | What changes |
|---|---|
| Intraday data unavailable (`MAIN_RV = "rv"`, `RV_MEASURES = ("rv",)`) | no code change; loops read config; `HAR_h` absent so its pairs are skipped and listed; `cumloss_M2_minus_M3_rv.png` is not built |
| Spec not frozen by Sun Nov 29 | Task 11 is skipped (`in_sample_only`); conditional GW dropped in Task 5 Step 2.4 |
| Full run not finished by Mon Nov 30 | Task 6 waits (06 stays `DEV`); 07 and Task 10 start; notebook 08 limited to placebo #1, HARX placebo #9, in-sample AI-GPR (Task 10 Steps 3, 4, 6) |
| Behind on Thu Dec 3 | no new work after the freeze; unfinished checks logged `cut` |
| Behind in notebook 06 | cut order of spec §11: drop the `rv_gk` tables first (keep the column), then GW conditional; `rv_hourly` and `rv` tables and the MVP items stay |
| E1 module branches not done by Thu 17:00 | E1 logged `cut` |
| Time short for OOS runs | `plan_oos_queue` drops placebo, then `tone`, then `ovx`, `gdelt_geo`, `aigpr`, `hourly_ctrl`; E1 is cut before all of them |
