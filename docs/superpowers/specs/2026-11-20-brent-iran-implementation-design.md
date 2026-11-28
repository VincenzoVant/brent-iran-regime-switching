# Brent × Iran Regime-Switching — Implementation Design

*Spec date: Fri Nov 20, 2026; updated Mon Nov 23, 2026. Data freeze: Fri Nov 6, 2026. Analysis freeze: Thu Dec 3. Due: Mon Dec 7, 2026, 6 pm.*
*Research plan (settled, implemented here, not changed): [`Plans/Brent-Iran Regime Switching - Feasibility and Plan.md`](<../../../Plans/Brent-Iran Regime Switching - Feasibility and Plan.md>), cited as "plan §N". Pitch deck: `Plans/Brent-Iran_Pitch_Modern-Sage.pptx` (not in git).*

**Labels.** **[decided]** = decided by the user. **[our call]** = filled in by this spec, with a one-line reason; changeable. Anything not labelled is taken directly from the plan.

**Change log**
- **2026-11-23: professor feedback — intraday RV, GDELT sentiment, GARCH-X/HAR-X.** After the intro presentation the professor asked for (a) the sentiment of GDELT Iran news (does positive vs. negative news matter), (b) a less noisy realized variance from intraday (1-hour) Brent prices, (c) GARCH-X and HAR-X models with lagged RV and a news indicator. The user approved all three **[decided]**. Additions: hourly RV from Dukascopy (§4.3, §6.8) as the main evaluation measure; conflict/cooperation shares and tone from GDELT (§6.4, §9.1) and an M3-tone variant (§7.4); HAR-X and GARCH-X as **extra models** (§4.4, §8.7), whose status (benchmark, replacement or appendix) is **open** and decided later from results, not here **[decided]**. Nothing had been built when this was written; the timeline (§11) is rewritten for the real start on Nov 23. Touched: §1–§4, §5.1–5.3, §5.5, §6 intro, §6.4–6.8, §7.1–7.9, §8.1, §8.3–8.12, §9.1–9.6, §10–§14.
- **2026-11-23 (later): aligned with Plans 2 and 3 after cross-document review.**
- **2026-11-23 (user decisions): weekly panel always committed (derived statistic); Thanksgiving is a normal working day; LSEG/UCSD-library route dropped; no comparison against deck claims (the deck was a feasibility pitch, already presented; it stays git-ignored).**
- **2026-11-23 (notebook 01 findings) [decided]:**
  - (1) Dukascopy hourly Brent starts **Thu 2010-12-02**: both `dukascopy-python` and `npx dukascopy-node` return nothing earlier. So `INTRADAY_START = 2010-12-02`, `HOURLY_SAMPLE_START = 2010-12-10`, and `STANDARDIZE_WINDOW_HOURLY = (2010-12-10, 2012-12-28)` (108 weeks); HAR-X/GARCH-X start Dec 2010. Older mentions of Nov 2006 / 2007-01-05 / 313 hourly weeks below are superseded.
  - (2) D1 read: `GDELT_FEED_RULE = "dateadded"` (the archive carries a Feb 2013 bulk `DATEADDED`; `SOURCEURL` is never NULL), and no 2.0 splice is needed. A third Q1 feed label `"stale"` marks daily-feed events dated more than `GDELT_STALE_AFTER_YEARS = 1` year before `DATEADDED` (537k pre-2013 events re-loaded in Jan 2020); `weekly.gdelt_weekly` always drops and counts them.
  - (3) FRED drops fake-browser user agents, so fetchers send requests' default.
  - (4) After Q1: the Jan 1–5, 2020 "stale" rows are current news with a broken `SQLDATE` (the normal feed nearly vanishes those days, and Jan 3 holds 348k Iran articles), so `GDELT_STALE_POLICY = "merge"`: stale rows count as daily feed. This supersedes "drops" in (2).
  - (5) GDELT 1.0 has no rows Jun 14 – Jul 1, 2025 (Israel–Iran war): those days are filled from GDELT 2.0 (`gdelt_v2_2025_daily.csv`), level-shifted on the other 2025 weeks with the §6.4 overlap rule (correlation ≥ 0.8), using `GDELT_V2_PATCH_DAYS`. The general 2.0 splice is still not needed.
  - (6) GDELT 2.0 is also empty Jun 15 – Jul 1, 2025 (it only fills Jun 14; overlap correlation 0.934, shift −0.109). The week ending 2025-06-27 has no GDELT day: `GDELT_EMPTY_WEEK_POLICY = "locf"` carries the previous week's GDELT values forward (no look-ahead; interpolation was rejected as look-ahead), flagged by the new panel column `gdelt_filled`. More than `GDELT_MAX_EMPTY_WEEKS = 1` consecutive empty weeks raise.
- **2026-11-27 (notebooks 02–05) [decided]:**
  - (7) `HOURLY_SAMPLE_START = 2013-06-07` and `STANDARDIZE_WINDOW_HOURLY = (2013-06-07, 2015-06-05)`: Dukascopy is unreliable Jan 2012 – May 2013 (stale quotes, holes). Supersedes the hourly dates in (1).
  - (8) Notebook 03: `PHI_SWITCHING = False` (LR test of φ_N = φ_H on 2005–2018, p = 0.846).
  - (9) Near-separation counts as degenerate: `MAX_TVTP_COEF = 50`; a fit with any |TVTP coefficient| above it is degenerate under §8.4's rule (every fit).
  - (10) `CONTROLS_ONLY_NH = True`: the TVTP controls (gpr, vix, log_rv) enter only the calm→turbulent logit; their turbulent→calm coefficients are fixed at 0 (the turbulent→calm logit keeps its constant). Reason: with expanding windows the turbulent→calm logit was near-separated at 3 of 6 checked sample ends (|coef| 89–562); the restriction removes it everywhere at a log-likelihood cost of 2–5. RQ2's test (M3 vs M2, df 1) is unchanged in meaning; M2 vs M1 now has df 3. Supersedes the "both logits" wording of §7.1 for the controls.
  - (11) Hourly benchmarks start when their data allows, without look-ahead: HAR_h/HARX_* from 2013-09-06 (13 weeks of hourly history), GARCHX_* from 2015-06-05 (end of the scale window). Model comparisons in notebook 06 use common weeks.
- 2026-11-20: approved.

---

## 1. Intent

Turn the settled research plan into working analysis: raw data → weekly panel → M1–M3, benchmarks and the extra linear news models (HAR-X, GARCH-X) → expanding-window out-of-sample (OOS) forecasts → evaluation tests and figures → event case studies → robustness. Since Nov 23, forecasts are evaluated mainly against realized variance from hourly Brent prices (`rv_hourly`), with the daily-return RV as a secondary measure. The written report (≤ 25 pages) is **out of scope** and comes later; this spec only has to produce the tables, figures and numbers it will cite.

The work is done in **Jupyter notebooks**, with Claude Code writing most of the code. This document has to be concrete enough that an agent can plan and build one notebook at a time without asking the user anything. A second, equally important goal: **keep local data minimal.** Only what the analysis uses is ever downloaded, and nothing large lands on disk.

---

## 2. Goals, non-goals, anti-goals

**Goals**
1. Every item in plan §3 (RQ1–RQ3), §8 (evaluation), §9 (events) and §10 (robustness), and every Nov 23 addition (change log), is produced by a named notebook (map in §7.9).
2. A teammate can clone the repo, run `uv sync`, and rerun notebooks 02 → 08 from committed files, without a Google account or network access. (One exception: if Dukascopy's terms forbid committing the derived daily file, rerunning notebook 02 first needs notebook 01's Dukascopy cell, which is free and needs no account; notebooks 03–08 still run offline from the committed panel. §6.8.)
3. One implementation of each risky piece (week mapping, lags, MS fit, forecaster, tests), imported everywhere and covered by small tests.
4. Expensive steps (BigQuery, the OOS loop) never rerun by accident and can resume after a crash.

**Non-goals**
- The written report, abstract and slides.
- Parked items from plan §17 (density forecasts/PIT/CRPS, prediction markets, PortWatch, Bloomberg pull, OPEC/Känzig/Kilian/dollar controls, GPR acts/threats split, placebo countries, 3-state model, winsorizing, `BZ=F`, VaR tests, GJR-GARCH).
- Brent implied vol from Bloomberg (plan §4.2 optional; on the cut list). The code accepts it as a drop-in `vix` replacement if someone gets the data.
- A general-purpose library, CLI, CI, or packaging beyond an editable install.
- Exchange-traded intraday Brent (FirstRate Data, LSEG Tick History, Databento). Only used if the Dukascopy validation fails (§4.3).
- Deciding now whether HAR-X/GARCH-X become benchmarks or replace anything. They are "extra models" until results are in **[decided]**.

**Anti-goals** (the project fails even if it runs)
- Large local data: raw GDELT zips, the full GDELT table, hourly price bars on disk, any file > 10 MB, total data > 20 MB.
- Lookahead: a predictor dated after the forecast origin; smoothed probabilities anywhere in a forecast; standardization stats, thresholds or instrument cutoffs computed on post-2012 data; using AI-GPR months before they were published.
- Using 2019–2026 to choose anything (specification, search settings, fallback, which linear model is "best", the roll rule) and above all using 2026 (plan §5, §7).
- Results a teammate cannot reproduce: unpinned environment, unseeded randomness, data pulled again after the freeze, notebooks that only work in the order someone happened to run cells.
- Two forecasters, two lag conventions, or two week mappings that disagree across notebooks.
- Silent label switching (H means "higher σ²" everywhere, after every fit).
- Hiding a failed estimation: every OOS fit records convergence, refits from scratch, degeneracy and restarts.

---

## 3. Constraints

| Constraint | Detail |
|---|---|
| Time | Work starts Mon Nov 23 with nothing built, about 3 days behind the original Nov 20 schedule; Thanksgiving (Nov 26) is a normal working day. Analysis freeze **Thu Dec 3**, hand-off to the report writer **Fri Dec 4**, due Mon Dec 7, 6 pm. Timeline in §11. |
| Intraday data | Free Dukascopy feed, no account. Instrument `BRENT.CMD/USD` is a CFD that tracks front-month ICE Brent futures: quotes, not exchange trades. Earliest data Nov 17, 2006. Hourly bars are never written to disk (§6.8). |
| Team | One builder (the user, with Claude Code as the coding agent), working through the notebooks in order **[decided]**. A teammate writes the report and presentation later from committed results and must be able to rerun notebooks 02 → 08. |
| Disk | Minimal data **[decided]**. Budget in §6.7. |
| BigQuery | Free sandbox: no credit card, 1 TB query bytes/month, tables expire after 60 days (we create no tables). **[decided]** |
| Tooling | Jupyter notebooks + a small shared module **[decided]**. uv 0.11.8 installed; system Python is 3.14. |
| statsmodels | `MarkovRegression` puts the same `exog_tvtp` regressors in every transition logit; no `fit_constrained`/`fix_params` for Markov-switching models; no `forecast()`; no Student-t errors. Parameter names are positional (`p[0->0].tvtp0`, `x1[0]`), not taken from DataFrame column names (checked on statsmodels 0.15.0). |
| Repo location | Inside an Obsidian vault on the Desktop; the `.venv` stays in the repo, git-ignored **[decided]**. Obsidian ignores dot-folders (`.venv`, `.git`). |

---

## 4. Approaches

### 4.1 Code organization

| Option | Verdict | Why |
|---|---|---|
| **A. Notebooks in pipeline order + small shared module `src/brent/`** | **Chosen [decided]** | Reusable pieces (fetchers, week logic, MS fitting, forecaster, OOS loop, tests, plot helpers) exist once and are tested. Notebooks keep the steps, tables, figures and interpretation. The user has worked this way before. |
| B. Notebooks only | Rejected | Functions copied between notebooks drift apart: two forecasters, two lag rules. `%run other.ipynb` hides where names come from and reruns side effects. |
| C. Script/make pipeline, notebooks only for viewing | Dropped | The user wants notebooks; a build system is overkill for 2.5 weeks and one builder. |

### 4.2 GDELT acquisition

| Option | Verdict | Why |
|---|---|---|
| **A. One aggregate SQL query on BigQuery `gdelt-bq.full.events` (GDELT 1.0)** | **Chosen [decided]** | Returns ~8,000 daily rows (< 1 MB). Scans roughly 30–60 GB of columns (estimate; confirmed by dry run), well inside 1 TB/month. Minutes, not hours. |
| B. Stream GDELT 1.0 yearly/monthly/daily zips from `data.gdeltproject.org/events/` and aggregate locally | Rejected | Tens of GB of transfer, hours of runtime, fragile resume logic. Kept only as a last resort if BigQuery access fails on every Google account (R1). |

### 4.3 Intraday Brent source (added Nov 23)

**Why this source.** No free source has intraday Brent before Nov 2006, and our evaluation targets run 2013–2026, so a source starting in late 2006 covers every target week and leaves six years (2007–2012) to check levels and standardize. Dukascopy is the only free source that does this.

| Option | Verdict | Why |
|---|---|---|
| **A. Dukascopy free historical feed, `BRENT.CMD/USD`, 1-hour bars** | **Chosen [decided]** | Free, no account, from Nov 17, 2006 (per the `dukascopy-node` README; verified in notebook 01). It is a CFD tracking front-month ICE Brent futures: dealer quotes, not exchange prices, and it rolls between contracts (§6.8 roll rule). Access via the Python package `dukascopy-python` (new optional group `intraday`); its API is unverified, so notebook 01 checks it. Fallback: the MIT-licensed `dukascopy-node` CLI via `npx`. |
| B. FirstRate Data, exchange ICE Brent hourly from Jan 2008 | Optional cross-check only | About $100 (price unverified). Bought only if the CFD level check in notebook 02 fails (§7.2). |
| C. LSEG Tick History via the UCSD library (RIC `LCO`, back to 2005) | Not pursued **[decided 2026-11-23]** | Access unknown; not worth the time before the deadline. |
| D. Oxford-Man Realized Library | Not usable | Has no oil, and stopped in mid-2022. |
| E. Bloomberg intraday | Not usable | Intraday lookback is about 140 days (forum claim, not checked); far too short. |
| F. Databento ICE Europe | Not usable | Starts Dec 2018, after the start of the evaluation window. |
| G. Oil ETFs (BNO, USO) | Not usable | US trading hours only; miss most of the ICE session and track the contract with a lag. |

### 4.4 GARCH-X estimation (added Nov 23)

| Option | Verdict | Why |
|---|---|---|
| **A. Hand-written Student-t GARCH-X likelihood, maximized with scipy** | **Chosen [our call]** | Standard GARCH-X (Engle 2002; Han & Kristensen 2014): nonnegative regressors with γ ≥ 0 added to the GARCH(1,1) variance, so h(t) stays positive without extra constraints, the model nests our GARCH(1,1)-t benchmark exactly (γ = 0), and γ has a clear reading ("extra variance per unit of the regressor"). About 60 lines; covered by a simulation-recovery test. |
| B. `arch` package | Rejected | `arch_model` cannot take regressors in the variance equation (only in the mean). |
| C. EGARCH-X / log-GARCH-X (regressors in logs, any sign) | Rejected | Would let us use the z-scored logs directly, but the model no longer nests our GARCH-t benchmark and its forecast needs a bias correction. |

---

## 5. Repo layout and environment

### 5.1 Layout

```
Brent-Iran Regime Switching/
├── README.md
├── pyproject.toml            # uv project, package "brent" (src layout)
├── uv.lock
├── .python-version           # 3.12
├── .gitignore
├── Plans/                    # research plan (+ deck, git-ignored)
├── docs/superpowers/specs/   # this spec
├── src/brent/
│   ├── __init__.py
│   ├── config.py             # all frozen constants and decisions
│   ├── paths.py              # repo-relative paths, size-budget check
│   ├── sources.py            # fetchers: FRED, GPR, AI-GPR, GDELT (BigQuery), Dukascopy (hourly → daily, in memory)
│   ├── weekly.py             # Friday-week mapping and weekly construction
│   ├── design.py             # THE place where lags and standardization happen
│   ├── ms.py                 # MS models: specs, fitting, relabeling, constrained M3
│   ├── forecast.py           # one-step MS forecaster
│   ├── benchmarks.py         # GARCH(1,1)-t, HAR-RV, random walk; extra models HAR-X, GARCH-X
│   ├── oos.py                # expanding-window loop, checkpoints, assembly
│   ├── insample.py           # RQ1/RQ2 tests, bootstrap LR, placebo shifts
│   ├── evaluate.py           # losses and forecast tests
│   ├── events.py             # event table and deck fact checks
│   ├── robustness.py         # robustness run builders, variant fits, comparisons, placebo helpers, run-time budget
│   └── plots.py              # shared figure helpers
├── notebooks/                # committed with outputs (§5.4)
│   ├── 01_fetch_data.ipynb
│   ├── 02_weekly_panel.ipynb
│   ├── 03_rq1_regimes.ipynb
│   ├── 04_rq2_explain.ipynb
│   ├── 05_oos_forecasts.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_events.ipynb
│   └── 08_robustness.ipynb
├── data/
│   ├── raw/                  # frozen trimmed snapshots + manifest.csv  (committed)
│   └── processed/            # weekly_panel.csv                         (committed)
├── results/
│   ├── forecasts/            # forecasts_<run>.csv                      (committed)
│   ├── fits/                 # params_<run>.csv, fitlog_<run>.csv       (committed)
│   ├── tables/               # *.csv for the report                     (committed)
│   ├── figures/              # *.png                                    (committed)
│   ├── frozen/               # spec_freeze.json, rq1_decision.json      (committed)
│   └── checkpoints/          # OOS loop partial outputs                 (git-ignored)
└── tests/
    ├── conftest.py           # synthetic raw data
    ├── test_config.py
    ├── test_sources.py
    ├── test_sources_intraday.py   # Dukascopy hourly → daily on synthetic bars
    ├── test_weekly.py
    ├── test_design.py
    ├── test_ms.py
    ├── test_forecast.py
    ├── test_benchmarks.py    # HAR-X design/forecast, GARCH-X recovery on simulated data
    ├── test_insample.py      # chi-bar-squared p-value, Wald equal-coefficient test
    ├── test_evaluate.py
    ├── test_plots.py         # figure helpers build on toy data
    ├── test_events.py        # event table, deck fact checks
    ├── test_robustness.py    # run-id guards, variant fits, placebo helpers
    ├── test_oos.py
    └── test_budget.py
```

### 5.2 Environment **[our call: uv project, src layout, editable install]**

- **Python 3.12**, pinned in `.python-version` and `requires-python = ">=3.12,<3.13"`. Why: statsmodels, arch, scipy and the Google Cloud client libraries (grpc) all ship mature wheels for 3.12; 3.14 is new enough that a compiled dependency may lack a wheel and try to build from source. statsmodels 0.15.0 and arch 9.0.0 were checked to install on 3.12.
- **Dependencies** (`uv add`): `numpy pandas scipy "statsmodels>=0.15" arch matplotlib requests xlrd`.
  - `statsmodels>=0.15` because it has `statsmodels.tsa.stattools.diebold_mariano_test` with the Harvey-Leybourne-Newbold option and a custom-loss callable (answers plan §8's "check for a built-in DM test": yes).
  - `xlrd` reads the GPR `.xls`.
- **Dev group** (`uv add --dev`): `ipykernel pytest`. People who prefer JupyterLab over VS Code add `jupyterlab` to the dev group.
- **`gdelt` group** (`uv add --group gdelt pandas-gbq google-cloud-bigquery`): only the builder needs it, once, for notebook 01 (`uv sync --group gdelt`). Anyone rerunning from notebook 02 skips the Google libraries entirely. **[our call: keeps the default install small and Google-free]**
- **`intraday` group** (`uv add --group intraday dukascopy-python`): only for the Dukascopy cell of notebook 01 (`uv sync --group intraday`, or `uv sync --group gdelt --group intraday` for both). The `npx dukascopy-node` fallback needs Node.js (`node --version`), not a Python dependency. `dukascopy_python` is imported inside the fetch function only. **[our call: same reason as `gdelt`]**
- Setup per clone:
  ```
  uv sync                                  # creates .venv, installs brent editable
  uv run pytest                            # should pass before any notebook work
  ```
  Kernel: in VS Code pick `.venv/bin/python`. For JupyterLab: `uv run python -m ipykernel install --user --name brent-iran --display-name "Python (brent-iran)"`.
- Initial scaffold (one-time): `uv init --package --name brent --python 3.12` in the repo root, then move nothing; it creates `src/brent/__init__.py`.
- Every notebook starts with:
  ```python
  %load_ext autoreload
  %autoreload 2
  import brent
  from brent import config as C
  ```
  `autoreload` makes module edits visible without restarting the kernel.

### 5.3 `.gitignore` changes

`*.pptx` is already ignored (deck shared outside git). Add:

```
# Python / tooling
.pytest_cache/
*.egg-info/
build/
dist/

# Expensive intermediate outputs (rebuildable, can be large)
results/checkpoints/

# Guard against accidental bulk downloads
*.zip
*.xls
*.xlsx
*.parquet
```

**[our call]** The `*.zip`/`*.xls` guards stop a stray GDELT zip or the GPR workbook from being committed; fetchers read those into memory only.

**Conditional (decided in notebook 01, §6.8):** if Dukascopy's terms of use forbid redistribution (`config.INTRADAY_REDISTRIBUTE = False`), also add
```
data/raw/brent_intraday_daily.csv
data/raw/brent_intraday_removed.csv
```
The weekly aggregates in `weekly_panel.csv` stay committed either way **[decided 2026-11-23: a derived weekly statistic, not the vendor's data]**.

### 5.4 Notebook outputs: kept **[our call]**

Executed notebooks are committed **with outputs**. Why: with one builder there are no notebook merge conflicts, and the report author can read every table, figure and interpretation on GitHub without installing or running anything. The files in `results/` remain the source the report cites; notebook outputs are the readable record of how they were made.

Rules that keep this small and honest:
- **Restart kernel and run all** before committing a notebook, so outputs match the code (expensive cells skip via `FORCE_*` flags and load saved results).
- Inline figures at 100 dpi (`plt.rcParams["figure.dpi"] = 100`); saved PNGs in `results/figures/` at 150 dpi.
- Long loops print one summary line per 50 origins, not per fit; warnings from repeated fits are captured into the fitlog, not printed.
- Commit executed notebooks at milestones (end of a notebook, after a rerun with new results), not after every exploratory run, to limit history growth.
- Size limits per notebook in §6.7, checked by `check_budget()`.

### 5.5 Working mode **[decided: one builder, sequential]**

- One builder works notebook by notebook in pipeline order, with Claude Code writing the code. Module code is written just before the notebook that first needs it, together with its tests (§11).
- Frozen decisions (roll rule and terms finding in notebook 01; φ after notebook 03; settings and M3 variant after notebook 05's validation stage) are written into `config.py` by hand at those points and committed with the matching file in `results/frozen/`.
- Module changes: `uv run pytest` before each commit. Commit after each notebook milestone.
- Reproducibility for the report author: everything from notebook 02 onward runs from committed files, with no Google account and no network (§2, goal 2).

---

## 6. Data acquisition (notebook 01 only)

Every fetch reads into memory, trims to the needed columns and window, and writes one small CSV to `data/raw/`. Each write appends a row to `data/raw/manifest.csv`: `file, source, url_or_table, fetched_at_utc, first_date, last_date, n_rows, sha256, bytes`. **[our call: documents the freeze and makes silent changes visible in git]**

Window for daily sources: **2004-12-01 → 2026-11-06** (the first weekly return needs the Dec 31, 2004 close; a month of margin covers holidays). **Exception: Brent is fetched from 1987-05-20** (its first FRED observation), so the plan §16 fact check "largest weekly gain since 1987" runs on frozen local data in notebook 07; notebook 02 trims Brent to the panel window. **[our call]** Dukascopy hourly Brent starts at its first available bar (Nov 17, 2006) and ends at 2026-11-06 22:00 UTC (§6.8). No notebook after 01 touches the network.

### 6.1 Brent spot, VIX, OVX (FRED, no key)

| Series | URL | Output | ≈ Size |
|---|---|---|---|
| Brent | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU&cosd=1987-05-20&coed=2026-11-06` | `data/raw/brent_daily.csv` (`date, brent`), 1987-05-20 → 2026-11-06 | 250 KB |
| VIX | same with `id=VIXCLS` | `data/raw/vix_daily.csv` (`date, vix`) | 100 KB |
| OVX | same with `id=OVXCLS&cosd=2007-05-01` | `data/raw/ovx_daily.csv` (`date, ovx`) | 80 KB |

- `cosd`/`coed` trim on FRED's side, so only the window is transferred.
- The header's date column is `observation_date` (older files used `DATE`); accept both. Missing values arrive as `.` or empty; drop those rows.
- Send a generic `User-Agent` header with `requests`; FRED sometimes refuses default library agents. Fallback: FRED API with a free key (store the key in `.env`, already ignored).
- Do **not** use `WCOILBRENTEU` (weekly average; plan §4.2).

### 6.2 Global GPR (Caldara & Iacoviello, daily)

- URL: `https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls` (read with `pd.read_excel(io.BytesIO(resp.content))`; never saved).
- Keep: the date column (`DAY` as `YYYYMMDD`, or `date`; verify) and `GPRD`. Window as above.
- Output: `data/raw/gpr_daily.csv` (`date, gprd`), ≈ 160 KB.
- Check: last date ≥ 2026-11-06. If the file stops earlier, the last weeks have missing GPR; record it in the manifest and raise it with the team (do not fill silently).

### 6.3 AI-GPR Iran (Iacoviello & Tong, monthly)

- File `ai_gpr_country_monthly.csv`, column `Iran_all` (plan §4.2). The exact URL is not in the plan; the builder finds it on the AI-GPR page of matteoiacoviello.com and records it in `config.AIGPR_URL`.
- Keep: month, `Iran_all`, months 2004-01 → 2026-10 (later months could not have been published by Nov 6).
- Output: `data/raw/aigpr_iran_monthly.csv` (`month, aigpr_iran`), ≈ 6 KB.

### 6.4 GDELT Iran share (BigQuery) **[decided: BigQuery sandbox; our call: auth path]**

**One-time setup (builder only; nobody rerunning from notebook 02 needs it):**
1. Use a Google account that can open the Cloud console. Try the UCSD account first. If Google Cloud is blocked for it (UCSD's Workspace admins may disable Cloud), use a **personal Google account** (R1).
2. Open `https://console.cloud.google.com/bigquery`, accept the terms. The sandbox creates or asks for a project; note the **project id**.
3. Set the project id **only** in the environment variable `BRENT_GCP_PROJECT` (e.g. in the git-ignored `.envrc` or `.env`). There is no config constant for it. It is not a secret, but it is personal, so it never lands in `config.py` or git. **[our call]**
4. `uv sync --group gdelt`. First query opens a browser for OAuth (via `pydata_google_auth`). The cached token lives in the user's home directory, outside the repo. No gcloud CLI needed.

Auth flow in code: get credentials once with `pydata_google_auth.get_user_credentials(["https://www.googleapis.com/auth/bigquery"])`, pass them to both `google.cloud.bigquery.Client(project=..., credentials=...)` (dry runs) and `pandas_gbq.read_gbq(..., credentials=...)` (real queries). Every real query also sets `maximumBytesBilled` = `config.GDELT_MAX_BYTES` (**[our call]** 200 GB), so BigQuery refuses a query that would scan more.

**Query D1 — diagnostic (dry run, then run):** confirms column names and types, what `DATEADDED` holds for backfile rows, and coverage through Nov 6, 2026.

```sql
-- D1: verify columns/types on the dry run. Result: a few thousand rows, inspected in memory, not saved.
SELECT
  DIV(SQLDATE, 10000)        AS event_year,
  DIV(DATEADDED, 100)        AS added_yyyymm,      -- verify: INTEGER YYYYMMDD in GDELT 1.0
  COUNT(*)                   AS n_events,
  SUM(NumArticles)           AS articles,
  COUNTIF(SOURCEURL IS NULL) AS n_no_url           -- verify: SOURCEURL exists only for the daily feed
FROM `gdelt-bq.full.events`
GROUP BY event_year, added_yyyymm
ORDER BY added_yyyymm, event_year
```

What the builder reads from D1:
- **Coverage:** `added_yyyymm` continues through 202611 with no multi-month gaps. If the 1.0 daily feed stops before Nov 6, 2026, apply plan §4.3 (only in that case):
  - Run Q1 against `gdelt-bq.gdeltv2.events` (2.0 starts Feb 18, 2015). In 2.0, `DATEADDED` is `YYYYMMDDHHMMSS`, so the day key is `DIV(DATEADDED, 1000000)`, and the `feed` label is `v2`. Dry-run it too. If `gdelt-bq.gdeltv2.events_partitioned` exists, prefer it and filter on `_PARTITIONTIME` to cut bytes.
  - **Overlap check:** from the week ending 2015-02-20 to the last full Sat–Fri week of 1.0 data, compute weekly `iran_gdelt` from both feeds. Require a correlation of the weekly log shares ≥ 0.8 **[our call]**, and report the mean difference (1.0 − 2.0). If the check fails, stop and ask the team.
  - **Splice date:** the first Friday-week after the last full week of 1.0 data. From that week on, `iran_gdelt` = 2.0 log share + the mean overlap difference, so the level is continuous. Record the week in `config.GDELT_SPLICE_WEEK` and add a `gdelt_source` column (`v1`/`v2`) to the panel.
- **Backfile `DATEADDED`:** if backfile rows (event dates before Apr 2013) carry their event date or a pre-2013 value in `DATEADDED`, the feed rule below works as written. If they carry a 2013 bulk-load date, classify backfile rows another way (e.g. `SOURCEURL IS NULL`) and record the rule in `config.GDELT_FEED_RULE`. Without this check, the whole backfile could pile onto one day.

**Query Q1 — main daily aggregate.** Classifying each event row (backfile vs. daily feed, Iran or not, conflict or cooperation) needs event-level fields, so it happens in SQL. **All calendar logic** (Friday weeks, weekend rule, the Apr 1, 2013 switch, the freeze cut, the weekly shares and tone means) **is in Python** (`weekly.gdelt_weekly`), tested once. **[our call: one tested place for week logic]**

**Sentiment columns (added Nov 23) [decided].** Q1 also carries the CAMEO `QuadClass` (1 verbal cooperation, 2 material cooperation, 3 verbal conflict, 4 material conflict), `AvgTone` and `GoldsteinScale` of each event, so the conflict/cooperation split and the article-weighted tone are built from the same single pass. Q1 has not been run yet, so these columns are added before its first run; there is no second query.

```sql
-- Q1: verify column names on the dry run (Actor1CountryCode, Actor2CountryCode,
-- ActionGeo_CountryCode, NumArticles, SQLDATE, DATEADDED, QuadClass, AvgTone, GoldsteinScale).
WITH ev AS (
  SELECT
    IF(DATEADDED >= 20130401, 'daily', 'backfile')            AS feed,     -- per D1; see GDELT_FEED_RULE
    IF(DATEADDED >= 20130401, DATEADDED, SQLDATE)             AS day_key,  -- daily feed: ingestion day; backfile: event day
    NumArticles,
    QuadClass,
    AvgTone,
    GoldsteinScale,
    (IFNULL(Actor1CountryCode, '') = 'IRN'
       OR IFNULL(Actor2CountryCode, '') = 'IRN')              AS iran_actor,
    (IFNULL(ActionGeo_CountryCode, '') = 'IR')                AS iran_geo   -- FIPS code
  FROM `gdelt-bq.full.events`
)
SELECT
  feed,
  day_key,
  COUNT(*)                                                                 AS n_events,
  SUM(NumArticles)                                                         AS articles_all,
  SUM(IF(iran_actor, NumArticles, 0))                                      AS articles_iran_actor,      -- main measure
  SUM(IF(iran_actor OR iran_geo, NumArticles, 0))                          AS articles_iran_actor_geo,  -- robustness variant
  COUNTIF(iran_actor)                                                      AS events_iran_actor,
  SUM(IF(iran_actor AND QuadClass IN (3, 4), NumArticles, 0))              AS articles_iran_conflict,
  SUM(IF(iran_actor AND QuadClass IN (1, 2), NumArticles, 0))              AS articles_iran_coop,
  SUM(IF(iran_actor AND AvgTone IS NOT NULL, AvgTone * NumArticles, 0))    AS tone_x_articles_iran,
  SUM(IF(iran_actor AND AvgTone IS NOT NULL, NumArticles, 0))              AS articles_iran_tone_n,
  SUM(IF(iran_actor AND GoldsteinScale IS NOT NULL, GoldsteinScale * NumArticles, 0)) AS goldstein_x_articles_iran,
  SUM(IF(iran_actor AND GoldsteinScale IS NOT NULL, NumArticles, 0))       AS articles_iran_goldstein_n
FROM ev
WHERE day_key BETWEEN 20050101 AND 20261106
GROUP BY feed, day_key
ORDER BY day_key, feed
```

- One pass returns the main measure, the robustness variant and the sentiment columns.
- **[our call]** The two `_n` columns are the denominators of the tone and Goldstein means: they count only articles whose event has a non-null value, so a null `AvgTone` or `GoldsteinScale` cannot bias the mean toward 0. Rows with a null `QuadClass` count in neither the conflict nor the cooperation share.
- The table is not partitioned, so bytes scanned = full size of the referenced columns regardless of `WHERE`. Run with `dry_run=True, use_query_cache=False` first; print `total_bytes_processed`; stop if above `GDELT_MAX_BYTES`. The three sentiment columns add roughly 5–15 GB (estimate; the dry run gives the exact figure), still far below the 200 GB cap.
- Output: `data/raw/gdelt_daily.csv` (`feed, date, n_events, articles_all, articles_iran_actor, articles_iran_actor_geo, events_iran_actor, articles_iran_conflict, articles_iran_coop, tone_x_articles_iran, articles_iran_tone_n, goldstein_x_articles_iran, articles_iran_goldstein_n`), ≈ 8,050 rows, ≈ 0.9 MB.
- Expected total BigQuery use: D1 (30–60 GB) + Q1 (40–75 GB) (estimates), < 15% of the monthly sandbox quota. Optional diagnostic D2 (daily feed keyed by `SQLDATE`, to separate a dating effect from a feed effect at Apr 2013) only if the level-shift plot in notebook 02 shows a large break.
- **GDELT 2.0 splice and sentiment:** `gdeltv2.events` has the same three fields. If the splice is needed, the same overlap check and level adjustment (mean overlap difference) are applied to each of `iran_conflict`, `iran_coop`, `iran_tone`, `iran_goldstein` separately **[our call]**; a sentiment column failing the 0.8 correlation is reported, and only the main `iran_gdelt` check stops the work.

### 6.5 Freeze rules

- Every source is cut at **2026-11-06** (AI-GPR: months up to 2026-10; Dukascopy: bars starting before 2026-11-06 22:00 UTC, the end of Nov 6 under the §6.8 day rule).
- Snapshots are fetched once (today or soon) and committed; values revised by a source between Nov 6 and the fetch date cannot be undone and are disclosed as such.
- After notebook 01 has run, nothing re-downloads unless `FORCE_FETCH[source] = True` is set by hand.

### 6.6 Notebook 01 skip logic

Each source cell: `if paths.RAW[source].exists() and not FORCE_FETCH[source]: skip`. The GDELT cells also skip when `BRENT_GCP_PROJECT` is unset, printing a message, so someone without a Google account can run the notebook top to bottom without error. The Dukascopy cell (`FORCE_FETCH["intraday"]`) skips with a message when neither `dukascopy_python` nor `npx` is available.

### 6.7 Data-volume budget **[our call]**

| Rule | Limit |
|---|---|
| Any single file written anywhere in the repo (excluding `.venv`, `.git`) | ≤ 10 MB |
| `data/` total | ≤ 5 MB (expected ≈ 2.4 MB: the 1987+ Brent series, `gdelt_daily.csv` ≈ 0.9 MB with the sentiment columns, `brent_intraday_daily.csv` ≈ 0.35 MB, `weekly_panel.csv` ≈ 0.35 MB with the new columns) |
| `data/` + `results/` (excl. checkpoints) | ≤ 20 MB (expected ≈ 7–11 MB, mostly PNGs; `forecasts_main.csv` ≈ 1.1 MB) |
| Each executed notebook in `notebooks/` | ≤ 5 MB; all notebooks together ≤ 15 MB (expected ≈ 5–8 MB) |
| Downloads | Only the seven sources above (FRED ×3, GPR, AI-GPR, GDELT, Dukascopy); nothing written except the trimmed CSVs. Hourly bars (≈ 110,000 rows) stay in memory; the `npx` fallback writes them to a temporary directory outside the repo that is deleted when the cell ends |

Enforced by `paths.check_budget()` (walks the repo, raises with the offending paths), called in the last cell of every notebook and in `tests/test_budget.py`. Figures are saved as PNG at 150 dpi.

### 6.8 Brent intraday from Dukascopy (added Nov 23) **[decided: source, hourly bars, minimal data; our call: details marked]**

**Access.**
- Primary: `dukascopy-python` (group `intraday`). Its API is unverified; notebook 01 checks the call that returns 1-hour bid candles for `BRENT.CMD/USD` and writes the exact call into `sources.fetch_dukascopy_hourly`.
- Fallback: `npx dukascopy-node -i brentcmdusd -from 2006-11-17 -to 2026-11-07 -t h1 -p bid -f csv` run into a `tempfile.TemporaryDirectory()` outside the repo, read with pandas, then deleted (flag names verified in notebook 01).
- Fetch in yearly chunks with retries, concatenated in memory. Expect minutes, not hours.
- Bid side only **[our call: one download; bid-quote returns carry no bid-ask bounce]** (`config.INTRADAY_PRICE_SIDE = "bid"`).

**Normalizing the bars** (`sources.normalize_hourly`):
- Timestamps are bar **start** times, converted to UTC. Naive timestamps are taken as UTC; tz-aware ones are converted with `tz_convert("UTC")`. All later logic uses UTC only, so DST never moves a bar.
- Sort, drop duplicate timestamps, and drop bars with zero volume if the feed returns flat bars for closed hours **[our call: a flat filler bar would add a zero return and inflate the hour count]**.
- Drop bars starting at or after 2026-11-06 22:00 UTC (freeze).

**Hourly returns.** r_h = 100·ln(close_h / close_{h−1}), where h−1 is the **previous bar in the whole series**, however long the gap. So the overnight break, the weekend and any missing hours go into the first bar after the gap: no return is dropped and none is counted twice. The first bar of the series has no return.

**Day rule [our call].** Day d holds the bars that **start in [d−1 22:00, d 22:00) UTC**, i.e. `date = (timestamp_utc + 2h).normalize()` (`config.INTRADAY_DAY_END_UTC = 22`). Why 22:00 UTC: it is the ICE Brent close (23:00 London) in summer and an hour before it in winter; a fixed UTC hour avoids DST logic. Consequences, both intended:
- The Sunday-evening reopening bars (from 22:00 or 23:00 UTC) and their weekend-gap return land on **Monday**.
- A winter Friday bar starting at 22:00 UTC lands on **Saturday**, which `week_ending` maps to the next Friday. So each week runs from Friday 22:00 UTC to Friday 22:00 UTC, and summing a Sat–Fri week's days gives Friday-close to Friday-close, with "close" = last quote before 22:00 UTC.

**Roll jumps [decided in notebook 01 with the user; stored in `config.INTRADAY_ROLL_RULE`].** The CFD switches to the next futures contract around expiry; a return across a switch is a price-level jump, not variance.
1. Notebook 01 runs `sources.roll_candidates(hourly, **C.INTRADAY_ROLL_SCAN)`: every hourly return with |r| > k × the median |r| of the previous `window` bars (defaults k = 8, window = 120 ≈ one trading week **[our call]**), with its timestamp, ratio and the gap length before it.
2. The builder shows the table to the user, and they agree one rule.
   - What to look at: whether candidates cluster around a fixed day of the month, and whether they are the first bar after a gap.
   - **[our call]** The rule is chosen from the **2006–2018** candidates only (anti-goal: 2019–2026 chooses nothing).
   - The 2019–2026 candidates and removals are listed afterwards. Any removed return within ±1 week of a `config.EVENTS` date is shown to the user before the rule is frozen, because genuine war-news jumps must not be removed as rolls.

   The rule is one of:
   - `{"kind": "none"}`: no removals.
   - `{"kind": "threshold", "k": float, "window": int}`: remove returns whose ratio exceeds k.
   - `{"kind": "timestamps", "timestamps": [ISO UTC strings]}`: remove exactly these bars.
3. The rule goes into `config.INTRADAY_ROLL_RULE` (default `None`; `intraday_daily` refuses to run while it is `None`). Removed returns are dropped from `rv_intraday` and from `n_hours`, and logged in `data/raw/brent_intraday_removed.csv` (`timestamp_utc, date, ret, ratio, reason`).
4. A removed return breaks the identity "sum of returns = close-to-close change" on purpose.

**Daily aggregation** (`sources.intraday_daily`, in memory): per day d, `rv_intraday` = Σ r_h² over retained returns, `n_hours` = number of retained returns, and `open, high, low, close` = first open, max high, min low, last close of the day's bars.

**Output** (the only intraday files written are this one and the small removed-returns log): `data/raw/brent_intraday_daily.csv` with columns `date, rv_intraday, n_hours, open, high, low, close`; ≈ 5,000 rows (Mon–Fri plus a few winter Saturdays), ≈ 0.35 MB. Plus a manifest row (`source = "Dukascopy BRENT.CMD/USD h1 bid"`, `url_or_table` = the package or CLI used).

**Terms of use.** Notebook 01 reads Dukascopy's terms for historical data. It records the finding (with the page URL and date) in a markdown cell and in `config.INTRADAY_REDISTRIBUTE`:
- `True`: both intraday CSVs are committed.
- `False`: both are git-ignored (§5.3), and the README says to run notebook 01's Dukascopy cell before notebook 02.

**Checks (notebook 01):**
- The first bar is on or before 2006-11-20, and the actual first date is written into the manifest (verifies `INTRADAY_START`).
- The last daily row is 2026-11-06.
- No duplicate dates; `n_hours` ≤ 24; `rv_intraday` ≥ 0.
- Row counts by weekday are printed (Sundays must be 0).
- The number of removed returns is printed.
- Days with `n_hours` < 12 are listed.

---

## 7. Notebook-by-notebook design

Conventions for every notebook:
- First cell: imports, `autoreload`, `np.random.seed(C.SEED)` (a backstop only: all fits and draws are seeded explicitly, §8.4), a `FORCE_*` flag block.
- Reads only files written by earlier notebooks; writes only to its own declared outputs.
- Ends with a **checks** cell (asserts) and `paths.check_budget()`.
- Markdown cells carry interpretation; tables/figures are saved via `plots.save_fig(name)` / `paths.save_table(df, name)`.

### 7.1 `01_fetch_data.ipynb`

- **Purpose:** the only notebook that touches the network. Plan §4.2, §4.3, §4.1 freeze.
- **Inputs:** URLs/table/instrument from `config`; `BRENT_GCP_PROJECT` env var for GDELT.
- **Outputs:** `data/raw/{brent,vix,ovx,gpr}_daily.csv`, `aigpr_iran_monthly.csv`, `gdelt_daily.csv` (with sentiment columns), `brent_intraday_daily.csv`, `brent_intraday_removed.csv`, `manifest.csv`.
- **Steps:** FRED ×3 → GPR → AI-GPR → GDELT (dry run D1 → run D1 → decide feed rule → dry run Q1 → run Q1) → Dukascopy (§6.8): check the `dukascopy-python` call (else `npx`) → read the terms, set `INTRADAY_REDISTRIBUTE` → fetch hourly bars into memory → verify the first date → `roll_candidates` → **decide the roll rule with the user**, set `INTRADAY_ROLL_RULE` → `intraday_daily` → save both files.
- **Checks:** each file's date range covers the window; no duplicate dates; Brent starts by 1987-05-20, has a close on Dec 31, 2004 (or the last trading day before it) and on or before 2026-11-06; GDELT has no feed overlap on the same day and no gap > 3 consecutive calendar days after Apr 2013 (gaps listed, not filled); GDELT sentiment columns non-negative where they are counts, `articles_iran_conflict + articles_iran_coop ≤ articles_iran_actor` per row; the Dukascopy checks in §6.8; bytes scanned printed and below cap; budget.

### 7.2 `02_weekly_panel.ipynb`

- **Purpose:** build `weekly_panel.csv` and the data section. Plan §4.1, §4.3 (level-shift check, coverage), §4.4, §5 (weekend rule).
- **Inputs:** `data/raw/*.csv`.
- **Outputs:** `data/processed/weekly_panel.csv`; `results/tables/data_summary.csv`, `rv_measures_check.csv`; figures `data_series.png` (one panel per variable), `gdelt_share_2013_break.png`, `gdelt_main_vs_geo.png`, `rv_hourly_vs_daily.png`, `gdelt_tone.png`.
- **Steps:**
  1. `weekly.build_weekly_panel(raw)` (spec in §9.1).
  2. Data table: for `r, rv, log_rv, rv_hourly, log_rv_hourly, rv_gk, iran_gdelt, iran_conflict, iran_coop, iran_tone, iran_goldstein, gpr, vix, ovx, aigpr_iran`: N, mean, SD, min, max, AR(1) coefficient, ADF statistic and p-value (`statsmodels.tsa.stattools.adfuller`, constant, AIC lag choice). **[our call: ADF spec]**
  3. Time-series plots.
  4. Apr 2013 level-shift check: plot `iran_gdelt` with a vertical line at the week ending Apr 5, 2013; report the mean of `iran_gdelt` 52 weeks before vs. after and a simple OLS on a post-2013 dummy (descriptive only). The dummy is carried to robustness (notebook 08).
  5. Coverage: GDELT has a row for every calendar day, so a complete Sat–Fri week has `gdelt_days` = 7. Weeks with `gdelt_days` < 6 are listed and flagged (kept, not filled). **[our call]**
  6. Zero values: a week with 0 Iran articles (main or geo variant, conflict or cooperation) or an AI-GPR month ≤ 0 cannot be logged. `build_weekly_panel` **stops with an error** listing them; nothing is filled or offset (no ln(1 + x)). The team decides what to do.
  7. **Hourly vs. daily RV check (added Nov 23).** On `STANDARDIZE_WINDOW_HOURLY` (2007–2012): correlation of `rv_hourly` with `rv` (levels and logs), and the ratio mean(`rv_hourly`)/mean(`rv`). Repeat on 2013–2018 for information. Save these to `rv_measures_check.csv`. Figure `rv_hourly_vs_daily.png`: log-log scatter, plus both series over time. Spot (FRED, daily returns) and the futures CFD (hourly) can differ in level. **No rescaling [our call: the variance forecasts are scored against each measure as it is]**. If the 2007–2012 ratio is outside `RV_LEVEL_RATIO_BAND` = [0.8, 1.25], flag it to the team (the FirstRate cross-check of §4.3 becomes an option). Also: list weeks with `n_hours_week < MIN_HOURS_WEEK`, count weeks that contain a removed roll return, and report mean `rv_gk`/`rv_hourly` in those weeks vs. the others.
  8. **Tone plots (added Nov 23).** `gdelt_tone.png`: `iran_conflict` and `iran_coop` (log shares) and `iran_tone` (level) over 2005–2026, with the Apr 2013 line and the nine event weeks marked. Report the correlation of `iran_conflict` and `iran_coop` with `iran_gdelt`. This is a collinearity warning for M3-tone; it is not used to choose anything.
- **Checks:** 1,140 rows, first 2005-01-07, last 2026-11-06, all Fridays; `rv > 0`; `n_days` between 1 and 5 (list weeks with < 4); `iran_gdelt`, `iran_conflict`, `iran_coop`, `iran_tone`, `iran_goldstein` finite; `rv_hourly` NaN before `HOURLY_SAMPLE_START` and finite and > 0 from it on, and the first non-NaN week equals `config.HOURLY_SAMPLE_START` (if not, update config and say so); `rv_hourly` has no NaN in 2013–2026 (the main evaluation measure); window counts 417 / 313 / 365 / 45 (train / validation / test / holdout); Abqaiq (Sat Sep 14, 2019) maps to week 2019-09-20 and Feb 28, 2026 to 2026-03-06 (also unit-tested).

### 7.3 `03_rq1_regimes.ipynb`

- **Purpose:** RQ1 on **2005–2018** and the frozen φ decision. Plan §3 RQ1, §6.1, §6.4 pitfalls, §14.4. Also develops and verifies the constrained M3 fit and the forecaster on real data.
- **Inputs:** `weekly_panel.csv`.
- **Outputs:** `results/frozen/rq1_decision.json`; `results/tables/rq1_tests.csv`, `rq1_params_M1.csv`; figure `rq1_filtered_prob_2005_2018.png`.
- **Steps:**
  1. Fit M1 (constant transition probabilities, switching μ, φ, σ²) on 2005–2018 with `search_reps = C.SEARCH_REPS_FIRST` (**[our call]** 30, inside plan's 20–50).
  2. **1 vs. 2 regimes:** one-regime Gaussian AR(1) vs. M1. Plain LR, BIC of both, and a **parametric-bootstrap LR**: simulate `C.BOOT_REPS` (**[our call]** 199) series from the fitted one-regime model, fit both models to each, p = (1 + #{LR* ≥ LR}) / (B + 1). The actual fit and every replicate use the **same** search effort (`search_reps = C.SEARCH_REPS_FIRST`; replicate b seeded with `C.SEED + b`) and the same degeneracy rule (§8.4). Degenerate replicates are logged and counted, and the count is reported next to the p-value. Est. ≈ 20 min (199 × ≈ 6 s); drop to 99 replicates if slow.
  3. **μ(N)=μ(H)** and **φ(N)=φ(H):** LR tests by refitting M1 with `switching_trend=False` and with `switching_exog=False` (both are native statsmodels options, χ²(1)); Wald tests from the unrestricted covariance as a second number. **[our call: tests run in M1, the regime-definition model without TVTP]**
  4. Expected durations 1/(1 − pᵢᵢ), number of weeks with filtered P(H) > 0.5, regime σ² and μ.
  5. **φ decision:** reject φ(N)=φ(H) at 5% (LR) → `PHI_SWITCHING = True`, else `False`. **[our call: 5%; the plan gives no level, see §14]** Write `rq1_decision.json` (`phi_switching`, LR stat, p-value, sample end, git commit hash). The builder then sets `config.PHI_SWITCHING` by hand and commits; later notebooks assert they match.
  6. **Constrained M3 dry run** on 2005–2018: fit M2 and M3 with the chosen φ; confirm the constraint holds, LL(M3) ≥ LL(M2) ≥ LL(M1), β sign mapping (§8.4).
  7. **Forecaster check on real data:** `forecast.predicted_probs(...)` vs. statsmodels' `predicted_marginal_probabilities` for M1, M2, M3 (max abs diff < 1e-10).
- **Checks:** all of the above asserted; H has the higher σ² in every reported fit; no regime with < 10 weeks of filtered P > 0.5 (flag, not fail).

### 7.4 `04_rq2_explain.ipynb`

- **Purpose:** RQ2 in sample on **2005–2025**, supplement on **2005–2026**. Plan §3 RQ2, §6.1–6.2, §14.1.
- **Inputs:** `weekly_panel.csv`, `config.PHI_SWITCHING`.
- **Outputs:** `results/tables/rq2_coefficients.csv` (all transition coefficients for M2 and M3 in the deck's sign convention, SEs, z, p), `rq2_lr_tests.csv` (columns `test, model_restricted, model_full, lr_raw, stat, df, pvalue`; rows M2 vs M1, M3 vs M2, M3tone vs M2), `rq2_fit_summary.csv` (LL, AIC, BIC, durations), `rq1_tests_2005_2025.csv`, `rq2_m3tone.csv`, `harx_insample.csv`, `harx_ftest.csv` (columns `test, F, df_num, df_resid, n, pvalue`; rows `HARX_CI_vs_HARX_C`, `HARX_C_vs_HAR_h`), `garchx_insample.csv`; `results/fits/insample_filtered_2005_2026.csv` (date, model, xi_H filtered; used by notebook 07); figures `rq2_filtered_prob.png` (2005–2026, M1/M2/M3; same idea as deck slide 11, filtered not smoothed), `rq2_transition_probs.png` (fitted P(N→H) over time, M2 vs M3).
- **Steps:**
  1. Fit M1, M2, M3 (constrained) on 2005–2025. Standardize TVTP regressors with 2005–2012 stats (§9.3).
  2. β = Iran's coefficient in P(N→H), SE from the numerical Hessian of the constrained log-likelihood (§8.4).
  3. LR tests with df = the difference in **free** parameters, computed by `insample.lr_test` from the two fits, never typed in. M3 vs M2: df = 1 under `constrained`, df = 2 under `iran_both_logits`. M2 vs M1: df = 6 (three controls in two logits). Report the joint LR and, separately, the N→H Iran coefficient β with its z-test (the two coincide only when df = 1).
  4. **RQ1 reported fit on 2005–2025** (plan §3): rerun the 1-vs-2 LR and BIC, the μ and φ equality tests, expected durations and weeks per regime on 2005–2025 → `rq1_tests_2005_2025.csv`. The bootstrap is optional here (only if time). φ stays frozen from 2005–2018 whatever these show.
  5. Supplement: refit all three on 2005–2026, same tables with suffix `_2026`, clearly labelled as a supplement.
  6. Filtered probabilities for 2026 from the 2005–2025 parameters (filter run forward with fixed params; this is what notebook 07 uses for in-sample event plots).
  7. **M3-tone (added Nov 23) [decided].** Spec `M3tone`: M2 plus `iran_conflict` and `iran_coop` (`C.IRAN_TONE_VARS`) in the TVTP, **both restricted to the N→H logit** (same constraint idea as M3, both fixed to 0 in `p[1->0]`). It replaces `iran_gdelt`; it does not add to it. Fit on 2005–2025 with nesting start from M2 (both β = 0). Report:
     - β_conflict, β_coop (deck sign convention), SEs, z;
     - LR vs. M2 with df from free parameters (2 under `constrained`, 4 under `iran_both_logits`; computed by `lr_test`, never typed);
     - a Wald test of β_conflict = β_coop from `param_cov` (χ²(1)).

     Interpretation (for the markdown cell): the Wald test asks whether **the tone of coverage matters beyond its amount**. Equal coefficients mean only the volume of Iran news matters. Good news can also raise volatility (e.g. the 2015 nuclear deal moved supply expectations), so a positive β_coop is not a contradiction. Also compare LL and BIC of M3tone vs. M3 (non-nested, descriptive). OOS for M3tone only in notebook 08, if time. Outputs `rq2_m3tone.csv`.
  8. **Extra linear news models in sample (added Nov 23).** On the hourly sample (`HOURLY_SAMPLE_START` → 2025-12-26):
     - `HAR_h`, `HARX_C`, `HARX_CI` by OLS (`benchmarks.harx_insample`): coefficients, Newey-West SEs (4 lags), the F-test of HARX_CI vs. HARX_C, adjusted R².
     - `GARCH_t` (same custom likelihood with no regressors), `GARCHX_C`, `GARCHX_CI` (`benchmarks.garchx_fit`): γ's, SEs, LL, and LRs. For GARCHX_C vs. GARCH_t, the p-value is chi-bar-squared with k = 3; for GARCHX_CI vs. GARCHX_C it is 50:50 χ²₀/χ²₁, because γ = 0 is on the boundary (`insample.chibar_pvalue`).

     Outputs `harx_insample.csv`, `garchx_insample.csv`. These are descriptive for now; the status of the extra models stays open.
- **Checks:** constraint holds exactly (coefficient == 0.0) under `constrained`, for every restricted variable; LL nesting (incl. LL(M3tone) ≥ LL(M2), LL(GARCHX_CI) ≥ LL(GARCHX_C) ≥ LL(GARCH_t) up to tolerance); relabel applied; no degenerate fit; `config.M3_VARIANT` equals the value in `spec_freeze.json` (rerun this notebook if validation switched the variant; the variant applies to M3tone too).

### 7.5 `05_oos_forecasts.ipynb` — expensive

- **Purpose:** the expanding-window OOS loop. Plan §6.3–6.4, §7.
- **Inputs:** `weekly_panel.csv`, `config`.
- **Outputs:** `results/forecasts/forecasts_main.csv`, `results/fits/params_main.csv`, `results/fits/fitlog_main.csv`, `results/frozen/spec_freeze.json`; figure `oos_runtime_and_convergence.png` (fit time and restarts by week).
- **Stages** (a `STAGE` variable at the top):
  1. **`timing`:** run the **first 10 validation origins** (2012-12-28 → 2013-03-01) for each model, never a test-window origin; print seconds per fit; extrapolate total runtime; choose `C.N_RESTARTS` and `C.RESTART_EVERY` so the full main run fits in ≈ 2 h. Output goes to a throwaway run (`run_id = "timing"`) that is never resumed. **[our call]**
  2. **`validation`:** origins 2012-12-28 → 2018-12-21 (targets 2013-01-04 → 2018-12-28, 313 weeks). All models. Then review on validation only: convergence rate, label flips, restarts that improved LL, constrained-M3 stability. Decide and freeze in `config`: `SEARCH_REPS_*`, `N_RESTARTS`, `RESTART_EVERY`, `SIGMA2_RATIO_MIN`, `VAR_FLOOR_FRAC`, `MIN_DURATION`, `M3_VARIANT` (`"constrained"` or the plan §6.4 variant fallback `"iran_both_logits"`). Write `spec_freeze.json` (all frozen settings, their `settings_hash`, git hash) and set `config.SPEC_FROZEN = True`.
     - **[our call] stability rule for the variant fallback:** switch to `"iran_both_logits"` only if, on validation, the constrained fit fails to converge or violates LL(M3) ≥ LL(M2) − 1e-6 after restarts in more than 5% of weeks.
     - The extra models have nothing to freeze. Their validation-window losses are only used later to name the "best linear model" for one figure (§7.6).
  3. **`full`:** refuses to run unless `SPEC_FROZEN`. Resumes from the validation checkpoint **only if** the checkpoint's `settings_hash` equals the one in `spec_freeze.json`. Otherwise (validation changed a setting) it discards the checkpoint and reruns from origin 2012-12-28 with the frozen settings (≈ +40% runtime). Continues to the last origin 2026-10-30 (target 2026-11-06). Test and holdout are produced in one pass; splits are assigned by target date. The M3 variant never changes inside the run (§8.8).
- **Models per origin:** M1, M2, M3 (MS); GARCH(1,1)-t; HAR-RV; RW (return only); and the extra models (added Nov 23) `HAR_h`, `HARX_C`, `HARX_CI`, `GARCHX_C`, `GARCHX_CI` (`C.BENCHMARKS_MAIN`; registry in §9.6). The extra models are in the main run from the start (timing, validation, full) so the run is never repeated for them.
- **Main steps per origin t** (inside `oos.run_oos`, §8.8): build design through t → fit each MS model (warm start from t−1's canonical params, restarts per schedule, relabel) → one-step forecasts for t+1 → benchmarks and extra models → append to checkpoint.
- **Runtime estimate:** loglike ≈ 0.4 ms and an unconstrained TVTP fit ≈ 1.5 s on 1,100 obs (measured on synthetic data). Per origin: M1 ≈ 0.3 s, M2 ≈ 1–2 s, M3 ≈ 2–6 s (constrained refit + restarts), GARCH ≈ 0.1 s, HAR family ≈ ms, GARCH-X ≈ 0.1–0.5 s each (estimate; warm-started SLSQP, timed in the `timing` stage). 723 origins ≈ **1–2 h** for the main run (the extra models add < 15 min). Run overnight or while doing other work; the checkpoint makes interruption harmless.
- **Checks:** 723 target weeks per model and target; no target date ≤ its origin; every forecast uses `origin = date − 7 days`; `var` forecasts > 0; π_H in [0, 1]; the frozen `settings_hash` is stored in every checkpoint row and in the params and fitlog files (the forecasts file, §9.2, has no such column); fitlog convergence rate, `refit_from_scratch` count, non-converged M3 weeks and degenerate fits printed; GARCH-X non-converged weeks and HAR-family floored weeks printed.

### 7.6 `06_evaluation.ipynb`

- **Purpose:** every test and figure in plan §8, split **validation / test (2019–2025) / holdout (2026)**, test and holdout reported separately. Plan §3 RQ3, §8, §14.3.
- **Inputs:** `forecasts_main.csv`, `weekly_panel.csv` (actuals), `config` (events, thresholds).
- **Development:** if the full run is not finished yet, the code can be built against a fake `forecasts_main.csv` generated by `oos.fake_forecasts(panel)` (same schema; for building code only, never reported).
- **Guard:** test and holdout metrics only when `config.SPEC_FROZEN` is True. Validation metrics are shown as an appendix table.
- **Outputs:** tables in `results/tables/`; figures in `results/figures/` (names below).
- **RV measures (added Nov 23) [decided].** Every variance forecast (`target = "var"`) is scored against each RV measure in `C.RV_MEASURES`: **`rv_hourly` (main)**, `rv` (daily returns, secondary), `rv_gk` (Garman-Klass, appendix). Every volatility table has a `measure` column; tables and statements in the report lead with `rv_hourly`. The test and holdout windows (2019–2026) and validation (2013–2018) are fully covered by `rv_hourly`. Following Patton (2011), QLIKE ranks forecasts consistently against any conditionally unbiased proxy, so one forecast per model is scored against all three. Models fitted on one measure (e.g. `HAR` on `rv`, `HAR_h` on `rv_hourly`) may gain from level matching when scored on their own measure; the clean ladder comparisons are therefore made on `rv_hourly`.

| Target | Tables | Figures |
|---|---|---|
| Volatility (main) | `vol_losses.csv` (QLIKE, RMSE per model × split × measure); `vol_mz.csv` (a, b, HAC SEs, joint Wald (0,1) p; every model × split × measure); `vol_dm.csv` (QLIKE and squared error, DM-HLN; pairs: M3 vs M2, M1, GARCH_t, HAR, HAR_h, **HARX_CI, GARCHX_CI** ("do regimes beat a linear news effect?"); **HARX_CI vs HARX_C** and **GARCHX_CI vs GARCHX_C** (Iran in a linear model); HARX_C vs HAR_h and GARCHX_C vs GARCH_t (controls in a linear model); nested pairs carry the nested-model caveat); `vol_cw_gw.csv` (Clark-West on squared error for the nested pairs M3 vs M2, HARX_CI vs HARX_C, GARCHX_CI vs GARCHX_C; Giacomini-White unconditional and conditional, QLIKE, M3 vs M2 only; caveats in §14 items 5 and 7) | `vol_fc_vs_actual_{test,holdout}.png` (`rv_hourly` and forecasts of M2, M3, HARX_CI, GARCHX_CI, Iran event weeks shaded); **`cumloss_M2_minus_M3.png`** (Σ(L_M2 − L_M3), QLIKE vs `rv_hourly`, 2019–2026, split line at Jan 1, 2026, event weeks marked; the same figure vs `rv` saved as `cumloss_M2_minus_M3_rv.png`); **`cumloss_bestlinear_minus_M3.png`** (Σ(L_best − L_M3), QLIKE vs `rv_hourly`; "best linear" defined below); `vol_error_acf.png` |
| Returns (check) | `ret_eval.csv` (RMSE vs RW, ratio, sign hit rate, Pesaran-Timmermann stat and p, per MS model × split) | `ret_fc_vs_actual.png`; `ret_error_acf.png` |
| High-vol prob (side) | `phigh_eval.csv` (Brier, hit rate on weeks with \|r\| > training 90th pct, base rate; optional Brier of mixture-implied P(\|r\| > thr)) | `phigh_reliability.png` (10 bins); `phigh_vs_bigmoves.png` |
| All | `ljung_box.csv` (forecast errors per model × target, and per measure for `var`; lags 4, 8, 13) | — |

- **[our call] details:** QLIKE = RV/f − ln(RV/f) − 1 (Patton 2011). DM loss: QLIKE primary, squared error secondary. DM: `lags=0` with `harvey_adj=True, horizon=1` (at h = 1 the loss differential is serially uncorrelated under the null, which is the case the HLN correction is built for). MZ uses Newey-West with 4 lags. GW instrument: `1{iran_gdelt(t) > 75th percentile of 2013–2018}`, dated at the **origin** t (it conditions the forecast for t+1; `gw_instrument` takes origins). The threshold comes from the validation window because 2013–2018 is the daily-feed era; the 2005–2012 backfile may sit at a different level (Apr 2013 break). Assert that the instrument's share of 1s in the test window is between 10% and 90%; otherwise stop and report (§14, item 6). π_H "hit" = π_H > 0.5. Mixture-implied exceedance probability is already in `forecasts_main.csv` (`target = p_exceed`), so the optional score costs one line.
- **"Best linear model" [our call].** Among `HAR, HAR_h, HARX_C, HARX_CI, GARCH_t, GARCHX_C, GARCHX_CI`, take the one with the lowest mean QLIKE vs `rv_hourly` on the **validation** window (2013–2018), via `evaluate.best_linear_model`. Choosing it on the test window would favor it by construction. Its name is printed and saved in `vol_best_linear.csv`.
- **Multiple comparisons [our call].** The headline remains M3 vs M2 on `rv_hourly` (pre-specified). The comparisons with extra models are labelled secondary in every table (`family` column: `main`, `benchmark`, `extra`), and no p-value is adjusted. The markdown states how many tests were run.
- **Checks:** forecasts and actuals aligned on target date; no NaN losses for any measure in 2013–2026; DM/CW/GW reproduce the toy cases in `tests/test_evaluate.py`; cumulative-loss series ends at the sum of per-week differences.

### 7.7 `07_events.ipynb`

- **Purpose:** event case studies, plan §9; fact checks, plan §16.
- **Inputs:** `weekly_panel.csv`, `data/raw/brent_daily.csv` (1987+, for fact checks), `data/raw/brent_intraday_daily.csv` (optional, for intraday highs), `forecasts_main.csv`, `insample_filtered_2005_2026.csv`, `config.EVENTS`, `config.NON_IRAN_EVENTS`.
- **Outputs:** `results/tables/event_table.csv` (MVP item: per event, event week, M2 and M3 values at weeks −4…+4 for in-sample filtered P(H), OOS π_H at the target week where available, weekly QLIKE losses vs `rv_hourly` for M2, M3 and HARX_CI where available, and `iran_conflict`, `iran_coop` at each week); `results/tables/deck_fact_checks.csv`; figures `event_<id>.png` (one per event, 9 Iran + 3 non-Iran) and `events_grid.png`.
- **Steps:**
  1. Map each event date to its event week with `weekly.week_ending` (weekend rule built in).
  2. ±4-week window, five panels:
     - Top: in-sample filtered P(H) (M2, M3; 2026 filtered with 2005–2025 params).
     - Second: OOS π_H (M2, M3), plotted at its **target** date. The x-axis is the event-week offset k = −4…+4, so the value at k = 0 is the forecast made at the Friday close before the event week. Y-axis label: "P(high vol in week k), forecast at end of week k−1".
     - Third (added Nov 23): `rv_hourly` (actual) with the variance forecasts of M2, M3 and HARX_CI, log scale.
     - Fourth: weekly QLIKE loss vs `rv_hourly` (M2, M3, HARX_CI).
     - Bottom (added Nov 23): the conflict/cooperation split, `iran_conflict` and `iran_coop`.

     Only plot what exists: Dec 2011 and 2008 fall in the training window, so they have no OOS rows (§14, item 14); their OOS panels show only the actual `rv_hourly`.
  3. Non-Iran comparisons: 2008, COVID-19, Russia–Ukraine (anchor dates in config, §8.1).
  4. Fact checks for the report (plan §16; the deck is not compared, it was a feasibility pitch): Brent Feb 27 → 2026 peak and its date; the week-to-Mar-6, 2026 weekly change and its rank among all Friday-to-Friday changes since 1987, from the frozen `brent_daily.csv` (starts 1987-05-20; same week rules as the panel); the reaction move and 5-day move for every event in `config.EVENTS` (spot series). Where helpful, the Dukascopy daily `high` (`brent_intraday_daily.csv`) gives the intraday futures-CFD peak next to the spot close (e.g. the 2026 peak). It is labelled as a different instrument, never mixed into the spot numbers. No network access.
- **Checks:** every event week exists in the panel; holdout event marked "holdout, not in the test" in every output.

### 7.8 `08_robustness.ipynb`

- **Purpose:** plan §10 in its priority order (§12), plus the post-2013 GDELT dummy (§4.3). Each section has its own `FORCE_<check>` flag and saves its own outputs, so a half-finished notebook still yields completed checks.
- **In-sample runs use 2005–2025; OOS runs reuse `oos.run_oos` with `run_id=<check>` and are evaluated with the same `evaluate` functions as notebook 06.** **[our call: which checks are in-sample vs. OOS, see table]**
- **OOS robustness runs** start at origin 2018-12-28 (first target 2019-01-04) and end at target 2025-12-26 (`C.ROBUST_OOS_TARGETS`; the 2026 holdout is untouched) with a fresh first fit (full first-fit procedure, `SEARCH_REPS_FIRST`) on data through 2018, then warm-start weekly with the frozen settings. They never rerun validation. **If trimming is needed**, AI-GPR, GDELT-geo and OVX run **in-sample only**.

| # | Check | In sample (2005–2025) | OOS | Output |
|---|---|---|---|---|
| 1 | **Placebo** (circular shifts of Iran) | 200 shifts: LR gain M3 vs M2 vs. the actual LR; p = (1 + #{placebo LR ≥ actual}) / (B + 1) | ~20 shifts, M3 only (M2 has no Iran), 365 origins (targets 2019–2025), QLIKE gain M2 − M3; p = (1 + #{placebo gain ≥ actual}) / (B + 1). **Cut first** if time is short | `placebo_insample.csv`, `placebo_oos.csv`, `placebo_hist.png` |
| 2 | **Iran measure: AI-GPR** (`aigpr_iran` replaces `iran_gdelt`) | β, LR | M3 run `aigpr` (if time) | `rob_aigpr_*.csv` |
| 2b | **Iran measure: GDELT actor + location** (`iran_gdelt_geo`) | β, LR | M3 run `gdelt_geo` (if time) | `rob_gdelt_geo_*.csv` |
| 2c | **Post-2013 dummy** in both logits (M2 and M3) | β with the dummy | — | `rob_post2013.csv` |
| 3 | **AR term**: the φ setting not chosen in RQ1 | fit comparison | M1–M3 run `ar_alt` (plan: "forecasts with and without") | `rob_ar_*.csv` |
| 4 | **Distribution**: fitted two-normal mixture vs. weekly returns | Histogram + the model's average one-step predictive density (1/T) Σₜ Σᵢ πᵢ(t\|t−1) N(mᵢ(t), σᵢ²); tail table: empirical vs. model P(\|r\| > thr). Student-t: **cut first**; otherwise a limitation | — | `rob_distribution.png`, `rob_tails.csv` |
| 5 | **Iran in the mean** (`iran_gdelt(t−1)` added to the mean, one common coefficient) | coefficient, SE, LR vs. M3 | — | `rob_iran_mean.csv` |
| 6 | **Market fear: OVX** replaces VIX; sample starts with OVX (May 2007), standardization on 2007–2012 | β, LR | M2, M3 run `ovx` (if time) | `rob_ovx_*.csv` |
| 7 | **Hourly lagged control** (added Nov 23): `log_rv_hourly` replaces `log_rv` as the TVTP control in M2 and M3 (`controls=C.TVTP_CONTROLS_HOURLY`); sample from `HOURLY_SAMPLE_START`, standardization on `STANDARDIZE_WINDOW_HOURLY` | β, LR M3 vs M2 | M2, M3 run `hourly_ctrl` (if time) | `rob_hourly_ctrl_*.csv` |
| 8 | **Tone variants** (added Nov 23): `M3tone` OOS; `M3toneL` = M3 + `iran_tone`; `M3gold` = M3 + `iran_goldstein` (in both, the added variable is also N→H only) | M3toneL, M3gold: β's, LR vs M3 (df from free params). M3tone is already in notebook 04 | M3tone run `tone` (if time) | `rob_tone_*.csv` |
| 9 | **Placebo for HARX_CI** (added Nov 23): circular shifts of `iran_gdelt`, same offsets rule as #1 | 200 shifts: in-sample gain LR = n·ln(SSR_C / SSR_CI) vs. the actual | 200 shifts (OLS is cheap): QLIKE gain HARX_C − HARX_CI vs `rv_hourly`, targets 2019–2025 | `placebo_harx.csv`, `placebo_harx_hist.png` |
| E1 | **Optional extension: Markov-switching HAR** (added Nov 23), spec below. **First to cut** | LR MSHAR_CI vs MSHAR_C, β | run `mshar` (if time) | `ext_mshar_*.csv` |

- **Order of work in notebook 08 [our call]:**
  1. placebo #1 (in-sample);
  2. HARX placebo #9 (minutes);
  3. hourly control #7 (in-sample);
  4. AI-GPR #2;
  5. tone variants #8 (in-sample);
  6. AR term #3;
  7. distribution #4;
  8. Iran in mean #5;
  9. OVX #6;
  10. GDELT geo #2b and post-2013 dummy #2c;
  11. the OOS runs, queued overnight in this order: `ar_alt`, `hourly_ctrl`, `aigpr`, `tone`, `gdelt_geo`, `ovx`, OOS placebo #1;
  12. E1 last.
- **E1 Markov-switching HAR [our call: specification].** Built only if everything above is done.
  - Model: `MarkovRegression` on y(t) = `log_rv_hourly(t)` with `trend="c"`, `switching_variance=True` and non-switching HAR exog [ln rv_hourly(t−1), ln mean rv_hourly(t−4..t−1), ln mean rv_hourly(t−13..t−1)] (`switching_exog=False`, to keep it small).
  - TVTP: `exog_tvtp` = [1, gpr(t−1), vix(t−1)] for `MSHAR_C`, plus `iran_gdelt(t−1)` in N→H only for `MSHAR_CI`, fitted with the same `fit_constrained`. Lagged RV is already in the mean.
  - Labels: H = the regime with the **higher intercept** (`MSSpec.canonical_by = "const"`), since in a log-RV model "high volatility" is a level, not a residual variance.
  - Forecast: E[RV(T+1)] = Σᵢ πᵢ exp(mᵢ + σᵢ²/2), with mᵢ = cᵢ + b′x_next (log-normal mean within each regime).
  - Sample from `HOURLY_SAMPLE_START` + 13 weeks; standardization on `STANDARDIZE_WINDOW_HOURLY`. In sample (to 2025) first; OOS run `mshar` (from origin 2018-12-28, like the other robustness runs) only if time.
  - It reuses `ms.py`, `design.py`, `forecast.py` and `oos.py` through `MSSpec.kind = "mshar"` (§8.4–8.6), with no new module but **new additive code built in Plan 3**: `ms.make_mshar_specs` (`MSHAR_C`, `MSHAR_CI`), and the `mshar` branches in `design.build_design`, `ms.fit_model`, `forecast.one_step` and `oos.run_oos`. Existing signatures do not change.

- **Placebo details [our call]:** shifts drawn uniformly from offsets at least 52 weeks away from 0 (mod the sample length), seed `C.SEED`. Shifts wrap within **2005–2025 only**, so 2026 values never enter (plan §5: 2026 never used to choose anything). OOS placebos are evaluated on 2019–2025 only. The HARX placebo (#9) wraps within `HOURLY_SAMPLE_START` → 2025-12-26 (its estimation sample) and reuses the same 200 offsets for the in-sample and OOS gains; HARX_C's forecasts do not depend on Iran, so they are computed once.
- **Iran-in-the-mean [our call]:** common (non-switching) coefficient on M3, using `switching_exog=[PHI_SWITCHING, False]`; one parameter, one test.
- **Extra outputs of notebook 08 (Plan 3):** `results/tables/rob_status.csv` (one row per check: `done`, `cut`, `in_sample_only`, `failed`; feeds the limitations table), `rob_fit_health.csv` (per run and model: convergence share, non-converged, degenerate and `refit_from_scratch` counts, median seconds), `handoff_numbers.csv` (`name, value, source_file, note`, built from the tables, no hand-typed numbers). Notebook 07 writes `results/tables/deck_fact_checks.csv` (computed facts only; no deck claims are compared).
- **Runtime:** in-sample placebo ≈ 200 × 5–10 s ≈ 20–35 min. Each extra OOS run ≈ 0.5–1.5 h. OOS placebo ≈ 20 × 365 origins × 4 s ≈ 8 h (only with spare time; parallelizable with `joblib` across shifts). HARX placebo ≈ 200 × 365 OLS fits, a few minutes. E1 in sample ≈ minutes; E1 OOS ≈ 1 h (estimate).
- **Checks:** each robustness run's fitlog convergence printed; the placebo's actual-data LR equals notebook 04's LR (same code path); the HARX placebo's unshifted gain equals notebook 04's HARX_CI vs HARX_C numbers and notebook 06's QLIKE difference on 2019–2025.

### 7.9 Coverage map: plan → notebook

| Plan item | Notebook |
|---|---|
| §3 RQ1: 1 vs 2 regimes (bootstrap LR, BIC) | 03 |
| §3 RQ1: Wald/LR μ(N)=μ(H), φ(N)=φ(H); durations; weeks per regime | 03 |
| §3 RQ1: φ decided on 2005–2018 and frozen; 2005–2025 reported fit (equality tests, durations, weeks per regime, 1-vs-2 LR/BIC) | 03 (decision), 04 (`rq1_tests_2005_2025.csv`) |
| §3 RQ2: β sign/significance; LR M3 vs M2; 2005–2026 supplement | 04 |
| §3 RQ3: QLIKE, DM, CW, GW (M3 vs M2); DM vs M1, GARCH-t, HAR; test and holdout separately | 05 → 06 |
| §4.3 GDELT level shift at Apr 2013; coverage through Nov 6; 2.0 fallback | 01, 02 (plot), 08 (dummy) |
| §4.4 data table (mean, SD, AR(1), ADF) and plots | 02 |
| §8 volatility: QLIKE, RMSE, MZ (HAC, joint), DM-HLN, CW, GW | 06 |
| §8 returns: RMSE vs RW, sign hit rate, Pesaran-Timmermann | 06 |
| §8 high-vol prob: Brier, reliability, hit rate; optional mixture exceedance Brier | 06 |
| §8 figures: forecast vs actual (events shaded), cumulative loss difference, error ACF + Ljung-Box | 06 |
| §9 nine events ±4 weeks, filtered P(H) M2 vs M3, weekly losses; non-Iran comparisons | 07 |
| §10 Iran in mean, AR term, Iran measure (AI-GPR, GDELT geo), placebo (in-sample 200+, OOS ~20), distribution, OVX | 08 |
| §12 MVP event table with M2 vs M3 probabilities | 07 |
| §16 facts to verify | 07 |
| **Nov 23 additions (professor feedback)** | |
| Intraday RV: Dukascopy fetch, roll rule, terms, daily file | 01 |
| `rv_hourly`, `rv_gk` in the panel; hourly vs daily level check | 02 |
| `rv_hourly` as main evaluation measure, `rv` secondary, `rv_gk` appendix | 06 |
| GDELT sentiment: conflict/coop shares, tone, Goldstein (columns, plots) | 01, 02 |
| M3-tone in sample (LR vs M2, Wald β_conflict = β_coop) | 04 |
| M3-tone OOS; tone/Goldstein-level variants | 08 |
| HAR_h, HARX_C, HARX_CI, GARCHX_C, GARCHX_CI: in-sample coefficients, γ and LR (chi-bar) | 04 |
| Same models OOS | 05 |
| DM M3 vs HARX_CI/GARCHX_CI; HARX_CI vs HARX_C, GARCHX_CI vs GARCHX_C (DM, CW); MZ for all; best-linear cumulative loss | 06 |
| HARX_CI forecast and conflict/coop split in event plots; Dukascopy intraday highs in fact checks | 07 |
| M2/M3 with `log_rv_hourly` control; HARX placebo; E1 MS-HAR (optional) | 08 |

---

## 8. Module design (`src/brent/`)

Signatures show names, inputs and outputs; types are indicative. Dates are `pd.Timestamp` Fridays unless noted.

### 8.1 `config.py`

All frozen constants and decisions; nothing else imports constants from anywhere else.

```python
SEED = 405
SAMPLE_START = "2005-01-07"; FREEZE_DATE = "2026-11-06"; FETCH_START = "2004-12-01"
TRAIN = ("2005-01-07", "2012-12-28")        # 417 weeks
VALID = ("2013-01-04", "2018-12-28")        # 313
TEST = ("2019-01-04", "2025-12-26")         # 365
HOLDOUT = ("2026-01-02", "2026-11-06")      # 45
RQ1_END = "2018-12-28"; RQ2_END = "2025-12-26"
STANDARDIZE_WINDOW = TRAIN                  # OVX run: ("2007-05-11", "2012-12-28")
HIGHVOL_PCTL = 90                           # of |r| on TRAIN
GW_IRAN_PCTL = 75; GW_WINDOW = VALID        # our call: threshold from 2013-2018 (daily-feed era)
TVTP_CONTROLS = ["gpr", "vix", "log_rv"]; IRAN_VAR = "iran_gdelt"
TVTP_CONTROLS_HOURLY = ["gpr", "vix", "log_rv_hourly"]       # robustness #7 only
IRAN_TONE_VARS = ("iran_conflict", "iran_coop")              # M3tone, both N->H only
# --- Intraday RV (added 2026-11-23) ---
DUKASCOPY_INSTRUMENT = "BRENT.CMD/USD"      # dukascopy-node id "brentcmdusd"
INTRADAY_START = "2006-11-17"               # earliest Dukascopy bar per README; verified in notebook 01
INTRADAY_END_UTC = "2026-11-06 22:00"       # bars starting at/after this are dropped (freeze)
INTRADAY_PRICE_SIDE = "bid"
INTRADAY_DAY_END_UTC = 22                   # day d = bars starting in [d-1 22:00, d 22:00) UTC
INTRADAY_ROLL_SCAN = {"k": 8.0, "window": 120}   # roll_candidates defaults
INTRADAY_ROLL_RULE: dict | None = None      # set in notebook 01 with the user (§6.8)
INTRADAY_REDISTRIBUTE: bool | None = None   # set in notebook 01 from Dukascopy's terms
HOURLY_SAMPLE_START = "2006-11-24"          # first full Sat-Fri week with rv_hourly; verified in notebook 02
STANDARDIZE_WINDOW_HOURLY = ("2007-01-05", "2012-12-28")   # 313 weeks; hourly-sample models and E1
MIN_HOURS_WEEK = 60                         # flag (not drop) weeks with fewer hourly returns
RV_LEVEL_RATIO_BAND = (0.8, 1.25)           # mean(rv_hourly)/mean(rv) on 2007-2012; outside -> tell the team
RV_MEASURES = ("rv_hourly", "rv", "rv_gk"); MAIN_RV = "rv_hourly"
# --- Extra models (added 2026-11-23; status open) ---
BENCHMARKS_MAIN = ("GARCH_t", "HAR", "RW", "HAR_h", "HARX_C", "HARX_CI", "GARCHX_C", "GARCHX_CI")
LINEAR_MODELS = ("HAR", "HAR_h", "HARX_C", "HARX_CI", "GARCH_t", "GARCHX_C", "GARCHX_CI")  # best-linear pool
X_CONTROLS = ("gpr", "vix")                 # news/fear regressors of HARX_C and GARCHX_C (+ IRAN_VAR in *_CI)
X_RV_COL = "rv_hourly"                      # RV used by HAR_h, HARX_*, GARCHX_*; "rv" only in the no-intraday fallback (§11)
X_SAMPLE_START = HOURLY_SAMPLE_START        # their sample start; SAMPLE_START in that fallback
GARCHX_SCALE_WINDOW = STANDARDIZE_WINDOW_HOURLY   # level regressors divided by their mean here
PHI_SWITCHING: bool | None = None           # set from notebook 03, then frozen
M3_VARIANT = "constrained"                  # or "iran_both_logits" (set in notebook 05 validation)
SPEC_FROZEN = False                         # True after notebook 05 validation stage
SEARCH_REPS_FIRST = 30; N_RESTARTS = 3; RESTART_EVERY = 13   # last two revisited in 05 timing
SIGMA2_RATIO_MIN = 1.5                      # restart trigger; frozen at validation
VAR_FLOOR_FRAC = 0.05; MIN_DURATION = 2.0   # degeneracy rule (§8.4); frozen at validation
GDELT_SPLICE_WEEK = None                    # set only if the GDELT 2.0 splice is needed
# --- Robustness (added 2026-11-23, Plan 3) ---
ROBUST_OOS_TARGETS = ("2019-01-04", "2025-12-26")   # robustness OOS targets = TEST; the 2026 holdout is untouched
OVX_START = "2007-05-11"; OVX_STD_WINDOW = ("2007-05-11", "2012-12-28")   # first week with OVX; OVX run standardization
DUMMY_POST2013_WEEK = "2013-04-05"          # first Friday week on the daily-feed side of the GDELT switch
TAIL_PCTLS = (90, 95, 99)                   # TRAIN percentiles of |r| for the tail table (rob_tails.csv)
MZ_HAC_LAGS = 4                             # Newey-West lags in Mincer-Zarnowitz
RV_FLOOR_FRAC = 0.01                        # HAR floor = 1% of min training RV
BOOT_REPS = 199; PLACEBO_IN = 200; PLACEBO_OOS = 20; PLACEBO_MIN_SHIFT = 52
GDELT_TABLE = "gdelt-bq.full.events"; GDELT_MAX_BYTES = 200 * 10**9
GDELT_SWITCH_DATE = "2013-04-01"; GDELT_FEED_RULE = "dateadded"   # or "sourceurl"; decided after D1
AIGPR_URL = "<fill in>"; AIGPR_PUBLICATION_LAG_MONTHS = 1   # see §9.1
EVENTS = [  # id, date, label, confounder (plan §9, frozen)
  ("hormuz_2011", "2011-12-27", "Iran threatens to close Hormuz; EU embargo Jan 23", "euro debt crisis"),
  ("jcpoa_2018", "2018-05-08", ...), ("tankers_2019", "2019-06-13", ...), ("abqaiq_2019", "2019-09-14", ...),
  ("soleimani_2020", "2020-01-03", ...), ("iran_israel_apr2024", "2024-04-13", ...),
  ("iran_israel_oct2024", "2024-10-01", ...), ("israel_iran_2025", "2025-06-13", ...),
  ("hormuz_2026", "2026-02-28", ...),
]
NON_IRAN_EVENTS = [  # our call: anchor dates (plan names the episodes only)
  ("lehman_2008", "2008-09-15"), ("covid_2020", "2020-03-09"), ("russia_ukraine_2022", "2022-02-24"),
]
SIZE_LIMIT_FILE = 10 * 2**20; SIZE_LIMIT_DATA = 5 * 2**20; SIZE_LIMIT_TOTAL = 20 * 2**20
SIZE_LIMIT_NOTEBOOK = 5 * 2**20; SIZE_LIMIT_NOTEBOOKS = 15 * 2**20
```

**Decisions that wait for data, and where they live.** Plans branch on these config values; nothing else carries the decision.

| Config value | Default | Set in | Possible values and effect |
|---|---|---|---|
| `GDELT_FEED_RULE` | `"dateadded"` | 01 (after D1) | `"dateadded"` or `"sourceurl"`: how `render_q1` labels backfile vs. daily rows |
| `GDELT_SPLICE_WEEK` | `None` | 01 (only if the 1.0 feed stops early) | a Friday: from that week on, GDELT columns come from 2.0 + level shift; panel gets `gdelt_source` |
| `AIGPR_URL` | `"<fill in>"` | 01 | the file URL |
| `INTRADAY_ROLL_RULE` | `None` | 01, with the user | `{"kind": "none"}`, `{"kind": "threshold", "k", "window"}`, `{"kind": "timestamps", "timestamps": [...]}`; `intraday_daily` refuses `None` |
| `INTRADAY_REDISTRIBUTE` | `None` | 01 (terms of use) | `True`: intraday CSVs committed; `False`: git-ignored (§5.3) |
| `INTRADAY_START`, `HOURLY_SAMPLE_START` | `"2006-11-17"`, `"2006-11-24"` | 01, 02 (verified) | corrected to the real first bar / first full week if they differ |
| `PHI_SWITCHING` | `None` | 03 | `True`/`False`: `switching_exog` in every MS-AR spec |
| `M3_VARIANT` | `"constrained"` | 05 validation | `"constrained"` or `"iran_both_logits"`; applies to M3, M3tone, M3toneL, M3gold, MSHAR_CI and every M3 robustness run |
| `SPEC_FROZEN` and the search/restart/degeneracy settings | as above | 05 validation | frozen in `spec_freeze.json` |

### 8.2 `paths.py`

- `ROOT`, `RAW`, `PROCESSED`, `RESULTS`, `FORECASTS`, `FITS`, `TABLES`, `FIGURES`, `FROZEN`, `CHECKPOINTS` (all `Path`, found from the package location, never from the notebook's cwd).
- `save_table(df, name) -> Path`, `check_budget(raise_=True) -> pd.DataFrame` (file sizes over the limits), `git_hash() -> str`.

### 8.3 `sources.py`

```python
fetch_fred(series_id: str, start: str, end: str) -> pd.DataFrame          # [date, value]
fetch_gpr_daily(url: str, start: str, end: str) -> pd.DataFrame           # [date, gprd]
fetch_aigpr_iran(url: str, start_month: str, end_month: str) -> pd.DataFrame  # [month, aigpr_iran]
gdelt_credentials() -> google.oauth2.credentials.Credentials
gdelt_dry_run(sql: str, project_id: str, credentials) -> int              # bytes processed
gdelt_query(sql: str, project_id: str, credentials, max_bytes: int) -> pd.DataFrame
GDELT_SQL_D1: str                                                         # template (§6.4)
render_q1(table: str, feed_rule: str, version: str = "v1",
          start: int = 20050101, end: int = 20261106) -> str             # Q1 incl. sentiment columns (§6.4)
gdelt_daily_frame(q1: pd.DataFrame) -> pd.DataFrame                       # day_key -> date; columns GDELT_DAILY_COLUMNS
GDELT_DAILY_COLUMNS: tuple[str, ...]                                      # the gdelt_daily.csv columns (§6.4)
save_snapshot(df: pd.DataFrame, name: str, source: str, url: str) -> Path  # CSV + manifest row
load_raw(name: str) -> pd.DataFrame

# Dukascopy (added Nov 23; §6.8). Hourly frames: columns timestamp_utc (tz-aware UTC, bar start),
# open, high, low, close, volume. Never written to disk.
fetch_dukascopy_hourly(instrument: str = C.DUKASCOPY_INSTRUMENT, start: str = C.INTRADAY_START,
                       end: str = "2026-11-07", side: str = C.INTRADAY_PRICE_SIDE,
                       method: str = "python") -> pd.DataFrame       # "python" (dukascopy-python) or "node" (npx, temp dir)
normalize_hourly(raw: pd.DataFrame, end_utc: str = C.INTRADAY_END_UTC) -> pd.DataFrame  # UTC, sort, dedupe, drop 0-volume, freeze cut
intraday_day(ts, day_end_utc: int = C.INTRADAY_DAY_END_UTC) -> pd.DatetimeIndex  # naive date = (ts_utc + 2h).normalize()
hourly_returns(hourly: pd.DataFrame) -> pd.Series                   # 100*diff(ln close) across gaps; index = timestamp_utc; first NaN
roll_candidates(hourly: pd.DataFrame, k: float = 8.0, window: int = 120) -> pd.DataFrame
    # [timestamp_utc, ret, med_abs, ratio, gap_hours], ratio = |ret| / median |ret| of the previous `window` returns; rows with ratio > k
roll_mask(hourly: pd.DataFrame, rule: dict) -> pd.Series            # bool per bar: True = return removed
intraday_daily(hourly: pd.DataFrame, rule: dict | None,
               day_end_utc: int = C.INTRADAY_DAY_END_UTC) -> tuple[pd.DataFrame, pd.DataFrame]
    # (daily [date, rv_intraday, n_hours, open, high, low, close], removed [timestamp_utc, date, ret, ratio, reason]); rule None -> ValueError
fetch_dukascopy_daily(rule: dict, **fetch_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]  # fetch -> normalize -> intraday_daily
```

The Google imports happen inside the GDELT functions, and `dukascopy_python` inside `fetch_dukascopy_hourly`, so `import brent` works without the `gdelt` or `intraday` groups. Returns in `sources.py` use `np.diff`/`.diff()`, never `.shift(` (grep rule, §8.5).

### 8.4 `ms.py`

```python
@dataclass(frozen=True)
class MSSpec:
    name: str                       # "M1" | "M2" | "M3" | "M3tone" | "M3toneL" | "M3gold" | "MSHAR_C" | "MSHAR_CI" | robustness names
    tvtp_vars: tuple[str, ...]      # () for M1; controls for M2; controls + iran_vars for M3-type
    iran_vars: tuple[str, ...]      # restricted variables (N->H only when iran_only_NH); () for M1/M2.
                                    # M3: ("iran_gdelt",); M3tone: IRAN_TONE_VARS; M3toneL: ("iran_gdelt", "iran_tone")
    iran_only_NH: bool              # True under M3_VARIANT == "constrained"
    phi_switching: bool             # ignored for kind "mshar" (HAR slopes do not switch)
    mean_exog: tuple[str, ...] = () # e.g. ("iran_gdelt",) for Iran-in-the-mean
    kind: str = "msar"              # "msar": y = r, MS-AR(1) (main); "mshar": y = log_rv_hourly, HAR exog (E1)
    canonical_by: str = "sigma2"    # H = higher sigma2 ("msar") or higher intercept ("const", for "mshar")
    std_window: tuple[str, str] = C.STANDARDIZE_WINDOW   # STANDARDIZE_WINDOW_HOURLY for hourly-sample specs
    start: str = C.SAMPLE_START     # first panel week used; HOURLY_SAMPLE_START for hourly-sample specs

make_specs(phi_switching: bool, variant: str = C.M3_VARIANT,
           controls: tuple[str, ...] = tuple(C.TVTP_CONTROLS),
           iran_var: str = C.IRAN_VAR) -> dict[str, MSSpec]    # "M1", "M2", "M3" (the ladder; robustness passes other args)
with_iran(spec: MSSpec, iran_vars: tuple[str, ...], name: str) -> MSSpec  # swap restricted vars (AI-GPR, geo, placebo, tone)

@dataclass
class MSFit:
    spec: MSSpec; params: pd.Series  # canonical labels: regime 0 = N, regime 1 = H
    loglike: float; converged: bool; n_restarts_used: int; relabeled: bool
    refit_from_scratch: bool; degenerate: bool; seconds: float; model: MarkovRegression

make_model(design: Design, spec: MSSpec) -> MarkovRegression
fit_ms(design, spec, start_params: pd.Series | None, search_reps: int, n_restarts: int, seed: int) -> MSFit
canonicalize(params: pd.Series, model, by: str = "sigma2") -> tuple[pd.Series, bool]   # → N=0, H=1 (H = higher sigma2, or higher const for "mshar")
fit_constrained(model, start_params: pd.Series, fixed: dict[str, float], n_restarts: int, seed: int) -> tuple[pd.Series, float, bool]
transition_coefs_deck(params: pd.Series, tvtp_names: list[str]) -> pd.DataFrame  # rows: N->H, H->N logits
expected_durations(params, model, exog_tvtp=None) -> dict[str, float]
param_cov(model, params, fixed: dict[str, float]) -> pd.DataFrame  # numeric Hessian on free params
```

Implementation notes (verified on statsmodels 0.15.0):
- Model: `MarkovRegression(y, k_regimes=2, trend="c", exog=X, switching_exog=phi_switching, switching_variance=True, exog_tvtp=Z)`; M1 omits `exog_tvtp`. Not `MarkovAutoregression` (plan §6.4).
- Parameter names are **interleaved by TVTP column**: `p[0->0].tvtp0, p[1->0].tvtp0, p[0->0].tvtp1, p[1->0].tvtp1, …, const[0], const[1], x1[0], x1[1], sigma2[0], sigma2[1]`. Always index by name through a mapping `tvtp_name_map = {"const": 0, "gpr": 1, …}` built in `make_model`; never by position.
- For k = 2 statsmodels models the logits of **p[0->0] = P(Sₜ=0 | Sₜ₋₁=0)** and **p[1->0] = P(Sₜ=0 | Sₜ₋₁=1)** (transitions *into* regime 0), with the same regressors in both.
- **Canonical labels [our call]:** after every fit, `canonicalize` permutes so that regime 0 = N and regime 1 = H (higher σ²). The log-likelihood is unchanged (tested). If statsmodels returned H = 0, the permutation is: swap `const`, `x1`, `sigma2` between regimes, and set new `p[0->0]` coefs = −old `p[1->0]` coefs, new `p[1->0]` coefs = −old `p[0->0]` coefs (because P(S=0|·) in the new labels is 1 − P(S=0|·) in the old ones). For M1 (constant probabilities, transformed scale) the same swap is done on probabilities: new p00 = 1 − old p10, new p10 = 1 − old p00. Why: downstream code then has one fixed mapping instead of two cases, and warm starts stay in the same labeling week to week.
- **In canonical labels:**

  | Deck quantity | statsmodels logit | Sign |
  |---|---|---|
  | P(N→H) = Λ(c + γ′x + β·Iran) | P(N→H) = 1 − p[0->0] | deck coef = **−** coef in `p[0->0]` |
  | P(H→N) = Λ(d + δ′x) | P(H→N) = p[1->0] | deck coef = **+** coef in `p[1->0]` |

  So "Iran only in N→H" = **fix Iran's coefficient in `p[1->0]` to 0**, and **β = −(Iran's coefficient in `p[0->0]`)**. (Without canonicalization this flips with the label; that is what the user's note "opposite sign of the coefficient in p[N->N]" refers to, case H = 1.)
- **Constrained fit:** statsmodels has no `fix_params` for MS models. `fit_constrained` maximizes `model.loglike(model.transform_params(full(u)))` over the free *untransformed* parameters `u` with `scipy.optimize.minimize` (BFGS, then Nelder-Mead polish if not converged), where `full()` inserts the fixed values. TVTP coefficients are not transformed (identity), so fixing in either space is the same; variances are transformed (untransformed = √σ²), so always go through `untransform_params`/`transform_params`. Restarts: perturb `u` with N(0, 0.5²) noise from `np.random.default_rng(seed + k)`, keep the best non-degenerate LL.
- **First fit procedure (plan §6.4):** unconstrained M3 → canonicalize → constrained refit from those params with Iran's `p[1->0]` coef set to 0. **Warm-started refits [our call]:** start directly from last week's canonical constrained params (labels already canonical); if the result has σ²[0] > σ²[1], rerun the full first-fit procedure (logged as `refit_from_scratch = True`). Saves one unconstrained fit per week.
- **Nesting starts [our call]:** M2 is also started from M1 (logit of M1's probabilities as TVTP constants, slopes 0) and M3 from M2 (β = 0). Keep the best LL. This guarantees LL(M3) ≥ LL(M2) ≥ LL(M1) up to optimizer tolerance, which is checked.
- **Variant fallback `iran_both_logits`** (a frozen choice for a whole run, set at validation; never a per-week switch): plain unconstrained M3; β = −coef in `p[0->0]`; the `p[1->0]` Iran coef is reported too.
- **Several restricted variables (added Nov 23).** Every variable in `spec.iran_vars` is treated like Iran in M3: under `constrained`, each one's `p[1->0]` coefficient is fixed to 0, and each gets its own β = −coef in `p[0->0]`. Nesting starts: M3tone starts from M2 with all β = 0; M3toneL and M3gold start from M3 with the added β = 0. LR df always come from `lr_test`'s free-parameter count.
- **E1 (`kind = "mshar"`).** Same `make_model`/`fit_ms`/`fit_constrained` path, with y = `log_rv_hourly`, HAR exog with `switching_exog=False`, and `canonicalize(by="const")`. The degeneracy rule applies unchanged, with var(y) in place of var(r).
- **Seeding (contract of `fit_ms` and `fit_constrained`):** statsmodels 0.15 `MarkovSwitching.fit(search_reps=…, rng=None)` draws its search perturbations from fresh OS entropy when `rng` is None, and `np.random.seed` does not affect it (two unseeded identical fits differ). So every `.fit(...)` call with `search_reps > 0` passes `rng=seed` explicitly, and every other random draw (restart noise, bootstrap simulation, placebo offsets) uses `np.random.default_rng(seed)`. Same inputs + same seed ⇒ identical params (tested).
- **Degeneracy rule [our call; thresholds frozen at validation]:** a candidate fit is *degenerate* if min σ² < `VAR_FLOOR_FRAC` × var(r) on the estimation sample (σ² → 0 spikes), or either regime's expected duration, evaluated with the standardized TVTP regressors at 0 (training means), is < `MIN_DURATION` weeks. Among search and restart candidates, keep the best-LL non-degenerate one; if all are degenerate, keep the best LL and set `degenerate = True`. The same rule applies to the RQ1/RQ2 fits, bootstrap replicates, placebos and the OOS loop.
- **SEs:** `statsmodels.tools.numdiff.approx_hess3` of the log-likelihood in the *constrained* (natural) parameter space, over free params only; cov = inv(−H).

### 8.5 `design.py` — the one place lags happen

```python
@dataclass
class Design:
    dates: pd.DatetimeIndex     # estimation rows (Fridays), first = SAMPLE_START + 1 week
    y: pd.Series               # r(t)
    X: pd.DataFrame            # mean exog: r_lag = r(t-1) [+ iran(t-1) if mean_exog]
    Z: pd.DataFrame | None     # exog_tvtp: const + standardized z(t-1)
    x_next: pd.Series          # mean exog for target t+1: r(T) [+ iran(T)]
    z_next: pd.Series | None   # tvtp row for transition T -> T+1: const + standardized z(T)
    origin: pd.Timestamp       # T
    std_stats: pd.DataFrame    # mean, sd per tvtp var (from STANDARDIZE_WINDOW)

standardization_stats(panel, vars, window) -> pd.DataFrame   # `dummy_*` columns pass through untouched
freeze_std_stats(panel) -> Path       # writes results/frozen/std_stats.csv (TVTP, iran/tone, ovx, hourly-window and GARCH-X scale stats)
build_design(panel: pd.DataFrame, spec: MSSpec, end: str | pd.Timestamp) -> Design
    # sample start and standardization window come from spec.start / spec.std_window;
    # Iran variables from spec.iran_vars (swap them with ms.with_iran, never here)
har_design(panel: pd.DataFrame, end: str | pd.Timestamp, rv_col: str = "rv",
           x_vars: tuple[str, ...] = (), start: str = C.SAMPLE_START,
           std_window=C.STANDARDIZE_WINDOW) -> tuple[pd.DataFrame, pd.Series, pd.Series]   # (X, y, x_next)
garchx_design(panel: pd.DataFrame, end: str | pd.Timestamp, x_vars: tuple[str, ...],
              rv_col: str = C.X_RV_COL, start: str = C.X_SAMPLE_START,
              scale_window=C.GARCHX_SCALE_WINDOW) -> tuple[pd.Series, pd.DataFrame, pd.Series]   # (r, X, x_next)
```

- Row t of `X` holds r(t−1); row t of `Z` holds [1, z(t−1)]: the regressors that drive S(t−1) → S(t) (plan §6.4).
- The forecast for t+1 uses `x_next` = r(T) and `z_next` = z(T): known at the Friday close of T (plan §5).
- The first panel week (2005-01-07) has no r(t−1), so estimation starts at **2005-01-14** (1,139 rows for the full sample). **[our call: simpler than fetching Dec 2004 weekly returns]** Hourly-sample specs start one week after `spec.start` for the same reason.
- **`kind = "mshar"` (E1):** y(t) = `log_rv_hourly(t)`; row t of `X` = [ln rv_hourly(t−1), ln mean rv_hourly(t−4..t−1), ln mean rv_hourly(t−13..t−1)]; `Z` as above. The first estimation row is the 14th week from `spec.start`.
- **`har_design`:** rows are indexed by the **target** week s. Row s holds X = [1, RV(s−1), mean RV(s−4..s−1), mean RV(s−13..s−1), z_x(s−1) for x in `x_vars`] and y = RV(s), where RV = `panel[rv_col]` and z_x are the logs z-scored with `std_window` stats (OLS forecasts do not depend on this scaling; it only makes coefficients comparable with the MS models). The last row has s ≤ `end`. `x_next` = the same regressors at T = `end`. The first row is the first s with 13 lags of RV from `start` on.
  - `HAR`: `rv_col="rv"`, `x_vars=()`, start `SAMPLE_START`.
  - `HAR_h`: `rv_col=X_RV_COL` (`"rv_hourly"`), `x_vars=()`, start `X_SAMPLE_START` (`HOURLY_SAMPLE_START`).
  - `HARX_C`: as `HAR_h` with `x_vars=X_CONTROLS`.
  - `HARX_CI`: as `HAR_h` with `x_vars=X_CONTROLS + (IRAN_VAR,)`.
- **`garchx_design`:** `r` = weekly returns `r(t)` for t from `start` + 1 week to `end`. Row t of `X` holds **nonnegative** regressors dated t−1:
  - `panel[rv_col](t−1)` raw (`rv_hourly` by default; in %², the same units as ε²);
  - for each name in `x_vars`, the level exp(col(t−1)) divided by its mean over `scale_window` (so `gpr`, `vix` and `iran_gdelt` → GPR, VIX and the Iran share, each ≈ 1 on average in 2007–2012).

  Columns are named `rv_col` (e.g. `rv_hourly`) and the `x_vars`. `x_next` = the same regressors at T.
  - `GARCHX_C`: `x_vars=X_CONTROLS`.
  - `GARCHX_CI`: `x_vars=X_CONTROLS + (IRAN_VAR,)`.
  - Plain `GARCH_t` does not use this design (it uses `arch` on `r` from `SAMPLE_START`, unchanged).
- Robustness swaps (`aigpr_iran`, `iran_gdelt_geo`, a placebo-shifted column) go through `ms.with_iran` for MS specs and through `x_vars` for the HAR-X/GARCH-X designs, never by editing lag code.
- Outside `design.py`, only `weekly.py` may call `.shift()` (to build daily and weekly returns from prices); placebo shifts use `np.roll` in `insample.circular_shift`. A test greps for this. **[our call]**

### 8.6 `forecast.py`

```python
@dataclass
class OneStep:
    pi_H: float          # π_H(T+1 | T)
    var_fc: float        # Σᵢ πᵢ (σᵢ² + mᵢ²)  — compared with RV(T+1)
    mean_fc: float       # Σᵢ πᵢ mᵢ           — compared with r(T+1)
    p_exceed: float      # Σᵢ πᵢ P(|r| > thr | i), mixture-implied
    xi_H_origin: float   # filtered ξ_H(T | T)

one_step(fit: MSFit, design: Design, thr: float) -> OneStep
predicted_probs(model, params) -> np.ndarray     # in-sample π(t | t-1), for the check vs statsmodels
highvol_threshold(panel) -> float                # 90th pct of |r| on TRAIN (lives here so oos.py does not import evaluate.py)
filter_fixed(model_for_longer_sample, params) -> pd.Series  # filtered ξ_H with fixed params (notebook 04/07)
```

- ξ(T|T) = `results.filtered_marginal_probabilities` at T (never smoothed); P(T+1) = `model.regime_transition_matrix(params, exog_tvtp=z_next[None, :])[:, :, 0]` (M1: no exog); π = P · ξ; mᵢ = μᵢ + φᵢ·r(T) (+ θ·Iran(T) if Iran-in-the-mean).
- Columns of P sum to 1, P[i, j] = P(S(T+1)=i | S(T)=j). Verified: P[:, :, t] · ξ(t−1|t−1) reproduces statsmodels' `predicted_marginal_probabilities[t]` to 1e-15.
- **`kind = "mshar"` (E1):** same π; mᵢ = cᵢ + b′x_next; `var_fc` = Σᵢ πᵢ exp(mᵢ + σᵢ²/2); `mean_fc` and `p_exceed` are NaN (not written to the forecasts file).
- One function, imported by notebooks 03–08. No notebook computes a forecast by hand.

### 8.7 `benchmarks.py`

```python
garch_t_forecast(r_hist: pd.Series, start_params: pd.Series | None = None) -> dict
    # {"var_fc": h + mu^2, "mean_fc": mu, "params": pd.Series, "converged": bool}
harx_forecast(panel, end, rv_col: str = "rv", x_vars: tuple[str, ...] = (),
              start: str = C.SAMPLE_START) -> dict      # {"var_fc", "floored": bool, "params": pd.Series}
har_forecast(panel, end) -> dict                         # = harx_forecast(panel, end, "rv")
model_forecast(name: str, panel, end, state: dict) -> dict
    # dispatch for BENCHMARKS_MAIN names; `state` carries warm starts between origins
harx_insample(panel, end, rv_col: str, x_vars: tuple[str, ...], start: str,
              hac_lags: int = 4) -> pd.DataFrame        # coef, NW SE, t, p; + rows "adj_r2", "n"
rw_forecast() -> float                                   # 0.0 (return)

@dataclass
class GarchXFit:
    params: pd.Series     # mu, omega, alpha, beta, gamma_<col> for each X column, nu
    loglike: float; converged: bool; seconds: float
    h: pd.Series          # fitted conditional variances h(t), same index as r
    x_names: tuple[str, ...]

garchx_loglike(theta: np.ndarray, r: np.ndarray, X: np.ndarray, h1: float) -> float  # Student-t; X may have 0 columns
garchx_fit(r: pd.Series, X: pd.DataFrame, start_params: pd.Series | None = None) -> GarchXFit
garchx_forecast(fit: GarchXFit, r: pd.Series, x_next: pd.Series) -> dict    # {"var_fc": h(T+1) + mu^2, "h_next"}
garchx_simulate(params: pd.Series, X: pd.DataFrame, seed: int) -> pd.Series  # returns, for the recovery test
```

- GARCH(1,1)-t: `arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")`, warm-started from last week's params. **[our call]** `var_fc = h(T+1) + μ̂²` so it targets the same uncentered second moment as the MS forecast (RV is uncentered; plan §6.4).
- HAR: RV(t+1) = b₀ + b₁·RV(t) + b₄·mean(RV(t−3..t)) + b₁₃·mean(RV(t−12..t)) + e; OLS in **levels** **[our call: standard Corsi form; log-HAR needs a bias correction]**; floor = 1% of the minimum training RV, floors counted in the fitlog.
- **HAR-X ladder (added Nov 23; extra models, status open) [decided].** `HAR_h` → `HARX_C` → `HARX_CI` (design in §8.5): same form, OLS in levels and floor as HAR. The floor is `RV_FLOOR_FRAC` × the minimum of `rv_col` over `TRAIN` ∩ the estimation sample (for `rv_hourly`: 2006-11-24 → 2012-12-28). The lagged RV is already in the HAR terms; `gpr(t)`, `vix(t)` and `iran_gdelt(t)` are the same z-scored logs the MS models use, dated t to forecast t+1. The existing `HAR` (daily `rv`, from 2005) stays as the plan's benchmark.
- **GARCH-X (added Nov 23; extra models, status open) [our call: specification; reasoning in §4.4].**
  - Model on weekly returns with Student-t errors:
    - r(t) = μ + ε(t), ε(t) = √h(t)·z(t), with z standardized Student-t(ν), as in `arch`'s `dist="t"`;
    - h(t) = ω + α·ε²(t−1) + β·h(t−1) + Σₖ γₖ·xₖ(t−1).
  - Regressors xₖ ≥ 0 (from `garchx_design`): `rv_hourly(t−1)` and the scaled **levels** of GPR, VIX (and the Iran share for `GARCHX_CI`).
  - Restrictions: ω > 0, α ≥ 0, β ≥ 0, α + β ≤ 0.999, γₖ ≥ 0, 2.05 ≤ ν ≤ 100.
  - h(1) = sample variance of r over the estimation sample **[our call]**.
  - Optimizer: `scipy.optimize.minimize(method="SLSQP")` with these bounds and the one linear constraint.
  - Starts: last week's params (warm start); at the first origin, the plain GARCH-t solution with all γ = 0 (nesting start, so LL(GARCH-X) ≥ LL(GARCH-t on the same sample)). Best LL kept.
  - Forecast `var_fc` = h(T+1) + μ̂², with h(T+1) = ω + α·ε²(T) + β·h(T) + Σ γₖ·xₖ(T), like `GARCH_t`.
  - Nonnegative regressors with γ ≥ 0 keep h > 0 with no extra checks; that is why levels (scaled to mean ≈ 1 on 2007–2012) are used instead of the z-scored logs.
- **GARCH-X inference.** γ = 0 is on the boundary of the parameter space, so the LR for γ = 0 is not χ²:
  - one γ (GARCHX_CI vs GARCHX_C): 50:50 mixture of χ²₀ and χ²₁, p = 0.5·P(χ²₁ > LR);
  - k γ's jointly (GARCHX_C vs GARCH-t, k = 3): chi-bar-squared, p = Σⱼ C(k, j)/2ᵏ · P(χ²ⱼ > LR), with P(χ²₀ > LR) = 0 for LR > 0. This treats the components as independent, an approximation that is stated.

  SEs come from the numerical Hessian (`approx_hess3`); they are unreliable when γ̂ sits at 0, which the table says.

### 8.8 `oos.py`

```python
origins(start_target: str, end_target: str) -> list[pd.Timestamp]   # origin = target - 7 days
run_oos(panel, run_id: str, ms_specs: list[MSSpec],
        benchmarks: tuple[str, ...] = C.BENCHMARKS_MAIN,
        origins: list[pd.Timestamp] = ..., iran_var: str = C.IRAN_VAR,
        force: bool = False, max_origins: int | None = None) -> None
    # benchmarks: any subset of BENCHMARKS_MAIN (() for MS-only robustness runs);
    # iran_var: the Iran column used by HARX_CI and GARCHX_CI in this run (MS specs carry their own iran_vars)
assemble(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]  # forecasts, params, fitlog → results/
load_forecasts(run_id: str, with_actuals: bool = True) -> pd.DataFrame
    # with_actuals adds actual_rv, actual_rv_hourly, actual_rv_gk, actual_r, actual_exceed (joined on date)
fake_forecasts(panel, models: tuple[str, ...] = ("M1", "M2", "M3") + C.BENCHMARKS_MAIN) -> pd.DataFrame
    # schema-correct dummy (§9.2) for building notebook 06 early
```

- **Expanding window**, refit every week, origins 2012-12-28 → 2026-10-30. Every model uses all data from its own sample start (§9.6) through the origin.
- **Extra models in the loop (added Nov 23):** per origin, after the MS models, `benchmarks.model_forecast(name, panel, origin, state)` for each name in `benchmarks`. GARCH-X warm starts live in `state` (like GARCH_t); HAR-family fits are closed-form. These fits are deterministic (no seed needed). Their `params` rows use readable names (`b0, b_rv1, b_rv4, b_rv13, c_gpr, c_vix, c_iran_gdelt` for the HAR family; `mu, omega, alpha, beta, gamma_rv_hourly, gamma_gpr, gamma_vix, gamma_iran_gdelt, nu` for GARCH-X). Their `fitlog` rows fill `loglike, converged, seconds, har_floored` and leave the MS columns empty.
- **Warm starts and restarts:** each MS model starts from its previous canonical params. Random restarts (`N_RESTARTS`) run every `RESTART_EVERY` weeks and whenever the warm-start fit fails a sanity check: not converged; the new LL is below `model.loglike(prev_params)` evaluated on the **current** sample (the optimizer ended below its own start; last week's LL is not comparable because the sample grew); σ²_H/σ²_N < `SIGMA2_RATIO_MIN` (frozen at validation); degenerate; or nesting violated. **[our call: restarts every week would roughly triple runtime]**
- **M3 failure inside a frozen run:** the variant never changes mid-run. If no candidate converges, use the best-LL candidate, set `converged = False`, and log it; notebook 06 reports the number of such weeks.
- **Seeds:** for each origin, `seed = C.SEED + origin_index`, where the index is the origin's position in the full origin list (2012-12-28 = 0), so a week gets the same seed in every run. The seed goes to `fit_ms` (→ statsmodels `rng`) and to restart noise; a resumed run gives identical numbers.
- **Checkpoint/resume:** after each origin, append that origin's rows to `results/checkpoints/<run_id>_{forecasts,params,fitlog}.csv` (flush). Every row stores `settings_hash` (hash of the frozen settings: search reps, restarts, restart schedule, thresholds, M3 variant, φ, `iran_var`, model list incl. `benchmarks`, `INTRADAY_ROLL_RULE`). On start, resume from the last completed origin **only if** every row's hash equals the current one (for the `full` stage, also the one in `spec_freeze.json`); otherwise discard the checkpoint and start again from the run's first origin. `force=True` deletes the run's checkpoints first. `assemble` writes the final files to `results/forecasts/` and `results/fits/`.
- **Skip logic:** notebook cell calls `run_oos` only if the assembled output is missing or `FORCE_OOS` is True.

### 8.9 `insample.py`

```python
lr_test(fit_restricted: MSFit, fit_full: MSFit) -> dict                      # df = difference in free-parameter counts
bootstrap_lr_1v2(y, X, n_reps: int, seed: int, search_reps: int) -> dict    # RQ1
equality_tests(design, spec) -> pd.DataFrame                                  # μ, φ: LR + Wald (RQ1)
fit_ladder(panel, end, phi_switching, iran_var=C.IRAN_VAR) -> dict[str, MSFit]  # M1, M2, M3 with nesting starts
fit_tone(panel, end, phi_switching, ladder: dict[str, MSFit]) -> dict[str, MSFit]  # M3tone (from M2), M3toneL, M3gold (from M3)
wald_equal(fit: MSFit, a: str, b: str) -> dict                 # H0: beta_a = beta_b (deck convention), chi2(1), from param_cov
chibar_pvalue(lr: float, k: int) -> float                      # sum_j C(k,j)/2^k * P(chi2_j > lr); k = 1 gives 0.5*P(chi2_1 > lr)
garchx_lr(panel, end, x_small: tuple[str, ...] | None, x_big: tuple[str, ...]) -> dict
    # x_small None = GARCH-t via garchx_fit with no X; same sample; {lr, k, pvalue_chibar, gammas, ses}
circular_shift(series: pd.Series, k: int, window: tuple[str, str]) -> pd.Series  # wraps inside window only
placebo_lr(panel, n_shifts, seed, window=("2005-01-07", "2025-12-26")) -> pd.DataFrame   # columns k, ll_m3, lr_raw
placebo_harx(panel, n_shifts, seed, window=(C.HOURLY_SAMPLE_START, "2025-12-26"),
             oos_targets=("2019-01-04", "2025-12-26")) -> pd.DataFrame   # columns k, lr_insample, qlike_gain_oos
```

`lr_test` returns a dict with keys `lr_raw, stat, df, pvalue, negative, k_restricted, k_full` (`stat = max(lr_raw, 0)`; `negative` flags an optimizer shortfall). Tables report `stat` and `lr_raw`; placebos compare `lr_raw` (actual vs. shifted), so a negative raw LR is never hidden.

```python
```

### 8.10 `evaluate.py`

```python
qlike(rv, f) -> np.ndarray; sq_err(y, f) -> np.ndarray
loss_table(fc: pd.DataFrame, measure: str = C.MAIN_RV, split_col="split") -> pd.DataFrame
    # fc = load_forecasts(...) rows with target "var"; actual = fc[f"actual_{measure}"]; output has a `measure` column
best_linear_model(fc: pd.DataFrame, measure: str = C.MAIN_RV, window=C.VALID,
                  pool=C.LINEAR_MODELS) -> str              # lowest mean QLIKE on the validation window
dm_test(y, f_a, f_b, loss: str = "qlike") -> dict           # wraps statsmodels diebold_mariano_test(lags=0, harvey_adj=True, horizon=1)
clark_west(y, f_small, f_big) -> dict                        # adjusted MSPE, one-sided, HAC t
giacomini_white(loss_a, loss_b, instruments: pd.DataFrame | None) -> dict   # Wald on mean(h_t * dL), chi2(q)
mincer_zarnowitz(y, f, hac_lags=4) -> dict                   # a, b, SEs, joint Wald (a, b) = (0, 1)
pesaran_timmermann(y, f) -> dict
sign_hit_rate(y, f) -> float
brier(p, event) -> float; reliability_table(p, event, n_bins=10) -> pd.DataFrame
ljung_box(errors, lags=(4, 8, 13)) -> pd.DataFrame           # statsmodels acorr_ljungbox
cumulative_loss_diff(loss_a, loss_b) -> pd.Series
highvol_threshold(panel) -> float                            # re-exported from forecast.py (same function)
gw_instrument(panel, origins) -> pd.Series                   # 1{iran(origin) > 75th pct of VALID}; index = origins
```

Each test returns a dict with `stat`, `pvalue`, `n` and any coefficients, so tables are built by stacking dicts. The tests take plain arrays; notebook 06 picks the actual column per measure, so no test function knows about measures.

### 8.11 `plots.py`

`setup_style()`, `shade_events(ax, events, dates)`, `mark_split(ax)` (Jan 1, 2026 line), `save_fig(fig, name)` (PNG, 150 dpi, `results/figures/`), `event_window_plot(...)`. One consistent palette: N/H, M1/M2/M3 (M3tone a lighter M3), benchmarks (GARCH_t, HAR, HAR_h), extra models (HARX_*, GARCHX_* as dashed versions of their base model's color).

### 8.12 `weekly.py` (listed here since Nov 23; built in Plan 1)

```python
week_ending(d)                                   # d + ((4 - weekday) mod 7) days; Timestamp, DatetimeIndex or Series
fridays(start=C.SAMPLE_START, end=C.FREEZE_DATE) -> pd.DatetimeIndex
weekly_brent(brent: pd.DataFrame, weeks) -> pd.DataFrame          # price, r, rv, n_days
weekly_last(daily, col, weeks) -> pd.Series; weekly_mean(daily, col, weeks) -> pd.Series
gdelt_weekly(gd: pd.DataFrame, weeks, switch=C.GDELT_SWITCH_DATE) -> tuple[pd.DataFrame, int]
    # iran_share, iran_gdelt, iran_gdelt_geo, gdelt_days, iran_conflict, iran_coop, iran_tone, iran_goldstein; int = rows dropped
aigpr_weekly(monthly, weeks, lag_months=C.AIGPR_PUBLICATION_LAG_MONTHS) -> pd.Series
intraday_weekly(daily: pd.DataFrame, weeks, first_week: str = C.HOURLY_SAMPLE_START) -> pd.DataFrame
    # rv_hourly, log_rv_hourly, n_hours_week, rv_gk; NaN before first_week; raises if a week >= first_week has n_hours_week == 0
build_weekly_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame   # keys brent, vix, ovx, gpr, aigpr, gdelt, intraday
```

- Shares are Σ numerator / Σ denominator over the week's days, never a mean of daily shares; the tone and Goldstein means use their own `_n` denominators (§9.1).
- `rv_gk` per day d (all prices from `brent_intraday_daily.csv`, C_{d−1} = the previous row's close):

  100²·[ (ln O_d − ln C_{d−1})² + 0.5·(ln H_d − ln L_d)² − (2 ln 2 − 1)·(ln C_d − ln O_d)² ]

  The first term is the gap since the previous close, so the weekly sum spans Friday close to Friday close like `rv_hourly` **[our call: Garman-Klass plus the opening gap]**. Days are not roll-corrected (appendix measure only).

---

## 9. File formats and interfaces

### 9.1 `data/processed/weekly_panel.csv`

One row per Friday, 2005-01-07 → 2026-11-06 (1,140 rows). All values **unlagged**: each row holds what is known at that Friday's close.

| Column | Definition |
|---|---|
| `date` | Friday (ISO). Week = Saturday → Friday. |
| `price` | Brent close on the last trading day of the week (Friday, or earlier if Friday is a holiday) |
| `r` | 100·ln(price(t)/price(t−1)); first row uses the Dec 31, 2004 close |
| `rv` | Σ (100·daily log return)² over the week's trading days; the first daily return uses the previous week's last close |
| `log_rv` | ln(rv) |
| `n_days` | Brent trading days in the week **[our call: diagnostic]** |
| `iran_share` | Σ Iran-actor articles / Σ all articles over the week's days (level) **[our call: for plots]** |
| `iran_gdelt` | ln(iran_share) — main measure |
| `iran_gdelt_geo` | ln of the actor + location share — robustness **[our call: extra column]** |
| `gdelt_days` | days with GDELT data in the week **[our call: diagnostic]** |
| `gpr` | ln(mean of daily GPRD over the 7 days Sat–Fri) |
| `vix` | ln(VIX close, last trading day of the week) |
| `ovx` | ln(OVX close, last trading day); NaN before May 2007 |
| `aigpr_iran` | ln(AI-GPR Iran of the last month published by this Friday); see below |
| `iran_conflict` | ln(Σ `articles_iran_conflict` / Σ `articles_all`) over the week's days (QuadClass 3–4) **[decided, added Nov 23]** |
| `iran_coop` | ln(Σ `articles_iran_coop` / Σ `articles_all`) (QuadClass 1–2) **[decided]** |
| `iran_tone` | Σ `tone_x_articles_iran` / Σ `articles_iran_tone_n`: article-weighted mean `AvgTone` of Iran events, **level** (can be negative) **[decided]** |
| `iran_goldstein` | Σ `goldstein_x_articles_iran` / Σ `articles_iran_goldstein_n`, level **[decided]** |
| `rv_hourly` | Σ `rv_intraday` over the week's Dukascopy days (Fri 22:00 UTC → Fri 22:00 UTC, gap returns included, roll returns removed; §6.8). NaN before `HOURLY_SAMPLE_START` **[decided: main evaluation measure]** |
| `log_rv_hourly` | ln(`rv_hourly`) |
| `n_hours_week` | Σ `n_hours`: hourly returns in the week; weeks < `MIN_HOURS_WEEK` (60) are flagged in notebook 02, not dropped **[our call: normal week ≈ 110–115 bars; because gap returns are kept, a short week still spans close to close]** |
| `rv_gk` | Σ daily Garman-Klass + opening-gap variance (§8.12), NaN before `HOURLY_SAMPLE_START` **[decided: appendix measure]** |
| `gdelt_source` | `v1` or `v2`; only present if the GDELT 2.0 splice was needed (§6.4) |

- **Column order:** `date, price, r, rv, log_rv, n_days, iran_share, iran_gdelt, iran_gdelt_geo, gdelt_days, gpr, vix, ovx, aigpr_iran, iran_conflict, iran_coop, iran_tone, iran_goldstein, rv_hourly, log_rv_hourly, n_hours_week, rv_gk` (+ `gdelt_source` last if present). Units: `rv`, `rv_hourly`, `rv_gk` all in %² (returns × 100).
- Variable names follow plan §4.4; `iran_gdelt`, `iran_conflict`, `iran_coop`, `gpr`, `vix`, `ovx`, `aigpr_iran` hold **logs** (as used in the models); `iran_tone` and `iran_goldstein` are levels. Level series can be recovered with `exp`.
- **Zero values:** a week with 0 Iran articles (main or geo variant, conflict or cooperation), 0 toned or Goldstein-scored Iran articles, or an AI-GPR month ≤ 0 makes `build_weekly_panel` raise with the list; nothing is filled or offset. A week on or after `HOURLY_SAMPLE_START` with no hourly return raises too (its gap return would otherwise land in the next week).
- **Week mapping:** `week_ending(d) = d + ((4 − d.weekday()) mod 7) days` (Friday = 4). Saturday and Sunday map to the next Friday, so the weekend rule (plan §5) is automatic.
- **GDELT:** daily rows with `date < 2013-04-01` come from the `backfile` feed (dated by `SQLDATE`), rows on or after from the `daily` feed (dated by `DATEADDED`); then mapped to weeks and summed (numerator and denominator separately) before taking the share.
- **AI-GPR availability [our call, conservative]:** month m is treated as published from the first Friday of month m + 2 (one full month of publication lag, `AIGPR_PUBLICATION_LAG_MONTHS = 1`). This availability rule is applied in the panel (it defines what is known at t); the model's one-week lag is then applied in `design.py` like every other variable. No other lag is applied in the panel.

### 9.2 `results/forecasts/forecasts_<run>.csv` (long format)

| Column | Meaning |
|---|---|
| `date` | target week t+1 (Friday) |
| `origin` | forecast origin t = date − 7 days |
| `split` | `validation` / `test` / `holdout`, by target date |
| `run` | `main`, `ar_alt`, `aigpr`, `gdelt_geo`, `ovx`, `hourly_ctrl`, `tone`, `mshar`, `placebo_<k>` (`timing` is a throwaway run, never evaluated) |
| `model` | `M1`, `M2`, `M3`, `GARCH_t`, `HAR`, `RW`; extra models `HAR_h`, `HARX_C`, `HARX_CI`, `GARCHX_C`, `GARCHX_CI`; robustness runs only (model column; run in brackets): `M1_alt, M2_alt, M3_alt` (`ar_alt`), `M3_aigpr` (`aigpr`), `M3_geo` (`gdelt_geo`), `M2_ovx, M3_ovx` (`ovx`), `M2_hctrl, M3_hctrl` (`hourly_ctrl`), `M3tone` (`tone`), `MSHAR_C, MSHAR_CI` (`mshar`), `M3` (`placebo_<k>`, shifted Iran series). `M3toneL` and `M3gold` are in-sample only (no OOS run) |
| `target` | `var` (variance forecast for week t+1), `r` (return forecast), `p_high` (π_H), `p_exceed` (mixture P(\|r\| > thr)) |
| `forecast` | value |

- **Changed Nov 23 [our call]:** the variance target is called `var`, not `rv`. Each model makes **one** variance forecast; which RV measure it is scored against (`rv_hourly`, `rv`, `rv_gk`) is chosen at evaluation via the `actual_<measure>` columns. Writing the same number three times under three target names would only invite mismatches. The RV measures are named exactly as their panel columns: `rv` (daily returns, FRED spot), `rv_hourly` (hourly Dukascopy), `rv_gk` (Garman-Klass).
- Plan §7's minimal format `date, model, target, forecast` is a subset; extra columns are additions. **[our call]**
- Actuals are **not stored**; `load_forecasts(with_actuals=True)` joins from the panel, on `date`: `actual_rv`, `actual_rv_hourly`, `actual_rv_gk`, `actual_r`, `actual_exceed` (1{|r| > training 90th pct}). So the panel stays the single source of truth. **[our call]**
- Which models produce which targets: MS-AR models (M1–M3, M3tone, robustness specs) → all four; `MSHAR_*` → `var`, `p_high`; `GARCH_t`, `GARCHX_*`, `HAR`, `HAR_h`, `HARX_*` → `var`; `RW` → `r`.
- Size: ≈ 723 × 20 rows ≈ 14.5k rows ≈ 1.1 MB for `main`.

### 9.3 Standardization **[our call]**

TVTP regressors are z-scored with the mean and SD of **2005–2012** (the training window) in every fit, in-sample and OOS (OVX run: 2007-05-11 → 2012-12-28, `OVX_STD_WINDOW`). Why: no lookahead in any window, coefficients comparable across origins and specifications, and LR tests are invariant to it. Stats are stored in `Design.std_stats` and written once to `results/frozen/std_stats.csv` by `design.freeze_std_stats(panel) -> Path` (columns `var, window, window_start, window_end, mean, sd`). Columns named `dummy_*` (e.g. `dummy_post2013`) pass through standardization untouched (they are 0/1 and never z-scored).

**Hourly-sample specs (added Nov 23):** specs whose sample starts at `HOURLY_SAMPLE_START` (robustness #7 with `log_rv_hourly`, E1) use `STANDARDIZE_WINDOW_HOURLY` = 2007-01-05 → 2012-12-28 (whole calendar years inside both the hourly sample and TRAIN), like the OVX run. The HAR-X regressors use the `TRAIN` stats (forecasts are invariant to the scaling). GARCH-X level regressors are divided by their mean over `GARCHX_SCALE_WINDOW` (= `STANDARDIZE_WINDOW_HOURLY`). These stats are added to `std_stats.csv` with a `window` column.

### 9.4 `results/fits/params_<run>.csv` and `fitlog_<run>.csv`

- **params (wide):** `origin, model, <param_name_1>, …` in canonical labels, with readable names added by `make_model` (e.g. `p00_iran`, `p10_vix`, `mu_N`, `mu_H`, `phi_N`, `phi_H`, `sigma2_N`, `sigma2_H`). ≈ 2,200 rows.
- **fitlog:** `origin, model, settings_hash, loglike, converged, n_restarts_used, relabeled, refit_from_scratch, degenerate, nesting_ok, seconds, sigma2_N, sigma2_H, dur_N, dur_H, xi_H_origin, har_floored`.

### 9.5 Frozen-decision files

- `results/frozen/rq1_decision.json` — φ decision and evidence (notebook 03).
- `results/frozen/spec_freeze.json` — every setting frozen after validation + git hash (notebook 05). Since Nov 23 it also records `INTRADAY_ROLL_RULE`, `INTRADAY_REDISTRIBUTE`, `HOURLY_SAMPLE_START` and `BENCHMARKS_MAIN`, so a changed roll rule cannot slip into a resumed run.
- Notebooks 04–08 assert `config` matches these files.

### 9.6 Model registry (added Nov 23)

One table for plan writers: what each model is fitted on, from when, and its status.

| Model | Family / status | Fitted on | Sample start (first estimation row) | Regressors (dated t, for t+1) | Targets written |
|---|---|---|---|---|---|
| `M1`, `M2`, `M3` | main | weekly `r` (MS-AR) | `SAMPLE_START` (row 2005-01-14) | TVTP: `TVTP_CONTROLS` (daily `log_rv`), + `iran_gdelt` (M3) | var, r, p_high, p_exceed |
| `M3tone` | main variant, in sample (04); OOS optional (08) | `r` | `SAMPLE_START` | controls + `iran_conflict`, `iran_coop` (N→H) | as M3 |
| `GARCH_t` | benchmark (plan) | `r`, `arch` | `SAMPLE_START` | — | var |
| `HAR` | benchmark (plan) | `rv` | `SAMPLE_START` (+13 weeks of lags) | HAR lags of `rv` | var |
| `RW` | benchmark (plan) | — | — | — | r |
| `HAR_h` | **extra, status open** | `rv_hourly` | `HOURLY_SAMPLE_START` (+13 weeks) | HAR lags of `rv_hourly` | var |
| `HARX_C` | **extra, status open** | `rv_hourly` | same | + `gpr`, `vix` (z-scored logs) | var |
| `HARX_CI` | **extra, status open** | `rv_hourly` | same | + `iran_gdelt` | var |
| `GARCHX_C` | **extra, status open** | `r`, custom MLE | `HOURLY_SAMPLE_START` (+1 week) | `rv_hourly(t)`, scaled levels of GPR, VIX | var |
| `GARCHX_CI` | **extra, status open** | `r`, custom MLE | same | + scaled Iran share | var |
| `M1_alt, M2_alt, M3_alt` | robustness #3 (OOS run `ar_alt`) | `r` | `SAMPLE_START` | as M1–M3 with the φ setting not chosen in RQ1 | var, r, p_high, p_exceed |
| `M3_aigpr`, `M3_geo` | robustness #2, #2b (runs `aigpr`, `gdelt_geo`) | `r` | `SAMPLE_START` | M3 with `aigpr_iran` / `iran_gdelt_geo` in N→H | as M3 |
| `M2_ovx`, `M3_ovx` | robustness #6 (run `ovx`) | `r` | `OVX_START` | `ovx` replaces `vix`; std on `OVX_STD_WINDOW` | as M3 |
| `M2_hctrl`, `M3_hctrl` | robustness #7 (run `hourly_ctrl`) | `r` | `HOURLY_SAMPLE_START` | `log_rv_hourly` replaces `log_rv`; std on `STANDARDIZE_WINDOW_HOURLY` | as M3 |
| `M3toneL`, `M3gold` | robustness #8, in sample only | `r` | `SAMPLE_START` | M3 + `iran_tone` / `iran_goldstein` (N→H) | — |
| `MSHAR_C`, `MSHAR_CI` | optional extension E1 (run `mshar`) | `log_rv_hourly` (MS-HAR) | `HOURLY_SAMPLE_START` (+13 weeks) | HAR log lags; TVTP `gpr`, `vix` (+ `iran_gdelt` N→H) | var, p_high |

"Status open" means the report will call these models benchmarks, replacements or appendix material only after the results are in **[decided]**; no code or table name presumes it (tables use `family = "extra"`).

---

## 10. Testing and verification

Small `pytest` suite for the risky parts only. Run `uv run pytest` (target < 1 min). Synthetic data, no network.

| Test file | What it checks |
|---|---|
| `test_sources.py`, `test_sources_intraday.py` (Dukascopy part, added Nov 23) | On synthetic hourly bars, no network:<br>• **Weekend gap:** bars Fri 2025-01-10 20:00, 21:00 UTC and Sun 2025-01-12 23:00, Mon 2025-01-13 00:00 UTC → the Sunday-evening bar and its gap return (vs. Friday 21:00's close) land on Monday 2025-01-13; there is no Sunday row; the sum of retained returns equals 100·ln(last close / first close).<br>• **UTC/DST:** a bar at 22:00 Europe/London in July (21:00 UTC) stays on its day, a bar at 23:00 London in July (22:00 UTC) moves to the next day, and a bar at 22:00 London in January (22:00 UTC) moves to the next day; naive timestamps are treated as UTC.<br>• **Roll jump:** one +5% bar in a series of ±0.1% returns is the top `roll_candidates` row; the `threshold` and `timestamps` rules remove exactly that return; `n_hours` drops by 1 and the removed log has one row; `{"kind": "none"}` removes nothing; rule `None` raises.<br>• **Data hygiene:** zero-volume bars are dropped before returns; bars at or after 2026-11-06 22:00 UTC are dropped.<br>Also: `render_q1` contains the sentiment aggregates. |
| `test_weekly.py` | `week_ending`: Mon–Fri → same Friday, Sat/Sun → next Friday; Sat Sep 14, 2019 → Sep 20; Sat Feb 28, 2026 → Mar 6. Holiday Friday (e.g. Good Friday Apr 18, 2025): price = Thursday close, `rv` sums 4 returns, the next week's first return uses that Thursday. GDELT: backfile/daily split at Apr 1, 2013; weekly share = Σnum/Σden (not a mean of daily shares); a week with 0 Iran articles raises; `iran_conflict`, `iran_coop`, `iran_tone` on a hand-computed two-day week; a week with 0 conflict articles raises. AI-GPR: month m first appears in month m + 2. Intraday: a Saturday row (from a Friday 22:00 UTC bar) goes to the next week; `rv_hourly` = Σ daily values; NaN before `HOURLY_SAMPLE_START`; a zero-hour week after it raises; `rv_gk` on a hand-computed day. |
| `test_design.py` | Row t of `X` = r(t−1); row t of `Z` = standardized z(t−1); `x_next`/`z_next` use T; standardization uses only the training window (changing post-2012 data does not change the stats); no `.shift(` outside `design.py` and `weekly.py`. `har_design`: row s holds RV(s−1) and the 4- and 13-week means ending at s−1, plus z_x(s−1); changing any value dated after `end` changes nothing. `garchx_design`: all X ≥ 0; row t holds x(t−1); the scaled levels have mean 1 on `GARCHX_SCALE_WINDOW`. |
| `test_ms.py` | `canonicalize`: LL unchanged, σ²[1] > σ²[0], idempotent; for M1 and TVTP models; with `by="const"`, the H regime has the higher intercept. Constrained fit: fixed coef is exactly 0, LL(constrained) ≤ LL(unconstrained) + tol, LL(M3) ≥ LL(M2) − tol. β sign: simulate data where Iran raises P(N→H) and check `transition_coefs_deck` returns β > 0. Seeding: two `fit_ms` calls with the same seed and `search_reps > 0` return identical params. Degeneracy: a candidate with σ² below the floor is rejected when a non-degenerate one exists. LR df: 1 for constrained M3 vs M2, 2 for `iran_both_logits`; 2 for constrained M3tone vs M2, 4 for `iran_both_logits`. |
| `test_forecast.py` | `predicted_probs` = statsmodels `predicted_marginal_probabilities` (< 1e-10) for M1 and a TVTP model; `one_step` var_fc = Σπ(σ²+m²) on a hand-computed case; π sums to 1; for `kind="mshar"`, var_fc = Σπ exp(m + σ²/2) on a hand-computed case. |
| `test_benchmarks.py` (added Nov 23) | `harx_forecast` with `x_vars=()` equals `har_forecast` on the same `rv_col`; the HAR-X OLS forecast equals `numpy.linalg.lstsq` on the same design; the floor applies and is flagged. **GARCH-X recovery:** simulate n = 4,000 weekly returns with `garchx_simulate` from (μ, ω, α, β, γ, ν) = (0.1, 0.3, 0.07, 0.80, 0.5, 8), seed 405, with one regressor x = exp(AR(1), φ = 0.9, sd 0.3); `garchx_fit` returns α within ±0.04, β within ±0.08, γ within ±0.25, ν in [5, 15], and LL(fit) ≥ LL(true params) − 1e-6. With an extra regressor whose true γ = 0, γ̂ ≥ 0 and its LR p (`chibar_pvalue`, k = 1) > 0.01. `garchx_loglike` with zero X columns is finite and equals the Student-t GARCH(1,1) likelihood computed by hand on 5 observations. `garchx_forecast` = ω + α ε²(T) + β h(T) + γ′x(T) + μ². |
| `test_insample.py` (added Nov 23) | `chibar_pvalue(lr, 1)` = 0.5·P(χ²₁ > lr) (e.g. lr = 2.706 → 0.05); `chibar_pvalue(0, k)` = 1; for k = 3 the weights are 1/8, 3/8, 3/8, 1/8. `wald_equal` on a fit with identical coefficients gives stat 0. `circular_shift` never moves values across the window bounds. |
| `test_evaluate.py` | QLIKE = 0 when f = RV and > 0 otherwise; DM wrapper (`lags=0`, HLN) matches a 10-line hand implementation with the same settings; CW on a toy nested case gives a positive stat when the big model is right; GW with instrument = constant equals the unconditional version; PT on perfectly signed forecasts rejects; MZ recovers (a, b) = (0, 1) on f = y + noise. `loss_table` uses `actual_<measure>`; `best_linear_model` picks the lower-loss model on the validation window and ignores test-window losses. |
| `test_oos.py` | On 3 toy origins: an interrupted-then-resumed run equals an uninterrupted run; a checkpoint with a different `settings_hash` is discarded, not resumed; every model in `benchmarks` writes exactly one `var` row per origin. |
| `test_budget.py` | `check_budget()` passes. |

Verification beyond tests (in notebooks):
- Notebook 01 verifies the Dukascopy API, first date, terms and roll candidates on the real feed; notebook 02 checks the hourly vs. daily RV level (§7.2 step 7).
- Notebook 03 repeats the forecaster check on real data.
- Notebook 05 checks forecast alignment, nesting of LLs per week (incl. GARCHX_CI ≥ GARCHX_C ≥ GARCH-t-on-the-same-sample at the first origin) and the resume identity.
- Notebook 08's placebo uses notebook 04's code path; the unshifted LR must equal notebook 04's.
- **Reproducibility check** (plan §12, Fri Dec 4): fresh clone → `uv sync` → `uv run pytest` → (only if `INTRADAY_REDISTRIBUTE` is False: `uv sync --group intraday` and notebook 01's Dukascopy cell) → restart-and-run-all notebooks 02 → 08 with `FORCE` flags off (05 and 08 load committed outputs) → `git diff` shows no changes to committed tables. Optionally rerun 05 in full overnight and confirm identical forecasts (seeds make this exact).
- Before trusting AI-written module code, the builder reads it (the `verify-ai-code` habit).

---

## 11. Build order (one builder, in sequence)

**[decided]** One builder, notebooks in order. **[our call]** Module code is written just before the notebook that first needs it, with its tests. Critical path: **01 → 02 → 03 (freeze φ) → 05 timing → 05 validation → freeze → 05 full → 06 → 07**. Notebook 04 and the build of 06 fit into time the OOS loop runs unattended.

**Rewritten Nov 23 [our call].** Nothing was built by Mon Nov 23, about 3 days behind the original dates, and the Nov 23 additions add roughly a day of work (mostly in notebooks 01, 02 and `benchmarks.py`). Thanksgiving is a normal full working day **[decided]**. The three implementation plans map onto the dates as follows.
- **Plan 1** (scaffold, data, notebooks 01–02): Nov 23–25.
- **Plan 2** (`design`, `ms`, `forecast`, `insample`, `benchmarks` incl. HAR-X/GARCH-X, `oos`; notebooks 03–05 incl. the overnight OOS run): Nov 25–29.
- **Plan 3** (`evaluate`, `plots`; notebooks 06–08 incl. E1): Nov 30–Dec 3.

| Dates | Work (critical path in bold) | Runs unattended |
|---|---|---|
| **Mon Nov 23** | Plan 1: scaffold (`uv init`, deps incl. `gdelt` and `intraday` groups, `config.py`, `paths.py`). **BigQuery setup and D1 first** (riskiest: account access, backfile `DATEADDED`, coverage). **Dukascopy access check**: one week of hourly bars via `dukascopy-python`, else `npx`; read the terms page. `sources.py` HTTP and GDELT parts. | D1 (minutes) |
| **Tue Nov 24** | `sources.py` Dukascopy part + tests; **notebook 01**: FRED, GPR, AI-GPR, Q1 with sentiment columns, full Dukascopy fetch, **roll rule decided with the user**. | Q1 (minutes); Dukascopy fetch (minutes to ≈ 1 h) |
| **Wed Nov 25** | `weekly.py` + tests; **notebook 02** (panel with `rv_hourly` and tone columns, hourly vs. daily check). **Plan 1 done.** Plan 2 starts: `design.py`, `ms.py` + tests. | — |
| **Thu Nov 26** | `forecast.py`, `insample.py` + tests; **notebook 03 → freeze φ**. | Bootstrap LR ≈ 20 min |
| **Fri Nov 27** | `benchmarks.py` (incl. HAR-X, GARCH-X and the recovery test), `oos.py` + `test_oos.py`; **notebook 05 `timing`** → set restarts; **start `validation` in the evening**. | Validation stage ≈ 0.5–1 h |
| **Sat Nov 28** | **Review validation → freeze spec** (`spec_freeze.json`); **start `full`** and leave it overnight. Meanwhile notebook 04 (RQ2, M3-tone, extra models in sample). | Full run ≈ 1–2 h (≈ 2–3 h if validation changed a setting and it reruns from 2012-12-28) |
| **Sun Nov 29** | Notebook 05 checks on the full run; commit. Buffer. **Plan 2 done.** | — |
| **Mon Nov 30 – Tue Dec 1** | Plan 3: `evaluate.py` + tests, `plots.py`; **notebook 06** on the real forecasts, per RV measure. | — |
| **Tue Dec 1 – Wed Dec 2** | **Notebook 07** (event table and figures, fact checks). **Minimum viable version complete by Wed Dec 2.** Notebook 08 in-sample items start (order in §7.8); queue the OOS robustness runs Wed night. | In-sample placebos ≈ 30 min; OOS runs ≈ 0.5–1.5 h each, overnight |
| **Thu Dec 3** | Notebook 08: remaining items, evaluate the overnight runs; E1 only if everything else is done. **Analysis freeze in the evening:** all tables, figures and executed notebooks committed. | — |
| **Fri Dec 4** | Reproducibility check (§10); hand-off to Linnéa (report and presentation). Write-up is out of scope. | Optional full rerun of 05 |

**If dates slip [our call]:**
- **BigQuery:** still blocked on Mon Nov 23 evening → switch to a personal Google account. Still blocked on Tue Nov 24 → zip-streaming fallback (R1), about +1 day: everything shifts a day and all OOS robustness runs are dropped.
- **Dukascopy:** `dukascopy-python` fails → `npx dukascopy-node` the same day. If both fail by Tue Nov 24 evening, go on **without intraday data** and tell the team:
  - set `MAIN_RV = "rv"`, `RV_MEASURES = ("rv",)`, `X_RV_COL = "rv"`, `X_SAMPLE_START = SAMPLE_START`;
  - drop `HAR_h` from `BENCHMARKS_MAIN` (it would equal `HAR`);
  - HARX/GARCHX then run on the daily `rv` from 2005. The plans need no other branch.
- **φ** not frozen by Fri Nov 27 → cut the bootstrap to 99 replicates.
- **Spec** not frozen by Sun Nov 29 → robustness becomes in-sample only (no OOS robustness runs) and the conditional GW test is dropped.
- **Full run** not finished by Mon Nov 30 → minimum-viable mode: notebooks 06 and 07 produce only the MVP items; notebook 08 is limited to the in-sample placebo, the HARX placebo and in-sample AI-GPR.
- **Thu Dec 3** → nothing new is started; unfinished checks in 08 become stated limitations.

**Minimum viable version** (plan §12, extended Nov 23) → notebooks:
- M1–M3 with GDELT (03, 04, 05);
- GARCH-t and HAR (05), plus **HARX_CI and GARCHX_CI** (05; HAR_h, HARX_C and GARCHX_C come with them at no extra cost);
- main test and 2026 holdout (05, 06), evaluated against **`rv_hourly`** as the main measure (02, 06);
- QLIKE, MZ, DM (incl. **M3 vs HARX_CI and vs GARCHX_CI**), Clark-West (06);
- the cumulative-loss figure (06);
- the event table with M2 vs M3 probabilities (07).

Data notebooks 01–02 are prerequisites.

**Cut first** (in this order) **[our call, extending plan §12]:**
1. E1 Markov-switching HAR (08);
2. OOS placebos for the MS models (08 #1; keep in-sample; the HARX placebo is minutes and stays);
3. M3-tone OOS run (keep it in sample, notebook 04);
4. `rv_gk` (keep the column; drop its tables);
5. Student-t errors (08 §4);
6. the fixed 10-year rolling window (not built unless time; `oos.run_oos` would take a `window` argument);
7. Brent implied vol;
8. OOS runs for AI-GPR / GDELT-geo / OVX / hourly control (keep their in-sample versions);
9. Giacomini-White conditional (keep unconditional).

---

## 12. Left to the builder

- Exact AI-GPR URL; GPR date-column name; FRED header variant.
- Backfile classification after D1 (`GDELT_FEED_RULE`), and whether D2 is needed.
- Running the GDELT 2.0 splice (procedure in §6.4), only if D1 shows the 1.0 feed stops before Nov 6, 2026.
- `N_RESTARTS`, `RESTART_EVERY`, restart noise scale and optimizer tolerances, from the timing stage; `SIGMA2_RATIO_MIN`, `VAR_FLOOR_FRAC`, `MIN_DURATION` reviewed on validation. All frozen at validation.
- Readable parameter names, plot styling, figure layout, table formatting.
- Whether to parallelize robustness runs (`joblib`) if time is short.
- Handling of any week with missing GPR/VIX at the end of the sample (report, don't fill silently; ask the team if it affects the holdout).
- Dukascopy (added Nov 23): the exact `dukascopy-python` call (or `npx dukascopy-node` flags), chunk size and retries; whether the feed returns zero-volume filler bars; the terms-of-use finding (`INTRADAY_REDISTRIBUTE`). The **roll rule is not the builder's alone**: it is agreed with the user from the `roll_candidates` table (§6.8).
- GARCH-X optimizer details (tolerances, restart if SLSQP fails) within the §8.7 specification; event-plot layout for the five panels.
- Anything in §14 resolved differently later: update `config.py` and this spec together.

---

## 13. Implementation risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | UCSD Google account blocked from Google Cloud | Personal Google account for the sandbox (only the builder needs one). Last resort: stream GDELT zips with a resumable script that aggregates on the fly and deletes each zip (rejected approach, kept as a fallback). |
| R2 | GDELT 1.0 feed stops before Nov 6, 2026 or has gaps | D1 detects it; plan §4.3 splice with GDELT 2.0 after an overlap check; gaps listed and `gdelt_days` carried in the panel. |
| R3 | Backfile `DATEADDED` holds a 2013 load date | D1 + `GDELT_FEED_RULE`; panel check that no single day holds an outsized share of 2005–2013 articles. |
| R4 | Query scans more than expected | Dry run first; `maximumBytesBilled` hard cap; only needed columns referenced. |
| R5 | Constrained M3 fit unstable or LL(M3) < LL(M2) | Nesting starts, restarts, degeneracy rule; variant fallback `iran_both_logits` decided on validation only; inside a frozen run, failures are logged, never switched. |
| R6 | Label flips mid-loop | Canonicalize after every fit; `relabeled` logged; warm starts in canonical labels. |
| R7 | OOS runtime blows up | Timing stage first (validation origins only); restart schedule; checkpoint/resume with settings hash; run overnight; cut list. |
| R8 | Lookahead through lags, standardization, thresholds | Lags only in `design.py` (tested, grep test); standardization and the high-vol threshold from TRAIN, GW threshold from VALID, nothing from 2019+; `SPEC_FROZEN` guard; filtered never smoothed. |
| R9 | FRED refuses scripted requests | Generic User-Agent; FRED API key fallback. |
| R10 | statsmodels version differences (param names, DM function) | Pin via `uv.lock`; index params by name; tests cover both. |
| R11 | Stale kernels; committed outputs out of sync with code | `autoreload`; restart and run all before committing a notebook (§5.4). |
| R12 | Size creep (figures, checkpoints, notebook outputs) | `check_budget()` in every notebook and in tests (incl. per-notebook limits); checkpoints ignored; inline figures at 100 dpi; notebooks committed at milestones only. |
| R13 | A fresh clone picks Python 3.14 | `.python-version` + `requires-python` make uv fetch 3.12. |
| R14 | Dukascopy is a CFD (dealer quotes tracking front-month ICE futures), not an exchange price; level and dynamics may differ from FRED spot | Level check on 2007–2012 in notebook 02 (correlation, mean ratio band [0.8, 1.25]); no rescaling; flag to the team if outside. Optional exchange cross-check (FirstRate, ICE Brent hourly from 2008; §4.3). Daily `rv` stays as the secondary measure, so every headline result can be read on both. |
| R15 | Roll jumps inflate `rv_hourly` | Candidate lister, rule agreed with the user and stored in `INTRADAY_ROLL_RULE` (part of the settings hash), every removed return logged; notebook 02 compares roll weeks with the rest. |
| R16 | Dukascopy unavailable, API different from expected, rate-limited, or terms forbid redistribution | Access checked on day 1 (Nov 23); `npx dukascopy-node` fallback; yearly chunks with retries; no account needed. Terms → git-ignore the derived files (§5.3). Last resort: the no-intraday fallback in §11, a config switch. |
| R17 | GARCH-X: γ on the boundary (0), weak identification of ω vs. γ, optimizer stuck | Chi-bar-squared p-values (§8.7), SEs flagged when γ̂ = 0; nesting start from GARCH-t so LL never falls below it; SLSQP with bounds; simulation-recovery test; non-converged weeks logged. |
| R18 | More models and measures → many tests, some "significant" by chance | Headline pre-specified (M3 vs M2 on `rv_hourly`); extra-model comparisons labelled secondary (`family` column); "best linear model" chosen on validation only; number of tests stated in notebook 06; no model is promoted on test-window results. |
| R19 | Hourly data gaps (missing hours or days in the feed) | Gap returns go into the next bar, so the weekly measure still spans close to close; `n_hours_week` flagged below 60; a zero-hour week raises; days with `n_hours` < 12 listed in notebook 01. |
| R20 | Conflict and cooperation shares are highly collinear with each other and with `iran_gdelt`, so M3-tone's β's are imprecise | Correlations reported in notebook 02; the Wald test of equality is the main M3-tone statistic, not the individual β's; M3-tone is not used for the headline. |

---

## 14. Issues found in the plan (resolution log)

These came up while turning the plan into an implementation. A review on Nov 20 found all of them resolvable inside this spec without a new research decision. The Resolution column records what the spec does. The list stays as a log, so the report can disclose the choices.

| # | Where | Issue | Resolution |
|---|---|---|---|
| 1 | §3 RQ1 / §14.4 vs §7 | The φ decision is made by RQ1's in-sample test on 2005–2018 (§3, §14.4), but §7's validation row says validation decides "φ switching or not". | **Resolved:** follow plan §14.4 (settled). φ comes from the in-sample LR on 2005–2018 (notebook 03). Validation freezes only the search settings, fit thresholds and the M3 variant. |
| 2 | §3, §14.4 | No significance level for the φ decision. | Interim choice accepted: 5% LR test. |
| 3 | §3 RQ1 | "1 vs 2 regimes" does not say which 2-regime model or how many bootstrap draws. | Interim choice accepted: M1 (switching μ, φ, σ²) vs. one-regime AR(1); 199 draws with the same search effort as the actual fit; degenerate replicates counted (§7.3). |
| 4 | §8, §3 RQ3 | Clark-West is a squared-error test, but it is listed next to QLIKE with no loss stated. | Interim choice accepted: CW on squared error of RV forecasts; QLIKE comparisons via DM and GW. |
| 5 | §8 | Giacomini-White assumes a rolling (bounded) estimation window; the plan uses an expanding window. | Run GW as planned and report two caveats: (a) the rolling-window assumption; (b) with nested models and an expanding window the unconditional GW is near-degenerate: as β̂ → 0 the M3 − M2 loss differential shrinks toward zero and its variance collapses, so the test's size is unreliable. CW is the primary nested test. The fixed 10-year window (cut list) would address (a). |
| 6 | §8 | GW instrument "lagged Iran news is high" has no threshold. | **Resolved:** 1{iran_gdelt(t) > 75th pct of **2013–2018**}, dated at the origin t. The validation window is the daily-feed era; the 2005–2012 backfile may sit at a different level (Apr 2013 break). Assert the share of 1s in the test window is in [10%, 90%] (§7.6). |
| 7 | §8 | DM for M3 vs M1 (and M3 vs M2) compares nested models; DM's normal limit does not hold for nested models under the null (Clark & McCracken, 2001). | Interim choice accepted: DM-HLN reported with the nested-model caveat; CW is the primary nested test. |
| 8 | §8 | DM loss for M3 vs M1/GARCH/HAR unspecified. | Interim choice accepted: QLIKE primary, squared error secondary. |
| 9 | §8 / §14.3 | π_H "hit rate" needs a cutoff. | Interim choice accepted: π_H > 0.5. |
| 10 | §4.2 / §7 | "Training-window means and SDs": fixed 2005–2012 or each expanding window? | Interim choice accepted: fixed 2005–2012 everywhere (OVX: 2007–2012). |
| 11 | §4.2, §5, §4.4 | AI-GPR publication lag ("last published month") is unspecified; §4.4 says "all unlagged" while AI-GPR needs an availability rule. | Interim choice accepted: month m usable from month m + 2, applied in the panel; model lag in `design.py`. Verify the real release timing if possible. |
| 12 | §4.3 | The DATEADDED/SQLDATE switch assumes backfile rows have a usable `DATEADDED`; unverified for the BigQuery table. | Interim choice accepted: diagnostic D1 before Q1 (§6.4). |
| 13 | §5, deck slide 12 | "Known at the Friday close": Friday's GDELT daily file and GPRD are published after the close. Not a lookahead for RV(t+1), which starts with Monday's return. | Interim choice accepted: data dating unchanged; wording "after the Friday close, before Monday's open". |
| 14 | §9 | "Filtered high-vol probability" for events: from which fit? Dec 2011 and 2008 lie in the training window, so they have no OOS forecasts or losses. | Interim choice accepted: in-sample filtered P(H) (2005–2025 fit) plus OOS π_H plotted at the target week; losses only where OOS exists (§7.7). |
| 15 | §9 | Non-Iran comparisons have no frozen anchor dates. | Interim choice accepted: Sep 15, 2008; Mar 9, 2020; Feb 24, 2022. Mar 9, 2020 is the OPEC+ price-war day (Saudi–Russia), still a non-Iran shock and the same event week (ending Mar 13) as the WHO pandemic declaration, so it is acceptable. |
| 16 | §10 | Which robustness checks are in-sample vs. OOS is not stated (except AR term and placebo). | Interim choice accepted: table in §7.8. Trimmed checks run in-sample only; OOS robustness runs start at origin 2018-12-28 with a fresh first fit. |
| 17 | §10 | Circular-shift placebos over the full sample would move 2026 values into earlier weeks. | Interim choice accepted: shifts wrap within 2005–2025 only. |
| 18 | §6.4 | The runtime estimate ("well under an hour") counts single fits only. | Interim choice accepted: timing stage, restart schedule, cut list; OOS placebo ≈ 8 h (365 origins). |
| 19 | §2, §6.3 | MS and GARCH forecast E[r(t+1)²]; HAR forecasts RV directly. They match in expectation only if daily returns are serially uncorrelated. | Interim choice accepted: GARCH uses h + μ²; MZ results read with this in mind. |
| 20 | Deck slide 4 vs plan §3 / deck slide 7 | Slide 4 lists DM for M3 vs M2; the plan uses CW + GW for M3 vs M2 and DM for the others. | **Resolved:** report DM, CW and GW for M3 vs M2 (covers both), with the nested-model caveat on DM and GW (§7.6). |
| 21 | §16 | "Largest weekly gain since 1987" needs Brent back to 1987. | **Resolved:** notebook 01 fetches Brent from 1987-05-20 into the frozen `brent_daily.csv` (≈ 250 KB); notebook 07 uses local data only. |

**Plan wording to update when convenient** (the plan stays the research record; these edits only bring it in line with the resolutions):
- §7, Validation row: replace "φ switching or not" with "search settings, fit thresholds and the M3 variant (φ is decided by RQ1, §14.4)".
- §3 RQ3 and §8: list DM, CW and GW for M3 vs M2 with the nested-model caveat; state that CW uses squared error.
- §8: define the GW instrument (75th percentile of the 2013–2018 Iran share, dated at the origin).
- §5 and deck slide 12: "forecast at the Friday close" → "after the Friday close, before Monday's open".
- §4.2: state the AI-GPR availability rule; §9: name the three non-Iran anchor dates.
- §6.4: note that the runtime estimate excludes constrained refits, restarts and robustness loops.
- §13: the team split no longer describes how the code is built (one builder; a teammate writes the report).

**Plan and deck wording implied by the Nov 23 additions** (the research plan itself is not edited here):
- §1 and §2: the main forecast target is still next-week Brent variance. It is now **scored mainly against realized variance from hourly prices** (Dukascopy `BRENT.CMD/USD`, a CFD on front-month ICE Brent futures, from Nov 2006), with the daily-return RV of spot as a secondary measure. "RV of Brent spot" → "RV from hourly Brent futures-CFD prices (main) and from daily spot returns (secondary)".
- §4.2 variables table: add a row "Brent intraday | Dukascopy `BRENT.CMD/USD`, 1-hour bid bars | Hourly | Nov 2006 | Σ squared hourly log returns, Fri 22:00 UTC → Fri 22:00 UTC, roll returns removed | Main RV measure for evaluation". Add GDELT `QuadClass`, `AvgTone`, `GoldsteinScale` to the GDELT row (conflict/cooperation shares, tone; robustness and M3-tone). Deck data slide (slide 10): same additions.
- §4.3: add the conflict/cooperation split and article-weighted tone to the GDELT definition (§9.1 here), dated the same way.
- §4.4 deliverable columns: add `iran_conflict, iran_coop, iran_tone, iran_goldstein, rv_hourly, log_rv_hourly, n_hours_week, rv_gk`.
- §6.2 model ladder: add M3-tone (conflict and cooperation shares in N→H instead of the total share) as a variant answering "does the tone of Iran news matter?".
- §6.3 benchmarks: add HAR-X (HAR on hourly RV + GPR, VIX, and + Iran) and GARCH-X (GARCH-t + lagged hourly RV, GPR, VIX, and + Iran in the variance) as **extra models whose role is decided after the results**. This is not GJR-GARCH, which stays parked (§17).
- §3 RQ3 and §8: DM of M3 vs HARX_CI and GARCHX_CI ("do regimes beat a linear news effect?"); HARX_CI vs HARX_C and GARCHX_CI vs GARCHX_C (DM and CW: does Iran help a linear model?); every loss table per RV measure.
- §10 robustness: M2/M3 with lagged hourly log RV as the control; tone and Goldstein variants; placebo for HARX_CI; optional Markov-switching HAR (E1).
- §12 timeline and minimum viable version: as in §11 here (start Nov 23, freeze Dec 3, hand-off Dec 4; MVP adds `rv_hourly`, HARX_CI, GARCHX_CI; E1 is cut first).
- §15 risks: CFD vs. exchange price; roll jumps; multiple comparisons.
- §18 references: add Engle (2002) and Han & Kristensen (2014) for GARCH-X, Corsi (2009) for HAR, Garman & Klass (1980), and Andersen & Bollerslev (1998) for realized variance from intraday returns.
- Deck model slides (after the intro presentation): mention the extra models only as "under evaluation", not as benchmarks.
