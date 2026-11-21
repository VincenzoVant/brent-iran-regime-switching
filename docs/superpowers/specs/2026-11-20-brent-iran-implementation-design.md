# Brent × Iran Regime-Switching — Implementation Design

*Spec date: Fri Nov 20, 2026. Data freeze: Fri Nov 6, 2026. Due: Mon Dec 7, 2026, 6 pm.*
*Research plan (settled, implemented here, not changed): [`Plans/Brent-Iran Regime Switching - Feasibility and Plan.md`](<../../../Plans/Brent-Iran Regime Switching - Feasibility and Plan.md>), cited as "plan §N". Pitch deck: `Plans/Brent-Iran_Pitch_Modern-Sage.pptx` (not in git).*

**Labels.** **[decided]** = decided by the user. **[our call]** = filled in by this spec, with a one-line reason; changeable. Anything not labelled is taken directly from the plan.

---

## 1. Intent

Turn the settled research plan into working analysis: raw data → weekly panel → M1–M3 and benchmarks → expanding-window out-of-sample (OOS) forecasts → evaluation tests and figures → event case studies → robustness. The written report (≤ 25 pages) is **out of scope** and comes later; this spec only has to produce the tables, figures and numbers it will cite.

The work is done in **Jupyter notebooks**, with Claude Code writing most of the code. This document has to be concrete enough that an agent can plan and build one notebook at a time without asking the user anything. A second, equally important goal: **keep local data minimal.** Only what the analysis uses is ever downloaded, and nothing large lands on disk.

---

## 2. Goals, non-goals, anti-goals

**Goals**
1. Every item in plan §3 (RQ1–RQ3), §8 (evaluation), §9 (events) and §10 (robustness) is produced by a named notebook (map in §7.9).
2. A teammate can clone the repo, run `uv sync`, and rerun notebooks 02 → 08 from committed files, without a Google account or network access.
3. One implementation of each risky piece (week mapping, lags, MS fit, forecaster, tests), imported everywhere and covered by small tests.
4. Expensive steps (BigQuery, the OOS loop) never rerun by accident and can resume after a crash.

**Non-goals**
- The written report, abstract and slides.
- Parked items from plan §17 (density forecasts/PIT/CRPS, prediction markets, PortWatch, Bloomberg pull, OPEC/Känzig/Kilian/dollar controls, GPR acts/threats split, placebo countries, 3-state model, winsorizing, `BZ=F`, VaR tests, GJR-GARCH).
- Brent implied vol from Bloomberg (plan §4.2 optional; on the cut list). The code accepts it as a drop-in `vix` replacement if someone gets the data.
- A general-purpose library, CLI, CI, or packaging beyond an editable install.

**Anti-goals** (the project fails even if it runs)
- Large local data: raw GDELT zips, the full GDELT table, any file > 10 MB, total data > 20 MB.
- Lookahead: a predictor dated after the forecast origin; smoothed probabilities anywhere in a forecast; standardization stats, thresholds or instrument cutoffs computed on post-2012 data; using AI-GPR months before they were published.
- Using 2019–2026 to choose anything (specification, search settings, fallback) and above all using 2026 (plan §5, §7).
- Results a teammate cannot reproduce: unpinned environment, unseeded randomness, data pulled again after the freeze, notebooks that only work in the order someone happened to run cells.
- Two forecasters, two lag conventions, or two week mappings that disagree across notebooks.
- Silent label switching (H means "higher σ²" everywhere, after every fit).
- Hiding a failed estimation: every OOS fit records convergence, refits from scratch, degeneracy and restarts.

---

## 3. Constraints

| Constraint | Detail |
|---|---|
| Time | 2.5 weeks incl. Thanksgiving (Nov 26). Plan §12 timeline; write-up starts Fri Dec 4, so analysis must be done by **Thu Dec 3**. |
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
│   ├── sources.py            # fetchers: FRED, GPR, AI-GPR, GDELT (BigQuery)
│   ├── weekly.py             # Friday-week mapping and weekly construction
│   ├── design.py             # THE place where lags and standardization happen
│   ├── ms.py                 # MS models: specs, fitting, relabeling, constrained M3
│   ├── forecast.py           # one-step MS forecaster
│   ├── benchmarks.py         # GARCH(1,1)-t, HAR-RV, random walk
│   ├── oos.py                # expanding-window loop, checkpoints, assembly
│   ├── insample.py           # RQ1/RQ2 tests, bootstrap LR, placebo shifts
│   ├── evaluate.py           # losses and forecast tests
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
    ├── test_weekly.py
    ├── test_design.py
    ├── test_ms.py
    ├── test_forecast.py
    ├── test_evaluate.py
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
- Frozen decisions (φ after notebook 03; settings and M3 variant after notebook 05's validation stage) are written into `config.py` by hand at those points and committed with the matching file in `results/frozen/`.
- Module changes: `uv run pytest` before each commit. Commit after each notebook milestone.
- Reproducibility for the report author: everything from notebook 02 onward runs from committed files, with no Google account and no network (§2, goal 2).

---

## 6. Data acquisition (notebook 01 only)

Every fetch reads into memory, trims to the needed columns and window, and writes one small CSV to `data/raw/`. Each write appends a row to `data/raw/manifest.csv`: `file, source, url_or_table, fetched_at_utc, first_date, last_date, n_rows, sha256, bytes`. **[our call: documents the freeze and makes silent changes visible in git]**

Window for daily sources: **2004-12-01 → 2026-11-06** (the first weekly return needs the Dec 31, 2004 close; a month of margin covers holidays). **Exception: Brent is fetched from 1987-05-20** (its first FRED observation), so the plan §16 fact check "largest weekly gain since 1987" runs on frozen local data in notebook 07; notebook 02 trims Brent to the panel window. **[our call]** No notebook after 01 touches the network.

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

**Query Q1 — main daily aggregate.** Classifying each event row (backfile vs. daily feed, Iran or not) needs event-level fields, so it happens in SQL. **All calendar logic** (Friday weeks, weekend rule, the Apr 1, 2013 switch, the freeze cut, the weekly share) **is in Python** (`weekly.gdelt_weekly`), tested once. **[our call: one tested place for week logic]**

```sql
-- Q1: verify column names on the dry run (Actor1CountryCode, Actor2CountryCode,
-- ActionGeo_CountryCode, NumArticles, SQLDATE, DATEADDED).
WITH ev AS (
  SELECT
    IF(DATEADDED >= 20130401, 'daily', 'backfile')            AS feed,     -- per D1; see GDELT_FEED_RULE
    IF(DATEADDED >= 20130401, DATEADDED, SQLDATE)             AS day_key,  -- daily feed: ingestion day; backfile: event day
    NumArticles,
    (IFNULL(Actor1CountryCode, '') = 'IRN'
       OR IFNULL(Actor2CountryCode, '') = 'IRN')              AS iran_actor,
    (IFNULL(ActionGeo_CountryCode, '') = 'IR')                AS iran_geo   -- FIPS code
  FROM `gdelt-bq.full.events`
)
SELECT
  feed,
  day_key,
  COUNT(*)                                          AS n_events,
  SUM(NumArticles)                                  AS articles_all,
  SUM(IF(iran_actor, NumArticles, 0))               AS articles_iran_actor,      -- main measure
  SUM(IF(iran_actor OR iran_geo, NumArticles, 0))   AS articles_iran_actor_geo,  -- robustness variant
  COUNTIF(iran_actor)                               AS events_iran_actor
FROM ev
WHERE day_key BETWEEN 20050101 AND 20261106
GROUP BY feed, day_key
ORDER BY day_key, feed
```

- One pass returns both the main and the robustness variant.
- The table is not partitioned, so bytes scanned = full size of the referenced columns regardless of `WHERE`. Run with `dry_run=True, use_query_cache=False` first; print `total_bytes_processed`; stop if above `GDELT_MAX_BYTES`.
- Output: `data/raw/gdelt_daily.csv` (`feed, date, n_events, articles_all, articles_iran_actor, articles_iran_actor_geo, events_iran_actor`), ≈ 8,050 rows, ≈ 0.5 MB.
- Expected total BigQuery use: D1 + Q1 ≈ 2 × 30–60 GB (estimate), < 15% of the monthly sandbox quota. Optional diagnostic D2 (daily feed keyed by `SQLDATE`, to separate a dating effect from a feed effect at Apr 2013) only if the level-shift plot in notebook 02 shows a large break.

### 6.5 Freeze rules

- Every source is cut at **2026-11-06** (AI-GPR: months up to 2026-10).
- Snapshots are fetched once (today or soon) and committed; values revised by a source between Nov 6 and the fetch date cannot be undone and are disclosed as such.
- After notebook 01 has run, nothing re-downloads unless `FORCE_FETCH[source] = True` is set by hand.

### 6.6 Notebook 01 skip logic

Each source cell: `if paths.RAW[source].exists() and not FORCE_FETCH[source]: skip`. The GDELT cells also skip when `BRENT_GCP_PROJECT` is unset, printing a message, so someone without a Google account can run the notebook top to bottom without error.

### 6.7 Data-volume budget **[our call]**

| Rule | Limit |
|---|---|
| Any single file written anywhere in the repo (excluding `.venv`, `.git`) | ≤ 10 MB |
| `data/` total | ≤ 5 MB (expected ≈ 1.4 MB, incl. the 1987+ Brent series) |
| `data/` + `results/` (excl. checkpoints) | ≤ 20 MB (expected ≈ 6–10 MB, mostly PNGs) |
| Each executed notebook in `notebooks/` | ≤ 5 MB; all notebooks together ≤ 15 MB (expected ≈ 5–8 MB) |
| Downloads | Only the six sources above; nothing written except the trimmed CSVs |

Enforced by `paths.check_budget()` (walks the repo, raises with the offending paths), called in the last cell of every notebook and in `tests/test_budget.py`. Figures are saved as PNG at 150 dpi.

---

## 7. Notebook-by-notebook design

Conventions for every notebook:
- First cell: imports, `autoreload`, `np.random.seed(C.SEED)` (a backstop only: all fits and draws are seeded explicitly, §8.4), a `FORCE_*` flag block.
- Reads only files written by earlier notebooks; writes only to its own declared outputs.
- Ends with a **checks** cell (asserts) and `paths.check_budget()`.
- Markdown cells carry interpretation; tables/figures are saved via `plots.save_fig(name)` / `paths.save_table(df, name)`.

### 7.1 `01_fetch_data.ipynb`

- **Purpose:** the only notebook that touches the network. Plan §4.2, §4.3, §4.1 freeze.
- **Inputs:** URLs/table from `config`; `BRENT_GCP_PROJECT` env var for GDELT.
- **Outputs:** `data/raw/{brent,vix,ovx,gpr}_daily.csv`, `aigpr_iran_monthly.csv`, `gdelt_daily.csv`, `manifest.csv`.
- **Steps:** FRED ×3 → GPR → AI-GPR → GDELT (dry run D1 → run D1 → decide feed rule → dry run Q1 → run Q1).
- **Checks:** each file's date range covers the window; no duplicate dates; Brent starts by 1987-05-20, has a close on Dec 31, 2004 (or the last trading day before it) and on or before 2026-11-06; GDELT has no feed overlap on the same day and no gap > 3 consecutive calendar days after Apr 2013 (gaps listed, not filled); bytes scanned printed and below cap; budget.

### 7.2 `02_weekly_panel.ipynb`

- **Purpose:** build `weekly_panel.csv` and the data section. Plan §4.1, §4.3 (level-shift check, coverage), §4.4, §5 (weekend rule).
- **Inputs:** `data/raw/*.csv`.
- **Outputs:** `data/processed/weekly_panel.csv`; `results/tables/data_summary.csv`; figures `data_series.png` (one panel per variable), `gdelt_share_2013_break.png`, `gdelt_main_vs_geo.png`.
- **Steps:**
  1. `weekly.build_weekly_panel(raw)` (spec in §9.1).
  2. Data table: for `r, rv, log_rv, iran_gdelt, gpr, vix, ovx, aigpr_iran`: N, mean, SD, min, max, AR(1) coefficient, ADF statistic and p-value (`statsmodels.tsa.stattools.adfuller`, constant, AIC lag choice). **[our call: ADF spec]**
  3. Time-series plots.
  4. Apr 2013 level-shift check: plot `iran_gdelt` with a vertical line at the week ending Apr 5, 2013; report the mean of `iran_gdelt` 52 weeks before vs. after and a simple OLS on a post-2013 dummy (descriptive only). The dummy is carried to robustness (notebook 08).
  5. Coverage: GDELT has a row for every calendar day, so a complete Sat–Fri week has `gdelt_days` = 7. Weeks with `gdelt_days` < 6 are listed and flagged (kept, not filled). **[our call]**
  6. Zero values: a week with 0 Iran articles (main or geo variant) or an AI-GPR month ≤ 0 cannot be logged. `build_weekly_panel` **stops with an error** listing them; nothing is filled or offset (no ln(1 + x)). The team decides what to do.
- **Checks:** 1,140 rows, first 2005-01-07, last 2026-11-06, all Fridays; `rv > 0`; `n_days` between 1 and 5 (list weeks with < 4); `iran_gdelt` finite; window counts 417 / 313 / 365 / 45 (train / validation / test / holdout); Abqaiq (Sat Sep 14, 2019) maps to week 2019-09-20 and Feb 28, 2026 to 2026-03-06 (also unit-tested).

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
- **Outputs:** `results/tables/rq2_coefficients.csv` (all transition coefficients for M2 and M3 in the deck's sign convention, SEs, z, p), `rq2_lr_tests.csv` (M2 vs M1, M3 vs M2), `rq2_fit_summary.csv` (LL, AIC, BIC, durations), `rq1_tests_2005_2025.csv`; `results/fits/insample_filtered_2005_2026.csv` (date, model, xi_H filtered; used by notebook 07); figures `rq2_filtered_prob.png` (2005–2026, M1/M2/M3; same idea as deck slide 11, filtered not smoothed), `rq2_transition_probs.png` (fitted P(N→H) over time, M2 vs M3).
- **Steps:**
  1. Fit M1, M2, M3 (constrained) on 2005–2025. Standardize TVTP regressors with 2005–2012 stats (§9.3).
  2. β = Iran's coefficient in P(N→H), SE from the numerical Hessian of the constrained log-likelihood (§8.4).
  3. LR tests with df = the difference in **free** parameters, computed by `insample.lr_test` from the two fits, never typed in. M3 vs M2: df = 1 under `constrained`, df = 2 under `iran_both_logits`. M2 vs M1: df = 6 (three controls in two logits). Report the joint LR and, separately, the N→H Iran coefficient β with its z-test (the two coincide only when df = 1).
  4. **RQ1 reported fit on 2005–2025** (plan §3): rerun the 1-vs-2 LR and BIC, the μ and φ equality tests, expected durations and weeks per regime on 2005–2025 → `rq1_tests_2005_2025.csv`. The bootstrap is optional here (only if time). φ stays frozen from 2005–2018 whatever these show.
  5. Supplement: refit all three on 2005–2026, same tables with suffix `_2026`, clearly labelled as a supplement.
  6. Filtered probabilities for 2026 from the 2005–2025 parameters (filter run forward with fixed params; this is what notebook 07 uses for in-sample event plots).
- **Checks:** constraint holds exactly (coefficient == 0.0) under `constrained`; LL nesting; relabel applied; no degenerate fit; `config.M3_VARIANT` equals the value in `spec_freeze.json` (rerun this notebook if validation switched the variant).

### 7.5 `05_oos_forecasts.ipynb` — expensive

- **Purpose:** the expanding-window OOS loop. Plan §6.3–6.4, §7.
- **Inputs:** `weekly_panel.csv`, `config`.
- **Outputs:** `results/forecasts/forecasts_main.csv`, `results/fits/params_main.csv`, `results/fits/fitlog_main.csv`, `results/frozen/spec_freeze.json`; figure `oos_runtime_and_convergence.png` (fit time and restarts by week).
- **Stages** (a `STAGE` variable at the top):
  1. **`timing`:** run the **first 10 validation origins** (2012-12-28 → 2013-03-01) for each model, never a test-window origin; print seconds per fit; extrapolate total runtime; choose `C.N_RESTARTS` and `C.RESTART_EVERY` so the full main run fits in ≈ 2 h. Output goes to a throwaway run (`run_id = "timing"`) that is never resumed. **[our call]**
  2. **`validation`:** origins 2012-12-28 → 2018-12-21 (targets 2013-01-04 → 2018-12-28, 313 weeks). All models. Then review on validation only: convergence rate, label flips, restarts that improved LL, constrained-M3 stability. Decide and freeze in `config`: `SEARCH_REPS_*`, `N_RESTARTS`, `RESTART_EVERY`, `SIGMA2_RATIO_MIN`, `VAR_FLOOR_FRAC`, `MIN_DURATION`, `M3_VARIANT` (`"constrained"` or the plan §6.4 variant fallback `"iran_both_logits"`). Write `spec_freeze.json` (all frozen settings, their `settings_hash`, git hash) and set `config.SPEC_FROZEN = True`.
     - **[our call] stability rule for the variant fallback:** switch to `"iran_both_logits"` only if, on validation, the constrained fit fails to converge or violates LL(M3) ≥ LL(M2) − 1e-6 after restarts in more than 5% of weeks.
  3. **`full`:** refuses to run unless `SPEC_FROZEN`. Resumes from the validation checkpoint **only if** the checkpoint's `settings_hash` equals the one in `spec_freeze.json`. Otherwise (validation changed a setting) it discards the checkpoint and reruns from origin 2012-12-28 with the frozen settings (≈ +40% runtime). Continues to the last origin 2026-10-30 (target 2026-11-06). Test and holdout are produced in one pass; splits are assigned by target date. The M3 variant never changes inside the run (§8.8).
- **Models per origin:** M1, M2, M3 (MS); GARCH(1,1)-t; HAR-RV; RW (return only).
- **Main steps per origin t** (inside `oos.run_oos`, §8.8): build design through t → fit each MS model (warm start from t−1's canonical params, restarts per schedule, relabel) → one-step forecasts for t+1 → benchmarks → append to checkpoint.
- **Runtime estimate:** loglike ≈ 0.4 ms and an unconstrained TVTP fit ≈ 1.5 s on 1,100 obs (measured on synthetic data). Per origin: M1 ≈ 0.3 s, M2 ≈ 1–2 s, M3 ≈ 2–6 s (constrained refit + restarts), GARCH ≈ 0.1 s, HAR ≈ ms. 723 origins ≈ **1–2 h** for the main run. Run overnight or while doing other work; the checkpoint makes interruption harmless.
- **Checks:** 723 target weeks per MS model and target; no target date ≤ its origin; every forecast uses `origin = date − 7 days`; `rv` forecasts > 0; π_H in [0, 1]; every checkpoint and output row carries the frozen `settings_hash`; fitlog convergence rate, `refit_from_scratch` count, non-converged M3 weeks and degenerate fits printed.

### 7.6 `06_evaluation.ipynb`

- **Purpose:** every test and figure in plan §8, split **validation / test (2019–2025) / holdout (2026)**, test and holdout reported separately. Plan §3 RQ3, §8, §14.3.
- **Inputs:** `forecasts_main.csv`, `weekly_panel.csv` (actuals), `config` (events, thresholds).
- **Development:** if the full run is not finished yet, the code can be built against a fake `forecasts_main.csv` generated by `oos.fake_forecasts(panel)` (same schema; for building code only, never reported).
- **Guard:** test and holdout metrics only when `config.SPEC_FROZEN` is True. Validation metrics are shown as an appendix table.
- **Outputs:** tables in `results/tables/`; figures in `results/figures/` (names below).

| Target | Tables | Figures |
|---|---|---|
| Volatility (main) | `vol_losses.csv` (QLIKE, RMSE per model × split); `vol_mz.csv` (a, b, HAC SEs, joint Wald (0,1) p); `vol_dm.csv` (M3 vs M2, M1, GARCH-t, HAR; QLIKE and squared error; DM-HLN; the nested pairs M3 vs M2 and M3 vs M1 carry the nested-model caveat); `vol_cw_gw.csv` (M3 vs M2: Clark-West on squared error; Giacomini-White unconditional and conditional, QLIKE; caveats in §14 items 5 and 7) | `vol_fc_vs_actual_{test,holdout}.png` (RV and forecasts, Iran event weeks shaded); **`cumloss_M2_minus_M3.png`** (Σ(L_M2 − L_M3), QLIKE, 2019–2026, split line at Jan 1, 2026, event weeks marked); `vol_error_acf.png` |
| Returns (check) | `ret_eval.csv` (RMSE vs RW, ratio, sign hit rate, Pesaran-Timmermann stat and p, per MS model × split) | `ret_fc_vs_actual.png`; `ret_error_acf.png` |
| High-vol prob (side) | `phigh_eval.csv` (Brier, hit rate on weeks with \|r\| > training 90th pct, base rate; optional Brier of mixture-implied P(\|r\| > thr)) | `phigh_reliability.png` (10 bins); `phigh_vs_bigmoves.png` |
| All | `ljung_box.csv` (forecast errors, lags 4, 8, 13) | — |

- **[our call] details:** QLIKE = RV/f − ln(RV/f) − 1 (Patton 2011). DM loss: QLIKE primary, squared error secondary. DM: `lags=0` with `harvey_adj=True, horizon=1` (at h = 1 the loss differential is serially uncorrelated under the null, which is the case the HLN correction is built for). MZ uses Newey-West with 4 lags. GW instrument: `1{iran_gdelt(t) > 75th percentile of 2013–2018}`, dated at the **origin** t (it conditions the forecast for t+1; `gw_instrument` takes origins). The threshold comes from the validation window because 2013–2018 is the daily-feed era; the 2005–2012 backfile may sit at a different level (Apr 2013 break). Assert that the instrument's share of 1s in the test window is between 10% and 90%; otherwise stop and report (§14, item 6). π_H "hit" = π_H > 0.5. Mixture-implied exceedance probability is already in `forecasts_main.csv` (`target = p_exceed`), so the optional score costs one line.
- **Checks:** forecasts and actuals aligned on target date; no NaN losses; DM/CW/GW reproduce the toy cases in `tests/test_evaluate.py`; cumulative-loss series ends at the sum of per-week differences.

### 7.7 `07_events.ipynb`

- **Purpose:** event case studies, plan §9; fact checks, plan §16.
- **Inputs:** `weekly_panel.csv`, `data/raw/brent_daily.csv` (1987+, for fact checks), `forecasts_main.csv`, `insample_filtered_2005_2026.csv`, `config.EVENTS`, `config.NON_IRAN_EVENTS`.
- **Outputs:** `results/tables/event_table.csv` (MVP item: per event, event week, M2 and M3 values at weeks −4…+4 for in-sample filtered P(H), OOS π_H at the target week where available, weekly QLIKE losses where available); `results/tables/deck_fact_checks.csv`; figures `event_<id>.png` (one per event, 9 Iran + 3 non-Iran) and `events_grid.png`.
- **Steps:**
  1. Map each event date to its event week with `weekly.week_ending` (weekend rule built in).
  2. ±4-week window: top panel = in-sample filtered P(H) (M2, M3; 2026 filtered with 2005–2025 params); middle = OOS π_H (M2, M3) plotted at its **target** date: x = event-week offset k = −4…+4, so the value at k = 0 is the forecast made at the Friday close before the event week; y-axis label "P(high vol in week k), forecast at end of week k−1"; bottom = weekly QLIKE loss (M2, M3). Only plot what exists: Dec 2011 and 2008 fall in the training window, so they have no OOS rows (§14, item 14).
  3. Non-Iran comparisons: 2008, COVID-19, Russia–Ukraine (anchor dates in config, §8.1).
  4. Deck fact checks (plan §16): Brent Feb 27 → 2026 peak and its date; the week-to-Mar-6, 2026 weekly change and its rank among all Friday-to-Friday changes since 1987, from the frozen `brent_daily.csv` (starts 1987-05-20; same week rules as the panel); the seven event moves on deck slide 2 (1-day and 5-day changes from the spot series). No network access.
- **Checks:** every event week exists in the panel; holdout event marked "holdout, not in the test" in every output.

### 7.8 `08_robustness.ipynb`

- **Purpose:** plan §10 in its priority order (§12), plus the post-2013 GDELT dummy (§4.3). Each section has its own `FORCE_<check>` flag and saves its own outputs, so a half-finished notebook still yields completed checks.
- **In-sample runs use 2005–2025; OOS runs reuse `oos.run_oos` with `run_id=<check>` and are evaluated with the same `evaluate` functions as notebook 06.** **[our call: which checks are in-sample vs. OOS, see table]**
- **OOS robustness runs** start at origin 2018-12-28 (first target 2019-01-04) with a fresh first fit (full first-fit procedure, `SEARCH_REPS_FIRST`) on data through 2018, then warm-start weekly with the frozen settings. They never rerun validation. **If trimming is needed**, AI-GPR, GDELT-geo and OVX run **in-sample only**.

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

- **Placebo details [our call]:** shifts drawn uniformly from offsets at least 52 weeks away from 0 (mod the sample length), seed `C.SEED`. Shifts wrap within **2005–2025 only**, so 2026 values never enter (plan §5: 2026 never used to choose anything). OOS placebos are evaluated on 2019–2025 only.
- **Iran-in-the-mean [our call]:** common (non-switching) coefficient on M3, using `switching_exog=[PHI_SWITCHING, False]`; one parameter, one test.
- **Runtime:** in-sample placebo ≈ 200 × 5–10 s ≈ 20–35 min. Each extra OOS run ≈ 0.5–1.5 h. OOS placebo ≈ 20 × 365 origins × 4 s ≈ 8 h (only with spare time; parallelizable with `joblib` across shifts).
- **Checks:** each robustness run's fitlog convergence printed; the placebo's actual-data LR equals notebook 04's LR (same code path).

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
PHI_SWITCHING: bool | None = None           # set from notebook 03, then frozen
M3_VARIANT = "constrained"                  # or "iran_both_logits" (set in notebook 05 validation)
SPEC_FROZEN = False                         # True after notebook 05 validation stage
SEARCH_REPS_FIRST = 30; N_RESTARTS = 3; RESTART_EVERY = 13   # last two revisited in 05 timing
SIGMA2_RATIO_MIN = 1.5                      # restart trigger; frozen at validation
VAR_FLOOR_FRAC = 0.05; MIN_DURATION = 2.0   # degeneracy rule (§8.4); frozen at validation
GDELT_SPLICE_WEEK = None                    # set only if the GDELT 2.0 splice is needed
RV_FLOOR_FRAC = 0.01                        # HAR floor = 1% of min training RV
BOOT_REPS = 199; PLACEBO_IN = 200; PLACEBO_OOS = 20; PLACEBO_MIN_SHIFT = 52
GDELT_TABLE = "gdelt-bq.full.events"; GDELT_MAX_BYTES = 200 * 10**9
GDELT_SWITCH_DATE = "2013-04-01"; GDELT_FEED_RULE = "dateadded>=20130401"  # revisit after D1
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
GDELT_SQL_D1: str; GDELT_SQL_Q1: str                                      # templates (§6.4)
save_snapshot(df: pd.DataFrame, name: str, source: str, url: str) -> Path  # CSV + manifest row
load_raw(name: str) -> pd.DataFrame
```

The Google imports happen inside the GDELT functions, so `import brent` works without the `gdelt` group.

### 8.4 `ms.py`

```python
@dataclass(frozen=True)
class MSSpec:
    name: str                       # "M1" | "M2" | "M3" | robustness names
    tvtp_vars: tuple[str, ...]      # () for M1; controls for M2; controls + iran for M3
    iran_var: str | None            # restricted variable (None for M1/M2)
    iran_only_NH: bool              # True for constrained M3
    phi_switching: bool
    mean_exog: tuple[str, ...] = () # e.g. ("iran_gdelt",) for Iran-in-the-mean

@dataclass
class MSFit:
    spec: MSSpec; params: pd.Series  # canonical labels: regime 0 = N, regime 1 = H
    loglike: float; converged: bool; n_restarts_used: int; relabeled: bool
    refit_from_scratch: bool; degenerate: bool; seconds: float; model: MarkovRegression

make_model(design: Design, spec: MSSpec) -> MarkovRegression
fit_ms(design, spec, start_params: pd.Series | None, search_reps: int, n_restarts: int, seed: int) -> MSFit
canonicalize(params: pd.Series, model) -> tuple[pd.Series, bool]   # → N=0, H=1 (H = higher sigma2)
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

standardization_stats(panel, vars, window) -> pd.DataFrame
build_design(panel: pd.DataFrame, spec: MSSpec, end: str | pd.Timestamp,
             start: str = SAMPLE_START, iran_var: str | None = None,
             std_window=STANDARDIZE_WINDOW) -> Design
```

- Row t of `X` holds r(t−1); row t of `Z` holds [1, z(t−1)]: the regressors that drive S(t−1) → S(t) (plan §6.4).
- The forecast for t+1 uses `x_next` = r(T) and `z_next` = z(T): known at the Friday close of T (plan §5).
- The first panel week (2005-01-07) has no r(t−1), so estimation starts at **2005-01-14** (1,139 rows for the full sample). **[our call: simpler than fetching Dec 2004 weekly returns]**
- `har_design(panel, end) -> (X, y, x_next)` also lives here: X = (RV(t), mean RV(t−3..t), mean RV(t−12..t)), y = RV(t+1), x_next = the same regressors at T.
- `iran_var` lets robustness swap in `aigpr_iran`, `iran_gdelt_geo`, or a placebo-shifted column without touching lag code. Benchmarks (`benchmarks.py`) get their lagged inputs from the same module via `har_design(panel, end)`.
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
filter_fixed(model_for_longer_sample, params) -> pd.Series  # filtered ξ_H with fixed params (notebook 04/07)
```

- ξ(T|T) = `results.filtered_marginal_probabilities` at T (never smoothed); P(T+1) = `model.regime_transition_matrix(params, exog_tvtp=z_next[None, :])[:, :, 0]` (M1: no exog); π = P · ξ; mᵢ = μᵢ + φᵢ·r(T) (+ θ·Iran(T) if Iran-in-the-mean).
- Columns of P sum to 1, P[i, j] = P(S(T+1)=i | S(T)=j). Verified: P[:, :, t] · ξ(t−1|t−1) reproduces statsmodels' `predicted_marginal_probabilities[t]` to 1e-15.
- One function, imported by notebooks 03–08. No notebook computes a forecast by hand.

### 8.7 `benchmarks.py`

```python
garch_t_forecast(r_hist: pd.Series) -> dict      # {"var_fc": h + mu^2, "mean_fc": mu, "params": ...}
har_forecast(panel, end) -> float                # OLS in levels, floored at RV_FLOOR_FRAC × min training RV
rw_forecast() -> float                           # 0.0 (return)
```

- GARCH(1,1)-t: `arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")`, warm-started from last week's params. **[our call]** `var_fc = h(T+1) + μ̂²` so it targets the same uncentered second moment as the MS forecast (RV is uncentered; plan §6.4).
- HAR: RV(t+1) = b₀ + b₁·RV(t) + b₄·mean(RV(t−3..t)) + b₁₃·mean(RV(t−12..t)) + e; OLS in **levels** **[our call: standard Corsi form; log-HAR needs a bias correction]**; floor = 1% of the minimum training RV, floors counted in the fitlog.

### 8.8 `oos.py`

```python
origins(start_target: str, end_target: str) -> list[pd.Timestamp]   # origin = target - 7 days
run_oos(panel, run_id: str, ms_specs: list[MSSpec], benchmarks: bool,
        origins: list[pd.Timestamp], iran_var: str = C.IRAN_VAR,
        force: bool = False, max_origins: int | None = None) -> None
assemble(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]  # forecasts, params, fitlog → results/
load_forecasts(run_id: str, with_actuals: bool = True) -> pd.DataFrame
fake_forecasts(panel) -> pd.DataFrame            # schema-correct dummy for building notebook 06 early
```

- **Expanding window**, refit every week, origins 2012-12-28 → 2026-10-30.
- **Warm starts and restarts:** each MS model starts from its previous canonical params. Random restarts (`N_RESTARTS`) run every `RESTART_EVERY` weeks and whenever the warm-start fit fails a sanity check: not converged; the new LL is below `model.loglike(prev_params)` evaluated on the **current** sample (the optimizer ended below its own start; last week's LL is not comparable because the sample grew); σ²_H/σ²_N < `SIGMA2_RATIO_MIN` (frozen at validation); degenerate; or nesting violated. **[our call: restarts every week would roughly triple runtime]**
- **M3 failure inside a frozen run:** the variant never changes mid-run. If no candidate converges, use the best-LL candidate, set `converged = False`, and log it; notebook 06 reports the number of such weeks.
- **Seeds:** for each origin, `seed = C.SEED + origin_index`, where the index is the origin's position in the full origin list (2012-12-28 = 0), so a week gets the same seed in every run. The seed goes to `fit_ms` (→ statsmodels `rng`) and to restart noise; a resumed run gives identical numbers.
- **Checkpoint/resume:** after each origin, append that origin's rows to `results/checkpoints/<run_id>_{forecasts,params,fitlog}.csv` (flush). Every row stores `settings_hash` (hash of the frozen settings: search reps, restarts, restart schedule, thresholds, M3 variant, φ, `iran_var`, model list). On start, resume from the last completed origin **only if** every row's hash equals the current one (for the `full` stage, also the one in `spec_freeze.json`); otherwise discard the checkpoint and start again from the run's first origin. `force=True` deletes the run's checkpoints first. `assemble` writes the final files to `results/forecasts/` and `results/fits/`.
- **Skip logic:** notebook cell calls `run_oos` only if the assembled output is missing or `FORCE_OOS` is True.

### 8.9 `insample.py`

```python
lr_test(fit_restricted: MSFit, fit_full: MSFit) -> dict                      # df = difference in free-parameter counts
bootstrap_lr_1v2(y, X, n_reps: int, seed: int, search_reps: int) -> dict    # RQ1
equality_tests(design, spec) -> pd.DataFrame                                  # μ, φ: LR + Wald (RQ1)
fit_ladder(panel, end, phi_switching, iran_var=C.IRAN_VAR) -> dict[str, MSFit]  # M1, M2, M3 with nesting starts
circular_shift(series: pd.Series, k: int, window: tuple[str, str]) -> pd.Series  # wraps inside window only
placebo_lr(panel, n_shifts, seed, window=("2005-01-07", "2025-12-26")) -> pd.DataFrame
```

### 8.10 `evaluate.py`

```python
qlike(rv, f) -> np.ndarray; sq_err(y, f) -> np.ndarray
loss_table(fc: pd.DataFrame, split_col="split") -> pd.DataFrame
dm_test(y, f_a, f_b, loss: str = "qlike") -> dict           # wraps statsmodels diebold_mariano_test(lags=0, harvey_adj=True, horizon=1)
clark_west(y, f_small, f_big) -> dict                        # adjusted MSPE, one-sided, HAC t
giacomini_white(loss_a, loss_b, instruments: pd.DataFrame | None) -> dict   # Wald on mean(h_t * dL), chi2(q)
mincer_zarnowitz(y, f, hac_lags=4) -> dict                   # a, b, SEs, joint Wald (a, b) = (0, 1)
pesaran_timmermann(y, f) -> dict
sign_hit_rate(y, f) -> float
brier(p, event) -> float; reliability_table(p, event, n_bins=10) -> pd.DataFrame
ljung_box(errors, lags=(4, 8, 13)) -> pd.DataFrame           # statsmodels acorr_ljungbox
cumulative_loss_diff(loss_a, loss_b) -> pd.Series
highvol_threshold(panel) -> float                            # 90th pct of |r| on TRAIN
gw_instrument(panel, origins) -> pd.Series                   # 1{iran(origin) > 75th pct of VALID}; index = origins
```

Each test returns a dict with `stat`, `pvalue`, `n` and any coefficients, so tables are built by stacking dicts.

### 8.11 `plots.py`

`setup_style()`, `shade_events(ax, events, dates)`, `mark_split(ax)` (Jan 1, 2026 line), `save_fig(fig, name)` (PNG, 150 dpi, `results/figures/`), `event_window_plot(...)`. One consistent palette: N/H, M1/M2/M3, benchmarks.

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
| `gdelt_source` | `v1` or `v2`; only present if the GDELT 2.0 splice was needed (§6.4) |

- Variable names follow plan §4.4; `iran_gdelt`, `gpr`, `vix`, `ovx`, `aigpr_iran` hold **logs** (as used in the models). Level series can be recovered with `exp`.
- **Zero values:** a week with 0 Iran articles (either variant) or an AI-GPR month ≤ 0 makes `build_weekly_panel` raise with the list; nothing is filled or offset.
- **Week mapping:** `week_ending(d) = d + ((4 − d.weekday()) mod 7) days` (Friday = 4). Saturday and Sunday map to the next Friday, so the weekend rule (plan §5) is automatic.
- **GDELT:** daily rows with `date < 2013-04-01` come from the `backfile` feed (dated by `SQLDATE`), rows on or after from the `daily` feed (dated by `DATEADDED`); then mapped to weeks and summed (numerator and denominator separately) before taking the share.
- **AI-GPR availability [our call, conservative]:** month m is treated as published from the first Friday of month m + 2 (one full month of publication lag, `AIGPR_PUBLICATION_LAG_MONTHS = 1`). This availability rule is applied in the panel (it defines what is known at t); the model's one-week lag is then applied in `design.py` like every other variable. No other lag is applied in the panel.

### 9.2 `results/forecasts/forecasts_<run>.csv` (long format)

| Column | Meaning |
|---|---|
| `date` | target week t+1 (Friday) |
| `origin` | forecast origin t = date − 7 days |
| `split` | `validation` / `test` / `holdout`, by target date |
| `run` | `main`, `ar_alt`, `aigpr`, `gdelt_geo`, `ovx`, `placebo_<k>` |
| `model` | `M1`, `M2`, `M3`, `GARCH_t`, `HAR`, `RW` |
| `target` | `rv` (variance forecast), `r` (return forecast), `p_high` (π_H), `p_exceed` (mixture P(\|r\| > thr)) |
| `forecast` | value |

- Plan §7's minimal format `date, model, target, forecast` is a subset; extra columns are additions. **[our call]**
- Actuals are **not stored**; `load_forecasts(with_actuals=True)` joins `rv`, `r`, and the exceedance indicator from the panel, so the panel stays the single source of truth. **[our call]**
- Which models produce which targets: MS models → all four; `GARCH_t` → `rv`; `HAR` → `rv`; `RW` → `r`.
- Size: ≈ 723 × 15 rows ≈ 11k rows ≈ 0.8 MB for `main`.

### 9.3 Standardization **[our call]**

TVTP regressors are z-scored with the mean and SD of **2005–2012** (the training window) in every fit, in-sample and OOS (OVX run: 2007-05 → 2012). Why: no lookahead in any window, coefficients comparable across origins and specifications, and LR tests are invariant to it. Stats are stored in `Design.std_stats` and written once to `results/frozen/std_stats.csv`.

### 9.4 `results/fits/params_<run>.csv` and `fitlog_<run>.csv`

- **params (wide):** `origin, model, <param_name_1>, …` in canonical labels, with readable names added by `make_model` (e.g. `p00_iran`, `p10_vix`, `mu_N`, `mu_H`, `phi_N`, `phi_H`, `sigma2_N`, `sigma2_H`). ≈ 2,200 rows.
- **fitlog:** `origin, model, settings_hash, loglike, converged, n_restarts_used, relabeled, refit_from_scratch, degenerate, nesting_ok, seconds, sigma2_N, sigma2_H, dur_N, dur_H, xi_H_origin, har_floored`.

### 9.5 Frozen-decision files

- `results/frozen/rq1_decision.json` — φ decision and evidence (notebook 03).
- `results/frozen/spec_freeze.json` — every setting frozen after validation + git hash (notebook 05).
- Notebooks 04–08 assert `config` matches these files.

---

## 10. Testing and verification

Small `pytest` suite for the risky parts only. Run `uv run pytest` (target < 1 min). Synthetic data, no network.

| Test file | What it checks |
|---|---|
| `test_weekly.py` | `week_ending`: Mon–Fri → same Friday, Sat/Sun → next Friday; Sat Sep 14, 2019 → Sep 20; Sat Feb 28, 2026 → Mar 6. Holiday Friday (e.g. Good Friday Apr 18, 2025): price = Thursday close, `rv` sums 4 returns, the next week's first return uses that Thursday. GDELT: backfile/daily split at Apr 1, 2013; weekly share = Σnum/Σden (not a mean of daily shares); a week with 0 Iran articles raises. AI-GPR: month m first appears in month m + 2. |
| `test_design.py` | Row t of `X` = r(t−1); row t of `Z` = standardized z(t−1); `x_next`/`z_next` use T; standardization uses only the training window (changing post-2012 data does not change the stats); no `.shift(` outside `design.py` and `weekly.py`. |
| `test_ms.py` | `canonicalize`: LL unchanged, σ²[1] > σ²[0], idempotent; for M1 and TVTP models. Constrained fit: fixed coef is exactly 0, LL(constrained) ≤ LL(unconstrained) + tol, LL(M3) ≥ LL(M2) − tol. β sign: simulate data where Iran raises P(N→H) and check `transition_coefs_deck` returns β > 0. Seeding: two `fit_ms` calls with the same seed and `search_reps > 0` return identical params. Degeneracy: a candidate with σ² below the floor is rejected when a non-degenerate one exists. LR df: 1 for constrained M3 vs M2, 2 for `iran_both_logits`. |
| `test_forecast.py` | `predicted_probs` = statsmodels `predicted_marginal_probabilities` (< 1e-10) for M1 and a TVTP model; `one_step` var_fc = Σπ(σ²+m²) on a hand-computed case; π sums to 1. |
| `test_evaluate.py` | QLIKE = 0 when f = RV and > 0 otherwise; DM wrapper (`lags=0`, HLN) matches a 10-line hand implementation with the same settings; CW on a toy nested case gives a positive stat when the big model is right; GW with instrument = constant equals the unconditional version; PT on perfectly signed forecasts rejects; MZ recovers (a, b) = (0, 1) on f = y + noise. |
| `test_oos.py` | On 3 toy origins: an interrupted-then-resumed run equals an uninterrupted run; a checkpoint with a different `settings_hash` is discarded, not resumed. |
| `test_budget.py` | `check_budget()` passes. |

Verification beyond tests (in notebooks):
- Notebook 03 repeats the forecaster check on real data.
- Notebook 05 checks forecast alignment, nesting of LLs per week and the resume identity.
- Notebook 08's placebo uses notebook 04's code path; the unshifted LR must equal notebook 04's.
- **Reproducibility check** (plan §12, Fri Dec 4): fresh clone → `uv sync` → `uv run pytest` → restart-and-run-all notebooks 02 → 08 with `FORCE` flags off (05 and 08 load committed outputs) → `git diff` shows no changes to committed tables. Optionally rerun 05 in full overnight and confirm identical forecasts (seeds make this exact).
- Before trusting AI-written module code, the builder reads it (the `verify-ai-code` habit).

---

## 11. Build order (one builder, in sequence)

**[decided]** One builder, notebooks in order. **[our call]** Module code is written just before the notebook that first needs it, with its tests. Critical path: **01 → 02 → 03 (freeze φ) → 05 timing → 05 validation → freeze → 05 full → 06 → 07**. Notebook 04 and the build of 06 fit into time the OOS loop runs unattended.

| Dates | Work (critical path in bold) | Runs unattended |
|---|---|---|
| **Fri Nov 20 – Sat Nov 21** | Scaffold (`uv init`, deps, `config.py`, `paths.py`). **BigQuery setup and D1 first** (riskiest: account access, backfile `DATEADDED`, coverage). `sources.py`; **notebook 01**. | Q1 (minutes) |
| **Sun Nov 22** | `weekly.py` + `test_weekly.py`; **notebook 02** (panel, data table, plots). | — |
| **Mon Nov 23** | `design.py`, `ms.py`, `forecast.py`, `insample.py` + tests on synthetic data; **notebook 03 → freeze φ**. | Bootstrap LR ≈ 20 min |
| **Tue Nov 24** | `benchmarks.py`, `oos.py` + `test_oos.py`; **notebook 05 `timing`** (10 validation origins) → set restarts; **start `validation` in the evening**. | Validation stage ≈ 0.5–1 h |
| **Wed Nov 25** | **Review validation → freeze spec** (`spec_freeze.json`); **start `full`** and leave it overnight. Meanwhile notebook 04 (RQ2, in-sample, cheap). | Full run ≈ 1–2 h (≈ 2–3 h if validation changed a setting and it reruns from 2012-12-28) |
| **Thu Nov 26** *(Thanksgiving)* | Buffer. | — |
| **Fri Nov 27 – Sat Nov 28** | `evaluate.py` + tests, `plots.py`; **notebook 06** on the real forecasts. | — |
| **Sun Nov 29** | **Notebook 07** (event table and figures, deck fact checks). **Minimum viable version complete.** | — |
| **Mon Nov 30 – Thu Dec 3** | Notebook 08 in plan order: placebo (in-sample) → AI-GPR → AR term → distribution → Iran in mean → OVX. Queue the OOS robustness runs (AR term first) to run overnight. | In-sample placebo ≈ 30 min; each OOS robustness run ≈ 0.5–1.5 h, overnight |
| **Thu Dec 3, evening** | **Analysis freeze:** all tables, figures and executed notebooks committed for the report. | — |
| **Fri Dec 4** | Reproducibility check (§10); hand-off to the report author. Write-up is out of scope. | Optional full rerun of 05 |

**If dates slip [our call]:**
- BigQuery still blocked on Sat Nov 21 evening → switch to a personal Google account. Still blocked on Sun Nov 22 → zip-streaming fallback (R1), about +1 day: everything shifts a day and all OOS robustness runs are dropped.
- φ not frozen by Tue Nov 24 → cut the bootstrap to 99 replicates.
- Spec not frozen by Fri Nov 27 → robustness becomes in-sample only (no OOS robustness runs) and the conditional GW test is dropped.
- Full run not finished by Sun Nov 29 → minimum-viable mode: notebooks 06 and 07 produce only the MVP items; notebook 08 is limited to the in-sample placebo and in-sample AI-GPR.
- Thu Dec 3 → nothing new is started; unfinished checks in 08 become stated limitations.

**Minimum viable version** (plan §12) → notebooks: M1–M3 with GDELT (03, 04, 05), GARCH-t and HAR (05), main test and 2026 holdout (05, 06), QLIKE, MZ, DM, Clark-West (06), cumulative-loss figure (06), event table with M2 vs M3 probabilities (07). Data notebooks 01–02 are prerequisites.

**Cut first** (plan §12): Student-t errors (08 §4), the fixed 10-year rolling window (not built unless time; `oos.run_oos` would take a `window` argument), OOS placebos (08 §1; keep in-sample), Brent implied vol. **[our call]** Next to cut if still short: OOS runs for AI-GPR / GDELT-geo / OVX (keep their in-sample versions), then Giacomini-White conditional (keep unconditional).

---

## 12. Left to the builder

- Exact AI-GPR URL; GPR date-column name; FRED header variant.
- Backfile classification after D1 (`GDELT_FEED_RULE`), and whether D2 is needed.
- Running the GDELT 2.0 splice (procedure in §6.4), only if D1 shows the 1.0 feed stops before Nov 6, 2026.
- `N_RESTARTS`, `RESTART_EVERY`, restart noise scale and optimizer tolerances, from the timing stage; `SIGMA2_RATIO_MIN`, `VAR_FLOOR_FRAC`, `MIN_DURATION` reviewed on validation. All frozen at validation.
- Readable parameter names, plot styling, figure layout, table formatting.
- Whether to parallelize robustness runs (`joblib`) if time is short.
- Handling of any week with missing GPR/VIX at the end of the sample (report, don't fill silently; ask the team if it affects the holdout).
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
