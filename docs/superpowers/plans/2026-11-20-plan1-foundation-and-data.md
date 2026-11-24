# Plan 1 — Foundation and Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Go from an empty repo to a committed, frozen `data/processed/weekly_panel.csv` (1,140 Friday weeks, 2005-01-07 → 2026-11-06), with hourly realized variance from Dukascopy and GDELT sentiment columns. The panel is built by notebooks 01–02 on a tested `brent` module.

**Architecture:** A `uv` project with a `src/brent/` package imported by Jupyter notebooks.
- Notebook 01 is the only one that touches the network: FRED, GPR and AI-GPR over HTTP; GDELT through one BigQuery aggregate query (now with conflict/cooperation and tone columns); Dukascopy hourly Brent bars, aggregated to days in memory.
- Each fetch writes a small trimmed CSV plus a manifest row to `data/raw/` (committed, except the intraday files if Dukascopy's terms forbid it).
- Notebook 02 turns those into the weekly panel. All week logic lives in `weekly.py`.

**Tech Stack:** Python 3.12, uv, pandas, numpy, statsmodels ≥ 0.15, requests, xlrd, matplotlib, pytest. Optional groups: `gdelt` (`pandas-gbq`, `google-cloud-bigquery`) and `intraday` (`dukascopy-python`; fallback `npx dukascopy-node`, which needs Node.js).

**Spec:** `docs/superpowers/specs/2026-11-20-brent-iran-implementation-design.md` (approved 2026-11-20, updated 2026-11-23: see its change log). Section references below (§N) point to the spec. The research plan is `Plans/Brent-Iran Regime Switching - Feasibility and Plan.md` ("plan §N").

**Scope:** Spec §5, §6 (incl. §6.8 Dukascopy), §7.1, §7.2, §8.1–8.3, §8.12, §9.1, and the `config`, `sources`, `weekly` and `budget` rows of §10. Plan 2 (notebooks 03–05: `design`, `ms`, `forecast`, `insample`, `benchmarks` incl. HAR-X/GARCH-X, `oos`) and Plan 3 (notebooks 06–08: `evaluate`, the rest of `plots`, robustness incl. E1) are written from the spec by other writers.

**Dates:** Mon Nov 23 → Wed Nov 25 (spec §11). BigQuery D1 and the Dukascopy access check come first on Nov 23.

**Do first today (Nov 23), riskiest first:** Task 1, Task 3's `gcp_project_id`/`gdelt_*` functions, and Task 5 Steps 1, 3 and 5a (BigQuery D1 + one week of Dukascopy bars as an access check); then return to Tasks 2 and 4. (Task order below stays 1 → 7 for reading; the order above is the order of work.)

## Global Constraints

- Python pinned: `.python-version` = `3.12`; `requires-python = ">=3.12,<3.13"`.
- Dependencies: `numpy pandas scipy "statsmodels>=0.15" arch matplotlib requests xlrd`. Dev group: `ipykernel pytest`. `gdelt` group: `pandas-gbq google-cloud-bigquery`. `intraday` group: `dukascopy-python`.
- `import brent` must work **without** the `gdelt` and `intraday` groups. Google imports happen only inside the GDELT functions, and `dukascopy_python` only inside `fetch_dukascopy_hourly`.
- Fetch window for daily sources: `2004-12-01` → `2026-11-06`. Brent alone starts at `1987-05-20`. AI-GPR months `2004-01` → `2026-10`. Dukascopy: first available bar (`INTRADAY_START = "2006-11-17"`, verified) → bars starting before `2026-11-06 22:00` UTC. Nothing after the freeze.
- No notebook after 01 touches the network. Fetchers read into memory; only trimmed CSVs are written. Never save raw `.xls`, `.zip`, GDELT rows or **hourly bars**. The `npx` fallback writes into a `tempfile.TemporaryDirectory()` outside the repo, read and deleted in the same call.
- Intraday time logic is UTC only.
  - Day d = bars **starting** in [d−1 22:00, d 22:00) UTC (`INTRADAY_DAY_END_UTC = 22`).
  - Hourly returns are 100·Δln(close) against the previous bar of the **whole series**, so overnight and weekend gap returns land in the first bar after the gap.
  - Roll returns are removed only by the rule in `config.INTRADAY_ROLL_RULE`, which is decided with the user.
- Size budget (`paths.check_budget()`): any file ≤ 10 MB; `data/` ≤ 5 MB (expected ≈ 2.4 MB); `data/` + `results/` (excluding `results/checkpoints/`) ≤ 20 MB; each notebook ≤ 5 MB; all notebooks ≤ 15 MB. `.venv`, `.git` and other dot-folders are excluded.
- The GCP project id comes **only** from the env var `BRENT_GCP_PROJECT` (or a git-ignored `.env` line). It never goes in `config.py` or git. Every real BigQuery query sets `maximumBytesBilled` = `C.GDELT_MAX_BYTES` (200 GB).
- Outside `design.py` (Plan 2) and `weekly.py`, no code calls `.shift(`. `sources.py` computes returns with `np.diff`/`.diff()`.
- Paths are resolved from the package location, never from the notebook's working directory.
- Every notebook:
  - Starts with `%load_ext autoreload`, `%autoreload 2`, `import brent`, `from brent import config as C`, `np.random.seed(C.SEED)`, and a `FORCE_*` flag block.
  - Ends with a checks cell (asserts) and `paths.check_budget()`.
  - Is committed **with outputs** after a restart-and-run-all, with inline figures at 100 dpi.
- Commits are authored by the repo's git user, with **no co-author trailer** (the user asked for commits "in my name").
- Data in `data/raw/` and `data/processed/` is committed. Exception: `brent_intraday_daily.csv` and `brent_intraday_removed.csv` are git-ignored if `INTRADAY_REDISTRIBUTE` is set to `False` in Task 5. `results/checkpoints/` is git-ignored.

## Review Focus

1. **A source ends before the freeze.** FRED runs about a week behind, and GPR may stop early. The fetch must not silently give a short series: `check_coverage` raises with the actual last date (test in Task 2).
2. **The weekend-gap return is dropped or counted twice.** The Sunday-evening reopening bar's return against Friday's last close must be counted exactly once, on Monday, in the new week. A winter Friday bar starting at 22:00 UTC must go to the *next* week (tests in Task 4 for days and Task 6 for weeks).
3. **A week with no Brent trading day, or no hourly return after `HOURLY_SAMPLE_START`.** `build_weekly_panel` must raise, not write a NaN price or a zero RV (tests in Task 6).
4. **GDELT rows on the wrong side of the Apr 1, 2013 switch.** For example, a backfile row with a 2013-04 date. It must be dropped and counted, not double-counted (test in Task 6).
5. **Missing VIX or GPR in a week when Brent trades.** The value stays NaN and is listed, never forward-filled (test in Task 6).

(The manifest-upsert test from the first version of this list stays in Task 2.)

---

### Task 1: Scaffold, `config.py`, `paths.py`

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.python-version`, `src/brent/__init__.py` (via `uv init`), `src/brent/config.py`, `src/brent/paths.py`, `tests/test_config.py`, `tests/test_budget.py`, `.gitkeep` files in `data/raw/`, `data/processed/`, `notebooks/`, `results/{forecasts,fits,tables,figures,frozen}/`
- Modify: `.gitignore`, `README.md`

**Interfaces:**
- Produces:
  - `brent.config`: every constant in spec §8.1, verbatim, incl. the intraday and extra-model block. `EVENTS` is fully written out (below), `GDELT_FEED_RULE = "dateadded"`, `INTRADAY_ROLL_RULE = None`, `INTRADAY_REDISTRIBUTE = None`.
  - `brent.paths`: `ROOT, RAW, PROCESSED, RESULTS, FORECASTS, FITS, TABLES, FIGURES, FROZEN, CHECKPOINTS: Path`
  - `save_table(df: pd.DataFrame, name: str) -> Path` (CSV to `TABLES/<name>.csv`)
  - `check_budget(raise_: bool = True, root: Path = ROOT) -> pd.DataFrame` (columns `rule, path, bytes, limit`; empty when within budget; raises `ValueError` listing the violations when `raise_`)
  - `git_hash() -> str`

- [ ] **Step 1: Scaffold the uv project**

  Run from the repo root:
  ```bash
  uv init --package --name brent --python 3.12 --no-readme --vcs none
  uv add numpy pandas scipy "statsmodels>=0.15" arch matplotlib requests xlrd
  uv add --dev ipykernel pytest
  uv add --group gdelt pandas-gbq google-cloud-bigquery
  uv add --group intraday dukascopy-python
  ```
  Then set `requires-python = ">=3.12,<3.13"` in `pyproject.toml` and run `uv sync`.
  Expected: `uv run python -c "import brent, sys; print(sys.version)"` prints 3.12.x.
  If `dukascopy-python` does not resolve on PyPI under that name, leave the `intraday` group empty, note it in the README, and rely on the `npx` fallback (Task 4).

- [ ] **Step 2: Update `.gitignore` and the folder skeleton**

  Append the block from spec §5.3 (`.pytest_cache/`, `*.egg-info/`, `build/`, `dist/`, `results/checkpoints/`, `*.zip`, `*.xls`, `*.xlsx`, `*.parquet`). Do not add the conditional intraday lines yet (Task 5 decides). Create the folders listed under Files, each with an empty `.gitkeep`.

- [ ] **Step 3: Write the failing tests**

  ```python
  # tests/test_config.py
  import pandas as pd
  from brent import config as C

  def n_fridays(a, b):
      return len(pd.date_range(a, b, freq="W-FRI"))

  def test_window_counts():
      assert n_fridays(*C.TRAIN) == 417
      assert n_fridays(*C.VALID) == 313
      assert n_fridays(*C.TEST) == 365
      assert n_fridays(*C.HOLDOUT) == 45
      assert n_fridays(C.SAMPLE_START, C.FREEZE_DATE) == 1140
      assert n_fridays(*C.STANDARDIZE_WINDOW_HOURLY) == 313

  def test_window_bounds_are_fridays():
      for a, b in (C.TRAIN, C.VALID, C.TEST, C.HOLDOUT, C.STANDARDIZE_WINDOW_HOURLY):
          assert pd.Timestamp(a).weekday() == 4 and pd.Timestamp(b).weekday() == 4
      assert pd.Timestamp(C.HOURLY_SAMPLE_START).weekday() == 4

  def test_events_frozen():
      assert len(C.EVENTS) == 9 and len(C.NON_IRAN_EVENTS) == 3
      assert all(len(e) == 4 for e in C.EVENTS)

  def test_intraday_and_extra_model_names():
      assert C.INTRADAY_DAY_END_UTC == 22 and C.MAIN_RV == "rv_hourly"
      assert C.RV_MEASURES == ("rv_hourly", "rv", "rv_gk")
      assert C.BENCHMARKS_MAIN == ("GARCH_t", "HAR", "RW", "HAR_h", "HARX_C", "HARX_CI", "GARCHX_C", "GARCHX_CI")
      assert C.IRAN_TONE_VARS == ("iran_conflict", "iran_coop")
  ```
  ```python
  # tests/test_budget.py
  import pytest
  from brent import paths

  def test_repo_within_budget():
      assert paths.check_budget(raise_=False).empty

  def test_oversized_file_flagged(tmp_path):
      (tmp_path / "data").mkdir()
      (tmp_path / "data" / "big.csv").write_bytes(b"0" * (11 * 2**20))
      with pytest.raises(ValueError, match="big.csv"):
          paths.check_budget(root=tmp_path)

  def test_dot_folders_ignored(tmp_path):
      (tmp_path / ".venv").mkdir()
      (tmp_path / ".venv" / "huge.bin").write_bytes(b"0" * (11 * 2**20))
      assert paths.check_budget(raise_=False, root=tmp_path).empty
  ```

- [ ] **Step 4: Run the tests and confirm they fail**

  Run: `uv run pytest -q`
  Expected: FAIL (`brent.config` / `brent.paths` have no such attributes).

- [ ] **Step 5: Implement `config.py` and `paths.py`**

  `config.py`: copy spec §8.1. Write out `EVENTS` with labels and confounders from plan §9:
  `hormuz_2011` 2011-12-27 · `jcpoa_2018` 2018-05-08 · `tankers_2019` 2019-06-13 · `abqaiq_2019` 2019-09-14 · `soleimani_2020` 2020-01-03 · `iran_israel_apr2024` 2024-04-13 · `iran_israel_oct2024` 2024-10-01 · `israel_iran_2025` 2025-06-13 · `hormuz_2026` 2026-02-28.
  - Keep `AIGPR_URL = "<fill in>"` and set `GDELT_FEED_RULE = "dateadded"` (alternative `"sourceurl"`). Both are decided in Task 5.
  - Keep `INTRADAY_ROLL_RULE = None` and `INTRADAY_REDISTRIBUTE = None`. Both are decided in Task 5 with the user.

  `paths.py`:
  - `ROOT = Path(__file__).resolve().parents[2]`.
  - `check_budget` walks `root`, skipping any path component that starts with `.` and skipping `results/checkpoints`. It applies the five limits from config (notebook limits apply to `notebooks/*.ipynb`).
  - `git_hash` runs `git rev-parse --short HEAD` in ROOT.

- [ ] **Step 6: Run the tests and confirm they pass**

  Run: `uv run pytest -q`
  Expected: all PASS.

- [ ] **Step 7: Add a Setup section to `README.md`**

  Cover:
  - `uv sync`; the builder alone also runs `uv sync --group gdelt --group intraday`.
  - The VS Code kernel at `.venv/bin/python`.
  - `BRENT_GCP_PROJECT` in a git-ignored `.env` (only needed for GDELT in notebook 01).
  - Node.js is needed only for the Dukascopy `npx` fallback.
  - Running `uv run pytest`.
  - Notebooks 02–08 run offline from committed files. The one exception is filled in at Task 5: if the intraday files are git-ignored, run notebook 01's Dukascopy cell first.

- [ ] **Step 8: Commit**

  ```bash
  git add pyproject.toml uv.lock .python-version .gitignore README.md src tests data notebooks results
  git commit -m "Scaffold uv project with config, paths and budget check"
  ```

---

### Task 2: HTTP fetchers, snapshots and manifest (`sources.py`, part 1)

**Files:**
- Create: `src/brent/sources.py`, `tests/test_sources.py`

**Interfaces:**
- Consumes: `paths.RAW`, `config.FETCH_START`, `config.FREEZE_DATE`
- Produces:
  - `parse_fred_csv(text: str, value_name: str, start: str, end: str) -> pd.DataFrame` (columns `date, <value_name>`)
  - `fetch_fred(series_id: str, start: str, end: str) -> pd.DataFrame` (columns `date, value`; renamed by the caller)
  - `parse_gpr_frame(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame` (columns `date, gprd`)
  - `fetch_gpr_daily(url: str, start: str, end: str) -> pd.DataFrame`
  - `parse_aigpr(df: pd.DataFrame, start_month: str, end_month: str) -> pd.DataFrame` (columns `month` as the first day of the month, `aigpr_iran`)
  - `fetch_aigpr_iran(url: str, start_month: str, end_month: str) -> pd.DataFrame`
  - `check_coverage(df: pd.DataFrame, col: str, first_by: str, last_by: str) -> None` (raises `ValueError` naming the actual first/last date)
  - `save_snapshot(df: pd.DataFrame, name: str, source: str, url: str) -> Path` (writes `RAW/<name>.csv`; upserts one row per file in `RAW/manifest.csv` with columns `file, source, url_or_table, fetched_at_utc, first_date, last_date, n_rows, sha256, bytes`)
  - `load_raw(name: str) -> pd.DataFrame` (parses `date`/`month`/`timestamp_utc` columns)
  - `GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"`

- [ ] **Step 1: Write the failing tests**

  ```python
  # tests/test_sources.py
  import pandas as pd, pytest
  from brent import sources as S

  def test_fred_parses_both_headers_and_drops_missing():
      for head in ("observation_date", "DATE"):
          txt = f"{head},DCOILBRENTEU\n2004-11-30,1\n2004-12-01,40.1\n2004-12-02,.\n2004-12-03,\n2026-11-09,70\n"
          df = S.parse_fred_csv(txt, "brent", "2004-12-01", "2026-11-06")
          assert list(df.columns) == ["date", "brent"]
          assert df["date"].tolist() == [pd.Timestamp("2004-12-01")]
          assert df["brent"].tolist() == [40.1]

  def test_gpr_parses_int_and_datetime_dates():
      a = pd.DataFrame({"DAY": [20041201, 20261106, 20261107], "GPRD": [100.0, 120.0, 130.0], "X": 1})
      b = a.assign(DAY=pd.to_datetime(a["DAY"].astype(str)))
      for df in (a, b):
          out = S.parse_gpr_frame(df, "2004-12-01", "2026-11-06")
          assert list(out.columns) == ["date", "gprd"] and len(out) == 2

  def test_aigpr_trims_months():
      df = pd.DataFrame({"month": ["2003-12", "2004-01", "2026-10", "2026-11"], "Iran_all": [1, 2, 3, 4]})
      out = S.parse_aigpr(df, "2004-01", "2026-10")
      assert out["aigpr_iran"].tolist() == [2, 3]

  def test_check_coverage_raises_on_short_series():
      df = pd.DataFrame({"date": pd.to_datetime(["2004-12-01", "2026-10-30"])})
      with pytest.raises(ValueError, match="2026-10-30"):
          S.check_coverage(df, "date", "2004-12-31", "2026-11-06")

  def test_snapshot_manifest_upserts(tmp_path, monkeypatch):
      monkeypatch.setattr(S.paths, "RAW", tmp_path)
      df = pd.DataFrame({"date": pd.to_datetime(["2005-01-03"]), "vix": [14.0]})
      S.save_snapshot(df, "vix_daily", "FRED", "url1")
      S.save_snapshot(df.assign(vix=15.0), "vix_daily", "FRED", "url2")
      man = pd.read_csv(tmp_path / "manifest.csv")
      assert len(man) == 1 and man.loc[0, "url_or_table"] == "url2" and len(man.loc[0, "sha256"]) == 64
      assert S.load_raw("vix_daily")["date"].dtype.kind == "M"
  ```
  (The AI-GPR column names are guesses. If the real file differs, `parse_aigpr` maps them, and this test's fixture is updated to the real names in Task 5.)

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_sources.py -q`
  Expected: FAIL (`module 'brent.sources' has no attribute ...`).

- [ ] **Step 3: Implement the functions listed under Interfaces in `src/brent/sources.py`**

  Fetchers use `requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=60)` and `resp.raise_for_status()`.
  - FRED URL: `https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={start}&coed={end}`.
  - The GPR workbook is read with `pd.read_excel(io.BytesIO(resp.content))` and never written.
  - `parse_*` functions accept a date column named `observation_date`/`DATE` (FRED), or `DAY`/`date` (GPR). They drop non-numeric values and trim to `[start, end]` inclusive.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest tests/test_sources.py -q`
  Expected: all PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/sources.py tests/test_sources.py
  git commit -m "Add FRED, GPR and AI-GPR fetchers with snapshot manifest"
  ```

---

### Task 3: GDELT BigQuery functions (`sources.py`, part 2)

**Files:**
- Modify: `src/brent/sources.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `config.GDELT_TABLE`, `config.GDELT_SWITCH_DATE`, `config.GDELT_MAX_BYTES`, `paths.ROOT`
- Produces:
  - `gcp_project_id() -> str | None` (env var `BRENT_GCP_PROJECT`, else a `BRENT_GCP_PROJECT=` line in `ROOT/.env`, else `None`)
  - `GDELT_SQL_D1: str` (spec §6.4 D1, with a `{table}` placeholder)
  - `render_q1(table: str, feed_rule: str, version: str = "v1", start: int = 20050101, end: int = 20261106) -> str`
    - `feed_rule` is `"dateadded"` (`IF(DATEADDED >= 20130401, 'daily', 'backfile')`) or `"sourceurl"` (`IF(SOURCEURL IS NULL, 'backfile', 'daily')`).
    - With `version="v2"`, the feed label is `'v2'` and the day key is `DIV(DATEADDED, 1000000)`.
    - Otherwise backfile rows are keyed by `SQLDATE` and daily rows by `DATEADDED`.
    - Output columns, exactly as spec §6.4 Q1: `feed, day_key, n_events, articles_all, articles_iran_actor, articles_iran_actor_geo, events_iran_actor, articles_iran_conflict, articles_iran_coop, tone_x_articles_iran, articles_iran_tone_n, goldstein_x_articles_iran, articles_iran_goldstein_n`.
  - `GDELT_DAILY_COLUMNS: tuple[str, ...]`: the output columns with `day_key` renamed to `date`.
  - `gdelt_credentials()`: `pydata_google_auth.get_user_credentials(["https://www.googleapis.com/auth/bigquery"])`
  - `gdelt_dry_run(sql: str, project_id: str, credentials) -> int`: `bigquery.Client(...).query(sql, job_config=QueryJobConfig(dry_run=True, use_query_cache=False)).total_bytes_processed`
  - `gdelt_query(sql: str, project_id: str, credentials, max_bytes: int) -> pd.DataFrame`: `pandas_gbq.read_gbq(sql, project_id=..., credentials=..., configuration={"query": {"maximumBytesBilled": str(max_bytes)}})`
  - `gdelt_daily_frame(q1: pd.DataFrame) -> pd.DataFrame`: renames `day_key` → `date` (parsed from `YYYYMMDD`); columns `GDELT_DAILY_COLUMNS`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sources.py`)

  ```python
  import subprocess, sys

  def test_import_without_optional_groups():
      code = ("import brent.sources, sys; "
              "assert not any(m.startswith(('google.cloud', 'dukascopy_python')) for m in sys.modules)")
      assert subprocess.run([sys.executable, "-c", code]).returncode == 0

  def test_render_q1_variants():
      q = S.render_q1("gdelt-bq.full.events", "dateadded")
      assert "IF(DATEADDED >= 20130401" in q and "BETWEEN 20050101 AND 20261106" in q
      assert "articles_iran_actor_geo" in q and "'IRN'" in q and "= 'IR'" in q
      assert "QuadClass IN (3, 4)" in q and "QuadClass IN (1, 2)" in q
      for col in ("articles_iran_conflict", "articles_iran_coop", "tone_x_articles_iran", "articles_iran_tone_n",
                  "goldstein_x_articles_iran", "articles_iran_goldstein_n"):
          assert f"AS {col}" in q
      assert "AvgTone IS NOT NULL" in q and "GoldsteinScale IS NOT NULL" in q
      assert "SOURCEURL IS NULL" in S.render_q1("gdelt-bq.full.events", "sourceurl")
      assert "DIV(DATEADDED, 1000000)" in S.render_q1("gdelt-bq.gdeltv2.events", "dateadded", version="v2")

  def test_project_id_from_env_and_dotenv(tmp_path, monkeypatch):
      monkeypatch.delenv("BRENT_GCP_PROJECT", raising=False)
      monkeypatch.setattr(S.paths, "ROOT", tmp_path)
      assert S.gcp_project_id() is None
      (tmp_path / ".env").write_text("BRENT_GCP_PROJECT=my-sandbox-123\n")
      assert S.gcp_project_id() == "my-sandbox-123"
      monkeypatch.setenv("BRENT_GCP_PROJECT", "from-env")
      assert S.gcp_project_id() == "from-env"

  def test_gdelt_daily_frame_parses_dates():
      q1 = pd.DataFrame({"feed": ["backfile"], "day_key": [20050103], "n_events": [5], "articles_all": [50],
                         "articles_iran_actor": [2], "articles_iran_actor_geo": [3], "events_iran_actor": [1],
                         "articles_iran_conflict": [1], "articles_iran_coop": [1], "tone_x_articles_iran": [-4.0],
                         "articles_iran_tone_n": [2], "goldstein_x_articles_iran": [1.0], "articles_iran_goldstein_n": [2]})
      out = S.gdelt_daily_frame(q1)
      assert out.loc[0, "date"] == pd.Timestamp("2005-01-03") and "day_key" not in out
      assert tuple(out.columns) == S.GDELT_DAILY_COLUMNS
  ```

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_sources.py -q`
  Expected: the 4 new tests FAIL.

- [ ] **Step 3: Implement the functions listed under Interfaces**

  Import `google.cloud.bigquery`, `pandas_gbq` and `pydata_google_auth` inside the function bodies only. The Q1 text follows spec §6.4 exactly.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest -q`
  Expected: all PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/sources.py tests/test_sources.py
  git commit -m "Add GDELT BigQuery query templates with sentiment columns"
  ```

---

### Task 4: Dukascopy hourly → daily (`sources.py`, part 3)

**Files:**
- Modify: `src/brent/sources.py`
- Create: `tests/test_sources_intraday.py`

**Interfaces:**
- Consumes: `config.DUKASCOPY_INSTRUMENT`, `INTRADAY_START`, `INTRADAY_END_UTC`, `INTRADAY_PRICE_SIDE`, `INTRADAY_DAY_END_UTC`, `INTRADAY_ROLL_SCAN`
- Produces (spec §6.8, §8.3). Hourly frame = columns `timestamp_utc` (tz-aware UTC, bar **start**), `open, high, low, close, volume`. It is never written to disk.
  - `fetch_dukascopy_hourly(instrument: str = C.DUKASCOPY_INSTRUMENT, start: str = C.INTRADAY_START, end: str = "2026-11-07", side: str = C.INTRADAY_PRICE_SIDE, method: str = "python") -> pd.DataFrame`
    - `"python"`: `dukascopy_python`, imported inside the function, called in yearly chunks with up to 3 retries per chunk.
    - `"node"`: `npx dukascopy-node -i brentcmdusd -from <start> -to <end> -t h1 -p bid -f csv` into a `tempfile.TemporaryDirectory()`, read, then deleted.
    - Either way the result is renamed to the hourly schema. The exact library call / CLI flags are confirmed in Task 5 Step 5a; the body may change then, the signature does not.
  - `normalize_hourly(raw: pd.DataFrame, end_utc: str = C.INTRADAY_END_UTC) -> pd.DataFrame`: naive timestamps are taken as UTC, tz-aware ones converted with `tz_convert("UTC")`; then sort, drop duplicate timestamps (keep first), drop `volume == 0` bars, drop bars with `timestamp_utc >= end_utc`.
  - `intraday_day(ts: pd.Series | pd.DatetimeIndex, day_end_utc: int = C.INTRADAY_DAY_END_UTC) -> pd.DatetimeIndex`: naive dates, `(ts + (24 − day_end_utc) h).normalize()` with the tz removed.
  - `hourly_returns(hourly: pd.DataFrame) -> pd.Series`: 100·diff(ln close) across all gaps; index `timestamp_utc`; first value NaN.
  - `roll_candidates(hourly: pd.DataFrame, k: float = 8.0, window: int = 120) -> pd.DataFrame`: columns `timestamp_utc, ret, med_abs, ratio, gap_hours`.
    - `med_abs` = median |ret| of the previous `window` returns, NaN until `window` returns exist.
    - `ratio` = |ret| / `med_abs`; `gap_hours` = hours since the previous bar.
    - Rows are those with ratio > k, sorted by ratio, descending.
  - `roll_mask(hourly: pd.DataFrame, rule: dict) -> pd.Series` (bool per bar, True = return removed). Rule kinds:
    - `{"kind": "none"}`;
    - `{"kind": "threshold", "k": float, "window": int}`, same ratio as `roll_candidates`;
    - `{"kind": "timestamps", "timestamps": [ISO strings]}`, parsed as UTC.

    Any other kind raises `ValueError`.
  - `intraday_daily(hourly: pd.DataFrame, rule: dict | None, day_end_utc: int = C.INTRADAY_DAY_END_UTC) -> tuple[pd.DataFrame, pd.DataFrame]`:
    - Daily frame, columns `date, rv_intraday, n_hours, open, high, low, close`, one row per day that has bars:
      - `rv_intraday` = Σ ret² over retained returns;
      - `n_hours` = count of retained returns;
      - OHLC = first open, max high, min low, last close of the day's bars.
    - Removed frame, columns `timestamp_utc, date, ret, ratio, reason` (`reason = "roll"`).
    - `rule is None` raises `ValueError("INTRADAY_ROLL_RULE not set")`.
  - `fetch_dukascopy_daily(rule: dict, **fetch_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]`: `fetch_dukascopy_hourly` → `normalize_hourly` → `intraday_daily`.

- [ ] **Step 1: Write the failing tests**

  ```python
  # tests/test_sources_intraday.py
  import numpy as np, pandas as pd, pytest
  from brent import sources as S

  NONE = {"kind": "none"}

  def bars(ts, close, volume=1.0, tz="UTC"):
      idx = pd.DatetimeIndex(pd.to_datetime(ts))
      idx = idx.tz_localize(tz) if tz else idx
      c = np.asarray(close, float)
      return pd.DataFrame({"timestamp_utc": idx, "open": c, "high": c, "low": c, "close": c, "volume": volume})

  def test_weekend_gap_lands_on_monday_once():
      h = S.normalize_hourly(bars(["2025-01-10 20:00", "2025-01-10 21:00", "2025-01-12 23:00", "2025-01-13 00:00"],
                                  [100, 101, 103, 102]))
      daily, removed = S.intraday_daily(h, NONE)
      d = daily.set_index("date")
      fri, mon = pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-13")
      assert list(d.index) == [fri, mon]                          # no Saturday or Sunday row
      assert d.loc[fri, "n_hours"] == 1 and d.loc[mon, "n_hours"] == 2
      gap, last = 100 * np.log(103 / 101), 100 * np.log(102 / 103)
      assert d.loc[fri, "rv_intraday"] == pytest.approx((100 * np.log(101 / 100)) ** 2)
      assert d.loc[mon, "rv_intraday"] == pytest.approx(gap**2 + last**2)
      assert (d.loc[mon, "open"], d.loc[mon, "high"], d.loc[mon, "low"], d.loc[mon, "close"]) == (103, 103, 102, 102)
      assert S.hourly_returns(h).sum() == pytest.approx(100 * np.log(102 / 100))
      assert removed.empty

  def test_day_boundary_is_22_utc_in_summer_and_winter():
      ts = pd.DatetimeIndex(["2025-07-01 22:00", "2025-07-01 23:00",
                             "2025-01-14 21:00", "2025-01-14 22:00"]).tz_localize("Europe/London")
      h = S.normalize_hourly(bars(ts, [1, 1, 1, 1], tz=None))
      assert str(h["timestamp_utc"].dt.tz) == "UTC"
      assert list(S.intraday_day(h["timestamp_utc"])) == [
          pd.Timestamp("2025-01-14"), pd.Timestamp("2025-01-15"),    # 21:00 UTC stays, 22:00 UTC moves
          pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-02")]    # 22:00 BST = 21:00 UTC stays, 23:00 BST moves

  def test_naive_timestamps_are_utc():
      h = S.normalize_hourly(bars(["2025-01-14 22:00"], [1.0], tz=None))
      assert h["timestamp_utc"].iloc[0] == pd.Timestamp("2025-01-14 22:00", tz="UTC")

  def test_friday_22utc_bar_goes_to_saturday():
      h = S.normalize_hourly(bars(["2025-01-10 21:00", "2025-01-10 22:00"], [100, 101]))
      daily, _ = S.intraday_daily(h, NONE)
      assert daily["date"].tolist()[-1] == pd.Timestamp("2025-01-11")

  def test_roll_jump_found_and_removed():
      ts = pd.date_range("2025-03-03 00:00", periods=200, freq="h", tz="UTC")
      r = np.where(np.arange(199) % 2 == 0, 0.001, -0.001)
      r[150] = 0.05                                               # the return of bar ts[151]
      close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
      h = S.normalize_hourly(pd.DataFrame({"timestamp_utc": ts, "open": close, "high": close,
                                           "low": close, "close": close, "volume": 1.0}))
      cand = S.roll_candidates(h, k=8.0, window=120)
      assert len(cand) == 1 and cand["timestamp_utc"].iloc[0] == ts[151]
      assert cand["ratio"].iloc[0] == pytest.approx(50.0)
      d0, rm0 = S.intraday_daily(h, NONE)
      assert rm0.empty
      for rule in ({"kind": "threshold", "k": 8.0, "window": 120},
                   {"kind": "timestamps", "timestamps": [ts[151].isoformat()]}):
          d1, rm1 = S.intraday_daily(h, rule)
          assert len(rm1) == 1 and rm1["timestamp_utc"].iloc[0] == ts[151] and rm1["reason"].iloc[0] == "roll"
          assert d0["n_hours"].sum() - d1["n_hours"].sum() == 1
          assert d0["rv_intraday"].sum() - d1["rv_intraday"].sum() == pytest.approx(25.0)
      with pytest.raises(ValueError):
          S.intraday_daily(h, None)

  def test_zero_volume_and_freeze_cut():
      h = S.normalize_hourly(bars(["2026-11-06 20:00", "2026-11-06 21:00", "2026-11-06 22:00"],
                                  [100, 100, 101], volume=[1, 0, 1]))
      assert h["timestamp_utc"].tolist() == [pd.Timestamp("2026-11-06 20:00", tz="UTC")]
  ```

- [ ] **Step 2: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_sources_intraday.py -q`
  Expected: FAIL (`module 'brent.sources' has no attribute 'normalize_hourly'`).

- [ ] **Step 3: Implement the functions listed under Interfaces in `src/brent/sources.py`**

  - Returns use `np.diff(np.log(close))` or `.diff()`, never `.shift(`.
  - The daily aggregation is a `groupby(intraday_day(...))` over bars, where the removed returns are set to NaN before squaring.

- [ ] **Step 4: Run the tests and confirm they pass**

  Run: `uv run pytest -q`
  Expected: all PASS (incl. `test_import_without_optional_groups`).

- [ ] **Step 5: Commit**

  ```bash
  git add src/brent/sources.py tests/test_sources_intraday.py
  git commit -m "Add Dukascopy hourly-to-daily aggregation with roll-jump rule"
  ```

---

### Task 5: Notebook 01 — fetch and freeze all raw data

**Order of work:** Steps 1, 3 and 5a first (Nov 23), then the rest.

The user has to take part in this task: a Google login in the browser, the sandbox project id, a decision based on what D1 returns, and the **roll rule**. Run it in the main session, not in a subagent.

**Files:**
- Create: `notebooks/01_fetch_data.ipynb`, `data/raw/{brent_daily,vix_daily,ovx_daily,gpr_daily,aigpr_iran_monthly,gdelt_daily,brent_intraday_daily,brent_intraday_removed}.csv`, `data/raw/manifest.csv`, `.env` (git-ignored, user-created)
- Modify:
  - `src/brent/config.py`: `AIGPR_URL`, `GDELT_FEED_RULE`, `GDELT_SPLICE_WEEK` (only if needed), `INTRADAY_REDISTRIBUTE`, `INTRADAY_ROLL_RULE`, and `INTRADAY_START`/`HOURLY_SAMPLE_START` only if the first bar differs.
  - `.gitignore`, only if `INTRADAY_REDISTRIBUTE` is `False`.
  - `README.md`: the intraday rerun note.
  - `src/brent/sources.py` and its tests, only if real column names, the Dukascopy call or the CLI flags differ from the guesses.

**Interfaces:**
- Consumes: everything from Tasks 2–4
- Produces: the raw CSVs with these columns:
  - `brent_daily`: `date, brent` (1987-05-20 → ≤ 2026-11-06)
  - `vix_daily`: `date, vix`
  - `ovx_daily`: `date, ovx` (from 2007-05)
  - `gpr_daily`: `date, gprd`
  - `aigpr_iran_monthly`: `month, aigpr_iran`
  - `gdelt_daily`: `GDELT_DAILY_COLUMNS`
  - `brent_intraday_daily`: `date, rv_intraday, n_hours, open, high, low, close`
  - `brent_intraday_removed`: `timestamp_utc, date, ret, ratio, reason` (may have 0 rows)

- [ ] **Step 1: User setup for BigQuery and the optional groups** (spec §6.4 steps 1–4)

  The user opens `https://console.cloud.google.com/bigquery` (UCSD account first; a personal account if Cloud is blocked), notes the sandbox project id, and writes `BRENT_GCP_PROJECT=<id>` to `.env`. Then run `uv sync --group gdelt --group intraday`.
  Expected: `uv run python -c "from brent.sources import gcp_project_id; print(gcp_project_id())"` prints the id.

- [ ] **Step 2: Build the HTTP cells**

  - First cell: the standard notebook header, plus `FORCE_FETCH = {k: False for k in ["brent","vix","ovx","gpr","aigpr","gdelt","intraday"]}`.
  - One cell per source with skip logic: `if (paths.RAW / f"{name}.csv").exists() and not FORCE_FETCH[key]: skip`.
  - Fetch → `check_coverage` → `save_snapshot`, with these windows:
    - Brent: `1987-05-20 → 2026-11-06`; `check_coverage(first_by="1987-05-20", last_by="2026-11-06")`
    - VIX: `FETCH_START → FREEZE_DATE`
    - OVX: `2007-05-01 → FREEZE_DATE`
    - GPR: `GPR_URL`
    - AI-GPR: `2004-01 → 2026-10`
  - Before the AI-GPR cell, find the monthly country file on matteoiacoviello.com's AI-GPR page. Put its URL in `config.AIGPR_URL`. If its column names differ from the test fixture, fix `parse_aigpr` and its test (TDD), then rerun `uv run pytest`.
  - If GPR's last date is before 2026-11-06, record it in a markdown cell and tell the user. Don't fill the gap.

- [ ] **Step 3: Run D1 and decide the feed rule**

  Cells:
  1. `creds = gdelt_credentials()`
  2. `gdelt_dry_run(GDELT_SQL_D1.format(table=C.GDELT_TABLE), pid, creds)`. Print GB and assert ≤ `C.GDELT_MAX_BYTES`.
  3. `gdelt_query(...)`. The result is kept in memory only.

  Read it per spec §6.4 and write a markdown cell with the findings:
  - (a) Does `added_yyyymm` continue through `202611`?
  - (b) What do backfile rows (event years ≤ 2012) carry in `DATEADDED`?
  - (c) Do backfile rows have `SOURCEURL` null?

  Decision:
  - If (b) is a pre-2013 date per row, keep `GDELT_FEED_RULE = "dateadded"`.
  - If (b) is a 2013 bulk-load date and (c) cleanly separates the feeds, set `"sourceurl"`.
  - If neither works, stop and ask the user.

  Commit the `config.py` change.

  Notebook 01 is committed with outputs, so: never print the project id (print only "project id set"). Before committing, grep the executed notebook for the project id (read from `BRENT_GCP_PROJECT` / `.env`) and abort if found (Step 7).

- [ ] **Step 4: Run Q1 and save the GDELT daily aggregate**

  1. `sql = render_q1(C.GDELT_TABLE, C.GDELT_FEED_RULE)`.
  2. Dry run, print GB, assert ≤ cap. The sentiment columns add roughly 5–15 GB; if the dry run fails on a column name (`QuadClass`, `AvgTone`, `GoldsteinScale`), fix `render_q1` and its test.
  3. `gdelt_query` → `gdelt_daily_frame` → `save_snapshot(df, "gdelt_daily", "GDELT 1.0 BigQuery", C.GDELT_TABLE)`.

  Expected: about 8,050 rows and under 1 MB.

  If (a) showed the 1.0 feed stops before Nov 6, 2026, do the GDELT 2.0 splice from spec §6.4 (incl. the per-column sentiment overlap check):
  - `render_q1("gdelt-bq.gdeltv2.events", "dateadded", version="v2")`;
  - the overlap correlation ≥ 0.8 for `iran_gdelt` (stop and ask the user if lower);
  - `config.GDELT_SPLICE_WEEK` and a `gdelt_source` column.

  Otherwise skip the splice.

- [ ] **Step 5: Dukascopy: access, terms, roll rule, daily file** (spec §6.8)

  The whole block sits under one skip condition: `brent_intraday_daily.csv` exists and `FORCE_FETCH["intraday"]` is False.
  - **5a Access.** Fetch one week (`start="2025-01-06", end="2025-01-13"`) with `method="python"`; if that fails, `method="node"`. Print the columns, the first rows and the dtype of the timestamps. Confirm that the timestamps are bar **start** times in UTC: the first bar after the weekend should start Sunday 22:00 or 23:00 UTC, or Monday early UTC. Fix `fetch_dukascopy_hourly` if the call or columns differ (rerun Task 4's tests), and record the working method in a markdown cell.
  - **5b Terms.** Open Dukascopy's terms of use for historical data. Write the URL, the date read and the finding in a markdown cell, and set `config.INTRADAY_REDISTRIBUTE`.
    - If it is `False`, append the two intraday paths to `.gitignore` (spec §5.3), and add the README note "run notebook 01's Dukascopy cell (needs `uv sync --group intraday`) before notebook 02".
    - If the terms are unclear, ask the user and choose `False` until they answer.
  - **5c Full fetch** (in memory): `raw = fetch_dukascopy_hourly(method=...)`; `hourly = normalize_hourly(raw)`. Print the first bar's timestamp and the number of bars (≈ 110,000 expected).
    - If the first bar's date ≠ `2006-11-17`, set `config.INTRADAY_START` to it and set `config.HOURLY_SAMPLE_START` by this rule, using `pd.Timestamp` arithmetic (`weekly.py` does not exist yet): let `fri = first date + ((4 − weekday) mod 7)` days (the Friday ending its Sat–Fri week). If the first bar falls on or before the Saturday that starts that week (`first date <= fri − 6 days`, so the week is fully covered), use `fri`; otherwise use `fri + 7 days` (the next Friday). A flat `+ 7` would skip a fully covered week when the first date is Mon–Thu. The real first bar is Fri Nov 17, 2006, so `2006-11-24` stays.
  - **5d Roll rule (stop and ask the user).** Run `roll_candidates(hourly, **C.INTRADAY_ROLL_SCAN)` and show:
    - the table;
    - the candidates' count by day of month;
    - `gap_hours`;
    - a plot of hourly close with the candidates marked.

    The rule is chosen from the 2006–2018 candidates only (spec §6.8). The user picks one rule (`none`, `threshold` with k and window, or `timestamps`).
    - Then list the 2019–2026 removals it implies, and show any removal within ±1 week of a `C.EVENTS` date. A genuine news jump must not be removed as a roll.
    - Once the user confirms, write the rule into `config.INTRADAY_ROLL_RULE` and commit `config.py`.
  - **5e Aggregate and save.** `daily, removed = intraday_daily(hourly, C.INTRADAY_ROLL_RULE)`, then:
    - `save_snapshot(daily, "brent_intraday_daily", "Dukascopy BRENT.CMD/USD h1 bid", <"dukascopy-python <version>" or "npx dukascopy-node">)`;
    - `save_snapshot(removed, "brent_intraday_removed", ...)`;
    - `del raw, hourly`.

- [ ] **Step 6: Checks cell** (spec §7.1, §6.8). Each line is an `assert`, printing the offending rows on failure:
  - No duplicate dates per file; GDELT has no duplicate `(feed, date)`.
  - Brent's first date ≤ 1987-05-20, and it has a close on or before 2004-12-31 (within 4 days) and on or before 2026-11-06.
  - VIX, GPR and GDELT cover `FETCH_START`/`2005-01-01` → `2026-11-06`; OVX from ≤ 2007-05-11.
  - GDELT: after 2013-04-01, list (don't fill) any gap of more than 3 consecutive calendar days.
  - GDELT: count columns ≥ 0, and `articles_iran_conflict + articles_iran_coop ≤ articles_iran_actor` on every row.
  - Intraday:
    - first `date` ≤ 2006-11-20 and last `date` == 2026-11-06;
    - `n_hours` ≤ 24 and `rv_intraday` ≥ 0;
    - no Sunday rows; row counts by weekday are printed;
    - days with `n_hours < 12` are listed; the number of removed returns is printed.
  - Bytes scanned printed for D1 and Q1.
  - `paths.check_budget()`.

- [ ] **Step 7: Restart and run all, then commit**

  Expected: every source cell prints "skip" (files exist) and the checks pass.

  Project-id guard (the notebook is committed with outputs): read the id from the env var / `.env`, then `grep -c "$BRENT_GCP_PROJECT" notebooks/01_fetch_data.ipynb` must be 0 (also check `data/raw/manifest.csv`). If found, abort, clear the offending output and rerun. Never echo the id into the terminal log either.
  ```bash
  git add notebooks/01_fetch_data.ipynb data/raw src/brent/config.py .gitignore README.md
  git commit -m "Fetch and freeze raw data incl. Dukascopy hourly RV (notebook 01)"
  ```
  (If `INTRADAY_REDISTRIBUTE` is `False`, `git add data/raw` skips the two ignored files; check with `git status` that they are not staged.)

---

### Task 6: Weekly construction (`weekly.py`)

**Files:**
- Create: `src/brent/weekly.py`, `tests/test_weekly.py`, `tests/conftest.py`

**GDELT 2.0 splice (conditional).** Added only if D1 (Task 5 Step 3, question (a)) shows the 1.0 feed stops before Nov 6, 2026; otherwise nothing below is built. Then `gdelt_weekly` gets a `splice: str | None = None` argument (the first Friday week taken from the `v2` rows, `C.GDELT_SPLICE_WEEK`) and a `gdelt_source` column (`"v1"`/`"v2"`). The level shift is the mean difference over the overlap weeks, `mean(log share v1) − mean(log share v2)`, per share column, and is added to the v2 log shares. Write the test first, on a fixture with `v1` rows to the end of week W−1, `v2` rows from week W−4 on (overlap), and a known v1/v2 ratio:
  - weeks before the splice week are identical to the no-splice output;
  - weeks from the splice week on equal the v2 log share + shift (assert to 1e-12);
  - `gdelt_source` is `"v1"` before the splice week and `"v2"` from it on;
  - with `splice=None` the column is absent and the output equals the old one.

**Interfaces:**
- Consumes: `config` (`SAMPLE_START`, `FREEZE_DATE`, `GDELT_SWITCH_DATE`, `AIGPR_PUBLICATION_LAG_MONTHS`, `EVENTS`, `HOURLY_SAMPLE_START`), raw frames shaped as in Task 5
- Produces:
  - `week_ending(d: pd.Timestamp | pd.DatetimeIndex | pd.Series)`, returning the same type: `d + ((4 − weekday) mod 7)` days
  - `fridays(start: str = C.SAMPLE_START, end: str = C.FREEZE_DATE) -> pd.DatetimeIndex`
  - `weekly_brent(brent: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame`
    - Index `date`; columns `price, r, rv, n_days`.
    - `r` = 100·ln(price/previous week's price).
    - `rv` = Σ(100·Δln p)² over the week's trading days, where the first daily return uses the previous trading day's close.
  - `weekly_last(daily: pd.DataFrame, col: str, weeks) -> pd.Series`: the last observation within Sat–Fri, NaN if none
  - `weekly_mean(daily: pd.DataFrame, col: str, weeks) -> pd.Series`: the mean of observations within Sat–Fri, NaN if none
  - `gdelt_weekly(gd: pd.DataFrame, weeks, switch: str = C.GDELT_SWITCH_DATE, splice: str | None = None) -> tuple[pd.DataFrame, int]` (`splice` only if the 2.0 splice is needed; see the box at the top of this task)
    - Columns `iran_share, iran_gdelt, iran_gdelt_geo, gdelt_days, iran_conflict, iran_coop, iran_tone, iran_goldstein`.
    - The int is the number of daily rows dropped for being on the wrong side of the switch: `backfile` rows kept only if date < switch, `daily` (or `v2`) rows only if date ≥ switch.
    - Shares are Σ numerator / Σ denominator per week (`articles_all` for all shares). `iran_conflict`, `iran_coop` are logs of shares. `iran_tone` = Σ `tone_x_articles_iran` / Σ `articles_iran_tone_n` and `iran_goldstein` likewise, both levels.
  - `aigpr_weekly(monthly: pd.DataFrame, weeks, lag_months: int = C.AIGPR_PUBLICATION_LAG_MONTHS) -> pd.Series`: month m is usable from the first Friday on or after the 1st of month m + 1 + `lag_months`
  - `intraday_weekly(daily: pd.DataFrame, weeks, first_week: str = C.HOURLY_SAMPLE_START) -> pd.DataFrame`
    - Index `date`; columns `rv_hourly, log_rv_hourly, n_hours_week, rv_gk`.
    - Days map to weeks with `week_ending` (a Saturday row goes to the next Friday). `rv_hourly` = Σ `rv_intraday`; `n_hours_week` = Σ `n_hours`.
    - `rv_gk` = Σ over days of 100²·[(ln O_d − ln C_{d−1})² + 0.5(ln H_d − ln L_d)² − (2 ln 2 − 1)(ln C_d − ln O_d)²], with C_{d−1} the previous daily row's close (spec §8.12).
    - The first daily row has no C_{d−1}, so its `rv_gk` term is NaN, and a week containing it has NaN `rv_gk`. That week is always before `first_week` in real data.
    - All four are NaN for weeks before `first_week`. Raises `ValueError` listing the weeks if any week ≥ `first_week` has `n_hours_week == 0` or no row.
  - `build_weekly_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame`
    - Keys `brent, vix, ovx, gpr, aigpr, gdelt, intraday`.
    - Columns in this order (spec §9.1): `date, price, r, rv, log_rv, n_days, iran_share, iran_gdelt, iran_gdelt_geo, gdelt_days, gpr, vix, ovx, aigpr_iran, iran_conflict, iran_coop, iran_tone, iran_goldstein, rv_hourly, log_rv_hourly, n_hours_week, rv_gk`. Logs for `iran_gdelt, iran_gdelt_geo, iran_conflict, iran_coop, gpr, vix, ovx, aigpr_iran`. Add `gdelt_source` last, only if it is present in the raw GDELT frame.
    - Raises `ValueError` listing the weeks if any week has `n_days == 0`, zero Iran articles (main, geo, conflict or cooperation), zero `articles_iran_tone_n` or `articles_iran_goldstein_n`, an AI-GPR value ≤ 0, or (from `intraday_weekly`) no hourly return on or after `HOURLY_SAMPLE_START`.
    - Never fills NaN.

- [ ] **Step 1: Write the synthetic fixture** in `tests/conftest.py`

  `synthetic_raw()` returns a dict with:
  - Brent and VIX on business days 2004-12-01 → 2026-11-06, with positive random-walk prices, seed 0.
  - OVX from 2007-05-01.
  - GPR on every calendar day.
  - AI-GPR monthly 2004-01 → 2026-10, positive.
  - GDELT on every calendar day: `backfile` rows before 2013-04-01 and `daily` rows from then on. Values: `articles_all = 1000`, `articles_iran_actor = 10`, `articles_iran_actor_geo = 12`, `articles_iran_conflict = 4`, `articles_iran_coop = 5`, `tone_x_articles_iran = -20.0`, `articles_iran_tone_n = 10`, `goldstein_x_articles_iran = 10.0`, `articles_iran_goldstein_n = 10`.
  - Intraday daily on business days 2006-11-17 → 2026-11-06: `rv_intraday` positive (seed 1), `n_hours = 22`, and OHLC from a positive random walk with `high ≥ max(open, close)` and `low ≤ min(open, close)`.

- [ ] **Step 2: Write the failing tests** in `tests/test_weekly.py`

  ```python
  import numpy as np, pandas as pd, pytest
  from brent import weekly as W, config as C

  def T(s): return pd.Timestamp(s)

  SENT = {"articles_iran_conflict": 1, "articles_iran_coop": 1, "tone_x_articles_iran": -1.0,
          "articles_iran_tone_n": 1, "goldstein_x_articles_iran": 1.0, "articles_iran_goldstein_n": 1}

  def test_week_ending_and_weekend_rule():
      assert W.week_ending(T("2019-09-14")) == T("2019-09-20")   # Abqaiq, Saturday
      assert W.week_ending(T("2026-02-28")) == T("2026-03-06")   # Saturday
      assert W.week_ending(T("2026-03-01")) == T("2026-03-06")   # Sunday
      assert W.week_ending(T("2026-03-06")) == T("2026-03-06")   # Friday stays

  def test_holiday_friday_week():
      days = pd.to_datetime(["2025-04-11", "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-21"])
      brent = pd.DataFrame({"date": days, "brent": [60.0, 61, 62, 61, 63, 64]})
      out = W.weekly_brent(brent, pd.DatetimeIndex([T("2025-04-18"), T("2025-04-25")]))
      wk = out.loc[T("2025-04-18")]
      lr = 100 * np.diff(np.log([60.0, 61, 62, 61, 63]))
      assert wk["price"] == 63 and wk["n_days"] == 4
      assert wk["r"] == pytest.approx(lr.sum()) and wk["rv"] == pytest.approx((lr**2).sum())
      assert out.loc[T("2025-04-25"), "r"] == pytest.approx(100 * np.log(64 / 63))

  def test_week_without_trading_raises(synthetic_raw):
      raw = dict(synthetic_raw)
      b = raw["brent"]
      raw["brent"] = b[~b["date"].between("2010-03-06", "2010-03-12")]
      with pytest.raises(ValueError, match="2010-03-12"):
          W.build_weekly_panel(raw)

  def test_gdelt_share_split_and_drop():
      gd = pd.DataFrame({
          "feed": ["backfile", "backfile", "daily", "daily"],
          "date": pd.to_datetime(["2013-03-29", "2013-04-02", "2013-04-01", "2013-04-02"]),
          "n_events": 1, "articles_all": [100, 999, 300, 100],
          "articles_iran_actor": [1, 999, 3, 4], "articles_iran_actor_geo": [2, 999, 4, 5], "events_iran_actor": 1,
          **SENT})
      out, dropped = W.gdelt_weekly(gd, pd.DatetimeIndex([T("2013-03-29"), T("2013-04-05")]))
      assert dropped == 1
      assert out.loc[T("2013-04-05"), "iran_share"] == pytest.approx(7 / 400)   # sum/sum, not mean of shares
      assert out.loc[T("2013-04-05"), "gdelt_days"] == 2

  def test_gdelt_conflict_coop_tone():
      gd = pd.DataFrame({
          "feed": "daily", "date": pd.to_datetime(["2015-06-08", "2015-06-09"]),
          "n_events": 1, "articles_all": [100, 300], "articles_iran_actor": [10, 20],
          "articles_iran_actor_geo": [12, 22], "events_iran_actor": 1,
          "articles_iran_conflict": [3, 5], "articles_iran_coop": [6, 12],
          "tone_x_articles_iran": [-20.0, 10.0], "articles_iran_tone_n": [4, 6],
          "goldstein_x_articles_iran": [5.0, -15.0], "articles_iran_goldstein_n": [10, 10]})
      out, _ = W.gdelt_weekly(gd, pd.DatetimeIndex([T("2015-06-12")]))
      wk = out.loc[T("2015-06-12")]
      assert wk["iran_conflict"] == pytest.approx(np.log(8 / 400))
      assert wk["iran_coop"] == pytest.approx(np.log(18 / 400))
      assert wk["iran_tone"] == pytest.approx(-1.0)       # (-20 + 10) / (4 + 6)
      assert wk["iran_goldstein"] == pytest.approx(-0.5)  # (5 - 15) / 20

  @pytest.mark.parametrize("col", ["articles_iran_actor", "articles_iran_conflict"])
  def test_zero_iran_week_raises(synthetic_raw, col):
      raw = dict(synthetic_raw)
      g = raw["gdelt"].copy()
      g.loc[g["date"].between("2015-06-06", "2015-06-12"), [col]] = 0
      raw["gdelt"] = g
      with pytest.raises(ValueError, match="2015-06-12"):
          W.build_weekly_panel(raw)

  def test_aigpr_publication_lag():
      m = pd.DataFrame({"month": pd.to_datetime(["2020-01-01", "2020-02-01"]), "aigpr_iran": [10.0, 20.0]})
      s = W.aigpr_weekly(m, pd.DatetimeIndex([T("2020-02-28"), T("2020-03-06"), T("2020-04-03")]))
      assert np.isnan(s.loc[T("2020-02-28")])
      assert s.loc[T("2020-03-06")] == 10.0 and s.loc[T("2020-04-03")] == 20.0

  def test_missing_vix_stays_nan():
      vix = pd.DataFrame({"date": pd.to_datetime(["2020-01-06"]), "vix": [15.0]})
      s = W.weekly_last(vix, "vix", pd.DatetimeIndex([T("2020-01-10"), T("2020-01-17")]))
      assert s.loc[T("2020-01-10")] == 15.0 and np.isnan(s.loc[T("2020-01-17")])

  def test_intraday_weekly_sums_and_saturday_rule():
      days = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
                             "2025-01-11", "2025-01-13"])           # Sat Jan 11 = Friday 22:00 UTC bar
      daily = pd.DataFrame({"date": days, "rv_intraday": [1.0, 2, 3, 4, 5, 6, 7],
                            "n_hours": [22, 23, 23, 23, 21, 1, 23],
                            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
      weeks = pd.DatetimeIndex([T("2025-01-03"), T("2025-01-10"), T("2025-01-17")])
      out = W.intraday_weekly(daily, weeks, first_week="2025-01-10")
      assert out.loc[T("2025-01-03")].isna().all()
      assert out.loc[T("2025-01-10"), "rv_hourly"] == 15 and out.loc[T("2025-01-10"), "n_hours_week"] == 112
      assert out.loc[T("2025-01-17"), "rv_hourly"] == 13 and out.loc[T("2025-01-17"), "n_hours_week"] == 24
      assert out.loc[T("2025-01-17"), "log_rv_hourly"] == pytest.approx(np.log(13))

  def test_rv_gk_with_opening_gap():
      daily = pd.DataFrame({"date": pd.to_datetime(["2025-01-03", "2025-01-10"]), "rv_intraday": 1.0, "n_hours": 22,
                            "open": [98.0, 100.0], "high": [100.0, 104.0], "low": [97.0, 98.0], "close": [99.0, 102.0]})
      out = W.intraday_weekly(daily, pd.DatetimeIndex([T("2025-01-10")]), first_week="2025-01-10")
      gk = 1e4 * (np.log(100 / 99) ** 2 + 0.5 * np.log(104 / 98) ** 2 - (2 * np.log(2) - 1) * np.log(102 / 100) ** 2)
      assert out.loc[T("2025-01-10"), "rv_gk"] == pytest.approx(gk)

  def test_zero_hour_week_raises():
      daily = pd.DataFrame({"date": pd.to_datetime(["2025-01-06", "2025-01-20"]), "rv_intraday": 1.0, "n_hours": 22,
                            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
      weeks = pd.DatetimeIndex([T("2025-01-10"), T("2025-01-17"), T("2025-01-24")])
      with pytest.raises(ValueError, match="2025-01-17"):
          W.intraday_weekly(daily, weeks, first_week="2025-01-10")

  def test_full_panel_shape(synthetic_raw):
      p = W.build_weekly_panel(synthetic_raw)
      assert len(p) == 1140 and p["date"].iloc[0] == T("2005-01-07") and p["date"].iloc[-1] == T("2026-11-06")
      assert (p["date"].dt.weekday == 4).all() and (p["rv"] > 0).all()
      assert list(p.columns) == ["date", "price", "r", "rv", "log_rv", "n_days", "iran_share", "iran_gdelt",
                                 "iran_gdelt_geo", "gdelt_days", "gpr", "vix", "ovx", "aigpr_iran",
                                 "iran_conflict", "iran_coop", "iran_tone", "iran_goldstein",
                                 "rv_hourly", "log_rv_hourly", "n_hours_week", "rv_gk"]
      assert p.loc[p["date"] < "2007-05-04", "ovx"].isna().all()      # synthetic OVX starts 2007-05-01
      assert p.loc[p["date"] < C.HOURLY_SAMPLE_START, "rv_hourly"].isna().all()
      assert (p.loc[p["date"] >= C.HOURLY_SAMPLE_START, "rv_hourly"] > 0).all()
      assert p["iran_tone"].eq(-2.0).all()                             # fixture: -20 / 10 every day

  def test_event_weeks():
      wk = {e[0]: W.week_ending(T(e[1])) for e in C.EVENTS}
      assert wk["abqaiq_2019"] == T("2019-09-20") and wk["hormuz_2026"] == T("2026-03-06")
  ```

- [ ] **Step 3: Run the tests and confirm they fail**

  Run: `uv run pytest tests/test_weekly.py -q`
  Expected: FAIL (`module 'brent.weekly' has no attribute ...`).

- [ ] **Step 4: Implement the functions listed under Interfaces in `src/brent/weekly.py`**

  - Map every daily row to `week_ending(date)` and aggregate with `groupby`, then `reindex(weeks)`.
  - Daily log returns come from consecutive Brent observations, and C_{d−1} for `rv_gk` from the previous intraday row (`.shift()` is allowed in this file).
  - `build_weekly_panel` uses `weeks = fridays()`. The first week's `r` needs the last close before 2005-01-01, which the 2004-12 data provides.

- [ ] **Step 5: Run the tests and confirm they pass**

  Run: `uv run pytest -q`
  Expected: all PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/brent/weekly.py tests/test_weekly.py tests/conftest.py
  git commit -m "Add Friday-week construction, hourly RV and GDELT tone to the panel builder"
  ```

---

### Task 7: Notebook 02 — weekly panel, data table and plots

**Files:**
- Create:
  - `src/brent/plots.py` (minimal: `setup_style()`, `save_fig(fig, name) -> Path`, PNG at 150 dpi to `paths.FIGURES`; Plan 3 extends it);
  - `notebooks/02_weekly_panel.ipynb`;
  - `data/processed/weekly_panel.csv`;
  - `results/tables/{data_summary,rv_measures_check}.csv`;
  - `results/figures/{data_series,gdelt_share_2013_break,gdelt_main_vs_geo,rv_hourly_vs_daily,gdelt_tone}.png`.
- Modify: `src/brent/config.py` (only if the first non-NaN `rv_hourly` week ≠ `HOURLY_SAMPLE_START`)

**Interfaces:**
- Consumes: `sources.load_raw`, `weekly.build_weekly_panel`, `weekly.gdelt_weekly` (for the dropped-row count), `paths.save_table`
- Produces: `data/processed/weekly_panel.csv`, the input to every later notebook. Columns as in Task 6, `date` in ISO format.

- [ ] **Step 1: Build the panel**

  1. Load the eight raw files, using the six daily sources, AI-GPR and the intraday daily file (`raw["intraday"] = load_raw("brent_intraday_daily")`). If that file is missing because it is git-ignored, stop with a message pointing to notebook 01's Dukascopy cell.
  2. Run `build_weekly_panel` and print the GDELT dropped-row count.
  3. Save with `panel.to_csv(paths.PROCESSED / "weekly_panel.csv", index=False)`.

  If it raises on zero-Iran weeks (any variant), AI-GPR ≤ 0, or a zero-hour intraday week, stop and show the user the listed weeks. Do not offset or fill (spec §7.2 step 6).

- [ ] **Step 2: Data table** (spec §7.2 step 2)

  The table covers `r, rv, log_rv, rv_hourly, log_rv_hourly, rv_gk, iran_gdelt, iran_conflict, iran_coop, iran_tone, iran_goldstein, gpr, vix, ovx, aigpr_iran`, with NaNs dropped per column. For each, compute:
  - N, mean, SD, min, max;
  - the AR(1) coefficient (`series.autocorr(1)`);
  - the ADF statistic and p-value from `adfuller(x, regression="c", autolag="AIC")`.

  Save with `save_table(df, "data_summary")`.

- [ ] **Step 3: Figures**

  - `data_series.png`: one panel per variable, 2005–2026, with a dashed line at Jan 1, 2026.
  - `gdelt_share_2013_break.png`: `iran_gdelt` with a vertical line at the week ending 2013-04-05. Print the mean of the 52 weeks before vs. after, and the OLS `iran_gdelt ~ 1 + post2013` coefficient (descriptive only).
  - `gdelt_main_vs_geo.png`: `iran_gdelt` vs `iran_gdelt_geo`.
  - `gdelt_tone.png` (spec §7.2 step 8): `iran_conflict` and `iran_coop` (log shares) and `iran_tone` (level), with the Apr 2013 line and the nine event weeks marked. Print the correlations of `iran_conflict` and `iran_coop` with `iran_gdelt`.

  Add a markdown cell interpreting the break. If it is large, note "consider D2 (spec §6.4)" and tell the user.

- [ ] **Step 4: Hourly vs. daily RV check** (spec §7.2 step 7)

  - On `C.STANDARDIZE_WINDOW_HOURLY` (2007-01-05 → 2012-12-28) and, for information, on 2013–2018, compute:
    - the correlation of `rv_hourly` with `rv`, in levels and in logs;
    - the ratio mean(`rv_hourly`)/mean(`rv`).

    Save these to `save_table(df, "rv_measures_check")`.
  - Figure `rv_hourly_vs_daily.png`: a log-log scatter, plus both series over time.
  - If the 2007–2012 ratio is outside `C.RV_LEVEL_RATIO_BAND` (0.8, 1.25), write it in a markdown cell and tell the user. Nothing is rescaled.
  - List the weeks with `n_hours_week < C.MIN_HOURS_WEEK` (flag, keep).
  - Count the weeks containing a removed roll return (from `brent_intraday_removed.csv`, mapped with `week_ending`), and print mean(`rv_gk`/`rv_hourly`) for those weeks vs. the rest.

- [ ] **Step 5: Coverage cell**

  List:
  - weeks with `gdelt_days < 6` and weeks with `n_days < 4`;
  - weeks with NaN `gpr`/`vix`, especially in the holdout.

  Nothing is filled. If any NaN `gpr`/`vix` week falls in 2026, tell the user (spec §12).

- [ ] **Step 6: Checks cell** (spec §7.2)

  - 1,140 rows; first date 2005-01-07, last 2026-11-06; all Fridays.
  - `rv > 0`; `1 ≤ n_days ≤ 5`; `iran_gdelt`, `iran_conflict`, `iran_coop`, `iran_tone`, `iran_goldstein` finite.
  - `rv_hourly`:
    - NaN before `C.HOURLY_SAMPLE_START`, and finite and > 0 from it on;
    - its first non-NaN week == `C.HOURLY_SAMPLE_START` (if not, fix config, rerun, and tell the user);
    - no NaN in 2013-01-04 → 2026-11-06.
  - Window counts 417/313/365/45 on `panel["date"]`.
  - Abqaiq week = 2019-09-20; Hormuz 2026 week = 2026-03-06.
  - `paths.check_budget()`.

- [ ] **Step 7: Restart and run all, then commit**

  ```bash
  git add src/brent/plots.py src/brent/config.py notebooks/02_weekly_panel.ipynb data/processed/weekly_panel.csv results/tables/data_summary.csv results/tables/rv_measures_check.csv results/figures
  git commit -m "Build weekly panel with hourly RV and GDELT tone, data summary and plots (notebook 02)"
  ```

---

## Done when

- `uv run pytest` passes.
- Notebooks 01 and 02 restart-and-run-all cleanly; 02 runs with the network off, given the intraday daily file.
- `weekly_panel.csv` is committed with 1,140 rows and the 22 columns of spec §9.1.
- `check_budget()` passes.
- `config.py` holds the decided `GDELT_FEED_RULE`, `AIGPR_URL`, `INTRADAY_ROLL_RULE` and `INTRADAY_REDISTRIBUTE`; `INTRADAY_START` and `HOURLY_SAMPLE_START` are verified against the data.
- The user has seen and approved:
  - the D1 findings and the GDELT 2013-break figure;
  - the roll-rule decision;
  - the Dukascopy terms finding;
  - the hourly vs. daily RV level check (and any flag outside [0.8, 1.25]).

  These feed Plans 2 and 3.
