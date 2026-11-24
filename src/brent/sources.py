"""Fetchers for every raw data source (spec §6, §8.3). Only notebook 01 calls these.

The big picture
---------------
Every source is downloaded *into memory*, trimmed to the columns and dates we need, and only
then written to disk as one small CSV in data/raw/ (via `save_snapshot`). Big raw files (the GPR
Excel workbook, GDELT event rows, ~110,000 hourly Brent bars) are never saved.

Sources, in the order notebook 01 uses them:
    1. FRED      -> Brent spot price, VIX, OVX          (plain CSV download, no key)
    2. GPR       -> daily geopolitical risk index       (Excel file read in memory)
    3. AI-GPR    -> monthly Iran geopolitical risk      (CSV, robustness only)
    4. GDELT     -> daily Iran news counts + sentiment  (SQL on Google BigQuery; only daily totals come back)
    5. Dukascopy -> hourly Brent bars -> daily realized variance (RV) + daily open/high/low/close

Optional libraries (Google for GDELT, dukascopy_python for intraday) are imported *inside* the
functions that need them, so `import brent` works for teammates who never installed them.

Returns are computed with np.diff / .diff(), never with a lag/shift operator: lags happen only in
design.py (spec §8.5), so we can grep the code base to prove there is no look-ahead.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from brent import config as C
from brent import paths

# No custom User-Agent: since Nov 2026 FRED drops connections from fake browser user agents (and from
# custom ones), while requests' own default "python-requests/x.y" works. Checked 2026-11-23 for all 3 hosts.
GPR_URL = C.GPR_URL
# cosd/coed = start/end dates: FRED trims on its side, so only our window is transferred.
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={start}&coed={end}"
# One row per saved file. sha256 is a fingerprint: if a file ever changes, its hash changes and git shows it.
MANIFEST_COLUMNS = ["file", "source", "url_or_table", "fetched_at_utc", "first_date", "last_date",
                    "n_rows", "sha256", "bytes"]
DATE_COLUMNS = ("date", "month")


def _get(url: str, tries: int = 4, timeout: int = 120) -> requests.Response:
    """HTTP GET that retries on timeouts/connection errors (FRED is sometimes slow: seen 2026-11-23).

    Waits 10s, 20s, 30s between attempts. Raises on 404/500 instead of silently returning an
    error page, and gives up after `tries` attempts with the last error.
    """
    for attempt in range(tries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def _trim(df: pd.DataFrame, col: str, start, end) -> pd.DataFrame:
    """Keep rows with start <= df[col] <= end (both inclusive), sorted by date."""
    m = (df[col] >= pd.Timestamp(start)) & (df[col] <= pd.Timestamp(end))
    return df.loc[m].sort_values(col).reset_index(drop=True)


# =====================================================================================
# 1-3. FRED, GPR, AI-GPR (plain HTTP downloads)
# Each source has a `parse_*` function (pure, tested on fake data) and a `fetch_*` function
# (downloads, then calls the parser). Splitting them lets us test parsing without the internet.
# =====================================================================================

def parse_fred_csv(text: str, value_name: str, start: str, end: str) -> pd.DataFrame:
    """FRED CSV text -> DataFrame [date, value_name].

    FRED's header calls the date column `observation_date` (older files: `DATE`); we accept both.
    Missing values arrive as '.' or empty (e.g. Brent on Christmas); those rows are dropped,
    never filled.
    """
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    date_col = next(c for c in raw.columns if c in ("observation_date", "DATE"))
    val_col = next(c for c in raw.columns if c != date_col)          # the one other column = the series
    df = pd.DataFrame({"date": pd.to_datetime(raw[date_col]),
                       value_name: pd.to_numeric(raw[val_col], errors="coerce")}).dropna()  # '.' -> NaN -> dropped
    return _trim(df, "date", start, end)


def fetch_fred(series_id: str, start: str, end: str) -> pd.DataFrame:
    """Download one FRED series -> [date, value]. The notebook renames `value` to brent/vix/ovx."""
    text = _get(FRED_URL.format(id=series_id, start=start, end=end)).text
    return parse_fred_csv(text, "value", start, end)


def _parse_day_column(s: pd.Series) -> pd.Series:
    """GPR dates come either as real datetimes or as integers like 20041201 (YYYYMMDD)."""
    if s.dtype.kind == "M":
        return pd.to_datetime(s)
    return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def parse_gpr_frame(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """GPR daily workbook -> [date, gprd]. We keep only the headline daily index GPRD."""
    date_col = next(c for c in df.columns if c in ("DAY", "date", "Date", "day"))
    out = pd.DataFrame({"date": _parse_day_column(df[date_col]),
                        "gprd": pd.to_numeric(df["GPRD"], errors="coerce")}).dropna()
    return _trim(out, "date", start, end)


def fetch_gpr_daily(url: str, start: str, end: str) -> pd.DataFrame:
    """Download the GPR .xls into memory (io.BytesIO), never onto disk -> [date, gprd]."""
    return parse_gpr_frame(pd.read_excel(io.BytesIO(_get(url).content)), start, end)


def parse_aigpr(df: pd.DataFrame, start_month: str, end_month: str) -> pd.DataFrame:
    """AI-GPR country file -> [month, aigpr_iran]; `month` is the first day of the month.

    We keep only the `Iran_all` column. Months after 2026-10 are dropped: they could not have
    been published by our Nov 6 freeze date.
    """
    month_col = next(c for c in df.columns if str(c).lower() in ("month", "date", "yearmonth"))
    month = pd.to_datetime(df[month_col].astype(str)).dt.to_period("M").dt.to_timestamp()
    out = pd.DataFrame({"month": month, "aigpr_iran": pd.to_numeric(df["Iran_all"], errors="coerce")}).dropna()
    return _trim(out, "month", start_month, end_month)


def fetch_aigpr_iran(url: str, start_month: str, end_month: str) -> pd.DataFrame:
    """Download the AI-GPR monthly country file (CSV or Excel, in memory) -> [month, aigpr_iran]."""
    content = _get(url).content
    if url.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))
    return parse_aigpr(df, start_month, end_month)


# =====================================================================================
# Coverage check, snapshots and the manifest
# =====================================================================================

def check_coverage(df: pd.DataFrame, col: str, first_by: str, last_by: str) -> None:
    """Fail loudly if a series starts too late or stops too early.

    Example: FRED often runs a week behind, and GPR may stop before Nov 6. Without this
    check we would silently get a short series and missing values in the last weeks.
    The error message names the actual first/last date so we know how big the gap is.
    """
    if df.empty:
        raise ValueError(f"{col}: no rows")
    first, last = pd.Timestamp(df[col].min()), pd.Timestamp(df[col].max())
    if first.tzinfo is not None:                       # hourly data is tz-aware; compare as plain dates
        first, last = first.tz_localize(None), last.tz_localize(None)
    problems = []
    if first > pd.Timestamp(first_by):
        problems.append(f"first {col} {first.date()} is after {first_by}")
    if last < pd.Timestamp(last_by):
        problems.append(f"last {col} {last.date()} is before {last_by}")
    if problems:
        raise ValueError("; ".join(problems))


def _date_col(df: pd.DataFrame) -> str | None:
    """Which column holds the dates (used for the manifest's first/last date)."""
    for c in (*DATE_COLUMNS, "timestamp_utc"):
        if c in df.columns:
            return c
    return None


def _fmt(v) -> str | None:
    """Dates as YYYY-MM-DD; timestamps with a time or a time zone as full ISO strings."""
    if v is None or pd.isna(v):
        return None
    ts = pd.Timestamp(v)
    return ts.isoformat() if (ts.tzinfo is not None or ts != ts.normalize()) else ts.strftime("%Y-%m-%d")


def save_snapshot(df: pd.DataFrame, name: str, source: str, url: str) -> Path:
    """Write data/raw/<name>.csv and record it in data/raw/manifest.csv.

    The manifest has exactly one row per file ("upsert"): re-fetching a source replaces its row
    instead of adding a second one. Each row records where the data came from, when, its date
    range, row count, size, and a sha256 fingerprint of the file's bytes.
    """
    paths.RAW.mkdir(parents=True, exist_ok=True)
    out = paths.RAW / f"{name}.csv"
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    blob = out.read_bytes()
    col = _date_col(df)
    row = {
        "file": out.name, "source": source, "url_or_table": url,
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "first_date": _fmt(df[col].min()) if col and len(df) else None,
        "last_date": _fmt(df[col].max()) if col and len(df) else None,
        "n_rows": len(df), "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
    }
    man_path = paths.RAW / "manifest.csv"
    man = pd.read_csv(man_path) if man_path.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    man = man[man["file"] != out.name]                                 # drop this file's old row (upsert)
    man = pd.concat([man, pd.DataFrame([row])], ignore_index=True)[MANIFEST_COLUMNS]
    man.sort_values("file").to_csv(man_path, index=False)
    return out


def load_raw(name: str) -> pd.DataFrame:
    """Read data/raw/<name>.csv back, turning date / month / timestamp_utc columns into datetimes."""
    df = pd.read_csv(paths.RAW / f"{name}.csv")
    for c in DATE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, format="ISO8601")
    return df


# =====================================================================================
# 4. GDELT via Google BigQuery
#
# GDELT is a database of news "events" (who did what to whom, coded from news articles).
# The full table is far too big to download, so we let BigQuery add it up per day on Google's
# servers and download only the daily totals (~8,000 rows).
#
# The one subtle part: GDELT 1.0 has two parts.
#   - "backfile": events before Apr 1, 2013, loaded in bulk later. We date these by SQLDATE
#     (the event date).
#   - "daily" feed: from Apr 1, 2013, added every day. We date these by DATEADDED (the day the
#     article was processed = the day the news was available to traders).
# GDELT_FEED_RULE says how a row is labelled backfile vs daily. Query D1 tells us which rule works.
# =====================================================================================

def gcp_project_id() -> str | None:
    """Your Google Cloud project id: from env var BRENT_GCP_PROJECT, else from a line in .env.

    It is personal, so it never goes into config.py, git, or a printed notebook output.
    """
    pid = os.environ.get("BRENT_GCP_PROJECT", "").strip()
    if pid:
        return pid
    env = paths.ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("BRENT_GCP_PROJECT="):
                val = line.split("=", 1)[1].strip().strip("'\"")
                return val or None
    return None


# D1 = diagnostic query. Counts events per (event year, month added). From it we read:
#   (a) does the feed continue through Nov 2026?  (b) what DATEADDED do backfile rows carry?
#   (c) do backfile rows have an empty SOURCEURL?  -> decides GDELT_FEED_RULE.
GDELT_SQL_D1 = """
-- D1: coverage and how backfile rows are dated. A few thousand rows, inspected in memory, not saved.
SELECT
  DIV(SQLDATE, 10000)        AS event_year,
  DIV(DATEADDED, 100)        AS added_yyyymm,
  COUNT(*)                   AS n_events,
  SUM(NumArticles)           AS articles,
  COUNTIF(SOURCEURL IS NULL) AS n_no_url
FROM `{table}`
GROUP BY event_year, added_yyyymm
ORDER BY added_yyyymm, event_year
"""

# The two possible rules for "is this row backfile or daily feed?", and which date to key it by.
# D1 (2026-11-23) decided "dateadded"; "sourceurl" stays only for completeness.
#
# Third label, "stale": a daily-feed event whose event date (SQLDATE) is more than
# GDELT_STALE_AFTER_YEARS before the day it was added. We first suspected old news re-loaded in
# Jan 2020. Q1 showed the opposite: on Jan 1-5, 2020 almost the whole feed (incl. the Soleimani news)
# carries a broken SQLDATE, so these are CURRENT events. Policy (config.GDELT_STALE_POLICY = "merge"):
# weekly.py counts stale rows as daily feed and reports how many it merged. The label stays in
# gdelt_daily.csv as a record. Stale rows are keyed by their add date. {lag} = years x 10000
# because dates are YYYYMMDD integers: 20200115 - 10000 = 20190115, exactly one year earlier.
_FEED_EXPR = {
    "dateadded": ("CASE WHEN DATEADDED < 20130401 THEN 'backfile' "
                  "WHEN SQLDATE < DATEADDED - {lag} THEN 'stale' ELSE 'daily' END",
                  "IF(DATEADDED >= 20130401, DATEADDED, SQLDATE)"),
    "sourceurl": ("CASE WHEN SOURCEURL IS NULL THEN 'backfile' "
                  "WHEN SQLDATE < DATEADDED - {lag} THEN 'stale' ELSE 'daily' END",
                  "IF(SOURCEURL IS NULL, SQLDATE, DATEADDED)"),
}

# Q1 = the main query. One pass over the table; per (feed, day) it returns:
#   n_events, articles_all              -> denominators (all news that day)
#   articles_iran_actor                 -> MAIN measure: articles on events where Iran (IRN) is actor 1 or 2
#   articles_iran_actor_geo             -> robustness: also events located in Iran (FIPS code IR)
#   articles_iran_conflict / _coop      -> professor's sentiment idea: QuadClass 3-4 = conflict, 1-2 = cooperation
#   tone_x_articles_iran / _tone_n      -> article-weighted average tone = tone_x / tone_n (built later in weekly.py)
#   goldstein_x_articles_iran / _n      -> same for the Goldstein scale (-10 conflict ... +10 cooperation)
# The `_n` columns count only articles whose tone/Goldstein is not null, so nulls can't pull the mean to 0.
_Q1 = """
WITH ev AS (
  SELECT
    {feed_expr} AS feed,
    {day_expr} AS day_key,
    NumArticles,
    QuadClass,
    AvgTone,
    GoldsteinScale,
    (IFNULL(Actor1CountryCode, '') = 'IRN'
       OR IFNULL(Actor2CountryCode, '') = 'IRN')              AS iran_actor,
    (IFNULL(ActionGeo_CountryCode, '') = 'IR')                AS iran_geo
  FROM `{table}`
)
SELECT
  feed,
  day_key,
  COUNT(*)                                                                 AS n_events,
  SUM(NumArticles)                                                         AS articles_all,
  SUM(IF(iran_actor, NumArticles, 0))                                      AS articles_iran_actor,
  SUM(IF(iran_actor OR iran_geo, NumArticles, 0))                          AS articles_iran_actor_geo,
  COUNTIF(iran_actor)                                                      AS events_iran_actor,
  SUM(IF(iran_actor AND QuadClass IN (3, 4), NumArticles, 0))              AS articles_iran_conflict,
  SUM(IF(iran_actor AND QuadClass IN (1, 2), NumArticles, 0))              AS articles_iran_coop,
  SUM(IF(iran_actor AND AvgTone IS NOT NULL, AvgTone * NumArticles, 0))    AS tone_x_articles_iran,
  SUM(IF(iran_actor AND AvgTone IS NOT NULL, NumArticles, 0))              AS articles_iran_tone_n,
  SUM(IF(iran_actor AND GoldsteinScale IS NOT NULL, GoldsteinScale * NumArticles, 0)) AS goldstein_x_articles_iran,
  SUM(IF(iran_actor AND GoldsteinScale IS NOT NULL, NumArticles, 0))       AS articles_iran_goldstein_n
FROM ev
WHERE day_key BETWEEN {start} AND {end}
GROUP BY feed, day_key
ORDER BY day_key, feed
"""

# Columns of data/raw/gdelt_daily.csv (Q1's output with day_key renamed to date).
GDELT_DAILY_COLUMNS: tuple[str, ...] = (
    "feed", "date", "n_events", "articles_all", "articles_iran_actor", "articles_iran_actor_geo",
    "events_iran_actor", "articles_iran_conflict", "articles_iran_coop", "tone_x_articles_iran",
    "articles_iran_tone_n", "goldstein_x_articles_iran", "articles_iran_goldstein_n",
)


def render_q1(table: str, feed_rule: str, version: str = "v1",
              start: int = 20050101, end: int = 20261106) -> str:
    """Fill the Q1 template.

    version="v1": GDELT 1.0, feed labelled by `feed_rule` ("dateadded" or "sourceurl"):
                  'backfile', 'daily', or 'stale' (old events added late; see _FEED_EXPR).
    version="v2": GDELT 2.0 (only if 1.0 stops early). There DATEADDED is YYYYMMDDHHMMSS,
                  so the day is DATEADDED // 1,000,000, and every row is labelled 'v2'.
    """
    lag = C.GDELT_STALE_AFTER_YEARS * 10000                       # YYYYMMDD integer offset
    if version == "v2":
        feed_expr = f"CASE WHEN SQLDATE < DIV(DATEADDED, 1000000) - {lag} THEN 'stale' ELSE 'v2' END"
        day_expr = "DIV(DATEADDED, 1000000)"
    elif feed_rule in _FEED_EXPR:
        feed_expr, day_expr = _FEED_EXPR[feed_rule]
        feed_expr = feed_expr.format(lag=lag)
    else:
        raise ValueError(f"feed_rule must be 'dateadded' or 'sourceurl', got {feed_rule!r}")
    return _Q1.format(feed_expr=feed_expr, day_expr=day_expr, table=table, start=start, end=end)


def gdelt_credentials():
    """Google login for BigQuery. The first time it opens a browser; the token is cached in your home folder."""
    import pydata_google_auth
    return pydata_google_auth.get_user_credentials(["https://www.googleapis.com/auth/bigquery"])


def gdelt_dry_run(sql: str, project_id: str, credentials) -> int:
    """How many bytes the query WOULD scan. A dry run is free and returns no data.

    The free sandbox gives 1 TB of scanning per month, so we always check this first.
    """
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id, credentials=credentials)
    job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    return int(job.total_bytes_processed)


def gdelt_query(sql: str, project_id: str, credentials, max_bytes: int) -> pd.DataFrame:
    """Run the query for real. maximumBytesBilled is a hard cap: BigQuery refuses anything bigger."""
    import pandas_gbq
    return pandas_gbq.read_gbq(sql, project_id=project_id, credentials=credentials,
                               configuration={"query": {"maximumBytesBilled": str(max_bytes)}},
                               progress_bar_type=None)


def gdelt_daily_frame(q1: pd.DataFrame) -> pd.DataFrame:
    """Q1 result -> the gdelt_daily table: integer day_key 20050103 -> date 2005-01-03."""
    df = q1.rename(columns={"day_key": "date"}).copy()
    df["date"] = pd.to_datetime(df["date"].astype("int64").astype(str), format="%Y%m%d")
    df["feed"] = df["feed"].astype(str)
    return df[list(GDELT_DAILY_COLUMNS)].sort_values(["date", "feed"]).reset_index(drop=True)


# =====================================================================================
# 5. Dukascopy hourly Brent -> daily realized variance
#
# Why: the professor asked for a less noisy realized-variance (RV) measure. Weekly RV from 5
# daily returns is noisy; from ~110 hourly returns per week it is much more precise.
#
# Flow:   fetch_dukascopy_hourly  (download bars, in memory)
#      -> normalize_hourly        (UTC, sort, dedupe, drop empty bars, cut at the freeze)
#      -> hourly_returns          (100 x log price change between consecutive bars)
#      -> roll_candidates         (find suspicious jumps that may be futures-contract rolls)
#      -> intraday_daily          (sum of squared hourly returns per trading day = daily RV)
#
# The "hourly frame" has columns: timestamp_utc (bar START time, UTC), open, high, low, close, volume.
# =====================================================================================

HOURLY_COLUMNS = ["timestamp_utc", "open", "high", "low", "close", "volume"]


def intraday_method_available() -> str | None:
    """Which download route works on this machine: 'python' package, 'node' CLI fallback, or None."""
    try:
        import dukascopy_python  # noqa: F401
        return "python"
    except ImportError:
        return "node" if shutil.which("npx") else None


def dukascopy_frame_to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """dukascopy_python returns bars indexed by 'timestamp' with extra columns; keep our 6 columns."""
    df = raw.reset_index().rename(columns={"timestamp": "timestamp_utc"})
    return df[HOURLY_COLUMNS]


def _fetch_python(instrument: str, start: str, end: str, side: str, retries: int = 3) -> pd.DataFrame:
    """Download with the dukascopy_python package, one calendar year at a time, retrying on errors."""
    import dukascopy_python as dk
    logging.getLogger("dukascopy").setLevel(logging.WARNING)          # silence per-chunk progress lines
    code = instrument.replace("/", "-")                 # the package's code is "BRENT.CMD-USD" (dash)
    offer = dk.OfferSide.BID if side == "bid" else dk.OfferSide.ASK
    t0 = pd.Timestamp(start, tz="UTC")
    t_end = pd.Timestamp(end, tz="UTC")
    chunks = []
    while t0 < t_end:
        t1 = min(pd.Timestamp(year=t0.year + 1, month=1, day=1, tz="UTC"), t_end)   # end of this year's chunk
        for attempt in range(retries):
            try:
                candles = dk.fetch_candles(code, 1, dk.TimeUnit.HOUR, offer,
                                           t0.to_pydatetime(), t1.to_pydatetime())
                part = dk.to_dataframe(candles)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))                          # wait 5s, 10s before retrying
        if len(part):
            chunks.append(dukascopy_frame_to_hourly(part))
        t0 = t1
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=HOURLY_COLUMNS)


def _fetch_node(instrument: str, start: str, end: str, side: str) -> pd.DataFrame:
    """Fallback: the dukascopy-node command-line tool (needs Node.js).

    It can only write files, so it writes into a temporary folder OUTSIDE the repo that is
    deleted automatically when the `with` block ends.
    """
    node_id = instrument.replace("/", "").replace(".", "").lower()     # "BRENT.CMD/USD" -> "brentcmdusd"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["npx", "--yes", "dukascopy-node", "-i", node_id, "-from", start, "-to", end,
                        "-t", "h1", "-p", side, "-f", "csv", "-v", "true", "-dir", tmp],
                       check=True, capture_output=True)
        files = list(Path(tmp).glob("*.csv"))
        if not files:
            raise RuntimeError("dukascopy-node wrote no CSV")
        raw = pd.read_csv(files[0])
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)   # epoch ms -> UTC time
    if "volume" not in raw:
        raw["volume"] = np.nan
    return raw[HOURLY_COLUMNS]


def fetch_dukascopy_hourly(instrument: str = C.DUKASCOPY_INSTRUMENT, start: str = C.INTRADAY_START,
                           end: str = "2026-11-07", side: str = C.INTRADAY_PRICE_SIDE,
                           method: str = "python") -> pd.DataFrame:
    """Hourly Brent bars (bar-start times in UTC), kept in memory only.

    We use the BID side only: one download, and bid-to-bid returns have no bid-ask bounce.
    """
    if method == "python":
        return _fetch_python(instrument, start, end, side)
    if method == "node":
        return _fetch_node(instrument, start, end, side)
    raise ValueError(f"method must be 'python' or 'node', got {method!r}")


def normalize_hourly(raw: pd.DataFrame, end_utc: str = C.INTRADAY_END_UTC) -> pd.DataFrame:
    """Clean the bars.

    - Timestamps in UTC (timestamps without a zone are taken as UTC). Working only in UTC
      means daylight-saving changes never move a bar.
    - Sort by time and drop duplicate bars (keep the first).
    - Drop zero-volume bars: flat "filler" bars for closed hours would add fake zero returns.
    - Drop bars starting while the market is closed: any Saturday bar, and Sunday bars before 22:00 UTC
      (Brent reopens Sunday 22:00/23:00 UTC). Found 2026-11-23: in 2012-13 Dukascopy re-printed stale flat
      quotes on weekends; they would also distort Monday's first return. A Friday 22:00 bar is kept.
    - Drop bars starting at/after Nov 6, 2026 22:00 UTC (our freeze).
    """
    df = raw[HOURLY_COLUMNS].copy()
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp_utc"]))
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    df["timestamp_utc"] = ts
    df = df.sort_values("timestamp_utc", kind="stable").drop_duplicates("timestamp_utc", keep="first")
    df = df[~(df["volume"] == 0)]
    wd, hr = df["timestamp_utc"].dt.day_of_week, df["timestamp_utc"].dt.hour
    closed = (wd == 5) | ((wd == 6) & (hr < 22))                 # Saturday, or Sunday before the reopening
    df = df[~closed]
    df = df[df["timestamp_utc"] < pd.Timestamp(end_utc, tz="UTC")]
    return df.reset_index(drop=True)


def intraday_day(ts, day_end_utc: int = C.INTRADAY_DAY_END_UTC) -> pd.DatetimeIndex:
    """Assign each bar to a trading day. Day d = bars starting in [d-1 22:00, d 22:00) UTC.

    Trick: add 2 hours, then take the calendar date. A bar at 21:00 UTC Tuesday -> 23:00 -> Tuesday.
    A bar at 22:00 UTC Tuesday -> 00:00 Wednesday -> Wednesday.
    22:00 UTC is (about) the ICE Brent close. Consequences:
      - Sunday-evening reopening bars, and the weekend price gap, count for MONDAY.
      - A winter Friday 22:00 bar counts for Saturday, which weekly.py rolls into the NEXT week.
    """
    idx = pd.DatetimeIndex(ts)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return (idx + pd.Timedelta(hours=24 - day_end_utc)).normalize().tz_localize(None)


def hourly_returns(hourly: pd.DataFrame) -> pd.Series:
    """Hourly log returns in percent: r_h = 100 x ln(close_h / close_{h-1}).

    h-1 is the previous bar in the WHOLE series, however long the gap. So the overnight and
    weekend price changes land in the first bar after the gap: no return is lost and none is
    counted twice. The very first bar has no previous bar, so its return is NaN.
    """
    lc = np.log(hourly["close"].to_numpy(float))
    ret = np.concatenate([[np.nan], 100 * np.diff(lc)])
    return pd.Series(ret, index=pd.DatetimeIndex(hourly["timestamp_utc"]), name="ret")


def _scan(hourly: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per bar: its return, how 'normal' returns looked just before it, and how big it is relative to that.

    med_abs   = median |return| of the previous `window` returns (120 ~ one trading week).
                It uses strictly earlier returns (we move the rolling result down by one row).
    ratio     = |ret| / med_abs. A ratio of 50 means 50x a typical recent hourly move.
    gap_hours = hours since the previous bar (1 = normal; ~50 = weekend).
    """
    ret = hourly_returns(hourly)
    med = ret.abs().rolling(window, min_periods=window).median().to_numpy()
    med_prev = np.concatenate([[np.nan], med[:-1]])                   # value for bar i uses returns up to i-1
    ts = pd.DatetimeIndex(hourly["timestamp_utc"])
    # Hours between consecutive bars. Via total_seconds() so it is right whatever time unit pandas
    # stores (pandas 3 often uses microseconds, not nanoseconds).
    gap = pd.Series(ts).diff().dt.total_seconds().to_numpy() / 3600
    return pd.DataFrame({"timestamp_utc": ts, "ret": ret.to_numpy(), "med_abs": med_prev,
                         "ratio": np.abs(ret.to_numpy()) / med_prev, "gap_hours": gap})


def roll_candidates(hourly: pd.DataFrame, k: float = 8.0, window: int = 120) -> pd.DataFrame:
    """Suspicious jumps: hourly returns larger than k x the recent typical move, biggest first.

    Why: Dukascopy's Brent CFD switches to the next futures contract about once a month. The
    price difference between the old and new contract shows up as a one-off jump: a level
    shift, not volatility. Notebook 01 shows you these candidates and you pick the rule.
    """
    s = _scan(hourly, window)
    return s[s["ratio"] > k].sort_values("ratio", ascending=False).reset_index(drop=True)


def roll_mask(hourly: pd.DataFrame, rule: dict) -> pd.Series:
    """True for each bar whose return is removed as a contract roll, according to `rule`.

    Rule kinds (decided with you in notebook 01, stored in config.INTRADAY_ROLL_RULE):
      {"kind": "none"}                                  -> remove nothing
      {"kind": "threshold", "k": 8.0, "window": 120}    -> remove every return with ratio > k
      {"kind": "timestamps", "timestamps": [...]}       -> remove exactly these bars (ISO UTC strings)
      {"kind": "gap", "max_gap_hours": 84}              -> remove returns spanning a DATA HOLE: a gap since
                                                           the previous bar longer than max_gap_hours. A normal
                                                           weekend is ~52 h and Easter/Christmas ~76-80 h, so
                                                           these are kept. CHOSEN 2026-11-23 (notebook 01, 5d).
    """
    kind = rule.get("kind")
    ts = pd.DatetimeIndex(hourly["timestamp_utc"])
    if kind == "none":
        m = np.zeros(len(ts), bool)
    elif kind == "threshold":
        m = (_scan(hourly, int(rule["window"]))["ratio"] > float(rule["k"])).to_numpy()
    elif kind == "gap":
        m = (_scan(hourly, C.INTRADAY_ROLL_SCAN["window"])["gap_hours"] > float(rule["max_gap_hours"])).to_numpy()
    elif kind == "timestamps":
        wanted = pd.DatetimeIndex([pd.Timestamp(t) for t in rule["timestamps"]])
        wanted = wanted.tz_localize("UTC") if wanted.tz is None else wanted.tz_convert("UTC")
        m = ts.isin(wanted)
    else:
        raise ValueError(f"unknown roll rule kind: {kind!r}")
    return pd.Series(m, index=ts, name="removed")


def intraday_daily(hourly: pd.DataFrame, rule: dict | None,
                   day_end_utc: int = C.INTRADAY_DAY_END_UTC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly bars -> one row per trading day, plus a log of the removed (roll) returns.

    Daily columns:
      rv_intraday = sum of squared hourly returns that day (removed roll returns excluded)
      n_hours     = number of hourly returns that went into it
      open/high/low/close = first open, highest high, lowest low, last close of the day's bars
                            (used later for the Garman-Klass RV in the appendix)
    Refuses to run until the roll rule has been decided (rule=None), so we can't skip that step.
    """
    if rule is None:
        raise ValueError("INTRADAY_ROLL_RULE not set")
    window = int(rule.get("window", C.INTRADAY_ROLL_SCAN["window"]))
    scan = _scan(hourly, window)
    mask = roll_mask(hourly, rule).to_numpy()
    kept = np.where(mask, np.nan, scan["ret"].to_numpy())             # removed returns -> NaN
    day = intraday_day(hourly["timestamp_utc"], day_end_utc)
    g = pd.DataFrame({"date": day, "r2": kept**2, "kept": ~np.isnan(kept),
                      "open": hourly["open"].to_numpy(), "high": hourly["high"].to_numpy(),
                      "low": hourly["low"].to_numpy(), "close": hourly["close"].to_numpy()}).groupby("date")
    daily = pd.DataFrame({
        "rv_intraday": g["r2"].sum(min_count=0),                      # NaN (removed/first) squares are skipped
        "n_hours": g["kept"].sum().astype(int),
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
    }).reset_index()
    removed = scan.loc[mask, ["timestamp_utc", "ret", "ratio"]].reset_index(drop=True)
    removed.insert(1, "date", day[mask])
    # Why a return was removed: "data_gap" (return spans a hole in the data) or "roll" (contract switch).
    removed["reason"] = "data_gap" if rule.get("kind") == "gap" else "roll"
    return daily, removed


def fetch_dukascopy_daily(rule: dict, **fetch_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: download -> clean -> daily RV in one call."""
    return intraday_daily(normalize_hourly(fetch_dukascopy_hourly(**fetch_kwargs)), rule)
