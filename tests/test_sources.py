import subprocess
import sys

import pandas as pd
import pytest

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
    df = pd.DataFrame({"Date": ["2003-12-01", "2004-01-01", "2026-10-01", "2026-11-01"], "Iran_all": [1, 2, 3, 4]})  # real column names
    out = S.parse_aigpr(df, "2004-01", "2026-10")
    assert out["aigpr_iran"].tolist() == [2, 3]
    assert out["month"].tolist() == [pd.Timestamp("2004-01-01"), pd.Timestamp("2026-10-01")]


def test_check_coverage_raises_on_short_series():
    df = pd.DataFrame({"date": pd.to_datetime(["2004-12-01", "2026-10-30"])})
    with pytest.raises(ValueError, match="2026-10-30"):
        S.check_coverage(df, "date", "2004-12-31", "2026-11-06")
    S.check_coverage(df, "date", "2004-12-31", "2026-10-30")      # passes


def test_snapshot_manifest_upserts(tmp_path, monkeypatch):
    monkeypatch.setattr(S.paths, "RAW", tmp_path)
    df = pd.DataFrame({"date": pd.to_datetime(["2005-01-03"]), "vix": [14.0]})
    S.save_snapshot(df, "vix_daily", "FRED", "url1")
    S.save_snapshot(df.assign(vix=15.0), "vix_daily", "FRED", "url2")
    man = pd.read_csv(tmp_path / "manifest.csv")
    assert len(man) == 1 and man.loc[0, "url_or_table"] == "url2" and len(man.loc[0, "sha256"]) == 64
    assert S.load_raw("vix_daily")["date"].dtype.kind == "M"


# --- GDELT (Task 3) ---

def test_import_without_optional_groups():
    code = ("import brent.sources, sys; "
            "assert not any(m.startswith(('google.cloud', 'dukascopy_python', 'pandas_gbq')) for m in sys.modules)")
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
    v2 = S.render_q1("gdelt-bq.gdeltv2.events", "dateadded", version="v2")
    assert "DIV(DATEADDED, 1000000)" in v2 and "'v2'" in v2
    with pytest.raises(ValueError):
        S.render_q1("gdelt-bq.full.events", "other")


def test_project_id_from_env_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("BRENT_GCP_PROJECT", raising=False)
    monkeypatch.setattr(S.paths, "ROOT", tmp_path)
    assert S.gcp_project_id() is None
    (tmp_path / ".env").write_text("# comment\nBRENT_GCP_PROJECT=my-sandbox-123\n")
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


def test_http_get_retries_after_timeout(monkeypatch):
    import requests
    calls = []

    class Ok:
        text = "ok"
        def raise_for_status(self):
            pass

    def flaky(url, timeout, headers=None):
        calls.append(url)
        if len(calls) < 3:
            raise requests.exceptions.ReadTimeout("slow server")
        return Ok()

    monkeypatch.setattr(S.requests, "get", flaky)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    assert S._get("https://example.org").text == "ok" and len(calls) == 3


def test_http_get_sends_no_fake_browser_user_agent(monkeypatch):
    # FRED drops connections from fake browser user agents (seen 2026-11-23); requests' own default works.
    seen = {}

    class Ok:
        def raise_for_status(self):
            pass

    def capture(url, timeout, headers=None):
        seen["headers"] = headers or {}
        return Ok()

    monkeypatch.setattr(S.requests, "get", capture)
    S._get("https://example.org")
    assert "Mozilla" not in seen["headers"].get("User-Agent", "")


def test_render_q1_labels_stale_daily_rows():
    # D1 (2026-11-23): GDELT re-loaded 537k pre-2013 events in Jan 2020. Daily-feed events dated more
    # than C.GDELT_STALE_AFTER_YEARS before their DATEADDED get feed='stale' (kept in the file, dropped weekly).
    q = S.render_q1("gdelt-bq.full.events", "dateadded")
    assert "WHEN DATEADDED < 20130401 THEN 'backfile'" in q
    assert "WHEN SQLDATE < DATEADDED - 10000 THEN 'stale'" in q
    assert "IF(DATEADDED >= 20130401, DATEADDED, SQLDATE) AS day_key" in q     # stale rows keyed by add date
    assert "THEN 'stale'" in S.render_q1("gdelt-bq.full.events", "sourceurl")
    assert "SQLDATE < DIV(DATEADDED, 1000000) - 10000 THEN 'stale'" in S.render_q1(
        "gdelt-bq.gdeltv2.events", "dateadded", version="v2")
