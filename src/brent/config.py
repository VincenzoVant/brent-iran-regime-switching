"""All frozen constants and decisions (spec §8.1).

Nothing else in the project defines constants. Values marked "set in notebook NN"
are decisions that wait for data; they are edited here by hand at that point and committed.
The GCP project id is NOT here: it lives only in the env var BRENT_GCP_PROJECT or a git-ignored .env.
"""

SEED = 405

# --- Sample and windows (Friday weeks) ---
SAMPLE_START = "2005-01-07"
FREEZE_DATE = "2026-11-06"
FETCH_START = "2004-12-01"
TRAIN = ("2005-01-07", "2012-12-28")        # 417 weeks
VALID = ("2013-01-04", "2018-12-28")        # 313
TEST = ("2019-01-04", "2025-12-26")         # 365
HOLDOUT = ("2026-01-02", "2026-11-06")      # 45
RQ1_END = "2018-12-28"
RQ2_END = "2025-12-26"
STANDARDIZE_WINDOW = TRAIN                  # OVX run: OVX_STD_WINDOW
HIGHVOL_PCTL = 90                           # of |r| on TRAIN
GW_IRAN_PCTL = 75
GW_WINDOW = VALID                           # threshold from 2013-2018 (daily-feed era)
TVTP_CONTROLS = ["gpr", "vix", "log_rv"]
IRAN_VAR = "iran_gdelt"
TVTP_CONTROLS_HOURLY = ["gpr", "vix", "log_rv_hourly"]       # robustness #7 only
IRAN_TONE_VARS = ("iran_conflict", "iran_coop")              # M3tone, both N->H only

# --- Fetch sources (notebook 01) ---
BRENT_FETCH_START = "1987-05-20"            # first FRED DCOILBRENTEU observation
FRED_SERIES = {"brent": "DCOILBRENTEU", "vix": "VIXCLS", "ovx": "OVXCLS"}
OVX_FETCH_START = "2007-05-01"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
AIGPR_URL = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_country_monthly.csv"  # found 2026-11-23 on
                                            # matteoiacoviello.com/ai_gpr.html; columns Date (YYYY-MM-01), Iran_all
AIGPR_MONTHS = ("2004-01", "2026-10")
AIGPR_PUBLICATION_LAG_MONTHS = 1

# --- GDELT (BigQuery) ---
GDELT_TABLE = "gdelt-bq.full.events"
GDELT_V2_TABLE = "gdelt-bq.gdeltv2.events"  # only if the splice is needed
GDELT_MAX_BYTES = 200 * 10**9               # maximumBytesBilled on every real query
GDELT_SWITCH_DATE = "2013-04-01"
GDELT_FEED_RULE = "dateadded"               # DECIDED 2026-11-23 from D1: archive rows carry a Feb 2013 bulk
                                            # DATEADDED (99.5%, before the switch); "sourceurl" unusable (no NULL URLs)
GDELT_STALE_AFTER_YEARS = 1                 # daily-feed events dated > 1 year before DATEADDED -> feed "stale"
GDELT_STALE_POLICY = "merge"                # DECIDED 2026-11-23 after Q1: the Jan 1-5, 2020 "stale" rows are
                                            # CURRENT news with a broken SQLDATE (the normal feed nearly vanishes
                                            # those days), so weekly.py counts stale rows as daily feed (and reports
                                            # how many). Elsewhere stale is ~25 events/day: negligible.
# GDELT 1.0 has no daily-feed rows Jun 14 - Jul 1, 2025 (Israel-Iran war). DECIDED 2026-11-23: fill these days
# from GDELT 2.0, level-shifted to 1.0 on the other 2025 weeks (spec §6.4 splice rules). In practice 2.0 only
# has Jun 14: GDELT as a whole was down Jun 15 - Jul 1 (both tables empty), see GDELT_EMPTY_WEEK_POLICY.
GDELT_V2_PATCH_DAYS = ("2025-06-14", "2025-07-01")
GDELT_V2_PATCH_QUERY = (20250101, 20251231)  # GDELT 2.0 days fetched (patch + calibration overlap)
GDELT_SPLICE_MIN_CORR = 0.8                 # weekly log-share correlation 1.0 vs 2.0 required on the overlap
                                            # (2025 result: 0.934 over 48 weeks, shift -0.109 = 2.0 share x 0.90)
# A week with NO GDELT day at all (only the week ending 2025-06-27) takes the previous week's GDELT values
# (last observation carried forward = what a forecaster actually had while GDELT was down; no look-ahead).
# The panel flags it (gdelt_filled = 1, gdelt_days = 0). DECIDED 2026-11-23.
GDELT_EMPTY_WEEK_POLICY = "locf"
GDELT_MAX_EMPTY_WEEKS = 1                   # more consecutive empty weeks than this -> raise, never fill
GDELT_SPLICE_WEEK = None                    # set only if the GDELT 2.0 splice is needed

# --- Intraday RV (added 2026-11-23) ---
DUKASCOPY_INSTRUMENT = "BRENT.CMD/USD"      # dukascopy-python code "BRENT.CMD-USD"; dukascopy-node id "brentcmdusd"
INTRADAY_START = "2010-12-02"               # first Dukascopy hourly bar (Thu). Verified 2026-11-23 with both
                                            # dukascopy-python and npx dukascopy-node: nothing exists before it
                                            # (the Nov 2006 date in the old README was wrong). Accepted by the user.
INTRADAY_END_UTC = "2026-11-06 22:00"       # bars starting at/after this are dropped (freeze)
INTRADAY_PRICE_SIDE = "bid"
INTRADAY_DAY_END_UTC = 22                   # day d = bars starting in [d-1 22:00, d 22:00) UTC
INTRADAY_ROLL_SCAN = {"k": 8.0, "window": 120}   # roll_candidates defaults
INTRADAY_ROLL_RULE: dict | None = {"kind": "gap", "max_gap_hours": 84}
# DECIDED 2026-11-23 (notebook 01, 5d): no contract rolls are visible (k=8 candidates are spread over every
# day of the month and cluster at 13-16 UTC = US data/EIA news hours; Dukascopy vs FRED shows no monthly
# step), so no news jump is removed. Only returns spanning a DATA HOLE (> 84 h since the previous bar;
# weekends ~52 h, Easter/Christmas 74-80 h are kept) are removed: 9 returns, all before Aug 2013.
INTRADAY_REDISTRIBUTE: bool | None = False  # DECIDED 2026-11-23 by the user: the Dukascopy-derived daily files
                                            # stay local (git-ignored); the weekly panel is still committed
HOURLY_SAMPLE_START = "2013-06-07"          # DECIDED 2026-11-23 (start of notebook 02): first week of hourly RV.
# Why not Dec 2010: Jan 2012 - May 2013 Dukascopy is unreliable (stale quote 111.71 re-printed, 12-18 bars/day,
# weekend bars, holes Oct 18-Nov 15 2011 and Feb 20-May 22 2013). Clean from 2015, mostly clean from Jun 2013.
# Effects: HAR-X/GARCH-X (X_SAMPLE_START) train from here; rv_hourly is NaN before it.
STANDARDIZE_WINDOW_HOURLY = (HOURLY_SAMPLE_START, "2015-06-05")   # 105 weeks = the first two clean years.
# Only a choice of units for the hourly variables (GARCH-X scaling, E1 z-scores, RV level check).
MIN_HOURS_WEEK = 60                         # flag (not drop) weeks with fewer hourly returns
RV_LEVEL_RATIO_BAND = (0.8, 1.25)           # mean(rv_hourly)/mean(rv) on STANDARDIZE_WINDOW_HOURLY; outside -> tell the team
RV_MEASURES = ("rv_hourly", "rv", "rv_gk")
MAIN_RV = "rv_hourly"

# --- Extra models (added 2026-11-23; status open) ---
BENCHMARKS_MAIN = ("GARCH_t", "HAR", "RW", "HAR_h", "HARX_C", "HARX_CI", "GARCHX_C", "GARCHX_CI")
LINEAR_MODELS = ("HAR", "HAR_h", "HARX_C", "HARX_CI", "GARCH_t", "GARCHX_C", "GARCHX_CI")  # best-linear pool
X_CONTROLS = ("gpr", "vix")                 # news/fear regressors of HARX_C and GARCHX_C (+ IRAN_VAR in *_CI)
X_RV_COL = "rv_hourly"                      # "rv" only in the no-intraday fallback
X_SAMPLE_START = HOURLY_SAMPLE_START        # SAMPLE_START in that fallback
GARCHX_SCALE_WINDOW = STANDARDIZE_WINDOW_HOURLY

# --- Model settings (frozen later) ---
PHI_SWITCHING: bool | None = None           # set from notebook 03, then frozen
M3_VARIANT = "constrained"                  # or "iran_both_logits" (notebook 05 validation)
SPEC_FROZEN = False                         # True after notebook 05 validation stage
SEARCH_REPS_FIRST = 30
N_RESTARTS = 3
RESTART_EVERY = 13
SIGMA2_RATIO_MIN = 1.5
VAR_FLOOR_FRAC = 0.05
MIN_DURATION = 2.0

# --- Robustness (Plan 3) ---
ROBUST_OOS_TARGETS = ("2019-01-04", "2025-12-26")
OVX_START = "2007-05-11"
OVX_STD_WINDOW = ("2007-05-11", "2012-12-28")
DUMMY_POST2013_WEEK = "2013-04-05"
TAIL_PCTLS = (90, 95, 99)
MZ_HAC_LAGS = 4
RV_FLOOR_FRAC = 0.01
BOOT_REPS = 199
PLACEBO_IN = 200
PLACEBO_OOS = 20
PLACEBO_MIN_SHIFT = 52

# --- Events (research plan §9, frozen before any estimation): id, date, label, confounder ---
EVENTS = [
    ("hormuz_2011", "2011-12-27", "Iran threatens to close Hormuz; EU embargo Jan 23", "Euro debt crisis"),
    ("jcpoa_2018", "2018-05-08", "US withdraws from the JCPOA", "Sanctions return Nov"),
    ("tankers_2019", "2019-06-13", "Tanker attacks in the Gulf of Oman", "US-China trade war"),
    ("abqaiq_2019", "2019-09-14", "Abqaiq attack, attributed to Iran", "A surprise, not predictable"),
    ("soleimani_2020", "2020-01-03", "Soleimani killed; Iran retaliates Jan 8", "COVID-19 follows"),
    ("iran_israel_apr2024", "2024-04-13", "Iran's drone and missile attack on Israel", "Red Sea attacks"),
    ("iran_israel_oct2024", "2024-10-01", "Iran's missile attack; Israel responds Oct 26", "Red Sea attacks"),
    ("israel_iran_2025", "2025-06-13", "Israel strikes Iran; US strikes Jun 22", "Ceasefire Jun 24"),
    ("hormuz_2026", "2026-02-28", "US-Israel strikes; Hormuz disrupted", "Holdout, not in the test"),
]
NON_IRAN_EVENTS = [
    ("lehman_2008", "2008-09-15"),
    ("covid_2020", "2020-03-09"),
    ("russia_ukraine_2022", "2022-02-24"),
]

# --- Size budget (spec §6.7) ---
SIZE_LIMIT_FILE = 10 * 2**20
SIZE_LIMIT_DATA = 5 * 2**20
SIZE_LIMIT_TOTAL = 20 * 2**20
SIZE_LIMIT_NOTEBOOK = 5 * 2**20
SIZE_LIMIT_NOTEBOOKS = 15 * 2**20
