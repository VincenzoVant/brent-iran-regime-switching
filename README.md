# Brent × Iran Regime-Switching

MGTF 405 Business Forecasting — group final project (due **Mon Dec 7, 2026, 6 pm**).

Forecast next-week Brent volatility with a two-state Markov-switching AR(1) and test whether Iran-specific news (GDELT) explains or predicts switches into the high-volatility regime, beyond global geopolitical risk, VIX and lagged volatility.

## Design plan

See [the project plan](<Plans/Brent-Iran Regime Switching - Feasibility and Plan.md>) (and the pitch deck `Plans/Brent-Iran_Pitch_Modern-Sage.pptx`) for the research questions, data sources, models, out-of-sample design, timeline and settled design decisions.

## Setup

Uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                   # creates .venv and installs the `brent` package
uv run pytest                             # all tests should pass before any notebook work
```

- **Kernel:** in VS Code pick `.venv/bin/python`.
- **Only to re-download raw data (notebook 01):**
  - `uv sync --group gdelt --group intraday` installs the Google BigQuery and Dukascopy libraries.
  - GDELT needs your Google Cloud sandbox project id in a git-ignored `.env` file: `BRENT_GCP_PROJECT=<id>`. Never commit it.
  - Node.js is only needed for the `npx dukascopy-node` fallback.
- **Everyone else:** notebooks 02–08 run offline from the committed files in `data/raw/`. No Google account and no network needed.
  - **One exception:** the Dukascopy intraday files (`data/raw/brent_intraday_*.csv`) are not committed. Before notebook 02, run `uv sync --group intraday`, then run section 5 of notebook 01 (a few minutes, no account needed). The committed `weekly_panel.csv` already contains the weekly hourly-RV columns, so notebooks 03–08 do not need this step.

Code lives in `src/brent/` (`config.py` holds every constant and decision). Notebooks are in `notebooks/`.
