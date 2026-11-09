# Brent × Iran Regime-Switching — Feasibility Check & Plan

*Prepared 2026-11-09. Based on 10 parallel research passes covering: price data, event datasets, text-based risk indices, the event chronology, econometrics/software, literature, evaluation design, controls, novel data, and a red-team review. No project code yet.*

**Working idea:** forecast weekly Brent oil returns with a two-state Markov-switching (MS) model (normal vs crisis), and test whether Iran-related events explain or predict switches between the two regimes.

---

## 1. Verdict: GO, with three changes

The project is feasible in the 4 weeks to **Mon Dec 7, 6 pm**. The data is free and current, the software works, and the course covers the method (Lecture 4: Markov switching, Ang & Timmermann 2012). The timing is also good. A US/Israel–Iran war began on **Feb 28, 2026**, Hormuz has been shut or contested since, and Brent has ranged from about $72 to $126. That is the largest Iran-driven oil regime in modern data, and no published regime-switching paper covers it yet.

**Three changes are needed so the project doesn't end in an uninformative null result:**

1. **Make volatility the main forecast target, not the return.** Weekly oil returns are close to unpredictable. Alquist, Kilian & Vigfusson (2013) show the no-change forecast is very hard to beat, and they appear in the *Handbook of Economic Forecasting* co-edited by Timmermann. A two-state MS model mostly separates **variance** regimes anyway. So the primary target is **weekly realized variance**, and secondary targets are the **probability of a crisis week** and the **density forecast**. The return forecast stays in as the required RMSE / hit-rate check. Every required guideline test (Mincer-Zarnowitz, Diebold-Mariano, cumulative squared-error plot) still applies.
2. **Put Iran in the regime-switching probabilities, and race it against global geopolitical risk.** Use an MS model with **time-varying transition probabilities (TVTP)**: the chance of switching into the crisis regime depends on lagged variables. The sharp question is: *"Does Iran-specific risk raise the probability of entering the turbulent regime beyond what global geopolitical risk (GPR), market fear and lagged volatility already capture?"*
3. **Keep "explain" and "predict" separate.** Iran variables from the same week can *explain* a switch. Only variables known by the end of week *t* can *predict* the regime in week *t+1*. Surprise events (strikes, Abqaiq) can't be predicted. Build-ups (threats before acts, rising news attention, telegraphed retaliation) can.

---

## 2. Research questions

| # | Question | Type | How it's tested |
|---|---|---|---|
| RQ1 | Does weekly Brent show a distinct, persistent high-volatility regime, and when is it active? | Describe | 1 vs 2 regimes: bootstrap likelihood-ratio (LR) test; smoothed regime probabilities plotted against the event chronology |
| RQ2 | Do Iran risk measures **explain** entries into the crisis regime? | Explain (in-sample) | Same-week variable in the transition probabilities; LR test of TVTP vs constant transition probabilities |
| RQ3 | Do **lagged** Iran measures **predict** regime switches and improve volatility, density and crisis-probability forecasts out of sample? | Predict (OOS) | DM / Clark-West / Giacomini-White tests vs constant-transition MS and GARCH |
| RQ4 | Is the effect Iran-specific, or just generic geopolitics? | Robustness | Horse race against global GPR; placebo countries; shuffled event dates |

A plausible, still interesting outcome: before 2026, Iran shocks show up as **short-lived jumps** that fade within days (Abqaiq 2019, the June 2025 ceasefire drop), while 2026 is the first real **Iran regime**. "Iran risk became a regime driver only once physical Hormuz flows were hit" would be a good headline. A null result before 2026 is also reportable, phrased as "we can rule out effects larger than X."

---

## 3. Data (all checked live on 2026-11-09)

### Dependent variable: Brent
| Item | Choice |
|---|---|
| Primary series | FRED `DCOILBRENTEU` (Europe Brent spot, daily, 1987-05-20 → 2026-11-03), downloaded via `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU`. No API key needed; runs about a week behind |
| Backup / robustness | Yahoo `BZ=F` (ICE front-month futures, 2007 → today). It is unadjusted across contract rolls, so expect jumps at each roll |
| Weekly return | Friday close (Thursday if Friday is a holiday): r_t = 100·ln(P_t/P_{t−1}). **Don't use FRED's weekly series `WCOILBRENTEU`.** It is a weekly *average*, which introduces artificial autocorrelation, and it stops at 2026-09-04 |
| Weekly realized variance | RV_t = sum of squared daily log returns from Friday to Friday |
| Sample | About 2,050 weekly returns from 1987, or about 1,400 from 2000. End the sample at about **2026-10-30** and freeze the data |

Note: the largest weekly move in the whole sample is **+29.4% (week ending 2026-03-06)**. Six of the ten largest absolute weekly moves are in 1990–91, 2008–09 and 2020, so the crisis regime has competing causes.

### Iran measures (the key input)
**Important correction:** the standard Caldara–Iacoviello GPR files have **no Iran country index**; Iran is not among the 44 countries. The usable options are below.

| Measure | Freq | Span | Role |
|---|---|---|---|
| **AI-GPR `GPR_OIL_MiddleEast`** (Iacoviello & Tong 2026), `matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv` | Daily → weekly mean, log(1+x) | 1960 → 2026-10-31 | **Primary weekly driver, full sample.** Peaked at about 1,980 in mid-March 2026, against a median of 35 |
| **AI-GPR `Iran_all`**, plus the bilateral `USA\|Iran` and `Israel\|Iran` series (`ai_gpr_country_monthly.csv`) | Monthly; lag 1 month | 1960 → 2026-10 | The direct Iran measure; a check on the weekly series. Record high of 222 in March 2026 |
| **GDELT DOC 2.0 news attention and tone**, query `iran (oil OR hormuz)` | Daily → weekly | 2017 → today | **Novel / scraped data component** (the guidelines encourage this). 2017+ subsample |
| GDELT 2.0 events (BigQuery `gdelt-bq.gdeltv2.events`): Iran-involved material-conflict share | Weekly | Feb 2015 → today | Optional extension, about 1.5 days of work |
| Hand-coded event chronology (45 events, 2002–2026; see §9) | Event dummies | 2002 → 2026 | Validates the regimes; event-window table; separates anticipated from surprise events |

⚠️ **The AI-GPR index has a caveat to disclose.** It is a 2026 working-paper index scored by an LLM after the fact, with no archived past versions found. Using it in a pseudo-OOS test therefore carries some lookahead and revision risk. The keyword GPR (global) does have archived weekly versions on GitHub. GDELT news attention is real-time by construction. This is a good reason to have both AI-GPR and GDELT.

### Controls
All lagged one week; monthly series use only values already published by that date.
| Control | Source | Why |
|---|---|---|
| Global GPR, daily, split into Acts and Threats | `gpr_files/data_gpr_daily_recent.xls` (`GPRD`, `GPRD_ACT`, `GPRD_THREAT`) | The non-Iran geopolitics benchmark. Regress it on the Iran measure and keep the residual |
| VIX (1990+) or OVX (2007+) | FRED `VIXCLS`, `OVXCLS` | General market fear (2008, 2020). OVX is based on WTI, not Brent, and cuts the sample to 2007 |
| Lagged realized variance / absolute return | own data | The persistence any model must beat |
| OPEC meeting dummy / Känzig supply-news surprise | github.com/dkaenzig/oilsupplynews (to 2025-12) | Separates OPEC policy shocks |
| Optional: Kilian real-activity index (FRED `IGREA`, ~2-month release lag), broad dollar (`DTWEXBGS`), EIA inventory surprise | FRED / EIA | Demand side |

Placebo countries: AI-GPR series for Venezuela or Russia (oil-relevant) and Turkey or Korea (not oil-relevant).

---

## 4. Models

| Label | Model | Purpose |
|---|---|---|
| **B1** | Zero / prevailing-mean return; AR(1) | Required return benchmark |
| **B2** | GARCH(1,1)-t and GJR-GARCH-t (`arch` package); AR or HAR on log RV | Strong volatility benchmarks ("not obviously inferior") |
| **M1** | 2-state MS mean + variance with **constant** transition probabilities: `statsmodels MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True)` | The user's original idea; the nested benchmark |
| **M2** | MS-TVTP with **controls only** (global GPR, VIX, lagged RV) | Isolates "generic risk" |
| **M3** | MS-TVTP with **controls + Iran measure** (AI-GPR oil Middle East / Iran) | **Main model.** The M3 vs M2 comparison answers RQ3/RQ4 |
| M4 | M3 with the GDELT Iran-oil attention and tone variables (2017+) | Novel-data version |
| Robustness | Iran measure in the **mean** equation instead of the transitions; 3 regimes; winsorized returns; pre-2026 sample; `BZ=F` futures | Show the result isn't an artifact |

**Software (verified on statsmodels 0.15.0 with simulated data):**
- `exog_tvtp` works in `MarkovRegression` and `MarkovAutoregression`. Each transition is a logit, with parameters named like `p[0->0].tvtp1`.
- Row *t* of `exog_tvtp` drives the transition from S_{t−1} to S_t, so **pass the lag yourself** (row t = z_{t−1}).
- `results.forecast()` is **not implemented**, so the one-step forecast has to be written by hand. It is about 10 lines:
  - ξ_t = filtered probabilities (**never smoothed**)
  - P_{t+1} = `model.regime_transition_matrix(params, exog_tvtp=[[1, z_t]])`
  - π = P_{t+1} ξ_t
  - Predictive density = Σ π_i N(μ_i, σ_i²)
  - Variance forecast = Σ π_i(σ_i² + μ_i²) − mean²
- Fit times are about 0.2 s (constant transitions) and 0.75 s (TVTP) with `search_reps=20`. Re-fitting about 400 rolling windows takes roughly 10–15 minutes.

**Known pitfalls and fixes:**
- **Label switching:** after every fit, relabel so that crisis = the regime with higher σ².
- **Local optima:** use `search_reps` of 20–50 and keep the best likelihood.
- **Separation with rare dummies:** with event dummies the slope coefficient runs off to about −13 with a standard error of 0. Use **continuous** indices, not dummies, in the transition equation. Let the Iran variable affect only the normal→crisis transition (a constrained fit via `scipy.optimize`, since statsmodels has no `fix_params` for this). Report LR tests rather than Wald tests.
- **A regime that captures only a few outliers (e.g. 2020):** check expected regime durations and the number of weeks in each regime; winsorize as a robustness check.
- **No Student-t errors** in statsmodels MS. Note this as a limitation (R `MSGARCH` has t errors but only constant transition probabilities).

---

## 5. Out-of-sample design (maps onto the guideline checklist)

**Split** (choices made on the validation window; the test window is used once):

| Window | Period | Weeks | Contains |
|---|---|---|---|
| Estimation | 1988 – 2012 | ~1,300 | Gulf War, 2008, 2011–12 sanctions and Hormuz threats |
| Validation | 2013 – 2018 | ~310 | 2014–16 price collapse, JCPOA, 2018 US withdrawal |
| **Test** | **2019-01 – 2026-10** | **~410** | Abqaiq, Soleimani, COVID, Russia 2022, Iran–Israel strikes 2024, 12-Day War 2025, **2026 war** |

- **Main design:** rolling window of about 15 years, re-fit every week (or every 4 weeks if runtime becomes a problem). Expanding window as a robustness check.
- **Two test-window reports:** with 2026, and **excluding 2026**. This shows whether 2026 alone drives the result.

| Guideline item | What to produce |
|---|---|
| Benchmark | GARCH-t / GJR-t, constant-transition MS (M1), controls-only MS (M2), prevailing mean |
| Forecast vs actual plots | RV forecast vs RV, Iran weeks shaded; crisis probability vs realized crisis weeks |
| Error autocorrelation | ACF plot + Ljung-Box |
| RMSE / hit rate | RMSE **and QLIKE** for variance (QLIKE is robust to noisy RV; Patton 2011); return sign hit rate + Pesaran-Timmermann test |
| Mincer-Zarnowitz | RV_{t+1} = a + b·σ̂²_{t+1} with HAC standard errors, joint test (a,b) = (0,1); a linear-probability version for crisis probabilities |
| Diebold-Mariano | `statsmodels.tsa.stattools.diebold_mariano_test` with the Harvey-Leybourne-Newbold correction, on MSE and QLIKE |
| Cumulative squared-error plot | Σ(L_bench − L_model), with Iran events marked. **This is likely the most important figure** |
| Nested comparison (M3 vs M2) | Clark-West (an approximation for variance loss) + **Giacomini-White** with the instrument "lagged Iran risk is high". This directly tests "does the model help around Iran episodes" |
| Density | PIT histogram + Berkowitz test; log score / CRPS (`scoringrules`: `crps_mixnorm`, `logs_mixnorm`) |
| Crisis probability | Crisis week = \|r\| above the 90th percentile *of the training sample*. Brier score, reliability diagram, DM test on Brier loss |
| VaR (optional) | 5% VaR, Kupiec + Christoffersen tests |

Clark-West, Giacomini-White, Berkowitz, Kupiec and Pesaran-Timmermann have no ready-made function in statsmodels or `arch`. Each is roughly 10–25 lines with scipy and HAC OLS.

---

## 6. Identification and robustness (what keeps a reviewer happy)

- **Lookahead control.** Freeze the event list and index choice before estimating. Use filtered probabilities only. Pick the crisis threshold on the training sample. Date GDELT by `DATEADDED` (ingestion time), not by event date. Assign weekend events (Abqaiq, the Fordow strikes, Feb 28 2026) to the next trading week.
- **Reverse causality.** Use lags *t−1* and *t−2* only. Use Threats rather than Acts. Pre-scheduled events (sanction deadlines, snapback) can't be caused by oil moves. Placebo check: past Brent returns should not predict the Iran index.
- **Placebos.** (a) Placebo countries. (b) Shuffle the Iran index in blocks 500–1,000 times and see where the real likelihood gain falls. (c) Remove the 2008, 2014–16, 2020 and 2022 windows and check that the Iran effect survives.
- **Statistical power.** Pre-2026 there are only about 12–20 Iran episodes (30–60 event-weeks), so report confidence intervals and a minimum detectable effect. 2026 adds about 35 weeks of sustained Iran-driven turbulence.

---

## 7. Novel-data component (makes the project stand out)

- **Main analysis:** GDELT weekly Iran-oil news attention and tone, 2017–2026, as a driver of regime switches in M4. The API is verified, limited to one request per 5 seconds, and a one-off pull of about 45 minutes.
- **Appendix case study (ties to the Week 7 prediction-markets lecture):** compare **Polymarket "US strikes Iran by…?"** ($529M volume, Dec 2025 → Feb 28 2026) and **Kalshi `KXCLOSEHORMUZ`** prices with the model's crisis-regime probability over Dec 2025 – Apr 2026. *Did the market price the regime shift before the model did?* Both APIs give daily history for resolved markets (verified).
- **Optional overlay:** IMF PortWatch daily Hormuz tanker transits (2019 → 2026-11-01, one ArcGIS query). Transits fell from about 55/day to 0–1/day from Mar 1, 2026. This is a physical-disruption check on the regime timing; it moves at the same time as the crisis, so it is not a predictor.
- **Skip:** Google Trends (the 2026 peak swamps the scaling, and stitching windows leaks future information) and LLM-coded headlines (heavy lookahead bias; stretch goal only).

---

## 8. Timeline (4 weeks)

| Week | Dates | Deliverables |
|---|---|---|
| 1 | Nov 9 – 15 | Pull and freeze all data (Brent, AI-GPR, GPR, VIX/OVX, GDELT DOC). Build weekly returns and RV. **Freeze the event list.** Data table with mean, SD, AR(1), ADF tests; plots |
| 2 | Nov 16 – 22 | Benchmarks (GARCH-t, HAR). M1 → M3 in-sample. Regime plots against events. LR tests. 1-vs-2-regime bootstrap |
| 3 | Nov 23 – 29 *(Thanksgiving Nov 26)* | Hand-written OOS forecaster and rolling loop. All evaluation tests and figures. Placebo checks. GDELT M4 |
| 4 | Nov 30 – Dec 6 | Prediction-market appendix, write-up (≤25 pages), abstract, final checks |
| — | **Mon Dec 7, 6 pm** | Submit |

**Minimum viable version (if time runs short):** B1/B2 + M1 + M3 with AI-GPR, OOS from 2019, the required tests, and the event-window table.

**Cut first:** 3-state models, MS-GARCH, h=4 forecasts, GDELT event-level BigQuery data, LLM coding.

---

## 9. Event chronology (validation anchor; verify flagged items before use)

Pre-2024 price reactions are from memory and not checked online; 2024–2026 rows were checked against the cited news sources. Full table with 45 events and sources is in the research notes. Key anchors:

- **2011-12 → 2012-07:** Hormuz threat, then EU embargo (anticipated; mostly priced in)
- **2015-07-14 / 2016-01-16:** JCPOA signed / implemented (de-escalation)
- **2018-05-08:** US leaves the JCPOA (about +3%)
- **2019-05 → 09:** tanker attacks, drone downing, **Abqaiq attack on 2019-09-14** (Saturday; about +14.6% on Monday, then faded)
- **2020-01-03 / 01-08:** Soleimani killed / Al-Asad base strike
- **2024-04 and 2024-10:** Iran–Israel strike exchanges (Oct 1: +8% on the week; Oct 28: −4 to −6%)
- **2025-06-13 → 06-24:** 12-Day War, including US strikes on Fordow (June 21/22) and the ceasefire (−6%)
- **2025-09-28:** UN snapback sanctions (no reaction)
- **2026-02-28:** US–Israel strikes; Khamenei killed; **Hormuz shut.** +28% in the week to Mar 6
- **2026-03-09:** Brent at $119.50; record IEA stock release
- **2026-04-07/08:** ceasefire (−13%)
- **2026-04-13:** US naval blockade of Iranian ports; Brent reaches $126 on Apr 30
- **2026-06-17:** Islamabad Memorandum signed; Brent back near $71 by Jul 1
- **2026-07-08 → 14:** ceasefire collapses; blockade reimposed; Brent above $100
- **2026-09 → 10:** tanker attacks; Saudi East-West pipeline halted; about $105–115
- **2026-11-02 → 09:** Qatar-brokered plan to reopen Hormuz; Brent about $93–95

The 2026 entries come from news sources and Wikipedia's 2026 oil-market chronology; **verify them against Reuters or EIA before the write-up.** The late-October and early-November 2026 figures are the least certain.

---

## 10. Literature to cite (core)

- **Closest existing paper (cite it and explain the difference):** Wang, Ma, Hao & Gao (2021), *IRFA* 76:101756. An asymmetric TVTP-MS-GARCH in which *global* GPR drives oil regime switches. **This project's contribution:** an *Iran-specific* driver, weekly frequency, an explicit explain-vs-predict split, and an out-of-sample test that includes the 2026 Hormuz war.
- Also close: Qian, Zeng & Li (2022), *IREF*. GPR predicts oil volatility in an MS-AR model, through threats rather than acts.
- **Markov switching in oil:**
  - Fong & See (2002), *Energy Econ*: MS-GARCH with time-varying transitions driven by the futures basis.
  - Herrera, Hu & Pastor (2018), *IJF*: MS-GARCH vs EGARCH out of sample.
  - Nomikos & Pouliasis (2011); Vo (2009).
- **Geopolitical risk:**
  - Caldara & Iacoviello (2022), *AER*.
  - Iacoviello & Tong (2026), AI-GPR working paper.
  - Liu et al. (2019), *Energy Econ*: GPR in GARCH-MIDAS volatility forecasts.
  - Plakandaras, Gupta & Wong (2019), *Resources Policy*: modest GPR gains.
  - Noguera-Santaella (2016): geopolitics mattered less after 2000.
- **Iran:**
  - Mohaddes & Pesaran (2016), *Energy Econ*: Iranian supply shocks are largely offset by Saudi output.
  - Kilian, Plante & Richter (2026), Dallas Fed WP 2615, on the 2026 war.
- **Identifying oil shocks:** Kilian (2009), *AER*; Känzig (2021), *AER*; Baumeister & Hamilton (2019), *AER*.
- **Method:** Hamilton (1989); Diebold, Lee & Weinbach (1994); Filardo (1994), *JBES*.
- **Predictability baseline:** Alquist, Kilian & Vigfusson (2013), *Handbook of Economic Forecasting* vol. 2.

---

## 11. Top risks

| Risk | Severity | Mitigation |
|---|---|---|
| The crisis regime is mostly non-Iran shocks (2008, 2014–16, 2020, 2022) | High | Make it a tested hypothesis. Report the share of crisis weeks inside Iran windows. 2026 gives a real Iran regime |
| 2026 dominates the estimates | High | Report results with and without 2026. Consider a 3-state robustness check. 2026 sits in the test window |
| AI-GPR lookahead and revisions | Medium | Disclose. Cross-check with GDELT (real-time) and keyword GPR (archived versions) |
| MS estimation fragility (local optima, label flips, separation) | Medium | Multi-start fits, relabel by variance, continuous drivers, LR tests |
| A two-variance-level MS loses to GARCH on point volatility forecasts | Medium | Lean on density, crisis-probability and Giacomini-White results. A loss to GARCH is still a valid finding |
| 2026 facts from secondary sources | Low–Med | Check against Reuters or EIA before writing |

---

## 12. Decisions needed before starting

1. **Solo or group?** This sets how much of §7 is realistic.
2. **Bloomberg access at Rady?** If yes: roll-adjusted `CO1`, Brent options skew and futures-curve slope. Not required.
3. **How to treat 2026:** keep it in the test window (recommended), or also report a pre-2026-only analysis?
4. **Main Iran driver:** AI-GPR (long sample, lookahead caveat) vs GDELT (real-time, 2017+). Recommended: AI-GPR as the main driver, GDELT as the novel-data model M4.
