# Brent × Iran Regime-Switching — Project Plan

*Updated 2026-11-20 to match the pitch deck `Brent-Iran_Pitch_Modern-Sage.pptx` (Group 7). The deck is the source of truth for the project idea. This file is the working plan that turns it into code, results and a report. First version: 2026-11-09 (see git history); §17 lists what changed.*

**Main question:** does Iran-specific risk raise the probability that Brent enters its high-volatility regime, beyond global geopolitical risk, market fear (VIX) and lagged volatility? Does that improve next-week volatility forecasts?

**Deadline:** Mon Dec 7, 2026, 6 pm.

**Status (Nov 20):** pitch deck done. A feasibility run of the MS-AR(1) with lagged log RV in the transition probability works (deck slide 11). The repo has no code yet.

---

## 1. Project in one paragraph

We forecast **next-week Brent volatility** (weekly realized variance of Brent spot) with a **two-state Markov-switching AR(1)** in weekly returns. The two states are a normal regime and a high-volatility regime. The probability of switching from normal to high volatility follows a logit in lagged observables (**time-varying transition probabilities**, TVTP). The key comparison is **M3 vs M2**: does adding a real-time **Iran news measure (GDELT)** to the switching equation beat a model with only global GPR, VIX and lagged RV? We test this in sample (explain) and out of sample (predict). We treat **2026 (the Hormuz war) as an untouched holdout stress test**.

Working hypothesis (deck slide 2): spare capacity absorbed earlier Iran shocks (Mohaddes & Pesaran, 2016). Iran risk drives a regime only when physical Hormuz flows are hit. **Expected result:** little gain before 2026, a clear gain in 2026. **If we're wrong:** Iran news doesn't reliably help outside major physical disruptions, which is still a reportable result.

Terminology: the state is the **high-volatility regime**, never the "crisis" or "Iran" regime. 2008, COVID-19 and Russia–Ukraine also land there.

---

## 2. Forecast targets

| Role | Target | Forecast | Who uses it |
|---|---|---|---|
| **Main result** | Next-week realized variance RV(t+1) of Brent spot, Friday to Friday | Σᵢ πᵢ(σᵢ² + mᵢ²) from the MS model (§6.4) | Producers and traders (position size, option prices, margin); risk managers (VaR, stress capital) |
| **Side result** | P(next week is in the high-volatility regime) | π_H(t+1 \| t) | A hedge trigger for airlines and shippers; war-risk insurers repricing Gulf cover |
| **Check** | Next-week return r(t+1) | Σᵢ πᵢ mᵢ; tested against a random walk | The course's required return checks; confirms returns are near-unpredictable (Alquist, Kilian & Vigfusson, 2013) |

**Dropped:** the density forecast as an output (deck slide 3). The fitted mixture still appears in the distribution robustness check (§10).

Weekly horizon: long enough to act on (rebalance a hedge, reprice cover), short enough that regime timing matters.

---

## 3. Research questions

| # | Question | How we test it | Sample |
|---|---|---|---|
| **RQ1 Regimes** | Is there a distinct, persistent high-volatility regime? Do the mean and the AR(1) coefficient differ between the two regimes? | 1 vs. 2 regimes: parametric-bootstrap LR test (the plain LR is non-standard under the null), plus BIC. Wald/LR test of μ(N)=μ(H) and φ(N)=φ(H). Expected regime durations and weeks in each regime | 2005–2018 for the specification decision; 2005–2025 for the reported in-sample fit |
| **RQ2 Explain** | Does Iran risk raise the probability of switching from normal to high volatility? | Sign and significance of β; LR test of M3 vs. M2 | 2005–2025 in sample; 2005–2026 reported as a supplement |
| **RQ3 Predict** | Does adding Iran risk improve next-week volatility forecasts, and does that hold in the 2026 stress period? | M3 vs. M2: QLIKE loss, Clark-West, Giacomini-White; DM against M1, GARCH-t and HAR | Main test 2019–2025 and the 2026 holdout, **reported separately** |

**Rule:** the φ decision from RQ1 (keep the AR term regime-specific or not) is made on 2005–2018 data only and then frozen. Otherwise it would use the test window.

---

## 4. Data

### 4.1 Sample

- **Jan 7, 2005 – Nov 6, 2026**, Friday-dated weeks, **~1,140 weekly observations**. All data **frozen on Nov 6, 2026**.
- Weeks run from Saturday to Friday. The week's value is at the Friday close, or at the last trading day if Friday is a holiday. The first weekly return (week ending Jan 7, 2005) uses the Dec 31, 2004 close.

### 4.2 Variables

| Variable | Source / code | Frequency | Available from | Weekly construction | Role |
|---|---|---|---|---|---|
| Brent spot | EIA via FRED `DCOILBRENTEU` (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU`, no key) | Daily | May 1987 | r(t) = 100·ln(P(t)/P(t−1)) on Friday closes; RV(t) = Σ (100·daily log return)² within the week | Target |
| Iran news share | GDELT 1.0 event database | Daily | 1979 (backfilled before Apr 2013) | Weekly share of Iran-related coverage, in logs (§4.3) | **Main Iran measure** |
| Global GPR | Caldara & Iacoviello, `matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls` (`GPRD`) | Daily | 1985 | Weekly mean, in logs | Control |
| VIX | FRED `VIXCLS` | Daily | 1990 | Friday close, in logs | Control |
| Lagged RV | own data | Weekly | — | log RV(t) | Control |
| AI-GPR Iran | Iacoviello & Tong (2026), `ai_gpr_country_monthly.csv` (`Iran_all`) | Monthly | 1960 | Last *published* month only | Robustness |
| OVX | FRED `OVXCLS` | Daily | 2007 | Friday close, in logs | Robustness (replaces VIX) |
| Brent implied vol | Bloomberg (only if someone gets to a Rady terminal) | Daily | check | Friday close | Robustness (optional) |

Notes:
- **Don't use FRED's weekly Brent `WCOILBRENTEU`.** It is a weekly *average*, which introduces artificial autocorrelation.
- Global GPR's 44 country indexes **don't include Iran**, so it can't double as the Iran measure. It does contain some Iran news, which is why β measures Iran's effect *beyond* global GPR. We put both in the model and do **not** residualize one on the other.
- Controls are kept few on purpose. OVX starts in 2007 and Bloomberg is hard to reproduce, so both are robustness only.
- **Standardize** every TVTP regressor with training-window means and SDs. This helps the logit converge and makes β comparable across variables.

### 4.3 Building the GDELT Iran measure

**Definition (freeze before estimating):**

> Iran share(w) = Σ NumArticles over events in week w with `Actor1CountryCode = 'IRN'` or `Actor2CountryCode = 'IRN'`, divided by Σ NumArticles over all events in week w. Iran(w) = ln(Iran share(w)).

A raw count trends up with GDELT's growth, so we use the share in logs. This actor-based definition is frozen as the main measure (§14). Robustness variant: also count events with `ActionGeo_CountryCode = 'IR'` (FIPS code).

**Source files:** `http://data.gdeltproject.org/events/`. The backfile is in yearly zips (to 2005) and monthly zips (2006 – Mar 2013); daily files start Apr 1, 2013. Alternative: one aggregated query on BigQuery `gdelt-bq.full.events` (GDELT 1.0) selecting only `SQLDATE`, the actor codes and `NumArticles`. Check the bytes estimate before running it, against the free tier. The DOC 2.0 news API only starts in 2017, so it can't be the main source.

**Dating (lookahead):**
- From Apr 2013, date each event by `DATEADDED`, the day GDELT ingested it.
- Before Apr 2013, the data was backfilled in 2013, so there is no real-time ingestion date. Date by `SQLDATE` (event date) and **state this in the report**.
- Check for a level shift at Apr 2013 when the daily feed begins. Plot the series and, if needed, add a post-2013 dummy to the switching equation as a robustness check.
- Confirm that the GDELT 1.0 daily feed runs through Nov 6, 2026. If it doesn't, splice in GDELT 2.0 events (2015+) after checking that the two series move together over their overlap.

### 4.4 Deliverable: `weekly_panel.csv`

One row per Friday, Jan 7, 2005 – Nov 6, 2026. Columns: `date, price, r, rv, log_rv, iran_gdelt, gpr, vix, ovx, aigpr_iran`, all *unlagged*. Lags are applied in the model code, written down in one place. Also produce a data table for the report (mean, SD, AR(1), ADF test) and time-series plots.

---

## 5. Forecast information set (no lookahead)

The forecast for week t+1 uses only data known at the **Friday close of week t**.

| Variable | How it becomes weekly | Known at forecast time? |
|---|---|---|
| Brent RV | Sum of squared daily returns | Yes: complete at week t close |
| GDELT Iran news (main) | Weekly share, dated when added (event date before Apr 2013) | Yes: news through week t |
| Global GPR, VIX | Weekly mean (GPR) or Friday value (VIX) | Yes: by end of week t |
| AI-GPR Iran (robustness) | Monthly | Only the last published month |
| Regime probabilities | Filtered from data through t | Yes; **never smoothed** |

Rules:
- Freeze the event list and variable choices before estimating. The event list is already frozen in the deck (§9).
- Set the 90th-percentile high-volatility threshold on **training data only** (2005–2012).
- Weekend events go into the next trading week (e.g. Abqaiq, Sat Sep 14, 2019 → week ending Sep 20; Sat Feb 28, 2026 → week ending Mar 6).
- 2026 is never used to choose anything.

---

## 6. Models

### 6.1 Two-state MS-AR(1) with TVTP

r(t) = μ_S + φ_S · r(t−1) + σ_S · ε(t),  S ∈ {N, H},  ε ~ N(0,1)

P(S(t) = H | S(t−1) = N) = Λ(c + γ′x(t−1) + β · Iran(t−1))

P(S(t) = N | S(t−1) = H) = Λ(d + δ′x(t−1))   (controls only, no Iran; §14)

- Λ is the logistic function. x = (global GPR, VIX, log RV).
- The mean and the variance switch. φ stays regime-specific only if RQ1 rejects φ(N)=φ(H) on 2005–2018.
- Iran is continuous, not event dummies. Six to nine events are too few for dummies, and rare dummies cause separation in the logit.

### 6.2 Model ladder

| Model | Switching probabilities | Purpose |
|---|---|---|
| **M1** | Constant | Nested baseline |
| **M2** | Driven by the controls (GPR, VIX, lagged RV) | Generic risk only |
| **M3** | M2 + Iran(t−1) in the normal→high switch | **Main model.** M3 vs. M2 answers RQ2 and RQ3 |

### 6.3 Benchmarks

| Target | Benchmark |
|---|---|
| Volatility | GARCH(1,1)-t on weekly returns (`arch`); HAR-RV on weekly RV, using lags of 1 week, 4 weeks and 13 weeks |
| Returns | Random walk in price, i.e. a zero-return forecast |

### 6.4 Implementation (statsmodels)

- **Use `MarkovRegression`, not `MarkovAutoregression`.** The deck's equation has an intercept form (μ_S + φ_S·r(t−1)). `MarkovAutoregression` uses Hamilton's deviations-from-mean form, which makes the mean depend on S(t−1) too. Instead:
  `MarkovRegression(r, k_regimes=2, trend='c', exog=r_lag, switching_exog=True, switching_variance=True, exog_tvtp=Z)`.
  Set `switching_exog=False` for the common-φ version.
- **`exog_tvtp` convention:** row t drives the transition S(t−1) → S(t), so row t must hold z(t−1). Include a constant column. M1 uses no `exog_tvtp`.
- **Iran only in the normal→high switch.** statsmodels puts the same regressors in every transition logit. To restrict Iran to N→H:
  1. Fit unconstrained.
  2. Relabel so that H is the regime with the higher σ².
  3. Re-fit by maximizing `model.loglike(params)` with `scipy.optimize`, holding Iran's coefficient in the H→N logit at zero.

  The controls stay free in both logits (§14). Fallback if the constrained fit is unstable: let Iran enter both logits and report the N→H coefficient.
- **One-step forecaster** (`results.forecast()` is not implemented for MS models; about 15 lines):
  - ξ(t|t) = filtered probabilities, never smoothed.
  - P(t+1) = `model.regime_transition_matrix(params, exog_tvtp=z_t)`. Columns sum to 1: P[i, j] = P(S(t+1)=i | S(t)=j).
  - π(t+1|t) = P(t+1) · ξ(t|t).
  - mᵢ = μᵢ + φᵢ · r(t).
  - **Variance forecast (compared with RV):** Σᵢ πᵢ (σᵢ² + mᵢ²). RV is an uncentered sum of squares, so don't subtract the squared mean.
  - **Return forecast:** Σᵢ πᵢ mᵢ.
  - **High-vol probability:** π_H(t+1|t).
- **Pitfalls:**
  - Label switching: relabel after every fit (H = higher σ²).
  - Local optima: use `search_reps` of 20–50 on the first fit, then warm-start each rolling re-fit from the previous week's parameters, with a few random restarts.
  - A regime that captures only a few outliers: check the expected durations 1/(1 − pᵢᵢ) and the number of weeks per regime.
  - statsmodels MS has **no Student-t errors**. See the distribution check in §10.
- Runtime: about 0.2 s per fit with constant transitions, about 0.75 s with TVTP. About 720 weekly re-fits (validation + test + holdout) × 3 MS models is well under an hour.

---

## 7. Out-of-sample design

| Window | Period | ≈ Weeks | Used for |
|---|---|---|---|
| Training | 2005–2012 | 417 | First estimation window; the 90th-percentile threshold; standardization |
| Validation | 2013–2018 | 313 | **All remaining specification choices** (φ switching or not; search settings). Checking the constrained M3 fit is stable |
| **Main test** | 2019–2025 | 365 | RQ3 headline results |
| **Holdout stress test** | Jan 2 – Nov 6, 2026 | 45 | Forecast once; never used to choose anything; reported separately |

- **Expanding window, re-estimated every week.** We forecast one week ahead, add the realized week, and re-fit. (The earlier 15-year rolling window doesn't fit a 2005 start: there are only 14 years before 2019.) Robustness: a fixed 10-year rolling window, if time allows.
- The loop runs straight through 2019–2026. Weekly re-estimation inside 2026 uses only 2026 data already realized, so it is a true real-time forecast. Results are split at Jan 1, 2026.
- Forecast output format: `forecasts.csv` with columns `date, model, target, forecast`. The evaluation code can then start before the models are finished.

---

## 8. Forecast evaluation

| Target | Tests and figures |
|---|---|
| **Volatility (main)** | QLIKE and RMSE loss (QLIKE is robust to RV being a noisy proxy; Patton, 2011). Mincer-Zarnowitz: RV(t+1) = a + b·forecast, HAC SEs, joint test (a, b) = (0, 1). **Diebold-Mariano:** M3 vs. M1, GARCH-t, HAR (Harvey-Leybourne-Newbold correction). **Clark-West and Giacomini-White: M3 vs. M2** (nested; GW instrument = "lagged Iran news is high") |
| **Returns (check)** | RMSE vs. random walk; sign hit rate; Pesaran-Timmermann test of direction |
| **High-vol probability (side)** | Brier score; reliability plot; hit rate on weeks with \|r\| above the training 90th percentile. If time allows: Brier score of the mixture-implied P(\|r\| > threshold) (§14) |

Also produce the course's standard figures for every target:
- forecast vs. actual plots (RV forecast vs. RV with the Iran event weeks shaded)
- **cumulative loss difference** Σ(L_M2 − L_M3) with the events marked; likely the key figure
- forecast-error ACF and Ljung-Box test

Clark-West, Giacomini-White, Pesaran-Timmermann and DM-HLN are each about 10–25 lines with scipy and HAC OLS. Check whether the installed statsmodels has a built-in DM test before writing one.

---

## 9. Event case studies (dates frozen before any estimation)

Events are for interpretation, not estimation. We plot a ±4-week window around each event week (§5 weekend rule).

| Date | Event | Confounder / note |
|---|---|---|
| Dec 27, 2011 | Iran threatens to close Hormuz; EU embargo Jan 23 | Confounder: euro debt crisis |
| May 8, 2018 | US withdraws from the JCPOA | Confounder: sanctions return Nov |
| Jun 13, 2019 | Tanker attacks in the Gulf of Oman | Confounder: US–China trade war |
| Sep 14, 2019 | Abqaiq attack, attributed to Iran | A surprise, not predictable |
| Jan 3, 2020 | Soleimani killed; Iran retaliates Jan 8 | Confounder: COVID-19 follows |
| Apr 13, 2024 | Iran's drone and missile attack on Israel | Confounder: Red Sea attacks |
| Oct 1, 2024 | Iran's missile attack; Israel responds Oct 26 | Confounder: Red Sea attacks |
| Jun 13, 2025 | Israel strikes Iran; US strikes Jun 22 | Ceasefire Jun 24 |
| Feb 28, 2026 | US–Israel strikes; Hormuz disrupted | Holdout, not in the test |

For each event:
- filtered high-vol probability from M2 and M3
- forecast losses, week by week

Repeat the same plots for **2008, COVID-19 (Mar 2020) and Russia–Ukraine (Feb 2022)** as non-Iran comparisons.

**What supports our story:** M3 raises the high-vol probability *earlier* than M2 before Iran build-ups, but not before non-Iran shocks. Surprises such as Abqaiq can explain a switch, but no model should foresee them.

---

## 10. Robustness checks (after the main analysis)

| Check | What we do |
|---|---|
| Iran in the mean | Add Iran(t−1) to the mean equation: does it move the mean return, not only the switch? |
| AR term | Forecasts with and without a regime-specific AR(1) |
| Iran measure | AI-GPR Iran (monthly, last published month) instead of GDELT. Disclose that it is LLM-scored after the fact, so it carries lookahead risk. Also the actor + location GDELT variant |
| Placebo | Shift the Iran series to random dates with circular shifts, which keep its autocorrelation. Run 200+ shifts for the in-sample LR gain and about 20 for the out-of-sample QLIKE gain (compute). The gain should vanish |
| Distribution | Overlay the fitted two-normal mixture on the histogram of weekly returns. If the tails are too thin, switch to Student-t errors within each regime. This needs a hand-written likelihood (statsmodels can't); if time runs out, report it as a limitation |
| Market fear | OVX (2007+) or Brent implied vol instead of VIX |

A model with global GPR only is already M2, so it needs no separate check.

---

## 11. Contribution and literature positioning

**Contribution:** we test whether real-time, Iran-specific news improves forecasts of Brent's high-volatility regime beyond general geopolitical risk and market fear.

| Year | Paper | Idea |
|---|---|---|
| 1989 | Hamilton | Markov switching: data alternate between hidden regimes |
| 1994 | Filardo; Diebold, Lee & Weinbach | Time-varying transition probabilities driven by observed variables |
| 2002 | Fong & See | Markov switching for oil futures volatility |
| 2012 | Ang & Timmermann | Regime changes in financial markets |
| 2018 | Herrera, Hu & Pastor | MS-GARCH vs. GARCH for oil volatility, out of sample |
| 2021 | **Wang, Ma, Hao & Gao (closest)** | Global GPR drives oil regime switches and improves forecasts |

Geopolitical risk and oil volatility: Liu et al. (2019) add GPR to GARCH-MIDAS. Qian, Zeng & Li (2022) find GPR predicts oil volatility in an MS model, through threats more than acts. Evidence is mixed, and it relies on global GPR indices.

**What we add:**
1. An Iran-specific driver, raced against global GPR.
2. Explaining (in sample) and predicting (out of sample) tested separately.
3. The 2026 Hormuz shock as an untouched stress test.

---

## 12. Timeline (Nov 20 → Dec 7)

| Dates | Deliverables |
|---|---|
| **Fri Nov 20 – Sun Nov 22** | Pull Brent, GPR, VIX, OVX, AI-GPR. **Build the GDELT Iran share** (the longest data task; start first). `weekly_panel.csv`, data table with ADF tests, data plots. Agree on the `forecasts.csv` format |
| **Mon Nov 23 – Sun Nov 29** *(Thanksgiving Nov 26)* | GARCH-t and HAR benchmarks. M1–M3 in sample on 2005–2018 → RQ1 φ decision, constrained M3 fit. Hand-written forecaster tested against statsmodels' filtered output. Validation run 2013–2018 → freeze the specification. RQ2 in-sample results on 2005–2025 |
| **Mon Nov 30 – Thu Dec 3** | Expanding-window loop through 2019–2026. All §8 tests and figures. Event case studies (§9). Robustness checks (§10) in priority order: placebo → AI-GPR → AR term → distribution → Iran in mean → OVX |
| **Fri Dec 4 – Sun Dec 6** | Write-up (≤ 25 pages), abstract, final figure pass, reproducibility check (fresh run from raw data) |
| **Mon Dec 7, 6 pm** | Submit |

**Minimum viable version:**
- M1–M3 with GDELT
- GARCH-t and HAR benchmarks
- main test and 2026 holdout
- QLIKE, MZ, DM and Clark-West tests
- the cumulative-loss figure
- the event table with M2 vs. M3 probabilities

**Cut first:** Student-t errors, the fixed rolling window, out-of-sample placebos (keep the in-sample ones), Brent implied vol.

---

## 13. Team split (3 people)

| Role | Owns | Owner |
|---|---|---|
| **Data and events** | All data pulls, GDELT construction, `weekly_panel.csv`, data table and plots, event windows (§4, §9) | ___ |
| **Models and forecasts** | Benchmarks, M1–M3, constrained fit, forecaster, expanding-window loop, `forecasts.csv` (§6, §7) | ___ |
| **Evaluation and write-up lead** | All tests and figures, robustness, literature, intro and conclusion, merging the report (§8, §10, §11) | ___ |

Interfaces: `weekly_panel.csv` (Friday dates, unlagged columns, lag conventions in one function) and `forecasts.csv` (`date, model, target, forecast`).

---

## 14. Decisions (settled 2026-11-20)

1. **Transitions:** the controls (GPR, VIX, lagged RV) enter **both** transition logits (normal→high and high→normal). Iran enters **only normal→high**. The deck only writes out the normal→high equation; this fills in the other one.
2. **GDELT variant:** the **actor-based** share (`Actor1CountryCode` or `Actor2CountryCode` = `IRN`) is the main measure, frozen now. Actor + location (`ActionGeo_CountryCode = 'IR'`) is a robustness check.
3. **High-vol probability scoring:** report π_H against \|r\| > training 90th percentile, as in the deck. If time allows, add the Brier score of the mixture-implied P(\|r\| > threshold), which matches the scored event exactly.
4. **φ switching:** decided by RQ1's test of φ(N)=φ(H) on 2005–2018, then frozen (§3).

---

## 15. Top risks

| Risk | Severity | Mitigation |
|---|---|---|
| The high-vol regime is mostly non-Iran shocks (2008, 2015–16, 2020, 2022) | High | It is the hypothesis, not a flaw: the question is whether Iran predicts *entry*. Non-Iran comparison plots (§9) |
| No pre-2026 Iran event starts a regime (deck slide 11 already hints at this) | High | This is the expected pre-2026 result. Report it with confidence intervals; 2026 is where the test has power |
| GDELT measure is noisy or breaks at Apr 2013 | Medium | Use the share, not counts; plot it; post-2013 dummy as a check; AI-GPR robustness |
| MS estimation fragility (local optima, label flips, constrained fit) | Medium | Multi-start fits, warm starts, relabel by variance, fallback unconstrained spec |
| MS loses to GARCH or HAR on point volatility | Medium | Still a valid finding. The main comparison is M3 vs. M2 |
| Lookahead in AI-GPR and in pre-2013 GDELT dating | Medium | AI-GPR is robustness only; disclose the event-date dating before 2013 |
| Time: 2.5 weeks including Thanksgiving | Medium | Minimum viable version and cut list in §12 |

---

## 16. Facts to verify against the frozen data before the write-up

- **2026 peak:** the deck says Brent spot went from $71 (Feb 27) to **$138 on Apr 7**. An earlier news-based note said $126 on Apr 30. Recompute from `DCOILBRENTEU`.
- **+29% in the week to Mar 6, 2026,** described as the largest weekly gain since 1987. Recompute.
- **Event moves on deck slide 2:** Abqaiq +11% in a day (earlier notes: +14.6%); Oct 2024 +12% in 5 days (earlier notes: +8% on the week). Recompute all seven from the spot series.
- **Deck slide 10** lists AI-GPR Iran's start date as "To confirm". It is monthly from 1960; update the slide.
- **2026 chronology** (ceasefires, blockade, reopening talks): check against Reuters or EIA wherever the report cites it.

---

## 17. Changes from the Nov 9 plan

Changed to match the deck:
- Main Iran measure: AI-GPR → **GDELT event-database share** (AI-GPR is now robustness only).
- Sample: 1987 → **2005**, frozen on Nov 6.
- Model: switching mean only → **MS-AR(1)**.
- Model ladder: M1–M4 → **M1–M3**.
- 2026 moved from inside the test window to a **separate holdout**.
- Rolling 15-year window → **expanding window**.
- RQ4 dropped. The μ/φ test was added to RQ1.
- The 45-event chronology was replaced by **9 frozen events** with named confounders.
- "Crisis regime" → **high-volatility regime**.

**Parked (not in the deck; only if there is spare time):**
- density forecast (PIT, Berkowitz, CRPS)
- Polymarket/Kalshi prediction-market appendix
- IMF PortWatch Hormuz transits (would illustrate the "physical flows" hypothesis)
- Bloomberg pull (CO1 futures, curve slope)
- OPEC/Känzig, Kilian and dollar controls
- GPR acts/threats split
- placebo countries
- 3-state model
- winsorizing
- `BZ=F`
- VaR tests
- GJR-GARCH

---

## 18. References

Alquist, R., Kilian, L. & Vigfusson, R. (2013). Forecasting the price of oil. *Handbook of Economic Forecasting*, vol. 2.
Ang, A. & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*.
Caldara, D. & Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review*.
Clark, T. & West, K. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*.
Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
Diebold, F., Lee, J.-H. & Weinbach, G. (1994). Regime switching with time-varying transition probabilities.
Filardo, A. (1994). Business-cycle phases and their transitional dynamics. *Journal of Business & Economic Statistics*.
Fong, W. M. & See, K. H. (2002). A Markov switching model of the conditional volatility of crude oil futures prices. *Energy Economics*.
Hamilton, J. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*.
Herrera, A. M., Hu, L. & Pastor, D. (2018). Forecasting crude oil price volatility. *International Journal of Forecasting*.
Iacoviello, M. & Tong, J. (2026). The AI-GPR index: Measuring geopolitical risk using artificial intelligence. Working paper.
Liu, J., Ma, F., Tang, Y. & Zhang, Y. (2019). Geopolitical risk and oil volatility: A new insight. *Energy Economics*.
Mohaddes, K. & Pesaran, M. H. (2016). Country-specific oil supply shocks and the global economy. *Energy Economics*.
Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
Pesaran, M. H. & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*.
Qian, L., Zeng, Q. & Li, T. (2022). Geopolitical risk and oil price volatility: Evidence from Markov-switching model. *International Review of Economics & Finance*.
Wang, L., Ma, F., Hao, J. & Gao, X. (2021). Forecasting crude oil volatility with geopolitical risk. *International Review of Financial Analysis*, 76.

Further reading (from the first plan, not in the deck):
- Giacomini & White (2006), *Econometrica*: conditional predictive ability; we use their test.
- Leetaru & Schrodt (2013): GDELT; cite it for the data.
- Kilian, Plante & Richter (2026), Dallas Fed WP 2615: the 2026 war.
- Plakandaras, Gupta & Wong (2019), *Resources Policy*.
- Noguera-Santaella (2016).
- Kilian (2009), *AER*.
- Känzig (2021), *AER*.
