# Verified findings for notebooks/07_Extensions.ipynb (numbers checked against reports/ext/*.csv and the executed notebook)

prop = """
**Result**
* Delay at one timepoint correlates **0.92** with the delay at the next timepoint of the same trip.
* Once a bus is > 5 min late, it is still late at the next timepoint **88.1 %** of the time; only **11.9 %** recover.
  An on-time bus becomes late at the next timepoint only 6.8 % of the time.
* A trip that leaves its first timepoint **> 10 min late arrives late at the end 93 %** of the time (5-10 min: 67 %; on time: 9 %).
* Delay builds over the trip: median delay 1.1 min at 10 % of the route -> 3.1 min at 70-80 %; late rate 16 % -> 37 %. At the last decile it
  falls back to 1.5 min / 29 % because the terminal has schedule padding.
* Recovery is lowest on route **39 (6.0 %)**, 34E (7.9 %) and 28 (9.4 %), highest on 116 (24.9 %). By period it is lowest in the evening
  (10.5 %) and at night (10.3 %), highest early morning (14.6 %).
* The previous timepoint of the same bus explains a delay far better (r = 0.88-0.95) than the bus ahead at the same stop (r = 0.27-0.42).
  The bus-ahead link is strongest in the peaks (0.42 AM, 0.40 PM), when crowding spreads from bus to bus.

**Finding.** Delay is highly persistent and accumulates; recovery is the exception. This is the mechanism behind the operational model's
0.98 AUC and behind `prev_tp_delay` being its top SHAP feature. Recovery does not come from driving faster but from padding at the terminal.

**Decision.** (1) The dispatcher alert should act at the **first** timepoints of a trip: after a > 10 min late start, 93 % of trips end
late, so later intervention is too late. (2) The website shows recovery by route so planners see where padding is missing (39, 34E, 28).
(3) The live model uses the bus's own previous-timepoint delay as the main signal and the bus ahead only as context, as the correlations
suggest.
"""

chronic = """
**Result**
* **29 of 371** segments (7.8 %) are chronic: late >= 1.5 x the network average (29.6 %) in >= 80 % of the months Jan-Oct. Their late rate
  is 47-76 %.
* **Out-of-time check:** the segment ranking from Jan-Oct and Nov-Dec agrees with Spearman rho = **0.84**; 18 of the top 20 (90 %) are still
  in the top quartile in Nov-Dec. On the test months chronic segments are late **47.9 %** of the time vs **25.2 %** for all others.
* Concentrated on three routes: **28 (10 segments), 44 (8), 39 (6)**; 77 has 2, 57, 34E and 66 one each; the other 12 routes none.
* **Inherited, not created:** chronic segments have a median delay of 7.0 min at the timepoint, but the median delay *added* on the segment
  itself is only **0.23 min**; only 2 of 29 add >= 1 min. Non-chronic segments add -0.01 min (they slightly recover).
* Running-time suggestions (chronic segment x period, median added delay): 171 combinations, 106 with a positive addition, 20 with >= 1 min,
  maximum **2.3 min** (route 44 inbound at stop 396, PM peak). Most of the top suggestions are on route 44.

**Finding.** The fix-list is stable over time, so it is actionable. But most chronic segments are late because the bus *arrives* late
from upstream, not because the segment's own running time is too short. Adding running time at the chronic segment would treat the symptom.

**Decision.** The Planner Insights page shows both columns, *mean delay* and *added delay*, so a planner can separate **creating**
segments (fix the running time there, e.g. route 44 around stops 396 / 40001 / 64000) from **inheriting** segments (fix upstream or the trip
start). The suggested additions are reported per period and include zeros, so they are never inflated.
"""

severity = """
**Result** (test Nov-Dec, 1.69 M stop events; class shares 42 % on time, 33 % minor, 16 % moderate, 9 % major)

| Mode | Model | Accuracy | Macro-F1 | QW kappa | Recall major | Precision major |
|---|---|---|---|---|---|---|
| planning | B0 usual delay class | 0.332 | 0.239 | 0.171 | 0.016 | 0.262 |
| planning | XGB 4-class | **0.506** | **0.360** | **0.292** | 0.103 | 0.341 |
| operational | B1 persistence | 0.739 | 0.737 | 0.801 | 0.763 | 0.871 |
| operational | XGB 4-class | **0.819** | **0.814** | **0.864** | **0.801** | **0.928** |

* Day-ahead, XGB improves on the baseline (+0.17 accuracy, +0.12 macro-F1) but finds only **10 %** of major delays; most moderate and major
  events are predicted as on time or minor.
* With live information, XGB beats the strong persistence baseline on every metric: +8 points accuracy, +0.06 kappa, and major delays are
  found 80 % of the time with 93 % precision.
* Consistency check: P(moderate) + P(major) used as a "> 5 min" score gives AUC 0.732 (planning) and **0.976** (operational), close to the
  binary models (0.755 / 0.978, which were refitted on all of Jan-Oct).
* The operational model used all 1,000 trees (no early stop), i.e. it was still improving slowly.

**Finding.** *How late* is largely unpredictable a day ahead: large delays are caused by incidents the planning features cannot see.
In real time it is well predictable, because severity is persistent (the transition matrix in section 1 shows 88 % of major delays stay
major) and the model adds slack, the bus ahead and history on top of persistence.

**Decision.** Severity is shown for the **operational** mode only (dispatcher: "likely major delay"). For planning the website keeps the
binary probability and does not show a severity class, which would be misleading at 10 % recall of major delays. The 1,000-tree cap is
kept (decision 29): the planning run took 17 min and the operational run 20 min; more trees would cost time for a secondary model whose
binary counterpart already reaches AUC 0.978.
"""

regression = """
**Result** (test Nov-Dec, 1.69 M stop events, target clipped to [-10, 60] min)

| Mode | Model | MAE (min) | RMSE | Median AE | Within 2 min | Within 5 min | 80 % interval: coverage / median width |
|---|---|---|---|---|---|---|---|
| planning | B0 stop-hour usual delay | 4.39 | 5.90 | 3.34 | 31.4 % | 67.2 % | - |
| planning | XGB median | **3.59** | **5.31** | **2.40** | **43.5 %** | **76.9 %** | raw 75.5 % / 10.3 min; CQR **78.9 %** / 11.0 min |
| operational | B1 persistence | 1.71 | 2.95 | 1.03 | 74.5 % | 94.5 % | - |
| operational | XGB median | **1.08** | **2.09** | **0.64** | **89.0 %** | **97.3 %** | raw 78.3 % / 2.91 min; CQR **79.6 %** / 2.98 min |

* Day-ahead, XGB cuts the error of the "usual delay" baseline by 18 % (MAE 4.39 -> 3.59 min). With live data it cuts the error of
  persistence by 37 % (1.71 -> 1.08 min), and 89 % of predictions are within 2 minutes.
* Error by true delay band: the planning model **shrinks towards typical delays**. Its bias is +2.8 min for buses that were on time
  (< 1 min) and -9.9 / -21.0 min for buses 10-20 / 20+ min late. The operational model has MAE 0.84-0.87 min below 5 min and 4.2 min
  for 20+ min delays (bias -3.4).
* Intervals: the raw quantile interval under-covers. CQR calibration (section 4b) adds +0.33 min (planning) and +0.03 min (operational)
  to both ends and lifts coverage to 78.9 % / 79.6 %. It stays just below 80 % because the calibration months (autumn) and the test months
  (winter) differ. By route, planning coverage after CQR runs from 70.6 % (route 15) and 72.5 % (route 57) to 83.6 % (route 111);
  operational runs from 76.5 % (route 73) to 81.4 % (route 111).

**Finding.** Minutes-late is useful in real time (median error ~40 s, interval ~3 min wide). A day ahead, the median prediction is only a
"typical delay": it cannot anticipate large incidents (the same conclusion as severity), and the honest interval is about 11 min wide.

**Decision (31).** Rider messages use the **operational** median with the CQR-calibrated interval ("expected 4-7 min late"). Planning
shows the probability, not a minutes estimate. Routes 15 and 57 are flagged as having intervals that are too narrow day-ahead.
Conformal calibration was chosen over refitting because it costs seconds, needs no new training and has a known guarantee.
"""
bunching = """
**Result** (9.30 M headway events in 2025: 7.6 % bunched, 13.2 % big gap; test base rate 7.3 %)

| Model | ROC-AUC | PR-AUC (base 0.073) | Note |
|---|---|---|---|
| B0 gap-ratio feature, sign as first assumed (-ratio) | 0.305 | 0.071 | worse than random: wrong sign |
| B0 gap-ratio feature (+ratio, after diagnosis) | 0.695 | 0.228 | one feature, 99 % coverage |
| planning XGB | 0.726 | 0.158 | 89 trees |
| operational XGB | **0.915** | **0.444** | 802 trees; top feature `prev_bus_gap_ratio` |

* **Where and when:** the frequent routes bunch most: 1 (10.4 %), 32 (10.2 %), 66 (10.0 %). Low-frequency routes hardly bunch: 240, 34E, 8
  and 44 are around 1 %. Bunching grows steadily along the route, from 2.6 % in the first decile to 10.4 % in the last, the classic
  instability of bus headways. It peaks in the afternoon: 10-11 % between 14:00 and 18:00, against 8.5 % at 08:00.
* **Bunching and lateness:** bunched arrivals are more often late (35.5 % vs 23.9 %, mean delay 3.5 vs 2.4 min).
* **Diagnosis (5b, decision 30):** the gap-ratio feature has a **U-shaped** relation with bunching. Ratio <= 0.25 gives 10.3 % bunched,
  0.25-1 gives 2.4-2.8 %, and > 2 gives 35-36 % with a median headway ratio of 0.47. A large value means the direct leader has not
  passed the target stop yet (so it is close ahead), because the feature skips buses still en route. This explains the inverted
  baseline. It also exposed a **train/serve skew** in live mode, which measured the direct leader's gap; this was fixed (Chapter 10,
  `passed_bus_features`).

**Finding.** Bunching is predictable in real time: PR-AUC 0.444 is 6x the base rate, and the trees learn the U-shape that a single
threshold cannot. A day ahead it is only weakly predictable (PR-AUC 2x base), mainly *where* and *when* (route frequency, position,
afternoon peak).

**Decision.** Bunching stays an analysis and planning result, shown with the route, hour and position profiles. It is not a live alert
yet: at PR-AUC 0.44 a dispatcher alert would need a threshold study with MBTA staff, which we list as future work. The baseline was
corrected in `src/extensions/bunching.py` rather than silently reported as 1 - AUC.
"""
summary = """
## 6. Phase 5 summary

| Extension | Key result (test Nov-Dec) | Used in the website |
|---|---|---|
| Propagation | delay at consecutive timepoints correlates 0.92; once > 5 min late, 88 % stay late (only 12 % recover) | explains why the live model leans on `prev_tp_delay`; dispatcher guidance "act early" |
| Chronic segments | 29 of 371 segments are chronic (rank agreement train vs test 0.84); most *inherit* delay (median added 0.23 min) | Planner Insights: chronic list and running-time suggestions (max +2.3 min) |
| Severity | operational QWK 0.864, major delays recall 0.80 at precision 0.93; planning finds only 10 % of majors | Live Mode severity column (operational only, decision 29) |
| Minutes late | operational MAE 1.08 min (persistence 1.71), CQR interval coverage 79.6 % | Planner Insights table; rider message wording (decision 31) |
| Bunching | operational PR-AUC 0.444 (6x base); frequent routes, route end and afternoon bunch most | Planner Insights profiles; feature-definition fix in live mode (decision 30) |

**Overall finding.** The extensions repeat the main thesis result at a finer resolution. Real-time information makes *every* question
answerable (late or not, how late, how many minutes, bunched or not). Day-ahead features only give a reliable picture of *where and when*
risk is high. That is exactly the split between the planner pages (patterns) and the dispatcher pages (live predictions) of the website
built in Phase 6.

**Checks done in this phase:** every model has a naive baseline from the same information set; all numbers are on the untouched Nov-Dec
test months; one inverted baseline was found, diagnosed and fixed (decision 30); interval coverage was measured and calibrated
(decision 31).
"""
