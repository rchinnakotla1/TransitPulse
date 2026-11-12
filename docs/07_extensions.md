# Phase 5: Extensions — Propagation, Chronic Segments, Severity, Minutes Late and Bunching

Full executed analysis: [notebooks/07_Extensions.ipynb](../notebooks/07_Extensions.ipynb) · code `src/extensions/` (`propagation.py`,
`chronic.py`, `severity.py`, `regression.py`, `conformal.py`, `bunching.py`, `common.py`) · results `reports/ext/` · figures
`reports/figures/p5_*.png` · logs `logs/ext_*.log`

## 1. Why this phase

Phases 3 and 4 answer one question, *will this bus be more than 5 minutes late, and why?* A transit agency also asks follow-up questions.
Each user group needs a different one:

| # | Question | User | Method |
|---|---|---|---|
| 1 | How does delay **spread** along a trip and from bus to bus? | dispatcher | descriptive propagation analysis on all 2025 events |
| 2 | Which segments are **chronically** late, and should their running time change? | planner | stable-over-time segment ranking + added-delay decomposition |
| 3 | **How late**: on time / minor / moderate / major? | dispatcher, rider comms | 4-class XGBoost (ordinal metrics) |
| 4 | **How many minutes** late, with an honest interval? | rider comms | XGBoost median + quantile interval + conformal calibration |
| 5 | Will buses **bunch**? | dispatcher, planner | headway labels + XGBoost |

## 2. Methods and protocol

All extension models follow one protocol (decision 19). Each is fit on a stratified **2 M-row sample of train** (Jan–Aug), early-stopped
on a **300 k validation** sample (Sep–Oct) and scored **once** on the full **test** months (Nov–Dec). Hyper-parameters are reused from the
Phase 3 Optuna search, with no new tuning. The numbers are therefore slightly conservative but fully out-of-time. Every model is compared with a
naive baseline that uses the same information: the stop's usual delay for planning, and persistence (the delay at the previous
timepoint) for operational.

| Extension | Label | Model | Metrics |
|---|---|---|---|
| Severity | on time < 1 min, minor 1–5, moderate 5–10, major > 10 min | XGBoost `multi:softprob` | accuracy, macro-F1, quadratic weighted kappa (QWK), recall/precision of *major* |
| Minutes | delay clipped to [−10, 60] min | XGBoost absolute-error (median) + multi-quantile 10 %/90 % | MAE, RMSE, % within 2 / 5 min, interval coverage and width |
| Conformal | — | CQR (Romano et al., 2019) on the validation months | test coverage, width |
| Bunching | actual headway < 25 % of scheduled (decision 20) | XGBoost binary | ROC-AUC, PR-AUC (base rate 7.3 %) |

## 3. Results

### 3.1 Delay propagation

* The delay at one timepoint correlates **0.92** with the delay at the next. Once a bus is more than 5 min late it stays late at the next timepoint
  **88.1 %** of the time; only **11.9 %** recover. An on-time bus becomes late only 6.8 % of the time.
* A trip that leaves its first timepoint more than 10 min late arrives late at the end **93 %** of the time (5–10 min: 67 %; on time: 9 %).
* Delay builds along the route: median 1.1 min at 10 % of the route rises to 3.1 min at 70–80 %. It falls back at the last decile because of
  terminal padding.
* Recovery is lowest on route 39 (6.0 %), 34E (7.9 %) and 28 (9.4 %), and highest on 116 (24.9 %).
* The bus's own previous timepoint (r = 0.88–0.95) explains its delay far better than the bus ahead at the same stop (r = 0.27–0.42).

Figures: p5_propagation.png, p5_propagation_recovery.png.

### 3.2 Chronic segments

* **29 of 371** segments are chronic: late at ≥ 1.5 × the network average in ≥ 80 % of the months Jan–Oct. The ranking is stable out of time
  (Spearman ρ = **0.84** between Jan–Oct and Nov–Dec). On the test months chronic segments are late **47.9 %** of the time vs 25.2 %.
* They are concentrated on routes 28 (10), 44 (8) and 39 (6).
* **Inherited, not created:** median delay at the timepoint is 7.0 min, but the delay *added* on the segment itself is only **0.23 min**.
  Only 2 of 29 segments add ≥ 1 min. Of 171 running-time suggestions (segment × period), 20 are ≥ 1 min. The largest is 2.3 min (route 44
  inbound, stop 396, PM peak).

Figure: p5_chronic.png.

### 3.3 Severity (4 classes)

| Mode | Model | Accuracy | Macro-F1 | QWK | Recall major | Precision major |
|---|---|---|---|---|---|---|
| planning | B0 usual delay class | 0.332 | 0.239 | 0.171 | 0.016 | 0.262 |
| planning | XGB | **0.506** | **0.360** | **0.292** | 0.103 | 0.341 |
| operational | B1 persistence | 0.739 | 0.737 | 0.801 | 0.763 | 0.871 |
| operational | XGB | **0.819** | **0.814** | **0.864** | **0.801** | **0.928** |

Consistency check: P(moderate) + P(major), used as a "> 5 min" score, gives AUC 0.732 / **0.976**. That is close to the binary models
(0.755 / 0.978). Figure: p5_severity_confusion.png.

### 3.4 Minutes late and calibrated intervals

| Mode | Model | MAE | RMSE | Within 2 min | Within 5 min | 80 % interval raw → CQR (median width) |
|---|---|---|---|---|---|---|
| planning | B0 usual delay | 4.39 | 5.90 | 31.4 % | 67.2 % | — |
| planning | XGB median | **3.59** | **5.31** | **43.5 %** | **76.9 %** | 75.5 % → **78.9 %** (10.3 → 11.0 min) |
| operational | B1 persistence | 1.71 | 2.95 | 74.5 % | 94.5 % | — |
| operational | XGB median | **1.08** | **2.09** | **89.0 %** | **97.3 %** | 78.3 % → **79.6 %** (2.91 → 2.98 min) |

* The planning model shrinks towards typical delays. Its bias is +2.8 min for on-time buses and −9.9 / −21.0 min for buses 10–20 / 20+ min
  late. The operational MAE stays below 0.9 min up to 5 min of delay and is 4.2 min for 20+ min.
* **Conformal calibration (CQR).** The 80 % quantile Q of the conformity score max(lo − y, y − hi) on Sep–Oct is added to both interval
  ends: +0.33 min for planning and +0.03 min for operational. Test coverage rises to 78.9 % / 79.6 %. The remaining shortfall is the
  autumn → winter shift, which the exchangeability guarantee does not cover. By route, planning coverage stays low on 15 (70.6 %) and
  57 (72.5 %).

Figures: p5_regression.png, p5_conformal.png.

### 3.5 Bus bunching

9.30 M headway events in 2025 are 7.6 % bunched and 13.2 % big gaps. The test base rate is 7.3 %.

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| B0 gap ratio, first assumed sign (−ratio) | 0.305 | 0.071 |
| B0 gap ratio (+ratio, after diagnosis) | 0.695 | 0.228 |
| planning XGB | 0.726 | 0.158 |
| operational XGB | **0.915** | **0.444** |

* **Where and when:** frequent routes bunch most: 1 (10.4 %), 32 (10.2 %), 66 (10.0 %). Low-frequency routes (240, 34E, 8, 44) are
  around 1 %. The rate grows along the route from 2.6 % to 10.4 % and peaks at 10–11 % between 14:00 and 18:00.
* **Diagnosis of the inverted baseline (decision 30).** The first baseline was below 0.5, which signals a wrong sign, not a useless feature.
  `prev_bus_gap_ratio` is the time since the last bus that has **already passed** the target stop. A leader that is close is still en route
  and is skipped, so the feature then describes the bus two ahead. On 1.6 M test rows the relation is U-shaped:

  | prev_bus_gap_ratio | ≤ 0.25 | 0.25–1 | 1–1.5 | 1.5–2 | > 2 |
  |---|---|---|---|---|---|
  | bunched at target stop | 10.3 % | 2.4–2.8 % | 7.1–15.7 % | 27.6 % | 35.5–36.4 % |

  The trees learn this shape; a single-feature threshold cannot. The same analysis revealed a **train/serve skew** in live mode. Live mode
  had used the gap to the direct leader, which is wrong exactly for bunched pairs, and it was fixed (Chapter 10).

Figures: p5_bunching.png, p5_bunching_gap_ratio.png.

## 4. Discussion

The extensions repeat the main result of the thesis at a finer resolution. With **real-time** information every question is answerable
with good accuracy: late or not, how late (QWK 0.86), how many minutes (MAE about 1 min), and bunching (PR-AUC 6 × base). **Day-ahead**
features answer reliably only *where and when* risk is high. They cannot foresee the incidents that cause large delays, which is why
planning severity finds just 10 % of major delays and planning regression shrinks towards the mean. Propagation explains why: delay is
persistent and accumulates, so the most informative signal is the bus's own current delay.

Chronic segments show how a planning recommendation can mislead without decomposition. Most chronic segments *inherit* lateness. Adding
running time there would treat a symptom, so the website shows mean delay and added delay side by side.

## 5. Limitations

* Extension models use the decision-19 protocol (2 M sample, no retuning). The operational severity model reached the 1,000-tree cap.
* Conformal coverage is guaranteed only under exchangeability. Season shift leaves planning intervals about 1 point short overall, and
  about 10 points short on route 15.
* Bunching labels need both an actual and a scheduled headway (9.3 M of the events). Holding strategies were not simulated.
* Propagation and chronic results are associations, not causal effects.

## 6. Decisions carried forward

| # | Decision |
|---|---|
| 19 | Extension protocol: 2 M train sample, 300 k validation, full test, reuse P3 parameters |
| 20 | Bunching label: headway < 25 % of scheduled; big gap > 150 % |
| 29 | Severity shown for the operational mode only; 1,000-tree cap kept |
| 30 | `prev_bus_gap_ratio` semantics: baseline uses +ratio; live mode rebuilds the training definition |
| 31 | CQR-calibrated intervals; operational median + interval for rider messages; routes 15 and 57 flagged |
