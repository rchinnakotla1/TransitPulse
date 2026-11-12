# Phase 3: Modelling and Evaluation

Full executed analysis: [notebooks/05_Models.ipynb](../notebooks/05_Models.ipynb) · code `src/models/common.py`, `src/models/train.py`,
`src/models/experiments.py` · results `reports/metrics_{planning,operational}.csv`, `reports/exp_*.csv`, `reports/preds/` · figures `reports/figures/p3_*.png`
· models `models/{B0,LR,DT,RF,XGB,LGB}_{planning,operational}`

## 1. Protocol

| Step | Data | Decided here |
|---|---|---|
| Tune | train (Jan–Aug 2025) → validation (Sep–Oct) | hyper-parameters; decision threshold (max F1) |
| Refit | train + validation (Jan–Oct, 83 % of 2025) | nothing new; same settings |
| Test | Nov–Dec, scored once | ROC-AUC, PR-AUC, Brier, P/R/F1, 95 % **day-block bootstrap** CIs (200 resamples of whole service days) |

| Model | Tuning | Training rows |
|---|---|---|
| B0 | none: `hist_stop_hour`; operational adds the rule "late at previous timepoint stays late" | n/a |
| Logistic regression | C ∈ {0.01, 0.1, 1}; one-hot + scaled; **not** class-weighted (keeps probabilities calibrated; the threshold handles imbalance) | 2 M stratified |
| Decision tree | depth ∈ {6, 10, 14, 18}, leaf ≥ 200 | 2 M |
| Random forest | 150 trees, depth ∈ {14, 22}, leaf ≥ 50, √features | 2 M |
| XGBoost | Optuna TPE, 15 trials on 1 M / 300 k, early stopping on PR-AUC; refit with 1.15 × trees | all rows |
| LightGBM | 3-point grid (num_leaves 63/127/255, min_child_samples 50/200) on 1 M / 300 k, early stopping; refit 1.15 × trees | all rows |

*Deviation from the plan:* the plan said "class-balanced LR". Threshold tuning was chosen instead, because the dashboard shows probabilities
and balancing would inflate them.

## 2. Results (test, Nov–Dec 2025)

| Mode | Model | ROC-AUC [95 % CI] | PR-AUC [95 % CI] | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Planning | B0 | 0.657 [0.651, 0.665] | 0.378 [0.351, 0.406] | 0.442 | 0.297 | 0.857 |
| | LR | 0.720 [0.714, 0.727] | 0.466 [0.437, 0.490] | 0.493 | 0.388 | 0.673 |
| | DT | 0.700 [0.692, 0.708] | 0.440 [0.414, 0.467] | 0.473 | 0.347 | 0.742 |
| | RF | 0.735 [0.727, 0.743] | 0.499 [0.469, 0.527] | 0.495 | 0.364 | 0.773 |
| | **XGB** | **0.755 [0.748, 0.762]** | **0.526 [0.500, 0.552]** | **0.525** | 0.434 | 0.666 |
| | LGB | 0.751 [0.744, 0.758] | 0.520 [0.495, 0.547] | 0.521 | 0.421 | 0.683 |
| Operational | B0 | 0.892 [0.889, 0.893] | 0.773 [0.761, 0.784] | 0.823 | 0.871 | 0.780 |
| | LR | 0.967 | 0.937 | 0.856 | 0.897 | 0.819 |
| | DT | 0.969 | 0.938 | 0.856 | 0.888 | 0.826 |
| | RF | 0.973 | 0.945 | 0.861 | 0.891 | 0.833 |
| | **XGB** | **0.978 [0.977, 0.979]** | **0.954 [0.950, 0.957]** | **0.875** | 0.916 | 0.838 |
| | LGB | 0.977 [0.976, 0.978] | 0.953 [0.949, 0.956] | 0.875 | 0.916 | 0.837 |

* XGBoost wins in both modes. McNemar vs RF: p < 10⁻¹⁰⁰. The AUC CIs do not overlap in planning.
* **LightGBM cross-check:** a second boosting library with only a 3-point grid lands inside XGBoost's confidence interval in both modes
  (planning 0.751 vs 0.755, operational 0.977 vs 0.978) while training in 4–7 min instead of 45–68 min. The result is therefore a property of
  gradient boosting on these features, not of one library or of lucky tuning. XGBoost stays the deployed model (best point estimate, already
  tuned and explained); LightGBM is the fast fallback for frequent retraining and is used in P4 to test whether explanations are stable.
* Planning gain over agency averages (B0): **+0.10 AUC, +0.15 PR-AUC (+39 %)**.
* Calibration: planning XGB predicts 26.1 % vs 25.1 % actual; operational within 0.2 pp. No recalibration needed.
* Planning test > validation (0.755 vs 0.726): the refit on Jan–Oct includes the autumn schedule (see walk-forward).

**Operating points:** rider alert = planning ≥ 0.60 (precision 0.70, 6 % of trips flagged) · planner watch-list = planning ≥ 0.30 (recall 0.61) ·
dispatcher alert = operational ≥ 0.50 (precision 0.93, recall 0.82).

**Where it fails:** reliable routes (111: AUC 0.68, 10.7 % late) are hardest, because their delays are one-off incidents. Chronically late routes (240, 73, 34E: 0.77–0.78)
are easiest. **Live value by position:** planning ≈ operational at the first timepoint (0.71 vs 0.73); operational reaches 0.98+ from the second.

## 3. Ablation: the central research question

XGBoost, fixed params, 2 M Jan–Aug sample; test ROC-AUC.

| Cumulative | AUC | Δ |  | Drop-one from full planning | AUC lost |
|---|---|---|---|---|---|
| history | 0.678 | | | temporal+calendar | **0.027** |
| + temporal & calendar | 0.722 | **+0.044** | | history | **0.018** |
| + daylight | 0.713 | −0.008 | | spatial | **0.010** |
| + weather | 0.715 | +0.002 | | gtfs / ridership / weather | ≤ 0.001 |
| + events | 0.714 | −0.001 | | daylight / events | −0.003 (removal helps) |
| + GTFS | 0.718 | +0.004 | | | |
| + ridership | 0.720 | +0.003 | | | |
| + spatial | 0.730 | +0.010 | | | |
| + operational | 0.977 | **+0.247** | | | |

**Episodic check** (Sep–Dec rows each source should affect): weather adds **+0.007** AUC on rain rows and +0.005 on snow rows. Events **reduce** AUC by
0.010 on event rows. The binary flag does not transfer from summer Fenway games to autumn TD Garden events. This is a negative result; the features are kept
(removing them based on the test set would be test-set selection), and a better event representation is listed in `99_improvement_ideas.md`.

**Answer.** Day-ahead delay risk is driven mainly by recurring **time × place** patterns. External data refines the prediction where it applies
(weather on bad-weather days) but does not transform it. Live operational data is decisive.

## 4. Robustness experiments

| Question | Experiment | Result | Decision |
|---|---|---|---|
| Are we under-training with 8 months / an 80-20-style split? | learning curve 5 → 100 % of train | valid AUC 0.711 → 0.723 → 0.726 → **0.7265**; +0.0006 from 3.2 M to 6.4 M rows | no; volume is saturated |
| Does training on Jan–Aug miss autumn conditions? | walk-forward May–Dec | 0.74–0.75 every month **except September 0.715** (new schedule + school start); Oct–Dec 0.752–0.754 once September is in training | retrain monthly **and** after each schedule change |
| Would a 12-month window (with last autumn, DST, holidays) help? | A Jan–Aug 2025 vs B Sep 2024–Aug 2025 | Nov–Dec AUC 0.732 (A) vs **0.721** (B); 3 M-row B: 0.721 | **no**: older regime data hurts; recency > seasonal coverage |
| How fast does a model decay? | train 2024, score each 2025 month | AUC 0.64–0.70, 0.07–0.10 below retrained models; calibration still good (rolling rates) | rolling features keep probabilities valid; ranking needs retraining |

## 5. Reproduce

```powershell
.\.venv\Scripts\python -m src.models.train planning operational     # ≈ 2.5 h on 12 cores (re-run a subset: ... planning XGB)
.\.venv\Scripts\python -m src.models.experiments                    # ≈ 70 min (ablation learning walkfwd window drift episodic)
cd notebooks; ..\.venv\Scripts\python build_models_notebook.py
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 05_Models.ipynb
```
