# Conclusion

## Answers to the research questions

**RQ1 – Predictability.** Lateness of more than five minutes is *moderately* predictable a day ahead and *highly* predictable in real time.
On the untouched November–December test months, XGBoost reaches ROC-AUC **0.755** in planning mode, against 0.657 for the
agency-style historical average (+0.10 AUC, +39 % PR-AUC). In operational mode it reaches **0.978**; the dispatcher alert at 0.50
has precision 0.93 and recall 0.82. Six model families were compared. LightGBM lands inside XGBoost's confidence interval, so the
result reflects gradient boosting on these features, not one library. Probabilities are calibrated (26.1 % predicted vs 25.1 %
observed). Learning curves, walk-forward months and a 12-month-window experiment show the January–August training split is not
under-trained. Only September, with its new schedule and school start, is harder (AUC 0.715).

**RQ2 – Value of external data.** External sources add **little on average but help where they apply**. In the ablation, time and
calendar (+0.044) and place (+0.010) carry the day-ahead signal. Weather, GTFS and ridership each add ≤ 0.004 AUC overall, yet weather adds
+0.007 on rain rows and +0.005 on snow rows. Event flags did not transfer from summer baseball to autumn arena events (−0.010 on event
rows); this negative result is reported, not hidden. Live operational data dominates everything (+0.247 AUC).

**RQ3 – Explanation.** Day-ahead risk is a sum of many small, recurring effects. History is 32 % of |SHAP|, followed by time and place
at 16 % each, schedule tightness, and calendar. Real-time risk is mostly "already late upstream": the live features are 61 % of |SHAP|.
The explanations are **faithful** where it matters. A 14-leaf rule set reproduces the operational model (R² 0.89, AUC 0.951), whereas
planning needs per-prediction SHAP (rules reach only R² 0.38). They are **stable** across XGBoost and LightGBM: planning rank
correlation 0.98, operational group-level 0.88. GPT-4o-mini turns them into plain language under a numeric fact check. 5 % of
drafts were rejected for replacing the main factor, and 100 % of the texts shown are faithful. A missing or expired key falls back to
templates with a visible warning.

**RQ4 – Operations.** The same pipeline runs on live MBTA V3 data and answers the follow-up questions:

* **Live mode** scores every in-service bus at its next timepoint. In real-API tests, all buses that MBTA itself predicted to be more
  than 5 min late were flagged (p ≥ 0.91).
* **Propagation:** 88 % of late buses stay late, and a trip starting more than 10 min late ends late 93 % of the time, so dispatchers must act early.
* **Severity:** real-time QWK 0.864, with major delays found 80 % of the time at 93 % precision. A day ahead only 10 % of majors are found.
* **Minutes late:** real-time MAE 1.08 min (persistence 1.71), with conformal-calibrated 80 % intervals at 79.6 % coverage.
* **Bunching:** real-time PR-AUC 0.444, six times the base rate. It concentrates on frequent routes, late in the route and in the afternoon.
* **Chronic segments:** 29 of 371, stable over time (ρ = 0.84). Most *inherit* their delay rather than create it.

## Contributions

1. A reproducible, leakage-safe pipeline from raw MBTA arrival/departure files plus five free external sources to two model modes. It has
   a strict out-of-time protocol and 31 logged decisions.
2. An honest assessment of external data, including negative results (events) and a season proxy (temperature) that SHAP alone would
   have presented as a cause.
3. An explanation layer evaluated for faithfulness and stability, and an LLM narrator that cannot publish unverified numbers.
4. Agency-oriented extensions: severity, calibrated minutes, propagation, bunching, chronic segments. Each is compared with a naive
   baseline from the same information set.
5. A working Streamlit website with live mode, error handling for API and key failures, and 62 automated tests.

## Lessons learned

* **Baselines catch bugs.** A bunching baseline with AUC 0.305 (below random) led to the discovery that the bus-ahead feature skips a
  leader that is still en route. This also exposed a train/serve skew in live mode, which was then fixed (decision 30).
* **Check the training definition before serving.** Two live-mode fixes (decisions 26 and 30) came from comparing live feature values
  with their training definitions, not from accuracy metrics.
* **Time split beats random split.** Random 80/20 splits would have leaked recent history; the walk-forward analysis showed exactly which
  months are hard.

## Limitations

* One city, 19 routes and one year. September-type schedule changes need retraining, so monthly retraining is recommended.
* Live features approximate their training definitions: slack uses "now" as a lower bound, and the passing time of the bus ahead is estimated.
  The live model has been checked against MBTA's own predictions, not against observed arrivals.
* SHAP, what-if and propagation results are associations, not causal effects.
* Narrative usefulness was not rated by real agency staff; a rubric is provided for that study.

## Future work

See Chapter "Future Work". Priorities are: a holding-strategy simulation for bunching alerts; a better event representation (venue
capacity, end time); retraining on each new schedule ("pick"); a live accuracy log against observed arrivals; and a user study with MBTA
dispatchers and planners.
