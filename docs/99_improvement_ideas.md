# Ideas to Strengthen the Project

Grouped by effort. ⭐ = recommended for the final submission.

## A. Data & features
1. ⭐ **School-calendar feature** — EDA shows Jul–Aug and late Dec are much better; add Boston Public Schools + university term dates (`is_school_day`).
2. ⭐ **Weather** — hourly precipitation, snowfall, temperature, wind from Meteostat (Logan Airport). Explain spike days like 2025-04-04.
3. ⭐ **Lagged / upstream delay** — previous bus's delay on the same route-stop and this trip's delay at the previous timepoint. Gives two model modes: *planning* (no live info) vs *operational* (upstream delay known). Compare both: a really nice talking point.
4. **Events calendar** — Red Sox (Fenway), Celtics/Bruins (TD Garden), Boston Marathon (Apr 21, 2025), July 4th; flag routes passing near venues.
5. **GTFS static enrichment** — stop names, lat/lon, distance between timepoints, scheduled running time per segment → *schedule tightness* feature (scheduled minutes per km).
6. **Service-change awareness** — MBTA Bus Network Redesign phases in 2025; flag rating periods (pick-ups) so the model knows when schedules changed.
7. **Ridership proxy** — MBTA bus ridership by trip/season dataset (also on the same portal) to capture crowding/dwell time.
7b. ⭐ **Better event representation** (*from the Phase 3 negative result*) — the binary "event within 1 km" flag hurts on autumn event rows (−0.010 AUC). Use event **end time**, attendance, venue-specific effects, and whether the trip heads *away* from the venue in the hour after the event.
7c. ⭐ **Schedule-change ("rating") flag and days-since-new-schedule** (*from walk-forward*) — September, the first month of the new schedule, is the only weak month (AUC 0.715). Add `days_since_rating_start` and retrain 2–3 weeks into each new rating.

## B. Modelling
8. ⭐ **Proper class-imbalance handling** — `class_weight` / `scale_pos_weight`, threshold tuning on the validation month to maximise F1 or a cost-weighted metric.
9. ⭐ **Probability calibration** (isotonic / Platt) + reliability diagram, so "70 % chance of delay" on the dashboard is honest.
10. **LightGBM / CatBoost** as extra boosting baselines (CatBoost handles high-cardinality `stop_id` natively).
11. **Hyper-parameter search** with Optuna using a rolling time-series CV (expanding window, month-by-month).
12. **Concept drift check** — train on Jan–Jun and evaluate month by month; plot F1 per month.
13. **Fairness / equity lens** — compare reliability and model error across routes serving different neighbourhoods (MBTA equity-route designation). Strong discussion section.

## C. Explainability (XAI)
14. ⭐ **SHAP**: global beeswarm, dependence plots (hour × route), local waterfall per prediction.
15. ⭐ **Decision-tree rule extraction** — depth-4 surrogate tree on XGBoost predictions → human-readable rules ("PM peak AND route 28 inbound AND midpoint ⇒ 62 % late").
16. ⭐ **LLM narrator** — feed top-k SHAP contributions + context (route, time, weather) to an LLM with a strict prompt → 2-sentence rider-friendly explanation. Ground it: the LLM may only use the numbers it is given; add a faithfulness check that every factor it mentions is in the SHAP top-k.
17. **Counterfactual explanations** (DiCE): "this trip would likely be on time if it were scheduled 1 h earlier / on a dry day".
18. **Rider vs planner explanations** — two LLM prompt styles from the same SHAP output.

## D. Dashboard & deployment
19. ⭐ Streamlit pages: Reliability Explorer (heatmaps/maps), Model Comparison (metrics, ROC/PR curves, confusion matrices), Predict & Explain (form → probability + SHAP + LLM text), Bunching monitor.
20. **Map view** of stops coloured by delay rate (pydeck/plotly with GTFS lat/lon).
21. **Real-time mode** — MBTA V3 API (`/predictions`, `/vehicles`) polled every 30 s, scored on the fly.
22. Package model with a small FastAPI endpoint + Docker for a "production-ready" feel.

## E. Extensions from the proposal (prioritised)
23. ⭐ **Severity (multi-class)** — on-time / minor / moderate / major already labelled in the cleaned data (37 / 34 / 18 / 11 %).
24. **Delay regression** — predict minutes with XGBoost + quantile regression (P50/P90 → "likely 4–9 min late").
25. **Propagation & bunching** — sequence model along timepoints of a trip, plus classifier for "next arrival bunched" on headway routes.

## F. Report polish
26. Problem framing with MBTA's own reliability standard (headway vs schedule), limitations section (timepoints only, AVL gaps, missing = cancelled ambiguity), and a business-value section (which operational action each prediction enables).
