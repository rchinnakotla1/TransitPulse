# The TransitPulse Website (Phase 6)

## What
A multi-page **Streamlit** website (`app/`) that puts every result of the project in front of the three user
groups: dispatchers, service planners and rider-communications staff. Run it with

```
.\.venv\Scripts\streamlit run app/Home.py
```

## Why a website (and why Streamlit)
* Agency staff do not open notebooks; they need a tool that answers *their* question in a few clicks.
* Streamlit is pure Python, so the website calls exactly the same code as the notebooks (`src/`), with no second
  implementation that could drift. It is free, runs locally or on Streamlit Community Cloud, and supports maps,
  Plotly charts, auto-refreshing fragments (live mode) and an in-process test harness (`AppTest`).
* Alternatives considered: Dash (more boilerplate for the same result), a React front end + API (far more work,
  no analytical benefit for a graduate project), Power BI / Tableau (cannot run the model or the LLM live).

## Design principles
| Principle | How it is implemented |
|---|---|
| Fast start, no 10 M-row loads | `python -m src.app_data` pre-computes small tables in `reports/app/` (reliability by route × direction × hour × day type × month, 442 timepoint stops, daily series, two December weeks of test events, live-mode profiles, per-route AUC); all loaders are cached |
| One deployed model | XGBoost in both modes (best in Chapter 6); raw probabilities are shown because they are calibrated (planning 26.1 % predicted vs 25.1 % observed) |
| Agreed operating points | presets from Chapter 6: rider alert planning ≥ 0.60, planner watch-list ≥ 0.30, dispatcher alert operational ≥ 0.50, plus the best-F1 thresholds frozen on validation |
| Honest uncertainty | per-route test AUC shown as a "trust" label (low < 0.70 ≤ medium < 0.75 ≤ good < 0.90 ≤ very good); what-if results labelled "association, not causal effect" |
| LLM failure is visible | sidebar shows the OpenAI key status (connected / invalid / expired / quota / network) with a re-check button; any page that uses the LLM shows a warning banner and falls back to the template narrative; details in `logs/llm.log` |
| Every page works offline | only Live Mode needs the internet; when the MBTA API fails it shows a warning instead of an error |

## Pages
| # | Page | User | Content |
|---|---|---|---|
| — | Home | all | purpose, headline metrics, page guide, per-route trust table |
| 1 | Reliability Explorer | planner | late rate by route, direction, hour, month; timepoint map; daily trend |
| 2 | Model Comparison | analyst | the six models with 95 % confidence intervals, both modes |
| 3 | Predict & Explain | all | pick a real December 2025 stop event → risk, SHAP reasons, OpenAI explanation for the chosen audience, what-if curves (extra running time, weather, live delay) |
| 4 | Explainability | planner | global importance (feature and group), SHAP by time of day, readable rules with fidelity, stability XGB vs LGB |
| 5 | Propagation & Bunching | planner, dispatcher | delay growth along the trip, recovery rates, bunching by route and hour, bunching model |
| 6 | Planner Insights | planner | chronic segments, running-time suggestions, severity and delay-minutes models, route × period risk |
| 7 | Live Mode | dispatcher | buses in service now scored by the operational model, map, explanation of one bus (Chapter 10) |
| 8 | Data Sources | all | every dataset, why it is used, how it is joined, pipeline diagram |

## Testing
`tests/test_app.py` runs **every page headless** with `streamlit.testing.v1.AppTest` and fails on any exception.
The network is disabled inside the test (all `requests.get` calls raise, the OpenAI key is blanked), which proves
the offline guarantee and the LLM fallback at the same time. The same file tests live-mode error handling with
mocked HTTP 429 / 403 / network failure / empty responses and the bus-ahead logic.

Result: **all 9 pages render without exceptions** and all 7 live-mode tests pass (16 tests in this file; 62 in the whole suite `pytest tests`). A real server was also started
(`streamlit run app/Home.py`, health endpoint `ok`, HTTP 200), and the Planner and Live pages were run once against the real MBTA
API without exceptions. All `use_container_width` arguments were replaced by `width="stretch"` (deprecated in Streamlit 1.66).

## Decisions
* **D27 — pre-computed app tables.** Loading the 9.8 M-row feature files would take minutes and > 4 GB of RAM;
  aggregated tables (largest: 381 k December events) keep start-up to seconds and make the site deployable on a
  free tier.
* **D28 — raw XGBoost probabilities, no recalibration.** Calibration was already within 1 pp on the test months;
  recalibrating would add a step without benefit, and showing probabilities lets each user pick a threshold.
