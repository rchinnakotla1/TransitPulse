# Introduction

## Problem
A bus that is more than five minutes late is the most common complaint of transit riders and the measure the MBTA
uses for its bus reliability standard at timepoint stops. In 2025, **28.8 %** of the 9.82 million timepoint stop
events on the 19 MBTA bus routes studied here were more than five minutes late. Agencies already publish
countdown predictions (MBTA's are 81–87 % accurate), but these answer only "when will my bus arrive?". They do not
tell a dispatcher **which buses are at risk and why**, they do not tell a planner **which delays are predictable
and recurring**, and they give a rider communications team no plain-language reason to share.

## Objective
TransitPulse is a machine-learning system that predicts whether a bus will be **more than five minutes late at a
timepoint** and **explains why**, for three groups of transit-agency users:

| User | Question | TransitPulse answer |
|---|---|---|
| Dispatcher | Which buses on the road now will be > 5 min late at their next timepoint? | operational model + live mode, alert threshold 0.50 (precision 0.93) |
| Service planner | Where and when is lateness predictable and recurring? Which segments need more running time? | planning model, SHAP by group and period, chronic segments, running-time suggestions |
| Rider communications | Can we warn riders a day ahead with a reason? | planning model at 0.60 (precision 0.70) + fact-checked OpenAI narrative |

## Research questions
1. **RQ1 – Predictability.** How well can > 5-minute lateness be predicted a day ahead (planning mode) and in real
   time (operational mode), compared with the agency's own historical average?
2. **RQ2 – Value of external data.** Do weather, calendars, events, schedule design (GTFS) and ridership add
   predictive power beyond route, stop and time?
3. **RQ3 – Explanation.** Which factors drive lateness, are the explanations faithful and stable, and can a large
   language model turn them into correct plain-language text?
4. **RQ4 – Operations.** Can the model run on live data, and what further questions (how late, does delay
   propagate, do buses bunch, which segments are chronic) can the same pipeline answer?

## Approach
The project was run as a sequence of phases, each with code (`src/`), an executed notebook
(What → Why → How → Result → Finding → Decision) and a chapter of this report:

| Phase | Content | Chapter |
|---|---|---|
| P0 | Data source survey, ingestion, cleaning, exploratory analysis | 2, 3 |
| P1 | External data: GTFS, weather, calendars, events, ridership, 2024 drift | 4 |
| P2 | Feature engineering, leakage-safe history, planning vs operational modes, time split | 5 |
| P3 | Six models (baseline, LR, DT, RF, XGBoost, LightGBM), tuning, ablation, walk-forward, drift | 6 |
| P4 | Explainability: SHAP, rules, what-if, stability, OpenAI narratives with a fact check | 7 |
| P5 | Extensions: severity, delay minutes, propagation, bunching, chronic segments | 8 |
| P6 | Streamlit website for agency users | 9 |
| P7 | Live mode on the MBTA V3 API | 10 |
| P8 | This report | — |

Every non-obvious choice is recorded with its reason in the decision log (Appendix A).

## Scope
* **City / mode:** Boston, MBTA buses; 19 routes – the 14 key bus routes 1, 15, 22, 23, 28, 32, 39, 57, 66, 71, 73,
  77, 111, 116 plus 34E, 70, 240, 8 and 44.
* **Period:** calendar year 2025 for modelling (train Jan–Aug, validation Sep–Oct, test Nov–Dec), 2024 for the
  history seed and a drift study.
* **Target:** `delayed = 1` if actual − scheduled time at a timepoint > 300 s.
* **Data:** only official or well-established free sources (MBTA open data, GTFS, Open-Meteo ERA5, `holidays`,
  Boston Public Schools calendar, MLB/NHL schedules, MBTA ridership); no paid data and no scraping of restricted
  sites.
