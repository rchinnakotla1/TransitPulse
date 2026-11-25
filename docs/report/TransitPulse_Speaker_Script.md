# TransitPulse: Speaker Script (Slide by Slide)

> Target length: about 18–20 minutes plus demo and Q&A.
> Section-divider slides (3, 8, 11, 14, 16, 20, 23, 25, 29) take 5–10 seconds each. Read the transition line and move on.
> **Bold** marks the numbers worth stressing.

---

## Slide 1: Title, *TransitPulse* (≈45 s)

"Good [morning/afternoon], everyone. My project is **TransitPulse**. It predicts MBTA bus delays with machine learning, and it also explains them, using SHAP and plain-language narratives written by an LLM.

The three numbers on this slide sum it up. I worked with **9.8 million stop events** across **19 Boston bus routes**, and in real time my model reaches a **ROC-AUC of 0.978**.

I'll go through the whole pipeline: data, features, models, explainability, and finally the website and live mode, which I'll demo at the end."

## Slide 2: Agenda (≈20 s)

"Here's the plan. First the problem and objectives, then the data and what the exploratory analysis showed. After that come feature engineering, the models and how I evaluated them, explainability and the LLM narratives, a few extensions, the website and live mode, and finally the results and conclusions."

## Slide 3: Section 01, Problem & Objectives (≈5 s)

"Let's start with why bus lateness needs both a prediction and an explanation."

## Slide 4: The Problem (≈75 s)

"In 2025, **28.8 %** of all timepoint stop events on these routes were more than five minutes late. That's more than one in four.

Five minutes is not my number. It's the **MBTA's own reliability standard** at timepoint stops.

The MBTA does have countdown predictions, and they're about **81 to 87 % accurate**, but they only tell you *when* the bus will arrive. They never tell you *why* it's late.

Existing tools leave three groups without answers. Dispatchers want to know which buses on the road right now will be late at their next timepoint. Service planners want to know where lateness is predictable and recurring, and which segments need more running time. Rider communications teams want to warn riders a day ahead, with a reason.

So I set up a clear target: **delayed = 1 if actual minus scheduled time is more than 300 seconds**. The goal is to predict that and explain every single prediction."

## Slide 5: Objectives & Target Users (≈60 s)

"Each of those users gets a matching part of TransitPulse.

For the **dispatcher**, there's an operational model with live mode. When it alerts at a probability of 0.5 or higher, it's right **93 %** of the time.

For the **service planner**, there's a planning model, SHAP broken down by route group and time period, a list of chronic segments, and suggested running-time adjustments.

For **rider communications**, the planning model is used at a stricter 0.6 threshold, with 70 % precision, and paired with an OpenAI-generated narrative that is **fact-checked** before anyone sees it."

## Slide 6: Research Questions (≈60 s)

"I framed the work around four research questions.

**RQ1, Predictability.** How well can lateness be predicted a day ahead and in real time, compared with the agency's historical average?

**RQ2, Value of external data.** Do weather, calendars, events, GTFS schedule design and ridership add anything beyond route, stop and time?

**RQ3, Explanation.** Which factors drive lateness? Are the explanations faithful and stable? Can an LLM turn them into *correct* plain language?

**RQ4, Operations.** Can the system run on live data and answer the follow-up questions: how late, does delay propagate, bunching, chronic segments?

I'll come back to all four at the end."

## Slide 7: End-to-End Architecture (≈75 s)

"Here's the whole system, left to right.

I ingest **28.2 million rows** from 12 monthly CSVs, then clean and scope them down to 19 routes and 9.8 million events. Next I merge **six external data sources** and build **leakage-safe features** in two modes. I train **six model families** tuned with Optuna. On top of the models sits an explainability layer with SHAP, surrogate rules and an LLM narrator, followed by extensions such as severity and bunching. Everything is delivered through a Streamlit website with a live mode.

The code has four layers. The **data layer** handles ingestion, cleaning and external joins. The **ML layer** covers features, encoding and training. The **insight layer** holds SHAP, the rules, what-if analysis and the GPT-4o-mini narrator with its fact check. The **product layer** is the Streamlit app and the live MBTA API client, all backed by **62 automated tests**."

## Slide 8: Section 02, Data & Scope (≈5 s)

"Next, the data. I only used official, free and trusted sources."

## Slide 9: Core Dataset & Cleaning (≈60 s)

"The core dataset is the MBTA's bus arrival and departure times. For 2025 that's **28.2 million raw rows covering 229 routes**.

I scoped it to **19 routes**: the MBTA's 14 key routes plus five more, giving 288 timepoints and about 1.2 million one-way trips. Data from 2024 serves as a history seed and for a drift study.

The cleaning funnel is simple. Starting from 10.3 million rows on the scoped routes, I dropped about **500 thousand** events with no actual time, meaning cancelled or unobserved trips, which is 4.9 %. Then I removed just 909 extreme outliers. That leaves **9.82 million clean events, 95 % of the data**."

## Slide 10: External Data Integration (≈60 s)

"On top of the core data I joined six external sources.

**GTFS schedules** give scheduled running time, minutes per kilometre, slack and headways. In other words, how tight the schedule is.

**Hourly weather** comes from Open-Meteo, measured at Logan Airport: rain, snow, temperature and wind.

**Calendars** cover public holidays, Boston Public Schools days and university terms.

**Events** means Red Sox and Bruins games plus citywide events.

**Ridership** comes from automatic passenger counters.

I also use **MBTA's own prediction accuracy** as an external benchmark.

On the engineering side, every join checks that the row count is unchanged and writes a coverage audit, so no join can silently duplicate or drop data."

## Slide 11: Section 03, Exploratory Analysis (≈5 s)

"Before modelling, what does the raw data tell us about lateness?"

## Slide 12: When & Where Buses Run Late (≈60 s)

"A few key findings.

The target is **imbalanced**: 28.8 % late. The median delay is only +2.3 minutes, but the 95th percentile is **+14 minutes**.

**Time of day** matters a lot. About 12 % of events are late at 5 a.m., rising to **33–39 %** in the afternoon peak.

**Holidays** are much better, at about 20 % late, and weekends are worse in the afternoon.

**Seasonality** follows the school calendar.

**Route** matters too. Route 44 is late almost **48 %** of the time, while Route 111 is late only 16 %.

Most important for modelling, the linear correlations are weak. The patterns are non-linear and full of interactions, which pointed me towards **tree ensembles**."

## Slide 13: Delay Propagation Along a Trip (≈45 s)

"This slide is the key insight for real-time prediction: delay is sticky.

**93 %** of trips that leave more than 10 minutes late are still late at the last timepoint. **88 %** of late buses are still late at the next timepoint, and the correlation between start-of-trip and end-of-trip delay is 0.58.

So once a bus is late, it rarely recovers. That's why the bus's current state is so powerful as a feature, and why dispatchers need to act early."

## Slide 14: Section 04, Feature Engineering (≈5 s)

"Now feature engineering. The guiding principle was no leakage, with two prediction modes."

## Slide 15: Two Modes, One Leakage-Safe Pipeline (≈75 s)

"I built two modes from one pipeline.

**Planning mode** predicts a day ahead, so it only uses what's known before the day starts: time and calendar, location on the route, historical late rates for that stop and hour over the last four weeks, the weather forecast, schedule tightness, ridership and events.

**Operational mode** predicts in real time. It uses everything from planning plus the live state: delay at the previous timepoint, scheduled minutes remaining, delay at trip start, and the gap to the bus ahead.

To keep it honest I used a **strict time-based split with no shuffling**: train on January–August, validate on September–October, and test on November–December. I touched the test set **only once**.

Historical rates only ever use past data, and I have automated leakage tests that **fail the build** if any feature can see the future."

## Slide 16: Section 05, Models & Evaluation (≈5 s)

"I compared six model families across both modes."

## Slide 17: Model Comparison, Test Months (≈90 s)

"These are the results on the held-out November–December data.

The first row is the **historical baseline**, essentially what an agency would do with historical averages. Planning AUC is 0.657.

From there, performance climbs through logistic regression, decision tree and random forest. **XGBoost** comes out on top, with **0.755 AUC for planning and 0.978 for operational**. LightGBM is essentially tied.

Three things to highlight:
- **+39 % PR-AUC** over the agency-style baseline in planning mode.
- The dispatcher alert at p ≥ 0.5 has **0.93 precision and 0.82 recall**.
- The model is **well calibrated**: it predicts a 26.1 % late rate against 25.1 % observed, so the probabilities can be taken at face value.

XGBoost was tuned with Optuna, and the confidence intervals come from a day-block bootstrap. Because LightGBM falls inside XGBoost's confidence interval, the result reflects *gradient boosting on these features*, not one particular library."

## Slide 18: Robust Gains Over the Baseline (≈45 s)

"How should these numbers be read?

**A day ahead, lateness is moderately predictable.** Risk is the sum of many small recurring time-and-place patterns, with no single dominant cause.

**In real time it's highly predictable**, because the bus's own current delay decides most of it.

I also checked that the model isn't under-trained. Learning curves, walk-forward validation and 12-month-window experiments all agree. The one harder month is **September**, when the new schedule and the school start arrive together."

## Slide 19: RQ2, Does External Data Help? (≈60 s)

"This is the ablation study, answering RQ2.

**Time and calendar** add +0.044 AUC, the biggest gain among the planning features.

Weather, GTFS and ridership each add **0.004 or less** overall. That's small on average.

But it isn't the whole story. **On rainy rows, weather adds +0.007**, and +0.005 on snow. External data helps *where it applies*.

And the big one: **live operational data adds +0.247 AUC**. Real-time state dwarfs everything else."

## Slide 20: Section 06, Explainability & LLM (≈5 s)

"A prediction isn't enough. Next question: *why* is this bus at risk?"

## Slide 21: What Drives Lateness? (TreeSHAP) (≈60 s)

"For the global view I used TreeSHAP.

In **planning mode**, historical late rates account for **32 %** of total SHAP importance, followed by time and place at 16 % each, then schedule tightness and calendar.

In **operational mode**, live features make up **61 %**, mostly the bus being already late upstream.

Two checks on the explanations. Are they **faithful**? A 14-leaf rule set reproduces the operational model with R² 0.89 and AUC 0.951. Are they **stable**? XGBoost and LightGBM rank features with a correlation of **0.98**."

## Slide 22: Per-Prediction Explanation → Plain Language (≈60 s)

"For individual predictions, SHAP gives the reasons, and GPT-4o-mini turns them into plain language. Here's an example for rider communications:

*'Route 44 at this stop is very likely to run more than 5 minutes late. This stop is usually late at this hour and has been late often over the last four weeks, and the schedule here is tight.'*

LLMs can hallucinate, so I built a safety net. Every draft goes through a **numeric fact check against the SHAP values** before it's shown. About **5 % of drafts get rejected**, so **100 % of the texts users actually see are faithful**. If the API key is missing or expired, the system falls back to a template and shows a visible warning."

## Slide 23: Section 07, Extensions (≈5 s)

"Late or not late is only the first question. Here's how I answered the follow-ups."

## Slide 24: Beyond Late / Not Late (≈75 s)

"Six extensions:

- **Severity classes**: quadratic weighted kappa of 0.864. Major delays are caught 80 % of the time at 93 % precision.
- **Delay minutes**: a regression model with a mean absolute error of **1.08 minutes**, against 1.71 for simple persistence.
- **Conformal intervals**: calibrated 80 % intervals achieve **79.6 %** coverage, almost exactly on target.
- **Propagation**: 88 % of late buses stay late, which again says dispatchers must act early.
- **Bus bunching**: PR-AUC of 0.444, **six times the base rate**. Bunching concentrates on frequent routes, in the afternoon and late in the route.
- **Chronic segments**: 29 of 371 segments are consistently problematic and stable over time, with ρ = 0.84. Most of them inherit their delay from upstream."

## Slide 25: Section 08, Website & Live Mode (≈5 s)

"All of this is only useful if agency users can actually use it."

## Slide 26: TransitPulse Streamlit Website (≈60 s)

"So I built a Streamlit website with a home page and eight pages.

The **Home** page shows headline metrics and a per-route trust table. The **Reliability Explorer** shows late rates by route, hour and month, plus a map. **Model Comparison** lists all six models with confidence intervals. **Predict & Explain** takes a real December 2025 event and shows the risk, the SHAP reasons, the LLM text and a what-if tool. **Explainability** covers global importance and rules. **Propagation & Bunching**, **Planner Insights** with chronic segments and running-time suggestions, **Live Mode**, and finally **Data Sources**, which documents every dataset.

I'll show some of this in the demo."

## Slide 27: Live Mode on the MBTA V3 API (≈60 s)

"Live mode pulls vehicle positions, trips and predictions from the **MBTA V3 API** and current weather from Open-Meteo. It builds feature rows with the **same definitions as training** and scores them with the operational XGBoost model. Dispatchers get a map with SHAP reasons and a narrative.

Three things I'm proud of:
- **Validated**: every bus the MBTA itself predicted to be more than 5 minutes late was also flagged by my model, at p ≥ 0.91.
- **Graceful degradation**: rate limits, bad keys, no network or no buses produce a message, never a crash.
- **Train/serve skew caught and fixed**: I compared the bus-ahead feature against its training definition and corrected a mismatch.

It works without a key at 20 requests a minute, or up to 1000 with one, and every other page works offline."

## Slide 28: Engineering Quality & Reproducibility (≈45 s)

"On engineering quality:
- **62 automated tests** covering leakage, features, joins, explainability, the narrator, and every website page run headless.
- **31 logged design decisions** with reasons in a decision log.
- **9 double-click pipeline steps**, from download to the thesis.
- **One-click setup** on a new Windows PC, with the trained models shipped in the repo.

The stack is Python, pandas, XGBoost, LightGBM, Optuna, SHAP, GPT-4o-mini, Streamlit and Plotly."

## Slide 29: Section 09, Results & Conclusion (≈5 s)

"To wrap up, back to the research questions."

## Slide 30: Answers to the Research Questions (≈75 s)

"**RQ1, Predictability.** Lateness is moderately predictable a day ahead, at AUC 0.755 against 0.657 for the baseline, and highly predictable in real time at 0.978. The probabilities are calibrated.

**RQ2, External data.** It adds little on average but helps where it applies, in rain and snow. Event features didn't transfer to the test period, and I report that honestly.

**RQ3, Explanation.** The explanations are faithful and stable, and with the fact check the LLM narratives are 100 % faithful when shown.

**RQ4, Operations.** It runs live on the MBTA V3 API and answers severity, minutes, propagation, bunching and chronic-segment questions."

## Slide 31: Lessons, Limitations & Future Work (≈75 s)

"Some reflections.

**Lessons learned.** Baselines catch bugs: a bunching baseline that performed *below random* exposed a feature bug. Always check training definitions before serving; that led to two fixes in live mode. And a time-based split beats a random split, because it avoids leaking recent history.

**Limitations.** This covers one city, 19 routes and one year, and schedule changes require retraining. Live features approximate the training definitions. SHAP and what-if show **associations, not causal effects**. And real agency staff haven't rated the narratives yet.

**Future work.** A holding-strategy simulation for bunching alerts, richer event features, retraining on every new schedule 'pick', and a live accuracy log with a user study involving MBTA staff."

## Slide 32: Thank You (≈15 s, then demo)

"Thank you. The code is on GitHub at the link shown, and the whole app starts with one command. I'll now give a short live demo, and then I'm happy to take questions."

---

### Suggested demo flow (3–5 min)
1. **Home**: headline metrics and the trust table.
2. **Predict & Explain**: pick a Route 44 event, then show the risk, the SHAP bars and the LLM narrative, and move one what-if slider.
3. **Live Mode**: buses in service now on the map; click one high-risk bus.
4. (If time allows) **Planner Insights**: chronic segments.

### Likely questions to prepare for
- *Why AUC and PR-AUC rather than accuracy?* The target is imbalanced (28.8 % late), so accuracy is misleading.
- *Why is real-time so much better than day-ahead?* Delay is sticky (88 % persistence); the current delay dominates.
- *Is the 0.978 too good to be true / leakage?* Strict time split, past-only history, automated leakage tests, and the test set touched once.
- *Why didn't events help?* (Check the thesis for the exact reason.) A likely one: there are no Red Sox games in the Nov–Dec test window, and the event features lack venue capacity and end time, which is listed as future work.
- *Can the LLM make things up?* Numeric fact check against SHAP; about 5 % rejected; template fallback.
