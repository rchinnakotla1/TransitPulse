# Phase 4: Explainability (XAI) and Plain-Language Narratives

Full executed analysis: [notebooks/06_Explainability.ipynb](../notebooks/06_Explainability.ipynb) · code `src/xai/` (`shap_analysis.py`, `rules.py`,
`counterfactual.py`, `narrator.py`, `llm_client.py`, `feature_text.py`, `evaluate_narratives.py`) · results `reports/xai/` · figures
`reports/figures/p4_*.png` · tests `tests/test_narrator.py`, `tests/test_xai.py`

## 1. Why this phase

A delay probability on its own is not actionable. A dispatcher needs to know *why* this bus is flagged, a planner needs to know *which
design choice* (running time, frequency) is behind chronic risk, and a rider needs one honest sentence. Phase 4 answers five questions:

1. What drives the predicted risk overall and for one trip? (TreeSHAP)
2. Can the model be summarised as readable rules, and how faithful are they? (surrogate tree)
3. What could a planner or dispatcher change? (model-based what-ifs)
4. Are the explanations stable across two equally good models? (XGBoost vs LightGBM)
5. Can an LLM write plain language **without inventing facts**, and what happens if the API key fails? (OpenAI + fact check + fallback)

## 2. Methods

| Step | Method | Key choices (decision log) |
|---|---|---|
| SHAP | XGBoost native exact TreeSHAP (`pred_contribs=True`) on **50,000** stratified test events per mode | sample size #14. Additivity is asserted against the raw margin (`output_margin=True`), because log(p/(1−p)) is numerically unstable near 0/1 |
| Groups | features summed into the P2 groups (history, temporal, spatial, calendar, weather, GTFS, daylight, ridership, events, operational) | |
| Rules | depth-4 `DecisionTreeRegressor` fitted to XGBoost probabilities on 200 k **validation** rows; fidelity measured on the **test** SHAP sample; depth sweep 2–12 | #15 |
| What-if | change one input and re-score, keeping linked features consistent (`seg_sched_min` → `schedule_tightness`, `prev_tp_delay` → `prev_tp_delayed`, rain → `heavy_rain`); single case + population (16 k events) + operational tipping points (2 k events) | #16, labelled *not causal* |
| Stability | Spearman ρ of mean \|SHAP\| between XGB and LGB, top-10 overlap, at **feature and group** level | #23 |
| Narratives | facts sheet (top SHAP factors as percentage-point effects) → template (always) → OpenAI `gpt-4o-mini` rewrite for dispatcher / planner / rider → **faithfulness check** → fallback | #17, #18, #22 |

**Faithfulness check.** (a) Every number in the LLM text must appear in the fact sheet; a whole-number rounding of a fact value is allowed.
(b) The top factor must be named. If either fails, the template text is shown and the rejected draft is kept for audit.

**Key failure handling.** `llm_client.health_check()` classifies errors as missing key, invalid/expired key (401), quota exhausted /
rate limit (429), network, or unknown model (404). It logs the result to `logs/llm.log` without ever writing the key. The dashboard
shows a warning banner and silently uses templates, so a broken key never breaks the app.

## 3. Results

### 3.1 Sanity
Base value + Σ SHAP equals the model margin to within 5.8e-6 (planning) and 2.6e-5 (operational). The base rate of the stratified
sample is p = 0.294 / 0.288.

### 3.2 Global importance (share of total |SHAP|)

| Group | Planning | Operational |
|---|---|---|
| history | **32.1 %** | 3.1 % |
| temporal | 16.2 % | 4.4 % |
| spatial | 16.2 % | 12.3 % |
| calendar | 11.7 % | 2.0 % |
| weather | 8.8 % | 1.3 % |
| GTFS schedule | 8.0 % | 10.1 % |
| daylight | 3.6 % | 1.3 % |
| ridership | 3.2 % | 4.5 % |
| events | 0.3 % | 0.0 % |
| operational (live) | n/a | **61.0 %** |

Top planning features: recent 28-day stop late rate (12.3 %), usual stop-hour late rate (9.2 %), stop (7.0 %), minutes into trip (6.4 %),
hour of day (6.2 %). Top operational features: delay at the previous timepoint (24.9 %), minutes since the previous timepoint (19.9 %),
stop (8.2 %).

**Temperature as a season proxy.** Temperature ranks 7th (4.7 %), yet adding weather in the P3 ablation gave almost no AUC. Colder
readings get *lower* risk (−0.22 log-odds below 0 °C; mean −0.16 in December), and December is the low-demand holiday month. We
interpret this as seasonality absorbed by temperature, not a causal weather effect. SHAP explains the model, not the world.

### 3.3 Direction of effects
* The usual stop-hour late rate moves SHAP from −0.88 (≤ 10 %) to +0.43 (> 60 %). The recent 28-day rate covers a wider range
  (−0.67 to +0.85).
* **Schedule tightness:** ≤ 1.5 scheduled min/km → +0.26 to +0.31; > 6 min/km → −0.24. Under-scheduled segments are riskier.
* Crowding (load > 40) → +0.14. Rain has |SHAP| < 0.06.
* Operational: the previous-timepoint delay effect goes from −2.9 (on time) to +3.8 (> 10 min) and crosses zero between 4 and 5 min.
  A gap ratio to the bus ahead > 2 adds +0.35, while bunched buses (< 0.25) get −0.23. Phase 5 later showed why a *large* ratio is risky: the feature skips a leader that has not yet passed the target stop, so a ratio > 2 usually means a bus close ahead, i.e. a bunched pair (decision 30).

### 3.4 By period and route
The PM peak is the riskiest period (31.5 % actual, 32.1 % predicted). It is explained by clock time (temporal +0.20) and historically
late stops, whereas the early morning is low risk (temporal −0.59, history −0.49). The highest-risk route × direction × period cells
are 73 Outbound PM peak (60 % late, 2.4× average), 71 Outbound PM peak, 39 Outbound evening/night, and route 44 in most periods
(`reports/xai/route_period_risk.csv`).

### 3.5 Surrogate rules

| Mode | Leaves | R² vs XGB | AUC surrogate | AUC XGB |
|---|---|---|---|---|
| Planning | 16 | 0.380 | 0.683 | 0.754 |
| Operational | 14 | **0.894** | **0.951** | 0.977 |

Depth sweep, planning: 284 leaves still give only R² 0.48 / AUC 0.696. Operational reaches R² 0.83 already at 4 leaves. Readable rules are
therefore faithful for live operations ("previous timepoint > 6.8 min late → 99 % late"). For planning they are only a coarse summary
("stop usually > 65 % late at this hour and route > 30 % late in the last 4 weeks → 72 %"). Because the surrogate was fitted on
validation months (32.8 % late), its probabilities are about 10 points high on test; its ranking is what is used.

### 3.6 What-if

| Analysis | Result |
|---|---|
| Single case (route 73 Out, stop 2125, Fri 19:00, p = 0.80) | +10 min of running time does not cross the threshold; the risk is inherited |
| 16,103 high-risk daytime events, add running time | fixed with ≤ 1 min: 10.2 % · ≤ 3 min: **21.8 %** · ≤ 5 min: 26.4 % · ≤ 10 min: 29.3 % |
| Mean probability vs added minutes | 0.449 → 0.420 (+1) → 0.388 (+3) → 0.381 (+4) → 0.379 (+10): diminishing returns after 3–4 min |
| Weather scenarios (same case) | ≤ 3 points change |
| Operational tipping point (2,000 events) | median **6 min** late at the previous timepoint (IQR 4–7.5); 10 % are already over the threshold; 9.6 % never cross within 15 min |

### 3.7 Stability (XGB vs LGB)

| Mode | Feature ρ | Top-10 overlap | Group ρ | Max group diff |
|---|---|---|---|---|
| Planning | 0.981 | 10/10 | 0.983 | 3.6 pts |
| Operational | 0.545 | 4/10 | 0.879 | 5.8 pts |

The operational disagreement is caused by correlated features: XGB credits the previous-timepoint delay (25 %), while LGB credits
minutes since the previous timepoint (38 %) and gives `prev_tp_delayed` almost nothing. At group level the live block gets 54 % vs 52 %.

### 3.8 LLM narratives (60 = 20 events × 3 audiences)

| Version of the check | LLM drafts rejected | Why | Shown texts faithful |
|---|---|---|---|
| v1 exact numbers | 20 % (12) | all 12 were honest roundings (40.4 → "40") | 100 % (fallback) |
| v2 rounding allowed | **5 % (3)** | all genuine: the main factor was replaced | **100 %** |

LLM texts average 69–76 words. All rider texts include a "this is a prediction" reminder. With no key, status = `missing_key` and the
template plus a warning banner are used. A human-rating rubric (`reports/xai/narrative_rubric.csv`) is provided for clarity,
correctness and usefulness instead of an LLM judge.

## 4. Discussion

* **Two models, two explanation regimes.** Live risk is explained almost entirely by "already late upstream", which is simple,
  rule-like and stable. Planned risk is a sum of many small effects (history, clock, place, calendar), which is why rule lists fail
  for it and SHAP per prediction is needed.
* **Actionability.** The planner lever the model reacts to is schedule tightness. Padding helps a minority of high-risk events, with
  diminishing returns. Most risk is inherited, which motivates the propagation analysis in Phase 5.
* **Honesty.** SHAP is model attribution, not causality (temperature shows this clearly). What-ifs are associations. The LLM never sees
  raw data and is always checked.

## 5. Limitations

* SHAP is computed on a 50 k sample. Global shares are stable at this size, and stability across libraries was checked.
* Correlated features split credit arbitrarily, so we report groups.
* What-ifs ignore behavioural responses (drivers, the bus ahead) and are bounded by the 2025 data range.
* The fact check verifies numbers and the main factor, not every qualitative claim. Human rating is left to the rubric.
* The LLM depends on an external paid API; the template fallback guarantees availability.

## 6. Decisions carried forward
* The dashboard explains predictions with grouped SHAP bars and a checked narrative, and shows rule fidelity next to the rules.
* Planner Insights page: route × period risk, schedule-padding effect, chronic segments (P5).
* Live mode: alert framing around the ~6-minute tipping point at the previous timepoint.
