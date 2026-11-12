# Findings text for notebooks/06_Explainability.ipynb (read by build_xai_notebook.py).
# Every number below was checked against the executed outputs (reports/xai/*, logs/shap.log, logs/rules.log,
# logs/stability.log, logs/narratives.log) on 2026-11-08.

globals()["global"] = """
**Finding.** *Planning model:* history features explain the largest share (**32 %** of total |SHAP|). The top two features are the
stop's recent 28-day late rate (12.3 %) and its usual late rate at this hour (9.2 %). Next come temporal (16 %), spatial (16 %),
calendar (12 %), weather (9 %) and GTFS schedule design (8 %). *Operational model:* the live features take **61 %**; the delay at the
previous timepoint and the minutes since it alone account for 45 %. This matches the P3 ablation, where adding the operational group
raised AUC from 0.73 to 0.98.

**Caveat (temperature).** Temperature is the 7th planning feature (4.7 %), but in P3 the weather group added almost no AUC. The SHAP
values show why: colder means *lower* predicted risk (−0.22 log-odds below 0 °C), and the test months behave the same way
(mean SHAP −0.07 in November, −0.16 in December). Temperature is mostly acting as a **season proxy** (the low-demand December holiday
period), not as a causal weather effect. SHAP explains the *model*, not the world, so we report it that way.

**Decision.** Explanations shown to users are grouped by data source (history, time, place, schedule, weather, live). Temperature is
labelled "season/temperature" in the dashboard and is never presented as a cause.
"""

dependence = """
**Finding.** The effects are monotone and plausible:
* Usual late rate at this stop-hour: SHAP rises from −0.88 (≤ 10 %) to +0.43 (> 60 %). The recent 28-day rate rises even more
  steeply, from −0.67 to +0.85, which is why it ranks first: recent conditions (roadworks, detours) matter more than the long-run
  average.
* **Schedule tightness** (scheduled min/km): tight segments (≤ 1.5 min/km) push risk *up* (+0.26 to +0.31); generous ones
  (> 6 min/km) push it *down* (−0.24). The model has learned that under-scheduled running time causes lateness.
* Crowding (average load > 40) adds +0.14. Rain has only small effects (|SHAP| < 0.06), consistent with the small weather gain in P3.
* Operational: the previous-timepoint delay effect swings from −2.9 (early or on time) to +3.8 (> 10 min late) and crosses zero
  between 4 and 5 min. A big gap behind the bus ahead (gap ratio > 2) adds +0.35; a bunched bus (< 0.25) subtracts 0.23.

**Decision.** Schedule tightness is a lever a planner controls, so it becomes the basis of the what-if analysis (§8) and of the
chronic-segment running-time suggestions in Phase 5.
"""

period = """
**Finding.** The PM peak has the highest late rate (31.5 % actual vs 32.1 % predicted, so it is well calibrated by period). Its
explanation is mostly *time of day itself* (temporal group +0.20 log-odds) plus stops that are historically late at that hour
(history +0.03, against −0.49 in the early morning). Early morning is low risk because of both clock time (−0.59) and history (−0.49).
In the evening and at night, darkness (daylight group −0.14) lowers the risk, which reflects lighter traffic rather than darkness
itself. Weather is negative in every period (−0.06 to −0.15), the season-proxy effect noted above. By route, the riskiest
route-direction-periods are 73 Outbound in the PM peak (60 % actually late, 2.4× the average), 71 Outbound in the PM peak, 39 Outbound
at night/evening, and route 44 in almost every period.

**Decision.** The route × period risk table (`route_period_risk.csv`) goes into the Planner Insights page, because it tells a
planner *where and when* to look first.
"""

rules = """
**Finding.** Readable rules work well for the **operational** model: a depth-4 tree (14 leaves) reproduces XGBoost with R² = 0.89 and
AUC 0.951 (vs 0.977), e.g. *"delay at the previous timepoint > 6.8 min and not at the last stops → 99 % late"*. For the **planning**
model, a 16-leaf tree reaches only R² 0.38 and AUC 0.683, below even the P3 decision tree (0.700). The depth sweep shows the
trade-off: 284 leaves (depth 12) still give only R² 0.48 / AUC 0.696. Planning risk is spread over many small interacting effects
that no short rule list captures. The top planning rule is *"stop usually late at this hour (> 65 %) and route late in the last 4
weeks (> 30 %) → 72 % (actual 62 %)"*. The surrogate's probabilities run about 10 points high on test because it was fitted on
validation months with a higher late rate (32.8 % vs 25.1 %); its *ranking* is what matters.

**Decision.** Rules are shown as a *summary* with the fidelity printed next to them. Planners are told the rules cover the main
pattern and that individual predictions should use the SHAP explanation. This is an honest limitation of global surrogates for
diffuse models.
"""

whatif = """
**Finding.** *Single case* (route 73 Outbound, stop 2125, Friday 19:00, p = 0.80): adding up to 10 scheduled minutes does **not**
bring the risk below the alert threshold. The risk is inherited from upstream and from the stop's history, not created on this
segment. *Population view* (16,103 high-risk daytime events): adding running time to the segment moves **10 %** below the threshold
with 1 min, **22 %** with ≤ 3 min and **29 %** with ≤ 10 min. The mean probability falls from 0.449 to 0.381 at +4 min and then
plateaus (0.379 at +10). There are clear **diminishing returns after about 3–4 minutes**. Weather scenarios change this case by at
most 3 points. *Operational tipping point* (2,000 events): the median event crosses the dispatcher alert when the previous timepoint
is **6 min** late (IQR 4–7.5 min). 10 % are above the threshold even when early (bus ahead, history), and 9.6 % never cross within
15 min.

**Decision.** Planner advice is framed as "a small padding (≤ 3 min) fixes about 1 in 5 high-risk stop events; the rest need upstream
or route-level action". This links directly to the propagation analysis in Phase 5. Every what-if is labelled *model-based, not
causal*.
"""

stability = """
**Finding.** *Planning:* XGBoost and LightGBM agree almost perfectly on why (feature-rank ρ = 0.98, top-10 overlap 10/10, group-rank
ρ = 0.98, max group difference 3.6 points). *Operational:* feature-level agreement is weak (ρ = 0.55, top-10 overlap 4/10), but
group-level agreement is strong (ρ = 0.88; the live group gets 54 % vs 52 %). The cause is correlated features: XGBoost credits the
previous-timepoint delay (25 %), while LightGBM credits minutes since the previous timepoint (38 %) and gives the 0/1 flag
`prev_tp_delayed` almost nothing. These features carry the same information, so the credit moves between them.

**Decision.** (Decision log #23.) Stability is reported at both levels, and operational explanations are communicated by *factor
group* ("live delay information"), which is the level at which two equally accurate models agree.
"""

llm = """
**Finding.** OpenAI (gpt-4o-mini) was connected. On 60 narratives (20 test events × 3 audiences), the first run with an exact-number
fact check rejected **20 %** of LLM drafts. Manual inspection showed **all 12 were honest roundings** (e.g. "40 minutes" for 40.4 min).
After allowing whole-number rounding and nothing else (decision #22), **5 %** (3/60) were rejected, and all 3 were *genuine*: the LLM
swapped the main factor (minutes into the trip) for another one, or claimed the time of day was most important. Rejected drafts fall
back to the template, so **100 % of the texts shown are faithful**. LLM texts average about 70–76 words. The rider version is shorter
in the template (30 words), and all LLM rider texts include a "this is a prediction" reminder. With an empty or expired key the system returns
status `missing_key` / `invalid_key`, shows a warning banner and uses the template, so the app never breaks.

**Decision.** Keep the LLM *behind* the fact check, never as the source of truth. The human-rating rubric (`narrative_rubric.csv`,
clarity / correctness / usefulness 1–5) is provided for a reviewer instead of using an LLM as a judge.
"""

summary = """
## Phase 4 summary

| Question | Answer | Evidence |
|---|---|---|
| What drives planned risk? | stop/route late history (32 %), time of day (16 %), place (16 %), calendar (12 %) | SHAP groups |
| What drives live risk? | delay at the previous timepoint and time since it (61 % live group) | SHAP groups |
| Is the model sensible? | monotone, plausible effects; tight schedules raise risk | dependence |
| Can it be written as rules? | operational yes (R² 0.89); planning only roughly (R² 0.38) | surrogate fidelity |
| What can a planner change? | ≤ 3 min padding fixes ~22 % of high-risk events; diminishing returns after that | population what-if |
| When should a dispatcher act? | median tipping point is 6 min late at the previous timepoint | operational what-if |
| Are explanations stable? | yes at group level (ρ 0.98 / 0.88) | XGB vs LGB |
| Can an LLM explain safely? | yes with a fact check: 5 % of drafts rejected, 100 % of shown texts faithful | narrative evaluation |

**Next (Phase 5):** the what-ifs show that most late risk is *inherited* from upstream. Phase 5 measures this directly (delay
propagation, recovery), adds severity and minutes-late predictions, detects bunching, and lists chronic segments that persist out of
time.
"""
