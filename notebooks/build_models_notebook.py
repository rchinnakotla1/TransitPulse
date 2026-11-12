"""Builds notebooks/05_Models.ipynb. Execute with:
    ..\\.venv\\Scripts\\jupyter nbconvert --to notebook --execute --inplace 05_Models.ipynb
Requires: python -m src.models.train planning operational ; python -m src.models.experiments
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 05 — Modelling and Evaluation (Phase 3)

**Research questions answered here**
1. Which classifier predicts *delayed > 5 min* best? (B0 historical baseline → logistic regression → decision tree → random forest → XGBoost, plus LightGBM as a cross-check)
2. How much do the external sources add **on top of the historical averages agencies already use**? (ablation)
3. How much better can a dispatcher do *minutes ahead* (operational mode) than a planner *a day ahead* (planning mode)?
4. Is the model robust over time? Is 8 months of training enough (learning curve)? Would a 12-month window help with seasonal differences?
   How fast does a model decay (walk-forward, drift)?

> **What** · **Why** · **How** · **Result** · **Finding** · **Decision**

### Protocol (`src/models/train.py`)
| Step | Data | What is decided |
|---|---|---|
| Tune | train Jan–Aug → validation Sep–Oct | hyper-parameters (grid for LR/DT/RF, 15-trial Optuna for XGBoost) and the **decision threshold** (max F1) |
| Refit | train + validation Jan–Oct (83 % of 2025) | nothing new; same settings, more and more recent data |
| Test | Nov–Dec, scored **once** | ROC-AUC, PR-AUC, Brier, P/R/F1 at the frozen threshold, 95 % **day-block bootstrap** CIs |

Why the day-block bootstrap: stop events on the same day share weather, incidents and traffic, so they are not independent. Resampling
whole service days gives honest (wider) intervals.
""")

code("""
import sys, json, warnings
sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
from src.config import REPORTS, FIG_DIR
from src.models.common import calibration_table, mcnemar, metrics
sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi'] = 110
def save(n): plt.tight_layout(); plt.savefig(FIG_DIR / f'p3_{n}.png', bbox_inches='tight'); plt.show()
ORDER = ['B0', 'LR', 'DT', 'RF', 'XGB', 'LGB']
NAMES = {'B0': 'B0 historical rate', 'LR': 'Logistic regression', 'DT': 'Decision tree', 'RF': 'Random forest', 'XGB': 'XGBoost', 'LGB': 'LightGBM'}
M = pd.concat([pd.read_csv(REPORTS / f'metrics_{m}.csv') for m in ('planning', 'operational')])
P = {m: pd.read_parquet(REPORTS / 'preds' / f'test_{m}.parquet') for m in ('planning', 'operational')}
V = {m: pd.read_parquet(REPORTS / 'preds' / f'valid_{m}.parquet') for m in ('planning', 'operational')}
""")

md("""
## 1. Model comparison (test set, Nov–Dec 2025)
**What:** every model in both modes, scored once on the held-out test months.
**Why:** ROC-AUC (ranking quality) and PR-AUC (quality on the minority *delayed* class) do not depend on a threshold. That matters here because
the delay rate shifts from 32.8 % (validation) to 25.1 % (test). F1 / precision / recall use the threshold frozen on validation, which is
what an agency would actually deploy.
""")
code("""
t = M[M.split == 'test'].copy()
t['ROC-AUC [95% CI]'] = t.apply(lambda r: f"{r.roc_auc:.4f} [{r.roc_auc_lo:.4f}, {r.roc_auc_hi:.4f}]", axis=1)
t['PR-AUC [95% CI]'] = t.apply(lambda r: f"{r.pr_auc:.4f} [{r.pr_auc_lo:.4f}, {r.pr_auc_hi:.4f}]", axis=1)
t['model'] = pd.Categorical(t.model, ORDER); t = t.sort_values(['mode', 'model'])
cols = ['mode', 'model', 'ROC-AUC [95% CI]', 'PR-AUC [95% CI]', 'brier', 'threshold', 'precision', 'recall', 'f1', 'pred_rate', 'seconds']
display(t[cols].round(4).reset_index(drop=True))
vt = M.pivot_table(index=['mode', 'model'], columns='split', values='roc_auc').round(4)
vt['valid→test'] = (vt.test - vt.valid).round(4); vt
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, mode in zip(axes, ['planning', 'operational']):
    d = t[t['mode'] == mode]
    ax.barh(d.model.astype(str), d.roc_auc, xerr=[d.roc_auc - d.roc_auc_lo, d.roc_auc_hi - d.roc_auc], color='steelblue')
    ax.set(xlim=(0.5, 1), title=f'{mode}: test ROC-AUC with 95 % day-block CI', xlabel='ROC-AUC')
save('model_comparison')
""")
md("""
**Finding.**
* **Planning (day-ahead):** XGBoost is best with test ROC-AUC **0.755** [0.748, 0.762] and PR-AUC **0.526** [0.500, 0.552].
  Next come random forest (0.735), logistic regression (0.720) and the decision tree (0.700). The historical-average baseline B0 scores 0.657.
  XGBoost's PR-AUC is **+0.15 over B0**, a 39 % relative gain over what an agency gets from its own averages. The confidence intervals
  of XGBoost and RF do not overlap.
* **Operational (minutes-ahead):** every model scores above 0.96. XGBoost is best at **0.978** [0.977, 0.979], PR-AUC 0.954. Even the one-line
  rule "a bus already late at the previous timepoint stays late" reaches 0.892. Live information dominates: the question in this mode is
  not *whether* we can predict, but how much the model adds over the simple rule. It adds +0.18 PR-AUC and +5 pp F1 (0.875 vs 0.823).
* **Validation → test:** planning scores are *higher* on test than on validation (XGB 0.726 → 0.755). This is not leakage. The final models are
  refit on Jan–Oct, so they have seen the autumn schedule that validation (Sep–Oct) was missing. The walk-forward (§11) shows
  the same jump.
* **LightGBM cross-check:** with only a 3-point grid, LightGBM scores **0.751** [0.744, 0.758] (planning) and **0.977** (operational), inside the
  XGBoost confidence intervals, in 4–7 min instead of 45–68 min. The gain is a property of gradient boosting on these features, not of one library
  or of lucky tuning.
* **Cost:** XGBoost tuning + refit took about 45 min per mode on a 12-core laptop. LR and RF took 8–20 min. Training cost is irrelevant
  for daily or monthly retraining.

**Decision.** XGBoost drives the dashboard in both modes (LightGBM = fast retraining fallback). Logistic regression is kept as the transparent reference, and the decision tree as the readable-rules
model for the XAI chapter.
""")

md("""
## 2. ROC and precision–recall curves
**Why:** the curves show the full trade-off. A dispatcher who can only act on a few trips cares about the **high-precision, left end** of the PR curve.
""")
code("""
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for i, mode in enumerate(['planning', 'operational']):
    p = P[mode]; y = p.delayed
    for m in ORDER:
        if f'p_{m}' not in p: continue
        fpr, tpr, _ = roc_curve(y, p[f'p_{m}']); axes[i, 0].plot(fpr, tpr, label=m)
        pr, rc, _ = precision_recall_curve(y, p[f'p_{m}']); axes[i, 1].plot(rc[::200], pr[::200], label=m)
    axes[i, 0].plot([0, 1], [0, 1], 'k:'); axes[i, 0].set(title=f'{mode}: ROC', xlabel='FPR', ylabel='TPR'); axes[i, 0].legend()
    axes[i, 1].axhline(y.mean(), color='k', ls=':', label='base rate')
    axes[i, 1].set(title=f'{mode}: precision–recall', xlabel='recall', ylabel='precision'); axes[i, 1].legend()
save('roc_pr')
""")

md("""
## 3. Confusion matrices at the frozen threshold (best model per mode)
""")
code("""
best = {mode: t[t['mode'] == mode].sort_values('pr_auc').model.astype(str).iloc[-1] for mode in P}
print('best model per mode (test PR-AUC):', best)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, mode in zip(axes, P):
    m = best[mode]; thr = t[(t['mode'] == mode) & (t.model == m)].threshold.iloc[0]
    cm = confusion_matrix(P[mode].delayed, P[mode][f'p_{m}'] >= thr, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax, xticklabels=['on time', 'delayed'], yticklabels=['on time', 'delayed'])
    ax.set(title=f'{mode} · {m} · threshold {thr:.2f}', xlabel='predicted', ylabel='actual')
save('confusion')
""")

md("""
## 4. Calibration: can the probability be read as "chance of being late"?
**Why:** the dashboard shows probabilities to dispatchers and riders, so "70 %" should mean about 7 in 10 such trips are late. The test months are calmer
(25.1 % delayed vs 28.8 % in training), so we expect some **over-prediction**.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, mode in zip(axes, P):
    for m in ORDER:
        if f'p_{m}' not in P[mode]: continue
        c = calibration_table(P[mode].delayed.values, P[mode][f'p_{m}'].values)
        ax.plot(c.mean_pred, c.observed, marker='o', label=m)
    ax.plot([0, 1], [0, 1], 'k:'); ax.set(title=f'{mode}: reliability (test)', xlabel='mean predicted', ylabel='observed rate'); ax.legend()
save('calibration')
t[['mode', 'model', 'mean_prob', 'base_rate', 'brier']].round(4)
""")
md("""
**Finding.** Despite the calmer test months, the probabilities stay close to calibrated.
* **Planning XGBoost:** mean prediction **26.1 %** vs **25.1 %** observed. Logistic regression (25.3 %) is the best-calibrated planning
  model because it was deliberately *not* class-weighted.
* **Over-prediction:** RF (28.9 %), DT (28.3 %) and especially B0 (32.6 %) over-predict. B0 averages the busier history.
* **Operational:** all models are within 0.6 pp of the truth.

The 28-day rolling rates let the models follow the drop in delay level without retraining.
**Decision.** Raw XGBoost probabilities can be shown in the dashboard as "chance of being > 5 min late". No extra recalibration step is needed.
We will re-check this in the drift experiment.
""")

md("""
## 5. Choosing the operating threshold
**What:** precision, recall and the share of trips flagged as the threshold moves (best planning model, test set).
**Why:** max-F1 is a neutral default. An agency would choose based on cost: a **rider alert** needs high precision (don't cry wolf); a
**dispatcher watch-list** can accept more false alarms to catch more late buses.
""")
code("""
rows = []
for mode in P:
    m = best[mode]; y = P[mode].delayed.values; p = P[mode][f'p_{m}'].values
    for thr in np.arange(0.2, 0.85, 0.05):
        yh = p >= thr; tp = (yh & (y == 1)).sum()
        rows.append({'mode': mode, 'threshold': round(thr, 2), 'precision': tp / max(yh.sum(), 1), 'recall': tp / y.sum(), 'flagged %': yh.mean() * 100})
th = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, mode in zip(axes, P):
    d = th[th['mode'] == mode].set_index('threshold')[['precision', 'recall']]; d.plot(ax=ax, marker='.')
    ax.set(title=f'{mode} · {best[mode]}: precision / recall vs threshold')
save('threshold')
th.round(3).pivot(index='threshold', columns='mode', values=['precision', 'recall', 'flagged %'])
""")
md("""
**Finding / Decision: two operating points for two agency products.**

| Product | Mode | Threshold | Precision | Recall | Trips flagged |
|---|---|---|---|---|---|
| Day-ahead **planner watch-list** (which trips to reinforce) | planning | 0.30 | 0.46 | 0.61 | 33 % |
| Day-ahead **rider alert** (don't cry wolf) | planning | 0.60 | 0.70 | 0.17 | 6 % |
| **Dispatcher alert** | operational | 0.50 | 0.93 | 0.82 | 22 % |

A day ahead, the model can confidently single out only a small group of trips: 7 of 10 trips flagged at 0.60 really are late. Minutes ahead,
it is accurate enough to act on almost every alert. The dashboard exposes the threshold as a slider with these presets.
""")

md("""
## 6. Is the best model *significantly* better than the runner-up? (McNemar)
""")
code("""
rows = []
for mode in P:
    tt = t[t['mode'] == mode].sort_values('pr_auc', ascending=False)
    a, b = tt.model.astype(str).iloc[0], tt.model.astype(str).iloc[1]
    ta, tb = tt.threshold.iloc[0], tt.threshold.iloc[1]
    r = mcnemar(P[mode].delayed, P[mode][f'p_{a}'] >= ta, P[mode][f'p_{b}'] >= tb)
    rows.append({'mode': mode, 'A (best)': a, 'B (runner-up)': b, **r})
pd.DataFrame(rows)
""")
md("""
**Finding.** In both modes XGBoost vs random forest is highly significant (p < 10⁻¹⁰⁰).
* **Planning:** XGBoost is right where RF is wrong on 228,588 test events, and the reverse holds on 70,636. Part of this comes from XGBoost's
  higher threshold (it flags fewer trips), so the threshold-free AUC CIs above are the primary evidence.
* **Operational:** 26,134 vs 13,268.

With 1.7 M test events almost any difference is "significant". The practical gap is +0.02 AUC / +0.03 PR-AUC in planning, and that is
what matters.
""")

md("""
## 7. Where does the model fail? Errors by route
**Why:** an agency needs to know where it can and cannot trust the model.
""")
code("""
mode = 'planning'; m = best[mode]; p = P[mode]
def rauc(g):
    return roc_auc_score(g.delayed, g[f'p_{m}']) if g.delayed.nunique() == 2 else np.nan
r = p.groupby('route_id', observed=True).apply(lambda g: pd.Series({'rows': len(g), '% delayed': g.delayed.mean() * 100,
        'mean prob %': g[f'p_{m}'].mean() * 100, 'AUC': rauc(g)}))
r = r[r.rows > 5000].sort_values('AUC')
display(pd.concat([r.head(8), r.tail(5)]).round(3))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(r['% delayed'], r.AUC, s=r.rows / r.rows.max() * 200, alpha=.6)
ax.set(xlabel='% delayed on route (test)', ylabel='route-level ROC-AUC', title=f'{mode} · {m}: per-route discrimination'); save('per_route')
""")
md("""
**Finding.** Discrimination is weakest on the **most reliable routes**: 111 (10.7 % delayed, AUC 0.68), 23 (15.8 %, 0.69) and 22 (20.0 %, 0.69).
It is strongest on chronically late routes: 240 (0.78), 73 and 34E (0.77), 44 (0.76). On a reliable route, lateness is mostly caused by
one-off incidents that no planning feature can foresee. On a chronically late route, it follows repeatable patterns of time, load and schedule.
Per-route calibration holds: mean predicted vs actual is within 3.3 pp on every route listed.
**Decision.** The dashboard shows route-level AUC next to each prediction, so a planner knows how much to trust it.
""")

md("""
## 8. Planning vs operational: the value of live information
**What:** best model in each mode, AUC by **position along the trip** (timepoint order).
**Why:** operational features come from the bus's previous timepoint. At the first timepoint they are missing, so both modes should be
equal there, and the operational advantage should grow along the route.
""")
code("""
pp, po = P['planning'], P['operational']
d = pd.DataFrame({'tp': pp.time_point_order.clip(upper=12).values, 'y': pp.delayed.values,
                  'plan': pp[f"p_{best['planning']}"].values, 'oper': po[f"p_{best['operational']}"].values})
g = d.groupby('tp').apply(lambda x: pd.Series({'planning': roc_auc_score(x.y, x.plan), 'operational': roc_auc_score(x.y, x.oper), 'rows': len(x)}))
ax = g[['planning', 'operational']].plot(marker='o', figsize=(8, 4)); ax.set(xlabel='timepoint order along the trip (12 = 12+)', ylabel='test ROC-AUC',
       title='Value of live information grows along the route'); save('mode_by_position')
g.round(3)
""")
md("""
**Finding.** As expected, the two modes are almost equal at the **first timepoint** (0.712 vs 0.734), where no live delay exists yet. The operational
model jumps to **0.976 from the second timepoint** and reaches 0.99 further along the route, while the planning model stays at 0.74–0.78.
Delay is highly persistent along a trip. Once a bus is late, the next timepoint is close to determined.
**Decision.** The agency value of the *operational* model lies in the first few timepoints and in the start-of-trip prediction. That is where
dispatchers can still intervene (holding, short-turning, swapping a bus). The *planning* model's value is in schedule and resource decisions.
The two are complementary products, not competitors.
""")

md("""
## 9. Ablation: what does each data source add over historical averages?
**What (`experiments.ablation`):** XGBoost with fixed tuned settings and a fixed 2 M training sample. Groups are added cumulatively, starting from
the **history** group (what agencies already use). A *drop-one* run removes each group from the full planning model.
**Why:** this is the central research question of the project. The drop-one view guards against order effects: a group added late can look
useless only because an earlier group already carried the same signal.
""")
code("""
A = pd.read_csv(REPORTS / 'exp_ablation.csv')
cum = A[(A.kind == 'cumulative') & (A.eval_on == 'test')].reset_index(drop=True)
cum['Δ AUC vs previous'] = cum.roc_auc.diff(); cum['Δ PR-AUC vs previous'] = cum.pr_auc.diff()
display(cum[['step', 'added', 'n_features', 'roc_auc', 'pr_auc', 'Δ AUC vs previous', 'Δ PR-AUC vs previous']].round(4))
full = cum[cum.added == 'spatial'][['roc_auc', 'pr_auc']].iloc[0]
drop = A[(A.kind == 'drop_one') & (A.eval_on == 'test')].copy()
drop['AUC lost when removed'] = full.roc_auc - drop.roc_auc; drop['PR-AUC lost'] = full.pr_auc - drop.pr_auc
display(drop[['added', 'roc_auc', 'pr_auc', 'AUC lost when removed', 'PR-AUC lost']].rename(columns={'added': 'group removed'}).round(4))
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
c2 = cum[cum.added != 'operational']
axes[0].plot(c2.added, c2.roc_auc, marker='o'); axes[0].set(title='Cumulative ablation (planning, test ROC-AUC)', ylabel='ROC-AUC'); axes[0].tick_params(axis='x', rotation=35)
d2 = drop.sort_values('PR-AUC lost'); axes[1].barh(d2.added, d2['PR-AUC lost'], color='indianred'); axes[1].set(title='Drop-one: test PR-AUC lost', xlabel='PR-AUC lost')
save('ablation')
""")
md("""
**Finding: the answer to the central research question.** (Experiment models use a 2 M training sample on Jan–Aug, so the absolute values are lower than the final model's.)

| Step | Test ROC-AUC | Δ | Reading |
|---|---|---|---|
| History only (= agency averages) | 0.678 | | already better than the single-feature B0 (0.657) |
| + temporal & calendar | 0.722 | **+0.044** | the biggest planning gain: *when* (hour × weekday × school day) interacts with *where* |
| + daylight, weather, events | 0.714 | −0.008 | no gain on average |
| + GTFS schedule, ridership | 0.720 | +0.006 | small but consistent (tight schedules, heavy loads) |
| + spatial (route / stop / position) | **0.730** | +0.010 | |
| + operational (live) | **0.977** | **+0.247** | live information outweighs everything else |

The drop-one view confirms this. Removing temporal+calendar costs −0.027 AUC, history −0.018 and spatial −0.010. Removing **daylight, weather,
events, GTFS or ridership each changes test AUC by ≤ 0.003**. That is inside the ±0.007 day-block CI, i.e. statistically no effect *on average*.
The external sources are not useless (Phase 1 showed real local effects: +3.9 pp near venue events, +6 pp on January snow days). But they
affect few rows, and the history and calendar features already capture most of the systematic part. §9b tests them on the rows they affect.

**Decision.** Keep all groups in the deployed model. They cost nothing at prediction time, and the drop-one losses are within noise in both
directions. Report honestly in the thesis that **day-ahead delay risk is driven mainly by recurring time-and-place patterns**. External
data refines rather than transforms the prediction.
""")

md("""
## 9b. Do weather and events help where they matter?
**What (`experiments.episodic`):** the full planning model vs the same model without weather or without events, scored only on the rows they
should affect (venue event nearby, citywide event, ≥ 2 mm rain in 3 h, snow in the last 24 h), using validation + test (Sep–Dec).
**Why:** an average AUC over 3.4 M rows hides effects that apply to 2 % of them.
""")
code("""
E = pd.read_csv(REPORTS / 'exp_episodic.csv')
e = E.pivot_table(index='eval_on', columns='model', values='roc_auc')
e['Δ weather'] = e['full planning'] - e['without weather']; e['Δ events'] = e['full planning'] - e['without events']
e['rows'] = E.groupby('eval_on').n_eval.first(); e['% delayed'] = E.groupby('eval_on').base_rate.first() * 100
e.round(4)
""")
md("""
**Finding: weather helps where it should; events do not.**

| Rows (Sep–Dec) | n | AUC full | Δ from weather | Δ from events |
|---|---|---|---|---|
| all | 3.38 M | 0.731 | −0.001 | −0.002 |
| rain ≥ 2 mm in 3 h | 157 k | 0.724 | **+0.007** | −0.002 |
| snow in last 24 h | 274 k | 0.713 | **+0.005** | −0.003 |
| venue event nearby | 59 k | 0.715 | +0.003 | **−0.010** |
| citywide event | 10 k | 0.716 | +0.003 | **−0.009** |

* **Weather** is invisible in the average but adds +0.005 to +0.007 AUC on wet and snowy rows, exactly where a planner needs it.
* **Event** features make predictions slightly *worse* on event rows. A likely reason is that the training months are dominated by Red Sox
  home games at Fenway (Apr–Aug), while Sep–Dec events are mostly TD Garden concerts and games with different timing and crowd flows. The
  binary "event within 1 km" flag cannot transfer between them.

**Decision.** We do **not** remove the event features from the deployed model now. Dropping a group because it hurts on the *test* set would bias the
reported test score (test-set feature selection). The validation drop-one shows only +0.001 for removing it. Instead, this is recorded as a
negative result, and a better event representation goes on the improvement list: event **end time**, attendance, and trips heading *away* from the
venue after the event.
""")

md("""
## 10. Learning curve: are we under-training with 8 months?
**What (`experiments.learning`):** train on 5 / 10 / 25 / 50 / 100 % of Jan–Aug and score validation (Sep–Oct). Training-sample AUC is shown too.
**Why:** a common question is whether holding out 17 % + 17 % starves the model. If validation AUC is still rising at 100 %, more data would
help. If it has flattened, the split costs nothing. The final model is refit on Jan–Oct anyway.
""")
code("""
L = pd.read_csv(REPORTS / 'exp_learning_curve.csv')
lc = L.pivot_table(index=['frac', 'n_train'], columns='eval_on', values='roc_auc').reset_index()
display(lc.round(4))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lc.n_train / 1e6, lc.valid, marker='o', label='validation (Sep–Oct)'); ax.plot(lc.n_train / 1e6, lc.train_in_sample, marker='o', ls='--', label='training sample')
ax.set(xscale='log', xlabel='training rows (millions, log scale)', ylabel='ROC-AUC', title='Learning curve (XGBoost, planning)'); ax.legend(); save('learning_curve')
""")
md("""
**Finding: no, we are not under-training.**
* Validation AUC rises from 0.711 (5 %, 0.3 M rows) to 0.723 (25 %) and 0.726 (50 %), then only **+0.0006** more at 100 % (6.4 M rows).
* The train–validation gap narrows from 0.14 to 0.065, so the model is not over-fitting either.
* The curve is flat: holding out validation and test months costs essentially nothing in *amount* of data.

What does matter is how **recent** the data is. The same model trained on Jan–Oct (final refit) scores 0.755 on Nov–Dec, vs 0.730 when trained on
Jan–Aug only (§9, §11).
**Decision.** In deployment, retrain on the **most recent months** (rolling refit), not on "as much history as possible".
""")

md("""
## 11. Walk-forward: month by month
**What (`experiments.walkfwd`):** for each month M from May to December, train on all 2025 months before M and test on M.
**Why:** a single test window could be lucky. This mirrors how an agency would retrain monthly, and shows whether autumn and winter months are
systematically harder.
""")
code("""
W = pd.read_csv(REPORTS / 'exp_walk_forward.csv')
display(W[['test_month', 'train_months', 'n_train', 'roc_auc', 'pr_auc', 'base_rate', 'mean_prob']].round(4))
fig, ax = plt.subplots(figsize=(9, 4)); ax2 = ax.twinx()
ax.plot(W.test_month, W.roc_auc, marker='o', label='ROC-AUC'); ax2.bar(W.test_month, W.base_rate * 100, alpha=.2, color='grey')
ax.set(xlabel='test month (2025)', ylabel='ROC-AUC', title='Walk-forward monthly evaluation (bars = % delayed)'); ax2.set_ylabel('% delayed'); save('walk_forward')
""")
md("""
**Finding.**
* AUC is stable at **0.74–0.75 every month except September**, which drops to **0.715**. September is the month the MBTA's fall schedule
  starts, together with the school and university term. The model trained on Jan–Aug has never seen that schedule, and the adversarial
  check in notebook 04 flagged exactly these schedule features.
* Once September is in the training data, October to December recover to 0.752–0.754.
* Calibration follows the level: mean predicted vs actual is within 4.7 pp every month, and within 1.7 pp from July on.

**Decision (retraining policy).** Retrain **monthly**, and in addition **immediately after each schedule change**, e.g. after the first 2–3
weeks of a new schedule. This seasonal effect is a *schedule* effect more than a *weather* effect.
""")

md("""
## 12. Seasonal window: Jan–Aug 2025 vs Sep 2024 – Aug 2025
**What (`experiments.window`):** A = Jan–Aug 2025 (8 months). B = Sep 2024 – Aug 2025 (12 months, so it **includes an autumn, a DST change and a
holiday season**). Same sample size. B-large uses 3 M rows. All are tested on Sep–Oct and Nov–Dec 2025.
**Why:** this directly answers the concern that training on Jan–Aug misses the climate, daylight and holiday conditions of the test months.
""")
code("""
Wn = pd.read_csv(REPORTS / 'exp_window.csv')
display(Wn.pivot_table(index='window', columns='eval_on', values=['roc_auc', 'pr_auc', 'mean_prob']).round(4))
""")
md("""
**Finding: the 12-month window is *worse*, not better.**

| Window | Sep–Oct AUC | Nov–Dec AUC | Nov–Dec PR-AUC |
|---|---|---|---|
| A: Jan–Aug 2025 (8 mo) | **0.724** | **0.732** | **0.481** |
| B: Sep 2024 – Aug 2025 (12 mo, same size) | 0.711 | 0.721 | 0.462 |
| B-large (12 mo, 3 M rows) | 0.713 | 0.721 | 0.463 |

Including last autumn's DST change and holiday season does **not** help. The extra 2024 months come from a different regime: 37 % delayed
vs 29 %, an older schedule, and route changes. They teach the model patterns that no longer hold. A larger sample does not fix this,
which matches the flat learning curve. Calibration is unaffected in all three (mean prediction 23.8–24.1 % vs 25.1 % observed), because the
rolling history features carry the current level.
**Decision.** Your concern about seasonality is real, but the evidence says to handle it with **recency** (monthly retraining + daylight/calendar
features), not by adding last year's same season. The thesis reports this as a negative result.
""")

md("""
## 13. Drift: how fast does a 2024 model decay?
**What (`experiments.drift`):** train on 2024 only and score every 2025 month.
**Why:** 2024 had 37.3 % delays vs 28.8 % in 2025. This tests how long a model can run without retraining. The answer drives the
retraining policy in the deployment section.
""")
code("""
D = pd.read_csv(REPORTS / 'exp_drift.csv')
D['month'] = D.eval_on.str[-2:].astype(int)
display(D[['eval_on', 'roc_auc', 'pr_auc', 'base_rate', 'mean_prob']].round(4))
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(D.month, D.roc_auc, marker='o', label='trained on 2024'); axes[0].plot(W.test_month, W.roc_auc, marker='s', label='walk-forward 2025')
axes[0].set(title='ROC-AUC by 2025 month', xlabel='month'); axes[0].legend()
axes[1].plot(D.month, D.mean_prob * 100, marker='o', label='mean predicted % (2024 model)'); axes[1].plot(D.month, D.base_rate * 100, marker='s', label='actual %')
axes[1].set(title='Calibration drift', xlabel='month', ylabel='% delayed'); axes[1].legend(); save('drift')
""")
md("""
**Finding.** A model trained only on 2024 reaches 0.64–0.70 AUC on 2025 months. It decays from 0.67–0.70 in Q1 to **0.638 in September**,
about **0.07–0.10 below** a model retrained on recent 2025 data (walk-forward). Its *calibration*, however, stays good: mean prediction within
6 pp of the actual rate every month, and within 0.5 pp from September on. The leakage-safe rolling rates adjust the overall level
automatically, while the *ranking* of risky trips (which routes and stops are late at which times) changes as schedules change.
**Decision.** Rolling history features keep the probabilities trustworthy between retrains. Ranking quality needs retraining, consistent with the
monthly + schedule-change policy from §11.
""")

md("""
## 14. Phase 3 summary

| Question | Answer |
|---|---|
| Best model | **XGBoost** in both modes: planning ROC-AUC 0.755 / PR-AUC 0.526; operational 0.978 / 0.954 (test Nov–Dec, 95 % day-block CIs ±0.007 / ±0.025) |
| Gain over agency averages (B0) | planning **+0.10 AUC, +0.15 PR-AUC**; operational +0.09 AUC, +0.18 PR-AUC over the "late stays late" rule |
| Model ranking | XGB ≈ LGB > RF > LR > DT > B0 in planning (significant except XGB vs LGB); in operational every model exceeds 0.96 |
| What drives day-ahead risk | time × calendar (+0.044 AUC), history, route/stop/schedule; weather/events/daylight ≤ 0.003 on average; on affected rows weather adds +0.005–0.007, events −0.01 (negative result) |
| Value of live data | +0.25 AUC; equal to planning at the first timepoint, 0.98–0.99 from the second |
| Are we under-training? | No: the learning curve is flat beyond 50 % of train (+0.0006 AUC from 3.2 M → 6.4 M rows) |
| Seasonal shift | September (new schedule) is the only weak month (0.715); a 12-month window *hurts* (−0.011); recency matters more than volume or seasonal coverage |
| Drift | a 2024 model loses 0.07–0.10 AUC but stays calibrated → monthly retraining + retrain after schedule changes |
| Calibration | planning XGB 26.1 % predicted vs 25.1 % actual; operational within 0.2 pp |
| Operating points | rider alert: planning ≥ 0.60 (precision 0.70); dispatcher alert: operational ≥ 0.50 (precision 0.93, recall 0.82) |

**Next (Phase 4):** explain the XGBoost models with SHAP (global, grouped and per-trip), extract readable rules, and generate OpenAI narratives
for dispatchers and riders.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parent / "05_Models.ipynb"
nbf.write(nb, out)
print("wrote", out)
