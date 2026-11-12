"""Builds notebooks/07_Extensions.ipynb. Execute with:
    ..\\.venv\\Scripts\\jupyter nbconvert --to notebook --execute --inplace 07_Extensions.ipynb
Requires: python -m src.extensions.propagation ; chronic ; severity ; regression ; bunching  (outputs in reports/ext/)
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
F = {}   # findings text, written after checking the executed outputs
exec(open(Path(__file__).parent / "ext_findings.py", encoding="utf-8").read(), F)

md("""
# 07 - Extensions (Phase 5)

The core model answers one yes/no question: *will this bus be more than 5 minutes late at this timepoint?*
Agency users ask more than that. This notebook answers five follow-up questions with the same data and pipeline:

| # | Question (user) | Method | Code |
|---|---|---|---|
| 1 | Does delay **carry along a trip**, and do buses recover? (dispatcher, planner) | descriptive propagation analysis | `src/extensions/propagation.py` |
| 2 | Which segments are **chronically** late, and how much running time is missing? (planner) | persistent-segment detection + out-of-time check | `src/extensions/chronic.py` |
| 3 | **How late** - on time, minor, moderate or major? (dispatcher, rider comms) | 4-class XGBoost | `src/extensions/severity.py` |
| 4 | **How many minutes** late, with an interval? (rider comms) | XGBoost median + 80 % quantile interval | `src/extensions/regression.py` |
| 5 | Will buses **bunch**? (dispatcher) | bunching labels from headways + XGBoost | `src/extensions/bunching.py` |

> **What** -> **Why** -> **How** -> **Result** -> **Finding** -> **Decision**

**Protocol for the models (decision 19 in the decision log):** train on a stratified 2 M-row sample of Jan-Aug, early-stop on 300 k rows of
Sep-Oct, score the full Nov-Dec test period once. Tree settings are re-used from the Phase 3 Optuna search (no new tuning), so the numbers
are slightly conservative compared with the Phase 3 models, which were refitted on all of Jan-Oct.
""")

code("""
import sys, warnings
sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src.config import REPORTS, FIG_DIR, SEVERITY_BINS, SEVERITY_LABELS
EXT = REPORTS / 'ext'
sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi'] = 110
pd.set_option('display.width', 200, 'display.max_columns', 30)
def save(n): plt.tight_layout(); plt.savefig(FIG_DIR / f'p5_{n}.png', bbox_inches='tight'); plt.show()
def rd(n, **k): return pd.read_csv(EXT / f'{n}.csv', **k)
print(sorted(p.name for p in EXT.iterdir()))
""")

# ------------------------------------------------------------------ 1 propagation
md("""
## 1. Delay propagation along the trip
**What:** how a bus's delay at one timepoint relates to its delay at the next, how delay grows along the route, and how often a late bus recovers.
**Why:** the operational model's biggest gain comes from `prev_tp_delay` (Phase 4: 24.9 % of SHAP importance). This section shows the
mechanism behind that, and tells dispatchers *when* intervening is worth it (if late buses rarely recover, act early).
**How:** all 2025 stop events, ordered within each trip by `time_point_order`. *Recovery* = late (> 5 min) at one timepoint and not late
at the next. Bus-to-bus: correlation of this bus's delay with the delay of the previous bus at the same stop.
""")
code("""
ov = rd('prop_overall', index_col=0).iloc[:, 0]
print(ov.round(3).to_string())
g = rd('prop_growth'); ls = rd('prop_late_start'); tr = rd('prop_transition', index_col=0)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(g.pos_decile, g.mean_delay, marker='o', label='mean delay (min)'); ax[0].plot(g.pos_decile, g.median_delay, marker='s', label='median delay (min)')
a2 = ax[0].twinx(); a2.plot(g.pos_decile, g.late_rate * 100, color='C3', marker='^', label='late rate %'); a2.set_ylabel('late rate %'); a2.grid(False)
ax[0].set(xlabel='position along the route (% of timepoints)', ylabel='minutes', title='Delay builds up, then the terminal pads it'); ax[0].legend(loc='upper left')
sns.barplot(data=ls, x='start_band', y='p_late_at_end', ax=ax[1], color='C0')
for i, v in enumerate(ls.p_late_at_end): ax[1].text(i, v + 0.01, f'{v:.0%}', ha='center')
ax[1].set(xlabel='delay at the first timepoint', ylabel='P(late at the last timepoint)', title='A late start rarely recovers')
sns.heatmap(tr, annot=True, fmt='.2f', cmap='Blues', ax=ax[2], cbar=False)
ax[2].set(title='Severity at next timepoint given previous', xlabel='next timepoint', ylabel='previous timepoint')
save('propagation')
ls
""")
code("""
rp = rd('prop_recovery_period'); rr = rd('prop_recovery_route', dtype={'route_id': str}); bb = rd('prop_bus_to_bus'); sa = rd('prop_segment_added', dtype={'route_id': str})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
sns.barplot(data=rr.sort_values('recovery_rate'), x='route_id', y='recovery_rate', ax=ax[0], color='C2')
ax[0].axhline(ov['recovery_rate_overall'], ls='--', c='k'); ax[0].set(title='Recovery rate by route (late -> not late at next TP)', ylabel='recovery rate')
b = bb.melt(id_vars='period', value_vars=['corr_with_prev_timepoint', 'corr_with_bus_ahead'])
sns.barplot(data=b, x='period', y='value', hue='variable', ax=ax[1]); ax[1].set(title='What predicts this delay better?', ylabel='correlation')
sns.barplot(data=sa, x='route_id', y='share_segments_adding_2min', ax=ax[2], color='C1')
ax[2].set(title='Share of segments adding >= 2 min', ylabel='share')
save('propagation_recovery')
print(rp.round(3).to_string(index=False))
""")
md(F["prop"])

# ------------------------------------------------------------------ 2 chronic
md("""
## 2. Chronic delay segments (planner's fix-list)
**What:** segments (route x direction x timepoint) that are late consistently, not just in a bad week.
**Why:** planners can change running times, stop spacing or bus priority only in a few places per schedule change; they need a ranked,
*persistent* list, and they need to know whether a segment **creates** delay or only **inherits** it from upstream.
**How:** Jan-Oct 2025. Chronic = late rate >= 1.5 x network average in >= 80 % of months. Verified out of time on Nov-Dec. *Added delay* =
delay here minus delay at the previous timepoint (median per segment). Running-time suggestion = median added delay per period, for chronic
segments only.
""")
code("""
cs = rd('chronic_summary', index_col=0).iloc[:, 0]; print(cs.round(3).to_string())
seg = rd('chronic_segments', dtype={'route_id': str, 'stop_id': str}); cr = rd('chronic_by_route', dtype={'route_id': str})
adj = rd('chronic_adjustments', dtype={'route_id': str, 'stop_id': str})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
ok = seg.dropna(subset=['late_rate_test'])
sns.scatterplot(data=ok, x='late_rate', y='late_rate_test', hue='chronic', ax=ax[0], s=25)
ax[0].plot([0, 1], [0, 1], 'k--', lw=1); ax[0].set(xlabel='late rate Jan-Oct', ylabel='late rate Nov-Dec', title='Chronic segments stay bad out of time')
sns.scatterplot(data=seg, x='mean_delay', y='median_added', hue='chronic', ax=ax[1], s=25)
ax[1].axhline(0, c='k', lw=1); ax[1].set(xlabel='mean delay at the timepoint (min)', ylabel='median delay ADDED on the segment (min)', title='Created vs inherited delay')
sns.barplot(data=cr[cr.chronic > 0], x='route_id', y='chronic', ax=ax[2], color='C3'); ax[2].set(title='Chronic segments by route', ylabel='count')
save('chronic')
ch = seg[seg.chronic]
print('chronic segments:', len(ch), '| median mean_delay %.2f min | median added %.2f min | share adding >= 1 min: %.0f %%' % (
      ch.mean_delay.median(), ch.median_added.median(), (ch.median_added >= 1).mean() * 100))
print('suggested additions (chronic segment x period):', len(adj), '| > 0: %d | >= 1 min: %d | max %.1f min' % (
      (adj.suggest_add_min > 0).sum(), (adj.suggest_add_min >= 1).sum(), adj.suggest_add_min.max()))
adj.head(10).round(2)
""")
md(F["chronic"])

# ------------------------------------------------------------------ 3 severity
md("""
## 3. Severity: how late? (4 classes)
**What:** on time (< 1 min), minor (1-5), moderate (5-10), major (> 10 min late).
**Why:** a 6-minute and a 25-minute delay are both "late" for the binary model, but a dispatcher acts very differently on them.
**How:** XGBoost `multi:softprob`. Baselines: **B0** = class of the stop's usual delay at this hour; for operational also **B1 persistence** =
the class the bus is in at the previous timepoint (computed below). Metrics: macro-F1 (imbalanced classes), quadratic-weighted kappa
(ordered classes), recall of *major*.
""")
code("""
sev = rd('severity_metrics')
from src.encoding import load_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
te = load_split('test', ['prev_tp_delay', 'severity'])
y = pd.Categorical(te.severity.astype(str), categories=SEVERITY_LABELS).codes
pers = pd.cut(te.prev_tp_delay.fillna(0) * 60, SEVERITY_BINS, labels=False, right=False).astype(int).values
rep = classification_report(y, pers, labels=range(4), target_names=SEVERITY_LABELS, output_dict=True, zero_division=0)
b1 = {'mode': 'operational', 'model': 'B1 persistence (prev timepoint class)', 'accuracy': accuracy_score(y, pers),
      'macro_f1': f1_score(y, pers, average='macro'), 'qw_kappa': cohen_kappa_score(y, pers, weights='quadratic'),
      **{f'recall_{c}': rep[c]['recall'] for c in SEVERITY_LABELS}, **{f'precision_{c}': rep[c]['precision'] for c in SEVERITY_LABELS}}
sev = pd.concat([sev, pd.DataFrame([b1])], ignore_index=True)
sev.to_csv(EXT / 'severity_metrics_with_persistence.csv', index=False)
print('test rows', len(te), '| class shares', np.bincount(y, minlength=4) / len(y))
cols = ['mode', 'model', 'accuracy', 'macro_f1', 'qw_kappa', 'recall_major', 'precision_major', 'auc_gt5_from_multiclass']
sev[cols].round(3)
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, m in zip(ax, ('planning', 'operational')):
    cm = rd(f'severity_confusion_{m}', index_col=0)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Purples', ax=a, cbar=False, vmin=0, vmax=1)
    a.set(title=f'{m}: confusion (row-normalised = recall)', xlabel='predicted', ylabel='actual')
save('severity_confusion')
""")
md(F["severity"])

# ------------------------------------------------------------------ 4 regression
md("""
## 4. Delay in minutes, with an 80 % interval
**What:** a point prediction of the delay (minutes) and an interval that should contain the true delay 80 % of the time.
**Why:** rider messages are better as "expected 3-9 min late" than as "62 % chance of being late".
**How:** XGBoost with absolute-error loss (predicts the conditional **median**: robust to the long tail of 30-60 min incidents) and a
multi-quantile XGBoost (10 % / 90 %) on 1 M random training rows. Target clipped to [-10, 60] min. Baselines: B0 = stop's usual delay at
this hour; B1 (operational) = persistence, the delay at the previous timepoint.
""")
code("""
reg = rd('regression_metrics'); print(reg.round(3).to_string(index=False))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for i, m in enumerate(('planning', 'operational')):
    s = pd.read_parquet(EXT / f'regression_sample_{m}.parquet')
    s['y'] = s.delay_min.clip(-10, 60)
    ax[i].hexbin(s.y, s.pred, gridsize=45, bins='log', cmap='viridis', extent=(-10, 40, -10, 40))
    ax[i].plot([-10, 40], [-10, 40], 'r--', lw=1); ax[i].set(xlabel='actual delay (min)', ylabel='predicted median (min)', title=f'{m}: predicted vs actual')
bands = pd.concat([rd(f'regression_by_band_{m}').assign(mode=m) for m in ('planning', 'operational')])
sns.barplot(data=bands, x='band', y='MAE', hue='mode', ax=ax[2]); ax[2].set(title='MAE by true delay band', xlabel='actual delay (min)')
save('regression')
bands.round(2)
""")
code("""
# interval calibration by route (operational): is the 80 % interval honest everywhere?
rows = []
for m in ('planning', 'operational'):
    s = pd.read_parquet(EXT / f'regression_sample_{m}.parquet'); s['y'] = s.delay_min.clip(-10, 60)
    s['inside'] = (s.y >= s.lo) & (s.y <= s.hi)
    rows.append(s.groupby(s.route_id.astype(str)).inside.mean().rename(m))
cov = pd.concat(rows, axis=1).round(3)
print('coverage of the 80 % interval by route (20 k test sample):'); print(cov.describe().round(3).loc[['min', '50%', 'max']])
""")
md("""
### 4b. Fixing the under-coverage: conformalized quantile regression (CQR)
**Why:** the raw 80 % interval covers less than 80 % of test delays. Quantile trees fitted on a sub-sample with early stopping are known to
under-cover. **How** (`src/extensions/conformal.py`, Romano et al. 2019): on the validation months (Sep-Oct) compute the conformity score
E = max(lo - y, y - hi). Its 80 % quantile Q is added to both ends of every interval. The guarantee holds only if validation and test are
exchangeable, and autumn vs winter are not exactly exchangeable, so we check coverage on Nov-Dec.
""")
code("""
conf = rd('regression_conformal'); print(conf.round(3).to_string(index=False))
cr = rd('regression_conformal_by_route', dtype={'route_id': str})
fig, ax = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
for i, m in enumerate(('planning', 'operational')):
    d = cr[cr['mode'] == m].sort_values('cal')
    ax[i].bar(d.route_id, d.raw, color='lightgray', label='raw'); ax[i].plot(d.route_id, d.cal, 'o', color='C0', label='calibrated (CQR)')
    ax[i].axhline(80, color='r', ls='--', lw=1); ax[i].set(title=f'{m}: 80 % interval coverage by route (test)', ylim=(60, 90), ylabel='coverage %'); ax[i].legend()
save('conformal')
print(cr.pivot(index='route_id', columns='mode', values='cal').round(1).sort_values('planning').head(4))
""")
md(F["regression"])

# ------------------------------------------------------------------ 5 bunching
md("""
## 5. Bus bunching
**What:** *bunched* = actual headway < 25 % of the scheduled headway; *big gap* = > 150 % (decision 20).
**Why:** bunching is the visible symptom of unreliable service on frequent routes: two buses arrive together, then nobody comes. A dispatcher
can hold the second bus if bunching is predicted early.
**How:** headways from the cleaned MBTA file joined on (service_date, half_trip_id, time_point_order). Descriptive rates by route, hour and
position; models: B0 = the gap-ratio feature itself as a score, operational XGB, planning XGB. The sign of B0 needed a diagnosis
(cell below, decision 30).
""")
code("""
bm = rd('bunching_metrics'); print(bm.round(3).to_string(index=False))
br = rd('bunching_by_route_id', dtype={'route_id': str}); bh = rd('bunching_by_hour'); bp = rd('bunching_by_position'); bv = rd('bunching_by_vs_delay')
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
sns.barplot(data=br.sort_values('bunched_rate'), x='route_id', y='bunched_rate', ax=ax[0], color='C4'); ax[0].set(title='Bunching rate by route (test)')
ax[1].plot(bh.hour, bh.bunched_rate, marker='o', label='bunched'); ax[1].plot(bh.hour, bh.big_gap_rate, marker='s', label='big gap')
ax[1].set(title='By hour', xlabel='hour', ylabel='rate'); ax[1].legend()
ax[2].plot(bp.pos_decile, bp.bunched_rate, marker='o', label='bunched'); ax[2].plot(bp.pos_decile, bp.big_gap_rate, marker='s', label='big gap')
ax[2].set(title='Along the route', xlabel='position (% of route)'); ax[2].legend()
save('bunching')
print(bv.round(3)); print(rd('bunching_importance', index_col=0).iloc[:8].round(0))
""")
md("""
### 5b. Diagnosis: why a LARGE gap ratio predicts bunching
The first run scored B0 as `-prev_bus_gap_ratio` (assuming "small gap now -> bunched later") and obtained ROC-AUC **0.305**, i.e. worse
than random, so the relation runs the other way. In `src/features.py` the feature is the time since the last bus that has **already passed the
target stop**, divided by the planned headway. If the direct leader is close, it has *not* passed the target stop yet and is skipped,
so the feature then describes the bus two ahead and is large. The cell checks this on all test rows.
""")
code("""
from sklearn.metrics import roc_auc_score
from src.encoding import load_split
from src.extensions.bunching import headways, attach
te = attach(load_split('test', ['service_date', 'half_trip_id', 'time_point_order', 'prev_bus_gap_ratio']), headways())
s = te.prev_bus_gap_ratio.fillna(te.prev_bus_gap_ratio.median())
print(f"rows {len(te):,}  AUC(-ratio) = {roc_auc_score(te.bunched, -s):.3f}   AUC(+ratio) = {roc_auc_score(te.bunched, s):.3f}")
bins = pd.cut(te.prev_bus_gap_ratio, [-np.inf, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, np.inf])
u = te.groupby(bins, observed=True).agg(rows=('bunched', 'size'), bunched=('bunched', 'mean'), big_gap=('big_gap', 'mean'),
                                         median_headway_ratio=('hw_ratio', 'median'))
u.to_csv(EXT / 'bunching_by_gap_ratio.csv')
ax = u[['bunched', 'big_gap']].plot(kind='bar', figsize=(10, 3.8), title='Outcome at the target stop by prev_bus_gap_ratio (test)')
ax.set(xlabel='prev_bus_gap_ratio bin', ylabel='rate'); save('bunching_gap_ratio')
del te; u.round(3)
""")
md(F["bunching"])

md(F["summary"])

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).parent / "07_Extensions.ipynb")
print("written", len(C), "cells")
