"""Builds notebooks/06_Explainability.ipynb. Execute with:
    ..\\.venv\\Scripts\\jupyter nbconvert --to notebook --execute --inplace 06_Explainability.ipynb
Requires: python -m src.xai.shap_analysis ; python -m src.xai.rules ; python -m src.xai.evaluate_narratives
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
F = {}   # findings text, filled in from the verified outputs
exec(open(Path(__file__).parent / "xai_findings.py", encoding="utf-8").read(), F)

md("""
# 06 — Explainability (Phase 4)

**Research questions answered here**
1. *Why* does the model think a bus will be late? Which factors matter overall, and for one specific trip?
2. Can the model be summarised as simple **if-then rules** a planner can read, and how faithful are they?
3. *What if*: which planner-controllable change (running time, frequency) would lower the risk, and how do weather or a live delay change it?
4. Are the explanations **stable** (do two different boosting libraries agree on why)?
5. Can an LLM (OpenAI) turn the explanation into plain language **without inventing facts**, and what happens when the key fails?

> **What** · **Why** · **How** · **Result** · **Finding** · **Decision**

| Tool | Answers | Code |
|---|---|---|
| TreeSHAP (exact, XGBoost native) | global + local "why" | `src/xai/shap_analysis.py` |
| Surrogate tree (depth 4) | readable rules + fidelity | `src/xai/rules.py` |
| What-if | planner / dispatcher scenarios | `src/xai/counterfactual.py` |
| Narrator + OpenAI | plain-language text, fact check, fallback | `src/xai/narrator.py`, `src/xai/llm_client.py` |
""")

code("""
import sys, json, warnings
sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, shap
from src.config import REPORTS, FIG_DIR
from src.encoding import feature_list, apply_levels
from src.xai.feature_text import label
from src.xai.shap_analysis import XAI_DIR, load_model, threshold
sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi'] = 110
def save(n): plt.tight_layout(); plt.savefig(FIG_DIR / f'p4_{n}.png', bbox_inches='tight'); plt.show()
S = {m: pd.read_parquet(XAI_DIR / f'shap_{m}.parquet') for m in ('planning', 'operational')}
FEATS = {m: feature_list(m) for m in S}
def shap_matrix(m): return S[m][[f'shap__{f}' for f in FEATS[m]]].values
print({m: len(d) for m, d in S.items()}, 'rows; base value p =', {m: round(1/(1+np.exp(-d.base_value.iloc[0])), 3) for m, d in S.items()})
""")

md("""
## 1. Sanity check: SHAP values add up to the model output
**What:** for every explained row, base value + Σ SHAP = model log-odds (the *local accuracy* property).
**Why:** if this fails, every later chart is wrong. It is asserted in `shap_analysis.py`; re-checked here.
**How:** 50,000 test events (Nov–Dec), stratified by the label, explained with XGBoost's exact TreeSHAP (`pred_contribs=True`).
""")
code("""
for m in S:
    d = S[m]
    err = np.abs(shap_matrix(m).sum(1) + d.base_value - d.logit).max()
    print(f'{m:12s} max |base + sum(SHAP) - logit| = {err:.2e}')
""")

md("""
## 2. Global importance: what drives delay risk overall?
**What:** mean |SHAP| per feature (log-odds), and summed per feature group.
**Why:** SHAP importance is measured on held-out data and is consistent (unlike split counts / gain); grouping answers the
project's research question (which data source matters) from the model's own point of view, complementing the P3 ablation.
""")
code("""
G = {m: pd.read_csv(XAI_DIR / f'shap_global_{m}.csv') for m in S}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, m in zip(axes, S):
    g = G[m].head(15).iloc[::-1]
    ax.barh([label(f) for f in g.feature], g.mean_abs_shap, color='steelblue')
    ax.set(title=f'{m}: top 15 features (mean |SHAP|, log-odds)', xlabel='mean |SHAP|')
save('global_importance')
pd.concat({m: G[m].head(10).set_index('rank')[['feature', 'group', 'share_%']] for m in S}, axis=1).round(2)
""")
code("""
GR = pd.concat({m: pd.read_csv(XAI_DIR / f'shap_groups_{m}.csv').set_index('group')['share_%'] for m in S}, axis=1).round(1)
ax = GR.sort_values('planning').plot.barh(figsize=(8, 4.5)); ax.set(xlabel='share of total |SHAP| (%)', title='Feature groups: share of explanation')
save('group_importance'); GR.sort_values('planning', ascending=False)
""")
md(F["global"])

md("""
## 3. Direction of effects (beeswarm)
**What:** each dot is one stop event; colour = feature value, x = SHAP (push towards late > 0).
**Why:** importance says *how much*, the beeswarm says *which way*, e.g. high historical late rate → higher risk.
""")
code("""
for m in S:
    num = [f for f in FEATS[m] if S[m][f].dtype.kind in 'fiub']
    idx = [FEATS[m].index(f) for f in num]
    sub = S[m].sample(8000, random_state=0)
    ex = shap.Explanation(values=sub[[f'shap__{f}' for f in num]].values, data=sub[num].astype(float).values,
                          feature_names=[label(f) + f' [{f}]' for f in num])
    shap.plots.beeswarm(ex, max_display=12, show=False); plt.title(f'{m}: SHAP beeswarm (numeric features)'); save(f'beeswarm_{m}')
""")

md("""
## 4. Dependence: *how* do the main factors change the risk?
**What:** SHAP value vs feature value for the key planning drivers (+ the live delay in operational mode).
**Why:** answers questions like "from what historical late rate does risk take off?" or "does a tight schedule (min/km) really hurt?"
""")
code("""
plan = ['hist_stop_hour', 'recent_route_28d', 'schedule_tightness', 'avg_load', 'precip_3h', 'sun_elevation']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
d = S['planning'].sample(15000, random_state=1)
for ax, f in zip(axes.flat, plan):
    x = d[f].astype(float); q = x.quantile([0.01, 0.99])
    ok = x.between(*q)
    ax.scatter(x[ok], d[f'shap__{f}'][ok], s=3, alpha=0.3, c=d.hour[ok], cmap='viridis')
    ax.axhline(0, color='k', lw=0.6); ax.set(xlabel=label(f), ylabel='SHAP (log-odds)', title=f)
save('dependence_planning')
""")
code("""
d = S['operational'].sample(15000, random_state=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, f in zip(axes, ['prev_tp_delay', 'prev_bus_gap_ratio']):
    x = d[f].astype(float); ok = x.between(*x.quantile([0.01, 0.99]))
    ax.scatter(x[ok], d[f'shap__{f}'][ok], s=3, alpha=0.3); ax.axhline(0, color='k', lw=0.6)
    ax.axvline(5, color='r', ls=':') if f == 'prev_tp_delay' else None
    ax.set(xlabel=label(f), ylabel='SHAP (log-odds)', title=f'operational: {f}')
save('dependence_operational')
""")
md(F["dependence"])

md("""
## 5. Why are evening peaks late? SHAP by time period
**What:** average signed SHAP per feature group, by period of the day (planning model).
**Why:** turns "the PM peak is worse" into *which* factors make it worse, the kind of statement a planner needs.
""")
code("""
from src.xai.shap_analysis import group_of
gm = group_of(); d = S['planning']
grp = pd.DataFrame({g: d[[f'shap__{f}' for f in FEATS['planning'] if gm[f] == g]].sum(1) for g in sorted(set(gm[f] for f in FEATS['planning']))})
grp['period'] = d.period.astype(str).values
per = grp.groupby('period').mean().reindex(['early', 'am_peak', 'midday', 'pm_peak', 'evening', 'night'])
late = d.groupby(d.period.astype(str)).agg(predicted=('prob', 'mean'), actual=('delayed', 'mean')).reindex(per.index)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [2.2, 1]})
sns.heatmap(per.T, annot=True, fmt='+.2f', cmap='RdBu_r', center=0, ax=axes[0]); axes[0].set(title='mean SHAP by group and period (log-odds; + = later)')
late.plot.bar(ax=axes[1], rot=0); axes[1].set(title='late rate (stratified sample)')
save('shap_by_period'); late.round(3)
""")
code("""
# the same question per route: which routes are high-risk in which period (test predictions, all rows)
R = pd.read_csv(XAI_DIR / 'route_period_risk.csv')
R.head(12).round(3)
""")
md(F["period"])

md("""
## 6. Local explanations: one trip at a time
**What:** waterfall plots for three real test events: a confident *late* call, a confident *on-time* call and a borderline one.
**Why:** this is exactly what a dispatcher sees in the dashboard: the prediction and the few factors that produced it.
""")
code("""
d = S['planning']; thr = threshold('planning')
cases = {'high risk': d.prob.idxmax(), 'low risk': d.prob.idxmin(), 'borderline': (d.prob - thr).abs().idxmin()}
for name, i in cases.items():
    r = d.loc[i]
    ex = shap.Explanation(values=r[[f'shap__{f}' for f in FEATS['planning']]].values.astype(float), base_values=float(r.base_value),
                          data=[str(r[f]) for f in FEATS['planning']], feature_names=[label(f) for f in FEATS['planning']])
    print(f"{name}: route {r.route_id} {r.direction_id}, stop {r.stop_id}, {r.service_date:%Y-%m-%d} {int(r.hour):02d}:00 -> p={r.prob:.2f}, actual delayed={int(r.delayed)}")
    shap.plots.waterfall(ex, max_display=10, show=False); plt.title(f'planning · {name}'); save(f'waterfall_{name.replace(" ", "_")}')
""")

md("""
## 7. Readable rules: a surrogate decision tree
**What:** a depth-4 regression tree trained to mimic the XGBoost probability using only readable features.
**Why:** planners want statements like "evening peak + stop usually late → high risk". A surrogate explains the *model*, so we must
report **fidelity**: R² against XGBoost probabilities and ROC-AUC on the real labels, measured on test rows the surrogate never saw
(it was fitted on validation).
""")
code("""
FID = pd.read_csv(XAI_DIR / 'surrogate_fidelity.csv'); display(FID.round(3))
from src.xai.rules import rule_sentence
L = pd.read_csv(XAI_DIR / 'rule_leaves_planning.csv')
L['rule (plain)'] = L.rule.map(rule_sentence)
display(L[['surrogate_p', 'actual_late', 'share_%', 'rule (plain)']].round(3).head(8))
print(open(XAI_DIR / 'rules_planning.txt', encoding='utf-8').read()[:2500])
""")
md(F["rules"])

md("""
## 8. What-if: what could a planner change?
**What:** for the high-risk event from §6: (a) add scheduled running time on the segment, (b) change the planned headway,
(c) different weather. Operational: how the risk grows with the delay at the previous timepoint.
**Why:** an explanation is most useful when it points to an action. **Caveat:** these are *model-based* what-ifs, i.e. associations
learned from 2025 data, not causal effects.
""")
code("""
from src.xai.counterfactual import what_if, weather_scenarios, minimal_schedule_change, schedule_fix_population, delay_tipping_points
mp = load_model('planning'); d = S['planning']; thr = threshold('planning')
# Decision: the single highest-risk event is a 01:00 night trip with p~1.0, which is not actionable and not typical.
# Use a realistic planner case instead: weekday daytime (7-19 h), clearly above the alert threshold but not saturated (0.5-0.8),
# with a known segment running time of at least 3 scheduled minutes.
cand = d[d.seg_sched_min.ge(3) & d.hour.between(7, 19) & (d.is_weekend == 0) & d.prob.between(0.5, 0.8)].sort_values('prob', ascending=False)
row = apply_levels(cand.iloc[[0]].copy()); r0 = row.iloc[0]
print(f"case: route {r0.route_id} {r0.direction_id}, stop {r0.stop_id}, {r0.service_date:%Y-%m-%d} {int(r0.hour):02d}:00, scheduled {r0.seg_sched_min:.1f} min over {r0.segment_km:.2f} km -> p={r0.prob:.2f}")
res = minimal_schedule_change(mp, row, 'planning', threshold=thr)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != 'curve'})
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
c = res['curve']; axes[0].plot(c.added_min, c.prob, marker='o'); axes[0].axhline(thr, color='r', ls=':')
axes[0].set(xlabel='extra scheduled minutes on segment', ylabel='P(> 5 min late)', title='(a) add running time')
hw = what_if(mp, row, 'planning', 'scheduled_headway', [5, 8, 10, 12, 15, 20, 30])
axes[1].plot(hw.scheduled_headway, hw.prob, marker='o'); axes[1].set(xlabel='planned headway (min)', title='(b) planned frequency')
ws = weather_scenarios(mp, row, 'planning'); axes[2].bar(ws.scenario, ws.prob); axes[2].set(title='(c) weather scenario')
save('what_if_planning'); ws.round(3)
""")
code("""
# Population view: all high-risk daytime events (p >= alert threshold) with a known segment time in the SHAP sample
hi = apply_levels(d[d.seg_sched_min.ge(1) & d.hour.between(6, 21) & (d.prob >= thr)].copy())
fix = schedule_fix_population(mp, hi, 'planning', thr)
share = {f'<= {a} min': round(float((fix.min_add <= a).mean() * 100), 1) for a in (1, 2, 3, 5, 10)}
print(f'{len(hi):,} high-risk daytime events; share pushed below the alert threshold by adding ...', share)
mean_curve = fix[[c for c in fix if c.startswith('p_+')]].mean()
print('mean probability after adding minutes:', mean_curve.round(3).to_dict())
WF = pd.Series(share, name='% of high-risk events fixed'); WF.to_frame()
""")
code("""
mo = load_model('operational'); o = S['operational']; othr = threshold('operational')
rows = apply_levels(o[o.prev_tp_delay.notna()].sample(2000, random_state=3).copy())
C, tip = delay_tipping_points(mo, rows, 'operational', othr)
fig, ax = plt.subplots(figsize=(8, 4))
q = C.quantile([0.25, 0.5, 0.75])
ax.fill_between(C.columns, q.loc[0.25], q.loc[0.75], alpha=0.3, label='25-75 % of events'); ax.plot(C.columns, q.loc[0.5], label='median event')
ax.axvline(5, color='r', ls=':'); ax.axhline(othr, color='g', ls=':', label=f'alert threshold {othr:.2f}')
ax.set(xlabel='delay at previous timepoint (min)', ylabel='P(> 5 min late here)', title='operational what-if: how much live delay tips the alert?')
ax.legend(); save('what_if_operational')
print('tipping point (min of previous-timepoint delay) over 2,000 events:'); print(tip.describe(percentiles=[.1, .25, .5, .75, .9]).round(2))
print('share never crossing within -2..15 min:', f'{tip.isna().mean():.1%}')
""")
md(F["whatif"])

md("""
## 9. Stability: do XGBoost and LightGBM agree on *why*?
**What:** Spearman rank correlation of global feature importance (mean |SHAP|) between the two boosting libraries, and the overlap of
their top-10 features, on the same 50 k test rows.
**Why:** if two equally accurate models explain the data very differently, explanations would be an artefact of one model. Agreement
supports trusting them.
""")
code("""
ST = pd.concat([pd.read_csv(XAI_DIR / f'shap_stability_{m}.csv') for m in S]); display(ST.round(3))
for m in S:
    print(m, '- share of |SHAP| by group (%)'); display(pd.read_csv(XAI_DIR / f'shap_stability_groups_{m}.csv', index_col=0).sort_values('xgb_%', ascending=False).T)
# operational: the features where the two libraries split credit differently
pd.read_csv(XAI_DIR / 'shap_stability_features_operational.csv', index_col=0).head(10)
""")
md(F["stability"])

md("""
## 10. Plain-language explanations with OpenAI (and what happens when the key fails)
**What:** facts (top SHAP factors converted to percentage-point effects) → template text (always) → OpenAI rewrite for a dispatcher,
planner or rider → automatic **faithfulness check** (every number must exist in the facts; the main factor must be named) → fallback.
**Why:** SHAP plots are for analysts; dispatchers and riders need one sentence. LLMs can hallucinate, so the LLM never sees the raw data,
only the fact sheet, and its output is checked.
**Key-failure handling:** `llm_client.health_check()` classifies errors (missing key, invalid/expired key 401, quota 429, rate limit, network,
unknown model), logs to `logs/llm.log` (never the key) and the dashboard shows a warning banner while falling back to templates.
""")
code("""
from src.xai import llm_client, narrator
st = llm_client.health_check(force=True)
print('OpenAI status:', st.code, '-', st.message, f'(model {st.model})')
V1 = pd.read_csv(XAI_DIR / 'narrative_eval_v1_strict.csv')
print('v1 (exact-number check): LLM drafts rejected =', f'{(V1.status == "unfaithful").mean():.0%}')
EV = pd.read_csv(XAI_DIR / 'narrative_eval.csv')
print('v2 (rounding allowed):  LLM drafts rejected =', f'{(EV.status == "unfaithful").mean():.0%}', '| final texts faithful =', f'{EV.ok.mean():.0%}')
print(EV.groupby(['audience', 'source']).agg(n=('ok', 'size'), faithful=('ok', 'mean'), words=('words', 'mean')).round(2))
for _, r in EV.head(3).iterrows():
    print(f"\\n[{r.audience} · {r.source}] {r.text}")
""")
code("""
# fallback demo: simulate an expired key without touching the real one
import os
saved = os.environ.get('OPENAI_API_KEY', '')
os.environ['OPENAI_API_KEY'] = ''
f = json.loads(EV.facts.iloc[0])
r = narrator.llm_explanation(f, 'dispatcher')
print('status:', r['status'], '| source:', r['source']); print('warning shown in app:', r['warning']); print('text:', r['text'])
os.environ['OPENAI_API_KEY'] = saved; llm_client._STATUS = None
""")
md(F["llm"])

md(F["summary"])

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parent / "06_Explainability.ipynb"
nbf.write(nb, out)
print("wrote", out)
