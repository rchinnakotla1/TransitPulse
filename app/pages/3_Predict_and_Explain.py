from common import (APP, PRESETS, best_threshold, explain_row, llm_warning_banner, model, page, read_parquet, risk_badge,
                    route_trust, shap_bar)
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.encoding import apply_levels
from src.xai import narrator
from src.xai.counterfactual import minimal_schedule_change, weather_scenarios, what_if

page("Predict & Explain", "🔮")
st.caption("Pick a real stop event from 1–14 December 2025 (test period, never seen in training). The model's prediction is compared "
           "with what actually happened.")

preset = st.selectbox("Use case / alert threshold", list(PRESETS))
mode, thr = PRESETS[preset]
thr = thr if thr is not None else best_threshold(mode)
ev = read_parquet(str(APP / f"events_{mode}.parquet"))

c = st.columns(5)
routes = sorted(ev.route_id.unique(), key=lambda x: (len(x), x))
route = c[0].selectbox("Route", routes, index=routes.index("28") if "28" in routes else 0)
e = ev[ev.route_id == route]
day = c[1].selectbox("Date", sorted(e.service_date.dt.date.unique()))
e = e[e.service_date.dt.date == day]
direction = c[2].selectbox("Direction", sorted(e.direction_id.unique()))
e = e[e.direction_id == direction]
hour = c[3].selectbox("Hour", sorted(e.hour.unique()), index=min(len(e.hour.unique()) - 1, 8))
e = e[e.hour == hour].sort_values(["scheduled_dt", "time_point_order"])
if e.empty:
    st.info("No events for this choice."); st.stop()
labels = [f"{r.scheduled_dt:%H:%M} · stop {r.stop_id} · timepoint {int(r.time_point_order)}" for r in e.itertuples()]
pick = c[4].selectbox("Scheduled time · stop", range(len(e)), format_func=lambda i: labels[i])
row = apply_levels(e.iloc[[pick]].copy())
r0 = e.iloc[pick]

s, base, p = explain_row(mode, row)
k = st.columns([1.3, 1, 1, 1.2])
k[0].markdown(f"### {risk_badge(p, thr)}")
k[0].caption(f"chance of arriving > 5 min late · alert threshold {thr:.2f} ({mode} model)")
k[1].metric("Actual delay", f"{r0.delay_min:+.1f} min", "late > 5 min" if r0.delayed else "not late", delta_color="inverse" if r0.delayed else "normal")
k[2].metric("Typical (base) risk", f"{1 / (1 + np.exp(-base)):.0%}")
tr = route_trust(mode, route)
if tr:
    k[3].metric(f"Model trust on route {route}", f"{tr['trust']}", f"test ROC-AUC {tr['auc']:.2f}", delta_color="off")
correct = (p >= thr) == bool(r0.delayed)
st.caption(("✅ The alert decision matched reality." if correct else "❌ The alert decision did not match reality.")
           + " Single events can go either way; see Model Comparison for overall accuracy.")

left, right = st.columns([1.1, 1])
left.plotly_chart(shap_bar(s), width="stretch")

with right:
    st.subheader("Explanation")
    audience = st.radio("Audience", narrator.AUDIENCES, horizontal=True)
    use_llm = st.toggle("Use OpenAI (plain-language rewrite)", value=True)
    facts = narrator.facts_from_shap(r0, s, base, thr, mode)
    key = ("narr", mode, route, str(day), direction, int(hour), pick, audience, use_llm)
    if key not in st.session_state:
        with st.spinner("Writing explanation..."):
            st.session_state[key] = narrator.llm_explanation(facts, audience, use_llm=use_llm)
    res = st.session_state[key]
    llm_warning_banner(res if use_llm else None)
    st.info(res["text"])
    src = {"llm": "🤖 OpenAI, passed the automatic fact check", "template": "📝 template (deterministic)"}[res["source"]]
    st.caption(f"Source: {src} · status: {res['status']}")
    with st.expander("Facts given to the AI (the only information it may use)"):
        st.json(facts)

st.divider()
st.subheader("What if…? (model-based scenarios, not causal effects)")
m = model(mode)
w1, w2, w3 = st.columns(3)
with w1:
    st.markdown("**Add scheduled running time** on the segment before this timepoint")
    if pd.notna(r0.seg_sched_min):
        res2 = minimal_schedule_change(m, row, mode, thr)
        if "curve" in res2:
            st.plotly_chart(px.line(res2["curve"], x="added_min", y="prob", markers=True, height=280)
                            .add_hline(y=thr, line_dash="dot", line_color="red"), width="stretch")
        st.caption(("Adding about **%.1f min** brings the risk below the threshold." % res2["added_min"]) if res2.get("possible") and res2.get("added_min", 0) > 0
                   else res2.get("reason", ""))
    else:
        st.caption("First timepoint of the trip: no previous segment.")
with w2:
    st.markdown("**Weather scenario** for the same trip")
    ws = weather_scenarios(m, row, mode)
    st.plotly_chart(px.bar(ws, x="scenario", y="prob", height=280).add_hline(y=thr, line_dash="dot", line_color="red"),
                    width="stretch")
with w3:
    if mode == "operational":
        st.markdown("**Delay at the previous timepoint** (live information)")
        wi = what_if(m, row, mode, "prev_tp_delay", np.arange(-2, 15.5, 0.5))
        st.plotly_chart(px.line(wi, x="prev_tp_delay", y="prob", height=280).add_hline(y=thr, line_dash="dot", line_color="red"),
                        width="stretch")
    else:
        st.markdown("**Planned headway** (frequency)")
        hw = what_if(m, row, mode, "scheduled_headway", [5, 8, 10, 12, 15, 20, 30])
        st.plotly_chart(px.line(hw, x="scheduled_headway", y="prob", markers=True, height=280)
                        .add_hline(y=thr, line_dash="dot", line_color="red"), width="stretch")
