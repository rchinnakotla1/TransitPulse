"""Page 6: planner insights: chronic segments (verified out of time), running-time suggestions, severity and minutes-late models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st

from common import EXT, XAI, exists, page, read_csv, read_parquet

page("Planner insights", "🗺️")
st.caption("For service planners: where lateness is *chronic* (not one bad week), what a schedule adjustment could do, "
           "and how late buses get when they are late.")

t1, t2, t3, t4 = st.tabs(["Chronic segments", "Running-time suggestions", "How late? (severity & minutes)", "Route × period risk"])

with t1:
    if not exists(EXT / "chronic_segments.csv"):
        st.error("Run `python -m src.extensions.chronic`.")
    else:
        s = read_csv(str(EXT / "chronic_summary.csv"), index_col=0).iloc[:, 0]
        c = st.columns(4)
        c[0].metric("Chronic segments", f"{int(s['chronic_segments'])} / {int(s['segments'])}")
        c[1].metric("Rank agreement Jan–Oct vs Nov–Dec", f"ρ = {s['spearman_janoct_vs_novdec']:.2f}")
        c[2].metric("Top-20 still top quartile later", f"{s['top20_still_top_quartile_%']:.0f} %")
        c[3].metric("Late rate later: chronic vs others", f"{s['chronic_late_rate_test']:.0%} vs {s['other_late_rate_test']:.0%}")
        st.caption("A segment (route × direction × timepoint) is **chronic** if its late rate is ≥ 1.5 × the network rate in ≥ 80 % "
                   "of months (Jan–Oct, at least 500 events). It is then checked on Nov–Dec, which it never saw.")
        cs = read_csv(str(EXT / "chronic_segments.csv"), dtype={"route_id": str, "stop_id": str})
        only = st.toggle("Show chronic segments only", value=True)
        v = cs[cs.chronic] if only else cs
        st.dataframe(v.round(3), hide_index=True, width="stretch",
                     column_config={"late_rate": st.column_config.ProgressColumn("late rate Jan–Oct", min_value=0, max_value=1,
                                                                                 format="%.2f"),
                                    "late_rate_test": "late rate Nov–Dec", "median_added": "median min added on segment"})
        if exists(EXT / "chronic_by_route.csv"):
            r = read_csv(str(EXT / "chronic_by_route.csv"), dtype={"route_id": str})
            st.plotly_chart(px.bar(r, x="route_id", y=["segments", "chronic"], barmode="overlay",
                                   title="Chronic segments by route"), width="stretch")
        st.info("Most chronic segments add little delay themselves (median added ≈ 0–1 min): buses arrive already late. "
                "Fixes therefore belong upstream (earlier segments, terminal departure, layover) more than at the flagged stop.")

with t2:
    if exists(EXT / "chronic_adjustments.csv"):
        a = read_csv(str(EXT / "chronic_adjustments.csv"), dtype={"route_id": str, "stop_id": str})
        st.markdown("**Suggested extra scheduled minutes** for chronic segments by period = median minutes the segment itself "
                    "adds (Jan–Oct). A suggestion of 0 means the segment does not add time: padding it would not help, look upstream.")
        st.dataframe(a.round(2), hide_index=True, width="stretch")
    st.markdown("**What the model says about padding** (Phase 4, 16,103 high-risk daytime stop events):")
    c = st.columns(4)
    for col, (k, v) in zip(c, {"+1 min": "10 %", "+3 min": "22 %", "+5 min": "26 %", "+10 min": "29 %"}.items()):
        col.metric(f"Fixed with {k}", v)
    st.caption("'Fixed' = predicted risk drops below the planning alert threshold. Gains flatten after ~3–4 min: most risk is "
               "inherited from upstream. Model-based estimate, not a causal effect.")

with t3:
    if exists(EXT / "severity_metrics.csv"):
        st.markdown("**Severity classes:** on time (< 1 min) · minor (1–5) · moderate (5–10) · major (> 10 min late)")
        f = EXT / "severity_metrics_with_persistence.csv"
        f = f if exists(f) else EXT / "severity_metrics.csv"
        st.dataframe(read_csv(str(f)).round(3), hide_index=True, width="stretch")
        st.caption("Day-ahead (planning) the 4-class model finds only ~10 % of major delays, so severity is shown only in Live Mode "
                   "(operational: major delays found 80 % of the time at 93 % precision). Planning keeps the yes/no probability.")
        for m in ("planning", "operational"):
            p = EXT / f"severity_confusion_{m}.csv"
            if exists(p):
                cm = read_csv(str(p), index_col=0)
                st.plotly_chart(px.imshow(cm, text_auto=".2f", color_continuous_scale="Blues", aspect="auto",
                                          title=f"{m}: severity confusion (row-normalised, test)"), width="stretch")
    else:
        st.info("Run `python -m src.extensions.severity`.")
    if exists(EXT / "regression_metrics.csv"):
        st.markdown("**Minutes late** (regression, median absolute error) with an 80 % prediction interval")
        st.dataframe(read_csv(str(EXT / "regression_metrics.csv")).round(3), hide_index=True, width="stretch")
        if exists(EXT / "regression_conformal.csv"):
            st.markdown("**Calibrated 80 % interval** (conformalized quantile regression on Sep–Oct, checked on Nov–Dec; decision 31)")
            st.dataframe(read_csv(str(EXT / "regression_conformal.csv")).round(3), hide_index=True, width="stretch")
            st.caption("Q is the margin (minutes) added to both ends of the raw interval. Coverage stays slightly below 80 % "
                       "because the winter test months differ from the autumn calibration months.")
        p = EXT / "regression_sample_operational.parquet"
        if exists(p):
            s = read_parquet(str(p)).sample(3000, random_state=0) if len(read_parquet(str(p))) > 3000 else read_parquet(str(p))
            ycols = [c for c in s.columns if c.startswith("pred")]
            if "delay_min" in s and ycols:
                st.plotly_chart(px.scatter(s, x="delay_min", y=ycols[0], opacity=0.3, title="operational: predicted vs actual minutes",
                                           range_x=[-10, 30], range_y=[-10, 30]), width="stretch")
    else:
        st.info("Run `python -m src.extensions.regression`.")

with t4:
    if exists(XAI / "route_period_risk.csv"):
        r = read_csv(str(XAI / "route_period_risk.csv"), dtype={"route_id": str})
        piv = r.pivot_table(index="route_id", columns="period", values="actual", aggfunc="mean")
        st.plotly_chart(px.imshow(piv, text_auto=".0%", color_continuous_scale="Reds", aspect="auto",
                                  title="Actual late rate by route and period (test, both directions averaged)"),
                        width="stretch")
        st.dataframe(r.round(3), hide_index=True, width="stretch")
