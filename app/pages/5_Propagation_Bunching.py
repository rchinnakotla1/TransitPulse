"""Page 5: how delay spreads along a trip and between buses; bus bunching."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob

import plotly.express as px
import streamlit as st

from common import EXT, exists, page, read_csv

page("Delay propagation & bunching", "🌊")
st.caption("Descriptive analysis of all 9.8 M timepoint events in 2025 (Phase 5). Answers: once a bus is late, does it recover? "
           "Does lateness spread to the bus behind? How often do buses bunch?")

if not exists(EXT / "prop_transition.csv"):
    st.error("Phase 5 results not found. Run `python -m src.extensions.propagation`.")
    st.stop()

t1, t2, t3 = st.tabs(["Along the trip", "Recovery", "Bunching"])

with t1:
    o = read_csv(str(EXT / "prop_overall.csv"), index_col=0).iloc[:, 0]
    c = st.columns(4)
    c[0].metric("Corr. with previous timepoint", f"{o.get('corr_prev_tp', float('nan')):.2f}")
    c[1].metric("Late buses that recover by the next timepoint", f"{o.get('recovery_rate_overall', float('nan')):.0%}")
    c[2].metric("Late buses that stay late", f"{o.get('p_stay_late', float('nan')):.0%}")
    c[3].metric("On-time buses that become late", f"{o.get('p_become_late', float('nan')):.0%}")
    tr = read_csv(str(EXT / "prop_transition.csv"), index_col=0)
    st.plotly_chart(px.imshow(tr, text_auto=".0%", color_continuous_scale="Reds", aspect="auto",
                              title="Delay state at this timepoint given the previous one (row = previous)"),
                    width="stretch")
    c1, c2 = st.columns(2)
    if exists(EXT / "prop_late_start.csv"):
        ls = read_csv(str(EXT / "prop_late_start.csv"))
        c1.plotly_chart(px.bar(ls, x="start_band", y="p_late_at_end", text_auto=".0%",
                               title="Late start → late at the end of the trip"), width="stretch")
    if exists(EXT / "prop_growth.csv"):
        g = read_csv(str(EXT / "prop_growth.csv"))
        c2.plotly_chart(px.line(g, x="pos_decile", y=["mean_delay", "late_rate"], markers=True,
                                title="Delay grows along the route (position decile 0 = start)"), width="stretch")

with t2:
    c1, c2 = st.columns(2)
    if exists(EXT / "prop_recovery_period.csv"):
        rp = read_csv(str(EXT / "prop_recovery_period.csv"))
        c1.plotly_chart(px.bar(rp, x="period", y="recovery_rate", text_auto=".0%", title="Recovery rate by period"),
                        width="stretch")
    if exists(EXT / "prop_recovery_route.csv"):
        rr = read_csv(str(EXT / "prop_recovery_route.csv"), dtype={"route_id": str}).sort_values("recovery_rate")
        c2.plotly_chart(px.bar(rr, x="route_id", y="recovery_rate", text_auto=".0%", title="Recovery rate by route"),
                        width="stretch")
    if exists(EXT / "prop_bus_to_bus.csv"):
        st.markdown("**Does delay spread to the next bus?** Correlation of delay with the bus ahead (same stop) vs the same bus "
                    "at its previous timepoint.")
        st.dataframe(read_csv(str(EXT / "prop_bus_to_bus.csv")).round(3), hide_index=True, width="stretch")
    if exists(EXT / "prop_segment_added.csv"):
        st.markdown("**Where is delay created?** Minutes added per segment, by route.")
        st.dataframe(read_csv(str(EXT / "prop_segment_added.csv"), dtype={"route_id": str}).round(3), hide_index=True,
                     width="stretch")

with t3:
    st.markdown("**Bunched** = headway < 25 % of scheduled; **big gap** = headway > 150 % of scheduled (headway-managed and "
                "scheduled timepoints with a known scheduled headway).")
    files = sorted(glob.glob(str(EXT / "bunching_by_*.csv")))
    if not files:
        st.info("Run `python -m src.extensions.bunching` to add bunching results.")
    for f in files:
        k = Path(f).stem.replace("bunching_by_", "")
        b = read_csv(f).rename(columns=lambda c: str(c))
        x = b.columns[0]
        ys = [c for c in b.columns if "bunch" in c.lower() or "gap" in c.lower()]
        if ys:
            st.plotly_chart(px.bar(b, x=x, y=ys, barmode="group", title=f"Bunching by {k}"), width="stretch")
        else:
            st.dataframe(b, width="stretch")
    if exists(EXT / "bunching_metrics.csv"):
        st.markdown("**Can bunching be predicted?** Model vs simple gap-ratio baseline (test Nov–Dec).")
        st.dataframe(read_csv(str(EXT / "bunching_metrics.csv")).round(3), hide_index=True, width="stretch")
