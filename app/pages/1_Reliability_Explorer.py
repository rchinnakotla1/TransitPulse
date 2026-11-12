from common import APP, page, read_parquet
import pandas as pd
import plotly.express as px
import streamlit as st

page("Reliability Explorer", "📊")
st.caption("Observed 2025 data (all splits). Late = more than 5 minutes after schedule at a timepoint.")

rel = read_parquet(str(APP / "reliability.parquet"))
routes = sorted(rel.route_id.unique(), key=lambda x: (len(x), x))
c1, c2, c3 = st.columns(3)
route = c1.selectbox("Route", ["All"] + routes)
direction = c2.selectbox("Direction", ["Both", "Inbound", "Outbound"])
day = c3.selectbox("Day type", ["Both", "weekday", "weekend"])
d = rel.copy()
if route != "All": d = d[d.route_id == route]
if direction != "Both": d = d[d.direction_id == direction]
if day != "Both": d = d[d.day_type == day]

def wmean(g):
    return pd.Series({"late_rate": (g.late_rate * g.events).sum() / g.events.sum(), "events": g.events.sum()})

k = st.columns(3)
tot = wmean(d)
k[0].metric("Late rate", f"{tot.late_rate:.1%}"); k[1].metric("Stop events", f"{int(tot.events):,}")
k[2].metric("Mean delay (min)", f"{(d.mean_delay * d.events).sum() / d.events.sum():.1f}")

hm = d.groupby(["month", "hour"]).apply(wmean, include_groups=False).reset_index()
fig = px.density_heatmap(hm, x="hour", y="month", z="late_rate", histfunc="avg", nbinsx=24, nbinsy=12,
                         color_continuous_scale="Reds", title="Late rate by hour and month")
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
by_route = rel.groupby("route_id").apply(wmean, include_groups=False).reset_index().sort_values("late_rate")
c1.plotly_chart(px.bar(by_route, x="late_rate", y="route_id", orientation="h", title="Late rate by route (2025)",
                       height=550).update_xaxes(tickformat=".0%"), width="stretch")
by_hour = d.groupby(["hour", "day_type"]).apply(wmean, include_groups=False).reset_index()
c2.plotly_chart(px.line(by_hour, x="hour", y="late_rate", color="day_type", markers=True, title="Late rate by hour",
                        height=550).update_yaxes(tickformat=".0%"), width="stretch")

st.subheader("Timepoint map")
stops = read_parquet(str(APP / "stops.parquet"))
s = stops if route == "All" else stops[stops.route_id == route]
if direction != "Both": s = s[s.direction_id == direction]
s = s.dropna(subset=["lat", "lon"])
fig = px.scatter_map(s, lat="lat", lon="lon", color="late_rate", size="events", hover_name="stop_id",
                     hover_data={"route_id": True, "direction_id": True, "late_rate": ":.1%", "mean_delay": ":.1f"},
                     color_continuous_scale="RdYlGn_r", zoom=11, height=550, map_style="carto-positron")
st.plotly_chart(fig, width="stretch")

st.subheader("Daily trend")
daily = read_parquet(str(APP / "daily.parquet"))
dd = daily if route == "All" else daily[daily.route_id == route]
dd = dd.groupby("service_date").apply(lambda g: (g.late_rate * g.events).sum() / g.events.sum(), include_groups=False).rename("late_rate").reset_index()
dd["7-day mean"] = dd.late_rate.rolling(7, min_periods=1).mean()
st.plotly_chart(px.line(dd, x="service_date", y=["late_rate", "7-day mean"], title="Daily late rate").update_yaxes(tickformat=".0%"),
                width="stretch")
