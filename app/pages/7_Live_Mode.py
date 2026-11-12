"""Page 7: LIVE mode: buses running right now on a route, scored by the operational model, with explanations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from common import APP, best_threshold, exists, explain_row, llm_warning_banner, page, risk_badge, route_trust, shap_bar

page("Live mode", "📡")
st.caption("Live MBTA predictions (V3 API) + current weather (Open-Meteo) → TransitPulse operational model. For each bus in service we score "
           "its **next timepoint**: the probability it will be more than 5 min late there.")

if not exists(APP / "profiles.parquet"):
    st.error("Static profiles missing. Run `python -m src.app_data` first.")
    st.stop()

from src.config import SCOPED_ROUTES  # noqa: E402
from src.live import mbta_live  # noqa: E402
from src.xai import narrator  # noqa: E402



@st.cache_data
def stop_name_map() -> dict:
    """stop_id -> GTFS stop name (reports/app/stop_names.parquet, written by src.app_data)."""
    f = APP / "stop_names.parquet"
    if not exists(f):
        return {}
    s = pd.read_parquet(f).dropna()
    return dict(zip(s.stop_id.astype(str), s.stop_name))


c1, c2, c3 = st.columns([1, 1, 1.2])
route = c1.selectbox("Route", [str(r) for r in SCOPED_ROUTES], index=0)
thr = c2.slider("Dispatcher alert threshold", 0.1, 0.9, 0.50, 0.05,
                help=f"Phase 3: ≥ 0.50 gives precision ≈ 0.93 / recall ≈ 0.82; best-F1 = {best_threshold('operational'):.2f}")
auto = c3.toggle("Auto-refresh every 60 s", value=False,
                 help="Without an MBTA API key the limit is 20 requests/min; each refresh uses 2 requests.")
refresh = c3.button("🔄 Refresh now", width="stretch")

key = f"live_{route}"
if refresh or key not in st.session_state or (auto and time.time() - st.session_state[key]["t"] > 60):
    with st.spinner("Fetching live MBTA predictions..."):
        out, status, weather = mbta_live.live_snapshot(route)
    st.session_state[key] = {"t": time.time(), "out": out, "status": status, "weather": weather}

snap = st.session_state[key]
out, status, weather = snap["out"], snap["status"], snap["weather"]
age = int(time.time() - snap["t"])
if status.ok:
    st.success(f"✅ {status.message} · updated {age} s ago")
else:
    st.warning(f"⚠️ {status.message}")
    st.caption("Live mode depends on buses being in service and on the MBTA API. Other pages work offline.")

if weather:
    st.caption(f"Weather now ({weather.get('_source')}): {weather.get('temperature_2m', float('nan')):.1f} °C, "
               f"rain {weather.get('precipitation', 0):.1f} mm/h, 3-h rain {weather.get('precip_3h', 0):.1f} mm, "
               f"snow 24 h {weather.get('snow_24h', 0):.1f} cm")

t = route_trust("operational", route)
if t:
    st.caption(f"Model trust on route {route}: test AUC {t.get('auc', float('nan')):.3f} ({t.get('trust', '')})")

if not out.empty:
    v = out.copy()
    v["risk"] = [risk_badge(p, thr) for p in v.p_late]
    v["alert"] = v.p_late >= thr
    v["scheduled"] = pd.to_datetime(v.scheduled).dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    names = stop_name_map()
    v["stop_name"] = [names.get(str(s), f"stop {s}") for s in v.stop_id]
    k = st.columns(4)
    k[0].metric("Buses scored", len(v)); k[1].metric("Alerts (≥ threshold)", int(v.alert.sum()))
    k[2].metric("MBTA itself predicts > 5 min late", int(v.mbta_says_late.sum()))
    k[3].metric("Median current delay", f"{v.prev_tp_delay.median():.1f} min")
    cols = ["risk", "direction_id", "stop_name", "stop_id", "scheduled", "prev_tp_delay", "mbta_delay_min", "prev_bus_gap_ratio", "vehicle_id",
            "trip_id", "p_late"] + [c for c in ("severity_pred", "p_major") if c in v]
    show = v[cols].sort_values("p_late", ascending=False)
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"prev_tp_delay": st.column_config.NumberColumn("current delay (min)", format="%.1f"),
                                "mbta_delay_min": st.column_config.NumberColumn("MBTA predicted delay at next timepoint", format="%.1f"),
                                "prev_bus_gap_ratio": st.column_config.NumberColumn("gap to bus ahead / planned", format="%.2f"),
                                "p_late": st.column_config.ProgressColumn("P(> 5 min late)", min_value=0, max_value=1, format="%.2f"),
                                "stop_name": "next timepoint (stop)", "stop_id": "MBTA stop id",
                                "scheduled": "scheduled at that stop",
                                "severity_pred": "expected severity",
                                "p_major": st.column_config.NumberColumn("P(> 10 min late)", format="%.2f")})
    st.caption("TransitPulse vs MBTA: MBTA's prediction is a point estimate of arrival time; TransitPulse gives a *probability* of "
               "exceeding 5 min, using history, schedule design and the bus ahead. Agreement is expected when the bus is already late.")

    prof = pd.read_parquet(APP / "profiles.parquet", columns=["route_id", "stop_id", "stop_lat", "stop_lon"])
    prof = prof[prof.route_id.astype(str) == route].astype({"stop_id": str}).drop_duplicates("stop_id")
    m = v.assign(stop_key=v.stop_id.astype(str)).drop(columns=["stop_lat", "stop_lon"], errors="ignore") \
        .merge(prof[["stop_id", "stop_lat", "stop_lon"]].rename(columns={"stop_id": "stop_key"}), on="stop_key", how="left") \
        .dropna(subset=["stop_lat"])
    if not m.empty:
        m["size"] = m.p_late.clip(0.05) * 20
        fig = px.scatter_map(m, lat="stop_lat", lon="stop_lon", color="p_late", size="size",
                             color_continuous_scale="RdYlGn_r", range_color=(0, 1), hover_name="stop_name", hover_data=["trip_id", "prev_tp_delay"],
                             zoom=11.5, height=420, map_style="carto-positron", title="Next timepoints of buses in service")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Explain one bus")
    pick = st.selectbox("Bus (trip)", show.trip_id.tolist(),
                        format_func=lambda x: f"{x} → {show.set_index('trip_id').loc[x, 'stop_name']} · "
                                              f"{show.set_index('trip_id').loc[x, 'risk']}")
    from src.encoding import feature_list  # noqa: E402
    row = out[out.trip_id == pick].head(1)
    sv, base, p = explain_row("operational", row)
    a, b = st.columns([1.2, 1])
    a.plotly_chart(shap_bar(sv), width="stretch")
    r0 = row.iloc[0].copy()
    r0["service_date"] = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    r0["hour"] = int(r0["hour"])
    facts = narrator.facts_from_shap(r0, sv, float(base), thr, "operational")
    aud = b.radio("Explain for", narrator.AUDIENCES, horizontal=True, key="live_aud")
    use_llm = b.toggle("Use OpenAI", value=True, key="live_llm")
    res = narrator.llm_explanation(facts, aud, use_llm=use_llm)
    llm_warning_banner(res)
    b.markdown(f"**{risk_badge(p, thr)}**")
    b.write(res["text"])
    b.caption(f"source: {res['source']} · fact check: {'passed' if res.get('check', {}).get('ok', True) else 'failed → template'}")

st.divider()
st.caption("Approximations in live mode (documented in docs/08): start delay ≈ current delay; school calendar beyond June 2026 unknown; "
           "events not scored live; only the next timepoint per bus is scored, because the model was trained on that one-step-ahead task.")

if auto:
    @st.fragment(run_every=10)
    def _tick():
        """Checks every 10 s; when the snapshot is older than 60 s the whole page reruns and fetches fresh data."""
        left = 60 - int(time.time() - st.session_state[key]["t"])
        if left <= 0:
            st.rerun()
        st.caption(f"Next auto-refresh in about {left} s.")
    _tick()
