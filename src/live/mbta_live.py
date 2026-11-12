"""Phase 7: LIVE mode - score buses that are running right now with the operational model.

Data sources (all free)
* MBTA V3 API  https://api-v3.mbta.com  ``/predictions`` (with ``include=schedule``) -> for every active trip the predicted and
  scheduled time at each upcoming stop.  Works without a key (20 requests/min); with ``MBTA_API_KEY`` in .env the limit is 1000/min.
* Open-Meteo forecast API (no key) -> current precipitation, temperature, wind, snow.
* Calendar: ``src.external.calendar_days`` (holidays package + BPS calendar); daylight from ``features.solar_elevation``.
* Static features (history rates, GTFS segment, ridership, position on route) from ``reports/app/profiles.parquet``
  (medians over Nov-Dec 2025 per route x direction x stop x hour x weekend).

How a live feature row is built (faithful to training - see decision log)
* The operational model was trained to predict the **next timepoint** given the delay at the previous one.  Live, we take each
  trip's *current delay* = MBTA predicted - scheduled time at its next stop, and score only the **next timepoint stop** of that
  trip.  Timepoints further ahead are shown with MBTA's own predicted delay, not scored (the model never saw such gaps).
* ``prev_bus_delay`` / ``prev_bus_gap_ratio``: current delay of the bus directly ahead (same route/direction) and the gap to it,
  measured at the first stop both buses still have a prediction for, relative to the planned headway (see ``bus_ahead``).
  NaN if there is no bus ahead (as for the first trip of the day in training).
* ``start_delay`` is unknown live -> set to the current delay (approximation, documented).

Every failure (network, HTTP 429 rate limit, no buses at night) returns an empty frame + a status message; never raises.
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from src.config import BOSTON_LATLON, LOG_DIR, REPORTS, ROOT, SCOPED_ROUTES, TIMEZONE

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

API = "https://api-v3.mbta.com"
METEO = "https://api.open-meteo.com/v1/forecast"
APP_DIR = REPORTS / "app"

log = logging.getLogger("transitpulse.live")
if not log.handlers:
    h = logging.FileHandler(LOG_DIR / "live.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h); log.setLevel(logging.INFO)


@dataclass
class LiveStatus:
    ok: bool
    message: str


def _headers():
    k = os.getenv("MBTA_API_KEY", "").strip()
    return {"x-api-key": k} if k else {}


def _get(path: str, params: dict) -> tuple[dict | None, LiveStatus]:
    try:
        r = requests.get(f"{API}{path}", params=params, headers=_headers(), timeout=15)
        if r.status_code == 429:
            log.warning("MBTA rate limit %s", path)
            return None, LiveStatus(False, "MBTA API rate limit reached (no key: 20 requests/min). Wait a minute or add MBTA_API_KEY to .env.")
        if r.status_code in (401, 403):
            return None, LiveStatus(False, f"MBTA API rejected the key (HTTP {r.status_code}). Check MBTA_API_KEY in .env.")
        r.raise_for_status()
        return r.json(), LiveStatus(True, "ok")
    except requests.RequestException as e:
        log.warning("MBTA request failed %s: %s", path, e)
        return None, LiveStatus(False, f"Cannot reach the MBTA API ({type(e).__name__}).")


def route_directions(route: str) -> dict[int, str]:
    """MBTA direction_id (0/1) -> 'Outbound'/'Inbound' (the names used in the arrival-departure data)."""
    js, st = _get(f"/routes/{route}", {"fields[route]": "direction_names"})
    if not js:
        return {0: "Outbound", 1: "Inbound"}
    names = js["data"]["attributes"]["direction_names"]
    return {i: n for i, n in enumerate(names)}


def fetch_predictions(route: str) -> tuple[pd.DataFrame, LiveStatus]:
    js, st = _get("/predictions", {"filter[route]": route, "include": "schedule"})
    if js is None:
        return pd.DataFrame(), st
    sched = {s["id"]: s["attributes"] for s in js.get("included", []) if s["type"] == "schedule"}
    rows = []
    for p in js["data"]:
        a = p["attributes"]; rel = p["relationships"]
        pt = a.get("arrival_time") or a.get("departure_time")
        sid = (rel.get("schedule") or {}).get("data") or {}
        s = sched.get(sid.get("id"), {})
        stt = s.get("arrival_time") or s.get("departure_time")
        if not pt:
            continue
        rows.append({"trip_id": rel["trip"]["data"]["id"], "stop_id": rel["stop"]["data"]["id"],
                     "vehicle_id": ((rel.get("vehicle") or {}).get("data") or {}).get("id"),
                     "direction": a["direction_id"], "stop_sequence": a.get("stop_sequence"),
                     "predicted": pd.Timestamp(pt), "scheduled": pd.Timestamp(stt) if stt else pd.NaT})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, LiveStatus(False, f"No live predictions for route {route} right now (no buses in service?).")
    df["delay_min"] = (df.predicted - df.scheduled).dt.total_seconds() / 60
    return df, LiveStatus(True, f"{df.trip_id.nunique()} active trips, {len(df)} upcoming stop predictions")


def current_weather() -> dict:
    """Current-hour weather in the same units as training; falls back to dry/mild if the API fails."""
    try:
        r = requests.get(METEO, params=dict(latitude=BOSTON_LATLON[0], longitude=BOSTON_LATLON[1], timezone=TIMEZONE,
                                            hourly="precipitation,temperature_2m,wind_gusts_10m,snowfall,snow_depth",
                                            past_days=1, forecast_days=1), timeout=10)
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"]); h["time"] = pd.to_datetime(h.time)
        now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")
        h = h[h.time <= now]
        last = h.iloc[-1]
        return {"precipitation": float(last.precipitation), "precip_3h": float(h.precipitation.tail(3).sum()),
                "snow_24h": float(h.snowfall.tail(24).sum()), "snow_depth": float(last.snow_depth or 0),   # metres, as in training
                "temperature_2m": float(last.temperature_2m), "wind_gusts_10m": float(last.wind_gusts_10m),
                "heavy_rain": int(last.precipitation >= 4), "is_snow_day": int(h.snowfall.tail(24).sum() >= 2.5),
                "_source": "Open-Meteo forecast"}
    except Exception as e:  # noqa: BLE001
        log.warning("weather failed: %s", e)
        return {"precipitation": 0.0, "precip_3h": 0.0, "snow_24h": 0.0, "snow_depth": 0.0, "temperature_2m": 10.0,
                "wind_gusts_10m": 20.0, "heavy_rain": 0, "is_snow_day": 0, "_source": "unavailable (assumed dry)"}


def calendar_now(ts: pd.Timestamp) -> dict:
    from src.external.calendar_days import build
    d = build(str(ts.date() - pd.Timedelta(days=1)), str(ts.date() + pd.Timedelta(days=1))).iloc[1]
    return {k: int(d[k]) for k in ("is_holiday", "is_school_day", "is_university_term", "holiday_week", "days_to_holiday")}


def time_features(ts: pd.Timestamp, cal: dict) -> dict:
    from src.features import PERIODS, solar_elevation
    h = ts.hour; frac = h + ts.minute / 60
    wk = ts.dayofweek < 5 and not cal["is_holiday"]
    period = next(n for lo, hi, n in PERIODS if lo <= h < hi)
    sun = float(solar_elevation(pd.Series([ts.tz_localize(None) if ts.tzinfo else ts])).iloc[0])
    am, pm = int(wk and 7 <= h <= 9), int(wk and 16 <= h <= 18)
    return {"hour": h, "hour_sin": math.sin(2 * math.pi * frac / 24), "hour_cos": math.cos(2 * math.pi * frac / 24),
            "dow": ts.dayofweek, "is_weekend": int(ts.dayofweek >= 5), "am_peak": am, "pm_peak": pm, "period": period,
            "school_dismissal": int(cal["is_school_day"] and 14 <= h <= 15), "sun_elevation": sun,
            "is_dark": int(sun < -6), "dark_peak": int(sun < -6 and (am or pm))}


def bus_ahead(preds: pd.DataFrame) -> pd.DataFrame:
    """For each trip: the bus directly ahead on the same direction, its current delay and the headway gap between them.

    Gap = ETA(this bus) - ETA(other bus) at the first stop that *both* trips still have a prediction for. The bus ahead
    is the one with the smallest positive gap (shown in the app). ``leaders`` keeps every bus ahead, nearest first, so
    that :func:`passed_bus_features` can rebuild the training definition (decision 30).
    """
    p = preds.dropna(subset=["predicted"]).sort_values(["trip_id", "predicted"])
    seq = {t: g for t, g in p.groupby("trip_id")}
    first_delay = p.groupby("trip_id").delay_min.first()
    dir_of = p.groupby("trip_id").direction.first()
    out = []
    for t, g in seq.items():
        eta = dict(zip(g.stop_id, g.predicted))
        leaders = []
        for o, h in seq.items():
            if o == t or dir_of[o] != dir_of[t]:
                continue
            eo = dict(zip(h.stop_id, h.predicted))
            common = next((s for s in g.stop_id if s in eo), None)
            if common is None:
                continue
            gap = (eta[common] - eo[common]).total_seconds() / 60
            if gap > 0:
                leaders.append((o, gap))
        leaders.sort(key=lambda x: x[1])
        best = leaders[0] if leaders else None
        out.append({"trip_id": t, "ahead_trip": best[0] if best else None, "gap_min": best[1] if best else np.nan,
                    "prev_bus_delay": first_delay.get(best[0], np.nan) if best else np.nan,
                    "leaders": [(o, gap, first_delay.get(o, np.nan)) for o, gap in leaders]})
    return pd.DataFrame(out).set_index("trip_id")


def passed_bus_features(preds: pd.DataFrame, rows: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Training-faithful ``prev_bus_gap_ratio`` / ``prev_bus_delay`` (decision 30).

    In training (src/features.py) the "previous bus" is the most recent *other* bus that has **already passed the
    target stop** at the prediction moment. A leader that is still between our previous timepoint and the target stop
    is skipped - so when two buses are bunched the feature describes the bus *two* ahead and the ratio is large.
    Live we reproduce that: walk the buses ahead (nearest first) and skip every one that still has a prediction for
    our target stop. For the first bus that has passed it we estimate its passing time as ETA(us at stop) - gap
    (headways change slowly between neighbouring stops) and compute ``(now - passing time) / scheduled headway``.
    """
    has_stop = set(zip(preds.trip_id, preds.stop_id))
    ratio, pdelay = [], []
    for r in rows.itertuples():
        val = (np.nan, np.nan)
        for o, gap, odelay in (r.leaders if isinstance(r.leaders, list) else []):
            if (o, r.stop_id) in has_stop:          # this leader has not reached our target stop yet -> skipped
                continue
            passed_at = r.mbta_predicted - pd.Timedelta(minutes=gap)
            since = max((now - passed_at).total_seconds() / 60, 0.0)
            val = (since / r.scheduled_headway if r.scheduled_headway and r.scheduled_headway > 0 else np.nan, odelay)
            break
        ratio.append(val[0]); pdelay.append(val[1])
    out = rows.copy()
    out["prev_bus_gap_ratio"] = pd.Series(ratio, index=rows.index, dtype="float64").clip(upper=10)
    out["prev_bus_delay"] = pd.Series(pdelay, index=rows.index, dtype="float64")
    return out


def build_rows(route: str, preds: pd.DataFrame, weather: dict | None = None, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """One feature row per active trip: its next timepoint stop (stops that exist in the training profiles)."""
    from src.encoding import apply_levels
    if preds.empty:
        return pd.DataFrame()
    prof = pd.read_parquet(APP_DIR / "profiles.parquet")
    prof = prof[prof.route_id == route]
    dirs = route_directions(route)
    now = now or pd.Timestamp.now(tz=TIMEZONE)
    weather = weather or current_weather()
    cal = calendar_now(now)
    p = preds.copy()
    p["direction_id"] = p.direction.map(dirs)
    p = p.sort_values(["trip_id", "predicted"])
    # score only trips that are IN SERVICE and SCHEDULED (decision D26):
    # * not started yet (next stop is stop_sequence 1): there is no current delay to propagate - training uses the
    #   first-timepoint convention (no previous delay) and MBTA's 0-min "delay" at the origin is just the schedule copied;
    # * added / overload trips ("-OL" ids) have no schedule, so their delay is undefined.
    trip = p.groupby("trip_id").agg(first_seq=("stop_sequence", "min"), no_sched=("scheduled", lambda x: x.isna().all()))
    keep = trip.index[(trip.first_seq.fillna(1) > 1) & ~trip.no_sched]
    skipped = {"not_started": int((trip.first_seq.fillna(1) <= 1).sum()), "unscheduled": int(trip.no_sched.sum())}
    p = p[p.trip_id.isin(keep) & p.scheduled.notna()]
    cur = p.groupby("trip_id").first()[["delay_min", "predicted"]].rename(columns={"delay_min": "current_delay",
                                                                                   "predicted": "next_stop_eta"})
    tps = set(zip(prof.direction_id, prof.stop_id))
    p = p[[(d, s) in tps for d, s in zip(p.direction_id, p.stop_id)]]
    nxt = p.groupby("trip_id").first().reset_index().join(cur, on="trip_id")
    if nxt.empty:
        e = pd.DataFrame(); e.attrs["skipped"] = skipped
        return e
    rows = []
    for _, r in nxt.iterrows():
        sched = r.scheduled if pd.notna(r.scheduled) else r.predicted
        local = sched.tz_convert(TIMEZONE)
        tf = time_features(local, cal)
        key = prof[(prof.direction_id == r.direction_id) & (prof.stop_id == r.stop_id) & (prof.is_weekend == tf["is_weekend"])]
        if key.empty:
            continue
        k = key.iloc[(key.hour - tf["hour"]).abs().argsort()[:1]].iloc[0]   # nearest hour with a profile
        row = {**k.drop(["hour", "is_weekend"]).to_dict(), **cal, **tf,
               **{c: v for c, v in weather.items() if not c.startswith("_")},
               "route_id": route, "direction_id": r.direction_id, "stop_id": r.stop_id,
               "venue_event_nearby": 0, "citywide_event": 0,
               "prev_tp_delay": r.current_delay, "prev_tp_delayed": float(r.current_delay > 5),
               "start_delay": r.current_delay,
               # training: scheduled time here - actual departure at previous TP (negative = already behind schedule,
               # 27% of test rows). Live the departure time is unknown but <= now, so sched - now is a lower bound; NOT clipped.
               "min_since_prev_tp": (sched - now).total_seconds() / 60,
               "trip_id": r.trip_id, "vehicle_id": r.vehicle_id, "scheduled": sched, "mbta_predicted": r.predicted,
               "mbta_delay_min": r.delay_min}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["skipped"] = skipped
    if df.empty:
        return df
    # bus ahead on the same direction: gap measured at the first stop both trips still have a prediction for
    ahead = bus_ahead(preds)
    df = df.join(ahead[["ahead_trip", "gap_min", "leaders"]], on="trip_id")
    # gap_min (direct leader) is kept for display; the MODEL features follow the training definition (decision 30)
    df = passed_bus_features(preds, df, now).drop(columns=["leaders", "ahead_trip"])
    df = df.sort_values(["direction_id", "scheduled"])
    out = apply_levels(df.reset_index(drop=True))
    out.attrs["skipped"] = skipped
    return out


def score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    import xgboost as xgb
    from src.encoding import feature_list
    from src.xai.shap_analysis import MODEL_DIR
    m = xgb.XGBClassifier(); m.load_model(MODEL_DIR / "XGB_operational.json")
    feats = feature_list("operational")
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    df = df.copy()
    df["p_late"] = m.predict_proba(X)[:, 1]
    df["mbta_says_late"] = df.mbta_delay_min > 5
    # severity class (Phase 5, operational only - decision 29): on_time / minor / moderate / major
    sev = MODEL_DIR / "SEV_operational.json"
    if sev.exists():
        from src.config import SEVERITY_LABELS
        s = xgb.XGBClassifier(); s.load_model(sev)
        ps = s.predict_proba(X)
        df["severity_pred"] = [SEVERITY_LABELS[i] for i in ps.argmax(1)]
        df["p_major"] = ps[:, 3]
    return df


def live_snapshot(route: str) -> tuple[pd.DataFrame, LiveStatus, dict]:
    """Everything the Live page needs: scored rows, status, weather used."""
    t = time.time()
    preds, st = fetch_predictions(route)
    if not st.ok:
        return pd.DataFrame(), st, {}
    w = current_weather()
    rows = build_rows(route, preds, w)
    if rows.empty:
        return pd.DataFrame(), LiveStatus(False, f"{st.message}, but none is approaching a timepoint the model knows."), w
    sk = rows.attrs.get("skipped", {})
    out = score(rows)
    log.info("live route=%s trips=%d scored=%d skipped=%s %.1fs", route, preds.trip_id.nunique(), len(out), sk, time.time() - t)
    extra = f" (not yet started: {sk.get('not_started', 0)}, unscheduled added trips: {sk.get('unscheduled', 0)})" if sk else ""
    return out, LiveStatus(True, f"{st.message}; scored {len(out)} buses in service at their next timepoint{extra}"), w


SUPPORTED_ROUTES = SCOPED_ROUTES

if __name__ == "__main__":
    out, st, w = live_snapshot("28")
    print(st)
    if not out.empty:
        print(out[["trip_id", "direction_id", "stop_id", "scheduled", "prev_tp_delay", "p_late", "mbta_delay_min"]])
