"""Phase 6: pre-compute small tables for the Streamlit dashboard (so the app starts in seconds and never loads 10 M rows).

Outputs in ``reports/app/``
* ``reliability.parquet``      late rate / mean delay by route x direction x hour x weekday-type x month (2025)
* ``stops.parquet``            timepoint stops with location and late rate per route/direction (map)
* ``daily.parquet``            daily late rate per route (trend chart)
* ``events_planning.parquet`` / ``events_operational.parquet``
                               every test stop event of two December weeks with all model features (Predict & Explain page)
* ``profiles.parquet``         typical static features per route x direction x stop x hour x weekend (used by Live mode to build
                               feature rows for buses that are running right now)
* ``stop_names.parquet``       stop_id -> readable GTFS stop name for every timepoint in the app (e.g. "Dudley St @ Warren St")

Run: ``python -m src.app_data``
"""
from __future__ import annotations

import pandas as pd

from src.config import REPORTS
from src.encoding import feature_list, load_split, spec

APP_DIR = REPORTS / "app"; APP_DIR.mkdir(exist_ok=True)
EVENT_DAYS = ("2025-12-01", "2025-12-14")
STATIC = ["time_point_order", "stop_pos", "point_type", "standard_type", "scheduled_headway", "min_into_trip", "stop_lat",
          "stop_lon", "hist_route_hour", "hist_route_dow", "hist_stop_period", "hist_stop_hour", "hist_delay_route_hour",
          "hist_delay_stop_hour", "recent_route_28d", "recent_stop_28d", "hist_stop_hour_n", "segment_km", "stops_between",
          "seg_sched_min", "schedule_tightness", "avg_load", "avg_boardings", "avg_alightings", "load_vs_route",
          "trip_start_hour", "period"]


def stop_names():
    """Readable names for the timepoints used in the app, from the MBTA GTFS stops table (Phase 1 download)."""
    from src.config import ROOT
    st = pd.concat([pd.read_parquet(APP_DIR / f, columns=["stop_id"]) for f in ("stops.parquet", "profiles.parquet")])
    st = st.astype({"stop_id": str}).drop_duplicates()
    g = pd.read_parquet(ROOT / "data" / "external" / "gtfs_stops.parquet", columns=["stop_id", "stop_name"])
    g["stop_id"] = g.stop_id.astype(str)
    out = st.astype({"stop_id": str}).merge(g.drop_duplicates("stop_id"), on="stop_id", how="left")
    out.to_parquet(APP_DIR / "stop_names.parquet", index=False)
    print("stop names", len(out), "missing", int(out.stop_name.isna().sum()))


def main():
    cols = ["service_date", "route_id", "direction_id", "stop_id", "hour", "is_weekend", "delayed", "delay_min",
            "stop_lat", "stop_lon", "time_point_order"]
    df = pd.concat([load_split(s, cols) for s in ("train", "valid", "test")], ignore_index=True)
    for c in ("route_id", "direction_id", "stop_id"):
        df[c] = df[c].astype(str)
    df["month"] = df.service_date.dt.month
    df["day_type"] = df.is_weekend.map({0: "weekday", 1: "weekend"})
    rel = df.groupby(["route_id", "direction_id", "hour", "day_type", "month"]).agg(
        events=("delayed", "size"), late_rate=("delayed", "mean"), mean_delay=("delay_min", "mean")).reset_index()
    rel.to_parquet(APP_DIR / "reliability.parquet", index=False)
    stops = df.groupby(["route_id", "direction_id", "stop_id"]).agg(
        lat=("stop_lat", "median"), lon=("stop_lon", "median"), order=("time_point_order", "median"),
        events=("delayed", "size"), late_rate=("delayed", "mean"), mean_delay=("delay_min", "mean")).reset_index()
    stops.to_parquet(APP_DIR / "stops.parquet", index=False)
    daily = df.groupby(["service_date", "route_id"]).agg(late_rate=("delayed", "mean"), events=("delayed", "size")).reset_index()
    daily.to_parquet(APP_DIR / "daily.parquet", index=False)
    print("reliability", len(rel), "stops", len(stops), "daily", len(daily))
    del df

    keep = ["service_date", "route_id", "direction_id", "stop_id", "hour", "half_trip_id", "scheduled_dt", "delayed", "delay_min"]
    for mode in ("planning", "operational"):
        feats = feature_list(mode)
        te = load_split("test", list(dict.fromkeys(keep + feats)))
        te = te[(te.service_date >= EVENT_DAYS[0]) & (te.service_date <= EVENT_DAYS[1])].reset_index(drop=True)
        for c in spec()["categorical"]:
            te[c] = te[c].astype(str)
        te.to_parquet(APP_DIR / f"events_{mode}.parquet", index=False)
        print(mode, "events", len(te))

    # live-mode profiles from the most recent two months (test period = closest to "now")
    te = load_split("test", ["route_id", "direction_id", "stop_id", "hour", "is_weekend"] + [c for c in STATIC if c != "period"] + ["period"])
    for c in ("route_id", "direction_id", "stop_id", "point_type", "standard_type", "period"):
        te[c] = te[c].astype(str)
    num = [c for c in STATIC if c not in ("point_type", "standard_type", "period")]
    agg = {c: "median" for c in num} | {c: "first" for c in ("point_type", "standard_type", "period")}   # constant per stop / hour
    prof = te.groupby(["route_id", "direction_id", "stop_id", "hour", "is_weekend"]).agg(agg).reset_index()
    prof.to_parquet(APP_DIR / "profiles.parquet", index=False)
    print("profiles", len(prof))

    # route-level trust indicator: how well does the model rank on this route (test, XGB)?
    from sklearn.metrics import roc_auc_score
    rows = []
    for mode in ("planning", "operational"):
        p = pd.read_parquet(REPORTS / "preds" / f"test_{mode}.parquet", columns=["route_id", "delayed", "p_XGB"])
        for r, g in p.groupby(p.route_id.astype(str)):
            rows.append({"mode": mode, "route_id": r, "auc": roc_auc_score(g.delayed, g.p_XGB), "late_rate": g.delayed.mean(),
                         "events": len(g)})
    ra = pd.DataFrame(rows)
    ra["trust"] = pd.cut(ra.auc, [0, 0.70, 0.75, 0.90, 1], labels=["low", "medium", "good", "very good"])
    ra.to_csv(APP_DIR / "route_auc.csv", index=False)
    print(ra.pivot(index="route_id", columns="mode", values="auc").round(3))
    stop_names()


if __name__ == "__main__":
    main()
