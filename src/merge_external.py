"""Step 5g — Merge all external sources onto every stop event and audit the join coverage.

Input : data/processed/bus_YYYY_clean.parquet + data/external/*
Output: data/processed/bus_YYYY_enriched.parquet
        reports/join_coverage_YYYY.csv   (share of rows matched by each source)

Run:  python -m src.merge_external 2024 2025
"""
import sys

import numpy as np
import pandas as pd

from src.config import EXTERNAL_DIR, PROCESSED_DIR, REPORTS

WEATHER_COLS = ["temperature_2m", "precipitation", "rain", "snowfall", "snow_depth", "wind_speed_10m",
                "wind_gusts_10m", "precip_3h", "snow_24h", "heavy_rain", "is_snowing", "freezing", "is_snow_day"]


def add_trip_context(df):
    """Previous timepoint of the same trip -> scheduled running time of the segment, trip start hour."""
    df = df.sort_values(["service_date", "route_id", "half_trip_id", "time_point_order"]).reset_index(drop=True)
    # A few dozen rows per month have no half_trip_id; they cannot be linked to a trip, so they get
    # no previous segment and their own scheduled hour as the "trip start hour".
    has_id = df.half_trip_id.notna()
    same = (df.half_trip_id.eq(df.half_trip_id.shift()) & df.service_date.eq(df.service_date.shift())
            & df.route_id.eq(df.route_id.shift())).fillna(False).astype(bool) & has_id
    df["from_stop"] = df.stop_id.astype(str).shift().where(same)
    df["seg_sched_min"] = ((df.sched_sec - df.sched_sec.shift()) / 60).where(same).astype("float32")
    grp = df.groupby(["service_date", "route_id", "half_trip_id"], observed=True).sched_sec
    start = grp.transform("min").fillna(df.sched_sec)
    df["trip_start_hour"] = (start // 3600 % 24).astype("int8")
    return df


def main(year: int):
    df = pd.read_parquet(PROCESSED_DIR / f"bus_{year}_clean.parquet")
    n0 = len(df)
    cov = {}
    df = add_trip_context(df)
    df["route_id"] = df.route_id.astype(str)
    df["stop_id"] = df.stop_id.astype(str)
    df["direction_id"] = df.direction_id.astype(str)

    # 1) calendar --------------------------------------------------------------------------------
    cal = pd.read_parquet(EXTERNAL_DIR / "calendar_daily.parquet").drop(columns="holiday_name")
    df = df.merge(cal.rename(columns={"date": "service_date"}), on="service_date", how="left")
    cov["calendar"] = df.is_holiday.notna().mean()

    # 2) weather on local calendar date + hour of the *scheduled* time -----------------------------
    w = pd.read_parquet(EXTERNAL_DIR / "weather_hourly.parquet")[["date", "hour"] + WEATHER_COLS]
    df["w_date"] = df.scheduled_dt.dt.normalize()
    df["w_hour"] = df.scheduled_dt.dt.hour.astype("int8")
    df = df.merge(w.rename(columns={"date": "w_date", "hour": "w_hour"}), on=["w_date", "w_hour"], how="left")
    cov["weather"] = df.precipitation.notna().mean()
    df = df.drop(columns=["w_date", "w_hour"])

    # 3) events: hourly flags --------------------------------------------------------------------
    ev = pd.read_parquet(EXTERNAL_DIR / "events.parquet")
    er = pd.read_parquet(EXTERNAL_DIR / "event_routes.parquet")
    hrs = []
    for r in ev.itertuples():
        for h in pd.date_range(r.window_start.floor("h"), r.window_end.floor("h"), freq="h"):
            hrs.append((h, r.venue, r.scope))
    eh = pd.DataFrame(hrs, columns=["ev_hour", "venue", "scope"]).drop_duplicates()
    df["ev_hour"] = df.scheduled_dt.dt.floor("h")
    city = eh[eh.scope == "citywide"][["ev_hour"]].drop_duplicates().assign(citywide_event=1)
    df = df.merge(city, on="ev_hour", how="left")
    venue = eh[eh.scope == "venue"].merge(er, on="venue")[["ev_hour", "route_id"]].drop_duplicates()
    df = df.merge(venue.assign(venue_event_nearby=1), on=["ev_hour", "route_id"], how="left")
    df[["citywide_event", "venue_event_nearby"]] = df[["citywide_event", "venue_event_nearby"]].fillna(0).astype("int8")
    df = df.drop(columns="ev_hour")
    cov["events (rows flagged)"] = (df.citywide_event | df.venue_event_nearby).mean()

    # 4) GTFS segment geometry -------------------------------------------------------------------
    seg = pd.read_parquet(EXTERNAL_DIR / "gtfs_segments.parquet").rename(columns={"to_stop": "stop_id"})
    df = df.merge(seg, on=["route_id", "from_stop", "stop_id"], how="left")
    has_prev = df.from_stop.notna()
    cov["gtfs segment (non-first timepoints)"] = df.loc[has_prev, "segment_km"].notna().mean()
    df["schedule_tightness"] = (df.seg_sched_min / df.segment_km.where(df.segment_km > 0.05)).astype("float32")
    stops = pd.read_parquet(EXTERNAL_DIR / "gtfs_stops.parquet")
    df = df.merge(stops, on="stop_id", how="left")
    cov["gtfs stop coordinates"] = df.stop_lat.notna().mean()

    # 5) ridership (crowding) ---------------------------------------------------------------------
    rl = pd.read_parquet(EXTERNAL_DIR / "ridership_load.parquet")
    rl = rl[rl.year == year].drop(columns="year")
    df["day_type"] = np.select([df.is_holiday.eq(1) | df.dow.eq(6), df.dow.eq(5)], ["sunday", "saturday"], "weekday")
    k = ["route_id", "direction_id", "stop_id", "day_type"]
    df = df.merge(rl, on=k + ["trip_start_hour"], how="left")
    fb = rl.groupby(k)[["avg_load", "avg_boardings", "avg_alightings", "load_vs_route"]].mean().add_suffix("_fb").reset_index()
    df = df.merge(fb, on=k, how="left")
    exact = df.avg_load.notna()
    for c in ["avg_load", "avg_boardings", "avg_alightings", "load_vs_route"]:
        df[c] = df[c].fillna(df[c + "_fb"])
    df = df.drop(columns=[c for c in df.columns if c.endswith("_fb")] + ["route_avg_load"])
    cov["ridership exact hour"] = exact.mean()
    cov["ridership incl. fallback"] = df.avg_load.notna().mean()

    assert len(df) == n0, f"row count changed in merges: {n0} -> {len(df)}"
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    df.to_parquet(PROCESSED_DIR / f"bus_{year}_enriched.parquet", index=False)
    c = pd.Series(cov, name="coverage").mul(100).round(2)
    c.to_csv(REPORTS / f"join_coverage_{year}.csv")
    print(year, f"{len(df):,} rows, {df.shape[1]} cols\n", c.to_string())


if __name__ == "__main__":
    for y in (sys.argv[1:] or ["2024", "2025"]):
        main(int(y))
