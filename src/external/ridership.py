"""Step 5e — Passenger load (crowding) from MBTA automatic passenger counters.

Source  : MBTA OPMI "Bus Ridership by Trip, Season, Route/Line, and Stop" (APC sensor averages).
Why     : Busy buses spend longer at stops (boarding/alighting, fare payment), which shows up as
          delay. Load lets the model tell "slow because crowded" apart from "slow because of traffic",
          and it gives planners a lever (add capacity vs. fix the running time).
Leakage : For year Y we only use the most recent *Fall* season published BEFORE Y
          (2024 events <- Fall 2023, 2025 events <- Fall 2024). No future ridership is used.
Join    : route_id x direction x stop_id x day_type (weekday/saturday/sunday) x trip-start hour,
          falling back to route x direction x stop x day_type if the hour has no sample.
Output  : data/external/ridership_load.parquet
"""
import pandas as pd

from src.config import EXTERNAL_DIR, SCOPED_ROUTES

SRC = EXTERNAL_DIR / "ridership" / "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop"
SEASON_FOR_YEAR = {2024: "Fall_2023", 2025: "Fall_2024"}
# MBTA bus GTFS convention, verified against the arrival data in notebook 03
DIRECTION_MAP = {0: "Outbound", 1: "Inbound"}


def load_season(tag: str) -> pd.DataFrame:
    # NB: schema changed between seasons (Fall 2024 renamed load_ -> load, sample_size -> sample_count)
    df = pd.read_csv(SRC / f"{SRC.name}_{tag}.csv",
                     dtype={"route_id": str, "stop_id": str, "day_type_name": str, "trip_start_time": str})
    df = df.rename(columns={"load": "load_", "sample_count": "sample_size"})
    df = df[["route_id", "direction_id", "trip_start_time", "day_type_name", "stop_id",
             "boardings", "alightings", "load_", "sample_size"]]
    df = df[df.route_id.isin(SCOPED_ROUTES)].copy()
    df["direction_id"] = df["direction_id"].map(DIRECTION_MAP)
    df["trip_start_hour"] = df["trip_start_time"].str.split(":").str[0].astype(int) % 24
    df["day_type"] = df["day_type_name"].str.lower()
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    w = df["sample_size"].clip(lower=1)
    df = df.assign(wl=df.load_ * w, wb=df.boardings * w, wa=df.alightings * w, w=w)
    keys = ["route_id", "direction_id", "stop_id", "day_type", "trip_start_hour"]
    g = df.groupby(keys)[["wl", "wb", "wa", "w"]].sum()
    out = pd.DataFrame({"avg_load": g.wl / g.w, "avg_boardings": g.wb / g.w,
                        "avg_alightings": g.wa / g.w}).reset_index()
    # route-level reference so we can express "how crowded vs normal for this route"
    route_avg = (df.groupby(["route_id", "day_type"]).wl.sum() / df.groupby(["route_id", "day_type"]).w.sum()
                 ).rename("route_avg_load").reset_index()
    out = out.merge(route_avg, on=["route_id", "day_type"])
    out["load_vs_route"] = out.avg_load / out.route_avg_load.where(out.route_avg_load > 0)
    return out


def main():
    parts = []
    for year, tag in SEASON_FOR_YEAR.items():
        a = aggregate(load_season(tag))
        a["year"] = year
        parts.append(a)
        print(year, "<-", tag, f"{len(a):,} rows, routes={a.route_id.nunique()}")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(EXTERNAL_DIR / "ridership_load.parquet", index=False)
    print(out.describe().round(2))


if __name__ == "__main__":
    main()
