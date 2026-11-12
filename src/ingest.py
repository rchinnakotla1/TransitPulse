"""Step 2 — Ingest raw monthly MBTA CSVs into typed monthly parquet files.

* Parses the odd MBTA time encoding ("1900-01-01T13:40:00Z" = time of day,
  "1900-01-02T00:11:17Z" = past midnight on the same service day)
* Builds real timestamps = service_date + offset
* Computes delay in seconds
* Writes a data-quality log (rows, nulls, duplicates) to reports/data_quality_raw_{year}.csv

Run:  python -m src.ingest 2024 2025
"""
import time

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, REPORTS, raw_dir

BASE = pd.Timestamp("1900-01-01")


def _offset_seconds(s: pd.Series) -> pd.Series:
    """'1900-01-01T13:40:00Z' -> seconds after service-day midnight (float, NaN if missing)."""
    ts = pd.to_datetime(s.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    return (ts - BASE).dt.total_seconds()


def process_month(path):
    t0 = time.time()
    df = pd.read_csv(
        path,
        dtype={"route_id": "string", "direction_id": "category", "half_trip_id": "string",
               "stop_id": "string", "time_point_id": "string", "time_point_order": "float32",
               "point_type": "category", "standard_type": "category",
               "scheduled": "string", "actual": "string",
               "scheduled_headway": "float32", "headway": "float32"},
        parse_dates=["service_date"],
    )
    n_raw = len(df)
    q = {"file": path.name, "rows_raw": n_raw}
    for c in df.columns:
        q[f"null_{c}"] = int(df[c].isna().sum())

    # MBTA pads some route ids ("01") — normalise so "01" == "1"
    df["route_id"] = df["route_id"].str.strip().str.lstrip("0").replace("", "0")

    df["sched_sec"] = _offset_seconds(df["scheduled"]).astype("float32")
    df["actual_sec"] = _offset_seconds(df["actual"]).astype("float32")
    df["scheduled_dt"] = df["service_date"] + pd.to_timedelta(df["sched_sec"], unit="s")
    df["actual_dt"] = df["service_date"] + pd.to_timedelta(df["actual_sec"], unit="s")
    df["delay_sec"] = (df["actual_sec"] - df["sched_sec"]).astype("float32")
    df = df.drop(columns=["scheduled", "actual"])

    key = ["service_date", "route_id", "direction_id", "half_trip_id", "stop_id",
           "time_point_order", "sched_sec"]
    q["exact_duplicates"] = int(df.duplicated().sum())
    q["key_duplicates"] = int(df.duplicated(subset=key).sum())

    for c in ["route_id", "stop_id", "time_point_id"]:
        df[c] = df[c].astype("category")

    out = INTERIM_DIR / f"{path.stem[-7:]}.parquet"   # e.g. 2025-01.parquet
    df.to_parquet(out, index=False)
    q["seconds"] = round(time.time() - t0, 1)
    print(q["file"], n_raw, f"{q['seconds']}s", flush=True)
    return q


def main(year: int = 2025):
    files = sorted(raw_dir(year).glob("*.csv"))
    assert files, f"No CSVs found in {raw_dir(year)}"
    log = [process_month(f) for f in files]
    pd.DataFrame(log).to_csv(REPORTS / f"data_quality_raw_{year}.csv", index=False)
    print("done ->", INTERIM_DIR)


if __name__ == "__main__":
    import sys
    for y in (sys.argv[1:] or ["2025"]):
        main(int(y))
