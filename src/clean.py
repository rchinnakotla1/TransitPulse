"""Step 3 — Cleaning & scoping (per year).

Input : data/interim/YYYY-MM.parquet (all routes, typed)
Output: data/processed/bus_YYYY_clean.parquet       (scoped, cleaned, labelled)
        reports/cleaning_funnel_YYYY.csv             (rows removed at each rule)
        reports/route_summary_all_YYYY.csv           (system-wide per-route stats, all routes)
        reports/missing_by_route_YYYY.csv            (missing actual = proxy for cancelled / not observed)

Route scope: chosen once on 2025 (14 key routes + 5 busiest schedule-managed routes, see
`pick_routes`) and frozen in config.SCOPED_ROUTES so every year covers the same routes.

Run:  python -m src.clean 2024 2025
"""
import sys

import pandas as pd

from src.config import (DELAY_THRESHOLD_SEC, INTERIM_DIR, KEY_ROUTES, MAX_EARLY_SEC,
                        MAX_LATE_SEC, N_SCHEDULE_ROUTES, PROCESSED_DIR, REPORTS,
                        SCOPED_ROUTES, SEVERITY_BINS, SEVERITY_LABELS)


def files_for(year):
    return sorted(INTERIM_DIR.glob(f"{year}-*.parquet"))


def route_overview(year):
    """Light pass over all routes (3 columns only) to (a) describe the full system and (b) pick scope."""
    parts = []
    for f in files_for(year):
        d = pd.read_parquet(f, columns=["route_id", "standard_type", "delay_sec"])
        d["route_id"] = d["route_id"].astype(str)
        d["late"] = d["delay_sec"] > DELAY_THRESHOLD_SEC
        d["missing_actual"] = d["delay_sec"].isna()
        parts.append(d.groupby(["route_id", "standard_type"], observed=True).agg(
            rows=("late", "size"), late=("late", "sum"),
            missing=("missing_actual", "sum"), delay_sum=("delay_sec", "sum")))
    s = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    s["observed"] = s["rows"] - s["missing"]
    s["late_rate"] = s["late"] / s["observed"]
    s["mean_delay_min"] = s["delay_sum"] / s["observed"] / 60
    s["missing_rate"] = s["missing"] / s["rows"]
    s = s.drop(columns="delay_sum").sort_values("rows", ascending=False)
    s.to_csv(REPORTS / f"route_summary_all_{year}.csv", index=False)
    return s


def pick_routes(s):
    """How SCOPED_ROUTES was derived (documented for reproducibility)."""
    sched = (s[(s.standard_type == "Schedule") & (~s.route_id.isin(KEY_ROUTES))]
             .groupby("route_id")["rows"].sum().sort_values(ascending=False))
    extra = list(sched.index[:N_SCHEDULE_ROUTES])
    return [r for r in KEY_ROUTES if r in set(s.route_id)] + extra


def main(year: int):
    s = route_overview(year)
    if year == 2025:
        derived = pick_routes(s)
        assert derived == SCOPED_ROUTES, f"Route scope changed: {derived}"
    routes = SCOPED_ROUTES
    present = set(s.route_id)
    print(year, "scoped routes present:", [r for r in routes if r in present],
          "| missing:", [r for r in routes if r not in present])

    df = pd.concat([pd.read_parquet(f, filters=[("route_id", "in", routes)]) for f in files_for(year)],
                   ignore_index=True)
    for c in ["route_id", "stop_id", "time_point_id"]:
        df[c] = df[c].astype(str).astype("category")

    funnel = [("raw rows (scoped routes)", len(df))]

    def step(name, mask_keep):
        nonlocal df
        df = df[mask_keep].copy()
        funnel.append((name, len(df)))

    step("drop exact duplicates", ~df.duplicated())
    step("drop missing scheduled time", df["sched_sec"].notna())

    # Missing actual time = trip/stop not observed (cancelled trip, AVL dropout).
    miss = (df.assign(missing=df["actual_sec"].isna())
              .groupby(["route_id", "direction_id"], observed=True)["missing"]
              .agg(["size", "sum", "mean"]).rename(columns={"size": "rows", "sum": "missing",
                                                             "mean": "missing_rate"}))
    miss.to_csv(REPORTS / f"missing_by_route_{year}.csv")
    step("drop missing actual (cancelled / unobserved)", df["actual_sec"].notna())

    key = ["service_date", "route_id", "direction_id", "half_trip_id", "stop_id",
           "time_point_order", "sched_sec"]
    df = df.sort_values(key + ["actual_sec"])
    step("drop duplicate stop events (same trip/stop/sched)",
         ~df.duplicated(subset=key, keep="first"))
    step(f"drop outliers (delay < {MAX_EARLY_SEC//60} min or > {MAX_LATE_SEC//60} min)",
         df["delay_sec"].between(MAX_EARLY_SEC, MAX_LATE_SEC))

    # ---- basic derived fields needed for EDA (full feature engineering is Step 6)
    df["delay_min"] = df["delay_sec"] / 60
    df["delayed"] = (df["delay_sec"] > DELAY_THRESHOLD_SEC).astype("int8")
    df["severity"] = pd.cut(df["delay_sec"], SEVERITY_BINS, labels=SEVERITY_LABELS, right=True)
    df["hour"] = (df["sched_sec"] // 3600 % 24).astype("int8")
    df["dow"] = df["service_date"].dt.day_of_week.astype("int8")          # 0 = Monday
    df["month"] = df["service_date"].dt.month.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")

    df = df.reset_index(drop=True)
    df.to_parquet(PROCESSED_DIR / f"bus_{year}_clean.parquet", index=False)

    f = pd.DataFrame(funnel, columns=["step", "rows_remaining"])
    f["rows_removed"] = (-f["rows_remaining"].diff()).fillna(0).astype(int)
    f["pct_of_raw"] = (100 * f["rows_remaining"] / f["rows_remaining"].iloc[0]).round(2)
    f.to_csv(REPORTS / f"cleaning_funnel_{year}.csv", index=False)
    print(f.to_string(index=False))
    print("Delayed share:", round(df["delayed"].mean(), 4))


if __name__ == "__main__":
    for y in (sys.argv[1:] or ["2025"]):
        main(int(y))
