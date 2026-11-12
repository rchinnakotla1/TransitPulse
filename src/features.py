"""Phase 2: Feature engineering.

Turns ``bus_{year}_enriched.parquet`` into model-ready, leakage-safe feature tables.

Two prediction modes (see docs/04_feature_engineering.md):
* **planning**: day-ahead. Only information known the day before (calendar, route/stop, schedule,
  historical rates up to yesterday, weather as a forecast, events, ridership of the previous season).
* **operational**: minutes-ahead (dispatcher). Planning + what the AVL system already knows at the moment the bus
  leaves its previous timepoint: its delay there, and the most recent bus that already passed this stop.

Leakage rules
* Historical rates use only service days **strictly before** the row's service date (expanding window, plus a
  28-day rolling window), seeded with 2024, smoothed toward the global rate known so far (Bayesian / m-estimate).
* ``month`` is NOT a model feature: train = Jan–Aug, test = Nov–Dec, so the model would extrapolate a month
  it never saw. Seasonality is carried by the school / university calendar and the 28-day rolling rates.
* The ``headway`` column (actual headway) and anything derived from this bus's own arrival is excluded.

Outputs
* data/processed/features/{train,valid,test,drift_2024}.parquet
* reports/feature_spec.json: feature groups, mode lists, categorical columns

Run: ``python -m src.features``
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import BOSTON_LATLON, DELAY_THRESHOLD_SEC, PROCESSED_DIR, REPORTS, TIMEZONE, TRAIN_END, VALID_END

FEAT_DIR = PROCESSED_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

M_SMOOTH = 30          # pseudo-count for Bayesian smoothing of historical rates
ROLL_DAYS = 28         # recent-reliability window
DEFAULT_RATE, DEFAULT_DELAY = 0.30, 2.0   # prior on the very first day of 2024 only

# ---------------------------------------------------------------------------------------------
# Feature groups (used by Phase 3 ablation: baseline -> +calendar -> +weather -> ...)
# ---------------------------------------------------------------------------------------------
GROUPS = {
    "temporal": ["hour_sin", "hour_cos", "dow", "is_weekend", "am_peak", "pm_peak", "period"],
    "calendar": ["is_holiday", "is_school_day", "is_university_term", "holiday_week", "days_to_holiday",
                 "school_dismissal"],
    "daylight": ["sun_elevation", "is_dark", "dark_peak"],
    "spatial": ["route_id", "direction_id", "stop_id", "time_point_order", "stop_pos", "point_type",
                "standard_type", "scheduled_headway", "trip_start_hour", "min_into_trip", "stop_lat", "stop_lon"],
    "history": ["hist_route_hour", "hist_route_dow", "hist_stop_period", "hist_stop_hour",
                "hist_delay_route_hour", "hist_delay_stop_hour", "recent_route_28d", "recent_stop_28d",
                "hist_stop_hour_n"],
    "weather": ["precipitation", "precip_3h", "snow_24h", "snow_depth", "is_snow_day", "temperature_2m",
                "wind_gusts_10m", "heavy_rain"],
    "events": ["venue_event_nearby", "citywide_event"],
    "gtfs": ["segment_km", "stops_between", "seg_sched_min", "schedule_tightness"],
    "ridership": ["avg_load", "avg_boardings", "avg_alightings", "load_vs_route"],
    "operational": ["prev_tp_delay", "prev_tp_delayed", "start_delay", "prev_bus_delay",
                    "prev_bus_gap_ratio", "min_since_prev_tp"],
}
PLANNING = [f for g, fs in GROUPS.items() if g != "operational" for f in fs]
OPERATIONAL = PLANNING + GROUPS["operational"]
CATEGORICAL = ["route_id", "direction_id", "stop_id", "point_type", "standard_type", "period"]
ID_COLS = ["service_date", "scheduled_dt", "route_id", "direction_id", "stop_id", "half_trip_id",
           "time_point_order", "month", "hour"]
TARGETS = ["delayed", "delay_min", "severity"]
# never allowed as features: they contain (or are computed from) this bus's own arrival
FORBIDDEN = ["actual_sec", "actual_dt", "delay_sec", "delay_min", "delayed", "severity", "headway", "month"]

PERIODS = [(0, 4, "night"), (4, 7, "early"), (7, 10, "am_peak"), (10, 16, "midday"),
           (16, 19, "pm_peak"), (19, 23, "evening"), (23, 24, "night")]


def period_of(hour: pd.Series) -> pd.Series:
    out = pd.Series("night", index=hour.index)
    for lo, hi, name in PERIODS:
        out[(hour >= lo) & (hour < hi)] = name
    return out


# ---------------------------------------------------------------------------------------------
# 1. Temporal / calendar / spatial
# ---------------------------------------------------------------------------------------------
def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    frac_hour = (df.sched_sec % 86400) / 3600.0
    df["hour_sin"] = np.sin(2 * np.pi * frac_hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * frac_hour / 24).astype("float32")
    wk = df.is_weekend.eq(0) & df.is_holiday.eq(0)
    df["am_peak"] = (wk & df.hour.between(7, 9)).astype("int8")
    df["pm_peak"] = (wk & df.hour.between(16, 18)).astype("int8")
    df["period"] = period_of(df.hour).astype("category")
    # Phase 1 finding: school dismissal overlaps the PM peak on school days
    df["school_dismissal"] = (df.is_school_day.eq(1) & df.hour.between(14, 15)).astype("int8")
    return df


def solar_elevation(local_dt: pd.Series, lat: float = BOSTON_LATLON[0], lon: float = BOSTON_LATLON[1]) -> pd.Series:
    """Sun elevation angle (degrees) for naive local Boston timestamps (NOAA approximation, accurate to ~0.5°).

    Computed once per unique minute (the data has ~0.5 M unique minutes vs ~10 M rows).
    DST is handled by converting local wall-clock time to UTC first. That is exactly the point: the same clock time
    (e.g. 17:00) is daylight in October and dark after the 2 Nov 2025 clock change.
    """
    u = pd.Series(local_dt.dt.floor("min").unique())
    utc = u.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    utc = utc.fillna((u + pd.Timedelta(hours=5)).dt.tz_localize("UTC"))       # 1 ambiguous hour/yr -> EST
    doy = utc.dt.day_of_year.values
    hr = (utc.dt.hour + utc.dt.minute / 60).values
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)
    phi = np.radians(lat)
    cz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    elev = 90 - np.degrees(np.arccos(np.clip(cz, -1, 1)))
    lut = pd.Series(elev.astype("float32"), index=u.values)
    return pd.Series(lut.reindex(local_dt.dt.floor("min").values).values, index=local_dt.index)


def add_daylight(df: pd.DataFrame) -> pd.DataFrame:
    """Daylight features (added after the seasonal-shift discussion: test = Nov–Dec, after the DST change)."""
    df["sun_elevation"] = solar_elevation(df.scheduled_dt)
    df["is_dark"] = (df.sun_elevation < -6).astype("int8")              # below civil twilight
    df["dark_peak"] = (df.is_dark.eq(1) & (df.am_peak.eq(1) | df.pm_peak.eq(1))).astype("int8")
    return df


def sort_trips(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["service_date", "route_id", "half_trip_id", "time_point_order"]).reset_index(drop=True)


def trip_mask(df: pd.DataFrame) -> pd.Series:
    """True where the row belongs to the same trip as the row above (df must be trip-sorted)."""
    same = (df.half_trip_id.eq(df.half_trip_id.shift()) & df.service_date.eq(df.service_date.shift())
            & df.route_id.eq(df.route_id.shift()))
    return same.fillna(False).astype(bool) & df.half_trip_id.notna()


def add_spatial(df: pd.DataFrame, same: pd.Series) -> pd.DataFrame:
    trip_id = (~same).cumsum()
    g = df.groupby(trip_id)
    df["stop_pos"] = (df.time_point_order / g.time_point_order.transform("max")).astype("float32")
    df["min_into_trip"] = ((df.sched_sec - g.sched_sec.transform("min")) / 60).astype("float32")
    df["scheduled_headway"] = (df.scheduled_headway / 60).clip(upper=120).astype("float32")   # minutes
    return df


# ---------------------------------------------------------------------------------------------
# 2. Leakage-safe historical rates
# ---------------------------------------------------------------------------------------------
def _global_prior(rows: pd.DataFrame, target: str, default: float) -> pd.Series:
    """Global mean of `target` over all days strictly before each service date."""
    d = rows.groupby("service_date")[target].agg(["sum", "count"]).sort_index()
    prior = (d["sum"].cumsum() - d["sum"]) / (d["count"].cumsum() - d["count"])
    return prior.replace([np.inf, -np.inf], np.nan).fillna(default).rename("prior")


def expanding_rate(rows: pd.DataFrame, keys: list[str], target: str, name: str,
                   default: float, m: float = M_SMOOTH, with_n: bool = False) -> pd.DataFrame:
    """Smoothed mean of `target` per key over all *previous* service days.

    rate(key, D) = (sum_{d<D} y + m * prior_D) / (n_{d<D} + m)
    Returns one row per (keys, service_date) present in `rows`.
    """
    daily = (rows.groupby(keys + ["service_date"], observed=True)[target].agg(["sum", "count"])
             .reset_index().sort_values("service_date", kind="stable"))
    g = daily.groupby(keys, observed=True)
    cs = g["sum"].cumsum() - daily["sum"]
    cc = g["count"].cumsum() - daily["count"]
    prior = daily.service_date.map(_global_prior(rows, target, default))
    daily[name] = ((cs + m * prior) / (cc + m)).astype("float32")
    out = keys + ["service_date", name]
    if with_n:
        daily[name + "_n"] = np.log1p(cc).astype("float32")
        out.append(name + "_n")
    return daily[out]


def rolling_rate(rows: pd.DataFrame, keys: list[str], name: str, days: int = ROLL_DAYS,
                 m: float = M_SMOOTH) -> pd.DataFrame:
    """Smoothed delay rate per key over the `days` calendar days before each service date (yesterday back)."""
    k = rows[keys[0]].astype(str)
    for kk in keys[1:]:
        k = k + "|" + rows[kk].astype(str)
    tmp = pd.DataFrame({"k": k.values, "service_date": rows.service_date.values, "y": rows.delayed.values})
    s = tmp.pivot_table(index="service_date", columns="k", values="y", aggfunc="sum")
    c = tmp.pivot_table(index="service_date", columns="k", values="y", aggfunc="count")
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s, c = s.reindex(full).fillna(0), c.reindex(full).fillna(0)
    rs = s.rolling(days, min_periods=1).sum().shift(1)
    rc = c.rolling(days, min_periods=1).sum().shift(1)
    prior = _global_prior(rows, "delayed", DEFAULT_RATE).reindex(full).ffill().fillna(DEFAULT_RATE)
    rate = (rs.add(prior * m, axis=0)) / (rc + m)
    rate.index.name = "service_date"
    long = rate.stack().rename(name).astype("float32").reset_index()
    if len(keys) > 1:
        parts = long.pop("k").str.split("|", expand=True)
        for i, kk in enumerate(keys):
            long[kk] = parts[i]
    else:
        long[keys[0]] = long.pop("k")
    return long


def build_history(years=(2024, 2025)) -> list[pd.DataFrame]:
    """Compute all historical tables from the compact 2024+2025 label history."""
    cols = ["service_date", "route_id", "direction_id", "stop_id", "hour", "dow", "delayed", "delay_min"]
    rows = pd.concat([pd.read_parquet(PROCESSED_DIR / f"bus_{y}_enriched.parquet", columns=cols) for y in years],
                     ignore_index=True)
    for c in ("route_id", "direction_id", "stop_id"):
        rows[c] = rows[c].astype(str)
    rows["period"] = period_of(rows.hour)
    stop = ["route_id", "direction_id", "stop_id"]
    tables = [
        expanding_rate(rows, ["route_id", "hour"], "delayed", "hist_route_hour", DEFAULT_RATE),
        expanding_rate(rows, ["route_id", "dow"], "delayed", "hist_route_dow", DEFAULT_RATE),
        expanding_rate(rows, stop + ["period"], "delayed", "hist_stop_period", DEFAULT_RATE),
        expanding_rate(rows, stop + ["hour"], "delayed", "hist_stop_hour", DEFAULT_RATE, with_n=True),
        expanding_rate(rows, ["route_id", "hour"], "delay_min", "hist_delay_route_hour", DEFAULT_DELAY),
        expanding_rate(rows, stop + ["hour"], "delay_min", "hist_delay_stop_hour", DEFAULT_DELAY),
        rolling_rate(rows, ["route_id"], "recent_route_28d"),
        rolling_rate(rows, stop, "recent_stop_28d"),
    ]
    return tables


def add_history(df: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.DataFrame:
    n0 = len(df)
    df["period_s"] = df.period.astype(str)
    for t in tables:
        t = t.rename(columns={"period": "period_s"})
        t["service_date"] = t.service_date.astype(df.service_date.dtype)
        keys = [c for c in t.columns if c in ("route_id", "direction_id", "stop_id", "hour", "dow",
                                              "period_s", "service_date")]
        df = df.merge(t, on=keys, how="left")
    assert len(df) == n0, "history merge changed row count"
    return df.drop(columns="period_s")


# ---------------------------------------------------------------------------------------------
# 3. Operational (real-time) features
# ---------------------------------------------------------------------------------------------
def add_operational(df: pd.DataFrame, same: pd.Series) -> pd.DataFrame:
    """Known when the bus departs its previous timepoint (prediction moment t0)."""
    prev_actual = df.actual_sec.shift().where(same)
    df["prev_tp_delay"] = df.delay_min.shift().where(same).astype("float32")
    df["prev_tp_delayed"] = (df.prev_tp_delay > DELAY_THRESHOLD_SEC / 60).astype("float32").where(same)
    trip_id = (~same).cumsum()
    df["start_delay"] = df.groupby(trip_id).delay_min.transform("first").where(same).astype("float32")
    # prediction moment: departure from previous timepoint; at the first timepoint, 1 min before scheduled time
    t0 = prev_actual.fillna(df.sched_sec - 60)
    df["min_since_prev_tp"] = ((df.sched_sec - t0) / 60).where(same).astype("float32")   # scheduled slack left

    # most recent *other* bus of the same route/direction that already passed this stop before t0
    gk = df.groupby(["service_date", "route_id", "direction_id", "stop_id"], observed=True, sort=False).ngroup()
    left = pd.DataFrame({"gk": gk.values, "t0": t0.values, "rid": np.arange(len(df))}).sort_values("t0")
    right = pd.DataFrame({"gk": gk.values, "p_actual": df.actual_sec.values,
                          "p_delay": df.delay_min.values, "p_rid": np.arange(len(df))}).sort_values("p_actual")
    m = pd.merge_asof(left, right, left_on="t0", right_on="p_actual", by="gk",
                      allow_exact_matches=False, direction="backward").sort_values("rid")
    own = m.p_rid.values == m.rid.values            # early departure at first stop could match itself
    p_delay = np.where(own, np.nan, m.p_delay.values)
    p_actual = np.where(own, np.nan, m.p_actual.values)
    df["prev_bus_delay"] = p_delay.astype("float32")
    # time since the previous bus passed, relative to the planned headway (>1 = a gap is opening ahead of us)
    df["prev_bus_gap_ratio"] = ((t0.values - p_actual) / 60 / df.scheduled_headway.values).astype("float32")
    df["prev_bus_gap_ratio"] = df.prev_bus_gap_ratio.clip(upper=10)
    return df


# ---------------------------------------------------------------------------------------------
# 4. Assemble
# ---------------------------------------------------------------------------------------------
LOAD_COLS = (ID_COLS + TARGETS + ["sched_sec", "actual_sec", "hour", "dow", "is_weekend", "point_type", "standard_type",
             "scheduled_headway", "trip_start_hour", "stop_lat", "stop_lon"]
             + GROUPS["calendar"][:-1] + GROUPS["weather"] + GROUPS["events"] + GROUPS["gtfs"] + GROUPS["ridership"])


def build_year(year: int, tables: list[pd.DataFrame]) -> pd.DataFrame:
    t = time.time()
    df = pd.read_parquet(PROCESSED_DIR / f"bus_{year}_enriched.parquet", columns=list(dict.fromkeys(LOAD_COLS)))
    for c in ("route_id", "direction_id", "stop_id"):
        df[c] = df[c].astype(str)
    df = sort_trips(df)
    same = trip_mask(df)
    df = add_temporal(df)
    df = add_daylight(df)
    df = add_spatial(df, same)
    df = add_operational(df, same)
    df = add_history(df, tables)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    keep = list(dict.fromkeys(ID_COLS + TARGETS + OPERATIONAL))
    df = df[keep]
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    print(f"  {year}: {len(df):,} rows, {len(OPERATIONAL)} features ({time.time() - t:.0f}s)")
    return df


def split_of(dates: pd.Series) -> pd.Series:
    return pd.Series(np.select([dates <= TRAIN_END, dates <= VALID_END], ["train", "valid"], "test"), index=dates.index)


def main():
    t = time.time()
    print("history tables ...")
    tables = build_history()
    print(f"  done ({time.time() - t:.0f}s)")
    d24 = build_year(2024, tables)
    d24.to_parquet(FEAT_DIR / "drift_2024.parquet", index=False)
    del d24
    d25 = build_year(2025, tables)
    sp = split_of(d25.service_date)
    sizes = {}
    for s in ("train", "valid", "test"):
        part = d25[sp.values == s]
        part.to_parquet(FEAT_DIR / f"{s}.parquet", index=False)
        sizes[s] = {"rows": len(part), "pct_delayed": round(float(part.delayed.mean()) * 100, 2),
                    "from": str(part.service_date.min().date()), "to": str(part.service_date.max().date())}
    spec = {"groups": GROUPS, "planning": PLANNING, "operational": OPERATIONAL, "categorical": CATEGORICAL,
            "id_cols": ID_COLS, "targets": TARGETS, "forbidden": FORBIDDEN, "splits": sizes,
            "smoothing_m": M_SMOOTH, "rolling_days": ROLL_DAYS}
    (REPORTS / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(sizes, indent=2), f"\ntotal {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
