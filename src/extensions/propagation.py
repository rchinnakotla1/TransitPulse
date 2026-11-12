"""Phase 5c: delay PROPAGATION - how lateness carries forward along a trip and from one bus to the next.

Descriptive analysis on all 2025 stop events (train+valid+test feature files; no model is fitted, so no leakage issue).
Questions for planners / dispatchers:
1. If a bus is late at one timepoint, how likely is it to still be late at the next? (severity transition matrix)
2. Do late buses recover?  Recovery rate = P(<=5 min late now | >5 min late at previous timepoint), by route & period.
3. Where along the route is delay added?  Mean delay by position-along-route decile; delay added per segment.
4. Does a late start doom the trip?  P(late at the final timepoint | delay at trip start).
5. Bus-to-bus: does the bus ahead's delay predict this bus's delay? (correlation, by period)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.extensions.common import EXT_DIR, log
from src.encoding import load_split

COLS = ["service_date", "route_id", "direction_id", "half_trip_id", "time_point_order", "point_type", "stop_pos",
        "hour", "period", "delay_min", "delayed", "prev_tp_delay", "start_delay", "prev_bus_delay"]
BANDS = [-np.inf, 1, 5, 10, np.inf]
NAMES = ["on time (<1)", "minor (1-5)", "moderate (5-10)", "major (>10)"]


def load_all() -> pd.DataFrame:
    df = pd.concat([load_split(s, COLS) for s in ("train", "valid", "test")], ignore_index=True)
    for c in ("route_id", "direction_id", "period", "point_type"):
        df[c] = df[c].astype(str)
    return df


def transition_matrix(df) -> pd.DataFrame:
    d = df.dropna(subset=["prev_tp_delay"])
    a = pd.cut(d.prev_tp_delay, BANDS, labels=NAMES); b = pd.cut(d.delay_min, BANDS, labels=NAMES)
    return pd.crosstab(a, b, normalize="index").rename_axis(index="previous timepoint", columns="this timepoint")


def recovery(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[df.prev_tp_delay > 5]
    rec = (d.delay_min <= 5)
    by_route = d.assign(recovered=rec).groupby("route_id").agg(
        late_events=("recovered", "size"), recovery_rate=("recovered", "mean")).sort_values("recovery_rate")
    by_period = d.assign(recovered=rec).groupby("period").agg(
        late_events=("recovered", "size"), recovery_rate=("recovered", "mean"))
    return by_route, by_period


def growth_along_route(df) -> pd.DataFrame:
    d = df.dropna(subset=["stop_pos"])
    d = d.assign(pos_decile=(d.stop_pos.clip(0, 0.999) * 10).astype(int) * 10)
    return d.groupby("pos_decile").agg(mean_delay=("delay_min", "mean"), median_delay=("delay_min", "median"),
                                       late_rate=("delayed", "mean"), n=("delayed", "size"))


def segment_added(df) -> pd.DataFrame:
    d = df.dropna(subset=["prev_tp_delay"])
    add = d.delay_min - d.prev_tp_delay
    return d.assign(added=add).groupby("route_id").agg(
        mean_added_min=("added", "mean"), p90_added_min=("added", lambda x: x.quantile(0.9)),
        share_segments_adding_2min=("added", lambda x: (x > 2).mean())).sort_values("mean_added_min", ascending=False)


def late_start_doom(df) -> pd.DataFrame:
    end = df[df.point_type == "Endpoint"].dropna(subset=["start_delay"])
    band = pd.cut(end.start_delay, [-np.inf, 0, 2, 5, 10, np.inf],
                  labels=["early/on time", "0-2 late", "2-5 late", "5-10 late", ">10 late"])
    return end.assign(start_band=band).groupby("start_band", observed=True).agg(
        trips=("delayed", "size"), p_late_at_end=("delayed", "mean"), mean_end_delay=("delay_min", "mean"))


def bus_to_bus(df) -> pd.DataFrame:
    d = df.dropna(subset=["prev_bus_delay"])
    return d.groupby("period").apply(lambda g: pd.Series({
        "corr_with_bus_ahead": g.delay_min.corr(g.prev_bus_delay),
        "corr_with_prev_timepoint": g.delay_min.corr(g.prev_tp_delay), "n": len(g)}), include_groups=False)


def main():
    df = load_all()
    log(f"propagation: {len(df):,} stop events")
    tm = transition_matrix(df); tm.to_csv(EXT_DIR / "prop_transition.csv")
    br, bp = recovery(df); br.to_csv(EXT_DIR / "prop_recovery_route.csv"); bp.to_csv(EXT_DIR / "prop_recovery_period.csv")
    g = growth_along_route(df); g.to_csv(EXT_DIR / "prop_growth.csv")
    s = segment_added(df); s.to_csv(EXT_DIR / "prop_segment_added.csv")
    ls = late_start_doom(df); ls.to_csv(EXT_DIR / "prop_late_start.csv")
    bb = bus_to_bus(df); bb.to_csv(EXT_DIR / "prop_bus_to_bus.csv")
    overall = {"corr_prev_tp": df.delay_min.corr(df.prev_tp_delay),
               "recovery_rate_overall": float((df[df.prev_tp_delay > 5].delay_min <= 5).mean()),
               "p_stay_late": float((df[df.prev_tp_delay > 5].delayed).mean()),
               "p_become_late": float(df[df.prev_tp_delay <= 5].delayed.mean())}
    pd.Series(overall).to_csv(EXT_DIR / "prop_overall.csv")
    log(f"{pd.Series(overall).round(3)}\n{tm.round(3)}\n{bp.round(3)}\n{ls.round(3)}\n{bb.round(3)}")


if __name__ == "__main__":
    main()
