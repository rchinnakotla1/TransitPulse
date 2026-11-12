"""Phase 5e: CHRONIC delay segments - the planner's "fix-list".

A segment = (route, direction, timepoint stop); the delay *added* on a segment = delay here - delay at the previous
timepoint (so a segment is not blamed for lateness it inherited).

Protocol (honest, out-of-time):
* identify chronic segments on Jan-Oct 2025 (train + valid)
* check that they are still the worst in Nov-Dec 2025 (test) - a fix-list that does not persist is useless

Chronic = late rate >= 1.5 x network average in >= 80 % of months (consistently bad, not one bad week).
Suggested running-time adjustment (per segment x period) = median delay added in that period, if positive: the
schedule would then match a typical day.  This is a starting point for planners, not an automatic schedule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.encoding import load_split
from src.extensions.common import EXT_DIR, log

COLS = ["service_date", "route_id", "direction_id", "stop_id", "period", "delay_min", "delayed", "prev_tp_delay",
        "seg_sched_min", "segment_km"]
SEG = ["route_id", "direction_id", "stop_id"]
MIN_N = 500


def _prep(df):
    for c in SEG + ["period"]:
        df[c] = df[c].astype(str)
    df["added"] = (df.delay_min - df.prev_tp_delay).astype("float32")
    df["month"] = df.service_date.dt.month
    return df


def segment_table(df: pd.DataFrame, network_rate: float) -> pd.DataFrame:
    g = df.groupby(SEG).agg(n=("delayed", "size"), late_rate=("delayed", "mean"), mean_delay=("delay_min", "mean"),
                            median_added=("added", "median"), mean_added=("added", "mean"),
                            sched_min=("seg_sched_min", "median"), km=("segment_km", "median"))
    monthly = df.groupby(SEG + ["month"]).delayed.mean().unstack()
    g["months_bad_%"] = ((monthly >= 1.5 * network_rate).sum(1) / monthly.notna().sum(1) * 100).reindex(g.index)
    g = g[g.n >= MIN_N]
    g["chronic"] = g["months_bad_%"] >= 80
    return g.sort_values("late_rate", ascending=False)


def run():
    hist = _prep(pd.concat([load_split(s, COLS) for s in ("train", "valid")], ignore_index=True))
    test = _prep(load_split("test", COLS))
    rate = hist.delayed.mean()
    seg = segment_table(hist, rate)
    seg_te = test.groupby(SEG).agg(n_test=("delayed", "size"), late_rate_test=("delayed", "mean"))
    seg = seg.join(seg_te, how="left")
    both = seg.dropna(subset=["late_rate_test"])
    rho = spearmanr(both.late_rate, both.late_rate_test).statistic
    top20 = both.head(20)
    q75 = both.late_rate_test.quantile(0.75)
    persist = float((top20.late_rate_test >= q75).mean())
    chronic = seg[seg.chronic]
    ch_te = both[both.chronic]
    summary = {"segments": len(seg), "chronic_segments": int(seg.chronic.sum()), "network_late_rate_janoct": rate,
               "spearman_janoct_vs_novdec": rho, "top20_still_top_quartile_%": persist * 100,
               "chronic_late_rate_test": ch_te.late_rate_test.mean() if len(ch_te) else np.nan,
               "other_late_rate_test": both[~both.chronic].late_rate_test.mean()}
    seg.reset_index().to_csv(EXT_DIR / "chronic_segments.csv", index=False)
    # running-time suggestion for chronic segments, by period
    sub = hist.merge(chronic.reset_index()[SEG], on=SEG)
    adj = sub.groupby(SEG + ["period"]).agg(n=("added", "size"), median_added=("added", "median"),
                                            late_rate=("delayed", "mean"))
    adj = adj[adj.n >= 100]
    adj["suggest_add_min"] = adj.median_added.clip(lower=0).round(1)
    adj.reset_index().sort_values("suggest_add_min", ascending=False).to_csv(EXT_DIR / "chronic_adjustments.csv", index=False)
    route = seg.reset_index().groupby("route_id").agg(segments=("chronic", "size"), chronic=("chronic", "sum"),
                                                      mean_late_rate=("late_rate", "mean")).sort_values("chronic", ascending=False)
    route.to_csv(EXT_DIR / "chronic_by_route.csv")
    pd.Series(summary).to_csv(EXT_DIR / "chronic_summary.csv")
    log(f"{pd.Series(summary).round(3)}\n{seg.head(10).round(3)}\n{route.head(8)}")
    return summary


if __name__ == "__main__":
    run()
