"""Phase 5d: BUS BUNCHING - two buses of the same route arriving almost together, followed by a long gap.

Definition (common in the transit literature and close to MBTA's headway standard):
* bunched  : actual headway < 25 % of the scheduled headway
* big gap  : actual headway > 150 % of the scheduled headway
``headway`` (actual seconds since the previous bus at this stop) and ``scheduled_headway`` come from the cleaned MBTA
file and are joined to the feature rows on (service_date, half_trip_id, time_point_order).

Analysis: bunching rate by route, hour and position along the route.
Model: can we predict that *this* bus will arrive bunched at the next timepoint?
* operational XGBoost (knows the current gap to the bus ahead, ``prev_bus_gap_ratio``)
* planning XGBoost (day-ahead, no live data)
* baseline: the gap ratio itself as a score. NOTE (decision 30): ``prev_bus_gap_ratio`` is the time since the last
  bus that has *already passed* the target stop. When the leader is close it has not passed yet and is skipped, so
  a LARGE ratio signals a bunched pair (U-shaped relation). The first run scored ``-ratio`` and got AUC 0.305;
  the baseline therefore scores ``+ratio`` (AUC 0.695).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PROCESSED_DIR
from src.extensions.common import EXT_DIR, MODEL_DIR, load_sets, log, xgb_params

KEY = ["service_date", "half_trip_id", "time_point_order"]
BUNCH, GAP = 0.25, 1.5


def headways() -> pd.DataFrame:
    h = pd.read_parquet(PROCESSED_DIR / "bus_2025_clean.parquet", columns=KEY + ["headway", "scheduled_headway"])
    h = h.dropna(subset=["headway", "scheduled_headway"])
    h["hw_ratio"] = (h.headway / h.scheduled_headway).astype("float32")
    h["bunched"] = (h.hw_ratio < BUNCH).astype("int8")
    h["big_gap"] = (h.hw_ratio > GAP).astype("int8")
    h["half_trip_id"] = h.half_trip_id.astype("string")
    return h.drop_duplicates(KEY)[KEY + ["hw_ratio", "bunched", "big_gap"]]


def attach(df: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df["half_trip_id"] = df.half_trip_id.astype("string")
    return df.merge(h, on=KEY, how="inner")


def descriptive(h_all: pd.DataFrame) -> dict:
    out = {}
    d = h_all
    for c in ("route_id", "hour"):
        out[c] = d.groupby(c, observed=True).agg(events=("bunched", "size"), bunched_rate=("bunched", "mean"),
                                                 big_gap_rate=("big_gap", "mean"))
    d = d.dropna(subset=["stop_pos"])
    d = d.assign(pos_decile=(d.stop_pos.clip(0, 0.999) * 10).astype(int) * 10)
    out["position"] = d.groupby("pos_decile").agg(bunched_rate=("bunched", "mean"), big_gap_rate=("big_gap", "mean"))
    # bunching vs lateness: is a bunched bus usually a late bus (catching up is impossible -> the *following* bus catches up)
    out["vs_delay"] = d.groupby("bunched").agg(mean_delay=("delay_min", "mean"), late_rate=("delayed", "mean"))
    return out


def run():
    h = headways()
    log(f"headway rows {len(h):,}; bunched {h.bunched.mean():.3%}, big gap {h.big_gap.mean():.3%}")
    rows = []
    for mode in ("operational", "planning"):
        feats, tr, va, te = load_sets(mode, extra=["stop_pos"] if mode == "operational" else ["stop_pos", "prev_bus_gap_ratio"])
        tr, va, te = attach(tr, h), attach(va, h), attach(te, h)
        if mode == "operational":
            desc = descriptive(te.assign(route_id=te.route_id.astype(str)))
            for k, v in desc.items():
                v.to_csv(EXT_DIR / f"bunching_by_{k}.csv")
            log(f"bunching by route (test):\n{desc['route_id'].sort_values('bunched_rate').round(3).tail(8)}")
            ok = te.prev_bus_gap_ratio.notna()
            # +ratio: a large 'time since the last passed bus' means the direct leader is still en route (decision 30)
            score = te.prev_bus_gap_ratio.fillna(te.prev_bus_gap_ratio.median())
            rows.append({"model": "B0 current gap ratio", "mode": "operational",
                         "roc_auc": roc_auc_score(te.bunched, score), "pr_auc": average_precision_score(te.bunched, score),
                         "base_rate": te.bunched.mean(), "coverage_%": ok.mean() * 100})
        p = {**xgb_params(mode), "eval_metric": "aucpr"}
        m = xgb.XGBClassifier(**p).fit(tr[feats], tr.bunched, eval_set=[(va[feats], va.bunched)], verbose=False)
        m.save_model(MODEL_DIR / f"BUNCH_{mode}.json")
        pr = m.predict_proba(te[feats])[:, 1]
        rows.append({"model": "XGB", "mode": mode, "roc_auc": roc_auc_score(te.bunched, pr),
                     "pr_auc": average_precision_score(te.bunched, pr), "base_rate": te.bunched.mean(),
                     "coverage_%": 100.0, "trees": m.best_iteration + 1})
        if mode == "operational":
            imp = pd.Series(m.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
            imp.head(15).to_csv(EXT_DIR / "bunching_importance.csv")
        log(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(EXT_DIR / "bunching_metrics.csv", index=False)
    log(f"\n{res.round(3).to_string(index=False)}")
    return res


if __name__ == "__main__":
    run()
