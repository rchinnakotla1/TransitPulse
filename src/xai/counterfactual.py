"""Phase 4: what-if (counterfactual) analysis on the deployed XGBoost models.

Questions a planner / dispatcher can ask:
* "If we add 2 scheduled minutes to this segment, does the risk drop below the alert threshold?"  (``seg_sched_min``)
* "What if it rains 5 mm in the next 3 hours?"  (weather scenario)
* "What if the bus is already 4 min late at the previous timepoint?"  (operational mode, ``prev_tp_delay``)

Caveat (stated in the dashboard and the thesis): these are **model-based** what-ifs, i.e. what the model *would
predict* for a changed input.  They are associations learned from 2025 data, not causal effects; a schedule change also
alters driver behaviour and the bus ahead, which the model cannot see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.encoding import feature_list


def _linked(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Keep derived features consistent with the changed input."""
    if feature == "seg_sched_min" and "schedule_tightness" in df:
        km = df["segment_km"].replace(0, np.nan)
        df["schedule_tightness"] = (df["seg_sched_min"] / km).astype("float32")
    if feature == "prev_tp_delay" and "prev_tp_delayed" in df:
        df["prev_tp_delayed"] = (df["prev_tp_delay"] > 5).astype(df["prev_tp_delayed"].dtype)
    if feature in ("precip_3h", "precipitation") and "heavy_rain" in df:
        df["heavy_rain"] = (df["precipitation"] >= 4).astype(df["heavy_rain"].dtype)
    return df


def what_if(model, row: pd.DataFrame, mode: str, feature: str, values) -> pd.DataFrame:
    """Probability of >5 min lateness for one stop event (1-row frame) as ``feature`` takes each value."""
    feats = feature_list(mode)
    X = pd.concat([row[feats]] * len(values), ignore_index=True)
    X[feature] = np.asarray(values, dtype="float32")
    X = _linked(X, feature)
    return pd.DataFrame({feature: values, "prob": model.predict_proba(X[feats])[:, 1]})


def weather_scenarios(model, row: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Same trip under dry / rain / heavy rain / snow conditions."""
    feats = feature_list(mode)
    sc = {"dry": dict(precipitation=0, precip_3h=0, heavy_rain=0, snow_24h=0, snow_depth=0, is_snow_day=0),
          "light rain": dict(precipitation=1, precip_3h=2, heavy_rain=0, snow_24h=0, snow_depth=0, is_snow_day=0),
          "heavy rain": dict(precipitation=6, precip_3h=12, heavy_rain=1, snow_24h=0, snow_depth=0, is_snow_day=0),
          "snow day": dict(precipitation=1, precip_3h=3, heavy_rain=0, snow_24h=10, snow_depth=0.10, is_snow_day=1)}   # snow_depth in m
    X = pd.concat([row[feats]] * len(sc), ignore_index=True)
    for i, vals in enumerate(sc.values()):
        for k, v in vals.items():
            if k in X:
                X.loc[i, k] = v
    return pd.DataFrame({"scenario": list(sc), "prob": model.predict_proba(X[feats])[:, 1]})


def minimal_schedule_change(model, row: pd.DataFrame, mode: str, threshold: float,
                            max_add: float = 10.0, step: float = 0.5) -> dict:
    """Smallest extra scheduled running time (minutes) that moves the prediction below ``threshold``."""
    base = float(row["seg_sched_min"].iloc[0])
    if np.isnan(base):
        return {"possible": False, "reason": "segment running time unknown (first timepoint)"}
    adds = np.arange(0, max_add + step, step)
    res = what_if(model, row, mode, "seg_sched_min", base + adds)
    res["added_min"] = adds
    below = res[res.prob < threshold]
    p0 = float(res.prob.iloc[0])
    if p0 < threshold:
        return {"possible": True, "added_min": 0.0, "p_before": p0, "p_after": p0, "curve": res,
                "reason": "already below threshold"}
    if below.empty:
        return {"possible": False, "p_before": p0, "p_after": float(res.prob.iloc[-1]), "curve": res,
                "reason": f"adding up to {max_add:.0f} min is not enough according to the model"}
    r = below.iloc[0]
    return {"possible": True, "added_min": float(r.added_min), "p_before": p0, "p_after": float(r.prob), "curve": res,
            "reason": "model-based estimate"}


def schedule_fix_population(model, rows: pd.DataFrame, mode: str, threshold: float,
                            adds=(0, 1, 2, 3, 4, 5, 7.5, 10)) -> pd.DataFrame:
    """For many high-risk events at once: probability after adding each amount of scheduled time to the segment.

    Returns one row per event with ``p_+{a}`` columns and ``min_add`` (smallest tested amount that goes below
    ``threshold``, NaN if none). Planner use: "how many of the high-risk stop events could a small running-time
    adjustment fix, according to the model?"
    """
    feats = feature_list(mode)
    base = rows[feats].copy()
    out = pd.DataFrame(index=rows.index)
    for a in adds:
        X = base.copy()
        X["seg_sched_min"] = (rows["seg_sched_min"] + a).astype("float32")
        X = _linked(X, "seg_sched_min")
        out[f"p_+{a}"] = model.predict_proba(X[feats])[:, 1]
    below = np.column_stack([out[f"p_+{a}"] < threshold for a in adds])
    first = np.where(below.any(1), below.argmax(1), -1)
    out["min_add"] = [adds[i] if i >= 0 else np.nan for i in first]
    return out


def delay_tipping_points(model, rows: pd.DataFrame, mode: str, threshold: float,
                         grid=np.arange(-2, 15.5, 0.5)) -> tuple[pd.DataFrame, pd.Series]:
    """Operational: for each event, the previous-timepoint delay at which the alert threshold is first crossed.

    Returns (curves: events x grid probabilities, tipping point per event; NaN = never crosses within the grid).
    """
    feats = feature_list(mode)
    curves = {}
    for v in grid:
        X = rows[feats].copy()
        X["prev_tp_delay"] = np.float32(v)
        X = _linked(X, "prev_tp_delay")
        curves[v] = model.predict_proba(X[feats])[:, 1]
    C = pd.DataFrame(curves, index=rows.index)
    over = C.values >= threshold
    tip = pd.Series(np.where(over.any(1), np.asarray(grid)[over.argmax(1)], np.nan), index=rows.index)
    return C, tip
