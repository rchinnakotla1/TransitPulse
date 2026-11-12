"""Phase 4: human-readable rules that summarise the XGBoost model (global surrogate) + high-risk route x period table.

Why a surrogate?  SHAP tells *how much* each feature pushed one prediction; planners also asked for simple
"if-then" statements ("evening peak on route 28 inbound runs late").  We fit a shallow regression tree that mimics the
XGBoost probability on readable features only.  It is an *explanation of the model*, not a second predictor, so we
report its **fidelity** (how well it reproduces XGBoost) on data it was not fitted on.

* fit on a 200 k stratified VALIDATION sample (XGBoost probabilities as target)
* fidelity measured on the 50 k TEST sample used for SHAP (R^2 vs XGBoost prob, ROC-AUC vs actual labels)

Outputs (``reports/xai/``): ``rules_{mode}.txt``, ``rule_leaves_{mode}.csv``, ``surrogate_fidelity.csv``,
``route_period_risk.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.tree import DecisionTreeRegressor, export_text

from src.config import REPORTS
from src.encoding import apply_levels, feature_list, load_split
from src.xai.feature_text import label
from src.xai.shap_analysis import XAI_DIR, load_model

READABLE = ["hour", "dow", "is_weekend", "am_peak", "pm_peak", "is_school_day", "is_dark", "hist_stop_hour",
            "hist_route_hour", "recent_route_28d", "schedule_tightness", "seg_sched_min", "avg_load", "precip_3h",
            "snow_depth", "stop_pos", "time_point_order", "scheduled_headway"]
READABLE_OP = READABLE + ["prev_tp_delay", "start_delay", "prev_bus_gap_ratio"]
DEPTH, MIN_LEAF = 4, 2000


def _readable(mode: str) -> list[str]:
    return READABLE_OP if mode == "operational" else READABLE


def _paths(tree: DecisionTreeRegressor, names: list[str]) -> dict[int, str]:
    """leaf id -> 'cond AND cond ...' in plain words."""
    t = tree.tree_
    out = {}

    def walk(node, conds):
        if t.children_left[node] == -1:
            out[node] = " AND ".join(conds) or "(all)"
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.3g}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.3g}"])
    walk(0, [])
    return out


def surrogate(mode: str) -> dict:
    feats = feature_list(mode)
    cols = _readable(mode)
    model = load_model(mode)
    va = load_split("valid", columns=list(dict.fromkeys(["delayed", "hour"] + feats)), sample=200_000, seed=11)
    va = apply_levels(va)
    p_va = model.predict_proba(va[feats])[:, 1]
    Xva = va[cols].astype("float32").fillna(-1)
    tree = DecisionTreeRegressor(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=0).fit(Xva, p_va)

    te = pd.read_parquet(XAI_DIR / f"shap_{mode}.parquet", columns=list(dict.fromkeys(cols + ["prob", "delayed"])))
    Xte = te[cols].astype("float32").fillna(-1)
    s_te = tree.predict(Xte)
    fid = {"mode": mode, "depth": DEPTH, "leaves": tree.get_n_leaves(),
           "r2_vs_xgb": r2_score(te.prob, s_te),
           "auc_surrogate": roc_auc_score(te.delayed, s_te), "auc_xgb": roc_auc_score(te.delayed, te.prob)}
    # readability vs fidelity trade-off: how deep must a single tree be to mimic XGBoost?
    sweep = []
    for dpt in (2, 3, 4, 6, 8, 12):
        t2 = DecisionTreeRegressor(max_depth=dpt, min_samples_leaf=MIN_LEAF // 4, random_state=0).fit(Xva, p_va)
        s2 = t2.predict(Xte)
        sweep.append({"mode": mode, "depth": dpt, "leaves": t2.get_n_leaves(), "r2_vs_xgb": r2_score(te.prob, s2),
                      "auc_surrogate": roc_auc_score(te.delayed, s2)})
    pd.DataFrame(sweep).to_csv(XAI_DIR / f"surrogate_depth_{mode}.csv", index=False)

    (XAI_DIR / f"rules_{mode}.txt").write_text(export_text(tree, feature_names=cols, decimals=2), encoding="utf-8")
    paths = _paths(tree, cols)
    leaf = tree.apply(Xte)
    lv = pd.DataFrame({"leaf": leaf, "surrogate_p": s_te, "xgb_p": te.prob, "actual": te.delayed})
    tab = lv.groupby("leaf").agg(n=("actual", "size"), surrogate_p=("surrogate_p", "mean"),
                                 xgb_p=("xgb_p", "mean"), actual_late=("actual", "mean")).reset_index()
    tab["share_%"] = tab.n / tab.n.sum() * 100
    tab["rule"] = tab.leaf.map(paths)
    tab = tab.sort_values("surrogate_p", ascending=False)
    tab.to_csv(XAI_DIR / f"rule_leaves_{mode}.csv", index=False)
    print(pd.Series(fid).to_string())
    return {"fidelity": fid, "leaves": tab, "tree": tree, "features": cols}


def rule_sentence(rule: str) -> str:
    """'hist_stop_hour > 0.41 AND pm_peak > 0.5' -> plain words for reports / dashboard.

    Repeated conditions on the same feature along a tree path (e.g. > 0.32, > 0.42, <= 0.65) are merged into the
    tightest range ("between 42% and 65%") so the sentence stays short.
    """
    bounds: dict[str, list] = {}          # feature -> [lower (exclusive), upper (inclusive)], in order of first use
    for c in rule.split(" AND "):
        f, op, v = c.split(" ")
        lo, hi = bounds.setdefault(f, [None, None])
        v = float(v)
        if op == ">":
            bounds[f][0] = v if lo is None else max(lo, v)
        else:
            bounds[f][1] = v if hi is None else min(hi, v)
    parts = []
    for f, (lo, hi) in bounds.items():
        name = label(f) if f != "hour" else "hour of day"
        if f in ("is_weekend", "am_peak", "pm_peak", "is_school_day", "is_dark"):
            parts.append(f"{name}: {'yes' if lo is not None else 'no'}")
            continue
        if f == "dow" and (lo is None or hi is None):
            v = lo if lo is not None else hi
            parts.append("Sat/Sun" if lo is not None and v >= 4.5 else ("Mon-Fri" if hi is not None and v >= 4.5 else
                         f"weekday index {'>' if lo is not None else '≤'} {v:.1f}"))
            continue
        pct = f.startswith(("hist_", "recent_"))
        fmt = (lambda x: f"{x * 100:.0f}%") if pct else (lambda x: f"{x:.3g}")
        if lo is not None and hi is not None:
            parts.append(f"{name} between {fmt(lo)} and {fmt(hi)}")
        elif lo is not None:
            parts.append(f"{name} {'above' if pct else '>'} {fmt(lo)}")
        else:
            parts.append(f"{name} {'at most' if pct else '≤'} {fmt(hi)}")
    return "; ".join(parts)


def route_period_risk(mode: str = "planning", min_n: int = 2000) -> pd.DataFrame:
    """Where and when is the model (and reality) most pessimistic?  Test predictions by route x direction x period."""
    p = pd.read_parquet(REPORTS / "preds" / f"test_{mode}.parquet",
                        columns=["route_id", "direction_id", "hour", "delayed", "p_XGB"])
    bins = [-1, 6, 9, 15, 18, 21, 30]
    names = ["early (<7)", "AM peak (7-9)", "midday (10-15)", "PM peak (16-18)", "evening (19-21)", "night (22+)"]
    p["period"] = pd.cut(p.hour, bins, labels=names)
    g = p.groupby(["route_id", "direction_id", "period"], observed=True).agg(
        n=("delayed", "size"), predicted=("p_XGB", "mean"), actual=("delayed", "mean")).reset_index()
    g = g[g.n >= min_n]
    base = p.delayed.mean()
    g["lift_vs_avg"] = g.actual / base
    g = g.sort_values("predicted", ascending=False).reset_index(drop=True)
    g.to_csv(XAI_DIR / "route_period_risk.csv", index=False)
    return g


if __name__ == "__main__":
    rows = []
    for md in ("planning", "operational"):
        rows.append(surrogate(md)["fidelity"])
    pd.DataFrame(rows).to_csv(XAI_DIR / "surrogate_fidelity.csv", index=False)
    print(route_period_risk().head(15).round(3).to_string())
