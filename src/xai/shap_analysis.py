"""Phase 4: SHAP explanations for the deployed XGBoost models.

We use XGBoost's built-in TreeSHAP (``pred_contribs=True``). It computes exact Shapley values for tree ensembles, supports native
categorical splits (``route_id``, ``stop_id``) and runs multi-threaded. The ``shap`` package is only used for plotting.

Outputs (per mode) in ``reports/xai/``
* ``shap_{mode}.parquet``: a stratified 50 k test sample with features, probability, base value and one SHAP column per feature
* ``shap_global_{mode}.csv``: mean |SHAP| per feature (log-odds) plus rank
* ``shap_groups_{mode}.csv``: SHAP summed within each feature group (temporal, history, weather, ...)
* ``shap_stability.csv``: rank agreement of feature importance between XGBoost and LightGBM (explanation robustness)

Run: ``python -m src.xai.shap_analysis``
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import REPORTS, ROOT
from src.encoding import apply_levels, feature_list, load_split, spec

XAI_DIR = REPORTS / "xai"; XAI_DIR.mkdir(exist_ok=True)
MODEL_DIR = ROOT / "models"
N_SAMPLE = 50_000
ID = ["service_date", "route_id", "direction_id", "stop_id", "hour", "half_trip_id", "time_point_order", "delayed", "delay_min"]


def load_model(mode: str) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier()
    m.load_model(MODEL_DIR / f"XGB_{mode}.json")
    return m


def threshold(mode: str, model: str = "XGB") -> float:
    m = pd.read_csv(REPORTS / f"metrics_{mode}.csv")
    return float(m[(m.model == model) & (m.split == "test")].threshold.iloc[0])


def contributions(model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP in log-odds space; returns (n × features SHAP matrix, base value)."""
    dm = xgb.DMatrix(X, enable_categorical=True)
    c = model.get_booster().predict(dm, pred_contribs=True)
    return c[:, :-1], float(c[0, -1])


def group_of() -> dict[str, str]:
    g = {}
    for grp, fs in spec()["groups"].items():
        for f in fs:
            g[f] = grp
    return g


def explain_mode(mode: str) -> pd.DataFrame:
    t = time.time()
    feats = feature_list(mode)
    te = load_split("test", columns=list(dict.fromkeys(ID + feats)), sample=N_SAMPLE, seed=7)
    te = apply_levels(te).reset_index(drop=True)
    model = load_model(mode)
    sv, base = contributions(model, te[feats])
    prob = model.predict_proba(te[feats])[:, 1]
    # sanity: SHAP values must add up to the model's raw log-odds output (local accuracy property).
    # Compare with the margin directly: log(p / (1-p)) is numerically unstable for p close to 0 or 1 (operational mode).
    logit = model.get_booster().predict(xgb.DMatrix(te[feats], enable_categorical=True), output_margin=True)
    err = np.abs(sv.sum(1) + base - logit).max()
    assert err < 1e-3, f"SHAP additivity violated ({err})"
    out = te.copy()
    out["prob"] = prob.astype("float32"); out["logit"] = logit.astype("float32"); out["base_value"] = base
    for i, f in enumerate(feats):
        out[f"shap__{f}"] = sv[:, i].astype("float32")
    out.to_parquet(XAI_DIR / f"shap_{mode}.parquet", index=False)

    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)})
    imp["group"] = imp.feature.map(group_of())
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp["share_%"] = imp.mean_abs_shap / imp.mean_abs_shap.sum() * 100
    imp.to_csv(XAI_DIR / f"shap_global_{mode}.csv", index=False)

    # grouped SHAP: sum the signed SHAP within a group per row first, then take mean |.| (Shapley values are additive)
    gmap = group_of()
    groups = sorted({gmap[f] for f in feats})
    gsv = pd.DataFrame({g: sv[:, [i for i, f in enumerate(feats) if gmap[f] == g]].sum(1) for g in groups})
    gimp = gsv.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()
    gimp["share_%"] = gimp.mean_abs_shap / gimp.mean_abs_shap.sum() * 100
    gimp.index.name = "group"
    gimp.to_csv(XAI_DIR / f"shap_groups_{mode}.csv")
    print(f"{mode}: {len(te):,} rows, additivity error {err:.1e}, base p={1 / (1 + np.exp(-base)):.3f} ({time.time() - t:.0f}s)")
    print(imp.head(10)[["rank", "feature", "group", "share_%"]].round(2).to_string(index=False))
    return imp


def stability(mode: str = "planning") -> pd.DataFrame:
    """Do two different boosting libraries agree on *why*? Spearman rank correlation of mean |SHAP|."""
    import lightgbm as lgb
    from scipy.stats import spearmanr
    path = MODEL_DIR / f"LGB_{mode}.txt"
    if not path.exists():
        return pd.DataFrame()
    feats = feature_list(mode)
    df = pd.read_parquet(XAI_DIR / f"shap_{mode}.parquet")
    booster = lgb.Booster(model_file=str(path))
    c = booster.predict(df[feats], pred_contrib=True)[:, :-1]
    a = pd.read_csv(XAI_DIR / f"shap_global_{mode}.csv").set_index("feature").mean_abs_shap
    b = pd.Series(np.abs(c).mean(0), index=feats)
    rho = spearmanr(a[feats], b[feats]).statistic
    top10 = len(set(a.sort_values().index[-10:]) & set(b.sort_values().index[-10:]))
    # Correlated features (e.g. prev_tp_delay / prev_tp_delayed / min_since_prev_tp) can swap credit between models,
    # so we also compare at the feature-GROUP level, which is how explanations are communicated.
    g = group_of()
    ga = (a[feats] / a[feats].sum()).groupby(lambda f: g[f]).sum()
    gb = (b / b.sum()).groupby(lambda f: g[f]).sum()
    res = pd.DataFrame([{"mode": mode, "spearman_rho_all": rho, "top10_overlap": top10,
                         "group_spearman": spearmanr(ga, gb[ga.index]).statistic,
                         "group_max_abs_diff_pts": float((ga - gb[ga.index]).abs().max() * 100)}])
    pd.DataFrame({"xgb_%": ga * 100, "lgb_%": gb[ga.index] * 100}).round(2).to_csv(XAI_DIR / f"shap_stability_groups_{mode}.csv")
    pd.DataFrame({"xgb_%": a[feats] / a[feats].sum() * 100, "lgb_%": b / b.sum() * 100}).round(2) \
        .sort_values("xgb_%", ascending=False).to_csv(XAI_DIR / f"shap_stability_features_{mode}.csv")
    res.to_csv(XAI_DIR / f"shap_stability_{mode}.csv", index=False)
    print(res.round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    for md in ("planning", "operational"):
        explain_mode(md)
        stability(md)
