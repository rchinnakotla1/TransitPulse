"""Phase 3: research experiments run with the tuned XGBoost configuration.

All experiments use the *same* hyper-parameters (the Optuna winner from ``train.py``, read from
``reports/metrics_planning.csv``) and the same training-sample size, so that the only thing that changes is the
factor being studied.

| key        | question                                                                                   | output                       |
|------------|--------------------------------------------------------------------------------------------|------------------------------|
| ablation   | How much does each external source add on top of historical averages? (cumulative + drop-one) | reports/exp_ablation.csv     |
| learning   | Is 8 months of training data enough, or are we under-training? (5 → 100 % of train)        | reports/exp_learning_curve.csv |
| walkfwd    | Does performance hold month by month when we always train on everything before it?         | reports/exp_walk_forward.csv |
| window     | Does a 12-month window (Sep 2024 – Aug 2025) beat Jan–Aug 2025 on the autumn test months?  | reports/exp_window.csv       |
| drift      | How fast does a model trained on 2024 decay over 2025?                                     | reports/exp_drift.csv        |
| episodic   | Do weather / events help *on the rows they affect* (event nights, rain, snow)?              | reports/exp_episodic.csv     |

Run: ``python -m src.models.experiments [ablation learning walkfwd window drift episodic]``
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import REPORTS
from src.encoding import align_categories, feature_list, load_split, spec

SEED = 42
SAMPLE = 2_000_000
EXTRA = ["service_date", "month", "delayed"]
ABLATION_ORDER = ["history", "temporal+calendar", "daylight", "weather", "events", "gtfs", "ridership", "spatial"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def xgb_params() -> dict:
    """Tuned planning-mode XGBoost params (fallback = sensible defaults if train.py has not run yet)."""
    p = {"max_depth": 8, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
         "max_cat_threshold": 64, "n_estimators_refit": 400}
    path = REPORTS / "metrics_planning.csv"
    if path.exists():
        m = pd.read_csv(path)
        row = m[(m.model == "XGB") & (m.split == "valid")]
        if len(row):
            p = json.loads(row.params.iloc[0])
    n = int(p.pop("n_estimators_refit") / 1.15)
    return dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, random_state=SEED,
                learning_rate=0.1, n_estimators=n, **p)


def sample(df, n=SAMPLE, seed=SEED):
    return df if len(df) <= n else df.groupby("delayed", group_keys=False).sample(frac=n / len(df), random_state=seed)


def fit_eval(train: pd.DataFrame, feats: list[str], tests: dict[str, pd.DataFrame], **extra) -> list[dict]:
    t = time.time()
    m = xgb.XGBClassifier(**xgb_params()).fit(train[feats], train.delayed)
    out = []
    for name, te in tests.items():
        p = m.predict_proba(te[feats])[:, 1]
        out.append({**extra, "eval_on": name, "n_train": len(train), "n_eval": len(te), "n_features": len(feats),
                    "roc_auc": roc_auc_score(te.delayed, p), "pr_auc": average_precision_score(te.delayed, p),
                    "base_rate": te.delayed.mean(), "mean_prob": p.mean(), "seconds": round(time.time() - t)})
    log("  ", extra, {o["eval_on"]: round(o["roc_auc"], 4) for o in out})
    return out


_CACHE: dict = {}


def data():
    """All 2025 rows (train/valid/test) + 2024, with operational features, categories aligned across all of them."""
    if not _CACHE:
        cols = list(dict.fromkeys(EXTRA + feature_list("operational")))
        frames = {s: load_split(s, columns=cols) for s in ("train", "valid", "test", "drift_2024")}
        align_categories(list(frames.values()), spec()["categorical"])
        _CACHE.update(frames)
    return _CACHE


def groups_of(names: list[str]) -> list[str]:
    return [g for n in names for g in n.split("+")]


# ------------------------------------------------------------------------------------------------
def ablation():
    d = data(); tr = sample(d["train"]); tests = {"valid": d["valid"], "test": d["test"]}
    rows = []
    # cumulative: start from what agencies already have (historical averages) and add one source at a time
    for i in range(len(ABLATION_ORDER)):
        names = ABLATION_ORDER[: i + 1]
        rows += fit_eval(tr, feature_list("planning", groups_of(names)), tests, kind="cumulative",
                         step=i, added=ABLATION_ORDER[i])
    rows += fit_eval(tr, feature_list("operational"), tests, kind="cumulative", step=len(ABLATION_ORDER),
                     added="operational")
    # drop-one from the full planning model (order-independent view of each group's unique contribution)
    for g in ABLATION_ORDER:
        keep = [x for x in ABLATION_ORDER if x != g]
        rows += fit_eval(tr, feature_list("planning", groups_of(keep)), tests, kind="drop_one", step=-1, added=g)
    pd.DataFrame(rows).to_csv(REPORTS / "exp_ablation.csv", index=False)


def learning():
    d = data(); feats = feature_list("planning"); rows = []
    for frac in (0.05, 0.10, 0.25, 0.50, 1.00):
        tr = d["train"] if frac == 1 else d["train"].groupby("delayed", group_keys=False).sample(frac=frac,
                                                                                              random_state=SEED)
        tr_eval = sample(tr, 600_000, seed=1)
        rows += fit_eval(tr, feats, {"train_in_sample": tr_eval, "valid": d["valid"]}, frac=frac)
    pd.DataFrame(rows).to_csv(REPORTS / "exp_learning_curve.csv", index=False)


def walkfwd():
    d = data(); feats = feature_list("planning"); rows = []
    allr = pd.concat([d["train"], d["valid"], d["test"]], ignore_index=True)
    for m in range(5, 13):
        tr = sample(allr[allr.month < m]); te = allr[allr.month == m]
        rows += fit_eval(tr, feats, {f"2025-{m:02d}": te}, test_month=m, train_months=f"1-{m - 1}")
    pd.DataFrame(rows).to_csv(REPORTS / "exp_walk_forward.csv", index=False)


def window():
    d = data(); feats = feature_list("planning"); rows = []
    tests = {"valid_SepOct": d["valid"], "test_NovDec": d["test"]}
    autumn24 = d["drift_2024"][d["drift_2024"].month >= 9]
    win12 = pd.concat([autumn24, d["train"]], ignore_index=True)
    rows += fit_eval(sample(d["train"]), feats, tests, window="A: Jan-Aug 2025 (8 mo)")
    rows += fit_eval(sample(win12), feats, tests, window="B: Sep 2024-Aug 2025 (12 mo)")
    # B keeps the same sample size as A (fair); B-large lets the longer window contribute more rows
    rows += fit_eval(sample(win12, 3_000_000), feats, tests, window="B-large: 12 mo, 3M sample")
    pd.DataFrame(rows).to_csv(REPORTS / "exp_window.csv", index=False)


def drift():
    d = data(); feats = feature_list("planning"); rows = []
    allr = pd.concat([d["train"], d["valid"], d["test"]], ignore_index=True)
    tests = {f"2025-{m:02d}": allr[allr.month == m] for m in range(1, 13)}
    rows += fit_eval(sample(d["drift_2024"]), feats, tests, trained_on="2024 full year")
    pd.DataFrame(rows).to_csv(REPORTS / "exp_drift.csv", index=False)


def episodic():
    """Weather / events are rare: an average-AUC ablation can hide a real effect on the few days they matter.
    Compare the full planning model with drop-weather / drop-events models *on the affected rows only*."""
    d = data(); tr = sample(d["train"])
    ev = pd.concat([d["valid"], d["test"]], ignore_index=True)
    subsets = {"all valid+test": ev,
               "venue event nearby": ev[ev.venue_event_nearby == 1],
               "citywide event": ev[ev.citywide_event == 1],
               "rain ≥ 2 mm in 3 h": ev[ev.precip_3h >= 2],
               "snow in last 24 h": ev[ev.snow_24h > 0]}
    subsets = {k: v for k, v in subsets.items() if len(v) > 2000 and v.delayed.nunique() == 2}
    rows = []
    for label, drop in (("full planning", None), ("without weather", "weather"), ("without events", "events")):
        keep = [x for x in ABLATION_ORDER if x != drop]
        rows += fit_eval(tr, feature_list("planning", groups_of(keep)), subsets, model=label)
    pd.DataFrame(rows).to_csv(REPORTS / "exp_episodic.csv", index=False)


EXPERIMENTS = {"ablation": ablation, "learning": learning, "walkfwd": walkfwd, "window": window, "drift": drift,
               "episodic": episodic}

if __name__ == "__main__":
    log("xgb params:", xgb_params())
    for name in [a for a in sys.argv[1:] if a in EXPERIMENTS] or list(EXPERIMENTS):
        log(f"== {name}")
        EXPERIMENTS[name]()
    log("done")
