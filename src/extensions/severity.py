"""Phase 5a: delay SEVERITY (multi-class) - on_time (<1 min) / minor (1-5) / moderate (5-10) / major (>10 min late).

Why: a yes/no ">5 min" flag hides the difference between 6 and 25 minutes late; dispatchers act very differently on
those.  Classes come from ``SEVERITY_BINS`` in config (already in the feature files as ``severity``).
Model: XGBoost ``multi:softprob`` with the P3 planning / operational tree parameters.
Baseline: map the stop's usual delay at this hour (``hist_delay_stop_hour``) to a class.
Metrics: macro-F1 (classes are imbalanced), quadratic-weighted kappa (classes are ordered), per-class recall
(especially *major*), normalised confusion matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score

from src.config import SEVERITY_BINS, SEVERITY_LABELS
from src.extensions.common import EXT_DIR, MODEL_DIR, load_sets, log, xgb_params

CLASSES = SEVERITY_LABELS


def _codes(s: pd.Series) -> np.ndarray:
    return pd.Categorical(s.astype(str), categories=CLASSES).codes


def baseline(df: pd.DataFrame) -> np.ndarray:
    sec = df.hist_delay_stop_hour.fillna(df.hist_delay_stop_hour.median()) * 60
    return pd.cut(sec, SEVERITY_BINS, labels=False, right=False).astype(int).values


def run(mode: str) -> dict:
    feats, tr, va, te = load_sets(mode)
    ytr, yva, yte = _codes(tr.severity), _codes(va.severity), _codes(te.severity)
    p = {**xgb_params(mode), "objective": "multi:softprob", "num_class": 4, "eval_metric": "mlogloss"}
    m = xgb.XGBClassifier(**p).fit(tr[feats], ytr, eval_set=[(va[feats], yva)], verbose=False)
    m.save_model(MODEL_DIR / f"SEV_{mode}.json")
    proba = m.predict_proba(te[feats]); yhat = proba.argmax(1)
    yb = baseline(te)
    rows = []
    for name, pred in (("B0 hist-delay class", yb), ("XGB multi-class", yhat)):
        rep = classification_report(yte, pred, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0)
        rows.append({"mode": mode, "model": name, "accuracy": accuracy_score(yte, pred),
                     "macro_f1": f1_score(yte, pred, average="macro"),
                     "qw_kappa": cohen_kappa_score(yte, pred, weights="quadratic"),
                     **{f"recall_{c}": rep[c]["recall"] for c in CLASSES},
                     **{f"precision_{c}": rep[c]["precision"] for c in CLASSES}})
    res = pd.DataFrame(rows)
    cm = pd.DataFrame(confusion_matrix(yte, yhat, labels=range(4), normalize="true"), index=CLASSES, columns=CLASSES)
    cm.to_csv(EXT_DIR / f"severity_confusion_{mode}.csv")
    # consistency with the binary task: P(moderate)+P(major) is a ">5 min late" score
    from sklearn.metrics import roc_auc_score
    res["auc_gt5_from_multiclass"] = [np.nan, roc_auc_score(te.delayed, proba[:, 2:].sum(1))]
    res["trees"] = [np.nan, m.best_iteration + 1]
    log(f"severity {mode}:\n{res.round(3).T}")
    return {"metrics": res, "confusion": cm}


if __name__ == "__main__":
    out = pd.concat([run(md)["metrics"] for md in ("planning", "operational")])
    out.to_csv(EXT_DIR / "severity_metrics.csv", index=False)
