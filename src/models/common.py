"""Phase 3: shared evaluation utilities (metrics, threshold choice, day-block bootstrap, McNemar)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, confusion_matrix)


def best_f1_threshold(y, p) -> float:
    """Threshold maximising F1 (chosen on VALIDATION only, then frozen for test)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


def metrics(y, p, thr: float) -> dict:
    y = np.asarray(y); p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
        "threshold": thr, "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "accuracy": (tp + tn) / len(y),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "pred_rate": yhat.mean(), "mean_prob": p.mean(), "base_rate": y.mean(),
    }


def day_block_bootstrap(y, p, days, stat=roc_auc_score, n: int = 200, seed: int = 0, max_rows: int = 300_000):
    """95 % CI resampling whole service days (stop events on the same day are correlated, so a row bootstrap is too optimistic)."""
    y, p, days = np.asarray(y), np.asarray(p), np.asarray(days)
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        idx = rng.choice(len(y), max_rows, replace=False)
        y, p, days = y[idx], p[idx], days[idx]
    ud, inv = np.unique(days, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(ud))]
    vals = []
    for _ in range(n):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(ud), len(ud))])
        vals.append(stat(y[pick], p[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, yhat_a, yhat_b) -> dict:
    """McNemar test (continuity-corrected) on the disagreements of two classifiers."""
    y, a, b = map(np.asarray, (y, yhat_a, yhat_b))
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "chi2": stat, "p_value": float(chi2.sf(stat, 1))}


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True)
