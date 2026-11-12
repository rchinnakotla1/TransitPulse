"""Phase 5 extensions: shared helpers.

Protocol for all extension models (simpler than P3 on purpose, logged in decision_log D-P5-1):
fit on a stratified 2 M sample of TRAIN (Jan-Aug), early-stop on a 300 k VALID sample (Sep-Oct), score TEST (Nov-Dec)
once.  Hyper-parameters are re-used from the P3 Optuna search for the same mode (no new tuning), so test numbers are
slightly conservative (no Jan-Oct refit) but fully out-of-time.
"""
from __future__ import annotations

import time

import pandas as pd

from src.config import REPORTS, ROOT
from src.encoding import align_categories, feature_list, load_split, spec

EXT_DIR = REPORTS / "ext"; EXT_DIR.mkdir(exist_ok=True)
MODEL_DIR = ROOT / "models"
ID = ["service_date", "route_id", "direction_id", "stop_id", "hour", "half_trip_id", "time_point_order",
      "delayed", "delay_min", "severity"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def xgb_params(mode: str) -> dict:
    """Best P3 Optuna parameters for this mode (tree structure only)."""
    d = pd.read_csv(REPORTS / f"optuna_xgb_{mode}.csv").sort_values("value").iloc[-1]
    p = {k.replace("params_", ""): d[k] for k in d.index if k.startswith("params_")}
    for k in ("max_depth", "max_cat_threshold"):
        p[k] = int(p[k])
    return {**p, "tree_method": "hist", "enable_categorical": True, "max_cat_to_onehot": 1, "n_jobs": -1,
            "random_state": 42, "learning_rate": 0.1, "n_estimators": 1000, "early_stopping_rounds": 30}


def load_sets(mode: str, extra: list[str] | None = None, n_train: int = 2_000_000, n_valid: int = 300_000,
              n_test: int | None = None):
    feats = feature_list(mode)
    cols = list(dict.fromkeys(ID + feats + (extra or [])))
    tr = load_split("train", cols, sample=n_train, seed=42)
    va = load_split("valid", cols, sample=n_valid, seed=42)
    te = load_split("test", cols, sample=n_test, seed=42)
    align_categories([tr, va, te], spec()["categorical"])
    return feats, tr, va, te
