"""Phase 2: encodings for the two model families.

* Linear models (logistic regression): one-hot categoricals, median-impute + missing-indicator + standard-scale numerics.
* Tree models (DT / RF / XGBoost): native pandas categories, NaN left as is (XGBoost learns a default direction;
  for sklearn trees we impute with a sentinel, see ``tree_matrix``).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROCESSED_DIR, REPORTS

FEAT_DIR = PROCESSED_DIR / "features"


def spec() -> dict:
    return json.loads((REPORTS / "feature_spec.json").read_text())


def feature_list(mode: str = "planning", groups: list[str] | None = None) -> list[str]:
    """Features for a prediction mode, optionally restricted to some groups (ablation)."""
    s = spec()
    allowed = set(s[mode])
    if groups is None:
        return s[mode]
    return [f for g in groups for f in s["groups"][g] if f in allowed]


def load_split(name: str, columns: list[str] | None = None, sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load train / valid / test / drift_2024. ``sample`` = stratified row sample (for RF/DT memory limits)."""
    df = pd.read_parquet(FEAT_DIR / f"{name}.parquet", columns=columns)
    if sample and sample < len(df):
        frac = sample / len(df)
        df = df.groupby("delayed", group_keys=False).sample(frac=frac, random_state=seed)
    return df


def linear_preprocessor(features: list[str]) -> ColumnTransformer:
    cat = [f for f in features if f in spec()["categorical"]]
    num = [f for f in features if f not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True), cat),
    ], sparse_threshold=0.3)


def tree_matrix(df: pd.DataFrame, features: list[str], sentinel: float = -999.0) -> pd.DataFrame:
    """sklearn trees: categories -> integer codes, NaN -> sentinel. (XGBoost uses the raw frame with enable_categorical.)"""
    X = df[features].copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes.astype("int16")
    return X.astype("float32").fillna(sentinel)


def align_categories(frames: list[pd.DataFrame], cols: list[str]) -> None:
    """Give every split identical category sets (required by XGBoost and consistent integer codes)."""
    for c in cols:
        cats = sorted(set().union(*[set(f[c].cat.categories) for f in frames if c in f]))
        for f in frames:
            if c in f:
                f[c] = f[c].cat.set_categories(cats)


def category_levels(refresh: bool = False) -> dict:
    """The exact category sets used in training (union over train/valid/test, sorted), saved once to
    ``reports/category_levels.json`` so that the dashboard / live mode encode new rows identically."""
    path = REPORTS / "category_levels.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    cats = spec()["categorical"]
    levels = {c: set() for c in cats}
    for s in ("train", "valid", "test"):
        df = pd.read_parquet(FEAT_DIR / f"{s}.parquet", columns=cats)
        for c in cats:
            levels[c] |= {str(x) for x in df[c].cat.categories}
    out = {c: sorted(v) for c, v in levels.items()}
    path.write_text(json.dumps(out))
    return out


def apply_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns of any frame (e.g. one live trip) to the training category sets."""
    for c, lv in category_levels().items():
        if c in df:
            df[c] = pd.Categorical(df[c].astype(str), categories=lv)
    return df
