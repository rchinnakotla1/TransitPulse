"""Checks on the produced feature tables (reads the real parquet files; skipped if not built yet)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR, REPORTS, TRAIN_END, VALID_END

FEAT = PROCESSED_DIR / "features"
pytestmark = pytest.mark.skipif(not (REPORTS / "feature_spec.json").exists(), reason="features not built")


@pytest.fixture(scope="module")
def spec():
    return json.loads((REPORTS / "feature_spec.json").read_text())


@pytest.fixture(scope="module")
def test_df(spec):
    return pd.read_parquet(FEAT / "test.parquet").sample(200_000, random_state=0)


def test_no_forbidden_features(spec):
    for mode in ("planning", "operational"):
        assert not set(spec[mode]) & set(spec["forbidden"]), mode


def test_planning_excludes_operational(spec):
    assert not set(spec["planning"]) & set(spec["groups"]["operational"])


def test_splits_are_time_ordered_and_disjoint():
    d = {s: pd.read_parquet(FEAT / f"{s}.parquet", columns=["service_date"]).service_date
         for s in ("train", "valid", "test")}
    assert d["train"].max() <= pd.Timestamp(TRAIN_END) < d["valid"].min()
    assert d["valid"].max() <= pd.Timestamp(VALID_END) < d["test"].min()
    drift = pd.read_parquet(FEAT / "drift_2024.parquet", columns=["service_date"]).service_date
    assert drift.dt.year.eq(2024).all()


def test_row_counts_match_clean_data():
    n25 = sum(len(pd.read_parquet(FEAT / f"{s}.parquet", columns=["delayed"])) for s in ("train", "valid", "test"))
    clean = pd.read_parquet(PROCESSED_DIR / "bus_2025_clean.parquet", columns=["delayed"])
    assert n25 == len(clean)


def test_value_ranges(test_df):
    for c in [c for c in test_df.columns if c.startswith(("hist_route", "hist_stop_period", "recent_"))] + ["hist_stop_hour"]:
        assert test_df[c].between(0, 1).all(), c
        assert test_df[c].notna().all(), c
    assert test_df.hour_sin.between(-1, 1).all() and test_df.hour_cos.between(-1, 1).all()
    assert test_df.stop_pos.dropna().between(0, 1).all()
    assert set(test_df.prev_tp_delayed.dropna().unique()) <= {0.0, 1.0}


def test_prev_tp_missing_only_at_trip_start(test_df):
    """prev_tp_delay is missing for the first timepoint of a trip (and for the few rows without trip id)."""
    miss = test_df.prev_tp_delay.isna()
    assert miss.mean() < 0.20
    assert (test_df.loc[miss, "time_point_order"] <= 2).mean() > 0.9


def test_history_is_informative_but_not_perfect(test_df):
    """Sanity: a leaky rate would be suspiciously predictive (AUC ~ 1)."""
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test_df.delayed, test_df.hist_stop_hour)
    assert 0.55 < auc < 0.85
