"""Unit tests for the Phase 4 rule text and what-if helpers (no trained model needed: a tiny dummy model is used)."""
import numpy as np
import pandas as pd

from src.xai import counterfactual as cf
from src.xai.rules import rule_sentence


def test_rule_sentence_merges_repeated_conditions():
    s = rule_sentence("hist_stop_hour > 0.32 AND recent_route_28d > 0.30 AND hist_stop_hour > 0.42 AND hist_stop_hour <= 0.65")
    assert "between 42% and 65%" in s and s.count("usual late rate at this hour") == 1


def test_rule_sentence_flags_and_weekend():
    s = rule_sentence("pm_peak > 0.5 AND dow > 4.5")
    assert "yes" in s and "Sat/Sun" in s


class _Dummy:
    """P(late) rises with schedule_tightness and with prev_tp_delay."""
    def predict_proba(self, X):
        z = X.get("schedule_tightness", 0) * -1.0 + X.get("prev_tp_delay", 0) * 0.8 + 1.0
        p = 1 / (1 + np.exp(-np.asarray(z, dtype=float)))
        return np.column_stack([1 - p, p])


def _rows(monkeypatch, feats):
    monkeypatch.setattr(cf, "feature_list", lambda mode: feats)


def test_schedule_fix_population(monkeypatch):
    feats = ["seg_sched_min", "segment_km", "schedule_tightness"]
    _rows(monkeypatch, feats)
    rows = pd.DataFrame({"seg_sched_min": [1.0, 1.0], "segment_km": [1.0, 1.0], "schedule_tightness": [1.0, 1.0]})
    out = cf.schedule_fix_population(_Dummy(), rows, "planning", threshold=0.4)
    # more scheduled time -> looser schedule -> lower probability; must be monotone
    probs = out[[c for c in out if c.startswith("p_+")]].values[0]
    assert np.all(np.diff(probs) <= 1e-9)
    assert out.min_add.notna().all()


def test_delay_tipping_points(monkeypatch):
    feats = ["prev_tp_delay", "prev_tp_delayed"]
    _rows(monkeypatch, feats)
    rows = pd.DataFrame({"prev_tp_delay": [0.0], "prev_tp_delayed": [0]})
    C, tip = cf.delay_tipping_points(_Dummy(), rows, "operational", threshold=0.9)
    # sigmoid(0.8 d + 1) >= 0.9  <=>  d >= (ln 9 - 1) / 0.8 = 1.50
    assert tip.iloc[0] == 1.5
    assert C.shape == (1, len(np.arange(-2, 15.5, 0.5)))
