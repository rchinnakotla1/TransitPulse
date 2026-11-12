"""Phase 4 tests: narrator templates, faithfulness check, OpenAI error classification and graceful fallback."""
import pandas as pd
import pytest

from src.xai import llm_client, narrator


def _facts():
    row = pd.Series({"route_id": "28", "direction_id": "Inbound", "stop_id": "1234", "hour": 17,
                     "service_date": "2025-11-12", "hist_stop_hour": 0.45, "pm_peak": 1, "hour_sin": -0.96,
                     "hour_cos": -0.26, "avg_load": 30.0})
    shap = pd.Series({"hist_stop_hour": 0.8, "pm_peak": 0.3, "hour_sin": 0.1, "hour_cos": 0.05, "avg_load": -0.2})
    return narrator.facts_from_shap(row, shap, base_value=-1.0, threshold=0.27, mode="planning")


def test_facts_structure():
    f = _facts()
    assert f["route"] == "28" and f["time"] == "17:00"
    assert f["factors"][0]["factor"] == "stop's usual late rate at this hour"
    assert f["factors"][0]["value"] == "45%" and f["factors"][0]["direction"] == "raises"
    # hour_sin + hour_cos merged into a single factor
    assert sum(x["factor"] == "time of day" for x in f["factors"]) == 1
    assert 0 <= f["probability_pct"] <= 100 and f["risk_level"] in ("HIGH", "ELEVATED", "LOW")


@pytest.mark.parametrize("aud", narrator.AUDIENCES)
def test_template_is_faithful(aud):
    f = _facts()
    t = narrator.template_explanation(f, aud)
    assert "28" in t and str(f["probability_pct"]) in t
    assert narrator.faithfulness(t, f)["ok"]


def test_faithfulness_catches_invented_number():
    f = _facts()
    bad = f"Route 28 has a {f['probability_pct']}% risk because of the stop's usual late rate, plus 3 accidents and 37 mm rain."
    chk = narrator.faithfulness(bad, f)
    assert not chk["ok"] and "37" in chk["invented_numbers"]


def test_faithfulness_requires_top_factor():
    f = _facts()
    chk = narrator.faithfulness(f"Route 28 is {f['probability_pct']}% likely late.", f)
    assert not chk["top_factor_mentioned"]


def test_faithfulness_allows_rounded_fact_but_not_other_numbers():
    f = _facts()
    f["factors"].append({"factor": "minutes into the trip", "value": "40.4 min", "effect_pts": -6, "direction": "lowers"})
    base = f"Route 28: the stop's usual late rate at this hour raises risk to {f['probability_pct']}%;"
    assert narrator.faithfulness(base + " the bus is 40 minutes into the trip.", f)["ok"]
    assert not narrator.faithfulness(base + " the bus is 43 minutes into the trip.", f)["ok"]


class _Err(Exception):
    def __init__(self, status=None, body="", name=None):
        super().__init__(body)
        self.status_code, self.body = status, body


def _named(name, **kw):
    return type(name, (_Err,), {})(**kw)


@pytest.mark.parametrize("exc,code", [
    (_Err(401, "Incorrect API key"), "invalid_or_expired_key"),
    (_named("AuthenticationError"), "invalid_or_expired_key"),
    (_Err(429, "insufficient_quota"), "quota_exceeded"),
    (_Err(429, "Rate limit reached"), "rate_limited"),
    (_Err(404, "model not found"), "model_not_found"),
    (_named("APIConnectionError"), "network"),
    (_named("APITimeoutError"), "network"),
    (ValueError("weird"), "error"),
])
def test_classify_error(exc, code):
    assert llm_client.classify_error(exc) == code


def test_missing_key_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    st = llm_client.health_check(force=True)
    assert not st.ok and st.code == "missing_key"
    r = narrator.llm_explanation(_facts(), "dispatcher")
    assert r["source"] == "template" and r["status"] == "missing_key" and r["warning"]
    llm_client._STATUS = None   # do not leak the fake status into other tests


def test_unfaithful_llm_output_rejected(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: ("Route 28 will be 99 minutes late due to a parade.", "ok"))
    r = narrator.llm_explanation(_facts(), "rider")
    assert r["source"] == "template" and r["status"] == "unfaithful"


def test_failed_call_marks_status(monkeypatch):
    def boom(*a, **k):
        llm_client.mark_failed("invalid_or_expired_key")
        return None, "invalid_or_expired_key"
    monkeypatch.setattr(llm_client, "chat", boom)
    r = narrator.llm_explanation(_facts(), "planner")
    assert r["source"] == "template" and "expired" in r["warning"]
    assert llm_client.health_check().code == "invalid_or_expired_key"
    llm_client._STATUS = None
