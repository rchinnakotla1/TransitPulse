"""Phase 4: turn a prediction + its SHAP values into a plain-language explanation.

Pipeline (same for every stop event)
1. ``facts_from_shap``   - top contributing features -> structured *facts* (plain names, values, effect in
                           percentage points).  Everything numeric the user will read comes from here.
2. ``template_explanation`` - deterministic text from the facts (always available, used as fallback).
3. ``llm_explanation``   - OpenAI rewrites the *same facts* in a friendlier voice for a given audience.
                           A strict prompt forbids new facts; ``faithfulness`` then checks that every number in
                           the LLM text exists in the facts and that the main factor is mentioned.  If the key is
                           missing / expired / over quota, or the check fails, we return the template text and a
                           warning that the dashboard displays.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.xai import llm_client
from src.xai.feature_text import fmt_value, label

AUDIENCES = ("dispatcher", "planner", "rider")
SYSTEM_PROMPT = (
    "You explain bus delay predictions for the MBTA (Boston). You receive FACTS as JSON produced by a machine-learning "
    "model and its SHAP explanation. Rules: (1) Use ONLY the facts given; never invent causes, numbers, events, "
    "weather or advice that is not in the facts. (2) Every number you write must appear in the facts exactly. "
    "(3) Mention the most important factor first. (4) Say it is a prediction, not a certainty. "
    "(5) Keep it under 90 words, no markdown headings, no bullet lists."
)
STYLE = {
    "dispatcher": "Audience: a bus dispatcher deciding whether to act now. Be direct and operational.",
    "planner": "Audience: a service planner reviewing schedules. Focus on recurring, schedule-related factors.",
    "rider": "Audience: a passenger waiting at the stop. Be friendly and simple; avoid jargon like SHAP or log-odds.",
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def facts_from_shap(row: pd.Series, shap_row: pd.Series, base_value: float, threshold: float, mode: str,
                    k: int = 4) -> dict:
    """Build the fact sheet.  ``shap_row`` is indexed by feature name (log-odds contributions)."""
    logit = base_value + float(shap_row.sum())
    p = float(_sigmoid(logit))
    # merge features that share a plain-language name (hour_sin + hour_cos -> 'time of day')
    merged: dict[str, dict] = {}
    for f, s in shap_row.items():
        name = label(f)
        m = merged.setdefault(name, {"shap": 0.0, "features": []})
        m["shap"] += float(s)
        m["features"].append(f)
    top = sorted(merged.items(), key=lambda kv: -abs(kv[1]["shap"]))[:k]
    factors = []
    for name, m in top:
        f = m["features"][0]
        val = row.get("hour") if f in ("hour_sin", "hour_cos") else row.get(f)
        shown = f"{int(val):02d}:00" if f in ("hour_sin", "hour_cos") and val is not None else fmt_value(f, val)
        effect = (p - float(_sigmoid(logit - m["shap"]))) * 100      # pts this factor adds vs. leaving it out
        factors.append({"factor": name, "value": shown, "effect_pts": int(round(effect)),
                        "direction": "raises" if effect > 0 else "lowers"})
    level = "HIGH" if p >= max(threshold, 0.5) else ("ELEVATED" if p >= threshold else "LOW")
    hour = row.get("hour")
    return {
        "route": str(row.get("route_id", "?")), "direction": str(row.get("direction_id", "?")),
        "stop": str(row.get("stop_id", "?")), "time": f"{int(hour):02d}:00" if hour is not None else "?",
        "date": str(row.get("service_date", ""))[:10], "mode": mode,
        "late_threshold_min": 5, "probability_pct": int(round(p * 100)),
        "alert_threshold_pct": int(round(threshold * 100)), "risk_level": level,
        "typical_pct": int(round(_sigmoid(base_value) * 100)), "factors": factors,
    }


def template_explanation(facts: dict, audience: str = "dispatcher") -> str:
    f = facts
    reasons = "; ".join(f"{x['factor']} ({x['value']}) {x['direction']} the risk by {abs(x['effect_pts'])} pts"
                        for x in f["factors"])
    if audience == "rider":
        mood = {"HIGH": "is likely to be", "ELEVATED": "may be", "LOW": "should be close to on time and is unlikely to be"}
        return (f"Route {f['route']} ({f['direction']}) around {f['time']} {mood[f['risk_level']]} more than "
                f"{f['late_threshold_min']} minutes late (predicted chance {f['probability_pct']}%). "
                f"Main reason: {f['factors'][0]['factor']} ({f['factors'][0]['value']}). This is a prediction, not a guarantee.")
    head = (f"[{f['risk_level']}] Route {f['route']} {f['direction']}, stop {f['stop']}, {f['time']}: "
            f"{f['probability_pct']}% chance of >{f['late_threshold_min']} min late "
            f"(alert threshold {f['alert_threshold_pct']}%, typical {f['typical_pct']}%).")
    return f"{head} Why: {reasons}."


_NUM = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    out = set()
    for n in _NUM.findall(text):
        out.add(n.lstrip("0") or "0")
    return out


def faithfulness(text: str, facts: dict) -> dict:
    """Automatic check: (a) no number outside the facts, (b) the top factor is mentioned."""
    allowed = _numbers(json.dumps(facts))
    # Rounding a fact to a whole number is honest (40.4 min -> "40 minutes"), so the rounded/truncated integer is allowed too.
    for n in list(allowed):
        if "." in n:
            x = float(n)
            allowed |= {str(int(round(x))), str(int(x))}
    invented = sorted(n for n in _numbers(text) if n not in allowed)
    top = facts["factors"][0]["factor"].lower() if facts["factors"] else ""
    words = [w for w in re.findall(r"[a-z]+", top) if len(w) > 3]
    mentioned = (not words) or sum(w in text.lower() for w in words) >= max(1, len(words) // 2)
    return {"ok": not invented and mentioned, "invented_numbers": invented, "top_factor_mentioned": mentioned}


def llm_explanation(facts: dict, audience: str = "dispatcher", use_llm: bool = True) -> dict:
    """Returns {text, source ('llm'|'template'), status, warning, check}. Never raises."""
    tmpl = template_explanation(facts, audience)
    if not use_llm:
        return {"text": tmpl, "source": "template", "status": "disabled", "warning": None, "check": None}
    user = f"{STYLE.get(audience, STYLE['dispatcher'])}\nFACTS:\n{json.dumps(facts, ensure_ascii=False)}"
    text, status = llm_client.chat(SYSTEM_PROMPT, user)
    if text is None:
        return {"text": tmpl, "source": "template", "status": status,
                "warning": llm_client.MESSAGES.get(status, llm_client.MESSAGES["error"]), "check": None}
    chk = faithfulness(text, facts)
    if not chk["ok"]:
        llm_client.log.warning("faithfulness check failed: %s", chk)
        return {"text": tmpl, "source": "template", "status": "unfaithful",
                "warning": "The AI text did not pass the fact check, so the template explanation is shown.",
                "check": chk, "rejected_text": text}
    return {"text": text, "source": "llm", "status": "ok", "warning": None, "check": chk}
