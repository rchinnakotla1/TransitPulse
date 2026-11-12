"""Shared helpers for the TransitPulse Streamlit dashboard: cached loaders, model access, sidebar status badges."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd
import streamlit as st

from src.config import REPORTS

APP = REPORTS / "app"
XAI = REPORTS / "xai"
EXT = REPORTS / "ext"
FIG = REPORTS / "figures"

PRESETS = {  # threshold presets agreed in Phase 3 (docs/05)
    "Rider alert (planning ≥ 0.60, precision ≈ 0.70)": ("planning", 0.60),
    "Planner watch-list (planning ≥ 0.30, recall ≈ 0.61)": ("planning", 0.30),
    "Best F1 (planning, frozen on validation)": ("planning", None),
    "Dispatcher alert (operational ≥ 0.50, precision ≈ 0.93)": ("operational", 0.50),
    "Best F1 (operational, frozen on validation)": ("operational", None),
}


def page(title: str, icon: str = "🚌"):
    st.set_page_config(page_title=f"TransitPulse · {title}", page_icon=icon, layout="wide")
    sidebar_status()
    st.title(f"{icon} {title}")


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def read_csv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def exists(p: Path) -> bool:
    return Path(p).exists()


@st.cache_resource(show_spinner=False)
def model(mode: str, kind: str = "XGB"):
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(ROOT / "models" / f"{kind}_{mode}.json")
    return m


@st.cache_data(show_spinner=False)
def metrics() -> pd.DataFrame:
    return pd.concat([pd.read_csv(REPORTS / f"metrics_{m}.csv") for m in ("planning", "operational")])


def best_threshold(mode: str) -> float:
    m = metrics()
    return float(m[(m["mode"] == mode) & (m.model == "XGB") & (m.split == "test")].threshold.iloc[0])


@st.cache_data(ttl=600, show_spinner=False)
def _llm_status() -> dict:
    from src.xai import llm_client
    return llm_client.health_check().to_dict()


def llm_status(force: bool = False) -> dict:
    if force:
        _llm_status.clear()
        from src.xai import llm_client
        llm_client.health_check(force=True)
    return _llm_status()


def sidebar_status():
    with st.sidebar:
        st.markdown("### System status")
        s = llm_status()
        if s["ok"]:
            st.success(f"🟢 OpenAI connected ({s['model']})")
        else:
            st.error(f"🔴 OpenAI: {s['code'].replace('_', ' ')}")
            st.caption(s["message"] + " Details in `logs/llm.log`.")
        if st.button("Re-check OpenAI key", width="stretch"):
            llm_status(force=True); st.rerun()
        import os
        st.caption("MBTA live API: " + ("key configured" if os.getenv("MBTA_API_KEY") else "no key (20 req/min limit)"))
        st.caption("Models: XGBoost (deployed) · test period Nov–Dec 2025")


def llm_warning_banner(result: dict | None = None):
    """Prominent warning when the LLM is unavailable (expired / invalid key, quota, network...)."""
    s = llm_status()
    if result and result.get("warning"):
        st.warning(f"⚠️ {result['warning']}")
    elif not s["ok"]:
        st.warning(f"⚠️ {s['message']}")


def explain_row(mode: str, row: pd.DataFrame):
    """SHAP values for one prepared row -> (Series of contributions by feature, base value, probability)."""
    import numpy as np
    from src.encoding import feature_list
    from src.xai.shap_analysis import contributions
    feats = feature_list(mode)
    m = model(mode)
    sv, base = contributions(m, row[feats])
    s = pd.Series(sv[0], index=feats)
    p = float(1 / (1 + np.exp(-(base + s.sum()))))
    return s, base, p


def risk_badge(p: float, thr: float) -> str:
    if p >= max(thr, 0.5):
        return f"🔴 HIGH risk · {p:.0%}"
    if p >= thr:
        return f"🟠 ELEVATED risk · {p:.0%}"
    return f"🟢 LOW risk · {p:.0%}"


def shap_bar(s: pd.Series, k: int = 10):
    import plotly.graph_objects as go
    from src.xai.feature_text import label
    merged = s.groupby(s.index.map(label)).sum()
    top = merged.reindex(merged.abs().sort_values(ascending=False).index[:k])[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                           marker_color=["#d62728" if v > 0 else "#1f77b4" for v in top.values]))
    fig.update_layout(height=40 * len(top) + 80, margin=dict(l=10, r=10, t=30, b=10),
                      title="What pushed this prediction (red = towards late, blue = towards on time; log-odds)")
    return fig


def route_trust(mode: str, route: str) -> dict | None:
    p = APP / "route_auc.csv"
    if not p.exists():
        return None
    r = read_csv(str(p), dtype={"route_id": str})
    r = r[(r["mode"] == mode) & (r.route_id == str(route))]
    return None if r.empty else r.iloc[0].to_dict()


def json_load(p: Path):
    return json.loads(Path(p).read_text())
