"""TransitPulse - home page.  Run:  .\\.venv\\Scripts\\streamlit run app/Home.py"""
from common import APP, metrics, page, read_csv
import streamlit as st

page("TransitPulse — MBTA bus delay prediction & explanation", "🚌")

st.markdown("""
**Who is this for?** Transit-agency staff:
* **Dispatchers**: which buses will be more than 5 minutes late at their next timepoint *right now* (operational model, live mode)
* **Service planners**: where and when lateness is predictable and recurring; which segments need more running time
* **Rider communications**: day-ahead alerts with a plain-language reason

**What it does:** predicts whether a bus will be **more than 5 minutes late** at a timepoint (MBTA's reliability standard), explains *why*
with SHAP, and turns the explanation into plain language with OpenAI (with a fact check and a template fallback).
""")

m = metrics()
t = m[(m.split == "test") & (m.model == "XGB")].set_index("mode")
c = st.columns(4)
c[0].metric("Stop events analysed (2025)", "9.8 M")
c[1].metric("Planning model (day-ahead) ROC-AUC", f"{t.loc['planning', 'roc_auc']:.3f}",
            f"+{t.loc['planning', 'roc_auc'] - m[(m.split == 'test') & (m.model == 'B0') & (m['mode'] == 'planning')].roc_auc.iloc[0]:.3f} vs agency average")
c[2].metric("Operational model (live) ROC-AUC", f"{t.loc['operational', 'roc_auc']:.3f}")
c[3].metric("Routes covered", "19 (14 key + 5 more: 34E, 70, 240, 8, 44)")

st.markdown("""
### Pages
| Page | Use it to |
|---|---|
| **Reliability Explorer** | see late rates by route, hour, month and on a map |
| **Model Comparison** | compare the six models (B0 baseline, LR, DT, RF, XGBoost, LightGBM) with confidence intervals |
| **Predict & Explain** | pick a real December 2025 trip, get the risk, the reasons, an AI explanation and what-if scenarios |
| **Explainability** | global drivers of lateness, rules, why evening peaks are late |
| **Propagation & Bunching** | how delay carries along a trip and how buses bunch |
| **Planner Insights** | chronic segments, suggested running-time changes, severity and delay-minutes models |
| **Live Mode** | score buses running now from the MBTA V3 API |
| **Data Sources** | every dataset used, why, and how it was joined |

> Probabilities are calibrated on the test months (planning: 26.1 % predicted vs 25.1 % observed), so "60 %" means roughly 6 in 10 such stop
> events are more than 5 minutes late. Model-based what-ifs are associations, not causal effects.
""")
if (APP / "route_auc.csv").exists():
    with st.expander("How much can I trust the model on each route? (test ROC-AUC by route)"):
        st.dataframe(read_csv(str(APP / "route_auc.csv"), dtype={"route_id": str}).pivot(
            index="route_id", columns="mode", values="auc").round(3), width="stretch")
