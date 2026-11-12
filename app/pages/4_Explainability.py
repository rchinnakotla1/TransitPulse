"""Page 4: global explainability: what drives delay risk, readable rules and how faithful they are, stability."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from common import FIG, XAI, exists, page, read_csv

page("Why are buses late? (Explainability)", "🔍")
st.caption("TreeSHAP on 50,000 held-out test events (Nov–Dec 2025) per model. SHAP explains the *model*, not causes in the world.")

if not exists(XAI / "shap_global_planning.csv"):
    st.error("XAI results not found. Run `python -m src.xai.shap_analysis` and `python -m src.xai.rules`.")
    st.stop()

mode = st.radio("Model", ["planning", "operational"], horizontal=True,
                help="planning = schedule-time forecast (no live data); operational = uses the live delay so far on the trip")

from src.xai.feature_text import label  # noqa: E402

tab1, tab2, tab3, tab4 = st.tabs(["Importance", "By time of day", "Readable rules", "Stability"])

with tab1:
    c1, c2 = st.columns([1.3, 1])
    g = read_csv(str(XAI / f"shap_global_{mode}.csv")).head(15).copy()
    g["label"] = g.feature.map(label)
    c1.plotly_chart(px.bar(g.iloc[::-1], x="share_%", y="label", color="group", orientation="h",
                           title="Top 15 features: share of total |SHAP| (%)", height=520), width="stretch")
    gr = read_csv(str(XAI / f"shap_groups_{mode}.csv"))
    c2.plotly_chart(px.pie(gr, names="group", values="share_%", title="Share by data source", hole=0.45),
                    width="stretch")
    if mode == "planning":
        st.info("**Reading tip:** temperature ranks high but acts mainly as a *season proxy* (colder = December holiday period = "
                "less traffic). The P3 ablation showed weather adds little accuracy, so do not read it as a cause.")
    else:
        st.info("**Reading tip:** live features (delay at the previous timepoint, time since it) explain ~61 % of the "
                "operational model: a bus that is already late tends to stay late.")
    if exists(FIG / f"p4_beeswarm_{mode}.png"):
        with st.expander("Direction of effects (beeswarm)"):
            st.image(str(FIG / f"p4_beeswarm_{mode}.png"))
    if exists(FIG / f"p4_dependence_{mode}.png"):
        with st.expander("Dependence plots"):
            st.image(str(FIG / f"p4_dependence_{mode}.png"))

with tab2:
    st.markdown("**Which factors make the evening peak late?** Mean signed SHAP by feature group and period "
                "(planning model; red = pushes towards late).")
    if exists(FIG / "p4_shap_by_period.png"):
        st.image(str(FIG / "p4_shap_by_period.png"))
    if exists(XAI / "route_period_risk.csv"):
        st.markdown("**Riskiest route × direction × period** (test predictions, at least 2,000 stop events)")
        r = read_csv(str(XAI / "route_period_risk.csv"), dtype={"route_id": str})
        st.dataframe(r.head(25).round(3), width="stretch", hide_index=True)

with tab3:
    fid = read_csv(str(XAI / "surrogate_fidelity.csv"))
    f = fid[fid["mode"] == mode].iloc[0]
    c = st.columns(4)
    c[0].metric("Leaves", int(f.leaves)); c[1].metric("R² vs XGBoost", f"{f.r2_vs_xgb:.2f}")
    c[2].metric("AUC rules", f"{f.auc_surrogate:.3f}"); c[3].metric("AUC XGBoost", f"{f.auc_xgb:.3f}")
    if f.r2_vs_xgb < 0.6:
        st.warning("Low fidelity: planning risk is spread over many small effects, so these rules are only a **coarse summary**. "
                   "Use the per-trip SHAP explanation (Predict & Explain page) for individual decisions.")
    else:
        st.success("High fidelity: these rules describe the live model well.")
    from src.xai.rules import rule_sentence  # noqa: E402
    L = read_csv(str(XAI / f"rule_leaves_{mode}.csv"))
    L["rule (plain words)"] = L.rule.map(rule_sentence)
    st.dataframe(L[["surrogate_p", "actual_late", "share_%", "rule (plain words)"]].round(3), width="stretch",
                 hide_index=True, column_config={"surrogate_p": "rule's risk", "actual_late": "actual late rate",
                                                 "share_%": "% of events"})
    if exists(XAI / f"surrogate_depth_{mode}.csv"):
        d = read_csv(str(XAI / f"surrogate_depth_{mode}.csv"))
        st.plotly_chart(px.line(d, x="leaves", y=["r2_vs_xgb", "auc_surrogate"], markers=True, log_x=True,
                                title="Readability vs fidelity: more leaves = more faithful but harder to read"),
                        width="stretch")

with tab4:
    st.markdown("Do two equally accurate models (XGBoost vs LightGBM) agree on *why*?")
    s = pd.concat([read_csv(str(XAI / f"shap_stability_{m}.csv")) for m in ("planning", "operational")])
    st.dataframe(s.round(3), hide_index=True, width="stretch")
    p = XAI / f"shap_stability_groups_{mode}.csv"
    if exists(p):
        g = read_csv(str(p), index_col=0).reset_index(names="group").melt("group", var_name="model", value_name="share %")
        st.plotly_chart(px.bar(g, x="group", y="share %", color="model", barmode="group",
                               title=f"{mode}: share of explanation by group"), width="stretch")
    st.caption("Operational features carrying the same information (delay at previous timepoint, minutes since it, the 0/1 late flag) "
               "swap credit between models; at group level the agreement is strong, so explanations are shown by group.")
