from common import FIG, REPORTS, metrics, page, read_csv
import plotly.express as px
import streamlit as st

page("Model Comparison", "🏆")
st.caption("Protocol: tune on Jan–Aug → Sep–Oct, refit on Jan–Oct, score Nov–Dec 2025 once. 95 % CIs from a day-block bootstrap.")

m = metrics()
mode = st.radio("Prediction mode", ["planning", "operational"], horizontal=True,
                help="planning = day-ahead (no live data); operational = minutes ahead, knows the delay at the previous timepoint")
t = m[(m.split == "test") & (m["mode"] == mode)].copy()
order = ["B0", "LR", "DT", "RF", "XGB", "LGB"]
names = {"B0": "B0 agency average", "LR": "Logistic regression", "DT": "Decision tree", "RF": "Random forest",
         "XGB": "XGBoost (deployed)", "LGB": "LightGBM (cross-check)"}
t["model_name"] = t.model.map(names); t = t.set_index("model").reindex([o for o in order if o in set(t.model)]).reset_index()
fig = px.bar(t, x="roc_auc", y="model_name", orientation="h", error_x=t.roc_auc_hi - t.roc_auc,
             error_x_minus=t.roc_auc - t.roc_auc_lo, range_x=(0.5, 1), title=f"{mode}: test ROC-AUC (95 % CI)")
st.plotly_chart(fig, width="stretch")
st.dataframe(t[["model_name", "roc_auc", "pr_auc", "brier", "threshold", "precision", "recall", "f1", "mean_prob", "base_rate", "seconds"]]
             .round(4), width="stretch", hide_index=True)

st.markdown("""
**Reading the table**
* **ROC-AUC / PR-AUC** do not depend on a threshold; PR-AUC focuses on the *late* class.
* **Threshold** was chosen on validation (max F1) and frozen before the test.
* **mean_prob vs base_rate** = calibration: close values mean probabilities can be read as chances.
* XGBoost and LightGBM are statistically tied; XGBoost is deployed (best point estimate, tuned, explained).
""")
for name, cap in [("p3_roc_pr.png", "ROC and precision–recall curves"), ("p3_calibration.png", "Calibration"),
                  ("p3_ablation.png", "Ablation: what each data source adds"), ("p3_walk_forward.png", "Walk-forward by month")]:
    if (FIG / name).exists():
        with st.expander(cap):
            st.image(str(FIG / name))
abl = REPORTS / "exp_ablation.csv"
if abl.exists():
    with st.expander("Ablation table"):
        st.dataframe(read_csv(str(abl)).round(4), width="stretch")
