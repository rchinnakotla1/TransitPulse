"""Page 8: data sources, methodology summary, limitations and the decision log."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from common import ROOT, page

page("Data sources & method", "📚")

st.markdown("""
### Data (all free, official or trusted)

| # | Source | Publisher | Used for | Join key |
|---|---|---|---|---|
| D1 | Bus arrival/departure times 2025 (9.8 M timepoint events, 19 key bus routes) | MBTA OPMI (MassDOT Open Data) | the target: > 5 min late | n/a |
| D2 | Bus arrival/departure times 2024 | MBTA OPMI | seed historical rates; drift check | n/a |
| D3 | GTFS schedules (8 seasonal feeds) | MBTA | segment length, stops between timepoints, **schedule tightness** | route + segment; stop_id |
| D4 | Hourly weather, Logan Airport | Open-Meteo (ERA5 reanalysis; forecast API live) | rain, snow, temperature, wind | date + hour |
| D5 | Holidays, Boston Public Schools & university calendars | `holidays` package, BPS | school / holiday effects | date |
| D6 | Red Sox / Bruins home games, citywide events | MLB Stats API, NHL API, City of Boston | event traffic | hour window + routes ≤ 1 km from venue |
| D7 | Passenger load (APC), previous Fall | MBTA OPMI | crowding / dwell time | route, direction, stop, day type, hour |
| D8 | MBTA bus prediction accuracy | MBTA OPMI | benchmark context | n/a |
| Live | MBTA V3 API `/predictions` + Open-Meteo forecast | MBTA, Open-Meteo | live mode | trip, stop |

### Method in one picture
""")
st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname=Helvetica];
  P0 [label="P0 Data choice\\n& cleaning"]; P1 [label="P1 External data\\n(GTFS, weather,\\ncalendar, events, APC)"];
  P2 [label="P2 Leakage-safe\\nfeatures + time split"]; P3 [label="P3 Models\\nB0, LR, DT, RF,\\nXGB, LGB"];
  P4 [label="P4 Explainability\\nSHAP, rules, what-if,\\nOpenAI narratives"]; P5 [label="P5 Extensions\\nseverity, minutes,\\npropagation, bunching,\\nchronic segments"];
  P6 [label="P6 Dashboard"]; P7 [label="P7 Live mode"];
  P0 -> P1 -> P2 -> P3 -> P4 -> P6; P3 -> P5 -> P6; P3 -> P7; P4 -> P7 }
""")

st.markdown("""
### Evaluation protocol
* **Time-based split:** train Jan–Aug 2025 · validate Sep–Oct (tuning, thresholds) · **test Nov–Dec, scored once**.
* Historical features use only data up to the previous day. Operational features use only what is known when the bus leaves its
  previous timepoint.
* 95 % confidence intervals by day-block bootstrap; ablation by feature group; walk-forward months; drift vs 2024.

### Key limitations
* 19 key bus routes in scope; results may differ on other routes.
* SHAP and what-ifs describe the model (associations), not causal effects.
* Live mode approximations: start delay ≈ current delay, events not scored live, school calendar only through June 2026,
  only the next timepoint per bus is scored.
* The LLM needs a paid OpenAI key. If it fails, templates are used and a warning is shown.
""")

dl = ROOT / "docs" / "decision_log.md"
if dl.exists():
    with st.expander("Decision log (every major choice and why)"):
        st.markdown(dl.read_text(encoding="utf-8"))
docs = sorted((ROOT / "docs").glob("0*.md"))
with st.expander("Thesis chapters (docs/)"):
    for d in docs:
        st.markdown(f"- `{d.name}`")
