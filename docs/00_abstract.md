TransitPulse is a machine-learning system that predicts whether an MBTA bus will be more than five minutes late at a
timepoint and explains why, for transit-agency dispatchers, service planners and rider-communication staff. The study uses
9.82 million timepoint stop events on 19 Boston bus routes in 2025, of which 28.8 % were late. It merges them with
free, trusted external sources: GTFS schedule design, Open-Meteo ERA5 weather, public-holiday and Boston school
calendars, MLB/NHL event schedules and MBTA ridership. Models are trained on January–August, tuned on September–October
and tested once on the untouched November–December months.

Six models were compared: a historical-average baseline, logistic regression, a decision tree, random forest, XGBoost and LightGBM.
They were evaluated in two modes. A day ahead (planning mode), XGBoost reaches ROC-AUC 0.755 against 0.657 for the agency's
historical average, a 39 % gain in PR-AUC. With live information (operational mode) it reaches 0.978, and the dispatcher alert has
precision 0.93 at recall 0.82. An ablation shows that day-ahead risk is driven by recurring time × place patterns. External data refines
predictions where it applies, for example on rain and snow days, but does not transform them. The bus's own current delay is
decisive in real time. Learning-curve, walk-forward and training-window experiments confirm that the January–August split
is not under-trained.

TreeSHAP, surrogate rules and what-if analysis explain the predictions, and the explanations were tested for faithfulness and for
stability across XGBoost and LightGBM. Plain-language narratives are written by OpenAI GPT-4o-mini under a numeric fact check
with a template fallback; 100 % of the texts shown are faithful. Extensions answer the follow-up questions. Severity classes reach
QWK 0.864 in real time. Delay minutes reach MAE 1.08 min, with conformal-calibrated 80 % intervals at 79.6 % coverage. The extensions
also cover delay propagation (88 % of late buses stay late), bus bunching (PR-AUC 0.444, six times the base rate) and 29 chronically
late segments, most of which inherit their delay from upstream. The analysis also uncovered and fixed a train/serve skew in a
bus-ahead feature. Finally, a Streamlit website with nine pages and a live mode on the MBTA V3 API delivers the results. The live mode
degrades gracefully when the API or the LLM key fails.
