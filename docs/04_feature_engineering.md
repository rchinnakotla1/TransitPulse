# Phase 2: Feature Engineering

Full executed analysis: [notebooks/04_Feature_Engineering.ipynb](../notebooks/04_Feature_Engineering.ipynb) · code `src/features.py`,
`src/encoding.py` · spec `reports/feature_spec.json` · figures `reports/figures/p2_*.png` · tests `tests/test_leakage.py`, `tests/test_features.py`, `tests/test_joins.py`

## 1. Guiding rule and the two prediction modes

Every feature must be something a transit agency **would know at the moment it makes the prediction**.

| Mode | Agency user | Prediction moment | Allowed information | # features |
|---|---|---|---|---|
| **Planning** (day-ahead) | Planners, rider alerts | day before | calendar, daylight, route/stop/schedule, history up to yesterday, weather (as forecast), events, previous-season ridership | 55 |
| **Operational** (minutes-ahead) | Dispatchers | bus leaves previous timepoint | planning + live AVL features | 61 |

## 2. Feature groups

| Group | Features | Why |
|---|---|---|
| Temporal | `hour_sin/cos`, `dow`, `is_weekend`, `am_peak`, `pm_peak`, `period` | demand cycle; cyclical so 23:00 ≈ 00:00 |
| Calendar | `is_holiday`, `is_school_day`, `is_university_term`, `holiday_week`, `days_to_holiday`, `school_dismissal` | Phase 1: school days +10–15 pp in peaks |
| Daylight | `sun_elevation` (NOAA solar position, DST-aware), `is_dark` (sun < −6°), `dark_peak` | the test period starts right after the 2 Nov clock change; "17:00" is a different hour in November than in October |
| Spatial | `route_id`, `direction_id`, `stop_id`, `time_point_order`, `stop_pos`, `point_type`, `standard_type`, `scheduled_headway`, `trip_start_hour`, `min_into_trip`, `stop_lat/lon` | EDA: delay builds mid-route, strong route effects |
| History | `hist_route_hour`, `hist_route_dow`, `hist_stop_period`, `hist_stop_hour` (+`_n`), `hist_delay_route_hour`, `hist_delay_stop_hour`, `recent_route_28d`, `recent_stop_28d` | "how this place/time usually performs" = what agencies use today |
| Weather | `precipitation`, `precip_3h`, `snow_24h`, `snow_depth`, `is_snow_day`, `temperature_2m`, `wind_gusts_10m`, `heavy_rain` | Phase 1: weak but episodic |
| Events | `venue_event_nearby`, `citywide_event` | Phase 1: +3.9 pp locally |
| GTFS | `segment_km`, `stops_between`, `seg_sched_min`, `schedule_tightness` | planner-controllable running time |
| Ridership | `avg_load`, `avg_boardings`, `avg_alightings`, `load_vs_route` | dwell time |
| Operational | `prev_tp_delay`, `prev_tp_delayed`, `start_delay`, `min_since_prev_tp`, `prev_bus_delay`, `prev_bus_gap_ratio` | delay propagation & bunching |

**Deliberately excluded:** `month` (train Jan–Aug vs test Nov–Dec, so trees would extrapolate), the actual `headway` and anything else derived from the bus's own arrival.

## 3. Leakage-safe historical rates

* Daily sums and counts per key. The cumulative sum **minus today** means a row on day *D* sees only days strictly before *D*.
* Bayesian smoothing toward the global rate known so far: $\hat p = (\text{late}_{<D} + 30\,\bar p_{<D}) / (n_{<D} + 30)$.
* Seeded with 2024, so January 2025 already has a full year of history.
* 28-day rolling versions follow the current level (they adapt to drift). The expanding version is stable but slow; see `p2_history_example.png`.

**Proof by test** (`tests/test_leakage.py`, 7 tests): flipping every label on or after *D* leaves all features up to *D* unchanged. A label more than
28 days old does not affect the rolling rate. Operational features never read the bus's own arrival (including the early-departure edge case).

**Demonstration** (test set, ROC-AUC of the single feature):

| Version | AUC |
|---|---|
| `hist_stop_hour` (safe, expanding) | 0.657 |
| `recent_stop_28d` (safe, 28 days) | 0.682 |
| naïve: same key over the whole test period | 0.721 |
| naïve: same key, same day incl. own row | **0.898** |

The naïve versions look better offline but could never be computed in deployment.

## 4. Operational features (definition of "already known")

The prediction moment *t0* is when the bus departs its previous timepoint (1 min before scheduled time at the first timepoint).
`prev_bus_delay` is the delay of the most recent **other** bus on the same route/direction that passed this stop **before t0** (`merge_asof`).
`prev_bus_gap_ratio` = time since that bus passed ÷ planned headway.

| Evidence | Result |
|---|---|
| prev timepoint delay 6–8 min / > 10 min | 84 % / ≈ 99 % late at the next timepoint |
| previous bus > 20 min late vs on time | 62 % vs 15 % |
| gap ratio > 1 (gap opened ahead) vs ≤ 1 | 43–52 % vs ≈ 17 % |

## 5. Time-based split

| Split | Dates | Rows | % delayed |
|---|---|---|---|
| Train | 2025-01-01 → 08-31 | 6,448,156 | 28.8 |
| Validation | 2025-09-01 → 10-31 | 1,690,047 | 32.8 |
| Test | 2025-11-01 → 12-31 | 1,685,083 | 25.1 |
| Drift | 2024 (all) | 9,258,150 | 37.3 |

The delay rate shifts between splits (busy Sep–Oct vs calm Nov–Dec). Consequences: threshold-free metrics (ROC-AUC, PR-AUC) as the main comparison,
a calibration check on test, and reliance on rolling rates to adapt.

## 6. Univariate signal (train, best AUC per group)

operational 0.96 · history 0.68 · spatial 0.58 · temporal 0.57 · ridership 0.56 · GTFS 0.53 · weather 0.52 · calendar 0.52 · daylight 0.52 · events 0.50

`is_school_day` alone is ≈ 0.50 even though school weekdays are clearly worse. Mixed over all days the effect cancels out (weekends are also bad),
so its value lies in interactions with time of day. **Decision:** features are not dropped on univariate AUC; the Phase 3 ablation decides.

## 6b. Seasonal shift: daylight, adversarial validation, covariate overlap

*Why this section exists:* the model trains on Jan–Aug and is tested on Nov–Dec, which differ in climate, daylight saving time and holidays.

| Check | Result | Interpretation |
|---|---|---|
| Weekday PM peak (16–18 h) in darkness | train 13.6 % · valid 8.2 % · **test 72.6 %** | biggest calendar difference between the periods; now captured explicitly by the daylight features |
| % delayed, dark vs daylight PM peak | train 26.7 vs 31.4 · valid 44.8 vs 41.7 · test 32.6 vs 31.6 | weak and season-confounded on its own; the ablation decides |
| Adversarial AUC (train vs test), all planning features | **1.000** | the periods are perfectly separable: the history count grows over time, temperature differs, and the autumn schedule "rating" changes `standard_type` / `seg_sched_min` |
| … without date-dependent features | 0.992 | day-level features (events, snow, wind, day of week) fingerprint individual dates, so this AUC cannot fall to 0.5 and is not by itself evidence of harmful shift |
| Covariate overlap (test rows inside train 0.5–99.5 % range) | temperature 100 %, precipitation 100 %, snow 100 %, load 99.6 %, hist_stop_hour 99.1 %, sun elevation 95.4 %, recent_route_28d 94.0 % | the model interpolates; it does not extrapolate |

**Decision.** Refit the final model on Jan–Oct so the autumn schedule is seen. Use ROC-/PR-AUC plus calibration as the headline metrics.
Measure the remaining seasonal risk directly in Phase 3 with the 12-month window (Sep 2024 – Aug 2025) and the walk-forward experiments.

## 7. Encodings (`src/encoding.py`)

| Model | Categoricals | Missing values | Scaling |
|---|---|---|---|
| Logistic regression | one-hot (levels < 50 rows pooled) | median + missing-indicator | standardised |
| Decision tree / Random forest | integer codes | sentinel −999 | none |
| XGBoost | native categorical | native | none |

Missingness is structural: GTFS segment 16.7 % (first timepoint plus unmatched pairs), previous-timepoint features 12 % (trip start), ridership 5.9 %.

## 8. Reproduce

```powershell
.\.venv\Scripts\python -m src.features          # ≈ 2 min
.\.venv\Scripts\python -m pytest tests -q        # 24 tests (incl. tests/test_daylight.py)
cd notebooks; ..\.venv\Scripts\python build_features_notebook.py
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 04_Feature_Engineering.ipynb
```
