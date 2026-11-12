# Steps 2–4 — Ingestion, Cleaning & Exploratory Data Analysis

Full executed analysis: [notebooks/02_EDA.ipynb](../notebooks/02_EDA.ipynb) · figures in [reports/figures](../reports/figures)

## 2. Ingestion (`src/ingest.py`)

* 12 monthly CSVs, **28,237,658 rows**, 229 routes, 13 columns.
* MBTA encodes times as `1900-01-01THH:MM:SSZ` (time-of-day; `1900-01-02` = after midnight on the same service day).
  Converted to seconds-after-service-midnight → real timestamps `service_date + offset`.
  Despite the `Z` suffix the times are **local Boston time** (verified: volume peaks at 07:00 and 17:00).
* Route ids normalised (`"01"` → `"1"`); categorical dtypes → 0.8 GB in memory for the scoped set.
* `delay_sec = actual − scheduled`.

## 3. Cleaning & scoping (`src/clean.py`)

**Scope:** 14 MBTA key bus routes + 5 busiest schedule-managed routes = 19 routes, 288 timepoint stops, 1.18 M one-way trips.
(Route 117 is in the key-route list but not present in the 2025 file.)

| Step | Rows remaining | Removed |
|---|---|---|
| Raw rows (scoped routes) | 10,324,982 | – |
| Exact duplicates | 10,324,982 | 0 |
| Missing scheduled time | 10,324,982 | 0 |
| Missing actual (cancelled / unobserved) | 9,824,195 | 500,787 (4.9 %) |
| Duplicate stop events | 9,824,195 | 0 |
| Outliers (< −30 min or > +120 min) | **9,823,286** | 909 |

The missing-actual rate per route (2–17 %, Silver Line worst) is kept as a separate reliability KPI.

## 4. Key EDA findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **28.8 %** of stop events are > 5 min late → imbalanced binary target. Median delay +2.3 min, 95th pct +14.3 min | `eda_target.png` |
| 2 | Risk lowest at 05:00 (≈12 %), highest **14:00–18:00** (≈33–39 %) | `eda_by_hour.png`, `eda_heatmap_dow_hour.png` |
| 3 | Weekends worse than weekdays in the afternoon (30.8 % vs 28.5 %); **holidays much better (19.8 %)** | `eda_by_hour.png` |
| 4 | Seasonality: Mar–Jun and Sep–Oct worst, Jul–Aug and late Dec best → matches **school calendar** | `eda_daily_timeseries.png`, `eda_dow_month.png` |
| 5 | Route matters a lot: 44 (47.9 %), 28 (43.4 %), 39 (40.4 %) vs 111 (16.1 %), 66 (21.6 %) | `eda_by_route.png` |
| 6 | Strong route × direction × period interactions (e.g. route 28 PM peak 51 %, route 39 outbound 48 %) | `eda_route_direction_period.png` |
| 7 | Delay builds over the first 40 % of a trip, then **recovers at the terminal** (schedule padding) | `eda_stop_position.png` |
| 8 | **Propagation:** start vs end delay r = 0.58; leaving > 10 min late ⇒ 93 % chance of arriving late | `eda_propagation.png` |
| 9 | **Bunching:** 8.7 % of headway-route arrivals come < 25 % of the planned headway behind the previous bus; 14.5 % have gaps > 1.5× | `eda_bunching.png` |
| 10 | Linear correlations with the target are weak → non-linear, interaction-heavy problem → tree ensembles expected to win | `eda_correlation.png` |

## Implications for Step 5 (feature engineering)

* Temporal: `hour` (sin/cos), `dow`, `is_weekend`, `peak`, `month`, `is_holiday`, `is_school_day`, hour × weekend.
* Spatial: `route_id`, `direction_id`, `stop_id`, `time_point_order`, `stop_pos`, `point_type`, `standard_type`, `scheduled_headway`.
* Historical (train-period only, to avoid leakage): delay rate per route, route × hour, stop × direction, route × dow.
* Lagged: delay of the previous trip on the same route/direction at the same stop (known before the bus arrives).
* External: hourly weather (precipitation, snow, temperature, wind) for Boston Logan, event calendar. *(Implemented in Chapter 3 with Open-Meteo ERA5 reanalysis instead of Meteostat: same station area, no gaps, no key.)*
* Split: **train Jan–Sep, validate Oct, test Nov–Dec** (no shuffling). *(Final split, fixed in Chapter 4: train Jan-Aug, validate Sep-Oct, test Nov-Dec - a two-month validation period for early stopping and threshold choice; decision 3 in the decision log.)*
