# Phase 1: External Data Integration & Validation

Full executed analysis: [notebooks/03_External_Data.ipynb](../notebooks/03_External_Data.ipynb) · figures `reports/figures/p1_*.png`
· join audit `reports/join_coverage_{2024,2025}.csv`

## 1. Why merge external data?

The arrival records show *when* and *where* buses are late, but not *why*. A transit agency already has, or can freely
obtain, several other sources that plausibly cause delay. Each one is added only if (a) a transit agency could realistically
use it, (b) it comes from an official or trusted source, and (c) it would be **known at prediction time** (no leakage).

| # | Source | Publisher | Why a transit agency cares | Code | Join key |
|---|---|---|---|---|---|
| D1 | Bus arrival/departure 2025 | MBTA OPMI | the target | `src/ingest.py`, `src/clean.py` | n/a |
| D2 | Bus arrival/departure 2024 | MBTA OPMI | seed the historical rates; measure drift | same | n/a |
| D3 | GTFS schedules (8 seasonal feeds) | MBTA | segment length, stops between timepoints, **schedule tightness** (planner-controllable) | `src/external/gtfs.py` | route + (prev timepoint → timepoint); stop_id |
| D4 | Hourly weather, Logan Airport | Open-Meteo (ERA5 reanalysis) | rain/snow slow traffic and boarding; snow schedules | `src/external/weather.py` | date + hour |
| D5 | Holidays, Boston Public Schools & university calendar | `holidays` pkg, BPS calendars | school trips and school-run traffic | `src/external/calendar_days.py` | date |
| D6 | Red Sox / Bruins home games, citywide events | MLB Stats API, NHL API, City of Boston | event service planning | `src/external/events.py` | hour (−2 h…+4 h) + routes within 1 km of venue |
| D7 | Passenger load (APC), previous Fall season | MBTA OPMI | crowding → dwell time; capacity vs running-time problems | `src/external/ridership.py` | route, direction, stop, day type, trip start hour |
| D8 | MBTA bus prediction accuracy | MBTA OPMI | benchmark context only | `src/external/prediction_accuracy.py` | n/a |

All merges happen in `src/merge_external.py`, which **asserts the row count is unchanged** (no duplicating joins) and writes
`data/processed/bus_{year}_enriched.parquet` (57 columns).

### Leakage rules applied
* Weather is joined on the **scheduled** hour (in planning mode it represents a forecast).
* Ridership comes from the **previous** Fall season (2024 rows ← Fall 2023, 2025 rows ← Fall 2024).
* Events and calendars are known in advance.
* GTFS uses the schedule in force on the service date.

## 2. Join coverage

| Source | 2024 | 2025 |
|---|---|---|
| Calendar | 100 % | 100 % |
| Weather | 100 % | 99.98 % |
| Events (rows inside an event window) | 3.3 % | 3.1 % |
| GTFS segment (non-first timepoints) | 94.9 % | 94.6 % |
| GTFS stop coordinates | 97.3 % | 96.6 % |
| Ridership (exact hour / with fallback) | 89.1 / 96.4 % | 93.4 / 94.0 % |

**Decision:** missing segment and ridership values are kept as NaN. XGBoost handles them natively; for the linear model we impute the route median and add a missing-flag.

**Bug found and fixed during merge:** ~40–100 rows per month have no `half_trip_id`. Group-wise operations silently produced NaN for them,
so these rows now get no previous segment and use their own scheduled hour as the trip start.

## 3. Findings per source

| Source | Signal? | Evidence | Carried forward as features |
|---|---|---|---|
| 2024 arrivals | ✅ **strong drift**: 37.3 % → 28.8 % delayed, lower in every month; uneven by route (22, 1, 23, 116, 66 improved 10–24 pp; 32, 34E, 71, 73 slightly worse) | `p1_year_comparison.png` | expanding-window historical rates; drift experiment |
| Weather | ⚠️ **weak**: > 4 mm/h rain gives 33 % vs 29 % dry, but is rare; only ≈ +1 pp within route × hour × day-type cells (positive in 54 % of cells). Snow and cold look *protective* only because they coincide with the low-demand winter months; within January, snow days were 6 pp worse (28.8 % vs 22.5 %), February showed no difference | `p1_weather.png` | precip, precip_3h, snow_24h, is_snow_day, temperature, gusts. **The ablation decides** |
| Calendar | ✅ **strong**: regular weekdays 23.8 % (no school) vs 28.5–30.6 % (school); +10–15 pp in AM and PM peaks | `p1_calendar.png` | is_school_day, is_university_term, is_holiday, holiday_week |
| Events | ✅ **local**: +3.9 pp vs same route/weekday/hour; 4 of the 8 worst days are event days (Red Sox opener 4 Apr = worst day of 2025; St Patrick's parade 16 Mar) | events table in notebook | venue_event_nearby, citywide_event |
| GTFS | ✅ tighter schedules are late more often (25 % at 6 km/h → 34.5 % at 17 km/h); number of stops alone r ≈ 0.02 | `p1_gtfs_map_tightness.png` | segment_km, stops_between, seg_sched_min, schedule_tightness, lat/lon |
| Ridership | ✅ load deciles 21 % → 35 %; busier than route average 25 % → 34 %; direction mapping verified (5× better than flipped) | `p1_ridership.png` | avg_load, avg_boardings, avg_alightings, load_vs_route |
| MBTA accuracy | context: MBTA countdowns are 81–87 % accurate | `p1_mbta_prediction_accuracy.png` | n/a (report & website) |

## 4. What this means for the project

1. **The research question becomes testable.** Phase 3 runs an ablation: historical baseline → +calendar → +weather → +events → +GTFS → +ridership → +operational,
   and reports how much each source adds. The prediction from this phase is that calendar, GTFS and ridership help, and weather barely does.
2. **Retraining matters.** The 2024→2025 drift means a static model would over-predict delays. The agency recommendation will be to retrain monthly or quarterly.
3. **Some spikes stay unexplained.** Four of the eight worst days have no public explanation (incidents, detours, staffing). This is an honest limitation for the thesis.
4. **Planner insights are already visible.** Tight segments in the Roxbury / Dorchester core, and school-day peaks, are concrete targets for added running time or school trippers.

## 5. Reproduce

```powershell
.\.venv\Scripts\python -m src.external.weather
.\.venv\Scripts\python -m src.external.calendar_days
.\.venv\Scripts\python -m src.external.gtfs
.\.venv\Scripts\python -m src.external.events
.\.venv\Scripts\python -m src.external.ridership
.\.venv\Scripts\python -m src.external.prediction_accuracy
.\.venv\Scripts\python -m src.merge_external 2024 2025
cd notebooks; ..\.venv\Scripts\python build_external_notebook.py
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 03_External_Data.ipynb
```
