# Step 1 — Data Source Survey & Selection

## Requirements derived from the proposal

| # | Requirement | Why it matters |
|---|---|---|
| R1 | One row = one bus at one stop, with **scheduled** and **actual** time | Needed to compute delay and the binary label |
| R2 | Route, stop, direction, date present | Spatial + temporal features |
| R3 | One city, one mode, a full recent year | Proposal scope; captures seasonality |
| R4 | Freely downloadable, documented, licence allows academic use | Reproducibility |
| R5 | Matching GTFS (stop names, coordinates, stop order) | Stop position, maps on the dashboard |
| R6 | A live feed from the same agency | "Real-time mode" extension |
| R7 | Weather/holiday data easy to join | "Weather and events" extension |

## Candidates evaluated

| Source | Granularity | Sched. vs actual per stop? | Years | Live feed | Verdict |
|---|---|---|---|---|---|
| **MBTA (Boston) – Bus Arrival Departure Times** (MassDOT Open Data / ArcGIS Hub) | One row per trip × timepoint stop | **Yes** – `scheduled`, `actual`, plus `scheduled_headway`, `headway` | 2020 → 2026, one file per year, monthly CSVs | Yes – MBTA V3 API + GTFS‑RT | ✅ **Selected** |
| Chicago CTA (via UChicago Mansueto *cta-stop-watch*) | Derived from BusTracker pings | Derived (must interpolate) | 2023+ | Yes | Good, but data is reconstructed by a third party |
| NYC MTA Bus (Kaggle "NYC Bus Data", 2017) | 10‑minute GPS snapshots | Partial – `ScheduledArrivalTime` for *next* stop only | Jun–Dec 2017 only | Yes (SIRI) | Old, not a full year, noisy matching |
| Toronto TTC Bus Delay (Open Data Toronto) | **Incident log** (only delays are recorded) | No – no on-time trips | 2014 → now | Limited | Unsuitable: no negative class, cannot build classifier |
| Dublin Bus GPS (data.gov.ie) | Raw AVL pings | No schedule join | 2012–2013 | No | Too old, needs heavy reconstruction |
| TriMet Portland / other GTFS‑RT archives | Must be self-collected | Only after months of logging | — | Yes | Too slow to build for a project |

## Decision

**MBTA Bus Arrival Departure Times 2025** (Boston, bus, full calendar year 2025).

* Fields: `service_date, route_id, direction_id, half_trip_id, stop_id, time_point_id, time_point_order, point_type, standard_type, scheduled, actual, scheduled_headway, headway`
* Satisfies R1–R7: official agency data, GTFS-compatible stop IDs, MBTA V3 API for real-time, Boston weather is easy to join (Open-Meteo ERA5 reanalysis at Logan Airport; Meteostat considered), US/MA holidays via `holidays` package.
* Important nuance: MBTA evaluates *frequent* routes on **headway** (gap between buses) and other routes on **schedule**. The `standard_type` column tells us which. This is a talking point for the report and the reason we also compute a headway-based reliability label as an extension.

Source portal: https://mbta-massdot.opendata.arcgis.com/ (search "MBTA Bus Arrival Departure Times 2025").
