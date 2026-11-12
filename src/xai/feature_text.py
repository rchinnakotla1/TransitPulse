"""Plain-language names for every model feature (used by SHAP plots, the narrator and the dashboard)."""
from __future__ import annotations

import math

LABEL = {
    # temporal / calendar / daylight
    "hour_sin": "time of day", "hour_cos": "time of day", "dow": "day of week", "is_weekend": "weekend",
    "am_peak": "morning peak", "pm_peak": "evening peak", "period": "time period",
    "is_holiday": "public holiday", "is_school_day": "school day", "is_university_term": "university term",
    "holiday_week": "holiday week", "days_to_holiday": "days to next holiday", "school_dismissal": "school dismissal time",
    "sun_elevation": "sun height (daylight)", "is_dark": "darkness", "dark_peak": "peak hour in darkness",
    # spatial
    "route_id": "route", "direction_id": "direction", "stop_id": "stop", "time_point_order": "timepoint number on trip",
    "stop_pos": "position along the route", "point_type": "stop type (start/mid/end)", "standard_type": "schedule type",
    "scheduled_headway": "planned gap between buses", "trip_start_hour": "trip start hour",
    "min_into_trip": "minutes into the trip", "stop_lat": "stop location (lat)", "stop_lon": "stop location (lon)",
    # history
    "hist_route_hour": "route's usual late rate at this hour", "hist_route_dow": "route's usual late rate on this weekday",
    "hist_stop_period": "stop's usual late rate in this period", "hist_stop_hour": "stop's usual late rate at this hour",
    "hist_delay_route_hour": "route's usual delay at this hour", "hist_delay_stop_hour": "stop's usual delay at this hour",
    "recent_route_28d": "route's late rate in the last 4 weeks", "recent_stop_28d": "stop's late rate in the last 4 weeks",
    "hist_stop_hour_n": "amount of history for this stop/hour",
    # weather
    "precipitation": "rain/snow this hour", "precip_3h": "rain/snow in last 3 h", "snow_24h": "snowfall in last 24 h",
    "snow_depth": "snow on the ground", "is_snow_day": "snow day", "temperature_2m": "season/temperature",
    "wind_gusts_10m": "wind gusts", "heavy_rain": "heavy rain",
    # events
    "venue_event_nearby": "event at a nearby venue", "citywide_event": "citywide event",
    # gtfs
    "segment_km": "distance from previous timepoint", "stops_between": "stops since previous timepoint",
    "seg_sched_min": "scheduled minutes from previous timepoint", "schedule_tightness": "scheduled minutes per km",
    # ridership
    "avg_load": "typical passenger load", "avg_boardings": "typical boardings", "avg_alightings": "typical alightings",
    "load_vs_route": "load compared with the route average",
    # operational
    "prev_tp_delay": "delay at the previous timepoint", "prev_tp_delayed": "already late at previous timepoint",
    "start_delay": "delay at trip start", "min_since_prev_tp": "scheduled minutes left to this stop",
    "prev_bus_delay": "delay of the bus ahead", "prev_bus_gap_ratio": "gap to the bus ahead vs plan",
}

# features that a *planner* can change (used by the what-if analysis)
CONTROLLABLE = {"seg_sched_min": "add / remove scheduled running time",
                "scheduled_headway": "change the planned frequency"}

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def label(f: str) -> str:
    return LABEL.get(f, f.replace("_", " "))


def fmt_value(f: str, v) -> str:
    """Human-readable value of a feature for explanations."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "unknown"
    try:
        if f in ("hour_sin", "hour_cos"):
            return "(see hour)"
        if f == "dow":
            return DOW[int(v)]
        if f in ("is_weekend", "am_peak", "pm_peak", "is_holiday", "is_school_day", "is_university_term", "holiday_week",
                 "school_dismissal", "is_dark", "dark_peak", "is_snow_day", "heavy_rain", "venue_event_nearby",
                 "citywide_event", "prev_tp_delayed"):
            return "yes" if float(v) >= 0.5 else "no"
        if f.startswith(("hist_route", "hist_stop", "recent_")) and "delay" not in f and not f.endswith("_n"):
            return f"{float(v) * 100:.0f}%"
        if f in ("prev_tp_delay", "start_delay", "prev_bus_delay", "hist_delay_route_hour", "hist_delay_stop_hour",
                 "seg_sched_min", "min_since_prev_tp", "scheduled_headway", "min_into_trip"):
            return f"{float(v):.1f} min"
        if f in ("precipitation", "precip_3h"):
            return f"{float(v):.1f} mm"
        if f == "snow_24h":
            return f"{float(v):.1f} cm"
        if f == "snow_depth":                      # stored in metres (Open-Meteo)
            return f"{float(v) * 100:.0f} cm"
        if f == "temperature_2m":
            return f"{float(v):.0f} °C"
        if f == "wind_gusts_10m":
            return f"{float(v):.0f} km/h"
        if f == "stop_pos":
            return f"{float(v) * 100:.0f}% along"
        if isinstance(v, float):
            return f"{v:.2f}"
    except (TypeError, ValueError):
        pass
    return str(v)
