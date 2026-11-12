"""Step 5d — Major events that load the Boston street network.

Sources : MLB Stats API (official)        -> Red Sox home games at Fenway Park (with start times)
          NHL web API (official)          -> Bruins home games at TD Garden (regular season + playoffs)
          City of Boston public calendar  -> citywide events (Marathon, St. Patrick's parade, July 4th,
                                             Pride, First Night), entered manually with sources.
Why     : Event crowds and road closures cause localized congestion. Agencies plan extra service for
          them, so knowing an event's effect is directly actionable. (EDA: the worst day of 2025, 4 Apr,
          was the Red Sox home opener.)
Join    : service_date + scheduled hour, and only for routes with a timepoint within VENUE_RADIUS_KM of
          the venue (venue events) or for all routes (citywide events).
Output  : data/external/events.parquet  (one row per event with start/end window)
          data/external/event_routes.parquet (venue -> affected routes)
"""
import numpy as np
import pandas as pd
import requests

from src.config import EXTERNAL_DIR, PROCESSED_DIR, SCOPED_ROUTES, TIMEZONE

VENUES = {"Fenway Park": (42.3467, -71.0972), "TD Garden": (42.3662, -71.0621)}
VENUE_RADIUS_KM = 1.0          # walking-distance catchment; 1.5 km wrongly captured Ruggles-area routes
PRE_H, POST_H = 2, 4          # window: 2 h before start .. 4 h after start (game + egress)

CITYWIDE = [  # date, name, approx start hour, duration h  (source: boston.gov events calendar / news)
    ("2024-03-17", "St. Patrick's Day Parade (South Boston)", 13, 4),
    ("2024-04-15", "Boston Marathon", 9, 8),
    ("2024-06-08", "Boston Pride for the People Parade", 12, 4),
    ("2024-07-04", "July 4th Fireworks (Esplanade)", 17, 7),
    ("2024-12-31", "First Night", 12, 12),
    ("2025-03-16", "St. Patrick's Day Parade (South Boston)", 13, 4),
    ("2025-04-21", "Boston Marathon", 9, 8),
    ("2025-06-14", "Boston Pride for the People Parade", 12, 4),
    ("2025-07-04", "July 4th Fireworks (Esplanade)", 17, 7),
    ("2025-12-31", "First Night", 12, 12),
]


def red_sox(season):
    r = requests.get("https://statsapi.mlb.com/api/v1/schedule",
                     params=dict(sportId=1, teamId=111, season=season, gameType="R,F,D,L,W"), timeout=60)
    games = [g for d in r.json()["dates"] for g in d["games"] if g["venue"]["name"] == "Fenway Park"]
    return pd.DataFrame({"venue": "Fenway Park", "event": "Red Sox home game",
                         "start_utc": [g["gameDate"] for g in games],
                         "status": [g["status"]["detailedState"] for g in games]})


def bruins(season_code):
    r = requests.get(f"https://api-web.nhle.com/v1/club-schedule-season/BOS/{season_code}", timeout=60)
    games = [g for g in r.json()["games"] if g["homeTeam"]["abbrev"] == "BOS" and g["gameType"] in (2, 3)]
    return pd.DataFrame({"venue": "TD Garden", "event": "Bruins home game",
                         "start_utc": [g["startTimeUTC"] for g in games], "status": "Final"})


def affected_routes():
    """Routes with any timepoint stop within VENUE_RADIUS_KM of each venue (uses GTFS coordinates)."""
    stops = pd.read_parquet(EXTERNAL_DIR / "gtfs_stops.parquet")
    d = pd.read_parquet(PROCESSED_DIR / "bus_2025_clean.parquet", columns=["route_id", "stop_id"]).drop_duplicates()
    d = d.astype(str).merge(stops, on="stop_id")
    rows = []
    for v, (lat, lon) in VENUES.items():
        dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin(np.radians(d.stop_lat - lat) / 2) ** 2 +
               np.cos(np.radians(lat)) * np.cos(np.radians(d.stop_lat)) * np.sin(np.radians(d.stop_lon - lon) / 2) ** 2))
        near = d[dist <= VENUE_RADIUS_KM].route_id.unique()
        rows += [(v, r) for r in near]
    return pd.DataFrame(rows, columns=["venue", "route_id"])


def main():
    ev = pd.concat([red_sox(2024), red_sox(2025), bruins("20232024"), bruins("20242025"), bruins("20252026")])
    ev = ev[~ev.status.str.contains("Postponed|Cancelled", na=False)]
    ev["start"] = pd.to_datetime(ev.start_utc, utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    ev["end"] = ev["start"] + pd.Timedelta(hours=3)
    ev["scope"] = "venue"
    cw = pd.DataFrame(CITYWIDE, columns=["date", "event", "h", "dur"])
    cw["start"] = pd.to_datetime(cw.date) + pd.to_timedelta(cw.h, unit="h")
    cw["end"] = cw["start"] + pd.to_timedelta(cw.dur, unit="h")
    cw["venue"], cw["scope"] = "Citywide", "citywide"
    ev = pd.concat([ev[["venue", "event", "start", "end", "scope"]], cw[["venue", "event", "start", "end", "scope"]]])
    ev = ev[(ev.start >= "2024-01-01") & (ev.start < "2026-01-01")].sort_values("start").reset_index(drop=True)
    ev["window_start"] = ev.start - pd.Timedelta(hours=PRE_H)
    ev["window_end"] = ev.start + pd.Timedelta(hours=POST_H)
    ev.to_parquet(EXTERNAL_DIR / "events.parquet", index=False)

    ar = affected_routes()
    ar.to_parquet(EXTERNAL_DIR / "event_routes.parquet", index=False)
    print(ev.groupby([ev.start.dt.year, "event"]).size())
    print(ar.groupby("venue").route_id.apply(list))


if __name__ == "__main__":
    main()
