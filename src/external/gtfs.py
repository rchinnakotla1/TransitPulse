"""Step 5c — GTFS static schedule enrichment (stops, coordinates, route geometry).

Source  : MBTA GTFS archive (https://cdn.mbta.com/archive/archived_feeds.txt), the official
          General Transit Feed Specification feeds MBTA publishes for every schedule change.
Why     : The arrival/departure file only lists *timepoints*. GTFS tells us
            * stop names + latitude/longitude      -> maps in the website, distance to event venues
            * how many regular stops lie between two timepoints  -> more stops = more dwell time
            * the distance (km) between two timepoints            -> with the scheduled running time
              this gives SCHEDULE TIGHTNESS (scheduled minutes per km), something planners control.
Join    : (route_id, from_stop_id, to_stop_id) for consecutive timepoints of a trip; stop_id for maps.
Output  : data/external/gtfs_stops.parquet, data/external/gtfs_segments.parquet

One representative feed per MBTA "rating" (seasonal schedule) is used: Winter/Spring/Summer/Fall 2024-2025.
"""
import csv
import io
import zipfile

import numpy as np
import pandas as pd
import requests

from src.config import EXTERNAL_DIR, PROCESSED_DIR, SCOPED_ROUTES

ARCHIVE = "https://cdn.mbta.com/archive/archived_feeds.txt"
SEASONS = [f"{s} {y}" for y in (2024, 2025) for s in ("Winter", "Spring", "Summer", "Fall")]
GTFS_DIR = EXTERNAL_DIR / "gtfs"
GTFS_DIR.mkdir(exist_ok=True)


def pick_feeds():
    rows = list(csv.DictReader(io.StringIO(requests.get(ARCHIVE, timeout=60).text)))
    feeds = []
    for season in SEASONS:
        cand = [r for r in rows if r["feed_version"].startswith(season)]
        if not cand:
            continue
        # longest validity window = the "main" schedule of that rating
        best = max(cand, key=lambda r: (pd.Timestamp(r["feed_end_date"]) - pd.Timestamp(r["feed_start_date"])).days)
        feeds.append((season, best["archive_url"]))
    return feeds


def load_feed(season, url):
    path = GTFS_DIR / f"{season.replace(' ', '_')}.zip"
    if not path.exists():
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    z = zipfile.ZipFile(path)
    rd = lambda n, **k: pd.read_csv(z.open(n), dtype=str, **k)
    stops = rd("stops.txt", usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"])
    trips = rd("trips.txt", usecols=["route_id", "trip_id", "direction_id"])
    trips = trips[trips.route_id.isin(SCOPED_ROUTES)]
    st = rd("stop_times.txt", usecols=["trip_id", "stop_id", "stop_sequence"])
    st = st[st.trip_id.isin(set(trips.trip_id))]
    st["stop_sequence"] = st["stop_sequence"].astype(int)
    return stops, trips, st


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def needed_pairs():
    """Consecutive timepoint pairs actually observed in the arrival data (both years)."""
    pairs = []
    for y in (2024, 2025):
        d = pd.read_parquet(PROCESSED_DIR / f"bus_{y}_clean.parquet",
                            columns=["service_date", "half_trip_id", "route_id", "stop_id", "time_point_order"])
        d = d.sort_values(["service_date", "half_trip_id", "time_point_order"])
        d["route_id"], d["stop_id"] = d.route_id.astype(str), d.stop_id.astype(str)
        same = (d.half_trip_id.eq(d.half_trip_id.shift()) & d.service_date.eq(d.service_date.shift()))
        d["from_stop"] = d.stop_id.shift().where(same)
        pairs.append(d.dropna(subset=["from_stop"])[["route_id", "from_stop", "stop_id"]].drop_duplicates())
    return pd.concat(pairs).drop_duplicates().rename(columns={"stop_id": "to_stop"})


def main():
    feeds = pick_feeds()
    print("Feeds:", feeds)
    all_stops, patterns = [], []
    for season, url in feeds:
        stops, trips, st = load_feed(season, url)
        all_stops.append(stops)
        s = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].astype(float)
        st = st.merge(trips, on="trip_id").sort_values(["trip_id", "stop_sequence"])
        st = st.join(s, on="stop_id")
        # one row per unique stop pattern (many trips share the same sequence)
        seq = st.groupby("trip_id").agg(route_id=("route_id", "first"), stops=("stop_id", tuple),
                                        lat=("stop_lat", list), lon=("stop_lon", list))
        seq = seq.drop_duplicates(subset=["route_id", "stops"])
        patterns.append(seq)
        print(season, "patterns:", len(seq))
    stops = pd.concat(all_stops).drop_duplicates("stop_id", keep="last")
    stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].astype(float)
    stops.to_parquet(EXTERNAL_DIR / "gtfs_stops.parquet", index=False)

    pats = pd.concat(patterns).drop_duplicates(subset=["route_id", "stops"])
    pairs = needed_pairs()
    out = []
    for route, g in pairs.groupby("route_id"):
        rp = pats[pats.route_id == route]
        prepared = []
        for _, p in rp.iterrows():
            lat, lon = np.array(p.lat), np.array(p.lon)
            cum = np.concatenate([[0], np.cumsum(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
            prepared.append(({s: i for i, s in enumerate(p.stops)}, cum))
        for _, row in g.iterrows():
            found = []
            for idx, cum in prepared:
                i, j = idx.get(row.from_stop), idx.get(row.to_stop)
                if i is not None and j is not None and j > i:
                    found.append((cum[j] - cum[i], j - i - 1))
            if found:
                km, n = np.median([f[0] for f in found]), int(np.median([f[1] for f in found]))
                out.append((route, row.from_stop, row.to_stop, km, n))
    seg = pd.DataFrame(out, columns=["route_id", "from_stop", "to_stop", "segment_km", "stops_between"])
    seg.to_parquet(EXTERNAL_DIR / "gtfs_segments.parquet", index=False)
    print(f"segments matched: {len(seg)}/{len(pairs)} ({len(seg)/len(pairs):.1%})")
    print(seg.describe().round(2))


if __name__ == "__main__":
    main()
