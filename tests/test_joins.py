"""Join integrity checks for Phase 1 merges (real files; skipped if not built)."""
import pandas as pd
import pytest

from src.config import EXTERNAL_DIR, PROCESSED_DIR, REPORTS

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "bus_2025_enriched.parquet").exists(), reason="not built")


@pytest.mark.parametrize("year", [2024, 2025])
def test_enrich_keeps_row_count(year):
    a = pd.read_parquet(PROCESSED_DIR / f"bus_{year}_clean.parquet", columns=["delayed"])
    b = pd.read_parquet(PROCESSED_DIR / f"bus_{year}_enriched.parquet", columns=["delayed"])
    assert len(a) == len(b) and a.delayed.sum() == b.delayed.sum()


@pytest.mark.parametrize("year", [2024, 2025])
def test_coverage_thresholds(year):
    c = pd.read_csv(REPORTS / f"join_coverage_{year}.csv", index_col=0).coverage
    assert c["calendar"] >= 99 and c["weather"] >= 99
    assert c["gtfs stop coordinates"] >= 90 and c["gtfs segment (non-first timepoints)"] >= 90
    assert c["ridership incl. fallback"] >= 90


def test_external_tables_unique_keys():
    w = pd.read_parquet(EXTERNAL_DIR / "weather_hourly.parquet")
    assert not w.duplicated(["date", "hour"]).any()
    cal = pd.read_parquet(EXTERNAL_DIR / "calendar_daily.parquet")
    assert not cal.date.duplicated().any()
    seg = pd.read_parquet(EXTERNAL_DIR / "gtfs_segments.parquet")
    assert not seg.duplicated(["route_id", "from_stop", "to_stop"]).any()


def test_no_duplicate_stop_events():
    d = pd.read_parquet(PROCESSED_DIR / "bus_2025_enriched.parquet",
                        columns=["service_date", "route_id", "half_trip_id", "stop_id", "sched_sec"])
    assert not d.dropna(subset=["half_trip_id"]).duplicated().any()
