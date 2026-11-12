"""Phase 6/7 tests: every dashboard page renders without an exception, and live mode degrades gracefully.

Pages are run headless with ``streamlit.testing.v1.AppTest``. Network calls (MBTA, Open-Meteo, OpenAI) are disabled so the
tests are deterministic and work offline: live mode must then show a warning, never crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.live import mbta_live  # noqa: E402

PAGES = ["Home.py"] + sorted(p.relative_to(APP_DIR).as_posix() for p in (APP_DIR / "pages").glob("*.py"))
HAVE_DATA = (ROOT / "reports" / "app" / "profiles.parquet").exists()


class _Resp:
    def __init__(self, code: int, js: dict | None = None):
        self.status_code, self._js = code, js or {}

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture
def offline(monkeypatch):
    """No network: every requests.get raises ConnectionError; OpenAI is disabled."""
    def boom(*a, **k):
        raise requests.ConnectionError("offline test")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setenv("OPENAI_API_KEY", "")


# ---------------------------------------------------------------- live mode: error handling (no Streamlit)
def test_live_rate_limit(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(429))
    df, st = mbta_live.fetch_predictions("28")
    assert df.empty and not st.ok and "rate limit" in st.message


def test_live_bad_key(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(403))
    df, st = mbta_live.fetch_predictions("28")
    assert df.empty and not st.ok and "key" in st.message


def test_live_network_failure(offline):
    df, st = mbta_live.fetch_predictions("28")
    assert df.empty and not st.ok and "Cannot reach" in st.message
    out, st2, w = mbta_live.live_snapshot("28")
    assert out.empty and not st2.ok


def test_live_no_buses(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, {"data": [], "included": []}))
    df, st = mbta_live.fetch_predictions("28")
    assert df.empty and not st.ok and "No live predictions" in st.message


def test_weather_fallback(offline):
    w = mbta_live.current_weather()
    assert w["_source"].startswith("unavailable") and w["precipitation"] == 0.0


def test_bus_ahead():
    t0 = pd.Timestamp("2026-01-05 08:00", tz="America/New_York")
    m = pd.Timedelta(minutes=1)
    rows = [  # trip A is ahead of B on the same direction; C runs the other way
        ("A", "s3", 0, t0 + 2 * m, 4.0), ("A", "s4", 0, t0 + 6 * m, 4.5),
        ("B", "s2", 0, t0 + 1 * m, 1.0), ("B", "s3", 0, t0 + 10 * m, 1.2), ("B", "s4", 0, t0 + 14 * m, 1.5),
        ("C", "s3", 1, t0 + 5 * m, 0.0),
    ]
    p = pd.DataFrame(rows, columns=["trip_id", "stop_id", "direction", "predicted", "delay_min"])
    a = mbta_live.bus_ahead(p)
    assert a.loc["B", "ahead_trip"] == "A"
    assert a.loc["B", "gap_min"] == pytest.approx(8.0)          # at s3, the first stop both still predict
    assert a.loc["B", "prev_bus_delay"] == pytest.approx(4.0)   # leader's current delay
    assert pd.isna(a.loc["A", "gap_min"]) and pd.isna(a.loc["C", "gap_min"])


def test_passed_bus_features_skips_leader_still_en_route():
    """Decision 30: like training, the 'previous bus' is the last bus that has ALREADY passed the target stop."""
    t0 = pd.Timestamp("2026-01-05 08:00", tz="America/New_York")
    m = pd.Timedelta(minutes=1)
    rows = [
        ("Z", "s5", 0, t0 + 5 * m, 7.0),                                   # two ahead: already past s3
        ("A", "s3", 0, t0 + 2 * m, 4.0), ("A", "s5", 0, t0 + 12 * m, 4.5),  # direct leader: not yet at s3
        ("B", "s3", 0, t0 + 10 * m, 1.2), ("B", "s5", 0, t0 + 20 * m, 1.5),
    ]
    p = pd.DataFrame(rows, columns=["trip_id", "stop_id", "direction", "predicted", "delay_min"])
    a = mbta_live.bus_ahead(p)
    assert a.loc["B", "ahead_trip"] == "A" and a.loc["B", "gap_min"] == pytest.approx(8.0)
    r = pd.DataFrame({"trip_id": ["B"], "stop_id": ["s3"], "mbta_predicted": [t0 + 10 * m],
                      "scheduled_headway": [10.0]}).join(a[["leaders"]], on="trip_id")
    out = mbta_live.passed_bus_features(p, r, now=t0)
    # Z passed s3 at about ETA_B(s3) - gap(B,Z)=15 -> t0-5 min, i.e. 5 min ago; headway 10 -> ratio 0.5
    assert out.loc[0, "prev_bus_gap_ratio"] == pytest.approx(0.5)
    assert out.loc[0, "prev_bus_delay"] == pytest.approx(7.0)


# ---------------------------------------------------------------- dashboard pages
@pytest.mark.skipif(not HAVE_DATA, reason="run `python -m src.app_data` first")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, offline):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_DIR / page), default_timeout=180)
    at.run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
