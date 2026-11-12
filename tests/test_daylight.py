"""Daylight feature checks against known Boston sun positions."""
import pandas as pd
import pytest

from src.features import solar_elevation


def elev(s):
    return float(solar_elevation(pd.Series(pd.to_datetime([s]))).iloc[0])


def test_summer_solstice_noon():
    assert elev("2025-06-21 12:42") == pytest.approx(71.1, abs=1.0)    # solar noon (EDT), 90 - 42.4 + 23.4


def test_winter_solstice_noon():
    assert elev("2025-12-21 11:44") == pytest.approx(24.2, abs=1.0)    # solar noon (EST)


def test_dst_change_makes_17h_dark():
    """Same clock time, opposite light: before vs after the 2 Nov 2025 clock change."""
    assert elev("2025-10-31 17:00") > 0          # sunset ~17:38 EDT
    assert elev("2025-11-03 17:00") < 0          # sunset ~16:33 EST


def test_night_is_negative():
    assert elev("2025-03-01 02:00") < -30
