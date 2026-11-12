"""Leakage tests for historical features (synthetic data, fast).

Core property: a feature for service date D must not change if we alter any label on day D or later.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import expanding_rate, rolling_rate, add_operational, trip_mask, sort_trips


@pytest.fixture
def rows():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    n = 6000
    df = pd.DataFrame({
        "service_date": rng.choice(dates, n),
        "route_id": rng.choice(["1", "28", "39"], n),
        "hour": rng.integers(5, 23, n).astype("int8"),
    })
    df["delayed"] = rng.binomial(1, 0.3, n)
    df["delay_min"] = rng.normal(2, 4, n)
    return df


def test_expanding_rate_ignores_today_and_future(rows):
    D = pd.Timestamp("2025-02-01")
    base = expanding_rate(rows, ["route_id"], "delayed", "r", 0.3)
    future = rows.copy()
    future.loc[future.service_date >= D, "delayed"] = 1 - future.loc[future.service_date >= D, "delayed"]
    pert = expanding_rate(future, ["route_id"], "delayed", "r", 0.3)
    a = base[base.service_date <= D].sort_values(["route_id", "service_date"]).r.values
    b = pert[pert.service_date <= D].sort_values(["route_id", "service_date"]).r.values
    np.testing.assert_allclose(a, b)
    # and the feature does react after D (the perturbation is visible from D+1 on)
    a2 = base[base.service_date > D].r.values
    b2 = pert[pert.service_date > D].r.values
    assert not np.allclose(a2, b2)


def test_expanding_rate_matches_manual(rows):
    m, D, key = 30, pd.Timestamp("2025-01-20"), "28"
    tbl = expanding_rate(rows, ["route_id"], "delayed", "r", 0.3, m=m)
    past = rows[rows.service_date < D]
    prior = past.delayed.mean()
    k = past[past.route_id == key]
    expected = (k.delayed.sum() + m * prior) / (len(k) + m)
    got = tbl[(tbl.route_id == key) & (tbl.service_date == D)].r.iloc[0]
    assert got == pytest.approx(expected, rel=1e-5)


def test_first_day_uses_prior_only(rows):
    tbl = expanding_rate(rows, ["route_id"], "delayed", "r", 0.3)
    first = tbl[tbl.service_date == tbl.service_date.min()]
    assert np.allclose(first.r, 0.3)


def test_rolling_rate_ignores_today_and_future(rows):
    D = pd.Timestamp("2025-02-10")
    base = rolling_rate(rows, ["route_id"], "x", days=7)
    future = rows.copy()
    future.loc[future.service_date >= D, "delayed"] = 1
    pert = rolling_rate(future, ["route_id"], "x", days=7)
    a = base[base.service_date <= D].sort_values(["route_id", "service_date"]).x.values
    b = pert[pert.service_date <= D].sort_values(["route_id", "service_date"]).x.values
    np.testing.assert_allclose(a, b)


def test_rolling_window_length(rows):
    """With days=7, a label 8+ days ago must not influence the feature."""
    D = pd.Timestamp("2025-02-10")
    base = rolling_rate(rows, ["route_id"], "x", days=7)
    old = rows.copy()
    sel = old.service_date < D - pd.Timedelta(days=7)
    old.loc[sel, "delayed"] = 1 - old.loc[sel, "delayed"]
    pert = rolling_rate(old, ["route_id"], "x", days=7)
    # the smoothing prior uses all history, so compare the un-smoothed part via m=0
    b0 = rolling_rate(rows, ["route_id"], "x", days=7, m=0)
    p0 = rolling_rate(old, ["route_id"], "x", days=7, m=0)
    a = b0[b0.service_date == D].sort_values("route_id").x.values
    b = p0[p0.service_date == D].sort_values("route_id").x.values
    np.testing.assert_allclose(a, b)


def _trip_frame():
    # two trips on the same route/direction passing the same 3 timepoints; trip B is 10 min behind trip A
    recs = []
    for trip, offset, delays in [("A", 0, [1, 3, 6]), ("B", 600, [0, 8, 12])]:
        for i, d in enumerate(delays):
            sched = 8 * 3600 + offset + i * 900
            recs.append(dict(service_date=pd.Timestamp("2025-03-03"), route_id="28", direction_id="Inbound",
                             half_trip_id=trip, stop_id=f"s{i}", time_point_order=float(i + 1),
                             sched_sec=float(sched), actual_sec=float(sched + d * 60), delay_min=float(d),
                             scheduled_headway=10.0))
    return pd.DataFrame(recs)


def test_operational_features_use_only_the_past():
    df = sort_trips(_trip_frame())
    same = trip_mask(df)
    out = add_operational(df.copy(), same)
    b = out[out.half_trip_id == "B"].set_index("stop_id")
    # previous timepoint delay of the same trip
    assert np.isnan(b.loc["s0", "prev_tp_delay"]) and b.loc["s1", "prev_tp_delay"] == 0 and b.loc["s2", "prev_tp_delay"] == 8
    # B leaves s1 at 08:25+8 = 08:33 -> trip A passed s2 at 08:30+6 = 08:36 -> NOT yet known -> no prev bus at s2
    assert np.isnan(b.loc["s2", "prev_bus_delay"])
    # at s1, B departs s0 at 08:10; A passed s1 at 08:18 -> not known either
    assert np.isnan(b.loc["s1", "prev_bus_delay"])
    # at s0 (prediction 1 min before 08:10) A passed s0 at 08:01 -> known, delay 1
    assert b.loc["s0", "prev_bus_delay"] == 1
    # own arrival never used: prev_bus for trip A must be NaN everywhere (no earlier bus)
    assert out[out.half_trip_id == "A"].prev_bus_delay.isna().all()


def test_operational_never_reads_own_row():
    df = sort_trips(_trip_frame())
    df.loc[0, "actual_sec"] = df.loc[0, "sched_sec"] - 300   # very early departure at first stop
    out = add_operational(df.copy(), trip_mask(df))
    assert np.isnan(out.loc[0, "prev_bus_delay"])
