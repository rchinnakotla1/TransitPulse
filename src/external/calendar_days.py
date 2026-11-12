"""Step 5b — Daily calendar: public holidays, school days, university terms.

Sources : `holidays` package (US federal + Massachusetts state holidays, e.g. Patriots' Day)
          Boston Public Schools (BPS) official calendars 2023-24, 2024-25, 2025-26
          Major Boston universities (BU / Northeastern / Harvard) typical term windows
Why     : EDA showed delay rate drops on holidays (19.8 % vs 28.5 %), in July-August and in late
          December -> school & university travel is a plausible driver of crowding and traffic.
Join    : service_date
Output  : data/external/calendar_daily.parquet
"""
import holidays
import pandas as pd

from src.config import EXTERNAL_DIR

# ---- Boston Public Schools (grades 1-12) ------------------------------------
# Sources: bostonpublicschools.org district calendars (2023-24, 2024-25, 2025-26)
BPS_TERMS = [("2023-09-07", "2024-06-26"), ("2024-09-05", "2025-06-23"), ("2025-09-04", "2026-06-26")]
BPS_BREAKS = [
    ("2023-12-22", "2024-01-01"),   # winter recess 2023-24
    ("2024-02-19", "2024-02-23"),   # February vacation 2024
    ("2024-04-15", "2024-04-19"),   # April vacation 2024
    ("2024-12-23", "2025-01-01"),   # winter recess 2024-25
    ("2025-02-17", "2025-02-21"),   # February vacation 2025
    ("2025-04-21", "2025-04-25"),   # April vacation 2025
    ("2025-12-22", "2026-01-02"),   # winter recess 2025-26
]
# ---- Universities (approximate common term windows of large Boston universities) ----
UNI_TERMS = [("2024-01-16", "2024-05-10"), ("2024-09-03", "2024-12-20"),
             ("2025-01-21", "2025-05-09"), ("2025-09-02", "2025-12-19")]


def _in_ranges(dates: pd.Series, ranges) -> pd.Series:
    m = pd.Series(False, index=dates.index)
    for a, b in ranges:
        m |= dates.between(pd.Timestamp(a), pd.Timestamp(b))
    return m


def build(start="2024-01-01", end="2025-12-31") -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    hol = holidays.US(state="MA", years=range(2024, 2027))
    d["is_holiday"] = d["date"].dt.date.map(lambda x: x in hol).astype("int8")
    d["holiday_name"] = d["date"].dt.date.map(lambda x: hol.get(x, ""))
    weekday = d["date"].dt.day_of_week < 5
    d["is_school_day"] = (_in_ranges(d["date"], BPS_TERMS) & ~_in_ranges(d["date"], BPS_BREAKS)
                          & weekday & (d["is_holiday"] == 0)).astype("int8")
    d["is_university_term"] = _in_ranges(d["date"], UNI_TERMS).astype("int8")

    # Distance to nearest holiday (captures pre-holiday travel / long weekends)
    hd = pd.Series(sorted(pd.Timestamp(x) for x in hol if pd.Timestamp(start) - pd.Timedelta(days=30)
                          <= pd.Timestamp(x) <= pd.Timestamp(end) + pd.Timedelta(days=30)))
    d["days_to_holiday"] = d["date"].map(lambda x: int((hd - x).abs().dt.days.min())).astype("int16")
    d["holiday_week"] = (d["days_to_holiday"] <= 3).astype("int8")
    return d


def main():
    d = build()
    out = EXTERNAL_DIR / "calendar_daily.parquet"
    d.to_parquet(out, index=False)
    print(d.groupby(d.date.dt.year)[["is_holiday", "is_school_day", "is_university_term"]].sum())
    print("->", out)


if __name__ == "__main__":
    main()
