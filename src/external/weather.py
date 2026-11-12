"""Step 5a — Hourly weather for Boston (Logan Airport), 2024-2025.

Source  : Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis, ECMWF). Free, no key.
Why     : Rain and snow slow traffic, lengthen boarding and reduce visibility. EDA showed isolated
          "spike days" in delay that calendar features cannot explain.
Join    : service-local date + hour  ->  every stop event (scheduled time).
Output  : data/external/weather_hourly.parquet

Run:  python -m src.external.weather
"""
import pandas as pd
import requests

from src.config import BOSTON_LATLON, EXTERNAL_DIR, TIMEZONE

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "precipitation", "rain", "snowfall", "snow_depth",
        "wind_speed_10m", "wind_gusts_10m", "weather_code"]   # visibility is not provided by the archive (all null)


def fetch(start="2024-01-01", end="2025-12-31") -> pd.DataFrame:
    params = dict(latitude=BOSTON_LATLON[0], longitude=BOSTON_LATLON[1], start_date=start,
                  end_date=end, hourly=",".join(VARS), timezone=TIMEZONE,
                  precipitation_unit="mm", wind_speed_unit="kmh")
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    df["precip_3h"] = df["precipitation"].rolling(3, min_periods=1).sum()
    df["snow_24h"] = df["snowfall"].rolling(24, min_periods=1).sum()
    df["heavy_rain"] = (df["precipitation"] >= 4).astype("int8")        # >= 4 mm/h
    df["is_snowing"] = (df["snowfall"] > 0).astype("int8")
    df["freezing"] = (df["temperature_2m"] <= 0).astype("int8")
    df["date"] = df["time"].dt.normalize()
    df["hour"] = df["time"].dt.hour.astype("int8")
    daily_snow = df.groupby("date")["snowfall"].transform("sum")
    df["is_snow_day"] = (daily_snow >= 2.5).astype("int8")                # >= 2.5 cm/day (~1 inch)
    return df


def main():
    df = derive(fetch())
    # DST "fall back" creates a duplicated local hour; keep the first so (date, hour) is unique
    df = df.drop_duplicates(subset=["date", "hour"], keep="first")
    out = EXTERNAL_DIR / "weather_hourly.parquet"
    df.to_parquet(out, index=False)
    print(df.describe().T.round(2).to_string())
    print(f"{len(df):,} hourly rows ->", out)


if __name__ == "__main__":
    main()
