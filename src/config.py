"""Central project configuration: paths, scope, and labelling thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_DIR = DATA / "raw" / "mbta_2025" / "MBTA_Bus_Arrival_Departure_Times_2025"


def raw_dir(year: int) -> Path:
    return DATA / "raw" / f"mbta_{year}" / f"MBTA_Bus_Arrival_Departure_Times_{year}"


EXTERNAL_DIR = DATA / "external"        # weather, calendar, events, GTFS, ridership
INTERIM_DIR = DATA / "interim"          # one parquet per month, all routes, typed
PROCESSED_DIR = DATA / "processed"      # cleaned, scoped dataset used for EDA / modelling
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
LOG_DIR = ROOT / "logs"

for d in (EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Scope -----------------------------------------------------------------
CITY, MODE = "Boston (MBTA)", "Bus"
YEARS = [2024, 2025]                    # 2024 = history + drift test, 2025 = main modelling year
YEAR = 2025
TIMEZONE = "America/New_York"
BOSTON_LATLON = (42.3656, -71.0096)     # Logan Airport weather station

# MBTA "key bus routes" (high-frequency, mostly headway-managed) ...
KEY_ROUTES = ["1", "15", "22", "23", "28", "32", "39", "57", "66",
              "71", "73", "77", "111", "116", "117"]
# ... plus the N busiest schedule-managed routes are added automatically in clean.py
N_SCHEDULE_ROUTES = 5
# Final scope fixed after Step 3 on 2025 data, re-used for every year so years are comparable
SCOPED_ROUTES = ["1", "15", "22", "23", "28", "32", "39", "57", "66", "71", "73", "77",
                 "111", "116", "34E", "70", "240", "8", "44"]

# ---- Target definition -----------------------------------------------------
DELAY_THRESHOLD_SEC = 5 * 60            # late by > 5 min  => "delayed" (label = 1)
# Severity bands (extension: multi-class)
SEVERITY_BINS = [-float("inf"), 60, 5 * 60, 10 * 60, float("inf")]
SEVERITY_LABELS = ["on_time", "minor", "moderate", "major"]

# ---- Cleaning rules --------------------------------------------------------
MAX_EARLY_SEC = -30 * 60                # earlier than 30 min => data error
MAX_LATE_SEC = 120 * 60                 # later than 2 h      => data error / mis-match

# ---- Time-based split (no shuffling!) ---------------------------------------
# 2024 is used only for historical features and the drift experiment.
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-08-31"                # Jan–Aug 2025 train
VALID_END = "2025-10-31"                # Sep–Oct validation
                                        # Nov–Dec test
