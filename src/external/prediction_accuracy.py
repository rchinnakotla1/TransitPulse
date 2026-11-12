"""Step 5f — MBTA's own bus prediction accuracy (benchmark context).

Source  : MBTA OPMI "Rapid Transit and Bus Prediction Accuracy Data" (weekly, by prediction horizon bin).
Why     : Shows how accurate the countdown clocks riders already see are. We report it next to our
          model to position the work. It is NOT a like-for-like comparison: MBTA predicts arrival time
          (minutes) at short horizons, we predict the risk of a > 5 min delay vs schedule.
Output  : data/external/mbta_bus_prediction_accuracy.parquet
"""
import pandas as pd

from src.config import EXTERNAL_DIR


def main():
    df = pd.read_csv(EXTERNAL_DIR / "prediction_accuracy.csv", parse_dates=["weekly"])
    bus = df[(df["mode"] == "bus") & (df.weekly.between("2024-01-01", "2025-12-31"))].copy()
    bus["accuracy"] = bus.num_accurate_predictions / bus.num_predictions
    bus.to_parquet(EXTERNAL_DIR / "mbta_bus_prediction_accuracy.parquet", index=False)
    s = bus.groupby([bus.weekly.dt.year, "bin"]).apply(
        lambda g: g.num_accurate_predictions.sum() / g.num_predictions.sum()).unstack().round(3)
    print(s)


if __name__ == "__main__":
    main()
