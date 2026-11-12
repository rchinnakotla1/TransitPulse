"""Phase 5b: delay in MINUTES (regression) with an 80 % prediction interval.

Why: riders and dispatchers ask "how late?", not only "late or not?".
* point model: XGBoost with absolute-error objective (predicts the conditional *median*, robust to the long right
  tail of delays - a few 60-min incidents should not drag every prediction up)
* interval: XGBoost multi-quantile model (alpha 0.1 / 0.9) -> "expected 2-9 min late"; we check empirical coverage
* baselines: (i) stop's usual delay at this hour (``hist_delay_stop_hour``), (ii) operational only:
  "delay stays the same as at the previous timepoint" (persistence)
Target clipped to [-10, 60] min for training (data-error guard; evaluation uses the same clip).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.extensions.common import EXT_DIR, MODEL_DIR, load_sets, log, xgb_params

CLIP = (-10, 60)


def _scores(name, mode, y, pred):
    err = np.abs(y - pred)
    return {"mode": mode, "model": name, "MAE": mean_absolute_error(y, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "median_AE": float(np.median(err)),
            "within_2min_%": float((err <= 2).mean() * 100), "within_5min_%": float((err <= 5).mean() * 100)}


def run(mode: str, n_train: int = 2_000_000) -> pd.DataFrame:
    feats, tr, va, te = load_sets(mode, n_train=n_train)
    y = {k: d.delay_min.clip(*CLIP).values for k, d in (("tr", tr), ("va", va), ("te", te))}
    base = {k: v for k, v in xgb_params(mode).items()}
    m = xgb.XGBRegressor(**base, objective="reg:absoluteerror", eval_metric="mae").fit(
        tr[feats], y["tr"], eval_set=[(va[feats], y["va"])], verbose=False)
    m.save_model(MODEL_DIR / f"REG_{mode}.json")
    pred = m.predict(te[feats])
    rows = [_scores("B0 stop-hour usual delay", mode, y["te"], te.hist_delay_stop_hour.fillna(0).values)]
    if mode == "operational":
        rows.append(_scores("B1 persistence (prev timepoint)", mode, y["te"], te.prev_tp_delay.fillna(0).values))
    rows.append(_scores("XGB median", mode, y["te"], pred))

    # 80 % interval with a single multi-quantile model (smaller sample: quantile loss is slower)
    qi = tr.sample(min(1_000_000, len(tr)), random_state=1).index     # random rows (stratified sample is sorted by class)
    q = xgb.XGBRegressor(**{**base, "n_estimators": 400}, objective="reg:quantileerror",
                         quantile_alpha=np.array([0.1, 0.9])).fit(
        tr.loc[qi, feats], tr.loc[qi, "delay_min"].clip(*CLIP).values, eval_set=[(va[feats], y["va"])], verbose=False)
    q.save_model(MODEL_DIR / f"REGQ_{mode}.json")
    lo_hi = np.sort(q.predict(te[feats]), axis=1)
    cover = float(((y["te"] >= lo_hi[:, 0]) & (y["te"] <= lo_hi[:, 1])).mean() * 100)
    width = float(np.median(lo_hi[:, 1] - lo_hi[:, 0]))
    res = pd.DataFrame(rows)
    res["interval80_coverage_%"] = np.nan; res["interval80_median_width_min"] = np.nan
    res.loc[res.model == "XGB median", ["interval80_coverage_%", "interval80_median_width_min"]] = [cover, width]

    # error by true-delay band: where is the regressor useful?
    band = pd.cut(y["te"], [-10, 1, 5, 10, 20, 60], labels=["<1", "1-5", "5-10", "10-20", "20+"], include_lowest=True)
    by = pd.DataFrame({"band": band, "abs_err": np.abs(y["te"] - pred), "bias": pred - y["te"]}).groupby(
        "band", observed=True).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"), bias=("bias", "mean"))
    by.to_csv(EXT_DIR / f"regression_by_band_{mode}.csv")
    sample = te[["route_id", "hour", "delay_min"]].assign(pred=pred, lo=lo_hi[:, 0], hi=lo_hi[:, 1]).sample(
        20_000, random_state=0)
    sample.to_parquet(EXT_DIR / f"regression_sample_{mode}.parquet", index=False)
    log(f"regression {mode}:\n{res.round(3).to_string(index=False)}\n{by.round(2)}")
    return res


if __name__ == "__main__":
    out = pd.concat([run(md) for md in ("planning", "operational")])
    out.to_csv(EXT_DIR / "regression_metrics.csv", index=False)
