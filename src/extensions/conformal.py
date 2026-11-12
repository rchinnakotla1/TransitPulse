"""Phase 5b+: conformal calibration of the 80 % delay interval (CQR, Romano et al. 2019).

Why: the quantile model's 80 % interval covered only 75.5 % (planning) / 78.3 % (operational) of test delays: quantile
regression is known to under-cover when the model is fitted with early stopping and on a sub-sample.
How (conformalized quantile regression): on the VALIDATION period (Sep-Oct, never used for fitting the quantile trees, only
for early stopping) compute the conformity score E = max(lo - y, y - hi). The 80 % empirical quantile Q of E is added to both
ends: [lo - Q, hi + Q]. Q < 0 shrinks an over-wide interval. Guarantee: marginal coverage >= 80 % if validation and test are
exchangeable; they are not exactly (seasons differ), so we check on Nov-Dec.
Output: reports/ext/regression_conformal.csv (coverage and width before / after, per mode and per route).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.extensions.common import EXT_DIR, MODEL_DIR, load_sets, log
from src.extensions.regression import CLIP

ALPHA = 0.20


def run(mode: str) -> tuple[dict, pd.DataFrame]:
    feats, _, va, te = load_sets(mode, n_train=1000)          # training rows are not needed
    q = xgb.XGBRegressor(); q.load_model(MODEL_DIR / f"REGQ_{mode}.json")
    yv, yt = va.delay_min.clip(*CLIP).values, te.delay_min.clip(*CLIP).values
    lv, lt = np.sort(q.predict(va[feats]), axis=1), np.sort(q.predict(te[feats]), axis=1)
    e = np.maximum(lv[:, 0] - yv, yv - lv[:, 1])
    n = len(e)
    Q = float(np.quantile(e, min(1, np.ceil((n + 1) * (1 - ALPHA)) / n)))
    lo, hi = lt[:, 0] - Q, lt[:, 1] + Q
    raw_in = (yt >= lt[:, 0]) & (yt <= lt[:, 1]); cal_in = (yt >= lo) & (yt <= hi)
    res = {"mode": mode, "Q_min": Q, "valid_raw_coverage_%": float(((yv >= lv[:, 0]) & (yv <= lv[:, 1])).mean() * 100),
           "test_raw_coverage_%": raw_in.mean() * 100, "test_raw_median_width": float(np.median(lt[:, 1] - lt[:, 0])),
           "test_cal_coverage_%": cal_in.mean() * 100, "test_cal_median_width": float(np.median(hi - lo))}
    by = pd.DataFrame({"route_id": te.route_id.astype(str).values, "raw": raw_in, "cal": cal_in}).groupby("route_id").mean() * 100
    by["mode"] = mode
    log(res)
    return res, by.reset_index()


if __name__ == "__main__":
    out = [run(m) for m in ("planning", "operational")]
    pd.DataFrame([o[0] for o in out]).to_csv(EXT_DIR / "regression_conformal.csv", index=False)
    pd.concat([o[1] for o in out]).to_csv(EXT_DIR / "regression_conformal_by_route.csv", index=False)
    print(pd.DataFrame([o[0] for o in out]).round(3).T)
