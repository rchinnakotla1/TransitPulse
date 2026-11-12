"""Phase 3: train and compare the models in both prediction modes.

Protocol (per mode):
1. **Tune on train (Jan–Aug) → validation (Sep–Oct).** Hyper-parameters and the decision threshold are chosen here only.
2. **Refit on train + validation (Jan–Oct, 83 % of 2025)** with the chosen hyper-parameters.
3. **Score the test set (Nov–Dec) once.** Threshold-free metrics (ROC-AUC, PR-AUC, Brier) plus P/R/F1 at the frozen threshold,
   with 95 % day-block bootstrap confidence intervals.

Models: B0 historical baseline · Logistic Regression · Decision Tree · Random Forest · XGBoost · LightGBM (cross-check).
Sample sizes: XGBoost uses all rows; LR / DT / RF use a 2 M stratified sample (the learning curve in experiments.py shows why that is enough).

Run: ``python -m src.models.train planning operational``

Fast mode: ``python -m src.models.train planning operational --fast`` skips every hyper-parameter search (Optuna for XGBoost,
the small grids for LR / DT / RF / LightGBM) and re-uses the winning settings frozen in ``reports/best_params.json``.
Same data, same seeds, same settings, so it re-creates the same models and test metrics, only much faster.
A full (non-fast) run updates ``reports/best_params.json`` with its own winners.
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import REPORTS, ROOT
from src.encoding import align_categories, feature_list, linear_preprocessor, load_split, spec, tree_matrix
from src.models.common import best_f1_threshold, day_block_bootstrap, metrics

MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)
PRED_DIR = REPORTS / "preds"; PRED_DIR.mkdir(exist_ok=True)
SAMPLE = 2_000_000
SEED = 42
KEEP = ["service_date", "route_id", "direction_id", "stop_id", "hour", "half_trip_id", "time_point_order",
        "delayed", "delay_min"]
optuna.logging.set_verbosity(optuna.logging.WARNING)
BEST_PARAMS = REPORTS / "best_params.json"
FIXED: dict = {}          # fast mode: {model_name: frozen params} for the current mode, empty = run the search


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_best_params(mode: str) -> dict:
    if not BEST_PARAMS.exists():
        raise SystemExit(f"--fast needs {BEST_PARAMS} (written by a full run). Run without --fast once.")
    return json.loads(BEST_PARAMS.read_text())[mode]


def save_best_params(mode: str, name: str, params: dict, n_valid: int | None = None):
    allp = json.loads(BEST_PARAMS.read_text()) if BEST_PARAMS.exists() else {}
    p = dict(params)
    if n_valid is not None:
        p["n_estimators_valid"] = int(n_valid)
    allp.setdefault(mode, {})[name] = p
    BEST_PARAMS.write_text(json.dumps(allp, indent=2))


def load(mode: str):
    feats = feature_list(mode)
    cols = list(dict.fromkeys(KEEP + feats))
    tr, va, te = (load_split(s, columns=cols) for s in ("train", "valid", "test"))
    align_categories([tr, va, te], spec()["categorical"])
    return feats, tr, va, te


def strat_sample(df, n, seed=SEED):
    return df if len(df) <= n else df.groupby("delayed", group_keys=False).sample(frac=n / len(df), random_state=seed)


def pr_auc(y, p):
    return average_precision_score(y, p)


# ------------------------------------------------------------------------------------------------
# model definitions: each returns (valid_prob_from_train_only_model, refit_fn(train_valid_df) -> predict_fn, params)
# ------------------------------------------------------------------------------------------------
def run_b0(feats, tr, va, mode):
    """Historical average = what agencies effectively do today. Operational: 'already late stays late' rule."""
    def prob(d):
        p = d.hist_stop_hour.values.astype(float)
        if mode == "operational":
            known = d.prev_tp_delayed.notna().values
            p = np.where(known, np.clip(0.03 + 0.92 * d.prev_tp_delayed.fillna(0).values, 0, 1), p)
        return p
    return prob(va), (lambda trva: prob), {"rule": "hist_stop_hour" + (" | prev_tp_delayed" if mode == "operational" else "")}


def run_lr(feats, tr, va, mode):
    s = strat_sample(tr, SAMPLE)
    best = None
    for C in ((FIXED["LR"]["C"],) if "LR" in FIXED else (0.01, 0.1, 1.0)):
        pipe = Pipeline([("prep", linear_preprocessor(feats)),
                         ("lr", LogisticRegression(C=C, max_iter=400, solver="lbfgs"))])
        pipe.fit(s[feats], s.delayed)
        p = pipe.predict_proba(va[feats])[:, 1]
        sc = pr_auc(va.delayed, p); log(f"   LR C={C}: valid PR-AUC {sc:.4f}")
        if best is None or sc > best[0]:
            best = (sc, C, p)

    def refit(trva):
        d = strat_sample(trva, SAMPLE)
        pipe = Pipeline([("prep", linear_preprocessor(feats)), ("lr", LogisticRegression(C=best[1], max_iter=400))])
        pipe.fit(d[feats], d.delayed)
        joblib.dump(pipe, MODEL_DIR / f"LR_{mode}.joblib", compress=3)
        return lambda x: pipe.predict_proba(x[feats])[:, 1]
    return best[2], refit, {"C": best[1]}


def _sk_tree(make, grid, name, feats, tr, va, mode):
    s = strat_sample(tr, SAMPLE)
    Xs, Xv = tree_matrix(s, feats), tree_matrix(va, feats)
    best = None
    if name in FIXED:
        grid = [{"max_depth": FIXED[name]["max_depth"]}]
    for params in grid:
        m = make(**params).fit(Xs, s.delayed)
        p = m.predict_proba(Xv)[:, 1]
        sc = pr_auc(va.delayed, p); log(f"   {name} {params}: valid PR-AUC {sc:.4f}")
        if best is None or sc > best[0]:
            best = (sc, params, p)
        del m

    def refit(trva):
        d = strat_sample(trva, SAMPLE)
        m = make(**best[1]).fit(tree_matrix(d, feats), d.delayed)
        joblib.dump(m, MODEL_DIR / f"{name}_{mode}.joblib", compress=3)
        return lambda x: m.predict_proba(tree_matrix(x, feats))[:, 1]
    return best[2], refit, best[1]


def run_dt(feats, tr, va, mode):
    make = lambda **k: DecisionTreeClassifier(min_samples_leaf=200, random_state=SEED, **k)
    return _sk_tree(make, [{"max_depth": d} for d in (6, 10, 14, 18)], "DT", feats, tr, va, mode)


def run_rf(feats, tr, va, mode):
    make = lambda **k: RandomForestClassifier(n_estimators=150, min_samples_leaf=50, max_features="sqrt",
                                              n_jobs=-1, random_state=SEED, **k)
    return _sk_tree(make, [{"max_depth": d} for d in (14, 22)], "RF", feats, tr, va, mode)


def run_xgb(feats, tr, va, mode, n_trials: int = 15):
    # tuning sample: 1 M train / 300 k valid keeps each Optuna trial to a few minutes (stop_id has ~1,000 levels)
    s = strat_sample(tr, 1_000_000); vs = strat_sample(va, 300_000)
    base = dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, random_state=SEED,
                eval_metric="aucpr", early_stopping_rounds=30, learning_rate=0.1, n_estimators=800)

    def objective(trial):
        t = time.time()
        p = dict(max_depth=trial.suggest_int("max_depth", 4, 10),
                 min_child_weight=trial.suggest_float("min_child_weight", 1, 200, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20, log=True),
                 max_cat_threshold=trial.suggest_int("max_cat_threshold", 16, 128, log=True))
        m = xgb.XGBClassifier(**base, **p).fit(s[feats], s.delayed, eval_set=[(vs[feats], vs.delayed)], verbose=False)
        trial.set_user_attr("best_iteration", m.best_iteration)
        log(f"   trial {trial.number}: PR-AUC {m.best_score:.4f} trees {m.best_iteration} ({time.time() - t:.0f}s) {p}")
        return m.best_score

    if "XGB" in FIXED:
        f = FIXED["XGB"]
        bp = {k: f[k] for k in ("max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda",
                                "max_cat_threshold")}
        n_best = int(f["n_estimators_valid"]) - 1
        log(f"   XGB fast mode: re-using saved params {bp} trees {n_best} (no Optuna search)")
    else:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials)
        bp = study.best_params; n_best = study.best_trial.user_attrs["best_iteration"]
        log(f"   XGB best valid PR-AUC {study.best_value:.4f} params {bp} trees {n_best}")
        study.trials_dataframe().to_csv(REPORTS / f"optuna_xgb_{mode}.csv", index=False)
        save_best_params(mode, "XGB", {**bp, "n_estimators_refit": int((n_best + 1) * 1.15)}, n_valid=n_best + 1)
    # train-only model on ALL training rows for validation predictions (threshold choice)
    fixed = {k: v for k, v in base.items() if k not in ("early_stopping_rounds", "n_estimators")}
    m = xgb.XGBClassifier(**fixed, **bp, n_estimators=n_best + 1).fit(tr[feats], tr.delayed)
    pv = m.predict_proba(va[feats])[:, 1]

    def refit(trva):
        n = int((n_best + 1) * 1.15)            # ~15 % more data -> slightly more trees
        mm = xgb.XGBClassifier(**fixed, **bp, n_estimators=n).fit(trva[feats], trva.delayed)
        mm.save_model(MODEL_DIR / f"XGB_{mode}.json")
        return lambda x: mm.predict_proba(x[feats])[:, 1]
    return pv, refit, {**bp, "n_estimators_refit": int((n_best + 1) * 1.15)}


def run_lgb(feats, tr, va, mode):
    """LightGBM: a second gradient-boosting library (leaf-wise growth, native categoricals) to check that the XGBoost
    result is not library-specific. Small grid instead of Optuna: the goal is a fair cross-check, not a second tuning race."""
    import lightgbm as lgb
    s = strat_sample(tr, 1_000_000); vs = strat_sample(va, 300_000)
    base = dict(learning_rate=0.1, n_estimators=800, subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                cat_smooth=20, min_data_per_group=200, n_jobs=-1, random_state=SEED, verbose=-1)
    best = None
    if "LGB" in FIXED:
        f = FIXED["LGB"]
        bp, n_best = {"num_leaves": f["num_leaves"], "min_child_samples": f["min_child_samples"]}, int(f["n_estimators_valid"])
        log(f"   LGB fast mode: re-using saved params {bp} trees {n_best} (no grid search)")
    else:
        for p in ({"num_leaves": 63, "min_child_samples": 200}, {"num_leaves": 255, "min_child_samples": 200},
                  {"num_leaves": 127, "min_child_samples": 50}):
            t = time.time()
            m = lgb.LGBMClassifier(**base, **p).fit(s[feats], s.delayed, eval_set=[(vs[feats], vs.delayed)],
                                                    eval_metric="average_precision",
                                                    callbacks=[lgb.early_stopping(30, verbose=False)])
            sc = m.best_score_["valid_0"]["average_precision"]; n = m.best_iteration_
            log(f"   LGB {p}: valid PR-AUC {sc:.4f} trees {n} ({time.time() - t:.0f}s)")
            if best is None or sc > best[0]:
                best = (sc, p, n)
        bp, n_best = best[1], best[2]
        save_best_params(mode, "LGB", {**bp, "n_estimators_refit": int(n_best * 1.15)}, n_valid=n_best)
    fixed = {**base, "n_estimators": n_best}
    m = lgb.LGBMClassifier(**fixed, **bp).fit(tr[feats], tr.delayed)
    pv = m.predict_proba(va[feats])[:, 1]

    def refit(trva):
        mm = lgb.LGBMClassifier(**{**fixed, "n_estimators": int(n_best * 1.15)}, **bp).fit(trva[feats], trva.delayed)
        mm.booster_.save_model(str(MODEL_DIR / f"LGB_{mode}.txt"))
        return lambda x: mm.predict_proba(x[feats])[:, 1]
    return pv, refit, {**bp, "n_estimators_refit": int(n_best * 1.15)}


RUNNERS = {"B0": run_b0, "LR": run_lr, "DT": run_dt, "RF": run_rf, "XGB": run_xgb, "LGB": run_lgb}


def main(mode: str, models=tuple(RUNNERS), fast: bool = False):
    log(f"=== mode: {mode} ===" + ("  (FAST: saved hyper-parameters, no search)" if fast else ""))
    FIXED.clear()
    if fast:
        FIXED.update(load_best_params(mode))
    feats, tr, va, te = load(mode)
    # memory: keep one concatenated frame and use row-slices of it as train / valid (copy-on-write views)
    ntr = len(tr)
    trva = pd.concat([tr, va], ignore_index=True); del tr, va
    tr, va = trva.iloc[:ntr], trva.iloc[ntr:].reset_index(drop=True)
    log(f"train {len(tr):,} valid {len(va):,} test {len(te):,} features {len(feats)}")
    # re-running a subset of models keeps the other models' results
    mpath, vpath, tpath = REPORTS / f"metrics_{mode}.csv", PRED_DIR / f"valid_{mode}.parquet", PRED_DIR / f"test_{mode}.parquet"
    vpred = va[["service_date", "route_id", "delayed"]].copy()
    tpred = te[KEEP].copy()
    rows = []
    if mpath.exists():
        rows = [r for r in pd.read_csv(mpath).to_dict("records") if r["model"] not in models]
    for path, frame in ((vpath, vpred), (tpath, tpred)):
        if path.exists():
            old = pd.read_parquet(path)
            if len(old) == len(frame):
                for c in old.columns:
                    if c.startswith("p_") and c[2:] not in models:
                        frame[c] = old[c].values
    for name in models:
        t0 = time.time(); log(f"-- {name}")
        pv, refit, params = RUNNERS[name](feats, tr, va, mode)
        if not fast and name in ("B0", "LR", "DT", "RF"):
            save_best_params(mode, name, params)
        thr = best_f1_threshold(va.delayed, pv)
        predict = refit(trva)
        pt = predict(te)
        secs = time.time() - t0
        vpred[f"p_{name}"] = pv.astype("float32"); tpred[f"p_{name}"] = pt.astype("float32")
        mv, mt = metrics(va.delayed, pv, thr), metrics(te.delayed, pt, thr)
        ci_auc = day_block_bootstrap(te.delayed, pt, te.service_date)
        ci_pr = day_block_bootstrap(te.delayed, pt, te.service_date, stat=pr_auc)
        rows.append({"mode": mode, "model": name, "split": "valid", **mv, "seconds": secs, "params": json.dumps(params)})
        rows.append({"mode": mode, "model": name, "split": "test", **mt, "roc_auc_lo": ci_auc[0], "roc_auc_hi": ci_auc[1],
                     "pr_auc_lo": ci_pr[0], "pr_auc_hi": ci_pr[1], "seconds": secs, "params": json.dumps(params)})
        log(f"   {name}: valid AUC {mv['roc_auc']:.4f} | TEST AUC {mt['roc_auc']:.4f} [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}] "
            f"PR-AUC {mt['pr_auc']:.4f} F1 {mt['f1']:.4f} ({secs:.0f}s)")
        pd.DataFrame(rows).to_csv(mpath, index=False)
        vpred.to_parquet(vpath, index=False)
        tpred.to_parquet(tpath, index=False)
    log("done")


if __name__ == "__main__":
    modes = [a for a in sys.argv[1:] if a in ("planning", "operational")] or ["planning", "operational"]
    only = [a for a in sys.argv[1:] if a in RUNNERS] or list(RUNNERS)
    fast = "--fast" in sys.argv
    for md in modes:
        main(md, only, fast=fast)
