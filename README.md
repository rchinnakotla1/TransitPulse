# TransitPulse: Predicting & Explaining MBTA Bus Delays

Graduate project: a machine-learning system that predicts whether a bus will be **> 5 minutes late** at a timepoint stop,
explains *why* (SHAP + LLM narratives), and presents it to transit-agency users (dispatchers, planners, performance team,
rider communications) in a Streamlit website with a live mode.

| | |
|---|---|
| City / mode | Boston (MBTA) · Bus |
| Years | 2025 (model) + 2024 (history seed, drift study) |
| Core source | [MBTA Bus Arrival Departure Times](https://mbta-massdot.opendata.arcgis.com/) (official, open). See [docs/01](docs/01_data_source_selection.md) |
| External sources | GTFS schedules, hourly weather, holidays + school/university calendars, Red Sox/Bruins/citywide events, passenger load (APC), MBTA prediction accuracy. See [docs/03](docs/03_external_data_integration.md) |
| Scope | 19 routes: 14 key routes (1, 15, 22, 23, 28, 32, 39, 57, 66, 71, 73, 77, 111, 116) + 34E, 70, 240, 8, 44 |
| Size | 2025: 9.82 M cleaned stop events (28.8 % delayed) · 2024: 9.26 M (37.3 % delayed) |
| Target | `delayed = 1` if actual − scheduled > 300 s |

## Quick start (new Windows PC)

1. `git clone https://github.com/rchinnakotla1/TransitPulse.git`
2. Double-click **`setup.bat`** (one time). It installs Python 3.13 if missing, creates `.venv`, installs the exact
   library versions from `requirements-lock.txt`, and creates `.env` from `.env.example`.
3. Open `.env` in Notepad and paste your `OPENAI_API_KEY` (and optionally `MBTA_API_KEY`).
4. Double-click **`run.bat`**. The website opens at http://localhost:8501.

No retraining is needed: the deployed XGBoost models (`models/XGB_*.json`) and all app data (`reports/`) are in the repo.

## Re-create everything from scratch (double-click scripts, no code editor)

Everything needed is in this repository: code, tests, notebooks, results, the tuned model settings and the inputs that
were downloaded by hand. The large data (~10 GB) and the other models are re-created by the scripts below.

| Script | What it does | Approx. time |
|---|---|---|
| **`run_all_steps.bat`** | runs steps 1-9 in order (asks Fast / Full training) | several hours |
| `steps\1_download_data.bat` | downloads MBTA bus data 2024 + 2025 (~720 MB, checked against the exact files used) and unpacks `inputs/` | 5-10 min |
| `steps\2_prepare_data.bat` | ingest, clean, external data (weather, GTFS, events, calendar, ridership, prediction accuracy), merge, features | ~1 h |
| `steps\3_run_tests.bat` | pytest: leakage, features, joins, explainability, website | few min |
| `steps\4_train_models.bat` | 6 models x 2 modes. **F = Fast**: re-uses the tuned settings in `reports/best_params.json` (same results, no search). **S = Full**: repeats the Optuna / grid search | Fast ≈ 1 h, Full ≈ 2.5-3 h |
| `steps\5_explain_and_experiments.bat` | learning curve, ablation, drift, SHAP, surrogate rules, LLM narratives | ~1.5 h |
| `steps\6_extensions.bat` | propagation, chronic segments, severity, regression, conformal, bunching | ~1 h |
| `steps\7_notebooks.bat` | rebuilds and executes notebooks 02-07 with outputs and figures (for presenting each step) | ~30 min |
| `steps\8_build_app_data.bat` | website tables (`reports/app`) | few min |
| `steps\9_build_thesis.bat` | `docs/report/TransitPulse_Thesis.docx` + `.pdf` | few min |

Requirements: Windows, internet, ~25 GB free disk, 16 GB RAM. Each step writes `logs/pipeline_<step>.log`. If a step
fails, fix the cause and re-run that step's script (or `python pipeline.py all --from <step>`). Same commands without
the .bat files: `python pipeline.py list`.

`inputs/` holds the two sources that cannot be downloaded by script (MBTA OPMI portals): bus ridership Fall 2023 + Fall
2024 and the MBTA prediction-accuracy CSV.

## Project structure

```
Roopa Project/
├── data/                       (git-ignored)
│   ├── raw/mbta_{2024,2025}/   downloaded monthly CSVs
│   ├── interim/                typed monthly parquet (all routes)
│   ├── external/               weather, calendar, events, GTFS, ridership, prediction accuracy
│   └── processed/              bus_{year}_clean.parquet, bus_{year}_enriched.parquet
├── src/
│   ├── config.py               paths, scope, thresholds, split dates
│   ├── ingest.py               CSV → parquet, time parsing, delay, raw quality log
│   ├── clean.py                scoping, cleaning funnel, labels
│   ├── external/               one module per external source (weather, calendar_days, events, gtfs, ridership, prediction_accuracy)
│   ├── merge_external.py       joins everything, asserts row count, writes coverage audit
│   ├── features.py             Phase 2: leakage-safe features, planning/operational modes, time split
│   ├── encoding.py             model-specific encodings, feature lists, split loader
│   ├── models/                 Phase 3: train.py (6 models x 2 modes, Optuna, bootstrap CIs), experiments.py
│   ├── xai/                    Phase 4: SHAP, surrogate rules, what-if, OpenAI narrator + fact check + key-health
│   ├── extensions/             Phase 5: propagation, chronic, severity, regression, conformal, bunching
│   ├── app_data.py             Phase 6: small pre-computed tables for the website (reports/app/)
│   ├── live/mbta_live.py       Phase 7: MBTA V3 API + Open-Meteo live feature rows and scoring
│   └── report/build_thesis.py  Phase 8: docs/*.md -> Word + PDF thesis
├── app/                        Streamlit website: Home.py + 8 pages (pages/1_..8_)
├── notebooks/
│   ├── build_*.py, *_findings.py  notebook generators (narrative + code + verified findings)
│   ├── 02_EDA.ipynb            cleaning & exploratory analysis
│   ├── 03_External_Data.ipynb  external data integration & validation
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Models.ipynb · 06_Explainability.ipynb · 07_Extensions.ipynb
├── models/                     trained models (only XGB_*.json in git; the rest re-created by steps\4_train_models.bat)
├── inputs/                     manually downloaded inputs bundled for reproducibility (ridership, prediction accuracy)
├── steps/                      double-click scripts for each pipeline step (see above); pipeline.py runs them
├── reports/                    CSV results, xai/, ext/, app/, figures/ (eda_, p1_ … p5_)
├── docs/                       thesis chapters 00–09, 99 future work, decision_log.md; report/ = Word + PDF
├── logs/                       run logs, llm.log (API key problems), live.log
├── tests/                      leakage, feature, join, XAI, narrator and website/live tests (pytest, 62 tests)
├── .env                        OPENAI_API_KEY, OPENAI_MODEL, MBTA_API_KEY (git-ignored; see .env.example)
└── requirements.txt
```

## Reproduce (manual commands, same as the scripts above)

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
# 1. core data (2025 item id 924df13d845f4907bb6a6c3ed380d57a, 2024 item id 96c77138c3144906bce93d0257531b6a)
Invoke-WebRequest "https://www.arcgis.com/sharing/rest/content/items/924df13d845f4907bb6a6c3ed380d57a/data" -OutFile data\raw\mbta_bus_2025.zip
Expand-Archive data\raw\mbta_bus_2025.zip data\raw\mbta_2025      # same for 2024 → data\raw\mbta_2024
.\.venv\Scripts\python -m src.ingest 2024 2025
.\.venv\Scripts\python -m src.clean 2024 2025
# 2. external data
foreach ($m in 'weather','calendar_days','gtfs','events','ridership','prediction_accuracy') { .\.venv\Scripts\python -m src.external.$m }
.\.venv\Scripts\python -m src.merge_external 2024 2025
# 3. notebooks
cd notebooks
..\.venv\Scripts\python build_eda_notebook.py; ..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 02_EDA.ipynb
..\.venv\Scripts\python build_external_notebook.py; ..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 03_External_Data.ipynb
# 4. features + tests (from project root)
.\.venv\Scripts\python -m src.features; .\.venv\Scripts\python -m pytest tests -q
# 5. models (P3, ~2.5 h + ~70 min) and explainability (P4)
.\.venv\Scripts\python -m src.models.train planning operational
.\.venv\Scripts\python -m src.models.experiments
foreach ($m in 'shap_analysis','rules','evaluate_narratives'   # what-ifs (counterfactual.py) run inside notebook 06) { .\.venv\Scripts\python -m src.xai.$m }
# 6. extensions (P5; run one at a time, 16 GB RAM)
foreach ($m in 'propagation','chronic','severity','regression','conformal','bunching') { .\.venv\Scripts\python -m src.extensions.$m }
# 7. notebooks 05-07 (same pattern as step 3: build_models / build_xai / build_extensions, then nbconvert --execute)
# 8. website (P6) + live mode (P7)
.\.venv\Scripts\python -m src.app_data
.\.venv\Scripts\streamlit run app/Home.py
# 9. thesis (P8): docs/report/TransitPulse_Thesis.docx + .pdf (PDF via Word, fallback docx2pdf / LibreOffice)
.\.venv\Scripts\python -m src.report.build_thesis
```

### Keys (`.env`, never committed)
| Variable | Needed for | Without it |
|---|---|---|
| `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o-mini`) | plain-language narratives | template narratives + a warning banner; expired/invalid keys are reported in the app and in `logs/llm.log` |
| `MBTA_API_KEY` (optional) | live mode at 1000 requests/min | live mode still works at 20 requests/min |

Live mode needs an internet connection (MBTA V3 API, Open-Meteo). Every failure (rate limit, bad key, no network, no buses) is shown
as a message, never a crash.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| P0 | Data source survey, ingestion, cleaning, EDA | ✅ |
| P1 | External data: GTFS, weather, calendars, events, ridership, 2024 drift | ✅ |
| P2 | Feature engineering: temporal, daylight, spatial, leakage-safe historical rates, operational features, time split | ✅ |
| P3 | Models: historical baseline, Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM; tuning, ablation, learning curve, walk-forward, 12-month window, drift ([docs/05](docs/05_models_and_evaluation.md)) | ✅ |
| P4 | Explainability: SHAP, surrogate rules, what-if, stability, OpenAI narrator with fact check, key-health warnings & fallback ([docs/06](docs/06_explainability.md)) | ✅ |
| P5 | Extensions: severity, delay minutes + conformal intervals, propagation, bunching, chronic segments ([docs/07](docs/07_extensions.md)) | ✅ |
| P6 | Streamlit website, 9 pages, tested headless ([docs/08_dashboard](docs/08_dashboard.md)) | ✅ |
| P7 | Live mode on the MBTA V3 API ([docs/08_live_mode](docs/08_live_mode.md)) | ✅ |
| P8 | Thesis report, Word + PDF ([docs/report](docs/report/)) | ✅ |
