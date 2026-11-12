"""TransitPulse - run the whole project step by step, with no code editor needed.

Every step is a thin wrapper around the project modules (src/...), run in the right order, with the same settings that
produced the results in reports/. Output is shown on screen and also saved to logs/pipeline_<step>.log.

    python pipeline.py list                   show the steps
    python pipeline.py download               1. download MBTA data + unpack the bundled inputs
    python pipeline.py prepare                2. ingest, clean, external data, merge, features
    python pipeline.py tests                  3. run the test suite (pytest)
    python pipeline.py train [--full]         4. train all models (default: saved best settings, no search)
    python pipeline.py explain                5. experiments + explainability (SHAP, rules, narratives)
    python pipeline.py extensions             6. propagation, chronic, severity, regression, conformal, bunching
    python pipeline.py notebooks              7. rebuild + execute notebooks 02-07 (with outputs, for presenting)
    python pipeline.py app_data               8. pre-compute the website tables
    python pipeline.py thesis                 9. build the thesis (docx + pdf)
    python pipeline.py all [--from STEP] [--full]   run 1-9 in order (optionally resume from a step)
    python pipeline.py app                    start the Streamlit website

The double-click .bat files in steps/ call these commands.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
DATA = ROOT / "data"
LOGS = ROOT / "logs"
INPUTS = ROOT / "inputs"

# MBTA Bus Arrival Departure Times (MassDOT Open Data / ArcGIS). sha256 + size = the exact files used for the results.
RAW = {
    2024: ("96c77138c3144906bce93d0257531b6a", 315387304,
           "DD45DBE56233D44FB8F8D41D8EC1E40D2D80C4261F4E95F691379AA4B47971FC"),
    2025: ("924df13d845f4907bb6a6c3ed380d57a", 404153122,
           "EAE5344638E61319185D5C1C996F8BEC3E9A71879702B3FC34D9991F04CE7706"),
}
ARCGIS = "https://www.arcgis.com/sharing/rest/content/items/{}/data"

NOTEBOOKS = [("build_eda_notebook.py", "02_EDA.ipynb"),
             ("build_external_notebook.py", "03_External_Data.ipynb"),
             ("build_features_notebook.py", "04_Feature_Engineering.ipynb"),
             ("build_models_notebook.py", "05_Models.ipynb"),
             ("build_xai_notebook.py", "06_Explainability.ipynb"),
             ("build_extensions_notebook.py", "07_Extensions.ipynb")]

_log_file = None


# ------------------------------------------------------------------------------------------------ helpers
def say(msg: str = ""):
    line = f"{time.strftime('%H:%M:%S')}  {msg}" if msg else ""
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n"); _log_file.flush()


def banner(title: str):
    say("=" * 78); say(title); say("=" * 78)


def run(cmd: list[str], cwd: Path = ROOT):
    """Run a command, stream its output to the screen and the log, stop the pipeline if it fails."""
    say("> " + " ".join(str(c) for c in cmd))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}
    t0 = time.time()
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
        if _log_file:
            _log_file.write(line); _log_file.flush()
    if p.wait() != 0:
        say(f"FAILED (exit code {p.returncode}) after {time.time() - t0:.0f}s: {' '.join(str(c) for c in cmd)}")
        raise SystemExit(p.returncode)
    say(f"ok ({time.time() - t0:.0f}s)")


def py(module: str, *args: str):
    run([PY, "-m", module, *args])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def download(url: str, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); done = 0; last = -1
        while chunk := r.read(1 << 20):
            f.write(chunk); done += len(chunk)
            pct = int(done * 100 / total) if total else -1
            if pct // 10 != last // 10:
                say(f"   {dest.name}: {done / 1e6:,.0f} MB" + (f" ({pct}%)" if total else "")); last = pct
    tmp.replace(dest)


# ------------------------------------------------------------------------------------------------ steps
def step_download(args):
    """1. Core MBTA data (2024 + 2025) and the bundled manual inputs."""
    raw = DATA / "raw"; raw.mkdir(parents=True, exist_ok=True)
    for year, (item, size, digest) in RAW.items():
        z = raw / f"mbta_bus_{year}.zip"
        if z.exists() and z.stat().st_size == size:
            say(f"{z.name} already downloaded")
        else:
            say(f"downloading MBTA Bus Arrival Departure Times {year} (~{size / 1e6:.0f} MB) ...")
            download(ARCGIS.format(item), z)
        got = sha256(z)
        if got != digest:
            say(f"WARNING: {z.name} differs from the file used for the published results "
                f"(MBTA may have re-published it). Results can differ slightly.")
        else:
            say(f"{z.name} verified (identical to the file used for the results)")
        target = raw / f"mbta_{year}"
        csv_dir = target / f"MBTA_Bus_Arrival_Departure_Times_{year}"
        if csv_dir.exists() and len(list(csv_dir.glob("*.csv"))) >= 12:
            say(f"{csv_dir.relative_to(ROOT)} already unpacked")
        else:
            say(f"unpacking {z.name} ...")
            zipfile.ZipFile(z).extractall(target)
    # inputs that were downloaded by hand from MBTA OPMI portals, bundled in the repo (inputs/)
    ext = DATA / "external"; ext.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUTS / "prediction_accuracy.csv", ext / "prediction_accuracy.csv")
    zipfile.ZipFile(INPUTS / "ridership_fall_2023_2024.zip").extractall(ext / "ridership")
    say("bundled inputs copied to data/external (ridership Fall 2023/2024, MBTA prediction accuracy)")
    say("note: weather, GTFS schedules and Red Sox/Bruins schedules are downloaded automatically in step 2")


def step_prepare(args):
    """2. Ingest, clean, external data, merge, leakage-safe features (+ time split)."""
    py("src.ingest", "2024", "2025")
    py("src.clean", "2024", "2025")
    for m in ("weather", "calendar_days", "gtfs", "events", "ridership", "prediction_accuracy"):
        py(f"src.external.{m}")
    py("src.merge_external", "2024", "2025")
    py("src.features")


def step_tests(args):
    """3. Test suite: leakage, features, joins, XAI, narrator, website."""
    run([PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])


def step_train(args):
    """4. Six models x two modes. Default = saved best settings (fast); --full = repeat the hyper-parameter search."""
    if args.full:
        say("FULL mode: repeating the Optuna / grid search (several hours)")
        py("src.models.train", "planning", "operational")
    else:
        say("FAST mode: re-using the tuned settings in reports/best_params.json (no search, same results)")
        py("src.models.train", "planning", "operational", "--fast")


def step_explain(args):
    """5. Experiments (learning curve, ablation, drift, ...) and explainability."""
    py("src.models.experiments")
    for m in ("shap_analysis", "rules", "evaluate_narratives"):
        py(f"src.xai.{m}")


def step_extensions(args):
    """6. Extensions, one at a time (memory)."""
    for m in ("propagation", "chronic", "severity", "regression", "conformal", "bunching"):
        py(f"src.extensions.{m}")


def step_notebooks(args):
    """7. Rebuild and execute notebooks 02-07 so they contain fresh outputs and figures."""
    nb_dir = ROOT / "notebooks"
    for builder, nb in NOTEBOOKS:
        run([PY, builder], cwd=nb_dir)
        run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=-1", nb], cwd=nb_dir)


def step_app_data(args):
    """8. Small pre-computed tables for the website (reports/app)."""
    py("src.app_data")


def step_thesis(args):
    """9. Thesis document (docs/report/TransitPulse_Thesis.docx + .pdf)."""
    py("src.report.build_thesis")


def step_app(args):
    """Start the Streamlit website."""
    run([PY, "-m", "streamlit", "run", "app/Home.py"])


STEPS = {"download": step_download, "prepare": step_prepare, "tests": step_tests, "train": step_train,
         "explain": step_explain, "extensions": step_extensions, "notebooks": step_notebooks,
         "app_data": step_app_data, "thesis": step_thesis}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="TransitPulse pipeline", formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("step", choices=["list", "all", "app", *STEPS])
    ap.add_argument("--full", action="store_true", help="train: repeat the hyper-parameter search (slow)")
    ap.add_argument("--from", dest="start", choices=list(STEPS), help="all: resume from this step")
    args = ap.parse_args()

    if args.step == "list":
        for i, (k, f) in enumerate(STEPS.items(), 1):
            print(f"{k:<11} {f.__doc__.strip()}")
        print(f"{'app':<11} {step_app.__doc__.strip()}")
        return
    if args.step == "app":
        return step_app(args)

    global _log_file
    LOGS.mkdir(exist_ok=True)
    names = list(STEPS) if args.step == "all" else [args.step]
    if args.step == "all" and args.start:
        names = names[names.index(args.start):]
    t_all = time.time()
    for name in names:
        _log_file = open(LOGS / f"pipeline_{name}.log", "w", encoding="utf-8")
        banner(f"STEP {list(STEPS).index(name) + 1}/9: {name} - {STEPS[name].__doc__.strip()}")
        t0 = time.time()
        STEPS[name](args)
        say(f"STEP {name} finished in {(time.time() - t0) / 60:.1f} min (log: logs/pipeline_{name}.log)")
        _log_file.close(); _log_file = None
    if len(names) > 1:
        print(f"\nAll done in {(time.time() - t_all) / 60:.1f} min.")


if __name__ == "__main__":
    main()
