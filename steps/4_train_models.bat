@echo off
REM STEP 4 - Train all 6 models in both modes (planning + operational).
echo.
echo   F = FAST : re-use the tuned settings saved in reports\best_params.json
echo             (same models and results, skips the hyper-parameter search)
echo   S = FULL : repeat the full Optuna / grid search (several extra hours)
echo.
choice /c FS /n /m "Fast or full search? [F/S]: "
if errorlevel 2 (
    call "%~dp0_run.bat" train --full
) else (
    call "%~dp0_run.bat" train
)
