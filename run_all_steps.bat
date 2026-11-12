@echo off
REM ============================================================
REM  TransitPulse - re-create EVERYTHING from scratch (steps 1-9)
REM  download -> prepare -> tests -> train -> explain -> extensions
REM  -> notebooks -> website tables -> thesis
REM  Needs internet, ~25 GB free disk and 16 GB RAM.
REM  Logs: logs\pipeline_<step>.log
REM  If it stops, fix the problem and resume with the single step
REM  scripts in the steps folder (or: pipeline.py all --from STEP).
REM ============================================================
cd /d "%~dp0"
echo.
echo   This re-creates all data, models, results, notebooks and the thesis.
echo   It takes several hours. Keep the laptop plugged in and awake.
echo.
echo   F = FAST : train with the saved tuned settings (recommended, same results)
echo   S = FULL : also repeat the hyper-parameter search (several extra hours)
echo.
choice /c FS /n /m "Fast or full search? [F/S]: "
if errorlevel 2 (
    call "%~dp0steps\_run.bat" all --full
) else (
    call "%~dp0steps\_run.bat" all
)
