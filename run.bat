@echo off
REM ============================================================
REM  TransitPulse - start the Streamlit website (Windows)
REM  Double-click this file. Your browser opens at http://localhost:8501
REM  Close this window (or press Ctrl+C) to stop the app.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo First run detected - running setup first...
    call setup.bat
    if not exist ".venv\Scripts\streamlit.exe" (
        echo Setup did not finish. Fix the errors above and try again.
        pause
        exit /b 1
    )
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo NOTE: Created .env - add your API keys to it for LLM narratives and live mode.
)

if not exist "models\XGB_planning.json" (
    echo WARNING: models\XGB_planning.json is missing - the Predict page will not work.
)

echo Starting TransitPulse at http://localhost:8501 ...
".venv\Scripts\python.exe" -m streamlit run app\Home.py
pause
