@echo off
REM Shared helper for the step scripts: makes sure setup was done, then runs one pipeline step.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found - running setup.bat first...
    call setup.bat
)
".venv\Scripts\python.exe" pipeline.py %*
set "ERR=%errorlevel%"
echo.
if not "%ERR%"=="0" (
    echo *** FAILED - read the messages above. Full log is in the logs folder. ***
) else (
    echo *** Finished successfully. ***
)
pause
exit /b %ERR%
