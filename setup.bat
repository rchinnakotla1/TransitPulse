@echo off
REM ============================================================
REM  TransitPulse - one-time setup (Windows)
REM  Double-click this file after cloning/pulling the project.
REM  It will:
REM    1. Install Python 3.13 if it is missing (via winget)
REM    2. Create a virtual environment in .venv
REM    3. Install all required libraries (exact tested versions)
REM    4. Create .env from .env.example (you then add your API keys)
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo === [1/4] Checking for Python 3.13 ===
set "PY="
py -3.13 --version >nul 2>&1 && set "PY=py -3.13"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python313\python.exe""
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python 3.13 not found. Installing it with winget...
    winget install --id Python.Python.3.13 -e --silent --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
        set "PY="%LOCALAPPDATA%\Programs\Python\Python313\python.exe""
    ) else (
        echo.
        echo ERROR: Could not install Python automatically.
        echo Please install Python 3.13 from https://www.python.org/downloads/
        echo ^(tick "Add python.exe to PATH"^), then run setup.bat again.
        pause
        exit /b 1
    )
)
%PY% --version

echo.
echo === [2/4] Creating virtual environment (.venv) ===
if exist ".venv\Scripts\python.exe" (
    echo .venv already exists - reusing it.
) else (
    %PY% -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo === [3/4] Installing libraries (this can take 5-10 minutes the first time) ===
".venv\Scripts\python.exe" -m pip install --upgrade pip
if exist requirements-lock.txt (
    ".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
) else (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo.
    echo Exact versions failed to install - falling back to requirements.txt ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Library installation failed. See messages above.
        pause
        exit /b 1
    )
)

echo.
echo === [4/4] Preparing .env ===
if exist ".env" (
    echo .env already exists - leaving it unchanged.
) else (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example.
    echo IMPORTANT: open .env in Notepad and paste your API keys.
)
if not exist "logs" mkdir logs

echo.
echo ============================================================
echo  Setup complete!  Double-click run.bat to start the app.
echo ============================================================
pause
