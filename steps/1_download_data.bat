@echo off
REM STEP 1 - Download the MBTA bus data (2024 + 2025, ~720 MB) and unpack the bundled inputs.
call "%~dp0_run.bat" download
