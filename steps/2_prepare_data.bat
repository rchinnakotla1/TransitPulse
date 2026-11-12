@echo off
REM STEP 2 - Ingest + clean the raw data, download/join external data (weather, GTFS, events, ...), build features.
call "%~dp0_run.bat" prepare
