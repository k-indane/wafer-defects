@echo off
REM ================================
REM Wafer Defect Dev Environment
REM ================================

REM --- Change this only if your repo moves ---
cd /d "C:\Users\custu\My Drive\MSDS 458\Report\wafer-defects"

REM --- Initialize conda (safe to call multiple times) ---
call "%ProgramData%\miniconda3\Scripts\activate.bat"

REM --- Activate your environment ---
call conda activate wafer-infer

REM --- Make repo importable as a package ---
set PYTHONPATH=%CD%

REM --- Friendly confirmation ---
echo.
echo =====================================
echo  Environment Ready
echo  Repo: %CD%
echo  Conda env: wafer-infer
echo  PYTHONPATH: %PYTHONPATH%
echo =====================================
echo.

REM --- Keep terminal open ---
cmd /k
