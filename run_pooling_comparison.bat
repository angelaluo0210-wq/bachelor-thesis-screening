@echo off
setlocal
python scripts\run_pooling_comparison.py %*
set "RUN_EXIT=%ERRORLEVEL%"
if not "%RUN_EXIT%"=="0" (
  echo Pooling comparison failed. See the error above.
  exit /b %RUN_EXIT%
)
echo Pooling comparison runner finished successfully.
exit /b 0
