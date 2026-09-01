@echo off
setlocal
python scripts\run_lr_tuning.py %*
if errorlevel 1 (
  echo LR tuning failed. See the error above. No completion is being claimed.
  exit /b %errorlevel%
)
echo LR tuning runner finished successfully.
exit /b 0
