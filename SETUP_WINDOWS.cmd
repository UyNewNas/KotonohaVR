@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -e ".[windows,local-asr]"
if errorlevel 1 goto failed
echo.
echo Setup complete. Run START_APP.cmd and open Settings.
echo The local speech model downloads on first use.
pause
exit /b 0
:failed
echo.
echo Setup failed. Check the error above and your Python/network setup.
pause
exit /b 1
