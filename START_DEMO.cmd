@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto verify
py -3.12 -m venv .venv
if errorlevel 1 goto failed
:verify
".venv\Scripts\python.exe" -c "import kotonoha_vr.ui" >nul 2>&1
if not errorlevel 1 goto run
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto failed
:run
".venv\Scripts\python.exe" -m kotonoha_vr --demo
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Startup failed. Install Python 3.12 with Python Launcher, then try again.
pause
exit /b 1
