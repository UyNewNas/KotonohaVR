@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -m kotonoha_vr
if errorlevel 1 goto failed
exit /b 0
:setup
echo Run SETUP_WINDOWS.cmd first.
pause
exit /b 1
:failed
echo.
echo Startup failed. Check the error above.
pause
exit /b 1
