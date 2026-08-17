@echo off
:: ============================================================
::  ProtIntel — Demo Stopper
::  Kills the backend and frontend console windows opened by
::  start_protintel.bat, then terminates any lingering uvicorn
::  or node processes bound to ports 8000 / 5173.
:: ============================================================

setlocal

echo.
echo  =========================================
echo   ProtIntel — Stopping Demo Services
echo  =========================================
echo.

:: Kill by window title (matches titles set by start_protintel.bat)
taskkill /FI "WINDOWTITLE eq ProtIntel — Backend*"  /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq ProtIntel — Frontend*" /T /F > nul 2>&1

:: Also terminate by image name in case the windows were renamed
:: Kill uvicorn processes
taskkill /IM uvicorn.exe /F > nul 2>&1

:: Kill node processes on port 5173 (Vite dev server)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5173 "') do (
    taskkill /PID %%p /F > nul 2>&1
)

:: Kill any process on port 8000 (FastAPI / uvicorn)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 "') do (
    taskkill /PID %%p /F > nul 2>&1
)

echo  Done. All ProtIntel services have been stopped.
echo.
timeout /t 2 /nobreak > nul
endlocal
