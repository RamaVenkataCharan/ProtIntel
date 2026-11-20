@echo off
rem ============================================================
rem  ProtIntel - Demo Launcher
rem  Starts the FastAPI backend and Vite frontend dev server.
rem  Works correctly whether launched from the project folder
rem  OR via a Desktop shortcut (uses %~dp0 for all paths).
rem ============================================================

setlocal enabledelayedexpansion

rem Resolve the project root from this script's own location
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%" == "\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo.
echo  =========================================
echo   ProtIntel - Starting Demo Services
echo  =========================================
echo.
echo  Project root : %PROJECT_ROOT%
echo.

rem -- Stop any stale services on 8000 / 5173 first -----------
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 "') do (
    taskkill /PID %%p /F > nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5173 "') do (
    taskkill /PID %%p /F > nul 2>&1
)

rem -- Backend ------------------------------------------------
echo  [1/2] Starting FastAPI backend (port 8000)...
start "ProtIntel - Backend" cmd /k "cd /d "%PROJECT_ROOT%" && call "%PROJECT_ROOT%\.venv\Scripts\activate.bat" && "%PROJECT_ROOT%\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

rem -- Frontend -----------------------------------------------
echo  [2/2] Starting Vite frontend dev server (port 5173)...
start "ProtIntel - Frontend" cmd /k "cd /d "%PROJECT_ROOT%\frontend" && npm run dev"

rem -- Wait for Backend & Open Browser ------------------------
echo.
echo  Waiting for backend API to initialize (loading ML model)...
powershell -NoProfile -Command "for ($i=0; $i -lt 30; $i++) { try { $r = Invoke-WebRequest -Uri http://localhost:8000/ -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {} Start-Sleep -Seconds 1 }; exit 0"

start http://localhost:5173

echo.
echo  Both services are running:
echo  Backend  - http://localhost:8000
echo  Frontend - http://localhost:5173
echo.
echo  Close this window or press any key to exit this launcher.
echo  (The backend and frontend will keep running in their own windows.)
echo  (To stop all services, run stop_protintel.bat)
pause > nul
endlocal
