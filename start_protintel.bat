@echo off
:: ============================================================
::  ProtIntel — Demo Launcher
::  Starts the FastAPI backend and Vite frontend dev server.
::  Works correctly whether launched from the project folder
::  OR via a Desktop shortcut (uses %~dp0 for all paths).
:: ============================================================

setlocal

:: Resolve the project root from this script's own location
set "PROJECT_ROOT=%~dp0"
:: Remove trailing backslash so paths compose cleanly
if "%PROJECT_ROOT:~-1%" == "\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo.
echo  =========================================
echo   ProtIntel — Starting Demo Services
echo  =========================================
echo.
echo  Project root : %PROJECT_ROOT%
echo.

:: ── Backend ──────────────────────────────────────────────────
:: Activate the virtual environment and launch uvicorn
echo  [1/2] Starting FastAPI backend (port 8000)...
start "ProtIntel — Backend" cmd /k "cd /d "%PROJECT_ROOT%" && call "%PROJECT_ROOT%\.venv\Scripts\activate.bat" && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

:: Brief pause so the backend window opens first
timeout /t 2 /nobreak > nul

:: ── Frontend ─────────────────────────────────────────────────
:: npm run dev (Vite) — uses node from PATH
echo  [2/2] Starting Vite frontend dev server (port 5173)...
start "ProtIntel — Frontend" cmd /k "cd /d "%PROJECT_ROOT%\frontend" && npm run dev"

echo.
echo  Both services are starting in separate windows.
echo  Backend  → http://localhost:8000
echo  Frontend → http://localhost:5173
echo.
echo  Close this window or press any key to exit this launcher.
echo  (The backend and frontend will keep running in their own windows.)
pause > nul
endlocal
