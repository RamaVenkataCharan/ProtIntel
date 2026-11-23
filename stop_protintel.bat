@echo off
rem ============================================================
rem  ProtIntel - Demo Stopper
rem  Kills backend and frontend services on ports 8000 / 5173.
rem ============================================================

setlocal

echo.
echo  =========================================
echo   ProtIntel - Stopping Demo Services
echo  =========================================
echo.

rem Kill console windows by title
taskkill /FI "WINDOWTITLE eq ProtIntel - Backend*"  /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq ProtIntel - Frontend*" /T /F > nul 2>&1

rem Terminate any processes bound to port 8000 or 5173
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 "') do (
    taskkill /PID %%p /F > nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5173 "') do (
    taskkill /PID %%p /F > nul 2>&1
)

echo  Done. All ProtIntel services have been stopped.
echo.
ping 127.0.0.1 -n 3 > nul
endlocal
