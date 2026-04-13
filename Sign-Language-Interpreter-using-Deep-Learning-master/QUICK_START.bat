@echo off
title Vani-Setu Quick Launch
color 0D

:: Navigate to the Code directory relative to this script
cd /d "%~dp0Code"

:: Kill any existing server on port 5000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING 2^>nul') do (
    taskkill /PID %%a /F >nul 2>nul
)

:: Install deps silently
pip install flask flask-cors deep-translator --quiet 2>nul

:: Open browser after 2 second delay (in background)
start /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:5000"

:: Run server
python server.py
pause
