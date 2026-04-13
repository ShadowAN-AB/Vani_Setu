@echo off
title Vani-Setu Launcher
color 0D

echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║                                                   ║
echo  ║        🤟  V A N I - S E T U  🤟                 ║
echo  ║        ASL Sign Language Translator               ║
echo  ║        Multi-Language • OTP Auth • AI             ║
echo  ║                                                   ║
echo  ╚═══════════════════════════════════════════════════╝
echo.

:: Navigate to Code directory
cd /d "%~dp0Code"

echo  [1/3] Checking dependencies...
pip install flask flask-cors deep-translator --quiet 2>nul
if %errorlevel% neq 0 (
    echo  [WARN] pip install had issues, trying anyway...
)
echo        ✓ Dependencies ready
echo.

echo  [2/3] Starting Vani-Setu server...
echo        → http://127.0.0.1:5000
echo.

echo  [3/3] Opening browser in 3 seconds...
:: Schedule browser to open after 3 seconds asynchronously
start /B cmd /c "ping 127.0.0.1 -n 4 > nul & start """" http://127.0.0.1:5000"

echo.
echo  ═══════════════════════════════════════════════════
echo   Vani-Setu is RUNNING at: http://127.0.0.1:5000
echo   Press Ctrl+C to stop the server
echo  ═══════════════════════════════════════════════════
echo.

:: Start server (this blocks until Ctrl+C)
python server.py

echo.
echo  Server stopped. Goodbye!
pause
