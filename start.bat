@echo off
title Auto EQ Desktop
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Auto EQ Python sanal ortami (.venv) bulunamadi!
    echo Lutfen once asagidaki komutlari calistirin:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Auto EQ Baslatiliyor...
"%~dp0.venv\Scripts\python.exe" -m auto_eq
endlocal
