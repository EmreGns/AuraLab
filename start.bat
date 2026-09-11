@echo off
title AuraLab Desktop
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] AuraLab Python sanal ortami .venv bulunamadi!
    echo Lutfen once asagidaki komutlari calistirin:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo AuraLab Baslatiliyor...
"%~dp0.venv\Scripts\python.exe" -m auto_eq
endlocal
