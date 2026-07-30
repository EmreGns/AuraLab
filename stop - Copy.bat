@echo off
title AI Bot Durdurucu
echo Yapay zeka ve Telegram botu kapatiliyor...

:: Arka planda çalısan yapay zeka motorunu kapatır
taskkill /f /im ollama.exe >nul 2>&1

:: Arka planda pusuya yatmıs Python botlarını kapatır
taskkill /f /im pythonw.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1

echo.
echo === SISTEM TAMAMEN DURDURULDU ===
echo Sen tekrar baslatana kadar bilgisayarini asla yormayacak.
echo.
timeout /t 3