@echo off
title Amy Bot Kontrol Paneli

:: 1. Eski Python süreçlerini temizle
taskkill /f /im python.exe >nul 2>&1

:: 2. Sadece Amy'yi Başlat (Loglar direkt bu ekranda akacak)
cd /d "D:\NVIDIA_Cache_Log"
"D:\NVIDIA_Cache_Log\stable-diffusion-webui\venv\Scripts\python.exe" "ai_brain.py"

pause