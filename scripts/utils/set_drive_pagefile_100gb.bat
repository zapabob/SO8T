@echo off
chcp 65001 >nul
echo ========================================
echo D Drive Page File Configuration (100GB)
echo ========================================
echo.
echo [INFO] This script will set D: drive page file size to 100GB
echo [INFO] Administrator privileges are required
echo.
echo Press any key to continue...
pause >nul

powershell -ExecutionPolicy Bypass -File "%~dp0set_drive_pagefile_100gb.ps1"

pause

