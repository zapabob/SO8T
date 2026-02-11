@echo off
chcp 65001 >nul
echo ========================================
echo Page File Size Configuration
echo ========================================
echo.
echo [INFO] This script will increase C: drive page file size
echo [INFO] Administrator privileges are required
echo.
echo Press any key to continue...
pause >nul

powershell -ExecutionPolicy Bypass -File "%~dp0increase_pagefile.ps1"

pause






