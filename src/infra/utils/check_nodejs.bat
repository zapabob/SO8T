@echo off
setlocal
chcp 65001 >nul

echo [CHECK] Verifying Node.js environment...

where node >nul 2>&1
if errorlevel 1 (
    echo [NG] Node.js not found in PATH.
    echo       Please install Node.js 18+ and ensure npm is available.
    goto :npm_check
) else (
    for /f "tokens=2 delims=v" %%i in ('node --version') do set NODE_VERSION=%%i
    echo [OK] Node.js version: %NODE_VERSION%
)

:npm_check
where npm >nul 2>&1
if errorlevel 1 (
    echo [NG] npm not found. Install Node.js from https://nodejs.org/ja/.
) else (
    for /f "tokens=2 delims=v" %%i in ('npm --version') do set NPM_VERSION=%%i
    echo [OK] npm version: %NPM_VERSION%
)

echo [CHECK] Completed Node.js verification.
endlocal

