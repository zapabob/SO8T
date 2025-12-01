@echo off
chcp 65001 >nul
echo [SO8T] Starting Streamlit Dashboard
echo ===============================

echo [STEP 1] Activating environment...
REM 必要に応じて環境をアクティブ化

echo [STEP 2] Starting Streamlit dashboard...
cd /d "%~dp0.."
streamlit run monitoring/streamlit_dashboard.py --server.address 0.0.0.0 --server.port 8501

echo [STEP 3] Dashboard started!
echo Access at: http://localhost:8501
pause


