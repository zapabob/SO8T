@echo off
REM SO8T Pipeline Test Runner

set PYTHONPATH=%CD%\src;%PYTHONPATH%

echo ============================================
echo SO8T Pipeline Tests
echo ============================================

py -3 tests/test_so8t_pipeline_integration.py

echo ============================================
echo Done
pause
