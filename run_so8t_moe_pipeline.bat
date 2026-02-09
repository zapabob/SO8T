@echo off
REM SO8T MoE Training Pipeline Launcher
REM Run from project root

echo ============================================
echo SO8T MoE Training Pipeline
echo ============================================

set PYTHONPATH=%CD%\src;%PYTHONPATH%

echo Running pipeline...
py -3 src/training/so8t_moe_unsloth_pipeline.py %*

echo ============================================
echo Done
