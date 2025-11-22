#!/usr/bin/env pwsh
# -*- coding: utf-8 -*-
# 電源投入時に自動的に訓練を再開するスクリプト

param(
    [string]$ConfigPath = "configs/train_borea_phi35_so8t_thinking.yaml",
    [string]$OutputDir = "D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking"
)

Write-Host "[AUTO-RESUME] Checking for training sessions..." -ForegroundColor Green

$sessionFile = Join-Path $OutputDir "training_session.json"

if (Test-Path $sessionFile) {
    Write-Host "[AUTO-RESUME] Found session file: $sessionFile" -ForegroundColor Yellow
    
    $sessionData = Get-Content $sessionFile | ConvertFrom-Json
    
    if ($sessionData.status -ne "completed") {
        Write-Host "[AUTO-RESUME] Resuming training..." -ForegroundColor Green
        Write-Host "[AUTO-RESUME] Session ID: $($sessionData.session_id)" -ForegroundColor Cyan
        Write-Host "[AUTO-RESUME] Progress: $($sessionData.current_step)/$($sessionData.total_steps)" -ForegroundColor Cyan
        
        # 訓練スクリプトを自動再開モードで実行
        py -3 scripts/training/train_borea_phi35_so8t_thinking.py `
            --config $ConfigPath `
            --dataset "D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl" `
            --output-dir $OutputDir `
            --auto-resume
    } else {
        Write-Host "[AUTO-RESUME] Training already completed. No resume needed." -ForegroundColor Green
    }
} else {
    Write-Host "[AUTO-RESUME] No session file found. Starting new training..." -ForegroundColor Yellow
    
    # 新しい訓練を開始
    py -3 scripts/training/train_borea_phi35_so8t_thinking.py `
        --config $ConfigPath `
        --dataset "D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl" `
        --output-dir $OutputDir
}






































































































