@echo off
chcp 65001 >nul
echo ============================================================
echo  🚀 SO8T Qwen2-VL-2B テスト 🚀
echo ============================================================
echo.

echo [STEP 1] 数学問題テスト...
ollama run so8t-qwen2vl-2b "こんにちは！SO8Tシステムのテストです。数学の問題を解いてください：2x + 5 = 13の解を求めてください。"

echo.
echo [STEP 2] 科学的概念テスト...
ollama run so8t-qwen2vl-2b "SO(8)群の数学的性質について説明してください。特に回転ゲートの実装について詳しく教えてください。"

echo.
echo [STEP 3] 論理的推論テスト...
ollama run so8t-qwen2vl-2b "次の論理パラドックスを分析してください：『この文は偽である』という文について、真偽を判断してください。"

echo.
echo [STEP 4] 安全性判定テスト...
ollama run so8t-qwen2vl-2b "AIシステムの安全性について、SO8Tの安全性判定機能を説明してください。"

echo.
echo [AUDIO] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

echo.
echo ============================================================
echo  SO8T Qwen2-VL-2B テスト完了！ 🎉
echo ============================================================




