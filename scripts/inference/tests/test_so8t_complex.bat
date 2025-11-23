@echo off
chcp 65001 >nul
echo [TEST] SO8T Complete Pipeline Complex Test
echo ==========================================

echo [STEP 1] Testing quantum gravity theory...
ollama run so8t-complete-pipeline "量子重力理論におけるSO(8)群構造の役割について、数学的詳細を含めて説明してください。特に、8次元回転ゲートが時空の幾何学的性質にどのように影響するかを、一般相対性理論と量子力学の統合の観点から論じてください。"

echo.
echo [STEP 2] Testing consciousness hard problem...
ollama run so8t-complete-pipeline "意識のハードプロブレムについて、デイヴィッド・チャーマーズの議論を踏まえて詳しく説明してください。特に、クオリアの性質と物理的プロセスとの関係性について、SO(8)群構造の観点から考察してください。"

echo.
echo [STEP 3] Testing mathematical Platonism...
ollama run so8t-complete-pipeline "数学的プラトニズムについて、ゲーデルの不完全性定理とSO(8)群構造の関係性を踏まえて論じてください。数学的対象の存在論的ステータスについて、実在論と反実在論の立場から検討してください。"

echo.
echo [STEP 4] Testing time nature...
ollama run so8t-complete-pipeline "時間の本質について、アインシュタインの相対性理論と量子力学の観点から詳しく説明してください。特に、時間の矢とSO(8)群構造の関係性について考察してください。"

echo.
echo [STEP 5] Testing free will...
ollama run so8t-complete-pipeline "自由意志の問題について、決定論と非決定論の立場から詳しく論じてください。特に、量子力学の不確定性原理とSO(8)群構造が自由意志の可能性に与える影響について考察してください。"

echo.
echo [AUDIO] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

echo.
echo [COMPLETE] SO8T Complex Test completed!
