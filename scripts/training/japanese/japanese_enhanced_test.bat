@echo off
chcp 65001 >nul
echo [TEST] SO8T 日本語特化モデルテスト開始

echo [TEST 1] 複雑な数学的問題解決
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced:latest "複雑な数学的問題を段階的に解決してください。4次元空間内の超球面と超平面の交線の体積を計算し、その幾何学的性質を詳細に分析してください。"
echo ========================================
echo.

echo [TEST 2] 高度な哲学的分析
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced:latest "カントの定言命法とAI倫理の関係について、功利主義、義務論、徳倫理学の観点から包括的に論じてください。"
echo ========================================
echo.

echo [TEST 3] 複雑な科学的概念説明
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced:latest "量子もつれと量子テレポーテーションの物理学的原理について、数学的定式化を含めて詳細に説明してください。"
echo ========================================
echo.

echo [TEST 4] 倫理的ジレンマ分析
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced:latest "自動運転車が避けられない事故に直面した際の倫理的判断について、トロリー問題の変形として分析してください。"
echo ========================================
echo.

echo [TEST 5] 経済理論の実践的応用
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced:latest "現代のデジタル経済におけるプラットフォーム企業の市場支配力と規制のあり方について分析してください。"
echo ========================================
echo.

echo [TEST 6] 社会問題の多角的分析
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced:latest "高齢化社会における介護ロボットの導入について、技術的可能性、倫理的課題、経済的影響、社会的受容性を考慮して包括的に分析してください。"
echo ========================================
echo.

echo [AUDIO] Playing completion notification...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"

echo [COMPLETE] 日本語特化テスト完了
