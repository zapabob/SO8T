@echo off
chcp 65001 >nul
echo [OLLAMA] SO8T 日本語特化ファインチューニングモデル V2 テスト開始！
echo モデル: so8t-qwen2vl-2b-japanese-enhanced-v2
echo 推論: 英語（内部）、回答: 日本語（最終出力）


echo ========================================
echo [TEST 1] 複雑な数学的問題を段階的に解決してください。4次元空間内の超球面と超平面の交線の体積を計算し、その幾何学的性質を詳細に分析してください。
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced-v2 "複雑な数学的問題を段階的に解決してください。4次元空間内の超球面と超平面の交線の体積を計算し、その幾何学的性質を詳細に分析してください。\n\n入力: 4次元空間、超球面、超平面、交線、体積計算"
echo.

echo ========================================
echo [TEST 2] カントの定言命法とAI倫理の関係について、功利主義、義務論、徳倫理学の観点から包括的に論じてください。
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced-v2 "カントの定言命法とAI倫理の関係について、功利主義、義務論、徳倫理学の観点から包括的に論じてください。\n\n入力: カント、定言命法、AI倫理、功利主義、義務論、徳倫理学"
echo.

echo ========================================
echo [TEST 3] 量子もつれと量子テレポーテーションの物理学的原理について、数学的定式化を含めて詳細に説明してください。
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced-v2 "量子もつれと量子テレポーテーションの物理学的原理について、数学的定式化を含めて詳細に説明してください。\n\n入力: 量子もつれ、量子テレポーテーション、数学的定式化、物理学的原理"
echo.

echo ========================================
echo [TEST 4] 複雑な社会問題の解決策を提案してください。気候変動、格差拡大、高齢化社会の3つの問題を統合的に分析し、持続可能な解決策を提示してください。
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced-v2 "複雑な社会問題の解決策を提案してください。気候変動、格差拡大、高齢化社会の3つの問題を統合的に分析し、持続可能な解決策を提示してください。\n\n入力: 気候変動、格差拡大、高齢化社会、統合的分析、持続可能な解決策"
echo.

echo ========================================
echo [TEST 5] 人工知能の限界と可能性について、チューリングテスト、中国語の部屋、フレーム問題などの観点から詳細に分析してください。
echo ========================================
ollama run so8t-qwen2vl-2b-japanese-enhanced-v2 "人工知能の限界と可能性について、チューリングテスト、中国語の部屋、フレーム問題などの観点から詳細に分析してください。\n\n入力: 人工知能、限界、可能性、チューリングテスト、中国語の部屋、フレーム問題"
echo.

echo [AUDIO] テスト完了通知を再生するで！
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"
