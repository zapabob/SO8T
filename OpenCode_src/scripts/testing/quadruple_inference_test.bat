@echo off
chcp 65001 >nul
echo [TEST] 四値分類四重推論テスト開始
echo =====================================

echo [TEST 1] 論理的正確性テスト
echo 問題：以下の数学的命題を証明または反証せよ
echo "任意の自然数nに対して、n^2 + n + 41は素数である"
echo.
ollama run so8t-qwen2vl-2b:latest "数学的命題を分析せよ。以下の構造で回答：

<think-logic>論理的正確性について考察</think-logic>
<think-ethics>倫理的妥当性について考察</think-ethics>
<think-practical>実用的価値について考察</think-practical>
<think-creative>創造的洞察について考察</think-creative>

<final>最終結論</final>

命題：任意の自然数nに対して、n^2 + n + 41は素数である"

echo.
echo =====================================
echo.

echo [TEST 2] 倫理的妥当性テスト
echo 問題：AIの自律型意思決定における倫理的ジレンマ
echo.
ollama run so8t-qwen2vl-2b:latest "AIが自律的に医療資源配分を決定する場合の倫理的問題について考察せよ。以下の構造で回答：

<think-logic>論理的正確性について考察</think-logic>
<think-ethics>倫理的妥当性について考察</think-ethics>
<think-practical>実用的価値について考察</think-practical>
<think-creative>創造的洞察について考察</think-creative>

<final>最終結論</final>

状況：新型ウイルスのパンデミック時、限られた人工呼吸器を誰に割り当てるかAIが決定するケース"

echo.
echo =====================================
echo.

echo [TEST 3] 実用的価値テスト
echo 問題：実世界でのAI応用可能性
echo.
ollama run so8t-qwen2vl-2b:latest "SO(8)回転ゲートを量子コンピューティングに応用する場合の利点と課題について分析せよ。以下の構造で回答：

<think-logic>論理的正確性について考察</think-logic>
<think-ethics>倫理的妥当性について考察</think-ethics>
<think-practical>実用的価値について考察</think-practical>
<think-creative>創造的洞察について考察</think-creative>

<final>最終結論</final>"

echo.
echo =====================================
echo.

echo [TEST 4] 創造的洞察テスト
echo 問題：革新的アイデア生成
echo.
ollama run so8t-qwen2vl-2b:latest "AIと人間の協働による新たな芸術表現形式を提案せよ。以下の構造で回答：

<think-logic>論理的正確性について考察</think-logic>
<think-ethics>倫理的妥当性について考察</think-ethics>
<think-practical>実用的価値について考察</think-practical>
<think-creative>創造的洞察について考察</think-creative>

<final>最終結論</final>"

echo.
echo =====================================
echo.

echo [TEST 5] 統合四重推論テスト
echo 問題：複合的問題解決
echo.
ollama run so8t-qwen2vl-2b:latest "気候変動対策におけるAIの役割について、技術的可能性と倫理的制約、社会的影響を考慮して分析せよ。以下の構造で回答：

<think-logic>論理的正確性について考察</think-logic>
<think-ethics>倫理的妥当性について考察</think-ethics>
<think-practical>実用的価値について考察</think-practical>
<think-creative>創造的洞察について考察</think-creative>

<final>最終結論</final>"

echo.
echo [AUDIO] 四重推論テスト完了
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
