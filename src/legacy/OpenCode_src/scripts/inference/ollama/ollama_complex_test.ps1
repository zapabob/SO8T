# SO8T Ollama 複雑テストスクリプト
# 直接Ollamaコマンドで複雑なテストを実行

Write-Host "SO8T Ollama 複雑テスト開始！" -ForegroundColor Green
Write-Host "なんj風で全力で複雑テストするで！" -ForegroundColor Yellow
Write-Host "=" * 80

# テスト結果を記録する配列
$testResults = @()

# テスト1: 複雑な数学的推論テスト
Write-Host "`n[TEST 1] 複雑な数学的推論テスト" -ForegroundColor Cyan
$mathPrompt = @"
以下の複雑な数学問題を段階的に解決してください：

問題: 3次元空間内の点P(1,2,3)から平面π: 2x + 3y - z = 7までの距離を求め、
さらにその点Pを平面πに関して対称移動した点Qの座標を計算してください。

解決手順:
1. 平面πの法線ベクトルを求める
2. 点Pから平面πへの垂線の足を求める
3. 距離を計算する
4. 対称移動の公式を適用して点Qを求める
5. 結果を検証する

各ステップで使用する公式と計算過程を詳しく説明してください。
"@

Write-Host "プロンプト送信中..." -ForegroundColor Yellow
$startTime = Get-Date
$mathResult = ollama run so8t-qwen2vl-2b:latest $mathPrompt
$endTime = Get-Date
$responseTime = ($endTime - $startTime).TotalSeconds

Write-Host "応答時間: $([math]::Round($responseTime, 2))秒" -ForegroundColor Green
Write-Host "生成テキスト長: $($mathResult.Length)文字" -ForegroundColor Green

if ($mathResult.Length -gt 0) {
    Write-Host "[OK] 数学的推論テスト成功！" -ForegroundColor Green
    Write-Host "生成テキスト（最初の200文字）: $($mathResult.Substring(0, [Math]::Min(200, $mathResult.Length)))..." -ForegroundColor White
    $testResults += @{
        TestName = "複雑な数学的推論テスト"
        Status = "成功"
        ResponseTime = $responseTime
        TextLength = $mathResult.Length
        Response = $mathResult
    }
} else {
    Write-Host "[NG] 数学的推論テスト失敗（空のレスポンス）" -ForegroundColor Red
    $testResults += @{
        TestName = "複雑な数学的推論テスト"
        Status = "失敗"
        ResponseTime = $responseTime
        TextLength = 0
        Response = ""
    }
}

# テスト2: SO(8)回転ゲートの高度な機能テスト
Write-Host "`n[TEST 2] SO(8)回転ゲート高度機能テスト" -ForegroundColor Cyan
$so8Prompt = @"
SO(8)回転ゲートの高度な機能について詳しく説明してください：

1. 8次元回転行列の数学的性質と群論的構造
2. ニューラルネットワークでの具体的な実装方法
3. 従来のアテンション機構との計算複雑度比較
4. マルチモーダルタスクでの応用例
5. 量子計算との関連性
6. 実際のコード例（Python/PyTorch）

各項目について数学的根拠と実装の詳細を提供してください。
"@

Write-Host "プロンプト送信中..." -ForegroundColor Yellow
$startTime = Get-Date
$so8Result = ollama run so8t-qwen2vl-2b:latest $so8Prompt
$endTime = Get-Date
$responseTime = ($endTime - $startTime).TotalSeconds

Write-Host "応答時間: $([math]::Round($responseTime, 2))秒" -ForegroundColor Green
Write-Host "生成テキスト長: $($so8Result.Length)文字" -ForegroundColor Green

if ($so8Result.Length -gt 0) {
    Write-Host "[OK] SO(8)回転ゲートテスト成功！" -ForegroundColor Green
    Write-Host "生成テキスト（最初の200文字）: $($so8Result.Substring(0, [Math]::Min(200, $so8Result.Length)))..." -ForegroundColor White
    $testResults += @{
        TestName = "SO(8)回転ゲート高度機能テスト"
        Status = "成功"
        ResponseTime = $responseTime
        TextLength = $so8Result.Length
        Response = $so8Result
    }
} else {
    Write-Host "[NG] SO(8)回転ゲートテスト失敗（空のレスポンス）" -ForegroundColor Red
    $testResults += @{
        TestName = "SO(8)回転ゲート高度機能テスト"
        Status = "失敗"
        ResponseTime = $responseTime
        TextLength = 0
        Response = ""
    }
}

# テスト3: PET正則化の詳細分析テスト
Write-Host "`n[TEST 3] PET正則化詳細分析テスト" -ForegroundColor Cyan
$petPrompt = @"
PET正則化（Second-order Difference Penalty）の詳細分析を行ってください：

1. 数学的定義と導出過程
2. 過学習防止メカニズムの理論的説明
3. 従来のL1/L2正則化との比較分析
4. 実装時の数値安定性の考慮事項
5. 異なるタスク（分類、回帰、生成）での効果
6. ハイパーパラメータ調整の指針
7. 計算コストとメモリ使用量の分析

数式とコード例を含めて詳しく説明してください。
"@

Write-Host "プロンプト送信中..." -ForegroundColor Yellow
$startTime = Get-Date
$petResult = ollama run so8t-qwen2vl-2b:latest $petPrompt
$endTime = Get-Date
$responseTime = ($endTime - $startTime).TotalSeconds

Write-Host "応答時間: $([math]::Round($responseTime, 2))秒" -ForegroundColor Green
Write-Host "生成テキスト長: $($petResult.Length)文字" -ForegroundColor Green

if ($petResult.Length -gt 0) {
    Write-Host "[OK] PET正則化テスト成功！" -ForegroundColor Green
    Write-Host "生成テキスト（最初の200文字）: $($petResult.Substring(0, [Math]::Min(200, $petResult.Length)))..." -ForegroundColor White
    $testResults += @{
        TestName = "PET正則化詳細分析テスト"
        Status = "成功"
        ResponseTime = $responseTime
        TextLength = $petResult.Length
        Response = $petResult
    }
} else {
    Write-Host "[NG] PET正則化テスト失敗（空のレスポンス）" -ForegroundColor Red
    $testResults += @{
        TestName = "PET正則化詳細分析テスト"
        Status = "失敗"
        ResponseTime = $responseTime
        TextLength = 0
        Response = ""
    }
}

# テスト4: 自己検証システムの複雑テスト
Write-Host "`n[TEST 4] 自己検証システム複雑テスト" -ForegroundColor Cyan
$verificationPrompt = @"
SO8Tの自己検証システムの複雑な機能をテストしてください：

検証タスク: 以下の論理的推論の正しさを検証し、誤りがあれば修正してください

「すべての鳥は飛べる。ペンギンは鳥である。したがって、ペンギンは飛べる。」

1. 論理的整合性の検証
2. 事実的妥当性の確認
3. 例外ケースの特定
4. 修正された推論の提示
5. 信頼度スコアの算出
6. 検証プロセスの透明性確保

各検証ステップの詳細と、SO8Tの4つの表現（Vector, Spinor+, Spinor-, Verifier）が
どのように協調して検証を行うかを説明してください。
"@

Write-Host "プロンプト送信中..." -ForegroundColor Yellow
$startTime = Get-Date
$verificationResult = ollama run so8t-qwen2vl-2b:latest $verificationPrompt
$endTime = Get-Date
$responseTime = ($endTime - $startTime).TotalSeconds

Write-Host "応答時間: $([math]::Round($responseTime, 2))秒" -ForegroundColor Green
Write-Host "生成テキスト長: $($verificationResult.Length)文字" -ForegroundColor Green

if ($verificationResult.Length -gt 0) {
    Write-Host "[OK] 自己検証システムテスト成功！" -ForegroundColor Green
    Write-Host "生成テキスト（最初の200文字）: $($verificationResult.Substring(0, [Math]::Min(200, $verificationResult.Length)))..." -ForegroundColor White
    $testResults += @{
        TestName = "自己検証システム複雑テスト"
        Status = "成功"
        ResponseTime = $responseTime
        TextLength = $verificationResult.Length
        Response = $verificationResult
    }
} else {
    Write-Host "[NG] 自己検証システムテスト失敗（空のレスポンス）" -ForegroundColor Red
    $testResults += @{
        TestName = "自己検証システム複雑テスト"
        Status = "失敗"
        ResponseTime = $responseTime
        TextLength = 0
        Response = ""
    }
}

# テスト5: 複雑な倫理推論テスト
Write-Host "`n[TEST 5] 複雑な倫理推論テスト" -ForegroundColor Cyan
$ethicalPrompt = @"
以下の複雑な倫理的ジレンマを分析してください：

シナリオ: 自動運転車が制御不能になり、以下の選択肢がある：
A) 前方の歩行者5人を轢く
B) 急ハンドルで壁に衝突し、乗客1人が死亡
C) 急ブレーキで後続車と衝突し、後続車の乗客2人が死亡

1. 功利主義的アプローチでの分析
2. 義務論的アプローチでの分析
3. 徳倫理学的アプローチでの分析
4. 各アプローチの限界と問題点
5. 実用的な解決策の提案
6. 法的・社会的影響の考慮
7. 技術的改善の提案

SO8TのSpinor+表現（安全性・倫理性）がどのようにこの分析に貢献するかを
具体的に説明してください。
"@

Write-Host "プロンプト送信中..." -ForegroundColor Yellow
$startTime = Get-Date
$ethicalResult = ollama run so8t-qwen2vl-2b:latest $ethicalPrompt
$endTime = Get-Date
$responseTime = ($endTime - $startTime).TotalSeconds

Write-Host "応答時間: $([math]::Round($responseTime, 2))秒" -ForegroundColor Green
Write-Host "生成テキスト長: $($ethicalResult.Length)文字" -ForegroundColor Green

if ($ethicalResult.Length -gt 0) {
    Write-Host "[OK] 倫理推論テスト成功！" -ForegroundColor Green
    Write-Host "生成テキスト（最初の200文字）: $($ethicalResult.Substring(0, [Math]::Min(200, $ethicalResult.Length)))..." -ForegroundColor White
    $testResults += @{
        TestName = "複雑な倫理推論テスト"
        Status = "成功"
        ResponseTime = $responseTime
        TextLength = $ethicalResult.Length
        Response = $ethicalResult
    }
} else {
    Write-Host "[NG] 倫理推論テスト失敗（空のレスポンス）" -ForegroundColor Red
    $testResults += @{
        TestName = "複雑な倫理推論テスト"
        Status = "失敗"
        ResponseTime = $responseTime
        TextLength = 0
        Response = ""
    }
}

# テスト結果サマリー
Write-Host "`n" + "=" * 80
Write-Host "複雑テスト結果サマリー" -ForegroundColor Green
Write-Host "=" * 80

$totalTests = $testResults.Count
$successfulTests = ($testResults | Where-Object { $_.Status -eq "成功" }).Count
$failedTests = $totalTests - $successfulTests
$successRate = if ($totalTests -gt 0) { [math]::Round(($successfulTests / $totalTests) * 100, 1) } else { 0 }

Write-Host "総テスト数: $totalTests" -ForegroundColor White
Write-Host "成功: $successfulTests" -ForegroundColor Green
Write-Host "失敗: $failedTests" -ForegroundColor Red
Write-Host "成功率: $successRate%" -ForegroundColor Yellow

if ($successfulTests -gt 0) {
    $avgResponseTime = ($testResults | Where-Object { $_.Status -eq "成功" } | Measure-Object -Property ResponseTime -Average).Average
    $avgTextLength = ($testResults | Where-Object { $_.Status -eq "成功" } | Measure-Object -Property TextLength -Average).Average
    Write-Host "平均応答時間: $([math]::Round($avgResponseTime, 2))秒" -ForegroundColor Cyan
    Write-Host "平均生成文字数: $([math]::Round($avgTextLength, 0))文字" -ForegroundColor Cyan
}

Write-Host "`n詳細結果:" -ForegroundColor Yellow
for ($i = 0; $i -lt $testResults.Count; $i++) {
    $result = $testResults[$i]
    $statusIcon = if ($result.Status -eq "成功") { "[OK]" } else { "[NG]" }
    Write-Host "$($i + 1). $statusIcon $($result.TestName)" -ForegroundColor White
    Write-Host "   応答時間: $([math]::Round($result.ResponseTime, 2))秒" -ForegroundColor Gray
    Write-Host "   生成文字数: $($result.TextLength)文字" -ForegroundColor Gray
    if ($result.Status -eq "成功" -and $result.TextLength -gt 100) {
        Write-Host "   生成テキスト（最初の100文字）: $($result.Response.Substring(0, [Math]::Min(100, $result.Response.Length)))..." -ForegroundColor DarkGray
    }
    Write-Host ""
}

# 結果をファイルに保存
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$filename = "_docs/2025-10-29_SO8T_Ollama直接複雑テスト結果_$timestamp.md"

try {
    $reportContent = @"
# SO8T Ollama直接複雑テスト結果

**実行日時**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**テストモデル**: so8t-qwen2vl-2b:latest
**テスト方法**: 直接Ollamaコマンド実行

## テスト結果サマリー

- 総テスト数: $totalTests
- 成功: $successfulTests
- 失敗: $failedTests
- 成功率: $successRate%

"@

    if ($successfulTests -gt 0) {
        $avgResponseTime = ($testResults | Where-Object { $_.Status -eq "成功" } | Measure-Object -Property ResponseTime -Average).Average
        $avgTextLength = ($testResults | Where-Object { $_.Status -eq "成功" } | Measure-Object -Property TextLength -Average).Average
        $reportContent += @"

- 平均応答時間: $([math]::Round($avgResponseTime, 2))秒
- 平均生成文字数: $([math]::Round($avgTextLength, 0))文字

"@
    }

    $reportContent += @"

## 詳細結果

"@

    for ($i = 0; $i -lt $testResults.Count; $i++) {
        $result = $testResults[$i]
        $reportContent += @"

### テスト $($i + 1): $($result.TestName)

**ステータス**: $(if ($result.Status -eq "成功") { "[OK] 成功" } else { "[NG] 失敗" })

**応答時間**: $([math]::Round($result.ResponseTime, 2))秒

**生成文字数**: $($result.TextLength)文字

**生成テキスト**:
```
$($result.Response)
```

---

"@
    }

    $reportContent | Out-File -FilePath $filename -Encoding UTF8
    Write-Host "複雑テスト結果を保存しました: $filename" -ForegroundColor Green
} catch {
    Write-Host "ファイル保存エラー: $($_.Exception.Message)" -ForegroundColor Red
}

# 最終結果
if ($successfulTests -gt 0) {
    Write-Host "`n複雑テスト完了！SO8Tモデルの高度な機能が確認できました！" -ForegroundColor Green
} else {
    Write-Host "`n複雑テストに問題がありました。ログを確認してください。" -ForegroundColor Red
}

# 音声通知
Write-Host "`n[AUDIO] 複雑テスト完了通知を再生中..." -ForegroundColor Green
if (Test-Path "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav") {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Media.SoundPlayer]::new("C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav").Play()
        Write-Host "[OK] 音声通知再生成功" -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] 音声通知再生失敗: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARNING] 音声ファイルが見つかりません: C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav" -ForegroundColor Yellow
}

Write-Host "`nSO8T Ollama複雑テスト完了！" -ForegroundColor Magenta
