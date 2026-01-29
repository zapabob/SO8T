# ABCテストパイプライン実行スクリプト (PowerShell)

Write-Host "=== ABCテストパイプライン実行開始 ===" -ForegroundColor Green

# 1. ABCテスト実行
Write-Host "[1/4] ABCテスト実行中..." -ForegroundColor Yellow
py -3 scripts/evaluation/run_comprehensive_abc_benchmark.py --num_samples 100 --num_seeds 10

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] ABCテスト実行に失敗しました" -ForegroundColor Red
    exit 1
}

# 2. 統計的分析実行
Write-Host "[2/4] 統計的分析実行中..." -ForegroundColor Yellow
py -3 scripts/evaluation/statistical_abc_analysis.py --results_file results/abc_testing/comprehensive_abc_results.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 統計的分析実行に失敗しました" -ForegroundColor Red
    exit 1
}

# 3. グラフ可視化実行
Write-Host "[3/4] グラフ可視化実行中..." -ForegroundColor Yellow
py -3 scripts/evaluation/visualize_abc_benchmark_statistics.py --results_file results/abc_testing/comprehensive_abc_results.json --statistical_analysis_file results/abc_testing/statistical_analysis.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] グラフ可視化実行に失敗しました" -ForegroundColor Red
    exit 1
}

# 4. HF README更新（repo_idが必要）
Write-Host "[4/4] HF README更新（repo_idを指定してください）" -ForegroundColor Yellow
Write-Host "実行コマンド:" -ForegroundColor Cyan
Write-Host "py -3 scripts/utils/update_hf_readme_with_benchmarks.py --repo_id <repo_id> --results_file results/abc_testing/comprehensive_abc_results.json --statistical_analysis_file results/abc_testing/statistical_analysis.json --visualization_dir results/abc_testing/visualizations" -ForegroundColor White

Write-Host "=== ABCテストパイプライン実行完了 ===" -ForegroundColor Green
