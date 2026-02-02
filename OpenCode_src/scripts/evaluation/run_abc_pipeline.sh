#!/bin/bash
# ABCテストパイプライン実行スクリプト

echo "=== ABCテストパイプライン実行開始 ==="

# 1. ABCテスト実行
echo "[1/4] ABCテスト実行中..."
py -3 scripts/evaluation/run_comprehensive_abc_benchmark.py --num_samples 100 --num_seeds 10

if [ $? -ne 0 ]; then
    echo "[ERROR] ABCテスト実行に失敗しました"
    exit 1
fi

# 2. 統計的分析実行
echo "[2/4] 統計的分析実行中..."
py -3 scripts/evaluation/statistical_abc_analysis.py --results_file results/abc_testing/comprehensive_abc_results.json

if [ $? -ne 0 ]; then
    echo "[ERROR] 統計的分析実行に失敗しました"
    exit 1
fi

# 3. グラフ可視化実行
echo "[3/4] グラフ可視化実行中..."
py -3 scripts/evaluation/visualize_abc_benchmark_statistics.py --results_file results/abc_testing/comprehensive_abc_results.json --statistical_analysis_file results/abc_testing/statistical_analysis.json

if [ $? -ne 0 ]; then
    echo "[ERROR] グラフ可視化実行に失敗しました"
    exit 1
fi

# 4. HF README更新（repo_idが必要）
echo "[4/4] HF README更新（repo_idを指定してください）"
echo "実行コマンド:"
echo "py -3 scripts/utils/update_hf_readme_with_benchmarks.py --repo_id <repo_id> --results_file results/abc_testing/comprehensive_abc_results.json --statistical_analysis_file results/abc_testing/statistical_analysis.json --visualization_dir results/abc_testing/visualizations"

echo "=== ABCテストパイプライン実行完了 ==="
