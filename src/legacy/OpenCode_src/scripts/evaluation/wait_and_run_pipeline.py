#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABCテストパイプライン実行待機スクリプト
ABCテストの実行完了を待って、自動的に次のステップを実行
"""

import time
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_file(file_path: Path, timeout: int = 3600, check_interval: int = 30):
    """ファイルの生成を待つ"""
    logger.info(f"[WAIT] Waiting for {file_path} (timeout: {timeout}s)")
    start_time = time.time()
    
    while not file_path.exists():
        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.error(f"[TIMEOUT] {file_path} not found after {timeout}s")
            return False
        
        logger.info(f"[WAIT] Waiting... ({int(elapsed)}s elapsed)")
        time.sleep(check_interval)
    
    logger.info(f"[OK] {file_path} found after {int(time.time() - start_time)}s")
    return True

def run_command(cmd: list, description: str):
    """コマンドを実行"""
    logger.info(f"[RUN] {description}")
    logger.info(f"[CMD] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"[ERROR] {description} failed")
        logger.error(f"[STDERR] {result.stderr}")
        return False
    
    logger.info(f"[OK] {description} completed")
    if result.stdout:
        logger.info(f"[STDOUT] {result.stdout[:500]}")  # 最初の500文字のみ
    
    return True

def main():
    """メイン関数"""
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results" / "abc_testing"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "comprehensive_abc_results.json"
    stats_file = results_dir / "statistical_analysis.json"
    viz_dir = results_dir / "visualizations"
    
    # 1. ABCテスト結果を待つ
    logger.info("=== Step 1: Waiting for ABC test results ===")
    if not wait_for_file(results_file, timeout=7200):  # 2時間タイムアウト
        logger.error("[FAIL] ABC test results not found")
        return 1
    
    # 2. 統計的分析実行
    logger.info("=== Step 2: Running statistical analysis ===")
    if not run_command(
        ["py", "-3", "scripts/evaluation/statistical_abc_analysis.py",
         "--results_file", str(results_file)],
        "Statistical analysis"
    ):
        return 1
    
    # 3. 統計的分析結果を待つ
    if not wait_for_file(stats_file, timeout=300):  # 5分タイムアウト
        logger.error("[FAIL] Statistical analysis results not found")
        return 1
    
    # 4. グラフ可視化実行
    logger.info("=== Step 3: Generating visualizations ===")
    if not run_command(
        ["py", "-3", "scripts/evaluation/visualize_abc_benchmark_statistics.py",
         "--results_file", str(results_file),
         "--statistical_analysis_file", str(stats_file)],
        "Visualization"
    ):
        return 1
    
    # 5. グラフファイルの確認
    viz_dir.mkdir(parents=True, exist_ok=True)
    chart_files = [
        "abc_industry_standard_benchmarks.png",
        "abc_advanced_benchmarks.png",
        "abc_elyza100_benchmark.png"
    ]
    
    logger.info("=== Step 4: Checking visualization files ===")
    for chart_file in chart_files:
        chart_path = viz_dir / chart_file
        if chart_path.exists():
            logger.info(f"[OK] {chart_file} generated")
        else:
            logger.warning(f"[WARN] {chart_file} not found")
    
    logger.info("=== Pipeline completed successfully ===")
    logger.info(f"[RESULTS] Results: {results_file}")
    logger.info(f"[STATS] Statistics: {stats_file}")
    logger.info(f"[VIZ] Visualizations: {viz_dir}")
    logger.info("[NEXT] Run HF README update with:")
    logger.info(f"  py -3 scripts/utils/update_hf_readme_with_benchmarks.py --repo_id <repo_id> --results_file {results_file} --statistical_analysis_file {stats_file} --visualization_dir {viz_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
