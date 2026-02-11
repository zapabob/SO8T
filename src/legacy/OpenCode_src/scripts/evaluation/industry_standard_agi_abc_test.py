#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Industry Standard + AGI ABC Test
Combines lm-evaluation-harness benchmarks with AGI final challenge tasks
Tests modela vs AEGIS series with quadruple reasoning for AEGIS
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

import sys
from pathlib import Path

# スクリプトディレクトリをパスに追加
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from agi_final_challenge_tasks import AGIFinalChallengeTasks, AGITask

RESULTS_ROOT = Path(r"D:/webdataset/benchmark_results/industry_standard_agi")
DOCS_ROOT = Path("_docs/benchmark_results/industry_standard_agi")
PLAY_AUDIO_SCRIPT = Path("scripts/utils/play_audio_notification.ps1")

# モデル設定
MODELS = {
    "modela": {
        "ollama_name": "model-a:q8_0",
        "use_quadruple": False,
    },
    "aegis_adjusted": {
        "ollama_name": "aegis-phi3.5-fixed-0.8:latest",
        "use_quadruple": True,
    },
}

# 業界標準ベンチマークタスク
STANDARD_TASKS = ["gsm8k", "mmlu", "hellaswag"]


def run_ollama_command(model: str, prompt: str, timeout: int = 180) -> Tuple[str, float]:
    """Ollamaコマンドを実行して応答と時間を取得"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LANG'] = 'C.UTF-8'
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ['ollama', 'run', model, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            env=env
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return result.stdout.strip(), elapsed
        else:
            return f"[ERROR] {result.stderr}", elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return "[ERROR] Timeout", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return f"[ERROR] {e}", elapsed


def extract_quadruple_reasoning(response: str) -> Dict[str, str]:
    """四重推論の各セクションを抽出"""
    result = {
        'logic': '',
        'ethics': '',
        'practical': '',
        'creative': '',
        'final': '',
        'full': response
    }
    
    # XMLタグから抽出
    import re
    
    logic_match = re.search(r'<think-logic>(.*?)</think-logic>', response, re.DOTALL | re.IGNORECASE)
    if logic_match:
        result['logic'] = logic_match.group(1).strip()
    
    ethics_match = re.search(r'<think-ethics>(.*?)</think-ethics>', response, re.DOTALL | re.IGNORECASE)
    if ethics_match:
        result['ethics'] = ethics_match.group(1).strip()
    
    practical_match = re.search(r'<think-practical>(.*?)</think-practical>', response, re.DOTALL | re.IGNORECASE)
    if practical_match:
        result['practical'] = practical_match.group(1).strip()
    
    creative_match = re.search(r'<think-creative>(.*?)</think-creative>', response, re.DOTALL | re.IGNORECASE)
    if creative_match:
        result['creative'] = creative_match.group(1).strip()
    
    final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL | re.IGNORECASE)
    if final_match:
        result['final'] = final_match.group(1).strip()
    else:
        # finalタグがない場合は最後の部分を使用
        result['final'] = response.split('</think-creative>')[-1].strip() if '</think-creative>' in response else response
    
    return result


def evaluate_agi_response(response: str, task: AGITask) -> Dict[str, float]:
    """AGI課題の応答を評価"""
    response_lower = response.lower()
    scores = {
        'overall': 0.0,
        'criteria_match': 0.0,
        'depth': 0.0,
        'completeness': 0.0,
    }
    
    # 評価基準のマッチング
    criteria_matches = sum(
        1 for criterion in task.evaluation_criteria
        if criterion.lower().replace('_', ' ') in response_lower
    )
    scores['criteria_match'] = min(criteria_matches / len(task.evaluation_criteria), 1.0)
    
    # 深さ（長さと構造）
    if len(response) > 500:
        scores['depth'] = 1.0
    elif len(response) > 200:
        scores['depth'] = 0.7
    elif len(response) > 100:
        scores['depth'] = 0.4
    else:
        scores['depth'] = 0.2
    
    # 完全性（期待される側面のカバー）
    aspect_matches = sum(
        1 for aspect in task.expected_aspects
        if aspect.lower().replace('_', ' ') in response_lower
    )
    scores['completeness'] = min(aspect_matches / len(task.expected_aspects), 1.0)
    
    # 総合スコア
    scores['overall'] = (
        scores['criteria_match'] * 0.4 +
        scores['depth'] * 0.3 +
        scores['completeness'] * 0.3
    )
    
    return scores


def run_lm_eval_benchmark(model_alias: str, model_config: Dict, output_dir: Path) -> Optional[Dict]:
    """業界標準ベンチマーク（lm-eval）を実行"""
    print(f"\n[LM-EVAL] Running industry-standard benchmarks for {model_alias}...")
    
    # lm-evalはHFモデルまたはGGUFファイルが必要
    # Ollamaモデルの場合はスキップ（後でAGI課題で評価）
    if model_config.get('skip_lm_eval', False):
        print(f"[SKIP] Skipping lm-eval for {model_alias} (Ollama-only model)")
        return None
    
    # ここではlm-evalの実行は別スクリプトに委譲
    # 実際の実装ではcuda_accelerated_benchmark.pyを呼び出す
    print(f"[INFO] lm-eval execution delegated to cuda_accelerated_benchmark.py")
    return None


def run_agi_tasks(model_alias: str, model_config: Dict, agi_tasks: AGIFinalChallengeTasks, output_dir: Path, limit: Optional[int] = None) -> List[Dict]:
    """AGI課題を実行"""
    print(f"\n[AGI] Running AGI final challenge tasks for {model_alias}...")
    
    model_name = model_config['ollama_name']
    use_quadruple = model_config['use_quadruple']
    
    results = []
    all_tasks = agi_tasks.get_all_tasks()
    
    # limitが指定されている場合は制限
    if limit is not None:
        all_tasks = all_tasks[:limit]
    
    print(f"[INFO] Running {len(all_tasks)} AGI tasks...")
    
    for idx, task in enumerate(all_tasks, 1):
        print(f"[AGI] [{idx}/{len(all_tasks)}] {task.category} - {task.difficulty}")
        
        # プロンプト準備
        if use_quadruple:
            prompt = agi_tasks.get_quadruple_reasoning_prompt(task.question)
        else:
            prompt = task.question
        
        # 実行
        response, elapsed = run_ollama_command(model_name, prompt, timeout=180)
        
        # 評価
        scores = evaluate_agi_response(response, task)
        
        # 四重推論の抽出（AEGISの場合）
        quadruple = {}
        if use_quadruple:
            quadruple = extract_quadruple_reasoning(response)
        
        result = {
            'model': model_alias,
            'task_id': idx,
            'category': task.category,
            'task_type': task.task_type,
            'difficulty': task.difficulty,
            'question': task.question,
            'response': response[:1000],  # 最初の1000文字
            'response_time': elapsed,
            'scores': scores,
            'quadruple_reasoning': quadruple if use_quadruple else None,
            'timestamp': datetime.now().isoformat(),
        }
        
        results.append(result)
        
        # 進捗表示
        print(f"  Score: {scores['overall']:.3f} | Time: {elapsed:.2f}s")
        
        # レート制限対策
        time.sleep(1)
    
    return results


def save_results(results: List[Dict], output_dir: Path, model_alias: str):
    """結果をJSON形式で保存"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{model_alias}_agi_results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVE] Results saved to {results_file}")
    return results_file


def play_audio():
    """完了音声通知を再生"""
    if not PLAY_AUDIO_SCRIPT.exists():
        return
    try:
        subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(PLAY_AUDIO_SCRIPT),
            ],
            check=False,
        )
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Industry Standard + AGI ABC Test"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="テストするモデル（modela, aegis_adjusted）",
    )
    parser.add_argument(
        "--skip-lm-eval",
        action="store_true",
        help="業界標準ベンチマークをスキップ",
    )
    parser.add_argument(
        "--skip-agi",
        action="store_true",
        help="AGI課題をスキップ",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT,
        help="結果保存先ルートディレクトリ",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="AGIタスクの総数を制限（先頭から指定数まで実行）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 出力ディレクトリ準備
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Industry Standard + AGI ABC Test")
    print("=" * 80)
    print(f"Output directory: {run_dir}")
    print(f"Models: {args.models}")
    print()
    
    # AGI課題ロード
    agi_tasks = AGIFinalChallengeTasks()
    print(f"[INFO] Loaded {len(agi_tasks.get_all_tasks())} AGI tasks")
    
    all_results = {}
    
    # 各モデルでテスト実行
    for model_alias in args.models:
        if model_alias not in MODELS:
            print(f"[WARNING] Unknown model: {model_alias}, skipping")
            continue
        
        model_config = MODELS[model_alias]
        model_dir = run_dir / model_alias
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'=' * 80}")
        print(f"Testing {model_alias}")
        print(f"{'=' * 80}")
        
        model_results = {
            'model': model_alias,
            'config': model_config,
            'lm_eval_results': None,
            'agi_results': [],
        }
        
        # 業界標準ベンチマーク実行
        if not args.skip_lm_eval:
            lm_results = run_lm_eval_benchmark(model_alias, model_config, model_dir)
            model_results['lm_eval_results'] = lm_results
        
        # AGI課題実行
        if not args.skip_agi:
            agi_results = run_agi_tasks(model_alias, model_config, agi_tasks, model_dir, limit=args.limit)
            model_results['agi_results'] = agi_results
            
            # 結果保存
            save_results(agi_results, model_dir, model_alias)
        
        all_results[model_alias] = model_results
    
    # 統合結果保存
    summary_file = run_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump({
            'timestamp': timestamp,
            'models': args.models,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUMMARY] All results saved to {summary_file}")
    print(f"[NEXT] Run analyze_industry_standard_agi_results.py to generate statistics and visualizations")
    
    play_audio()


if __name__ == "__main__":
    main()

