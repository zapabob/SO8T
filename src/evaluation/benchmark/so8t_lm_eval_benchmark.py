#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8)T LM-Evaluation-Harness Benchmark Script
HFモデルとGGUFモデルの両方をサポートしたベンチマーク実行
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SO8TLMEvaluator:
    """SO(8)T LM評価器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lm_eval_path = Path('./lm-evaluation-harness')
        self.results_dir = Path(config.get('results_dir', './benchmark_results/lm_eval'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_hf_benchmark(self, model_name_or_path: str, model_nickname: str,
                        tasks: List[str] = None, dtype: str = "bfloat16") -> Dict[str, Any]:
        """HFモデルベンチマーク実行"""
        logger.info(f"HFモデルベンチマーク開始: {model_nickname}")

        if tasks is None:
            tasks = ['hellaswag', 'mmlu']

        tasks_str = ','.join(tasks)

        # コマンド構築
        cmd = [
            'python', '-m', 'lm_eval',
            '--model', 'hf',
            '--model_args', f'pretrained={model_name_or_path},dtype={dtype}',
            '--tasks', tasks_str,
            '--device', 'cuda:0',
            '--batch_size', '8',
            '--output_path', str(self.results_dir / f'hf_{model_nickname}'),
            '--log_samples'
        ]

        logger.info(f"実行コマンド: {' '.join(cmd)}")

        try:
            # lm-evaluation-harnessディレクトリに移動して実行
            result = subprocess.run(
                cmd,
                cwd=str(self.lm_eval_path),
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"HFベンチマーク成功: {model_nickname}")

            # 結果ファイル読み込み
            results_file = self.results_dir / f'hf_{model_nickname}' / 'results.json'
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
            else:
                results_data = {'error': 'Results file not found'}

            return {
                'model_type': 'hf',
                'model_name': model_nickname,
                'full_model_path': model_name_or_path,
                'tasks': tasks,
                'dtype': dtype,
                'results': results_data,
                'command': ' '.join(cmd),
                'success': True
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"HFベンチマーク失敗: {model_nickname}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")

            return {
                'model_type': 'hf',
                'model_name': model_nickname,
                'full_model_path': model_name_or_path,
                'tasks': tasks,
                'dtype': dtype,
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr,
                'command': ' '.join(cmd),
                'success': False
            }

    def run_gguf_benchmark(self, gguf_dir: str, gguf_filename: str,
                          tokenizer_path: str, model_nickname: str,
                          tasks: List[str] = None) -> Dict[str, Any]:
        """GGUFモデルベンチマーク実行（HF backend経由）"""
        logger.info(f"GGUFモデルベンチマーク開始: {model_nickname}")

        if tasks is None:
            tasks = ['hellaswag', 'mmlu']

        tasks_str = ','.join(tasks)

        # コマンド構築
        cmd = [
            'python', '-m', 'lm_eval',
            '--model', 'hf',
            '--model_args', f'pretrained={gguf_dir},gguf_file={gguf_filename},tokenizer={tokenizer_path}',
            '--tasks', tasks_str,
            '--device', 'cuda:0',
            '--batch_size', '8',
            '--output_path', str(self.results_dir / f'gguf_{model_nickname}'),
            '--log_samples'
        ]

        logger.info(f"実行コマンド: {' '.join(cmd)}")

        try:
            # lm-evaluation-harnessディレクトリに移動して実行
            result = subprocess.run(
                cmd,
                cwd=str(self.lm_eval_path),
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"GGUFベンチマーク成功: {model_nickname}")

            # 結果ファイル読み込み
            results_file = self.results_dir / f'gguf_{model_nickname}' / 'results.json'
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
            else:
                results_data = {'error': 'Results file not found'}

            return {
                'model_type': 'gguf',
                'model_name': model_nickname,
                'gguf_dir': gguf_dir,
                'gguf_filename': gguf_filename,
                'tokenizer_path': tokenizer_path,
                'tasks': tasks,
                'results': results_data,
                'command': ' '.join(cmd),
                'success': True
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"GGUFベンチマーク失敗: {model_nickname}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")

            return {
                'model_type': 'gguf',
                'model_name': model_nickname,
                'gguf_dir': gguf_dir,
                'gguf_filename': gguf_filename,
                'tokenizer_path': tokenizer_path,
                'tasks': tasks,
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr,
                'command': ' '.join(cmd),
                'success': False
            }

    def run_ab_comparison(self, model_a_config: Dict[str, Any],
                         model_b_config: Dict[str, Any],
                         tasks: List[str] = None) -> Dict[str, Any]:
        """A/B比較ベンチマーク実行"""
        logger.info("A/B比較ベンチマーク開始")

        if tasks is None:
            tasks = ['hellaswag', 'mmlu']

        # model A実行
        logger.info("=== model A ベンチマーク実行 ===")
        if model_a_config['type'] == 'hf':
            result_a = self.run_hf_benchmark(
                model_a_config['path'],
                model_a_config['name'],
                tasks,
                model_a_config.get('dtype', 'bfloat16')
            )
        elif model_a_config['type'] == 'gguf':
            result_a = self.run_gguf_benchmark(
                model_a_config['gguf_dir'],
                model_a_config['gguf_filename'],
                model_a_config['tokenizer_path'],
                model_a_config['name'],
                tasks
            )

        # model B実行
        logger.info("=== model B ベンチマーク実行 ===")
        if model_b_config['type'] == 'hf':
            result_b = self.run_hf_benchmark(
                model_b_config['path'],
                model_b_config['name'],
                tasks,
                model_b_config.get('dtype', 'bfloat16')
            )
        elif model_b_config['type'] == 'gguf':
            result_b = self.run_gguf_benchmark(
                model_b_config['gguf_dir'],
                model_b_config['gguf_filename'],
                model_b_config['tokenizer_path'],
                model_b_config['name'],
                tasks
            )

        # 比較結果作成
        comparison = {
            'model_a': result_a,
            'model_b': result_b,
            'tasks': tasks,
            'timestamp': str(Path(self.results_dir) / 'comparison_results.json'),
            'summary': self.create_comparison_summary(result_a, result_b, tasks)
        }

        # 比較結果保存
        comparison_file = self.results_dir / 'comparison_results.json'
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info(f"A/B比較ベンチマーク完了: {comparison_file}")

        return comparison

    def create_comparison_summary(self, result_a: Dict[str, Any],
                                result_b: Dict[str, Any],
                                tasks: List[str]) -> Dict[str, Any]:
        """比較サマリー作成"""
        summary = {
            'model_a_name': result_a.get('model_name', 'Unknown'),
            'model_b_name': result_b.get('model_name', 'Unknown'),
            'tasks_evaluated': tasks,
            'task_results': {}
        }

        # 各タスクの比較
        for task in tasks:
            task_result = {
                'model_a_score': None,
                'model_b_score': None,
                'difference': None,
                'improvement': None
            }

            # model Aのスコア取得
            if result_a.get('success', False) and 'results' in result_a:
                results_a = result_a['results']
                if 'results' in results_a and task in results_a['results']:
                    task_result['model_a_score'] = results_a['results'][task].get('acc,none', None)

            # model Bのスコア取得
            if result_b.get('success', False) and 'results' in result_b:
                results_b = result_b['results']
                if 'results' in results_b and task in results_b['results']:
                    task_result['model_b_score'] = results_b['results'][task].get('acc,none', None)

            # 差分計算
            if task_result['model_a_score'] is not None and task_result['model_b_score'] is not None:
                diff = task_result['model_b_score'] - task_result['model_a_score']
                task_result['difference'] = diff
                task_result['improvement'] = diff > 0

            summary['task_results'][task] = task_result

        return summary

def create_so8t_config() -> Dict[str, Any]:
    """SO(8)Tベンチマーク設定"""
    return {
        'results_dir': './benchmark_results/lm_eval',
        'lm_eval_path': './lm-evaluation-harness',

        # model A: Boreas-phi3.5-instinct-jp (ベースモデル)
        'model_a': {
            'type': 'hf',
            'name': 'borea_phi35_base',
            'path': 'Boreas/phi-3.5-mini-instruct-Jp',
            'dtype': 'bfloat16'
        },

        # model B: SO(8)T学習済みモデル
        'model_b': {
            'type': 'gguf',
            'name': 'borea_phi35_so8t_ppo',
            'gguf_dir': 'D:/webdataset/gguf_models/borea_phi35_so8t_ppo',
            'gguf_filename': 'borea_phi35_so8t_ppo_Q8_0.gguf',
            'tokenizer_path': './checkpoints/sft_so8t/final_model'
        },

        # ベンチマークタスク
        'tasks': ['hellaswag', 'mmlu'],

        # バッチサイズ
        'batch_size': 8,

        # ログ設定
        'log_samples': True
    }

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='SO(8)T LM-Evaluation Benchmark')
    parser.add_argument('--model-a-only', action='store_true', help='Only benchmark model A')
    parser.add_argument('--model-b-only', action='store_true', help='Only benchmark model B')
    parser.add_argument('--tasks', nargs='+', default=['hellaswag', 'mmlu'],
                       help='Benchmark tasks')
    parser.add_argument('--ab-compare', action='store_true', help='Run A/B comparison')

    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 設定
    config = create_so8t_config()
    config['tasks'] = args.tasks

    # 評価器作成
    evaluator = SO8TLMEvaluator(config)

    if args.ab_compare:
        # A/B比較実行
        print("[START] SO(8)T A/B比較ベンチマーク実行")
        print("=" * 60)

        comparison = evaluator.run_ab_comparison(
            config['model_a'],
            config['model_b'],
            config['tasks']
        )

        print("[OK] A/B比較ベンチマーク完了!")
        print(f"[STATS] 結果保存先: {config['results_dir']}/comparison_results.json")

        # サマリー表示
        summary = comparison['summary']
        print(f"\n[TARGET] 比較サマリー:")
        print(f"Model A: {summary['model_a_name']}")
        print(f"Model B: {summary['model_b_name']}")
        print(f"評価タスク: {', '.join(summary['tasks_evaluated'])}")

        for task, result in summary['task_results'].items():
            print(f"\n{task.upper()}:")
            if result['model_a_score'] is not None and result['model_b_score'] is not None:
                print(f"  Model A スコア: {result['model_a_score']:.4f}")
                print(f"  Model B スコア: {result['model_b_score']:.4f}")
                print(f"  差分: {result['difference']:.4f}")
                print(f"  改善: {'[OK]' if result.get('improvement', False) else '[NG]'}")
            else:
                print("  データなし")

    elif args.model_a_only:
        # model Aのみ実行
        print("[START] Model A ベンチマーク実行")
        result = evaluator.run_hf_benchmark(
            config['model_a']['path'],
            config['model_a']['name'],
            config['tasks'],
            config['model_a'].get('dtype', 'bfloat16')
        )
        print(f"[OK] Model A ベンチマーク完了: {result.get('success', False)}")

    elif args.model_b_only:
        # model Bのみ実行
        print("[START] Model B ベンチマーク実行")
        result = evaluator.run_gguf_benchmark(
            config['model_b']['gguf_dir'],
            config['model_b']['gguf_filename'],
            config['model_b']['tokenizer_path'],
            config['model_b']['name'],
            config['tasks']
        )
        print(f"[OK] Model B ベンチマーク完了: {result.get('success', False)}")

    else:
        # デフォルト: A/B比較
        print("[START] SO(8)T A/B比較ベンチマーク実行（デフォルト）")
        comparison = evaluator.run_ab_comparison(
            config['model_a'],
            config['model_b'],
            config['tasks']
        )
        print("[OK] A/B比較ベンチマーク完了!")

    # 音声通知
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts\\utils\\play_audio_notification.ps1"
        ], check=True)
    except Exception as e:
        print(f"[WARNING] 音声通知失敗: {e}")

if __name__ == "__main__":
    main()
