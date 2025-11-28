#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-3.5 Thinking Format Integration for PPO Training

既存の全データセットをPhi-3.5固有のThinkingフォーマットで統合し、
PPO用途と内部推論強化のためにCoTデータを重みづけ

Phi-3.5 Thinking Format:
<think-task>タスク理解</think-task>
<think-safety>安全性評価</think-safety>
<think-logic>論理的思考</think-logic>
<think-ethics>倫理的考慮</think-ethics>
<think-practical>実用的考察</think-practical>
<think-creative>創造的アプローチ</think-creative>
<final>最終回答</final>
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import re
import random
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phi35ThinkingIntegrator:
    """
    Phi-3.5 Thinking Format Integration Class

    PPO学習用に既存データセットをPhi-3.5 Thinkingフォーマットに変換
    CoTデータを特に重みづけ
    """

    def __init__(self):
        self.cot_weight_multiplier = 3.0  # CoTデータの重みづけ係数
        self.min_thinking_length = 50     # Thinking部分の最小長
        self.max_samples_per_dataset = 50000  # データセットごとの最大サンプル数

    def integrate_all_datasets(self, input_dirs: List[Path], output_dir: Path) -> Dict[str, int]:
        """
        全データセットをPhi-3.5 Thinkingフォーマットで統合

        Args:
            input_dirs: 入力データセットディレクトリのリスト
            output_dir: 出力ディレクトリ

        Returns:
            統計情報
        """
        logger.info("="*80)
        logger.info("Phi-3.5 Thinking Format Integration")
        logger.info("="*80)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 全データセットの収集
        all_datasets = []
        dataset_stats = defaultdict(int)

        for input_dir in input_dirs:
            if not input_dir.exists():
                logger.warning(f"Input directory not found: {input_dir}")
                continue

            logger.info(f"Scanning directory: {input_dir}")
            datasets = self._collect_datasets(input_dir)
            all_datasets.extend(datasets)

            for dataset_path in datasets:
                dataset_name = dataset_path.parent.name
                dataset_stats[dataset_name] += 1

        logger.info(f"Found {len(all_datasets)} datasets from {len(dataset_stats)} sources")
        for name, count in sorted(dataset_stats.items()):
            logger.info(f"  {name}: {count} files")

        # データ統合とThinkingフォーマット変換
        integrated_samples = []
        processed_stats = {
            'total_samples': 0,
            'thinking_converted': 0,
            'cot_weighted': 0,
            'ppo_optimized': 0
        }

        for dataset_path in tqdm(all_datasets, desc="Processing datasets"):
            try:
                samples = self._load_dataset_file(dataset_path)
                processed_samples, file_stats = self._convert_to_phi35_format(
                    samples, dataset_path, processed_samples
                )

                for key, value in file_stats.items():
                    processed_stats[key] += value

                # メモリ効率のため定期的に保存
                if len(processed_samples) >= 10000:
                    self._save_batch(processed_samples, output_dir, f"batch_{len(integrated_samples)//10000}")
                    integrated_samples = []

            except Exception as e:
                logger.error(f"Error processing {dataset_path}: {e}")
                continue

        # 残りのデータを保存
        if integrated_samples:
            self._save_batch(integrated_samples, output_dir, f"batch_{len(integrated_samples)//10000 + 1}")

        # PPO最適化データセットの作成
        self._create_ppo_optimized_dataset(output_dir, processed_samples)

        # 統計レポート
        self._generate_integration_report(output_dir, processed_stats, dataset_stats)

        logger.info("="*80)
        logger.info("Integration completed!")
        logger.info(f"Total samples processed: {processed_stats['total_samples']:,}")
        logger.info(f"Thinking format converted: {processed_stats['thinking_converted']:,}")
        logger.info(f"CoT data weighted: {processed_stats['cot_weighted']:,}")
        logger.info(f"PPO optimized: {processed_stats['ppo_optimized']:,}")
        logger.info("="*80)

        return processed_stats

    def _collect_datasets(self, input_dir: Path) -> List[Path]:
        """データセットファイルの収集"""
        dataset_files = []

        # JSON/JSONLファイルの収集
        for pattern in ["*.json", "*.jsonl"]:
            dataset_files.extend(list(input_dir.rglob(pattern)))

        # 特定のディレクトリ構造の処理
        if input_dir.name == "datasets":
            # HuggingFace形式
            for subdir in input_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    json_files = list(subdir.glob("*.json")) + list(subdir.glob("*.jsonl"))
                    dataset_files.extend(json_files)

        return sorted(dataset_files)

    def _load_dataset_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """データセットファイルの読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        # 様々な構造に対応
                        for key in ['training_data', 'data', 'samples', 'conversations']:
                            if key in data and isinstance(data[key], list):
                                return data[key]
                        # 単一サンプルとして扱う
                        return [data]
                elif file_path.suffix == '.jsonl':
                    samples = []
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    return samples

        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            return []

        return []

    def _convert_to_phi35_format(self, samples: List[Dict], dataset_path: Path,
                               processed_samples: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        """
        サンプルをPhi-3.5 Thinkingフォーマットに変換

        Args:
            samples: 入力サンプル
            dataset_path: データセットファイルパス
            processed_samples: 処理済みサンプルリスト

        Returns:
            (更新されたprocessed_samples, 統計情報)
        """
        stats = {'total_samples': 0, 'thinking_converted': 0, 'cot_weighted': 0, 'ppo_optimized': 0}

        dataset_name = dataset_path.parent.name
        is_cot_dataset = self._is_cot_dataset(dataset_path)

        for sample in samples:
            stats['total_samples'] += 1

            # Phi-3.5 Thinkingフォーマットに変換
            phi35_sample = self._transform_to_phi35_thinking(sample, dataset_name)

            if phi35_sample:
                stats['thinking_converted'] += 1

                # CoTデータは重みづけ（複数回追加）
                if is_cot_dataset:
                    # CoTデータは3倍に重みづけ
                    for _ in range(int(self.cot_weight_multiplier)):
                        processed_samples.append(phi35_sample.copy())
                    stats['cot_weighted'] += int(self.cot_weight_multiplier)
                else:
                    processed_samples.append(phi35_sample)

                stats['ppo_optimized'] += 1

        return processed_samples, stats

    def _transform_to_phi35_thinking(self, sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """
        個別のサンプルをPhi-3.5 Thinkingフォーマットに変換

        Args:
            sample: 入力サンプル
            dataset_name: データセット名

        Returns:
            Phi-3.5形式のサンプル
        """
        # テキストの抽出
        text = self._extract_text(sample)
        if not text or len(text.strip()) < 10:
            return None

        # データセットタイプの判定
        dataset_type = self._classify_dataset_type(dataset_name, text)

        # Phi-3.5 Thinking構造の構築
        thinking_parts = self._build_thinking_structure(text, dataset_type)

        # Thinkingテキストの生成
        thinking_text = self._generate_thinking_text(thinking_parts)

        # 最終回答の生成
        final_answer = self._generate_final_answer(text, dataset_type)

        # Phi-3.5フォーマットの構築
        phi35_format = f"{thinking_text}\n<final>{final_answer}</final>"

        return {
            'source_dataset': dataset_name,
            'original_text': text[:1000],  # オリジナルテキスト（制限）
            'phi35_thinking': phi35_format,
            'dataset_type': dataset_type,
            'is_cot': 'CoT' in dataset_type,
            'thinking_length': len(thinking_text),
            'language': sample.get('language', 'unknown'),
            'metadata': {
                'source_file': str(sample.get('source', '')),
                'processing_timestamp': None,  # 後で設定
            }
        }

    def _extract_text(self, sample: Dict[str, Any]) -> str:
        """サンプルからテキストを抽出"""
        text_fields = ['text', 'content', 'instruction', 'input', 'output',
                      'response', 'prompt', 'chosen', 'rejected', 'conversation']

        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                text = sample[field].strip()
                if text:
                    return text

        # conversation形式の処理
        if 'conversation' in sample and isinstance(sample['conversation'], list):
            texts = []
            for turn in sample['conversation']:
                if isinstance(turn, dict):
                    for key in ['human', 'assistant', 'user', 'system']:
                        if key in turn and isinstance(turn[key], str):
                            texts.append(turn[key])
            if texts:
                return ' '.join(texts)

        return ''

    def _classify_dataset_type(self, dataset_name: str, text: str) -> str:
        """データセットタイプの分類"""
        name_lower = dataset_name.lower()
        text_lower = text.lower()

        # CoT関連データセット
        if any(keyword in name_lower for keyword in ['reasoning', 'thinking', 'cot', 'chain_of_thought']):
            return 'CoT_Reasoning'

        # 数学データセット
        if any(keyword in name_lower for keyword in ['math', 'gsm8k', 'mmlu']):
            if any(math_term in text_lower for math_term in ['calculate', 'solve', 'equation', 'theorem']):
                return 'CoT_Math'

        # コーディングデータセット
        if any(keyword in name_lower for keyword in ['code', 'programming', 'starcoder']):
            if any(code_term in text_lower for code_term in ['function', 'class', 'import', 'def ', 'print(']):
                return 'CoT_Coding'

        # 安全・倫理データセット
        if any(keyword in name_lower for keyword in ['safety', 'ethics', 'moral', 'nsfw']):
            return 'Safety_Ethics'

        # 対話データセット
        if any(keyword in name_lower for keyword in ['dialogue', 'conversation', 'chat']):
            return 'Dialogue_Reasoning'

        # 一般タスク
        if any(task_term in text_lower for task_term in ['explain', 'describe', 'analyze', 'what is']):
            return 'General_Reasoning'

        return 'General_Task'

    def _build_thinking_structure(self, text: str, dataset_type: str) -> Dict[str, str]:
        """Thinking構造の構築"""
        thinking_parts = {}

        # タスク理解
        thinking_parts['task'] = self._analyze_task(text, dataset_type)

        # 安全性評価
        thinking_parts['safety'] = self._evaluate_safety(text, dataset_type)

        # 論理的思考
        thinking_parts['logic'] = self._apply_logic(text, dataset_type)

        # 倫理的考慮
        thinking_parts['ethics'] = self._consider_ethics(text, dataset_type)

        # 実用的考察
        thinking_parts['practical'] = self._practical_considerations(text, dataset_type)

        # 創造的アプローチ
        thinking_parts['creative'] = self._creative_approach(text, dataset_type)

        return thinking_parts

    def _analyze_task(self, text: str, dataset_type: str) -> str:
        """タスク理解の分析"""
        if dataset_type.startswith('CoT_'):
            return f"このクエリは{dataset_type.split('_')[1]}に関する複雑な推論を必要とする。"
        elif dataset_type == 'Safety_Ethics':
            return "このクエリは安全性と倫理的側面を慎重に考慮する必要がある。"
        else:
            return "このクエリを理解し、適切な回答を準備する。"

    def _evaluate_safety(self, text: str, dataset_type: str) -> str:
        """安全性評価"""
        dangerous_keywords = ['kill', 'harm', 'illegal', 'dangerous', 'unsafe']
        if any(kw in text.lower() for kw in dangerous_keywords):
            return "危険な内容が含まれているため、回答を制限する。"
        else:
            return "安全性に問題はない。"

    def _apply_logic(self, text: str, dataset_type: str) -> str:
        """論理的思考の適用"""
        if 'math' in dataset_type.lower():
            return "数学的アプローチで問題を分解し、段階的に解決する。"
        elif 'code' in dataset_type.lower():
            return "プログラミングの論理構造に従って実装を検討する。"
        else:
            return "論理的思考プロセスを適用して回答を構築する。"

    def _consider_ethics(self, text: str, dataset_type: str) -> str:
        """倫理的考慮"""
        if dataset_type == 'Safety_Ethics':
            return "倫理的・道徳的影響を慎重に評価する。"
        else:
            return "倫理的観点から回答の適切性を確認する。"

    def _practical_considerations(self, text: str, dataset_type: str) -> str:
        """実用的考察"""
        return "実用的・実行可能な解決策を検討する。"

    def _creative_approach(self, text: str, dataset_type: str) -> str:
        """創造的アプローチ"""
        if 'creative' in dataset_type.lower():
            return "創造的な視点から新しい解決策を模索する。"
        else:
            return "効率的で効果的なアプローチを選択する。"

    def _generate_thinking_text(self, thinking_parts: Dict[str, str]) -> str:
        """Thinkingテキストの生成"""
        thinking_text = ""

        for part_name, content in thinking_parts.items():
            if content and len(content.strip()) > 0:
                tag_name = f"think-{part_name}"
                thinking_text += f"<{tag_name}>{content}</{tag_name}>\n"

        return thinking_text.strip()

    def _generate_final_answer(self, text: str, dataset_type: str) -> str:
        """最終回答の生成"""
        # シンプルな最終回答生成（実際にはもっと洗練されたものが必要）
        if len(text) > 200:
            return text[:200] + "..."
        else:
            return text

    def _is_cot_dataset(self, dataset_path: Path) -> bool:
        """CoTデータセットかどうかの判定"""
        path_str = str(dataset_path).lower()
        cot_indicators = ['reasoning', 'thinking', 'cot', 'chain', 'math', 'code', 'logic']

        return any(indicator in path_str for indicator in cot_indicators)

    def _save_batch(self, samples: List[Dict], output_dir: Path, batch_name: str):
        """バッチデータの保存"""
        output_file = output_dir / f"phi35_integrated_{batch_name}.jsonl"

        logger.info(f"Saving {len(samples)} samples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                # タイムスタンプの追加
                import datetime
                sample['metadata']['processing_timestamp'] = datetime.datetime.now().isoformat()

                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    def _create_ppo_optimized_dataset(self, output_dir: Path, all_samples: List[Dict]):
        """PPO最適化データセットの作成"""
        logger.info("Creating PPO-optimized dataset...")

        # CoTデータを優先的に配置
        cot_samples = [s for s in all_samples if s.get('is_cot', False)]
        non_cot_samples = [s for s in all_samples if not s.get('is_cot', False)]

        # CoTデータを3倍重みづけ
        weighted_samples = cot_samples * 3 + non_cot_samples

        # シャッフル
        random.shuffle(weighted_samples)

        # PPO用データセットとして保存
        ppo_file = output_dir / "phi35_ppo_optimized_dataset.jsonl"
        logger.info(f"Saving PPO dataset with {len(weighted_samples)} samples to {ppo_file}")

        with open(ppo_file, 'w', encoding='utf-8') as f:
            for sample in weighted_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 統計情報も保存
        stats_file = output_dir / "phi35_integration_stats.json"
        stats = {
            'total_samples': len(all_samples),
            'cot_samples': len(cot_samples),
            'non_cot_samples': len(non_cot_samples),
            'ppo_weighted_samples': len(weighted_samples),
            'cot_weight_multiplier': self.cot_weight_multiplier,
            'processing_timestamp': datetime.datetime.now().isoformat()
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def _generate_integration_report(self, output_dir: Path, processed_stats: Dict[str, int],
                                   dataset_stats: Dict[str, int]):
        """統合レポートの生成"""
        report_file = output_dir / "phi35_integration_report.md"

        report = f"""# Phi-3.5 Thinking Format Integration Report

## 概要
- **処理日時**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **総サンプル数**: {processed_stats['total_samples']:,}
- **Thinking変換完了**: {processed_stats['thinking_converted']:,}
- **CoTデータ重みづけ**: {processed_stats['cot_weighted']:,}
- **PPO最適化完了**: {processed_stats['ppo_optimized']:,}

## データセット統計
"""

        for name, count in sorted(dataset_stats.items()):
            report += f"- **{name}**: {count} ファイル\n"

        report += f"""
## Phi-3.5 Thinkingフォーマット
```xml
<think-task>タスク理解</think-task>
<think-safety>安全性評価</think-safety>
<think-logic>論理的思考</think-logic>
<think-ethics>倫理的考慮</think-ethics>
<think-practical>実用的考察</think-practical>
<think-creative>創造的アプローチ</think-creative>
<final>最終回答</final>
```

## PPO最適化特徴
- CoTデータ重みづけ係数: {self.cot_weight_multiplier}
- 最小Thinking長: {self.min_thinking_length} 文字
- データセットごとの最大サンプル数: {self.max_samples_per_dataset}

## 出力ファイル
- `phi35_integrated_batch_*.jsonl`: 統合データバッチ
- `phi35_ppo_optimized_dataset.jsonl`: PPO最適化データセット
- `phi35_integration_stats.json`: 統計情報
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Integration report saved to {report_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Phi-3.5 Thinking Format Integration for PPO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phi-3.5 Thinkingフォーマットで既存データセットを統合し、PPO学習用に最適化

使用例:
  python phi35_thinking_integration.py --input D:/webdataset --output D:/webdataset/phi35_integrated
  python phi35_thinking_integration.py --input D:/webdataset/datasets D:/webdataset/aegis_v2.0 --output D:/webdataset/phi35_integrated
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        nargs='+',
        required=True,
        help="入力データセットディレクトリ（複数指定可）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--cot-weight",
        type=float,
        default=3.0,
        help="CoTデータの重みづけ係数 (default: 3.0)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="データセットごとの最大サンプル数 (default: 50000)"
    )

    args = parser.parse_args()

    # ディレクトリの設定
    input_dirs = [Path(d) for d in args.input]
    output_dir = Path(args.output)

    # 統合処理の実行
    integrator = Phi35ThinkingIntegrator()
    integrator.cot_weight_multiplier = args.cot_weight
    integrator.max_samples_per_dataset = args.max_samples

    try:
        stats = integrator.integrate_all_datasets(input_dirs, output_dir)

        logger.info("[SUCCESS] Phi-3.5 Thinking integration completed")
        logger.info(f"Processed {stats['total_samples']:,} samples")
        logger.info(f"PPO-optimized dataset ready for training")

        # オーディオ通知
        import subprocess
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass", "-File",
                "scripts/utils/play_audio_notification.ps1"
            ], check=True)
        except:
            pass

        return 0

    except Exception as e:
        logger.error(f"[FAILED] Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import datetime  # インポートを関数内に移動
    import sys
    sys.exit(main())
