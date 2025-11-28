#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
既存統合データセットをPhi-3.5 Thinkingフォーマットに変換

PPO用途と内部推論強化のためにCoTデータを重みづけ
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import random
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phi35Converter:
    """Phi-3.5 Thinkingフォーマット変換クラス"""

    def __init__(self, cot_weight_multiplier: float = 3.0):
        self.cot_weight_multiplier = cot_weight_multiplier

    def convert_integrated_dataset(self, input_file: Path, output_dir: Path):
        """統合データセットをPhi-3.5フォーマットに変換"""
        logger.info("="*80)
        logger.info("Converting Integrated Dataset to Phi-3.5 Thinking Format")
        logger.info("="*80)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 入力ファイルの読み込み
        logger.info(f"Loading integrated dataset: {input_file}")
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading samples"):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(samples):,} samples")

        # Phi-3.5フォーマットに変換
        phi35_samples = []
        stats = {'total': 0, 'converted': 0, 'cot_weighted': 0}

        for sample in tqdm(samples, desc="Converting to Phi-3.5 format"):
            stats['total'] += 1

            # Phi-3.5 Thinkingフォーマットに変換
            phi35_sample = self._convert_sample_to_phi35(sample)
            if phi35_sample:
                stats['converted'] += 1
                phi35_samples.append(phi35_sample)

                # CoTデータは重みづけ
                if self._is_cot_sample(sample):
                    for _ in range(int(self.cot_weight_multiplier - 1)):
                        phi35_samples.append(phi35_sample.copy())
                    stats['cot_weighted'] += int(self.cot_weight_multiplier)

        # シャッフル
        random.shuffle(phi35_samples)

        # PPO最適化データセットとして保存
        output_file = output_dir / "phi35_ppo_optimized_integrated.jsonl"
        logger.info(f"Saving {len(phi35_samples):,} samples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in phi35_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 統計情報保存
        stats_file = output_dir / "phi35_conversion_stats.json"
        stats.update({
            'final_samples': len(phi35_samples),
            'cot_weight_multiplier': self.cot_weight_multiplier,
            'processing_timestamp': datetime.datetime.now().isoformat()
        })

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # レポート生成
        self._generate_conversion_report(output_dir, stats)

        logger.info("="*80)
        logger.info("Conversion completed!")
        logger.info(f"Original samples: {stats['total']:,}")
        logger.info(f"Converted samples: {stats['converted']:,}")
        logger.info(f"Final PPO dataset: {len(phi35_samples):,}")
        logger.info("="*80)

    def _convert_sample_to_phi35(self, sample: dict) -> dict:
        """個別のサンプルをPhi-3.5フォーマットに変換"""
        text = sample.get('text', '')
        if not text or len(text.strip()) < 10:
            return None

        dataset_name = sample.get('dataset', 'unknown')

        # データセットタイプの判定
        dataset_type = self._classify_dataset_type(dataset_name, text)

        # Phi-3.5 Thinking構造の構築
        thinking_parts = self._build_thinking_structure(text, dataset_type)

        # Thinkingテキストの生成
        thinking_text = ""
        for part_name, content in thinking_parts.items():
            if content:
                thinking_text += f"<think-{part_name}>{content}</think-{part_name}>\n"

        # 最終回答
        final_answer = self._generate_final_answer(text, dataset_type)

        phi35_format = f"{thinking_text}<final>{final_answer}</final>"

        return {
            'source_dataset': dataset_name,
            'original_text': text[:1000],
            'phi35_thinking': phi35_format,
            'dataset_type': dataset_type,
            'is_cot': 'CoT' in dataset_type,
            'language': sample.get('language', 'unknown'),
            'processing_timestamp': datetime.datetime.now().isoformat()
        }

    def _classify_dataset_type(self, dataset_name: str, text: str) -> str:
        """データセットタイプの分類"""
        name_lower = dataset_name.lower()
        text_lower = text.lower()

        # CoT関連
        if any(kw in name_lower for kw in ['reasoning', 'thinking', 'cot', 'chain']):
            return 'CoT_Reasoning'

        # 数学
        if any(kw in name_lower for kw in ['math', 'gsm8k', 'mmlu']):
            return 'CoT_Math'

        # コーディング
        if any(kw in name_lower for kw in ['code', 'programming', 'starcoder']):
            return 'CoT_Coding'

        # 安全・倫理
        if any(kw in name_lower for kw in ['safety', 'ethics', 'nsfw']):
            return 'Safety_Ethics'

        # 一般
        return 'General_Task'

    def _build_thinking_structure(self, text: str, dataset_type: str) -> dict:
        """Thinking構造の構築"""
        return {
            'task': self._analyze_task(text, dataset_type),
            'safety': self._evaluate_safety(text, dataset_type),
            'logic': self._apply_logic(text, dataset_type),
            'ethics': self._consider_ethics(text, dataset_type),
            'practical': "実用的観点から検討する。",
            'creative': "効果的な解決策を考える。"
        }

    def _analyze_task(self, text: str, dataset_type: str) -> str:
        if 'CoT' in dataset_type:
            return f"{dataset_type.split('_')[1]}に関する推論タスクを理解した。"
        return "タスクの内容を理解した。"

    def _evaluate_safety(self, text: str, dataset_type: str) -> str:
        dangerous = ['kill', 'harm', 'illegal', 'dangerous']
        if any(kw in text.lower() for kw in dangerous):
            return "安全性に問題があるため慎重に扱う。"
        return "安全性に問題はない。"

    def _apply_logic(self, text: str, dataset_type: str) -> str:
        if 'Math' in dataset_type:
            return "数学的論理を適用して問題を解決する。"
        elif 'Coding' in dataset_type:
            return "プログラミングの論理構造に従う。"
        return "論理的思考を適用する。"

    def _consider_ethics(self, text: str, dataset_type: str) -> str:
        if dataset_type == 'Safety_Ethics':
            return "倫理的影響を慎重に考慮する。"
        return "倫理的観点を考慮する。"

    def _generate_final_answer(self, text: str, dataset_type: str) -> str:
        """最終回答の生成（簡易版）"""
        if len(text) > 200:
            return text[:200] + "..."
        return text

    def _is_cot_sample(self, sample: dict) -> bool:
        """CoTサンプルかどうかの判定"""
        dataset_name = sample.get('dataset', '').lower()
        text = sample.get('text', '').lower()

        cot_indicators = [
            'reasoning', 'thinking', 'cot', 'chain', 'math', 'code',
            'calculate', 'solve', 'explain', 'analyze'
        ]

        return any(indicator in dataset_name or indicator in text for indicator in cot_indicators)

    def _generate_conversion_report(self, output_dir: Path, stats: dict):
        """変換レポートの生成"""
        report_file = output_dir / "phi35_conversion_report.md"

        report = f"""# Phi-3.5 Thinking Format Conversion Report

## 概要
- **処理日時**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **元サンプル数**: {stats['total']:,}
- **変換完了**: {stats['converted']:,}
- **CoT重みづけ**: {stats['cot_weighted']:,}
- **最終データセット**: {stats['final_samples']:,}

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
- CoTデータ重みづけ係数: {stats['cot_weight_multiplier']}
- シャッフル適用: 有効
- CoTサンプル判定: 自動

## 出力ファイル
- `phi35_ppo_optimized_integrated.jsonl`: PPO最適化統合データセット
- `phi35_conversion_stats.json`: 統計情報
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Conversion report saved to {report_file}")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert integrated dataset to Phi-3.5 Thinking format")
    parser.add_argument("--input", type=str, required=True, help="Input integrated dataset file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--cot-weight", type=float, default=3.0, help="CoT weight multiplier")

    args = parser.parse_args()

    converter = Phi35Converter(cot_weight_multiplier=args.cot_weight)
    converter.convert_integrated_dataset(Path(args.input), Path(args.output))

    # オーディオ通知
    import subprocess
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-File",
            "scripts/utils/play_audio_notification.ps1"
        ], check=True)
    except:
        pass


if __name__ == "__main__":
    main()
