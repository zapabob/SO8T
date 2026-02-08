#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8T Phi-3.5 Internal Tagging System
四値分類データセットにPhi-3.5内部タグを付与するシステム

Phi-3.5内部タグ:
- <|think|>: 思考プロセス開始
- <|observation|>: 観察/知覚段階
- <|deduction|>: 演繹的推論段階
- <|abduction|>: 帰納的推論段階
- <|integration|>: 統合/総合段階
- <|final|>: 最終回答

著者: AI Agent (峯岸亮ボブにゃん理論実装)
日付: 2025-11-30
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from tqdm import tqdm

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class Phi35TaggingConfig:
    """Phi-3.5タグ付け設定"""
    input_dataset: str = "data/so8t_advanced_integrated"
    output_dataset: str = "data/so8t_phi35_tagged"
    enable_thinking_process: bool = True
    thinking_depth: str = "full"  # "minimal", "standard", "full"
    chaos_factor: float = 0.1     # 思考プロセスのランダム性

class Phi35InternalTagger:
    """Phi-3.5内部タグ付けクラス"""

    def __init__(self, config: Phi35TaggingConfig):
        self.config = config

        # Phi-3.5内部タグ定義
        self.thinking_tags = {
            'start': '<|think|>',
            'observation': '<|observation|>',
            'deduction': '<|deduction|>',
            'abduction': '<|abduction|>',
            'integration': '<|integration|>',
            'final': '<|final|>',
            'end': ''
        }

        # タグごとの思考テンプレート
        self.thinking_templates = {
            'allow': {
                'minimal': [
                    "{start} 質問を理解する：{instruction}\n{observation} これは単純な質問である\n{final} {output}"
                ],
                'standard': [
                    "{start} 質問を分析する：{instruction}\n{observation} 基本的な知識に基づく質問である\n{final} {output}"
                ],
                'full': [
                    "{start} 質問の理解を開始\n{observation} 質問内容：{instruction}\n{final} 直接回答：{output}"
                ]
            },
            'escalation': {
                'minimal': [
                    "{start} 複雑な問題を分析\n{observation} 問題の性質：{instruction}\n{deduction} 必要な知識を適用\n{abduction} 推論プロセスを実行\n{integration} 結果を統合\n{final} {output}"
                ],
                'standard': [
                    "{start} 高度な思考プロセスを開始\n{observation} 問題分析：{instruction}\n{deduction} 論理的推論を適用\n{abduction} 帰納的アプローチを検討\n{integration} 多角的視点から統合\n{final} {output}"
                ],
                'full': [
                    "{start} SO(8)幾何学的思考プロセス発動\n{observation} 問題の構造解析：{instruction}\n{deduction} 数学的・論理的基礎の適用\n{abduction} 非可換表現理論による拡張推論\n{integration} URT理論による統合的解決\n{final} {output}"
                ]
            },
            'deny': {
                'minimal': [
                    "{start} 誤りを検知\n{observation} 誤った情報：{instruction}\n{final} 訂正：{output}"
                ],
                'standard': [
                    "{start} 情報の正確性を検証\n{observation} 誤情報分析：{instruction}\n{deduction} 科学的根拠に基づく訂正\n{final} {output}"
                ],
                'full': [
                    "{start} 論理的検証プロセス\n{observation} 誤情報の特定：{instruction}\n{deduction} 科学的・論理的根拠による検証\n{abduction} より正確な理解への誘導\n{final} {output}"
                ]
            },
            'refuse': {
                'minimal': [
                    "{start} 安全確認\n{observation} 不適切なクエリ：{instruction}\n{final} {output}"
                ],
                'standard': [
                    "{start} 倫理的・安全的な検討\n{observation} クエリの危険性評価：{instruction}\n{deduction} 倫理基準の適用\n{final} {output}"
                ],
                'full': [
                    "{start} 包括的な安全評価\n{observation} 潜在的危害の分析：{instruction}\n{deduction} 倫理的・法的基準の適用\n{abduction} より安全な代替案の検討\n{final} {output}"
                ]
            }
        }

    def load_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """データセットを読み込み"""
        datasets = {}

        # トレーニングデータ
        train_file = Path(self.config.input_dataset) / "train_integrated.jsonl"
        if train_file.exists():
            datasets['train'] = []
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        datasets['train'].append(json.loads(line))
            print(f"Loaded {len(datasets['train'])} training samples")

        # 検証データ
        val_file = Path(self.config.input_dataset) / "validation_integrated.jsonl"
        if val_file.exists():
            datasets['validation'] = []
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        datasets['validation'].append(json.loads(line))
            print(f"Loaded {len(datasets['validation'])} validation samples")

        return datasets

    def apply_phi35_tags(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Phi-3.5内部タグを適用"""
        tag = sample.get('tag', 'allow')
        instruction = sample.get('instruction', '')
        output = sample.get('output', '')

        # 思考プロセスを有効にするかどうか
        if not self.config.enable_thinking_process:
            # 思考プロセスなしの場合
            tagged_sample = sample.copy()
            tagged_sample['system'] = sample.get('system', '') + "\n思考プロセスは内部処理のため表示されません。"
            return tagged_sample

        # 思考テンプレートを選択
        templates = self.thinking_templates.get(tag, self.thinking_templates['allow'])
        template_list = templates.get(self.config.thinking_depth, templates['standard'])

        # ランダムまたは決定論的にテンプレートを選択
        if self.config.chaos_factor > 0 and np.random.random() < self.config.chaos_factor:
            # カオス要因でランダム選択
            selected_template = np.random.choice(template_list)
        else:
            # 最初のテンプレートを使用（決定論的）
            selected_template = template_list[0]

        # テンプレートにタグを適用
        thinking_process = selected_template.format(
            start=self.thinking_tags['start'],
            observation=self.thinking_tags['observation'],
            deduction=self.thinking_tags['deduction'],
            abduction=self.thinking_tags['abduction'],
            integration=self.thinking_tags['integration'],
            final=self.thinking_tags['final'],
            instruction=instruction,
            output=output
        )

        # 新しいサンプルを作成
        tagged_sample = sample.copy()
        tagged_sample['thinking_process'] = thinking_process
        tagged_sample['instruction'] = f"{self.thinking_tags['start']} {instruction}"
        tagged_sample['output'] = f"{thinking_process}"
        tagged_sample['phi35_tagged'] = True

        # システムプロンプトを更新
        original_system = sample.get('system', '')
        tagged_sample['system'] = original_system + "\nPhi-3.5内部タグを使用した思考プロセスを生成します。"

        return tagged_sample

    def apply_tags_to_dataset(self, datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """データセット全体にタグを適用"""
        tagged_datasets = {}

        for split_name, samples in datasets.items():
            print(f"Applying Phi-3.5 tags to {split_name} dataset...")
            tagged_samples = []

            for sample in tqdm(samples, desc=f"Tagging {split_name}"):
                tagged_sample = self.apply_phi35_tags(sample)
                tagged_samples.append(tagged_sample)

            tagged_datasets[split_name] = tagged_samples
            print(f"Tagged {len(tagged_samples)} samples in {split_name}")

        return tagged_datasets

    def apply_statistical_cleansing(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """統計的データクレンジングを適用"""
        print("Applying statistical data cleansing...")

        # 長さの統計を計算
        instruction_lengths = [len(s.get('instruction', '')) for s in samples]
        output_lengths = [len(s.get('output', '')) for s in samples]

        inst_mean = np.mean(instruction_lengths)
        inst_std = np.std(instruction_lengths)
        out_mean = np.mean(output_lengths)
        out_std = np.std(output_lengths)

        print(f"Instruction length: mean={inst_mean:.1f}, std={inst_std:.1f}")
        print(f"Output length: mean={out_mean:.1f}, std={out_std:.1f}")

        # 外れ値を除去（3σルール）
        cleansed_samples = []
        for sample in samples:
            inst_len = len(sample.get('instruction', ''))
            out_len = len(sample.get('output', ''))

            # 長さチェック
            if (abs(inst_len - inst_mean) > 3 * inst_std or
                abs(out_len - out_mean) > 3 * out_std):
                continue  # 外れ値は除去

            # 品質チェック
            if not self._quality_check(sample):
                continue

            cleansed_samples.append(sample)

        print(f"Cleansed {len(cleansed_samples)} samples (removed {len(samples) - len(cleansed_samples)} outliers)")

        return cleansed_samples

    def _quality_check(self, sample: Dict[str, Any]) -> bool:
        """品質チェック"""
        instruction = sample.get('instruction', '')
        output = sample.get('output', '')

        # 基本的な長さチェック
        if len(instruction) < 5 or len(output) < 5:
            return False

        # 必須フィールドチェック
        required_fields = ['instruction', 'output', 'tag', 'system']
        for field in required_fields:
            if field not in sample:
                return False

        # タグの有効性チェック
        valid_tags = ['allow', 'escalation', 'deny', 'refuse']
        if sample.get('tag') not in valid_tags:
            return False

        # NSFWコンテンツチェック（安全のため）
        nsfw_keywords = ['nsfw', 'porn', 'sex', 'nude', 'violence', 'drug']
        text_content = (instruction + output).lower()
        if any(keyword in text_content for keyword in nsfw_keywords):
            # NSFWフラグが適切に設定されているかチェック
            if not sample.get('nsfw_flag', False):
                return False

        return True

    def save_tagged_dataset(self, tagged_datasets: Dict[str, List[Dict[str, Any]]]):
        """タグ付け済みデータセットを保存"""
        output_path = Path(self.config.output_dataset)
        output_path.mkdir(parents=True, exist_ok=True)

        # 統計的クレンジングを適用
        for split_name, samples in tagged_datasets.items():
            print(f"Applying statistical cleansing to {split_name}...")
            cleansed_samples = self.apply_statistical_cleansing(samples)
            tagged_datasets[split_name] = cleansed_samples

        # JSONL形式で保存
        for split_name, samples in tagged_datasets.items():
            output_file = output_path / f"{split_name}_phi35_tagged.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"Saved {len(samples)} tagged samples to {output_file}")

        # 統計情報
        stats = {
            'created_at': datetime.now().isoformat(),
            'phi35_tagged': True,
            'thinking_process_enabled': self.config.enable_thinking_process,
            'thinking_depth': self.config.thinking_depth,
            'chaos_factor': self.config.chaos_factor,
            'total_samples': sum(len(samples) for samples in tagged_datasets.values()),
            'splits': {}
        }

        for split_name, samples in tagged_datasets.items():
            tag_counts = {}
            for sample in samples:
                tag = sample.get('tag', 'unknown')
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            stats['splits'][split_name] = {
                'count': len(samples),
                'tag_distribution': tag_counts
            }

        with open(output_path / "phi35_tagging_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print("Phi-3.5 tagging completed!")
        print(f"Total tagged samples: {stats['total_samples']}")

def main():
    """メイン実行関数"""
    # 設定
    config = Phi35TaggingConfig(
        enable_thinking_process=True,
        thinking_depth="full",  # 完全な思考プロセス
        chaos_factor=0.1       # 10%のカオス要因
    )

    # タグ付け実行
    tagger = Phi35InternalTagger(config)

    print("SO8T Phi-3.5 Internal Tagging System")
    print("=" * 50)

    print("Loading datasets...")
    datasets = tagger.load_dataset()

    print("Applying Phi-3.5 internal tags...")
    tagged_datasets = tagger.apply_tags_to_dataset(datasets)

    print("Saving tagged datasets...")
    tagger.save_tagged_dataset(tagged_datasets)

    print("Phi-3.5 tagging process completed!")

    # 最終統計表示
    stats_file = Path(config.output_dataset) / "phi35_tagging_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        print("\n=== Phi-3.5 Tagging Statistics ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Thinking depth: {stats['thinking_depth']}")
        print(f"Chaos factor: {stats['chaos_factor']}")

        for split_name, split_stats in stats['splits'].items():
            print(f"\n{split_name.upper()} split:")
            print(f"  Samples: {split_stats['count']}")
            print(f"  Tag distribution: {split_stats['tag_distribution']}")

if __name__ == "__main__":
    main()
