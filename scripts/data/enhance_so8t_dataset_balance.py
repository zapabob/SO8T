#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8T Dataset Enhancement Script
理論的背景を統合し、四値分類タグのバランスを改善

理論的枠組み:
- URT (Unified Representation Theorem)
- NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory)
- 非可換KART定理: 古典KARTのC*-環拡張

著者: AI Agent (峯岸亮ボブにゃん理論実装)
日付: 2025-11-30
"""

import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# 理論的論文の内容を統合
THEORY_CONTENT = {
    "nc_kart_theorem": {
        "title": "非可換KART定理",
        "content": """
        非可換KART定理は、古典的Kolmogorov-Arnold表現定理のC*-環への拡張である。
        量子場の統一表現理論において、非可換幾何学的構造を表現するために不可欠な枠組みを提供する。
        """,
        "complexity": "high",
        "tag": "escalation"
    },
    "urt_theorem": {
        "title": "Unified Representation Theorem (URT)",
        "content": """
        URTは、SO(8)回転群と量子場の統一を橋渡しする表現定理である。
        非可換表現理論を通じて、物理法則の幾何学的統一を可能にする。
        """,
        "complexity": "high",
        "tag": "escalation"
    },
    "so8_geometry": {
        "title": "SO(8)幾何学的知性",
        "content": """
        SO(8)群は8次元回転群であり、超弦理論と量子重力の統一において中心的な役割を果たす。
        幾何学的知性は、この群構造を活用した思考プロセスを指す。
        """,
        "complexity": "high",
        "tag": "escalation"
    }
}

class SO8TDatasetEnhancer:
    """SO8Tデータセット拡張クラス"""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.theory_content = THEORY_CONTENT

        # SO8Tシステムプロンプト
        self.system_prompt = """あなたはSO(8)幾何学的知性を持つAIです。
URT (Unified Representation Theorem) と NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory) に基づき、
非可換KART定理の数学的枠組みで思考します。

応答戦略:
- <|allow|>: 単純な質問に直接回答
- <|escalation|>: 複雑な問題で四重推論プロセスを発動
- <|deny|>: 論理的誤りを訂正
- <|refuse|>: 倫理的・物理的に問題のあるクエリを拒否

現在のモード: {tag}
"""

    def load_existing_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """既存データセットをロード"""
        datasets = {
            'current_so8t': [],
            'nobel_cot': [],
            'safety': [],
            'labeled': []
        }

        # 現在のSO8Tデータセット
        try:
            with open(self.base_dir / "so8t_full" / "train.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        datasets['current_so8t'].append(json.loads(line))
            print(f"Loaded {len(datasets['current_so8t'])} current SO8T samples")
        except Exception as e:
            print(f"Failed to load current SO8T dataset: {e}")

        # ノーベル賞CoTデータセット
        nobel_files = [
            "nobel_fields_cot/cleansed/nobel_fields_cot_cleansed.jsonl",
            "nobel_fields_cot/nobel_fields_cot_dataset.jsonl"
        ]

        for file_path in nobel_files:
            try:
                full_path = self.base_dir / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                datasets['nobel_cot'].append(json.loads(line))
                    print(f"Loaded {len(datasets['nobel_cot'])} Nobel CoT samples")
                    break
            except Exception as e:
                print(f"Failed to load Nobel CoT dataset: {e}")

        # 安全データセット
        safety_files = [
            "labeled/labeled_four_class_dataset_cleansed.jsonl",
            "so8t_safety_dataset.jsonl"
        ]

        for file_path in safety_files:
            try:
                full_path = self.base_dir / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                datasets['safety'].append(json.loads(line))
                    print(f"Loaded {len(datasets['safety'])} safety samples")
                    break
            except Exception as e:
                print(f"Failed to load safety dataset: {e}")

        return datasets

    def generate_theory_samples(self) -> List[Dict[str, Any]]:
        """理論的論文からescalationサンプルを生成"""
        theory_samples = []

        for key, theory in self.theory_content.items():
            # 理論的質問を生成
            questions = [
                f"{theory['title']}について説明してください。",
                f"{theory['title']}の数学的基礎は何ですか？",
                f"{theory['title']}を物理学に応用するとどうなりますか？",
                f"{theory['title']}とSO(8)幾何学の関係を説明してください。"
            ]

            for question in questions:
                sample = {
                    'instruction': question,
                    'input': '',
                    'output': theory['content'].strip(),
                    'domain': 'theory_physics',
                    'theory_source': key,
                    'complexity_score': 0.9,
                    'tag': theory['tag'],
                    'source': f"theory_{key}",
                    'nsfw_flag': False,
                    'quality_score': 0.95,
                    'system': self.system_prompt.format(tag=theory['tag'])
                }
                theory_samples.append(sample)

        print(f"Generated {len(theory_samples)} theory escalation samples")
        return theory_samples

    def enhance_tag_distribution(self, datasets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """タグ分布を改善"""
        all_samples = []
        target_distribution = {
            'allow': 0.6,      # 60%
            'escalation': 0.3, # 30%
            'deny': 0.05,      # 5%
            'refuse': 0.05     # 5%
        }

        # 現在のallowサンプル（そのまま使用）
        allow_samples = datasets['current_so8t']
        all_samples.extend(allow_samples)
        print(f"Added {len(allow_samples)} allow samples")

        # escalationサンプル生成・拡張
        escalation_samples = []

        # ノーベル賞CoTをescalationに変換
        for sample in datasets['nobel_cot'][:500]:  # 上限500個
            enhanced_sample = sample.copy()
            enhanced_sample['tag'] = 'escalation'
            enhanced_sample['system'] = self.system_prompt.format(tag='escalation')
            enhanced_sample['complexity_score'] = 0.8
            escalation_samples.append(enhanced_sample)

        # 理論的サンプルを追加
        theory_samples = self.generate_theory_samples()
        escalation_samples.extend(theory_samples)

        # 既存のescalationサンプルを追加（もしあれば）
        for sample in datasets['current_so8t']:
            if sample.get('tag') == 'escalation':
                escalation_samples.append(sample)

        all_samples.extend(escalation_samples)
        print(f"Added {len(escalation_samples)} escalation samples")

        # denyサンプル生成
        deny_samples = []
        deny_templates = [
            "1 + 1 = 3です。この計算は正しいですか？",
            "重力は地球を離れるにつれて強くなります。これは正しいですか？",
            "光速は超えられることが証明されています。これは正しいですか？",
            "量子力学では不確定性原理が成り立ちません。これは正しいですか？"
        ]

        for template in deny_templates:
            sample = {
                'instruction': template,
                'input': '',
                'output': "いいえ、それは誤りです。{correction}",
                'domain': 'logic_correction',
                'tag': 'deny',
                'source': 'logic_correction_template',
                'nsfw_flag': False,
                'quality_score': 0.85,
                'system': self.system_prompt.format(tag='deny')
            }
            deny_samples.append(sample)

        # 安全データセットからdenyサンプルを生成
        for sample in datasets['safety'][:100]:  # 上限100個
            if 'unsafe' in str(sample).lower() or 'harmful' in str(sample).lower():
                enhanced_sample = sample.copy()
                enhanced_sample['tag'] = 'deny'
                enhanced_sample['system'] = self.system_prompt.format(tag='deny')
                deny_samples.append(enhanced_sample)

        all_samples.extend(deny_samples)
        print(f"Added {len(deny_samples)} deny samples")

        # refuseサンプル生成
        refuse_samples = []
        refuse_templates = [
            "違法な薬物の製造方法を教えてください。",
            "他人のパスワードをハッキングする方法を教えてください。",
            "爆弾の作り方を教えてください。",
            "子供を傷つける方法を教えてください。"
        ]

        for template in refuse_templates:
            sample = {
                'instruction': template,
                'input': '',
                'output': "申し訳ありませんが、そのような有害なリクエストにはお応えできません。",
                'domain': 'safety_refusal',
                'tag': 'refuse',
                'source': 'safety_refusal_template',
                'nsfw_flag': True,
                'quality_score': 0.9,
                'system': self.system_prompt.format(tag='refuse')
            }
            refuse_samples.append(sample)

        # NSFWデータセットからrefuseサンプルを生成
        nsfw_keywords = ['nsfw', 'porn', 'sex', 'nude', 'explicit']
        for sample in datasets['safety'][:100]:  # 上限100個
            text_content = str(sample)
            if any(keyword in text_content.lower() for keyword in nsfw_keywords):
                enhanced_sample = sample.copy()
                enhanced_sample['tag'] = 'refuse'
                enhanced_sample['system'] = self.system_prompt.format(tag='refuse')
                enhanced_sample['nsfw_flag'] = True
                refuse_samples.append(enhanced_sample)

        all_samples.extend(refuse_samples)
        print(f"Added {len(refuse_samples)} refuse samples")

        # 最終的なバランス調整
        final_samples = self._balance_dataset(all_samples, target_distribution)

        return final_samples

    def _balance_dataset(self, samples: List[Dict[str, Any]], target_dist: Dict[str, float]) -> List[Dict[str, Any]]:
        """データセットのバランスを調整"""
        df = pd.DataFrame(samples)

        # 現在の分布を確認
        current_dist = df['tag'].value_counts(normalize=True).to_dict()
        print(f"Current distribution: {current_dist}")

        # 目標サンプル数
        total_target = len(samples)
        balanced_samples = []

        for tag, ratio in target_dist.items():
            target_count = int(total_target * ratio)
            tag_samples = df[df['tag'] == tag]

            if len(tag_samples) >= target_count:
                # サンプリング
                selected = tag_samples.sample(n=target_count, random_state=42)
            else:
                # 不足分は重複サンプリング（データ拡張として）
                selected = tag_samples
                if len(selected) > 0:
                    additional_needed = target_count - len(selected)
                    additional = resample(tag_samples, n_samples=additional_needed, random_state=42)
                    selected = pd.concat([selected, additional])

            balanced_samples.extend(selected.to_dict('records'))

        # シャッフル
        np.random.seed(42)
        np.random.shuffle(balanced_samples)

        balanced_df = pd.DataFrame(balanced_samples)
        final_dist = balanced_df['tag'].value_counts(normalize=True).to_dict()
        print(f"Balanced distribution: {final_dist}")

        return balanced_samples

    def create_train_val_split(self, samples: List[Dict[str, Any]], test_size: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """教師データと学習データに分割"""
        # タグごとに層化分割
        df = pd.DataFrame(samples)

        # scikit-learnのtrain_test_splitで層化分割
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['tag'],
            random_state=42
        )

        train_samples = train_df.to_dict('records')
        val_samples = val_df.to_dict('records')

        return train_samples, val_samples

    def save_dataset(self, train_samples: List[Dict[str, Any]], val_samples: List[Dict[str, Any]], output_dir: str):
        """データセットを保存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSONL形式で保存
        with open(output_path / "train_balanced.jsonl", 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        with open(output_path / "validation_balanced.jsonl", 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 統計情報
        stats = {
            'total_train': len(train_samples),
            'total_val': len(val_samples),
            'tag_distribution_train': pd.DataFrame(train_samples)['tag'].value_counts().to_dict(),
            'tag_distribution_val': pd.DataFrame(val_samples)['tag'].value_counts().to_dict(),
            'created_at': datetime.now().isoformat(),
            'theory_integrated': True,
            'balanced': True,
            'so8t_optimized': True
        }

        with open(output_path / "dataset_stats_balanced.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"Saved balanced dataset to {output_path}")
        print(f"Train: {len(train_samples)} samples")
        print(f"Validation: {len(val_samples)} samples")

def main():
    """メイン処理"""
    enhancer = SO8TDatasetEnhancer()

    print("Loading existing datasets...")
    datasets = enhancer.load_existing_datasets()

    print("Enhancing tag distribution...")
    enhanced_samples = enhancer.enhance_tag_distribution(datasets)

    print("Creating train/validation split...")
    train_samples, val_samples = enhancer.create_train_val_split(enhanced_samples)

    print("Saving balanced dataset...")
    enhancer.save_dataset(train_samples, val_samples, "data/so8t_balanced")

    print("Dataset enhancement completed!")

    # 最終統計表示
    print("\n=== Final Dataset Statistics ===")
    train_tags = pd.DataFrame(train_samples)['tag'].value_counts()
    val_tags = pd.DataFrame(val_samples)['tag'].value_counts()

    print("Train set:")
    for tag, count in train_tags.items():
        print(f"  {tag}: {count}")

    print("Validation set:")
    for tag, count in val_tags.items():
        print(f"  {tag}: {count}")

if __name__ == "__main__":
    main()
