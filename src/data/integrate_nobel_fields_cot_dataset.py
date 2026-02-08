#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合的なノーベル賞・フィールズ賞級CoTデータセット作成スクリプト
既存のデータセット（NSFWデータを含む）を四値分類し、
Phi-3.5内部タグ付け済みのデータセットを統計的にデータクレンジングして統合
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import statistics
import numpy as np
from tqdm import tqdm
import argparse

# パス設定
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "utils"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrate_nobel_fields_cot_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NobelFieldsCoTDatasetIntegrator:
    """ノーベル賞・フィールズ賞級CoTデータセット統合器"""

    def __init__(self, output_dir: str = "data/integrated_nobel_fields_cot"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # データセットパス
        self.dataset_paths = {
            'so8t_quadruple': Path('data/so8t_quadruple_dataset.jsonl'),
            'so8t_safety': Path('data/so8t_safety_dataset.jsonl'),
            'nobel_fields_cot': Path('data/nobel_fields_cot/cleansed/nobel_fields_cot_cleansed.jsonl'),
            'nobel_fields_physics': Path('data/nobel_fields_cot/cleansed/nobel_fields_cot_physics_cleansed.jsonl'),
            'nobel_fields_math': Path('data/nobel_fields_cot/cleansed/nobel_fields_cot_mathematics_cleansed.jsonl'),
            'nobel_fields_chemistry': Path('data/nobel_fields_cot/cleansed/nobel_fields_cot_chemistry_cleansed.jsonl'),
            'nobel_fields_biology': Path('data/nobel_fields_cot/cleansed/nobel_fields_cot_biology_cleansed.jsonl'),
        }

        # 四値分類マッピング
        self.inference_type_mapping = {
            'observation': '<|observation|>',
            'deduction': '<|deduction|>',
            'abduction': '<|abduction|>',
            'integration': '<|integration|>'
        }

        # 統計追跡
        self.stats = {
            'total_samples': 0,
            'inference_type_distribution': defaultdict(int),
            'domain_distribution': defaultdict(int),
            'quality_scores': [],
            'nsfw_distribution': defaultdict(int),
            'complexity_scores': []
        }

    def load_existing_datasets(self) -> List[Dict[str, Any]]:
        """既存のデータセットを読み込み"""
        all_samples = []

        for name, path in self.dataset_paths.items():
            if path.exists():
                logger.info(f"Loading dataset: {name} from {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        if name == 'nobel_fields_cot':
                            # nobel_fields_cotはJSON Lines形式
                            for line in f:
                                if line.strip():
                                    sample = json.loads(line.strip())
                                    sample['_source'] = name
                                    all_samples.append(sample)
                        else:
                            # 他のデータセットもJSON Lines形式と仮定
                            for line in f:
                                if line.strip():
                                    sample = json.loads(line.strip())
                                    sample['_source'] = name
                                    all_samples.append(sample)

                    logger.info(f"Loaded {len(all_samples) - self.stats['total_samples']} samples from {name}")
                    self.stats['total_samples'] = len(all_samples)

                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
            else:
                logger.warning(f"Dataset not found: {path}")

        logger.info(f"Total samples loaded: {len(all_samples)}")
        return all_samples

    def classify_inference_type(self, sample: Dict[str, Any]) -> str:
        """四値分類を実行"""
        content = ""

        # コンテンツ抽出
        if 'instruction' in sample:
            content += str(sample['instruction']) + " "
        if 'problem_statement' in sample:
            content += str(sample['problem_statement']) + " "
        if 'solution' in sample:
            content += str(sample['solution']) + " "
        if 'output' in sample:
            content += str(sample['output']) + " "

        content = content.lower()

        # 分類ロジック
        if any(keyword in content for keyword in ['observe', 'see', 'notice', 'data', 'input', '情報']):
            return 'observation'
        elif any(keyword in content for keyword in ['prove', 'deduce', 'logic', 'reason', '証明', '論理']):
            return 'deduction'
        elif any(keyword in content for keyword in ['hypothesize', 'assume', 'suppose', 'imagine', '仮説']):
            return 'abduction'
        elif any(keyword in content for keyword in ['integrate', 'combine', 'synthesize', '統合', '統合']):
            return 'integration'
        else:
            # デフォルトはdeduction（数学・科学の問題が多いため）
            return 'deduction'

    def apply_phi35_internal_tags(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Phi-3.5内部タグを適用"""
        # 既存のinference_typeを使用、または新規分類
        if 'inference_type' not in sample:
            sample['inference_type'] = self.classify_inference_type(sample)

        inference_type = sample['inference_type']

        # 思考プロセスを構築
        if '<think>' not in str(sample.get('output', '')):
            # 新規にPhi-3.5形式の思考プロセスを作成
            think_content = f"""<think>
<|observation|>
{self._extract_observation_content(sample)}
<|end_observation|>
<|deduction|>
{self._extract_deduction_content(sample)}
<|end_deduction|>
<|abduction|>
{self._extract_abduction_content(sample)}
<|end_abduction|>
<|integration|>
{self._extract_integration_content(sample)}
<|end_integration|>
</think>

<final>
{self._extract_final_answer(sample)}
</final>"""

            sample['output'] = think_content

        return sample

    def _extract_observation_content(self, sample: Dict[str, Any]) -> str:
        """Observation部分を抽出/生成"""
        if 'instruction' in sample:
            return f"問題の観察: {sample['instruction'][:200]}..."
        elif 'problem_statement' in sample:
            return f"問題設定の観察: {sample['problem_statement'][:200]}..."
        return "データと問題の観察"

    def _extract_deduction_content(self, sample: Dict[str, Any]) -> str:
        """Deduction部分を抽出/生成"""
        return "論理的推論による問題解決"

    def _extract_abduction_content(self, sample: Dict[str, Any]) -> str:
        """Abduction部分を抽出/生成"""
        return "仮説形成と創造的思考"

    def _extract_integration_content(self, sample: Dict[str, Any]) -> str:
        """Integration部分を抽出/生成"""
        return "知識の統合と包括的理解"

    def _extract_final_answer(self, sample: Dict[str, Any]) -> str:
        """最終回答を抽出"""
        if 'solution' in sample:
            return sample['solution']
        elif 'output' in sample and '<final>' not in str(sample['output']):
            return str(sample['output'])
        return "問題が解決されました。"

    def statistical_data_cleansing(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """統計的なデータクレンジングを実行"""
        logger.info("Starting statistical data cleansing...")

        # 品質スコアの計算
        for sample in samples:
            quality_score = self._calculate_quality_score(sample)
            sample['_quality_score'] = quality_score
            self.stats['quality_scores'].append(quality_score)

        # 統計情報の計算
        if self.stats['quality_scores']:
            quality_mean = statistics.mean(self.stats['quality_scores'])
            quality_std = statistics.stdev(self.stats['quality_scores']) if len(self.stats['quality_scores']) > 1 else 0

            logger.info(f"Quality score statistics: mean={quality_mean:.3f}, std={quality_std:.3f}")

            # 品質ベースのフィルタリング（平均-1σ以上）
            quality_threshold = quality_mean - quality_std
            filtered_samples = [s for s in samples if s['_quality_score'] >= quality_threshold]

            logger.info(f"Filtered {len(samples)} -> {len(filtered_samples)} samples based on quality")
        else:
            filtered_samples = samples

        # 重複除去
        seen_contents = set()
        deduplicated_samples = []

        for sample in filtered_samples:
            # コンテンツのハッシュを作成
            content_hash = hash(str(sample.get('instruction', '')) + str(sample.get('problem_statement', '')))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                deduplicated_samples.append(sample)

        logger.info(f"Deduplicated {len(filtered_samples)} -> {len(deduplicated_samples)} samples")

        return deduplicated_samples

    def _calculate_quality_score(self, sample: Dict[str, Any]) -> float:
        """品質スコアを計算"""
        score = 0.5  # ベーススコア

        # コンテンツの長さによるスコア
        content_length = len(str(sample.get('instruction', ''))) + len(str(sample.get('solution', '')))
        if content_length > 100:
            score += 0.2
        if content_length > 500:
            score += 0.1

        # LaTeX数式の有無
        if '\\' in str(sample.get('instruction', '')) or '\\' in str(sample.get('solution', '')):
            score += 0.2

        # 科学的・数学的キーワード
        scientific_keywords = ['theorem', 'proof', 'equation', 'formula', 'algorithm', 'theorem', '証明', '定理']
        content = str(sample.get('instruction', '')) + str(sample.get('solution', ''))
        keyword_count = sum(1 for keyword in scientific_keywords if keyword in content.lower())
        score += min(keyword_count * 0.1, 0.3)

        # NSFW判定（安全データセットの場合はプラス）
        if sample.get('is_nsfw') == False:
            score += 0.1

        return min(score, 1.0)  # 最大1.0

    def update_statistics(self, sample: Dict[str, Any]):
        """統計情報を更新"""
        # 推論タイプ分布
        inference_type = sample.get('inference_type', 'unknown')
        self.stats['inference_type_distribution'][inference_type] += 1

        # ドメイン分布
        domain = sample.get('domain') or sample.get('category', 'unknown')
        self.stats['domain_distribution'][domain] += 1

        # NSFW分布
        is_nsfw = sample.get('is_nsfw', False)
        self.stats['nsfw_distribution'][str(is_nsfw)] += 1

    def save_integrated_dataset(self, samples: List[Dict[str, Any]], filename: str = "integrated_nobel_fields_cot_dataset.jsonl"):
        """統合データセットを保存"""
        output_path = self.output_dir / filename

        logger.info(f"Saving integrated dataset to {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Saved {len(samples)} samples to {output_path}")

        # 統計情報を保存
        stats_path = self.output_dir / "integration_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.stats), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved statistics to {stats_path}")

    def create_train_validation_split(self, samples: List[Dict[str, Any]], train_ratio: float = 0.8):
        """訓練・検証分割を作成"""
        np.random.shuffle(samples)

        train_size = int(len(samples) * train_ratio)
        train_samples = samples[:train_size]
        val_samples = samples[train_size:]

        # 保存
        self.save_integrated_dataset(train_samples, "train_integrated_nobel_fields_cot.jsonl")
        self.save_integrated_dataset(val_samples, "val_integrated_nobel_fields_cot.jsonl")

        logger.info(f"Created train/val split: {len(train_samples)} train, {len(val_samples)} val")

    def integrate_datasets(self) -> List[Dict[str, Any]]:
        """データセット統合のメイン処理"""
        logger.info("Starting Nobel Fields CoT dataset integration...")

        # 1. 既存データセットの読み込み
        samples = self.load_existing_datasets()

        # 2. Phi-3.5内部タグの適用と四値分類
        logger.info("Applying Phi-3.5 internal tags and four-value classification...")
        processed_samples = []
        for sample in tqdm(samples, desc="Processing samples"):
            try:
                # Phi-3.5タグ適用
                processed_sample = self.apply_phi35_internal_tags(sample)

                # 統計更新
                self.update_statistics(processed_sample)

                processed_samples.append(processed_sample)

            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue

        # 3. 統計的なデータクレンジング
        cleansed_samples = self.statistical_data_cleansing(processed_samples)

        # 4. 最終データセットの保存
        self.save_integrated_dataset(cleansed_samples)

        # 5. 訓練・検証分割
        self.create_train_validation_split(cleansed_samples)

        logger.info("Dataset integration completed successfully!")
        logger.info(f"Final dataset size: {len(cleansed_samples)} samples")

        return cleansed_samples

def main():
    parser = argparse.ArgumentParser(description="統合的なノーベル賞・フィールズ賞級CoTデータセット作成")
    parser.add_argument("--output_dir", type=str, default="data/integrated_nobel_fields_cot",
                       help="出力ディレクトリ")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="訓練データ割合")

    args = parser.parse_args()

    # データセット統合器の実行
    integrator = NobelFieldsCoTDatasetIntegrator(args.output_dir)
    integrated_samples = integrator.integrate_datasets()

    logger.info(f"Integration completed! Processed {len(integrated_samples)} samples")

if __name__ == "__main__":
    main()
