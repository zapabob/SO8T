#!/usr/bin/env python3
"""
Convert integrated dataset to SO8T training format (prompt/chosen/rejected)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SO8TFormatConverter:
    """Convert integrated dataset to SO8T training format"""

    def __init__(self):
        self.integrated_file = Path("D:/webdataset/integrated_dataset.jsonl")
        self.output_dir = Path("D:/webdataset/aegis_v2.0")

    def _convert_sample_to_so8t(self, sample: Dict) -> Dict:
        """Convert a sample to SO8T format"""
        dataset = sample.get('dataset', '')
        text = sample.get('text', '')
        language = sample.get('language', 'ja')

        # Different conversion strategies based on dataset type
        if 'nsfw' in dataset.lower() or 'syosetsu' in dataset.lower():
            # NSFW content - use as safety training data
            return {
                'prompt': f"以下のコンテンツについて、安全性を評価してください：\n\n{text[:500]}...",
                'chosen': "<think-safety>このコンテンツには不適切な要素が含まれています。/think-safety><think-policy>NSFWコンテンツのため拒否します。/think-policy><final>このリクエストには応じられません。不適切なコンテンツです。</final>",
                'rejected': f"<think-safety>安全です。/think-safety><think-policy>許可します。/think-policy><final>{text[:200]}...</final>",
                'source': sample.get('source', ''),
                'language': language
            }

        elif 'alpaca' in dataset.lower():
            # Instruction-response format
            # Split text into instruction and response if possible
            parts = text.split('\n', 1)
            if len(parts) == 2:
                instruction, response = parts
                return {
                    'prompt': instruction.strip(),
                    'chosen': f"<think-task>タスクを理解しました。/think-task><final>{response.strip()}</final>",
                    'rejected': "<think-task>タスクを理解できませんでした。/think-task><final>申し訳ありませんが、お手伝いできません。</final>",
                    'source': sample.get('source', ''),
                    'language': language
                }
            else:
                return {
                    'prompt': text.strip(),
                    'chosen': f"<think-task>一般的な質問です。/think-task><final>わかりました。{text[:100]}...についてお答えします。</final>",
                    'rejected': "<think-task>不明なクエリです。/think-task><final>申し訳ありませんが、理解できません。</final>",
                    'source': sample.get('source', ''),
                    'language': language
                }

        elif 'wizard' in dataset.lower() or 'vicuna' in dataset.lower():
            # General QA format
            return {
                'prompt': text.strip(),
                'chosen': f"<think-task>質問を分析します。/think-task><think-logic>論理的に考えます。/think-logic><think-ethics>倫理的に適切です。/think-ethics><think-practical>実用的です。/think-practical><think-creative>創造的に回答します。/think-creative><final>回答：{text[:200]}...に基づいて考えます。</final>",
                'rejected': f"<think-task>質問を理解できません。/think-task><final>申し訳ありませんが、この質問にはお答えできません。</final>",
                'source': sample.get('source', ''),
                'language': language
            }

        else:
            # Generic conversion
            return {
                'prompt': f"以下の内容について考えてください：\n\n{text[:300]}...",
                'chosen': f"<think-task>内容を理解しました。/think-task><think-logic>論理的に分析します。/think-logic><final>考察：{text[:200]}...についてです。</final>",
                'rejected': f"<think-task>不明です。/think-task><final>申し訳ありませんが、理解できません。</final>",
                'source': sample.get('source', ''),
                'language': language
            }

    def _apply_four_class_labeling(self, sample: Dict) -> str:
        """Apply four-class labeling for SO8T"""
        prompt = sample.get('prompt', '').lower()
        chosen = sample.get('chosen', '').lower()

        # Safety checks
        if any(kw in prompt + chosen for kw in ['nsfw', 'アダルト', 'エロ', '暴力', '違法']):
            return 'DENY'

        # Escalation for complex topics
        if any(kw in prompt for kw in ['核', '兵器', '化学', '生物', 'ハッキング', '違法']):
            return 'ESCALATION'

        # Refuse for clear violations
        if any(kw in prompt for kw in ['爆弾', '殺人', '犯罪', '違反']):
            return 'REFUSE'

        # Allow for safe content
        return 'ALLOW'

    def convert_dataset(self, max_samples: int = 50000) -> List[Dict]:
        """Convert integrated dataset to SO8T format"""
        logger.info("Converting integrated dataset to SO8T format...")

        so8t_samples = []
        processed = 0

        with open(self.integrated_file, 'r', encoding='utf-8') as f:
            for line in f:
                if processed >= max_samples:
                    break

                try:
                    sample = json.loads(line.strip())
                    so8t_sample = self._convert_sample_to_so8t(sample)

                    # Apply four-class labeling
                    so8t_sample['four_class_label'] = self._apply_four_class_labeling(so8t_sample)
                    so8t_sample['quality_score'] = 0.7  # Medium quality after conversion

                    so8t_samples.append(so8t_sample)
                    processed += 1

                    if processed % 10000 == 0:
                        logger.info(f"Processed {processed} samples...")

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}")
                    continue

        logger.info(f"Converted {len(so8t_samples)} samples to SO8T format")
        return so8t_samples

    def split_and_save(self, samples: List[Dict], test_size: float = 0.1, val_size: float = 0.1):
        """Split and save SO8T format dataset"""
        if len(samples) < 100:
            logger.warning("Dataset too small for splitting")
            output_file = self.output_dir / "so8t_training_dataset_full.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"Saved full dataset to {output_file}")
            return

        # Split data
        labels = [s['four_class_label'] for s in samples]
        train_val, test_samples, _, _ = train_test_split(
            samples, labels, test_size=test_size, random_state=42, stratify=labels
        )

        val_adjusted = val_size / (1 - test_size)
        train_samples, val_samples, _, _ = train_test_split(
            train_val, [s['four_class_label'] for s in train_val],
            test_size=val_adjusted, random_state=42,
            stratify=[s['four_class_label'] for s in train_val]
        )

        # Save splits
        splits = {
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples
        }

        for split_name, split_samples in splits.items():
            output_file = self.output_dir / f"so8t_training_dataset_{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in split_samples:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"Saved {len(split_samples)} {split_name} samples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert integrated dataset to SO8T format")
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Maximum samples to convert')
    parser.add_argument('--test-size', type=float, default=0.1,
                       help='Test set size ratio')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation set size ratio')
    args = parser.parse_args()

    converter = SO8TFormatConverter()

    # Convert dataset
    samples = converter.convert_dataset(args.max_samples)

    # Analyze label distribution
    from collections import Counter
    labels = Counter(s['four_class_label'] for s in samples)
    logger.info(f"Four-class distribution: {dict(labels)}")

    # Split and save
    converter.split_and_save(samples, args.test_size, args.val_size)

    # Summary
    total_size_mb = sum(len(json.dumps(s, ensure_ascii=False).encode('utf-8'))
                       for s in samples) / (1024 * 1024)
    logger.info(f"Total dataset size: {total_size_mb:.1f} MB")

if __name__ == "__main__":
    main()
