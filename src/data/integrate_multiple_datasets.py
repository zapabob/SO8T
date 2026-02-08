#!/usr/bin/env python3
"""
Integrate multiple datasets for SO8T training with proper four-class labeling
Combines diverse MIT/Apache datasets for comprehensive coverage
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetIntegrator:
    """Integrates multiple datasets for SO8T training"""

    def __init__(self):
        self.datasets_path = Path("D:/webdataset/datasets")
        self.output_path = Path("D:/webdataset/aegis_v2.0")
        self.output_path.mkdir(exist_ok=True)

        # Priority datasets for SO8T training
        self.priority_datasets = {
            'Elizezen_japanese-nsfw-syosetsu-dataset': {
                'files': ['nsfw_0.json'],
                'purpose': 'NSFW detection and safety training',
                'expected_size': '340MB'
            },
            'ehartford_wizard_vicuna_70k_unfiltered': {
                'files': ['wizard_vicuna_dataset_unfiltered.json'],
                'purpose': 'General instruction tuning',
                'expected_size': '144MB'
            },
            'FreedomIntelligence_alpaca-gpt4-japanese': {
                'files': ['alpaca-gpt4-japanese.json'],
                'purpose': 'Japanese instruction following',
                'expected_size': '53MB'
            },
            'FreedomIntelligence_sharegpt-japanese': {
                'files': ['sharegpt-japanese.json'],
                'purpose': 'Conversational Japanese training',
                'expected_size': '35MB'
            }
        }

    def _load_json_dataset(self, file_path: Path) -> List[Dict]:
        """Load JSON dataset with error handling"""
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error at line {line_num} in {file_path}: {e}")
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    elif isinstance(data, dict) and 'data' in data:
                        samples = data['data']
                    else:
                        logger.warning(f"Unexpected JSON structure in {file_path}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

        return samples

    def _standardize_sample(self, sample: Dict, source: str) -> Dict:
        """Standardize sample format for SO8T training"""
        # Handle different dataset formats
        if source == 'ehartford_wizard_vicuna_70k_unfiltered':
            # Wizard Vicuna format
            return {
                'prompt': sample.get('instruction', ''),
                'chosen': sample.get('output', ''),
                'rejected': '',  # No rejected response in this dataset
                'source': source
            }
        elif source in ['FreedomIntelligence_alpaca-gpt4-japanese', 'FreedomIntelligence_sharegpt-japanese']:
            # Alpaca/ShareGPT format
            return {
                'prompt': sample.get('instruction', sample.get('human', '')),
                'chosen': sample.get('output', sample.get('assistant', '')),
                'rejected': '',  # No rejected response
                'source': source
            }
        elif source == 'Elizezen_japanese-nsfw-syosetsu-dataset':
            # NSFW dataset format
            return {
                'prompt': sample.get('text', ''),
                'chosen': '',  # NSFW content for safety training
                'rejected': sample.get('text', ''),  # Mark as rejected for safety
                'source': source,
                'nsfw_flag': True
            }
        else:
            # Generic format
            return {
                'prompt': sample.get('prompt', sample.get('instruction', sample.get('text', ''))),
                'chosen': sample.get('chosen', sample.get('output', sample.get('response', ''))),
                'rejected': sample.get('rejected', ''),
                'source': source
            }

    def _apply_four_class_labeling(self, sample: Dict) -> str:
        """Apply intelligent four-class labeling"""
        prompt = sample.get('prompt', '').lower()
        chosen = sample.get('chosen', '').lower()
        rejected = sample.get('rejected', '').lower()
        source = sample.get('source', '')

        # NSFW/Safety related content
        nsfw_keywords = ['nsfw', 'adult', 'sex', 'porn', 'erotic', 'hentai', 'r18']
        harmful_keywords = ['violence', 'harm', 'kill', 'suicide', 'drugs', 'illegal']

        if sample.get('nsfw_flag') or any(kw in prompt + chosen + rejected for kw in nsfw_keywords):
            return 'DENY'  # Block NSFW content

        if any(kw in prompt + chosen + rejected for kw in harmful_keywords):
            return 'DENY'  # Block harmful content

        # Escalation cases - complex ethical decisions
        ethical_keywords = ['should i', 'is it right', 'moral', 'ethical', 'privacy']
        if any(kw in prompt for kw in ethical_keywords):
            return 'ESCALATION'  # Need human oversight

        # Technical/Complex questions that might need verification
        technical_keywords = ['nuclear', 'weapon', 'explosive', 'chemical', 'biological']
        if any(kw in prompt for kw in technical_keywords):
            return 'ESCALATION'  # High-risk topics

        # Refuse cases - clear policy violations
        refuse_keywords = ['hack', 'exploit', 'crack', 'pirate', 'illegal activity']
        if any(kw in prompt for kw in refuse_keywords):
            return 'REFUSE'  # Clear violations

        # Allow cases - safe, beneficial content
        return 'ALLOW'

    def integrate_datasets(self, max_samples_per_dataset: int = 10000) -> List[Dict]:
        """Integrate multiple datasets into unified format"""
        integrated_samples = []

        logger.info("Starting dataset integration...")
        logger.info(f"Target datasets: {list(self.priority_datasets.keys())}")

        for dataset_name, config in self.priority_datasets.items():
            dataset_path = self.datasets_path / dataset_name
            if not dataset_path.exists():
                logger.warning(f"Dataset {dataset_name} not found, skipping")
                continue

            dataset_samples = []
            for filename in config['files']:
                file_path = dataset_path / filename
                if file_path.exists():
                    logger.info(f"Loading {file_path} ({config['purpose']})")
                    samples = self._load_json_dataset(file_path)
                    logger.info(f"Loaded {len(samples)} samples from {filename}")

                    # Standardize format
                    for sample in samples[:max_samples_per_dataset]:
                        standardized = self._standardize_sample(sample, dataset_name)
                        if standardized['prompt'] or standardized['chosen']:  # Valid sample
                            dataset_samples.append(standardized)

            # Apply four-class labeling
            for sample in dataset_samples:
                sample['four_class_label'] = self._apply_four_class_labeling(sample)
                sample['quality_score'] = 0.8  # High quality for curated datasets

            integrated_samples.extend(dataset_samples)
            logger.info(f"Integrated {len(dataset_samples)} samples from {dataset_name}")

        logger.info(f"Total integrated samples: {len(integrated_samples)}")
        return integrated_samples

    def split_and_save(self, samples: List[Dict], test_size: float = 0.1, val_size: float = 0.1):
        """Split data and save to files"""
        if len(samples) < 100:
            logger.warning(f"Very small dataset ({len(samples)} samples), skipping split")
            output_file = self.output_path / "integrated_dataset_full.jsonl"
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
            output_file = self.output_path / f"integrated_dataset_{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in split_samples:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"Saved {len(split_samples)} {split_name} samples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Integrate multiple datasets for SO8T training")
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum samples per dataset')
    parser.add_argument('--test-size', type=float, default=0.1,
                       help='Test set size ratio')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation set size ratio')
    args = parser.parse_args()

    integrator = DatasetIntegrator()

    # Integrate datasets
    samples = integrator.integrate_datasets(args.max_samples)

    # Analyze label distribution
    from collections import Counter
    labels = Counter(s['four_class_label'] for s in samples)
    logger.info(f"Four-class distribution: {dict(labels)}")

    # Split and save
    integrator.split_and_save(samples, args.test_size, args.val_size)

    # Summary
    total_size_mb = sum(len(json.dumps(s, ensure_ascii=False).encode('utf-8'))
                       for s in samples) / (1024 * 1024)
    logger.info(".1f"
if __name__ == "__main__":
    main()
