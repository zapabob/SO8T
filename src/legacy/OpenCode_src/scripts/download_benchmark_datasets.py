#!/usr/bin/env python3
"""
ベンチマークデータセットダウンロードスクリプト
MMLU, GSM8K, MATH, ELYZA-100, AGIテスト, 人類最後の試験などの全データをダウンロード
"""

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import logging
import requests
from urllib.parse import urljoin
import zipfile
import tarfile

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 保存先ディレクトリ
DATA_DIR = Path("D:/webdataset/benchmark_datasets")

class BenchmarkDatasetDownloader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_mmlu(self):
        """MMLU (Massive Multitask Language Understanding) データセットダウンロード"""
        logger.info("Downloading MMLU dataset...")

        try:
            # MMLUはHuggingFace datasetsから利用可能
            mmlu_subjects = [
                "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
                "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
                "college_medicine", "college_physics", "computer_security", "conceptual_physics",
                "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
                "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
                "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
                "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
                "high_school_physics", "high_school_psychology", "high_school_statistics",
                "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
                "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
                "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
                "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
                "professional_law", "professional_medicine", "professional_psychology", "public_relations",
                "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
            ]

            all_data = []
            for subject in tqdm(mmlu_subjects, desc="Downloading MMLU subjects"):
                try:
                    dataset = load_dataset("cais/mmlu", subject, split="test")
                    for item in dataset:
                        item['subject'] = subject
                        all_data.append(item)
                except Exception as e:
                    logger.warning(f"Failed to download {subject}: {e}")

            # 保存
            mmlu_df = pd.DataFrame(all_data)
            mmlu_path = self.data_dir / "mmlu" / "mmlu_full.json"
            mmlu_path.parent.mkdir(parents=True, exist_ok=True)
            mmlu_df.to_json(mmlu_path, orient='records', indent=2, force_ascii=False)
            logger.info(f"MMLU dataset saved to {mmlu_path} ({len(mmlu_df)} samples)")

        except Exception as e:
            logger.error(f"Failed to download MMLU: {e}")

    def download_gsm8k(self):
        """GSM8K (Grade School Math) データセットダウンロード"""
        logger.info("Downloading GSM8K dataset...")

        try:
            dataset = load_dataset("gsm8k", "main")
            all_data = []

            for split in ['train', 'test']:
                if split in dataset:
                    for item in tqdm(dataset[split], desc=f"Processing GSM8K {split}"):
                        all_data.append({
                            'question': item['question'],
                            'answer': item['answer'],
                            'split': split
                        })

            gsm8k_df = pd.DataFrame(all_data)
            gsm8k_path = self.data_dir / "gsm8k" / "gsm8k_full.json"
            gsm8k_path.parent.mkdir(parents=True, exist_ok=True)
            gsm8k_df.to_json(gsm8k_path, orient='records', indent=2, force_ascii=False)
            logger.info(f"GSM8K dataset saved to {gsm8k_path} ({len(gsm8k_df)} samples)")

        except Exception as e:
            logger.error(f"Failed to download GSM8K: {e}")

    def download_math(self):
        """MATH データセットダウンロード"""
        logger.info("Downloading MATH dataset...")

        try:
            dataset = load_dataset("competition_math")
            all_data = []

            for split in ['train', 'test']:
                if split in dataset:
                    for item in tqdm(dataset[split], desc=f"Processing MATH {split}"):
                        all_data.append({
                            'problem': item.get('problem', ''),
                            'solution': item.get('solution', ''),
                            'level': item.get('level', ''),
                            'type': item.get('type', ''),
                            'split': split
                        })

            math_df = pd.DataFrame(all_data)
            math_path = self.data_dir / "math" / "math_full.json"
            math_path.parent.mkdir(parents=True, exist_ok=True)
            math_df.to_json(math_path, orient='records', indent=2, force_ascii=False)
            logger.info(f"MATH dataset saved to {math_path} ({len(math_df)} samples)")

        except Exception as e:
            logger.error(f"Failed to download MATH: {e}")

    def download_elyza_100(self):
        """ELYZA-100 データセットダウンロード"""
        logger.info("Downloading ELYZA-100 dataset...")

        try:
            # ELYZA-100はHuggingFaceにないので、直接ダウンロード
            url = "https://huggingface.co/datasets/elyza/ELYZA-tasks-100/raw/main/elyza_tasks_100.json"
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            elyza_df = pd.DataFrame(data)

            elyza_path = self.data_dir / "elyza_100" / "elyza_100_full.json"
            elyza_path.parent.mkdir(parents=True, exist_ok=True)
            elyza_df.to_json(elyza_path, orient='records', indent=2, force_ascii=False)
            logger.info(f"ELYZA-100 dataset saved to {elyza_path} ({len(elyza_df)} samples)")

        except Exception as e:
            logger.error(f"Failed to download ELYZA-100: {e}")

    def download_agi_tests(self):
        """AGIテストデータセットダウンロード"""
        logger.info("Downloading AGI test datasets...")

        agi_datasets = [
            ("allenai/ai2_arc", "ARC-Challenge"),
            ("allenai/ai2_arc", "ARC-Easy"),
            ("facebook/anli", None),
            ("winogrande", "winogrande_xl"),
            ("piqa", None),
            ("hellaswag", None),
            ("super_glue", "boolq"),
            ("super_glue", "multirc"),
        ]

        for dataset_name, config in tqdm(agi_datasets, desc="Downloading AGI datasets"):
            try:
                if config:
                    dataset = load_dataset(dataset_name, config)
                else:
                    dataset = load_dataset(dataset_name)

                for split in ['train', 'validation', 'test']:
                    if split in dataset:
                        df = pd.DataFrame(dataset[split])
                        save_path = self.data_dir / "agi_tests" / f"{dataset_name.replace('/', '_')}_{config or 'main'}_{split}.json"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_json(save_path, orient='records', indent=2, force_ascii=False)
                        logger.info(f"AGI dataset {dataset_name} {config} {split} saved ({len(df)} samples)")

            except Exception as e:
                logger.warning(f"Failed to download {dataset_name} {config}: {e}")

    def download_domain_benchmarks(self):
        """ドメイン別LLMベンチマークデータセットダウンロード"""
        logger.info("Downloading domain-specific benchmarks...")

        domain_datasets = [
            ("bigbench", "abstract_narrative_understanding"),
            ("bigbench", "anachronisms"),
            ("bigbench", "analytic_entailment"),
            ("bigbench", "causal_judgement"),
            ("bigbench", "date_understanding"),
            ("bigbench", "disambiguation_qa"),
            ("bigbench", "fantasy_reasoning"),
            ("bigbench", "hindu_knowledge"),
            ("bigbench", "known_unknowns"),
            ("bigbench", "logical_args"),
            ("bigbench", "movie_recommendation"),
            ("bigbench", "novel_concepts"),
            ("bigbench", "strategyqa"),
            ("bigbench", "temporal_sequences"),
            ("bigbench", "understanding_fables"),
        ]

        for dataset_name, config in tqdm(domain_datasets, desc="Downloading domain benchmarks"):
            try:
                dataset = load_dataset(dataset_name, config)
                for split in ['train', 'validation']:
                    if split in dataset:
                        df = pd.DataFrame(dataset[split])
                        save_path = self.data_dir / "domain_benchmarks" / f"{config}_{split}.json"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_json(save_path, orient='records', indent=2, force_ascii=False)
                        logger.info(f"Domain benchmark {config} {split} saved ({len(df)} samples)")

            except Exception as e:
                logger.warning(f"Failed to download domain benchmark {config}: {e}")

    def download_final_exam(self):
        """人類最後の試験（最終試験）データセットダウンロード"""
        logger.info("Downloading 'Final Exam for Humanity' datasets...")

        # 人類最後の試験として、最も難易度の高いデータセットをダウンロード
        final_exam_datasets = [
            ("allenai/ai2_arc", "ARC-Challenge"),  # 最も難しいARC
            ("lukaemon/bbh", None),  # BigBench Hard
            ("lighteval/MATH-Hard", None),  # 難しい数学問題
            ("Idavidrein/gpqa", "gpqa_main"),  # Google-Proof Q&A
            ("meta-llama/Meta-Llama-3.1-405B-Instruct-evals", None),  # 高度な評価セット
        ]

        for dataset_name, config in tqdm(final_exam_datasets, desc="Downloading final exam datasets"):
            try:
                if config:
                    dataset = load_dataset(dataset_name, config, trust_remote_code=True)
                else:
                    dataset = load_dataset(dataset_name, trust_remote_code=True)

                for split in ['train', 'validation', 'test']:
                    if split in dataset:
                        df = pd.DataFrame(dataset[split])
                        save_path = self.data_dir / "final_exam" / f"{dataset_name.replace('/', '_')}_{config or 'main'}_{split}.json"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_json(save_path, orient='records', indent=2, force_ascii=False)
                        logger.info(f"Final exam dataset {dataset_name} {config} {split} saved ({len(df)} samples)")

            except Exception as e:
                logger.warning(f"Failed to download final exam dataset {dataset_name} {config}: {e}")

    def download_all_datasets(self):
        """全データセットをダウンロード"""
        logger.info("Starting comprehensive benchmark dataset download...")

        download_methods = [
            self.download_mmlu,
            self.download_gsm8k,
            self.download_math,
            self.download_elyza_100,
            self.download_agi_tests,
            self.download_domain_benchmarks,
            self.download_final_exam,
        ]

        for method in download_methods:
            try:
                method()
            except Exception as e:
                logger.error(f"Failed to execute {method.__name__}: {e}")
                continue

        logger.info("Benchmark dataset download completed!")

        # ダウンロード統計を表示
        self.show_download_stats()

    def show_download_stats(self):
        """ダウンロード統計を表示"""
        total_files = 0
        total_size = 0

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    total_files += 1
                    file_path = Path(root) / file
                    total_size += file_path.stat().st_size

        logger.info("=== Download Statistics ===")
        logger.info(f"Total JSON files: {total_files}")
        logger.info(f"Total size: {total_size / (1024**3):.2f} GB")
        logger.info(f"Dataset directory: {self.data_dir}")


def main():
    downloader = BenchmarkDatasetDownloader(DATA_DIR)
    downloader.download_all_datasets()


if __name__ == "__main__":
    main()
