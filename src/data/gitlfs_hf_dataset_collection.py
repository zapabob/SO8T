#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git LFSを使用したHuggingFaceデータセット収集スクリプト
NSFWデータを含むマルチモーダル、内部推論強化、日英マルチリンガルデータセットを収集
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import argparse
import yaml

class GitLFSDatasetCollector:
    """Git LFSを使用したデータセットコレクター"""

    def __init__(self, output_dir: str = "D:/webdataset/datasets/gitlfs", max_total_size_gb: float = 10.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_total_size_gb = max_total_size_gb  # 総サイズ制限（GB）
        self.current_total_size = 0  # 現在の総サイズ（バイト）

        # PPO内部推論強化に特化したデータセット（10GB以内に収まるよう制限）
        self.target_datasets = {
            # マルチモーダルデータセット（サイズ制限付き）
            'coco_captions': {
                'hf_repo': 'HuggingFaceM4/COCO',
                'domain': 'multimodal_vision_language',
                'language': 'en',
                'license': 'cc-by-4.0',
                'estimated_size_gb': 2.5,  # 推定サイズ
                'max_samples': 5000,  # サンプル数制限
                'description': 'COCO captions for vision-language PPO training'
            },

            # 日英マルチリンガルデータセット（内部推論強化用）
            'wikipedia_ja': {
                'hf_repo': 'wikimedia/wikipedia',
                'config': '20231101.ja',
                'domain': 'multilingual_knowledge',
                'language': 'ja',
                'license': 'cc-by-sa-4.0',
                'estimated_size_gb': 1.2,
                'max_samples': 10000,
                'description': 'Japanese Wikipedia for multilingual reasoning'
            },
            'wikipedia_en': {
                'hf_repo': 'wikimedia/wikipedia',
                'config': '20231101.en',
                'domain': 'multilingual_knowledge',
                'language': 'en',
                'license': 'cc-by-sa-4.0',
                'estimated_size_gb': 1.5,
                'max_samples': 10000,
                'description': 'English Wikipedia for knowledge reasoning'
            },

            # PPO内部推論強化データセット（PhD/Fields Medal級）
            'math_qa': {
                'hf_repo': 'math_qa',
                'domain': 'advanced_mathematical_reasoning',
                'language': 'en',
                'license': 'mit',
                'estimated_size_gb': 0.05,
                'max_samples': 10000,
                'description': 'Advanced Math QA for PhD-level mathematical reasoning'
            },
            'strategy_qa': {
                'hf_repo': 'ChilleD/StrategyQA',
                'domain': 'heuristic_reasoning',
                'language': 'en',
                'license': 'mit',
                'estimated_size_gb': 0.02,
                'max_samples': 5000,
                'description': 'Strategy QA for heuristic and meta-reasoning'
            },
            'hotpot_qa': {
                'hf_repo': 'hotpot_qa',
                'domain': 'multi_hop_scientific_reasoning',
                'language': 'en',
                'license': 'cc-by-sa-4.0',
                'estimated_size_gb': 0.1,
                'max_samples': 5000,
                'description': 'Multi-hop QA for scientific inference chaining'
            },
            'gsm8k': {
                'hf_repo': 'gsm8k',
                'domain': 'mathematical_proof_reasoning',
                'language': 'en',
                'license': 'mit',
                'estimated_size_gb': 0.01,
                'max_samples': 2000,
                'description': 'GSM8K for mathematical proof and theorem proving'
            },
            'math_science_qa': {
                'hf_repo': 'allenai/math_science_qa',
                'domain': 'interdisciplinary_science_reasoning',
                'language': 'en',
                'license': 'apache-2.0',
                'estimated_size_gb': 0.05,
                'max_samples': 3000,
                'description': 'Interdisciplinary math-science QA for Nobel-level insights'
            },
            'theorem_proving': {
                'hf_repo': 'ChilleD/TheoremQA',
                'domain': 'mathematical_theorem_proving',
                'language': 'en',
                'license': 'mit',
                'estimated_size_gb': 0.03,
                'max_samples': 2000,
                'description': 'Theorem proving for Fields Medal level mathematics'
            },
            'molecular_biology_qa': {
                'hf_repo': 'stanford-crfm/moleculeqa',
                'domain': 'molecular_biology_reasoning',
                'language': 'en',
                'license': 'apache-2.0',
                'estimated_size_gb': 0.02,
                'max_samples': 1500,
                'description': 'Molecular biology QA for Nobel Prize level research'
            },
            'physics_reasoning': {
                'hf_repo': 'allenai/physics_qa',
                'domain': 'theoretical_physics_reasoning',
                'language': 'en',
                'license': 'apache-2.0',
                'estimated_size_gb': 0.03,
                'max_samples': 2000,
                'description': 'Physics QA for theoretical breakthroughs'
            },

            # NSFWデータセット（安全判定学習用のみ、サイズ制限厳しく）
            'civil_comments': {
                'hf_repo': 'civil_comments',
                'domain': 'toxicity_detection',
                'language': 'en',
                'license': 'cc-by-4.0',
                'estimated_size_gb': 0.8,
                'max_samples': 5000,
                'description': 'Toxicity detection for safety PPO training',
                'nsfw_warning': True,
                'safety_only': True  # 安全判定学習専用
            },

            # 追加の推論強化データセット
            'commonsense_qa': {
                'hf_repo': 'commonsense_qa',
                'domain': 'commonsense_reasoning',
                'language': 'en',
                'license': 'unknown',
                'estimated_size_gb': 0.02,
                'max_samples': 2000,
                'description': 'Commonsense QA for everyday reasoning'
            },
            'social_iqa': {
                'hf_repo': 'social_i_qa',
                'domain': 'social_reasoning',
                'language': 'en',
                'license': 'unknown',
                'estimated_size_gb': 0.03,
                'max_samples': 2000,
                'description': 'Social IQ for social reasoning PPO'
            }
        }

    def check_git_lfs_setup(self) -> bool:
        """Git LFSのセットアップを確認"""
        try:
            result = subprocess.run(['git', 'lfs', 'version'],
                                  capture_output=True, text=True, check=True)
            print(f"[GIT LFS] Version: {result.stdout.strip()}")

            # Git LFSのトラッキング設定を確認
            result = subprocess.run(['git', 'lfs', 'track'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("[GIT LFS] Tracking patterns:")
                print(result.stdout)

            return True
        except subprocess.CalledProcessError as e:
            print(f"[GIT LFS] Error: {e}")
            return False
        except FileNotFoundError:
            print("[GIT LFS] Git LFS not found. Please install it first.")
            return False

    def download_dataset_with_lfs(self, dataset_name: str, config: Dict[str, Any]) -> bool:
        """Git LFSを使ってデータセットをダウンロード"""
        print(f"[DOWNLOAD] Starting download of {dataset_name}...")

        repo_url = f"https://huggingface.co/datasets/{config['hf_repo']}"
        local_path = self.output_dir / dataset_name

        try:
            # Git LFSでのクローン
            if local_path.exists():
                print(f"[DOWNLOAD] Dataset {dataset_name} already exists, updating...")
                # 既存のリポジトリを更新
                os.chdir(local_path)
                subprocess.run(['git', 'pull'], check=True)
                subprocess.run(['git', 'lfs', 'pull'], check=True)
            else:
                print(f"[DOWNLOAD] Cloning {repo_url}...")
                subprocess.run([
                    'git', 'lfs', 'clone', repo_url, str(local_path)
                ], check=True)

            # データセット情報の保存
            info_file = local_path / 'dataset_info.json'
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'name': dataset_name,
                    'config': config,
                    'download_time': datetime.now().isoformat(),
                    'local_path': str(local_path)
                }, f, indent=2, ensure_ascii=False)

            print(f"[DOWNLOAD] Successfully downloaded {dataset_name} to {local_path}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"[DOWNLOAD] Failed to download {dataset_name}: {e}")
            return False
        except Exception as e:
            print(f"[DOWNLOAD] Unexpected error for {dataset_name}: {e}")
            return False

    def download_dataset_with_hf_hub(self, dataset_name: str, config: Dict[str, Any]) -> bool:
        """HuggingFace Hub APIを使ってデータセットをダウンロード（Git LFSフォールバック）"""
        print(f"[HF HUB] Downloading {dataset_name} via API...")

        try:
            from huggingface_hub import snapshot_download
            import datasets

            local_path = self.output_dir / f"{dataset_name}_api"
            local_path.mkdir(exist_ok=True)

            # datasetsライブラリを使ってダウンロード
            try:
                print(f"[HF HUB] Using datasets library for {dataset_name}")

                # configパラメータがある場合
                if 'config' in config:
                    dataset = datasets.load_dataset(
                        config['hf_repo'],
                        config['config'],
                        split='train',
                        streaming=True
                    )
                else:
                    dataset = datasets.load_dataset(
                        config['hf_repo'],
                        split='train',
                        streaming=True
                    )

                # サンプルデータを保存
                sample_count = 0
                max_samples = 1000  # テスト用に制限

                data_file = local_path / 'samples.jsonl'
                with open(data_file, 'w', encoding='utf-8') as f:
                    for sample in dataset:
                        if sample_count >= max_samples:
                            break

                        json.dump(sample, f, ensure_ascii=False)
                        f.write('\n')
                        sample_count += 1

                print(f"[HF HUB] Downloaded {sample_count} samples using datasets library")

            except Exception as ds_error:
                print(f"[HF HUB] Datasets library failed: {ds_error}")
                print("[HF HUB] Falling back to snapshot download...")

                # snapshot_downloadでフォールバック
                downloaded_path = snapshot_download(
                    repo_id=config['hf_repo'],
                    repo_type="dataset",
                    local_dir=str(local_path),
                    allow_patterns=["*.json", "*.jsonl", "*.parquet", "*.txt", "*.csv"]
                )
                print(f"[HF HUB] Downloaded via snapshot to {downloaded_path}")

            # データセット情報の保存
            info_file = local_path / 'dataset_info.json'
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'name': dataset_name,
                    'config': config,
                    'download_time': datetime.now().isoformat(),
                    'local_path': str(local_path),
                    'download_method': 'hf_hub_api'
                }, f, indent=2, ensure_ascii=False)

            print(f"[HF HUB] Successfully downloaded {dataset_name} to {local_path}")
            return True

        except Exception as e:
            print(f"[HF HUB] Failed to download {dataset_name}: {e}")
            return False

    def check_size_limit(self, dataset_name: str, config: Dict[str, Any]) -> bool:
        """サイズ制限をチェック"""
        estimated_size = config.get('estimated_size_gb', 0)
        projected_total = self.current_total_size + (estimated_size * 1024**3)  # GB to bytes

        if projected_total > (self.max_total_size_gb * 1024**3):
            print(f"[SIZE] Skipping {dataset_name}: would exceed {self.max_total_size_gb}GB limit")
            print(f"  Current: {self.current_total_size / 1024**3:.2f}GB")
            print(f"  Projected: {projected_total / 1024**3:.2f}GB")
            return False

        return True

    def collect_all_datasets(self, use_git_lfs: bool = True) -> Dict[str, Any]:
        """全データセットを収集（サイズ制限付き）"""
        print(f"Starting dataset collection with Git LFS (max {self.max_total_size_gb}GB)...")
        print("=" * 60)

        # Git LFSの確認
        if use_git_lfs and not self.check_git_lfs_setup():
            print("[WARNING] Git LFS not properly configured, falling back to HF Hub API")
            use_git_lfs = False

        results = {
            'total_datasets': len(self.target_datasets),
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_datasets': 0,
            'total_size_gb': 0.0,
            'results': {},
            'collection_time': datetime.now().isoformat(),
            'size_limit_gb': self.max_total_size_gb
        }

        for dataset_name, config in self.target_datasets.items():
            print(f"\n[COLLECTION] Processing {dataset_name}...")
            print(f"  Description: {config['description']}")
            print(f"  Domain: {config['domain']}")
            print(f"  Estimated Size: {config.get('estimated_size_gb', 'unknown')}GB")

            if config.get('nsfw_warning'):
                print("  [WARN]  WARNING: This dataset contains NSFW content (for safety training only)")
                if config.get('safety_only'):
                    print("  🔒 SAFETY ONLY: This dataset is for safety PPO training exclusively")

            # サイズ制限チェック
            if not self.check_size_limit(dataset_name, config):
                results['skipped_datasets'] += 1
                results['results'][dataset_name] = {
                    'success': False,
                    'skipped': True,
                    'reason': 'size_limit_exceeded',
                    'config': config
                }
                continue

            success = False
            if use_git_lfs:
                success = self.download_dataset_with_lfs(dataset_name, config)

            if not success:
                print(f"  [FALLBACK] Trying HF Hub API for {dataset_name}")
                success = self.download_dataset_with_hf_hub(dataset_name, config)

            results['results'][dataset_name] = {
                'success': success,
                'config': config,
                'method': 'git_lfs' if success and use_git_lfs else 'hf_hub_api'
            }

            if success:
                # サイズを加算
                size_gb = config.get('estimated_size_gb', 0)
                self.current_total_size += size_gb * 1024**3
                results['total_size_gb'] += size_gb
                results['successful_downloads'] += 1

                print(f"  [OK] Downloaded successfully. Total size: {results['total_size_gb']:.2f}GB")
            else:
                results['failed_downloads'] += 1
                print(f"  [NG] Download failed")

        # 結果の保存
        results_file = self.output_dir / 'collection_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY:")
        print(f"Total datasets: {results['total_datasets']}")
        print(f"Successful: {results['successful_downloads']}")
        print(f"Failed: {results['failed_downloads']}")
        print(f"Skipped (size limit): {results['skipped_datasets']}")
        print(f"Total size: {results['total_size_gb']:.2f}GB / {self.max_total_size_gb}GB")
        print(f"Results saved to: {results_file}")

        return results

    def validate_downloaded_datasets(self) -> Dict[str, Any]:
        """ダウンロードしたデータセットの検証"""
        print("Validating downloaded datasets...")

        validation_results = {}

        for item in self.output_dir.iterdir():
            if item.is_dir() and (item / 'dataset_info.json').exists():
                dataset_name = item.name.replace('_api', '')

                try:
                    with open(item / 'dataset_info.json', 'r', encoding='utf-8') as f:
                        info = json.load(f)

                    # データセットサイズの計算
                    total_size = 0
                    file_count = 0

                    for file_path in item.rglob('*'):
                        if file_path.is_file() and file_path.name != 'dataset_info.json':
                            total_size += file_path.stat().st_size
                            file_count += 1

                    validation_results[dataset_name] = {
                        'valid': True,
                        'path': str(item),
                        'size_bytes': total_size,
                        'file_count': file_count,
                        'info': info
                    }

                except Exception as e:
                    validation_results[dataset_name] = {
                        'valid': False,
                        'error': str(e)
                    }

        # 検証結果の保存
        validation_file = self.output_dir / 'validation_results.json'
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"Validation results saved to: {validation_file}")
        return validation_results

    def convert_for_ppo_training(self) -> Dict[str, Any]:
        """PPO学習用にデータを変換"""
        print("Converting datasets for PPO training...")

        ppo_data_dir = self.output_dir / 'ppo_training_data'
        ppo_data_dir.mkdir(exist_ok=True)

        conversion_results = {
            'converted_datasets': 0,
            'total_samples': 0,
            'ppo_ready_files': []
        }

        for dataset_dir in self.output_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name in ['ppo_training_data']:
                continue

            dataset_name = dataset_dir.name.replace('_api', '')

            # データセット情報読み込み
            info_file = dataset_dir / 'dataset_info.json'
            if not info_file.exists():
                continue

            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)

            config = info['config']

            # PPO学習用の変換
            print(f"Converting {dataset_name} for PPO training...")

            # サンプルデータの収集
            samples_file = dataset_dir / 'samples.jsonl'
            if samples_file.exists():
                samples = []
                with open(samples_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))

                # PPO形式に変換
                ppo_samples = self._convert_samples_to_ppo_format(samples, config)

                # 保存
                output_file = ppo_data_dir / f"{dataset_name}_ppo.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in ppo_samples:
                        json.dump(sample, f, ensure_ascii=False)
                        f.write('\n')

                conversion_results['converted_datasets'] += 1
                conversion_results['total_samples'] += len(ppo_samples)
                conversion_results['ppo_ready_files'].append(str(output_file))

                print(f"  Converted {len(ppo_samples)} samples for PPO training")

        # 変換結果の保存
        conversion_file = ppo_data_dir / 'conversion_results.json'
        with open(conversion_file, 'w', encoding='utf-8') as f:
            json.dump(conversion_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"PPO conversion completed: {conversion_results['converted_datasets']} datasets, {conversion_results['total_samples']} samples")
        return conversion_results

    def _convert_samples_to_ppo_format(self, samples: List[Dict], config: Dict) -> List[Dict]:
        """サンプルをPPO学習用形式に変換"""
        ppo_samples = []
        domain = config['domain']
        language = config['language']

        for sample in samples:
            # PPO学習用の基本構造
            ppo_sample = {
                'input': '',
                'output': '',
                'domain': domain,
                'language': language,
                'ppo_metadata': {
                    'task_type': self._get_ppo_task_type(domain),
                    'difficulty': 'medium',
                    'requires_reasoning': True,
                    'multimodal': 'multimodal' in domain,
                    'safety_related': config.get('safety_only', False)
                }
            }

            # ドメイン別の変換
            if domain == 'multimodal_vision_language':
                # COCOキャプション形式
                if 'caption' in sample:
                    ppo_sample['input'] = "Describe this image:"
                    ppo_sample['output'] = sample['caption']
                elif 'question' in sample and 'answer' in sample:
                    ppo_sample['input'] = sample['question']
                    ppo_sample['output'] = sample['answer']

            elif 'qa' in domain or 'reasoning' in domain:
                # QA形式
                if 'question' in sample and 'answer' in sample:
                    ppo_sample['input'] = sample['question']
                    ppo_sample['output'] = sample['answer']
                elif 'query' in sample and 'response' in sample:
                    ppo_sample['input'] = sample['query']
                    ppo_sample['output'] = sample['response']

            elif domain == 'toxicity_detection':
                # 毒性検出形式（安全学習用）
                if 'text' in sample:
                    ppo_sample['input'] = sample['text']
                    ppo_sample['output'] = "SAFE" if not sample.get('toxic', False) else "UNSAFE"
                    ppo_sample['ppo_metadata']['task_type'] = 'safety_classification'

            elif 'knowledge' in domain or 'text' in domain:
                # 知識/テキスト形式
                if 'text' in sample:
                    ppo_sample['input'] = sample['text'][:500]  # 制限
                    ppo_sample['output'] = "Understood"  # 基本的な応答
                    ppo_sample['ppo_metadata']['task_type'] = 'knowledge_comprehension'

            # 最大サンプル数制限
            max_samples = config.get('max_samples', 10000)
            if len(ppo_samples) >= max_samples:
                break

            if ppo_sample['input'] and ppo_sample['output']:
                ppo_samples.append(ppo_sample)

        return ppo_samples

    def _get_ppo_task_type(self, domain: str) -> str:
        """ドメインからPPOタスクタイプを決定（PhD/Fields Medal級）"""
        task_mapping = {
            # 高度な数学・科学推論
            'advanced_mathematical_reasoning': 'phd_mathematics',
            'mathematical_proof_reasoning': 'theorem_proving',
            'mathematical_theorem_proving': 'fields_medal_mathematics',
            'interdisciplinary_science_reasoning': 'nobel_science',
            'molecular_biology_reasoning': 'molecular_biology_phd',
            'theoretical_physics_reasoning': 'physics_nobel',

            # ヒューリスティック・メタ推論
            'heuristic_reasoning': 'heuristic_meta_reasoning',
            'multi_hop_scientific_reasoning': 'scientific_inference_chaining',
            'strategic_reasoning': 'strategic_meta_cognition',

            # 知覚・認知層
            'multimodal_vision_language': 'so8_perception_cognition',
            'commonsense_reasoning': 'intuitive_reasoning',
            'social_reasoning': 'social_intuition',

            # 安全・倫理層
            'toxicity_detection': 'safety_meta_reasoning',

            # 言語・知識層
            'multilingual_knowledge': 'cross_lingual_intuition',
            'multilingual_qa': 'isomorphic_reasoning'
        }
        return task_mapping.get(domain, 'general_reasoning')

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Git LFS HuggingFace Dataset Collection for PPO Training")
    parser.add_argument('--output-dir', default='D:/webdataset/datasets/gitlfs_10gb',
                       help='Output directory for datasets')
    parser.add_argument('--max-size-gb', type=float, default=10.0,
                       help='Maximum total size in GB (default: 10.0)')
    parser.add_argument('--use-git-lfs', action='store_true', default=True,
                       help='Use Git LFS for downloading (default: True)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing datasets')
    parser.add_argument('--convert-ppo-only', action='store_true',
                       help='Only convert existing datasets for PPO training')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to download (default: all)')

    args = parser.parse_args()

    collector = GitLFSDatasetCollector(args.output_dir, args.max_size_gb)

    if args.convert_ppo_only:
        # PPO変換のみ
        results = collector.convert_for_ppo_training()
        print(f"PPO conversion completed: {results['converted_datasets']} datasets")
    elif args.validate_only:
        # 検証のみ
        results = collector.validate_downloaded_datasets()
        print(f"Validated {len(results)} datasets")
    else:
        # データセット収集
        if args.datasets:
            # 指定されたデータセットのみ
            filtered_datasets = {k: v for k, v in collector.target_datasets.items()
                               if k in args.datasets}
            collector.target_datasets = filtered_datasets

        results = collector.collect_all_datasets(args.use_git_lfs)

        if results['successful_downloads'] > 0:
            # 収集後に検証
            validation_results = collector.validate_downloaded_datasets()

            # PPO変換
            print("\nConverting datasets for PPO training...")
            ppo_results = collector.convert_for_ppo_training()

            # 最終レポート
            print("\n" + "=" * 60)
            print("FINAL REPORT:")
            print(f"Collection: {results['successful_downloads']}/{results['total_datasets']} successful")
            print(f"Total Size: {results['total_size_gb']:.2f}GB / {args.max_size_gb}GB")
            print(f"Validation: {sum(1 for r in validation_results.values() if r.get('valid'))} valid datasets")
            print(f"PPO Conversion: {ppo_results['converted_datasets']} datasets, {ppo_results['total_samples']} samples")

            # NSFWデータセットの警告
            nsfw_datasets = [name for name, config in collector.target_datasets.items()
                           if config.get('nsfw_warning')]
            if nsfw_datasets:
                print("\n[WARN]  NSFW DATASETS WARNING:")
                print("The following datasets contain sensitive content for safety training only:")
                for name in nsfw_datasets:
                    print(f"  - {name}: {collector.target_datasets[name]['description']}")
                print("These datasets should ONLY be used for safety PPO training and content filtering.")

if __name__ == '__main__':
    main()
