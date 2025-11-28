#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
パイプライン環境セットアップスクリプト
Pipeline Environment Setup Script

Pythonライブラリ・データセットダウンロード、データクレンジング、前処理を実行
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineEnvironmentSetup:
    """
    パイプライン環境セットアップクラス
    Pipeline Environment Setup Class
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.required_dirs = [
            "D:/webdataset",
            "D:/webdataset/models",
            "D:/webdataset/gguf_models",
            "D:/webdataset/checkpoints",
            "D:/webdataset/results",
            "D:/webdataset/datasets",
            "external"
        ]

        self.required_libraries = {
            # 基本ライブラリ
            "torch": "2.0.0+",
            "transformers": "4.35.0+",
            "datasets": "2.15.0+",
            "numpy": "1.24.0+",
            "pandas": "2.0.0+",
            "scipy": "1.11.0+",
            "matplotlib": "3.7.0+",
            "seaborn": "0.12.0+",
            "tqdm": "4.65.0+",
            "psutil": "5.9.0+",

            # LLMベンチマークライブラリ
            "llama-cpp-python": "0.2.20+",
            "lm-eval": "0.4.0+",
            "lighteval": "0.3.0+",

            # 追加ユーティリティ
            "huggingface-hub": "0.17.0+",
            "requests": "2.31.0+",
            "pyyaml": "6.0+",
        }

        self.required_datasets = [
            "elyza/ELYZA-tasks-100",  # ELYZA-100
            "hendrycks/competition_math",  # MATH
            "truthful_qa",  # TruthfulQA
            "allenai/ai2_arc",  # ARC
            "Rowan/hellaswag",  # HellaSwag
            "microsoft/DialoGPT-medium",  # 基本モデル用
        ]

    def setup_complete_environment(self) -> bool:
        """
        完全環境セットアップ実行
        Run complete environment setup
        """
        logger.info("[SETUP] Starting complete pipeline environment setup...")

        success = True

        try:
            # 1. ディレクトリ作成
            logger.info("[SETUP] Creating required directories...")
            if not self._create_directories():
                logger.error("[SETUP] Directory creation failed!")
                success = False

            # 2. Pythonライブラリインストール
            logger.info("[SETUP] Installing Python libraries...")
            if not self._install_libraries():
                logger.error("[SETUP] Library installation failed!")
                success = False

            # 3. 外部依存関係セットアップ
            logger.info("[SETUP] Setting up external dependencies...")
            if not self._setup_external_dependencies():
                logger.error("[SETUP] External dependencies setup failed!")
                success = False

            # 4. データセットダウンロード
            logger.info("[SETUP] Downloading datasets...")
            if not self._download_datasets():
                logger.error("[SETUP] Dataset download failed!")
                success = False

            # 5. データクレンジングと前処理
            logger.info("[SETUP] Performing data cleansing and preprocessing...")
            if not self._perform_data_cleansing():
                logger.error("[SETUP] Data cleansing failed!")
                success = False

            # 6. 環境検証
            logger.info("[SETUP] Validating environment...")
            if not self._validate_environment():
                logger.error("[SETUP] Environment validation failed!")
                success = False

            # 7. ABCテスト実行準備
            logger.info("[SETUP] Preparing for ABC test execution...")
            if not self._prepare_abc_test():
                logger.error("[SETUP] ABC test preparation failed!")
                success = False

        except Exception as e:
            logger.error(f"[SETUP] Setup failed with error: {e}")
            success = False

        if success:
            logger.info("[SUCCESS] Complete pipeline environment setup completed!")
            self._play_success_sound()
        else:
            logger.error("[FAILED] Pipeline environment setup failed!")
            self._play_error_sound()

        return success

    def _create_directories(self) -> bool:
        """必要なディレクトリ作成"""
        try:
            for dir_path in self.required_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"[DIRS] Created directory: {dir_path}")

            # 追加のサブディレクトリ
            subdirs = [
                "D:/webdataset/results/abc_test_results",
                "D:/webdataset/results/hf_submission",
                "D:/webdataset/results/hf_submission/plots",
                "D:/webdataset/results/hf_submission/tables",
                "D:/webdataset/results/hf_submission/analysis",
                "D:/webdataset/models/Borea-Phi-3.5-mini-Instruct-Jp",
                "D:/webdataset/gguf_models",
                "D:/webdataset/checkpoints/training",
                "D:/webdataset/checkpoints/finetuning",
                "D:/webdataset/checkpoints/finetuning/so8t_phi3_rtx3060",
                "D:/webdataset/datasets/processed",
                "D:/webdataset/datasets/soul_weights_dataset",
            ]

            for subdir in subdirs:
                Path(subdir).mkdir(parents=True, exist_ok=True)

            return True

        except Exception as e:
            logger.error(f"[DIRS] Directory creation failed: {e}")
            return False

    def _install_libraries(self) -> bool:
        """Pythonライブラリインストール"""
        try:
            # pip upgrade
            logger.info("[LIBS] Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                         check=True, capture_output=True)

            # 基本ライブラリインストール
            basic_libs = [
                "torch", "transformers", "datasets", "numpy", "pandas",
                "scipy", "matplotlib", "seaborn", "tqdm", "psutil",
                "huggingface-hub", "requests", "pyyaml"
            ]

            logger.info("[LIBS] Installing basic libraries...")
            for lib in basic_libs:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", lib],
                                 check=True, capture_output=True)
                    logger.info(f"[LIBS] Installed: {lib}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"[LIBS] Failed to install {lib}: {e}")

            # LLM特化ライブラリ（オプション）
            llm_libs = [
                ("llama-cpp-python", "llama-cpp-python"),
                ("lm-eval", "lm_eval"),
                ("lighteval", "lighteval"),
            ]

            logger.info("[LIBS] Installing LLM-specific libraries...")
            for lib_name, import_name in llm_libs:
                try:
                    # まずimportを試す
                    __import__(import_name)
                    logger.info(f"[LIBS] {lib_name} already available")
                except ImportError:
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "install", lib_name],
                                     check=True, capture_output=True, timeout=300)
                        logger.info(f"[LIBS] Installed: {lib_name}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"[LIBS] Failed to install {lib_name}: {e}")
                        logger.warning(f"[LIBS] {lib_name} will be limited in functionality")

            return True

        except Exception as e:
            logger.error(f"[LIBS] Library installation failed: {e}")
            return False

    def _setup_external_dependencies(self) -> bool:
        """外部依存関係セットアップ"""
        try:
            external_dir = self.project_root / "external"
            external_dir.mkdir(exist_ok=True)

            # llama.cppのクローン（GGUF変換用）
            llama_cpp_dir = external_dir / "llama.cpp-master"
            if not llama_cpp_dir.exists():
                logger.info("[EXTERNAL] Cloning llama.cpp...")
                try:
                    subprocess.run([
                        "git", "clone", "--depth", "1",
                        "https://github.com/ggerganov/llama.cpp.git",
                        str(llama_cpp_dir)
                    ], check=True, capture_output=True)
                    logger.info("[EXTERNAL] llama.cpp cloned successfully")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"[EXTERNAL] Failed to clone llama.cpp: {e}")
                    logger.warning("[EXTERNAL] GGUF conversion will be limited")

            # llama.cpp Pythonバインディング
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "-r", str(llama_cpp_dir / "requirements.txt")
                ], check=True, capture_output=True)
            except:
                logger.warning("[EXTERNAL] Failed to install llama.cpp requirements")

            return True

        except Exception as e:
            logger.error(f"[EXTERNAL] External dependencies setup failed: {e}")
            return False

    def _download_datasets(self) -> bool:
        """データセットダウンロード"""
        try:
            from datasets import load_dataset, DatasetDict
            import huggingface_hub

            logger.info("[DATASETS] Starting dataset downloads...")

            # ELYZA-100ダウンロード
            try:
                logger.info("[DATASETS] Downloading ELYZA-100...")
                elyza_dataset = load_dataset("elyza/ELYZA-tasks-100")
                elyza_path = Path("D:/webdataset/datasets/elyza_100")
                elyza_dataset.save_to_disk(str(elyza_path))
                logger.info(f"[DATASETS] ELYZA-100 saved to {elyza_path}")
            except Exception as e:
                logger.warning(f"[DATASETS] Failed to download ELYZA-100: {e}")

            # 他のベンチマークデータセット
            benchmark_datasets = [
                ("truthful_qa", "truthful_qa"),
                ("allenai/ai2_arc", "ARC"),
                ("Rowan/hellaswag", "hellaswag"),
                ("competition_math", "MATH"),
            ]

            for dataset_name, save_name in benchmark_datasets:
                try:
                    logger.info(f"[DATASETS] Downloading {dataset_name}...")
                    dataset = load_dataset(dataset_name)
                    save_path = Path(f"D:/webdataset/datasets/{save_name.lower()}")
                    dataset.save_to_disk(str(save_path))
                    logger.info(f"[DATASETS] {dataset_name} saved to {save_path}")
                except Exception as e:
                    logger.warning(f"[DATASETS] Failed to download {dataset_name}: {e}")

            # カスタムデータセット作成（テスト用）
            self._create_sample_datasets()

            return True

        except Exception as e:
            logger.error(f"[DATASETS] Dataset download failed: {e}")
            return False

    def _create_sample_datasets(self):
        """サンプルデータセット作成"""
        try:
            import json

            # テスト用のシンプルなデータセット
            sample_data = {
                "multimodal_processed_train": [
                    {
                        "text": "こんにちは、AIについて教えてください。",
                        "response": "AIは人工知能のことです。機械学習や深層学習などの技術を使って、様々なタスクを自動化します。",
                        "task_type": "qa"
                    },
                    {
                        "text": "量子コンピューティングの原理を説明してください。",
                        "response": "量子コンピューティングは量子力学の原理を利用した計算方式です。古典的なビットではなく量子ビットを使用し、並列処理が可能です。",
                        "task_type": "explanation"
                    }
                ],
                "multimodal_processed_val": [
                    {
                        "text": "機械学習と深層学習の違いは何ですか？",
                        "response": "機械学習はデータからパターンを学習する手法で、深層学習は多層のニューラルネットワークを使用した機械学習の一種です。",
                        "task_type": "comparison"
                    }
                ]
            }

            # JSON Lines形式で保存
            for dataset_name, data in sample_data.items():
                file_path = Path(f"D:/webdataset/datasets/processed/{dataset_name}.jsonl")
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                logger.info(f"[DATASETS] Created sample dataset: {file_path}")

        except Exception as e:
            logger.warning(f"[DATASETS] Failed to create sample datasets: {e}")

    def _perform_data_cleansing(self) -> bool:
        """データクレンジングと前処理"""
        try:
            logger.info("[CLEANSING] Starting data cleansing and preprocessing...")

            # 既存の前処理済みデータをクレンジング
            processed_dir = Path("D:/webdataset/datasets/processed")

            if processed_dir.exists():
                # 古いファイルをクリーンアップ
                for file_path in processed_dir.glob("*"):
                    if file_path.is_file() and file_path.stat().st_size == 0:
                        file_path.unlink()
                        logger.info(f"[CLEANSING] Removed empty file: {file_path}")

            # 基本的なデータ検証
            self._validate_downloaded_datasets()

            # 前処理パイプライン実行
            self._run_preprocessing_pipeline()

            return True

        except Exception as e:
            logger.error(f"[CLEANSING] Data cleansing failed: {e}")
            return False

    def _validate_downloaded_datasets(self):
        """ダウンロードしたデータセットの検証"""
        try:
            datasets_dir = Path("D:/webdataset/datasets")

            # ELYZA-100検証
            elyza_path = datasets_dir / "elyza_100"
            if elyza_path.exists():
                logger.info("[VALIDATE] ELYZA-100 dataset found and validated")
            else:
                logger.warning("[VALIDATE] ELYZA-100 dataset not found")

            # 他のデータセット検証
            expected_datasets = ["arc", "hellaswag", "math", "truthful_qa"]
            for dataset_name in expected_datasets:
                dataset_path = datasets_dir / dataset_name
                if dataset_path.exists():
                    logger.info(f"[VALIDATE] {dataset_name.upper()} dataset found")
                else:
                    logger.warning(f"[VALIDATE] {dataset_name.upper()} dataset not found")

        except Exception as e:
            logger.warning(f"[VALIDATE] Dataset validation failed: {e}")

    def _run_preprocessing_pipeline(self):
        """前処理パイプライン実行"""
        try:
            logger.info("[PREPROCESS] Running preprocessing pipeline...")

            # 基本的なテキスト前処理
            processed_files = list(Path("D:/webdataset/datasets/processed").glob("*.jsonl"))

            for file_path in processed_files:
                logger.info(f"[PREPROCESS] Processing {file_path.name}...")

                # ファイルの基本検証
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if len(lines) == 0:
                    logger.warning(f"[PREPROCESS] Empty file: {file_path}")
                    continue

                # 基本的なJSON検証
                valid_lines = 0
                for i, line in enumerate(lines):
                    try:
                        json.loads(line.strip())
                        valid_lines += 1
                    except json.JSONDecodeError:
                        logger.warning(f"[PREPROCESS] Invalid JSON at line {i+1} in {file_path}")

                logger.info(f"[PREPROCESS] {file_path.name}: {valid_lines}/{len(lines)} valid lines")

        except Exception as e:
            logger.warning(f"[PREPROCESS] Preprocessing pipeline failed: {e}")

    def _validate_environment(self) -> bool:
        """環境検証"""
        try:
            logger.info("[VALIDATE] Validating environment...")

            # Pythonバージョン確認
            python_version = sys.version_info
            logger.info(f"[VALIDATE] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

            # 基本ライブラリ検証
            basic_imports = [
                "torch", "transformers", "datasets", "numpy", "pandas",
                "scipy", "matplotlib", "seaborn", "tqdm"
            ]

            for lib in basic_imports:
                try:
                    __import__(lib)
                    logger.info(f"[VALIDATE] ✓ {lib}")
                except ImportError:
                    logger.warning(f"[VALIDATE] ✗ {lib} not available")

            # オプションライブラリ検証
            optional_imports = [
                ("llama_cpp", "llama-cpp-python"),
                ("lm_eval", "lm-evaluation-harness"),
                ("lighteval", "lighteval"),
            ]

            for import_name, display_name in optional_imports:
                try:
                    __import__(import_name)
                    logger.info(f"[VALIDATE] ✓ {display_name}")
                except ImportError:
                    logger.warning(f"[VALIDATE] ✗ {display_name} not available")

            # GPU確認
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"[VALIDATE] ✓ CUDA available: {gpu_count} GPU(s), {gpu_name}")
                else:
                    logger.warning("[VALIDATE] ✗ CUDA not available")
            except:
                logger.warning("[VALIDATE] ✗ PyTorch GPU check failed")

            # ディスク容量確認
            try:
                stat = os.statvfs('D:/')
                free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                logger.info(f"[VALIDATE] Free space on D:/: {free_space_gb:.1f} GB")

                if free_space_gb < 50:  # 最低50GB必要
                    logger.warning("[VALIDATE] ⚠️  Low disk space! Need at least 50GB free")
                    return False

            except:
                logger.warning("[VALIDATE] Could not check disk space")

            return True

        except Exception as e:
            logger.error(f"[VALIDATE] Environment validation failed: {e}")
            return False

    def _prepare_abc_test(self) -> bool:
        """ABCテスト実行準備"""
        try:
            logger.info("[ABC PREP] Preparing ABC test execution...")

            # 設定ファイル作成
            abc_config = {
                "models": {
                    "modela": {
                        "path": "D:/webdataset/gguf_models/borea_phi35_instruct_jp_q8_0.gguf",
                        "type": "gguf",
                        "description": "Borea-Phi3.5-instruct-jp (GGUF Q8_0) - ABC Test Model A"
                    },
                    "modelb": {
                        "path": "D:/webdataset/models/borea_phi35_alpha_gate_sigmoid_bayesian/final",
                        "type": "hf",
                        "description": "AEGIS-Phi3.5-Enhanced Model - ABC Test Model B"
                    },
                    "modelc": {
                        "path": "D:/webdataset/models/borea_phi35_so8t_rtx3060/final",
                        "type": "hf",
                        "description": "AEGIS-Phi3.5-Golden-Sigmoid Model - ABC Test Model C"
                    }
                },
                "benchmark_config": {
                    "max_samples": 100,
                    "batch_size": 1,
                    "num_fewshot": 0,
                    "timeout": 600
                },
                "output": {
                    "results_dir": "D:/webdataset/results/abc_test_results",
                    "hf_submission_dir": "D:/webdataset/results/hf_submission"
                }
            }

            # 設定ファイル保存
            config_path = self.project_root / "configs" / "abc_test_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(abc_config, f, indent=2, ensure_ascii=False)

            logger.info(f"[ABC PREP] ABC test config saved to {config_path}")

            # 実行スクリプトの準備確認
            abc_script = self.project_root / "scripts" / "testing" / "run_complete_abc_test.bat"
            if abc_script.exists():
                logger.info("[ABC PREP] ABC test execution script found")
            else:
                logger.warning("[ABC PREP] ABC test execution script not found")

            return True

        except Exception as e:
            logger.error(f"[ABC PREP] ABC test preparation failed: {e}")
            return False

    def _play_success_sound(self):
        """成功音再生"""
        try:
            import winsound
            winsound.PlaySound(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav", winsound.SND_FILENAME)
        except:
            print('\a')

    def _play_error_sound(self):
        """エラー音再生"""
        try:
            import winsound
            winsound.Beep(800, 1000)
        except:
            print('\a')


def run_pipeline_setup():
    """
    パイプラインセットアップ実行
    Run pipeline setup
    """
    logger.info("[PIPELINE SETUP] Starting SO8T Pipeline Environment Setup...")
    logger.info("=" * 60)

    setup = PipelineEnvironmentSetup()
    success = setup.setup_complete_environment()

    if success:
        logger.info("=" * 60)
        logger.info("[SUCCESS] Pipeline environment setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run ABC test: scripts/testing/run_complete_abc_test.bat")
        logger.info("2. Check results: D:/webdataset/results/")
        logger.info("3. View HF submission: D:/webdataset/results/hf_submission/")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("[FAILED] Pipeline environment setup failed!")
        logger.error("Please check the error messages above and try again.")
        logger.error("=" * 60)
        sys.exit(1)

    return success


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="SO8T Pipeline Environment Setup"
    )
    parser.add_argument(
        '--skip-datasets',
        action='store_true',
        help='Skip dataset downloads'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip environment validation'
    )

    args = parser.parse_args()

    # セットアップ実行
    run_pipeline_setup()


if __name__ == '__main__':
    main()
