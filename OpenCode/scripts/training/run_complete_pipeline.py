#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Complete Pipeline: 学習→焼き込み→GGUF変換→量子化→較正

日本語ドメイン別知識とコーディング能力向上を狙った完全自動化パイプライン

Usage:
    python scripts/training/run_complete_pipeline.py \
        --base_model models/Borea-Phi-3.5-mini-Instruct-Jp \
        --japanese_dataset D:/webdataset/japanese_training_dataset/train.jsonl \
        --coding_dataset D:/webdataset/coding_dataset/train.jsonl \
        --output_base D:/webdataset/so8t_complete
"""

import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class CompletePipeline:
    """完全自動化パイプライン"""
    
    def __init__(
        self,
        base_model: str,
        japanese_dataset: Optional[str] = None,
        coding_dataset: Optional[str] = None,
        thinking_dataset: Optional[str] = None,
        four_class_dataset: Optional[str] = None,
        domain_knowledge_dataset: Optional[str] = None,
        output_base: str = "D:/webdataset/so8t_complete",
        lora_r: int = 16,
        lora_alpha: int = 32,
        quantization: str = "Q5_K_M",
        llama_cpp_dir: Optional[str] = None,
        use_cursor_browser: bool = True
    ):
        """
        Args:
            base_model: ベースモデルパス
            japanese_dataset: 日本語データセットパス（JSONL）
            coding_dataset: コーディングデータセットパス（JSONL）
            thinking_dataset: 四重推論データセットパス（JSONL）
            four_class_dataset: 四値分類データセットパス（JSONL）
            output_base: 出力ベースディレクトリ
            lora_r: LoRAランク
            lora_alpha: LoRA alpha
            quantization: 量子化タイプ（Q4_K_M or Q5_K_M）
            llama_cpp_dir: llama.cppディレクトリパス
        """
        self.base_model = Path(base_model)
        self.japanese_dataset = Path(japanese_dataset) if japanese_dataset else None
        self.coding_dataset = Path(coding_dataset) if coding_dataset else None
        self.thinking_dataset = Path(thinking_dataset) if thinking_dataset else None
        self.four_class_dataset = Path(four_class_dataset) if four_class_dataset else None
        self.domain_knowledge_dataset = Path(domain_knowledge_dataset) if domain_knowledge_dataset else None
        self.output_base = Path(output_base)
        self.use_cursor_browser = use_cursor_browser
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.quantization = quantization
        self.llama_cpp_dir = Path(llama_cpp_dir) if llama_cpp_dir else PROJECT_ROOT / "external" / "llama.cpp-master"
        
        # 出力ディレクトリ
        self.checkpoint_dir = self.output_base / "checkpoints" / "training"
        self.baked_dir = self.output_base / "models" / "baked"
        self.gguf_dir = self.output_base / "gguf_models"
        self.calibration_dir = self.output_base / "calibration"
        
        # パイプライン状態
        self.state = {
            "training": False,
            "baking": False,
            "conversion": False,
            "quantization": False,
            "calibration": False
        }
        
        # ログファイル
        self.log_dir = PROJECT_ROOT / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def find_available_datasets(self) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
        """
        利用可能なデータセットを検索
        
        Returns:
            (日本語データセットパス, コーディングデータセットパス, 四重推論データセットパス, 四値分類データセットパス)
        """
        japanese_path = None
        coding_path = None
        thinking_path = None
        four_class_path = None
        
        # 指定されたパスを確認
        if self.japanese_dataset and self.japanese_dataset.exists():
            japanese_path = self.japanese_dataset
        else:
            # デフォルトパスを確認
            default_japanese = Path("D:/webdataset/japanese_training_dataset/train.jsonl")
            if default_japanese.exists():
                japanese_path = default_japanese
            else:
                # dataディレクトリ内の日本語データセットを検索
                data_dir = PROJECT_ROOT / "data"
                if data_dir.exists():
                    japanese_files = list(data_dir.glob("*japanese*.jsonl"))
                    if japanese_files:
                        japanese_path = japanese_files[0]
                        logger.info(f"Found Japanese dataset: {japanese_path}")
        
        if self.coding_dataset and self.coding_dataset.exists():
            coding_path = self.coding_dataset
        else:
            # デフォルトパスを確認
            default_coding = Path("D:/webdataset/coding_dataset/train.jsonl")
            if default_coding.exists():
                coding_path = default_coding
            else:
                # dataディレクトリ内のコーディングデータセットを検索
                data_dir = PROJECT_ROOT / "data"
                if data_dir.exists():
                    coding_files = list(data_dir.glob("*coding*.jsonl"))
                    if coding_files:
                        coding_path = coding_files[0]
                        logger.info(f"Found coding dataset: {coding_path}")
        
        # 四重推論データセット検索
        if self.thinking_dataset and self.thinking_dataset.exists():
            thinking_path = self.thinking_dataset
        else:
            # dataディレクトリ内の四重推論データセットを検索
            data_dir = PROJECT_ROOT / "data"
            if data_dir.exists():
                # processed/thinking/配下を検索
                thinking_dir = data_dir / "processed" / "thinking"
                if thinking_dir.exists():
                    thinking_files = list(thinking_dir.glob("thinking_*.jsonl"))
                    if thinking_files:
                        thinking_path = thinking_files[0]
                        logger.info(f"Found thinking dataset: {thinking_path}")
                # 直接検索
                if not thinking_path:
                    thinking_files = list(data_dir.glob("*thinking*.jsonl"))
                    if thinking_files:
                        thinking_path = thinking_files[0]
                        logger.info(f"Found thinking dataset: {thinking_path}")
        
        # 四値分類データセット検索
        if self.four_class_dataset and self.four_class_dataset.exists():
            four_class_path = self.four_class_dataset
        else:
            # dataディレクトリ内の四値分類データセットを検索
            data_dir = PROJECT_ROOT / "data"
            if data_dir.exists():
                # safetyデータセットを検索
                safety_files = list(data_dir.glob("*safety*.jsonl"))
                if safety_files:
                    four_class_path = safety_files[0]
                    logger.info(f"Found four-class dataset: {four_class_path}")
                # four_classデータセットを検索
                if not four_class_path:
                    four_class_files = list(data_dir.glob("*four_class*.jsonl"))
                    if four_class_files:
                        four_class_path = four_class_files[0]
                        logger.info(f"Found four-class dataset: {four_class_path}")
        
        return japanese_path, coding_path, thinking_path, four_class_path
    
    def collect_training_data(self, collect_domain_knowledge: bool = True) -> Optional[Path]:
        """
        Step 0-1: Playwrightで学習用データを収集（ドメイン別知識収集統合版）
        
        Args:
            collect_domain_knowledge: ドメイン別知識収集を実行するか
        
        Returns:
            収集済みデータセットパス（収集がスキップされた場合はNone）
        """
        logger.info("="*80)
        logger.info("Step 0-1: Collecting training data with Playwright (Domain Knowledge Integrated)")
        logger.info("="*80)
        
        # 既存のデータセットが十分にある場合はスキップ
        japanese_path, coding_path, thinking_path, four_class_path = self.find_available_datasets()
        total_existing = sum([
            len(list(Path(p).read_text(encoding='utf-8').split('\n'))) if p and Path(p).exists() else 0
            for p in [japanese_path, coding_path, thinking_path, four_class_path]
        ])
        
        if total_existing >= 10000:  # 既存データが10,000サンプル以上ある場合はスキップ
            logger.info(f"[SKIP] Sufficient existing data found ({total_existing} samples), skipping collection")
            return None
        
        # ドメイン別知識収集を実行
        if collect_domain_knowledge:
            logger.info("[DOMAIN] Collecting domain-specific knowledge...")
            domain_output = self.output_base / "domain_knowledge_collected"
            domain_output.mkdir(parents=True, exist_ok=True)
            
            # SO8T/thinkingモデルパスを検索
            so8t_model_path = None
            possible_paths = [
                Path("models/so8t_thinking"),
                Path("D:/webdataset/models/so8t_thinking"),
                Path("models/so8t_retrained/final_model"),
            ]
            for path in possible_paths:
                if path.exists():
                    so8t_model_path = str(path)
                    break
            
            domain_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "data" / "collect_domain_knowledge_with_playwright.py"),
                "--output", str(domain_output),
                "--domains", "defense,aerospace,transport,general,nsfw_detection,drug_detection",
                "--delay", "2.0",
                "--timeout", "30000",
                "--max_pages_per_domain", "100",
                "--max_depth", "3",
                "--quality_threshold", "0.7"
            ]
            
            if so8t_model_path:
                domain_cmd.extend(["--so8t_model_path", so8t_model_path])
                logger.info(f"[DOMAIN] Using SO8T/thinking model: {so8t_model_path}")
            else:
                logger.warning("[DOMAIN] SO8T/thinking model not found, continuing without labeling")
            
            if self.use_cursor_browser:
                domain_cmd.extend(["--use_cursor_browser", "--remote_debugging_port", "9222"])
            
            logger.info(f"Running domain knowledge collection: {' '.join(domain_cmd)}")
            
            try:
                domain_result = subprocess.run(
                    domain_cmd,
                    check=False,  # エラーでも続行
                    capture_output=False,
                    text=True
                )
                if domain_result.returncode == 0:
                    logger.info("[OK] Domain knowledge collection completed successfully")
                else:
                    logger.warning("[WARNING] Domain knowledge collection failed, continuing...")
            except Exception as e:
                logger.warning(f"[WARNING] Domain knowledge collection error: {e}, continuing...")
        
        # Playwrightデータ収集スクリプトを実行
        collection_output = self.output_base / "collected_data"
        collection_output.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "data" / "collect_training_data_with_playwright.py"),
            "--output", str(collection_output),
            "--sources", "wikipedia_ja,github,stackoverflow,qiita",
            "--target_samples", "50000",
            "--use_cursor_browser",
            "--delay", "2.0",
            "--timeout", "30000",
            "--max_depth", "3",
            "--max_pages_per_source", "500"
        ]
        
        logger.info(f"Running data collection: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            logger.info("[OK] Data collection completed successfully")
            
            # 収集されたデータファイルを検索
            collected_files = list(collection_output.glob("training_data_*.jsonl"))
            if collected_files:
                return collected_files[0]
            else:
                logger.warning("[WARNING] No collected data files found")
                return None
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Data collection failed: {e}")
            return None
    
    def merge_datasets(self) -> Path:
        """
        日本語データセットとコーディングデータセットをマージ
        
        Returns:
            マージ済みデータセットパス
        """
        logger.info("="*80)
        logger.info("Step 0: Merging datasets")
        logger.info("="*80)
        
        merged_dataset = self.output_base / "merged_dataset.jsonl"
        merged_dataset.parent.mkdir(parents=True, exist_ok=True)
        
        # 利用可能なデータセットを検索
        japanese_path, coding_path, thinking_path, four_class_path = self.find_available_datasets()
        
        # ドメイン別知識データセット検索
        domain_knowledge_path = None
        if self.domain_knowledge_dataset and self.domain_knowledge_dataset.exists():
            domain_knowledge_path = self.domain_knowledge_dataset
        else:
            # デフォルトパスを確認
            default_domain = Path("D:/webdataset/domain_knowledge_collected")
            if default_domain.exists():
                domain_files = list(default_domain.glob("domain_knowledge_*_cleaned.jsonl"))
                if domain_files:
                    domain_knowledge_path = domain_files[0]
                    logger.info(f"Found domain knowledge dataset: {domain_knowledge_path}")
        
        samples = []
        
        # ドメイン別知識データセット読み込み
        if domain_knowledge_path and domain_knowledge_path.exists():
            logger.info(f"Loading domain knowledge dataset: {domain_knowledge_path}")
            domain_count = 0
            with open(domain_knowledge_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        # ドメイン別知識データをinstruction形式に変換
                        instruction = sample.get("instruction", f"以下の{sample.get('title', 'コンテンツ')}について説明してください。")
                        output = sample.get("output", sample.get("text", ""))
                        
                        # 四重推論と四値分類ラベルがある場合は保持
                        merged_sample = {
                            "instruction": instruction,
                            "output": output
                        }
                        
                        if "quadruple_thinking" in sample:
                            merged_sample["quadruple_thinking"] = sample["quadruple_thinking"]
                        
                        if "four_class_label" in sample:
                            merged_sample["four_class_label"] = sample["four_class_label"]
                        
                        if "quality_score" in sample:
                            merged_sample["quality_score"] = sample["quality_score"]
                        
                        samples.append(merged_sample)
                        domain_count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"  Loaded {domain_count} domain knowledge samples")
        else:
            logger.warning("Domain knowledge dataset not found, skipping")
        
        # 日本語データセット読み込み
        if japanese_path and japanese_path.exists():
            logger.info(f"Loading Japanese dataset: {japanese_path}")
            japanese_count = 0
            with open(japanese_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                        japanese_count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"  Loaded {japanese_count} Japanese samples")
        else:
            logger.warning("Japanese dataset not found, skipping")
        
        # コーディングデータセット読み込み
        if coding_path and coding_path.exists():
            logger.info(f"Loading coding dataset: {coding_path}")
            coding_count = 0
            with open(coding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                        coding_count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"  Loaded {coding_count} coding samples")
        else:
            logger.warning("Coding dataset not found, skipping")
        
        # 四重推論データセット読み込み
        if thinking_path and thinking_path.exists():
            logger.info(f"Loading thinking dataset: {thinking_path}")
            thinking_count = 0
            with open(thinking_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        # 四重推論形式をinstruction形式に変換
                        if "quadruple_thinking" in sample or "thinking" in sample:
                            # 四重推論データをinstruction形式に変換
                            instruction = sample.get("instruction", sample.get("query", ""))
                            thinking_data = sample.get("quadruple_thinking", sample.get("thinking", {}))
                            if isinstance(thinking_data, dict):
                                output = f"""<think-task>{thinking_data.get('task', '')}</think-task>
<think-safety>{thinking_data.get('safety', '')}</think-safety>
<think-policy>{thinking_data.get('policy', '')}</think-policy>
<final>{thinking_data.get('final', sample.get('output', ''))}</final>"""
                            else:
                                output = sample.get("output", "")
                            samples.append({
                                "instruction": instruction,
                                "output": output
                            })
                        else:
                            samples.append(sample)
                        thinking_count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"  Loaded {thinking_count} thinking samples")
        else:
            logger.warning("Thinking dataset not found, skipping")
        
        # 四値分類データセット読み込み
        if four_class_path and four_class_path.exists():
            logger.info(f"Loading four-class dataset: {four_class_path}")
            four_class_count = 0
            with open(four_class_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        # 四値分類ラベルを含むデータをinstruction形式に変換
                        instruction = sample.get("instruction", sample.get("text", sample.get("query", "")))
                        output = sample.get("output", sample.get("response", ""))
                        four_class_label = sample.get("four_class_label", sample.get("safety_label", "ALLOW"))
                        
                        # 四値分類ラベルを出力に含める
                        formatted_output = f"[{four_class_label}] {output}"
                        samples.append({
                            "instruction": instruction,
                            "output": formatted_output,
                            "four_class_label": four_class_label
                        })
                        four_class_count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"  Loaded {four_class_count} four-class samples")
        else:
            logger.warning("Four-class dataset not found, skipping")
        
        # データセットが空の場合はエラー
        if len(samples) == 0:
            logger.error("[ERROR] No training samples found. Please prepare datasets first.")
            logger.error("  Japanese dataset: Use scripts/data/collect_japanese_training_dataset.py")
            logger.error("  Coding dataset: Use scripts/pipelines/prepare_coding_training_data.py")
            raise ValueError("No training samples available")
        
        # マージ済みデータセット保存
        logger.info(f"Saving merged dataset: {merged_dataset}")
        with open(merged_dataset, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"  Total samples: {len(samples)}")
        logger.info(f"  Merged dataset saved to: {merged_dataset}")
        
        return merged_dataset
    
    def run_training(self, dataset_path: Path) -> bool:
        """
        Step 1: SO8T QLoRA学習
        
        Args:
            dataset_path: 学習データセットパス
        
        Returns:
            成功したかどうか
        """
        logger.info("="*80)
        logger.info("Step 1: SO8T QLoRA Training")
        logger.info("="*80)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train_so8t_lora.py"),
            "--base_model", str(self.base_model),
            "--dataset", str(dataset_path),
            "--output_dir", str(self.checkpoint_dir),
            "--lora_r", str(self.lora_r),
            "--lora_alpha", str(self.lora_alpha),
            "--lora_dropout", "0.05",
            "--batch_size", "1",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", "2e-4",
            "--num_epochs", "3",
            "--load_in_4bit",
            "--save_steps", "500",
            "--logging_steps", "10"
        ]
        
        logger.info(f"Running training command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            self.state["training"] = True
            logger.info("[OK] Training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Training failed: {e}")
            return False
    
    def run_baking(self) -> bool:
        """
        Step 2: SO8T回転の焼き込み
        
        Returns:
            成功したかどうか
        """
        logger.info("="*80)
        logger.info("Step 2: SO8T Rotation Bake-in")
        logger.info("="*80)
        
        self.baked_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "bakein_o_proj.py"),
            "--model_path", str(self.checkpoint_dir),
            "--output_path", str(self.baked_dir),
            "--base_model", str(self.base_model)
        ]
        
        logger.info(f"Running baking command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            self.state["baking"] = True
            logger.info("[OK] Baking completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Baking failed: {e}")
            return False
    
    def run_conversion_and_quantization(self) -> bool:
        """
        Step 3: GGUF変換・量子化
        
        Returns:
            成功したかどうか
        """
        logger.info("="*80)
        logger.info("Step 3: GGUF Conversion & Quantization")
        logger.info("="*80)
        
        self.gguf_dir.mkdir(parents=True, exist_ok=True)
        
        # Windows用バッチファイルを使用
        import platform
        if platform.system() == "Windows":
            script_path = PROJECT_ROOT / "scripts" / "training" / "convert_and_quantize.bat"
            cmd = [
                str(script_path),
                str(self.baked_dir),
                str(self.gguf_dir),
                self.quantization,
                str(self.llama_cpp_dir)
            ]
        else:
            script_path = PROJECT_ROOT / "scripts" / "training" / "convert_and_quantize.sh"
            cmd = [
                "bash",
                str(script_path),
                str(self.baked_dir),
                str(self.gguf_dir),
                self.quantization,
                str(self.llama_cpp_dir)
            ]
        
        logger.info(f"Running conversion command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            self.state["conversion"] = True
            self.state["quantization"] = True
            logger.info("[OK] Conversion and quantization completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Conversion/quantization failed: {e}")
            return False
    
    def run_calibration(self, val_data_path: Optional[Path] = None) -> bool:
        """
        Step 4: AED較正
        
        Args:
            val_data_path: 検証データパス（オプション）
        
        Returns:
            成功したかどうか
        """
        logger.info("="*80)
        logger.info("Step 4: AED Calibration")
        logger.info("="*80)
        
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        
        # GGUFファイルを検索
        model_name = self.baked_dir.name
        gguf_file = self.gguf_dir / f"{model_name}_{self.quantization}.gguf"
        
        if not gguf_file.exists():
            logger.error(f"[ERROR] GGUF file not found: {gguf_file}")
            return False
        
        # 検証データが指定されていない場合はスキップ
        if val_data_path is None or not val_data_path.exists():
            logger.warning("Validation data not provided, skipping calibration")
            logger.info("  To run calibration manually:")
            logger.info(f"    python scripts/training/calibrate_aed.py \\")
            logger.info(f"        --model {gguf_file} \\")
            logger.info(f"        --val_data <val_data_path> \\")
            logger.info(f"        --output_dir {self.calibration_dir}")
            return True
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "calibrate_aed.py"),
            "--model", str(gguf_file),
            "--val_data", str(val_data_path),
            "--output_dir", str(self.calibration_dir)
        ]
        
        logger.info(f"Running calibration command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            self.state["calibration"] = True
            logger.info("[OK] Calibration completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Calibration failed: {e}")
            return False
    
    def run_complete_pipeline(
        self,
        val_data_path: Optional[Path] = None,
        skip_training: bool = False,
        skip_baking: bool = False,
        skip_conversion: bool = False,
        skip_calibration: bool = False
    ) -> bool:
        """
        完全パイプラインを実行
        
        Args:
            val_data_path: 検証データパス（較正用）
            skip_training: 学習をスキップ
            skip_baking: 焼き込みをスキップ
            skip_conversion: 変換をスキップ
            skip_calibration: 較正をスキップ
        
        Returns:
            成功したかどうか
        """
        logger.info("="*80)
        logger.info("SO8T Complete Pipeline: Training → Baking → GGUF → Quantization → Calibration")
        logger.info("="*80)
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Output base: {self.output_base}")
        logger.info(f"Quantization: {self.quantization}")
        logger.info("")
        
        start_time = datetime.now()
        
        # Step 0-1: Playwrightで学習用データを収集（オプション）
        collected_data = self.collect_training_data()
        if collected_data:
            logger.info(f"[OK] Collected data: {collected_data}")
        
        # Step 0-2: データセットマージ
        merged_dataset = self.merge_datasets()
        
        # Step 1: 学習
        if not skip_training:
            if not self.run_training(merged_dataset):
                logger.error("[ERROR] Pipeline stopped at training step")
                return False
        else:
            logger.info("[SKIP] Training step skipped")
        
        # Step 2: 焼き込み
        if not skip_baking:
            if not self.run_baking():
                logger.error("[ERROR] Pipeline stopped at baking step")
                return False
        else:
            logger.info("[SKIP] Baking step skipped")
        
        # Step 3: GGUF変換・量子化
        if not skip_conversion:
            if not self.run_conversion_and_quantization():
                logger.error("[ERROR] Pipeline stopped at conversion/quantization step")
                return False
        else:
            logger.info("[SKIP] Conversion/quantization step skipped")
        
        # Step 4: 較正
        if not skip_calibration:
            if not self.run_calibration(val_data_path):
                logger.warning("[WARNING] Calibration step failed or skipped")
        else:
            logger.info("[SKIP] Calibration step skipped")
        
        # 完了
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*80)
        logger.info("Pipeline Summary")
        logger.info("="*80)
        logger.info(f"Duration: {duration}")
        logger.info(f"Training: {'[OK]' if self.state['training'] else '[SKIP]'}")
        logger.info(f"Baking: {'[OK]' if self.state['baking'] else '[SKIP]'}")
        logger.info(f"Conversion: {'[OK]' if self.state['conversion'] else '[SKIP]'}")
        logger.info(f"Quantization: {'[OK]' if self.state['quantization'] else '[SKIP]'}")
        logger.info(f"Calibration: {'[OK]' if self.state['calibration'] else '[SKIP]'}")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  Checkpoints: {self.checkpoint_dir}")
        logger.info(f"  Baked model: {self.baked_dir}")
        logger.info(f"  GGUF models: {self.gguf_dir}")
        logger.info(f"  Calibration: {self.calibration_dir}")
        logger.info("="*80)
        logger.info("Pipeline completed!")
        logger.info("="*80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description="SO8T Complete Pipeline")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model path")
    parser.add_argument("--japanese_dataset", type=str, default=None,
                       help="Japanese dataset path (JSONL)")
    parser.add_argument("--coding_dataset", type=str, default=None,
                       help="Coding dataset path (JSONL)")
    parser.add_argument("--thinking_dataset", type=str, default=None,
                       help="Quadruple thinking dataset path (JSONL)")
    parser.add_argument("--four_class_dataset", type=str, default=None,
                       help="Four-class classification dataset path (JSONL)")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Validation dataset path for calibration (JSONL)")
    parser.add_argument("--output_base", type=str, default="D:/webdataset/so8t_complete",
                       help="Output base directory")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--quantization", type=str, default="Q5_K_M",
                       choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
                       help="Quantization type")
    parser.add_argument("--llama_cpp_dir", type=str, default=None,
                       help="llama.cpp directory path")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training step")
    parser.add_argument("--skip_baking", action="store_true",
                       help="Skip baking step")
    parser.add_argument("--skip_conversion", action="store_true",
                       help="Skip conversion/quantization step")
    parser.add_argument("--skip_calibration", action="store_true",
                       help="Skip calibration step")
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = CompletePipeline(
        base_model=args.base_model,
        japanese_dataset=args.japanese_dataset,
        coding_dataset=args.coding_dataset,
        thinking_dataset=args.thinking_dataset,
        four_class_dataset=args.four_class_dataset,
        output_base=args.output_base,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        quantization=args.quantization,
        llama_cpp_dir=args.llama_cpp_dir
    )
    
    val_data_path = Path(args.val_data) if args.val_data else None
    
    success = pipeline.run_complete_pipeline(
        val_data_path=val_data_path,
        skip_training=args.skip_training,
        skip_baking=args.skip_baking,
        skip_conversion=args.skip_conversion,
        skip_calibration=args.skip_calibration
    )
    
    if success:
        logger.info("Pipeline execution completed successfully!")
        return 0
    else:
        logger.error("Pipeline execution failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

