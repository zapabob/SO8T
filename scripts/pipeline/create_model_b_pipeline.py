#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデルB作成パイプライン

焼きこみ → 事後学習 → ファインチューニング → 温度較正の順で実行

Usage:
    python scripts/create_model_b_pipeline.py --config configs/ab_test_borea_phi35.yaml
"""

import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class ModelBPipeline:
    """モデルB作成パイプライン"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config = self._load_config(config_path)
        self.base_model_path = self.config['model']['base_model_path']
        self.output_base_dir = Path(self.config['output']['base_dir'])
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 各段階の出力ディレクトリ
        self.quantized_dir = self.output_base_dir / "quantized"
        self.post_trained_dir = self.output_base_dir / "post_trained"
        self.fine_tuned_dir = self.output_base_dir / "fine_tuned"
        self.calibrated_dir = self.output_base_dir / "calibrated"
        
        logger.info("="*80)
        logger.info("Model B Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Base model: {self.base_model_path}")
        logger.info(f"Output base dir: {self.output_base_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _update_so8t_tokenizer_info(self, model_dir: Path, tokenizer, model_name: str = "so8t-borea-phi35"):
        """
        convert_hf_to_gguf_update.pyを使用してSO8Tトークナイザー情報を更新
        
        Args:
            model_dir: モデルディレクトリパス
            tokenizer: トークナイザーオブジェクト
            model_name: SO8Tモデル名
        """
        try:
            import subprocess
            from hashlib import sha256
            
            # convert_hf_to_gguf_update.pyのパス
            update_script = PROJECT_ROOT / "external" / "llama.cpp-master" / "convert_hf_to_gguf_update.py"
            
            if not update_script.exists():
                logger.warning(f"convert_hf_to_gguf_update.py not found at {update_script}, skipping tokenizer update")
                # 直接convert_hf_to_gguf.pyに追加する方法にフォールバック
                self._add_so8t_tokenizer_directly(tokenizer, model_name)
                return
            
            # convert_hf_to_gguf_update.pyにSO8Tモデルを追加
            self._add_so8t_to_update_script(update_script, model_dir, model_name)
            
            # convert_hf_to_gguf.pyに直接追加
            self._add_so8t_tokenizer_directly(tokenizer, model_name)
                
        except Exception as e:
            logger.warning(f"Failed to update SO8T tokenizer info: {e}")
            logger.exception(e)
            # エラーが発生しても処理は続行
    
    def _add_so8t_to_update_script(self, update_script: Path, model_dir: Path, model_name: str):
        """
        convert_hf_to_gguf_update.pyのmodelsリストにSO8Tモデルを追加
        
        Args:
            update_script: convert_hf_to_gguf_update.pyのパス
            model_dir: モデルディレクトリパス
            model_name: SO8Tモデル名
        """
        try:
            update_script_content = update_script.read_text(encoding='utf-8')
            
            # modelsリストの最後にSO8Tモデルを追加
            models_pattern = r'(    {"name": "granite-docling",  "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/ibm-granite/granite-docling-258M", },\n])'
            so8t_entry = f'    {{"name": "{model_name}", "tokt": TOKENIZER_TYPE.BPE, "repo": "{model_dir}", }},\n]'
            
            if re.search(models_pattern, update_script_content):
                update_script_content = re.sub(
                    models_pattern,
                    r'\1'[:-2] + f',\n    {{"name": "{model_name}", "tokt": TOKENIZER_TYPE.BPE, "repo": "{model_dir}", }}\n]',
                    update_script_content
                )
                update_script.write_text(update_script_content, encoding='utf-8')
                logger.info(f"Added {model_name} to convert_hf_to_gguf_update.py models list")
            else:
                logger.warning("Could not find models list end marker in convert_hf_to_gguf_update.py")
        except Exception as e:
            logger.warning(f"Failed to add SO8T to update script: {e}")
    
    def _add_so8t_tokenizer_directly(self, tokenizer, model_name: str):
        """
        convert_hf_to_gguf.pyに直接SO8Tトークナイザー情報を追加
        
        Args:
            tokenizer: トークナイザーオブジェクト
            model_name: SO8Tモデル名
        """
        try:
            from hashlib import sha256
            
            # SO8Tモデルのトークナイザーハッシュを計算
            CHK_TXT = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'
            
            chktok = tokenizer.encode(CHK_TXT)
            chkhsh = sha256(str(chktok).encode()).hexdigest()
            
            logger.info(f"SO8T tokenizer hash: {chkhsh}")
            
            # convert_hf_to_gguf.pyを読み込み
            convert_script = PROJECT_ROOT / "external" / "llama.cpp-master" / "convert_hf_to_gguf.py"
            if not convert_script.exists():
                logger.warning(f"convert_hf_to_gguf.py not found at {convert_script}, skipping tokenizer update")
                return
            
            # convert_hf_to_gguf.pyにSO8Tトークナイザー情報を追加
            convert_py_content = convert_script.read_text(encoding='utf-8')
            
            # SO8Tトークナイザーハッシュが既に存在するかチェック
            so8t_pattern = f'if chkhsh == "{chkhsh}":'
            if so8t_pattern in convert_py_content:
                logger.info("SO8T tokenizer hash already exists in convert_hf_to_gguf.py")
                return
            
            # get_vocab_base_pre関数内にSO8Tトークナイザー情報を追加
            # Marker: End get_vocab_base_preの前に追加
            marker_pattern = r'( +# Marker: End get_vocab_base_pre)'
            so8t_entry = f"""        if chkhsh == "{chkhsh}":
            # ref: SO8T Model (Borea-Phi-3.5-mini-Instruct-Common with SO8T rotation baking)
            res = "{model_name}"
"""
            
            if re.search(marker_pattern, convert_py_content):
                convert_py_content = re.sub(
                    marker_pattern,
                    so8t_entry + r'\1',
                    convert_py_content
                )
                
                # ファイルに書き込み
                convert_script.write_text(convert_py_content, encoding='utf-8')
                logger.info(f"Added SO8T tokenizer hash to convert_hf_to_gguf.py")
            else:
                logger.warning("Could not find marker in convert_hf_to_gguf.py, skipping tokenizer update")
        except Exception as e:
            logger.warning(f"Failed to add SO8T tokenizer directly: {e}")
            logger.exception(e)
    
    def step1_burnin_and_quantize(self) -> Path:
        """
        ステップ1: 焼きこみ（SO8T Rotation Baking）と量子化
        
        Returns:
            quantized_model_path: 量子化済みモデルのパス
        """
        logger.info("="*80)
        logger.info("STEP 1: Burn-in (SO8T Rotation Baking) + Quantization")
        logger.info("="*80)
        
        try:
            # SO8T焼き込みパイプラインを使用
            # インポートパスを設定
            import sys
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            
            # 相対インポートを試す
            try:
                from scripts.so8t_burnin_pipeline import SO8TBurnInPipeline
            except ImportError:
                # 絶対パスでインポート
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "so8t_burnin_pipeline",
                    PROJECT_ROOT / "scripts" / "so8t_burnin_pipeline.py"
                )
                so8t_burnin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(so8t_burnin_module)
                SO8TBurnInPipeline = so8t_burnin_module.SO8TBurnInPipeline
            
            logger.info("Initializing SO8T burn-in pipeline...")
            burnin_pipeline = SO8TBurnInPipeline(
                hf_model_path=self.base_model_path,
                output_dir=str(self.quantized_dir / "burned"),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # モデル読み込みとSO8T統合
            logger.info("Loading HF model and integrating SO8T rotation gates...")
            burnin_pipeline.load_hf_model()
            
            # 焼き込み実行
            logger.info("Baking SO8T rotation gates into weights...")
            burnin_pipeline.bake_rotation_right_multiply()
            
            # ディスク容量不足のため、焼き込み済みモデルの保存をスキップ
            # 直接量子化してから保存する
            logger.info("Skipping intermediate model save due to disk space constraints...")
            logger.info("Applying quantization directly to save space...")
            
            # 量子化処理（メモリ内で実行）
            logger.info("Applying 8bit quantization...")
            from utils.so8t_quantization import SO8TQuantizer
            
            # 量子化器の作成
            quantizer = SO8TQuantizer(
                model=burnin_pipeline.model,
                quantization_type="8bit",
                calibration_samples=100,
                device=burnin_pipeline.device
            )
            
            # キャリブレーションデータ生成（簡易版）
            calibration_data = []
            for _ in range(100):
                data = torch.randn(1, 16, burnin_pipeline.model.config.hidden_size)
                calibration_data.append(data)
            
            quantizer.calibrate(calibration_data)
            
            # 量子化実行
            quantized_model = quantizer.quantize_model()
            
            # 量子化済みモデル保存（最小限のファイルのみ）
            quantized_model_dir = self.quantized_dir / "final_model"
            quantized_model_dir.mkdir(parents=True, exist_ok=True)
            
            # メモリから直接保存（中間ファイルを避ける）
            logger.info("Saving quantized model (minimal files)...")
            
            # トークナイザーのみ先に保存（軽量）
            burnin_pipeline.tokenizer.save_pretrained(str(quantized_model_dir))
            
            # 量子化済みモデルの重みを保存（8bitなのでサイズが小さい）
            try:
                # 量子化済みモデルを保存（可能な限り軽量に）
                quantized_model.save_pretrained(
                    str(quantized_model_dir),
                    safe_serialization=True,
                    max_shard_size="2GB"  # シャードサイズを制限
                )
            except Exception as e:
                logger.warning(f"Full model save failed: {e}")
                logger.info("Trying to save only state dict...")
                # フォールバック: ステートディクショナリのみ保存
                torch.save(quantized_model.state_dict(), quantized_model_dir / "pytorch_model.bin")
            
            # convert_hf_to_gguf_update.pyを使用してSO8Tトークナイザー情報を更新
            # （焼き込み済みモデルディレクトリの代わりに量子化済みモデルディレクトリを使用）
            logger.info("Updating tokenizer information using convert_hf_to_gguf_update.py...")
            self._update_so8t_tokenizer_info(quantized_model_dir, burnin_pipeline.tokenizer, model_name="so8t-borea-phi35")
            
            # 焼き込み済みモデルディレクトリの参照を量子化済みモデルディレクトリに変更
            baked_model_dir = quantized_model_dir
            
            logger.info(f"[OK] Step 1 completed. Quantized model saved to {quantized_model_dir}")
            return quantized_model_dir
            
        except Exception as e:
            logger.error(f"[ERROR] Step 1 failed: {e}")
            logger.exception(e)
            raise
    
    def step2_post_training(self, input_model_path: Path) -> Path:
        """
        ステップ2: 事後学習（言語モデル継続学習）
        
        Args:
            input_model_path: 入力モデルパス（量子化済みモデル）
        
        Returns:
            post_trained_model_path: 事後学習済みモデルのパス
        """
        logger.info("="*80)
        logger.info("STEP 2: Post-training (Continued Pre-training)")
        logger.info("="*80)
        
        try:
            # 既存のファインチューニングスクリプトを参考に実行
            from scripts.finetune_borea_japanese import BoreaJapaneseFinetuner
            
            # 設定ファイルのパスを取得
            config_path = self.config.get('post_training', {}).get('config_path', 'configs/finetune_borea_japanese.yaml')
            
            logger.info(f"Initializing post-training with config: {config_path}")
            
            # 設定を更新（ベースモデルを量子化済みモデルに変更）
            post_training_config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
            post_training_config['model']['base_model'] = str(input_model_path)
            post_training_config['training']['output_dir'] = str(self.post_trained_dir)
            
            # 一時設定ファイルを作成
            temp_config_path = self.output_base_dir / "post_training_config.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(post_training_config, f, allow_unicode=True)
            
            # ファインチューナーを初期化
            finetuner = BoreaJapaneseFinetuner(
                config_path=str(temp_config_path),
                auto_resume=False
            )
            
            # 学習実行
            logger.info("Starting post-training...")
            finetuner.train()
            
            # 最終モデルパス
            final_model_dir = self.post_trained_dir / "final_model"
            
            logger.info(f"[OK] Step 2 completed. Post-trained model saved to {final_model_dir}")
            return final_model_dir
            
        except Exception as e:
            logger.error(f"[ERROR] Step 2 failed: {e}")
            logger.exception(e)
            raise
    
    def step3_fine_tuning(self, input_model_path: Path) -> Path:
        """
        ステップ3: ファインチューニング（四値分類タスク特化）
        
        Args:
            input_model_path: 入力モデルパス（事後学習済みモデル）
        
        Returns:
            fine_tuned_model_path: ファインチューニング済みモデルのパス
        """
        logger.info("="*80)
        logger.info("STEP 3: Fine-tuning (Four-class Classification)")
        logger.info("="*80)
        
        try:
            from scripts.train_four_class_classifier import FourClassTrainer
            
            # 設定ファイルのパスを取得
            config_path = self.config.get('fine_tuning', {}).get('config_path', 'configs/train_four_class.yaml')
            
            logger.info(f"Initializing fine-tuning with config: {config_path}")
            
            # 設定を更新
            fine_tuning_config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
            fine_tuning_config['model']['base_model'] = str(input_model_path)
            fine_tuning_config['training']['output_dir'] = str(self.fine_tuned_dir)
            
            # 一時設定ファイルを作成
            temp_config_path = self.output_base_dir / "fine_tuning_config.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(fine_tuning_config, f, allow_unicode=True)
            
            # トレーナーを初期化
            trainer = FourClassTrainer(config_path=str(temp_config_path))
            
            # 学習実行
            logger.info("Starting fine-tuning...")
            trainer.train()
            
            # 最終モデルパス
            final_model_dir = self.fine_tuned_dir / "final_model"
            
            logger.info(f"[OK] Step 3 completed. Fine-tuned model saved to {final_model_dir}")
            return final_model_dir
            
        except Exception as e:
            logger.error(f"[ERROR] Step 3 failed: {e}")
            logger.exception(e)
            raise
    
    def step4_temperature_calibration(self, input_model_path: Path) -> Path:
        """
        ステップ4: 温度較正
        
        Args:
            input_model_path: 入力モデルパス（ファインチューニング済みモデル）
        
        Returns:
            calibrated_model_path: 温度較正済みモデルのパス
        """
        logger.info("="*80)
        logger.info("STEP 4: Temperature Calibration")
        logger.info("="*80)
        
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))
            from inference.temperature_calibration import TemperatureCalibrator
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from torch.utils.data import DataLoader
            
            # モデル読み込み
            logger.info(f"Loading model from {input_model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                str(input_model_path),
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(input_model_path),
                trust_remote_code=True
            )
            
            # 検証データ読み込み
            val_data_path = Path(self.config.get('calibration', {}).get('val_data', 'data/splits/val.jsonl'))
            logger.info(f"Loading validation data from {val_data_path}...")
            
            # 簡易データローダー作成
            val_texts = []
            with open(val_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        text = sample.get("text", "")
                        if text:
                            val_texts.append(text)
                            if len(val_texts) >= 100:  # 最大100サンプル
                                break
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(val_texts)} validation samples")
            
            # 温度較正器の作成
            calibrator = TemperatureCalibrator(
                model=model,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # データローダー作成（簡易版）
            from torch.utils.data import Dataset
            
            class SimpleDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length=512):
                    self.texts = texts
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = self.texts[idx]
                    encoded = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    return {
                        'input_ids': encoded['input_ids'].squeeze(),
                        'labels': encoded['input_ids'].squeeze()  # 簡易版：入力と同じ
                    }
            
            val_dataset = SimpleDataset(val_texts, tokenizer)
            val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
            
            # ロジットとラベルを収集
            logger.info("Collecting logits and labels...")
            logits, labels = calibrator.collect_logits_and_labels(val_dataloader)
            
            # 温度較正実行
            logger.info("Calibrating temperature...")
            optimal_temperature = calibrator.grid_search_temperature(logits, labels)
            
            logger.info(f"Optimal temperature: {optimal_temperature:.4f}")
            
            # 較正済みモデル保存
            calibrated_model_dir = self.calibrated_dir / "final_model"
            calibrated_model_dir.mkdir(parents=True, exist_ok=True)
            
            # モデルとトークナイザーを保存
            model.save_pretrained(str(calibrated_model_dir))
            tokenizer.save_pretrained(str(calibrated_model_dir))
            
            # 温度パラメータを保存
            calibration_info = {
                "optimal_temperature": float(optimal_temperature),
                "calibration_date": datetime.now().isoformat(),
                "validation_samples": len(val_texts)
            }
            with open(calibrated_model_dir / "calibration_info.json", 'w', encoding='utf-8') as f:
                json.dump(calibration_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[OK] Step 4 completed. Calibrated model saved to {calibrated_model_dir}")
            return calibrated_model_dir
            
        except Exception as e:
            logger.error(f"[ERROR] Step 4 failed: {e}")
            logger.exception(e)
            raise
    
    def run_pipeline(self, skip_steps: Optional[list] = None):
        """
        パイプライン全体を実行
        
        Args:
            skip_steps: スキップするステップのリスト（例: ['step1', 'step2']）
        """
        if skip_steps is None:
            skip_steps = []
        
        logger.info("="*80)
        logger.info("Starting Model B Pipeline")
        logger.info("="*80)
        
        current_model_path = Path(self.base_model_path)
        
        try:
            # Step 1: 焼きこみ + 量子化
            if 'step1' not in skip_steps:
                current_model_path = self.step1_burnin_and_quantize()
            else:
                logger.info("[SKIP] Step 1: Burn-in + Quantization")
                current_model_path = self.quantized_dir / "final_model"
            
            # Step 2: 事後学習
            if 'step2' not in skip_steps:
                current_model_path = self.step2_post_training(current_model_path)
            else:
                logger.info("[SKIP] Step 2: Post-training")
                current_model_path = self.post_trained_dir / "final_model"
            
            # Step 3: ファインチューニング
            if 'step3' not in skip_steps:
                current_model_path = self.step3_fine_tuning(current_model_path)
            else:
                logger.info("[SKIP] Step 3: Fine-tuning")
                current_model_path = self.fine_tuned_dir / "final_model"
            
            # Step 4: 温度較正
            if 'step4' not in skip_steps:
                current_model_path = self.step4_temperature_calibration(current_model_path)
            else:
                logger.info("[SKIP] Step 4: Temperature Calibration")
                current_model_path = self.calibrated_dir / "final_model"
            
            logger.info("="*80)
            logger.info("[SUCCESS] Model B Pipeline Completed!")
            logger.info(f"Final model path: {current_model_path}")
            logger.info("="*80)
            
            return current_model_path
            
        except Exception as e:
            logger.error("="*80)
            logger.error(f"[ERROR] Pipeline failed: {e}")
            logger.error("="*80)
            logger.exception(e)
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Model B Creation Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ab_test_borea_phi35.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        nargs='+',
        help="Steps to skip (e.g., --skip-steps step1 step2)"
    )
    
    args = parser.parse_args()
    
    # パイプライン初期化
    pipeline = ModelBPipeline(config_path=args.config)
    
    # パイプライン実行
    skip_steps = args.skip_steps if args.skip_steps else []
    final_model_path = pipeline.run_pipeline(skip_steps=skip_steps)
    
    logger.info(f"Model B created successfully at: {final_model_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

