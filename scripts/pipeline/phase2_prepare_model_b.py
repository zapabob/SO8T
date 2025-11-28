#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: モデルB準備（SO(8)T再学習）

Webスクレイピングデータのクレンジング、四重推論形式への変換、
SO(8)T再学習（QLoRA）、ベイズ最適化、GGUF変換を実行します。

Usage:
    python scripts/pipelines/phase2_prepare_model_b.py --config configs/complete_automated_ab_pipeline.yaml
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase2_prepare_model_b.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase2ModelBPreparer:
    """Phase 2: モデルB準備クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.model_b_config = config.get('model_b', {})
        
        # パス設定
        self.base_model = self.model_b_config.get('base_model', 'models/Borea-Phi-3.5-mini-Instruct-Jp')
        self.training_config_path = Path(self.model_b_config.get('training_config', 'configs/train_so8t_borea_qlora.yaml'))
        self.web_scraping_data = Path(self.model_b_config.get('web_scraping_data', 'data/web_scraping_cleaned.jsonl'))
        self.output_dir = Path(self.model_b_config.get('output_dir', 'D:/webdataset/gguf_models/so8t_borea_phi35_mini_q8_0'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイント設定
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', 'D:/webdataset/checkpoints/complete_ab_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # データ準備設定
        self.quadruple_thinking = self.model_b_config.get('quadruple_thinking', True)
        self.cleaned_data_path = Path("data/web_scraping_cleaned_quadruple.jsonl")
        
        # ベイズ最適化設定
        self.bayesian_config = self.model_b_config.get('bayesian_optimization', {})
        
        # ツールパス
        self.convert_script = PROJECT_ROOT / "external" / "llama.cpp-master" / "convert_hf_to_gguf.py"
        self.quantize_tool = self._find_quantize_tool()
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Phase 2: Model B Preparer Initialized")
        logger.info("="*80)
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Training config: {self.training_config_path}")
        logger.info(f"Web scraping data: {self.web_scraping_data}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Session ID: {self.session_id}")
    
    def _find_quantize_tool(self) -> Path:
        """llama-quantizeツールを検索"""
        possible_paths = [
            PROJECT_ROOT / "external" / "llama.cpp-master" / "build" / "bin" / "llama-quantize.exe",
            PROJECT_ROOT / "external" / "llama.cpp-master" / "build" / "bin" / "llama-quantize",
            PROJECT_ROOT / "external" / "llama.cpp-master" / "llama-quantize.exe",
            PROJECT_ROOT / "external" / "llama.cpp-master" / "llama-quantize",
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found quantize tool: {path}")
                return path
        
        raise FileNotFoundError(
            f"llama-quantize tool not found. Checked paths:\n" +
            "\n".join([f"  - {p}" for p in possible_paths])
        )
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_data = {
            'session_id': self.session_id,
            'phase': 'phase2',
            'base_model': self.base_model,
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"{self.session_id}_phase2_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def prepare_data(self) -> Path:
        """Step 2.1: データ準備（クレンジング + 四重推論形式変換）"""
        logger.info("="*80)
        logger.info("Step 2.1: Data Preparation (Cleaning + Quadruple Thinking Conversion)")
        logger.info("="*80)
        
        # 既に存在する場合はスキップ
        if self.cleaned_data_path.exists():
            logger.info(f"Cleaned data already exists: {self.cleaned_data_path}")
            logger.info("Skipping data preparation...")
            return self.cleaned_data_path
        
        # Webスクレイピングデータのクレンジング
        if not self.web_scraping_data.exists():
            logger.warning(f"Web scraping data not found: {self.web_scraping_data}")
            logger.info("Creating empty dataset...")
            self.cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
            # 空のデータセットを作成
            with open(self.cleaned_data_path, 'w', encoding='utf-8') as f:
                pass
            return self.cleaned_data_path
        
        # 四重推論形式への変換
        if self.quadruple_thinking:
            logger.info("Converting to quadruple thinking format...")
            convert_script = PROJECT_ROOT / "scripts" / "data" / "convert_to_quadruple_json.py"
            
            if convert_script.exists():
                cmd = [
                    sys.executable,
                    str(convert_script),
                    "--input", str(self.web_scraping_data),
                    "--output", str(self.cleaned_data_path),
                    "--instruction", "以下の文書の要点を要約してください。"
                ]
                
                logger.info(f"[CONVERT] Running: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    logger.info(f"[OK] Quadruple thinking conversion completed")
                    logger.info(f"Output: {self.cleaned_data_path}")
                    return self.cleaned_data_path
                except subprocess.CalledProcessError as e:
                    logger.error(f"[ERROR] Quadruple thinking conversion failed: {e}")
                    logger.error(f"stderr: {e.stderr}")
                    raise
            else:
                logger.warning(f"Convert script not found: {convert_script}")
                logger.info("Copying original data...")
                import shutil
                shutil.copy2(self.web_scraping_data, self.cleaned_data_path)
                return self.cleaned_data_path
        else:
            # 四重推論形式を使用しない場合はそのままコピー
            import shutil
            shutil.copy2(self.web_scraping_data, self.cleaned_data_path)
            return self.cleaned_data_path
    
    def train_model(self) -> Path:
        """Step 2.2: SO(8)T再学習（QLoRA）"""
        logger.info("="*80)
        logger.info("Step 2.2: SO(8)T Training (QLoRA)")
        logger.info("="*80)
        
        # 学習スクリプト実行
        training_script = PROJECT_ROOT / "scripts" / "training" / "train_so8t_with_bayesian.py"
        
        if not training_script.exists():
            raise FileNotFoundError(f"Training script not found: {training_script}")
        
        cmd = [
            sys.executable,
            str(training_script),
            "--config", str(self.training_config_path)
        ]
        
        logger.info(f"[TRAIN] Running: {' '.join(cmd)}")
        
        try:
            # 学習実行（長時間実行のため、プロセスを起動して監視）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                bufsize=1,
                universal_newlines=True
            )
            
            # 進捗監視
            for line in iter(process.stdout.readline, ''):
                if line:
                    logger.info(f"  {line.strip()}")
                    # チェックポイント保存のタイミングを検出
                    if "checkpoint" in line.lower() or "saved" in line.lower():
                        self._save_checkpoint()
            
            process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"Training failed with return code {process.returncode}")
            
            # 学習済みモデルのパスを取得
            checkpoint_dir = Path(self.config.get('model_b', {}).get('output_dir', 'D:/webdataset/checkpoints/training/so8t_borea_phi35'))
            # 最新のチェックポイントを検索
            checkpoint_paths = list(checkpoint_dir.glob("**/checkpoint-*"))
            if checkpoint_paths:
                latest_checkpoint = max(checkpoint_paths, key=lambda p: p.stat().st_mtime)
                logger.info(f"[OK] Training completed. Latest checkpoint: {latest_checkpoint}")
                return latest_checkpoint
            else:
                # 最終モデルを検索
                final_model_paths = list(checkpoint_dir.glob("**/final_model"))
                if final_model_paths:
                    latest_final = max(final_model_paths, key=lambda p: p.stat().st_mtime)
                    logger.info(f"[OK] Training completed. Final model: {latest_final}")
                    return latest_final
                else:
                    raise FileNotFoundError(f"No checkpoint or final model found in {checkpoint_dir}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Training failed: {e}")
            raise
    
    def optimize_hyperparameters(self, trained_model_path: Path) -> Dict[str, Any]:
        """Step 2.3: ベイズ最適化"""
        logger.info("="*80)
        logger.info("Step 2.3: Bayesian Optimization")
        logger.info("="*80)
        
        if not self.bayesian_config.get('optimize_hyperparameters', True):
            logger.info("Hyperparameter optimization disabled. Skipping...")
            return {}
        
        # ベイズ最適化スクリプト実行
        bayesian_script = PROJECT_ROOT / "scripts" / "training" / "train_so8t_with_bayesian.py"
        
        if not bayesian_script.exists():
            logger.warning(f"Bayesian optimization script not found: {bayesian_script}")
            logger.info("Skipping Bayesian optimization...")
            return {}
        
        n_trials = self.bayesian_config.get('n_trials', 50)
        
        cmd = [
            sys.executable,
            str(bayesian_script),
            "--config", str(self.training_config_path),
            "--n-trials", str(n_trials),
            "--optimize-temperature", str(self.bayesian_config.get('optimize_temperature', True)).lower()
        ]
        
        logger.info(f"[OPTIMIZE] Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600*24)  # 24時間タイムアウト
            
            # 最適化結果を読み込み
            study_path = self.checkpoint_dir / f"{self.session_id}_bayesian_study.json"
            if study_path.exists():
                with open(study_path, 'r', encoding='utf-8') as f:
                    optimization_results = json.load(f)
                logger.info(f"[OK] Bayesian optimization completed")
                logger.info(f"Best parameters: {optimization_results.get('best_params', {})}")
                return optimization_results
            else:
                logger.warning("Optimization results not found. Using default parameters...")
                return {}
                
        except subprocess.TimeoutExpired:
            logger.warning("Bayesian optimization timed out. Using default parameters...")
            return {}
        except subprocess.CalledProcessError as e:
            logger.warning(f"Bayesian optimization failed: {e}. Using default parameters...")
            return {}
    
    def convert_to_gguf(self, trained_model_path: Path) -> Path:
        """Step 2.4: GGUF変換"""
        logger.info("="*80)
        logger.info("Step 2.4: GGUF Conversion")
        logger.info("="*80)
        
        if not self.convert_script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found: {self.convert_script}\n"
                f"Please ensure llama.cpp is cloned in external/llama.cpp-master/"
            )
        
        # F16変換
        output_f16 = self.output_dir / "so8t_borea_phi35_mini_f16.gguf"
        
        if output_f16.exists():
            logger.info(f"F16 GGUF already exists: {output_f16}")
        else:
            cmd = [
                sys.executable,
                str(self.convert_script),
                str(trained_model_path),
                "--outfile", str(output_f16),
                "--outtype", "f16",
                "--model-name", "SO8T-Borea-Phi-3.5-mini-Instruct-Jp"
            ]
            
            logger.info(f"[CONVERT] Running: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.convert_script.parent),
                bufsize=1,
                universal_newlines=True
            )
            
            pbar = tqdm(desc="F16 conversion", unit="tensor", dynamic_ncols=True)
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    if "tensor" in line.lower() or "layer" in line.lower():
                        pbar.update(1)
                    if "successfully" in line.lower():
                        logger.info(f"  {line.strip()}")
            
            process.wait()
            pbar.close()
            
            if not output_f16.exists() or output_f16.stat().st_size == 0:
                raise RuntimeError("F16 conversion failed")
            
            logger.info(f"[OK] F16 conversion completed: {output_f16}")
        
        # Q8_0量子化
        output_q8_0 = self.output_dir / "so8t_borea_phi35_mini_q8_0.gguf"
        
        if output_q8_0.exists():
            logger.info(f"Q8_0 GGUF already exists: {output_q8_0}")
        else:
            cmd = [
                str(self.quantize_tool),
                str(output_f16),
                str(output_q8_0),
                "Q8_0"
            ]
            
            logger.info(f"[QUANTIZE] Running: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.quantize_tool.parent),
                bufsize=1,
                universal_newlines=True
            )
            
            pbar = tqdm(desc="Q8_0 quantization", unit="tensor", dynamic_ncols=True)
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    if "tensor" in line.lower() or "%" in line:
                        pbar.update(1)
                    if "successfully" in line.lower() or "done" in line.lower():
                        logger.info(f"  {line.strip()}")
            
            process.wait()
            pbar.close()
            
            if not output_q8_0.exists() or output_q8_0.stat().st_size == 0:
                raise RuntimeError("Q8_0 quantization failed")
            
            logger.info(f"[OK] Q8_0 quantization completed: {output_q8_0}")
        
        return output_q8_0
    
    def run(self) -> Dict[str, Any]:
        """Phase 2実行"""
        logger.info("="*80)
        logger.info("Starting Phase 2: Model B Preparation")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Step 2.1: データ準備
            cleaned_data_path = self.prepare_data()
            self._save_checkpoint()
            
            # Step 2.2: SO(8)T再学習
            trained_model_path = self.train_model()
            self._save_checkpoint()
            
            # Step 2.3: ベイズ最適化
            optimization_results = self.optimize_hyperparameters(trained_model_path)
            self._save_checkpoint()
            
            # Step 2.4: GGUF変換
            q8_0_path = self.convert_to_gguf(trained_model_path)
            self._save_checkpoint()
            
            duration = time.time() - start_time
            
            result = {
                'status': 'completed',
                'duration': duration,
                'cleaned_data_path': str(cleaned_data_path),
                'trained_model_path': str(trained_model_path),
                'optimization_results': optimization_results,
                'q8_0_path': str(q8_0_path),
                'output_dir': str(self.output_dir),
                'session_id': self.session_id
            }
            
            logger.info("="*80)
            logger.info("[SUCCESS] Phase 2 completed!")
            logger.info("="*80)
            logger.info(f"Duration: {duration/3600:.2f} hours")
            logger.info(f"Q8_0 model: {q8_0_path}")
            
            # 音声通知
            self._play_audio_notification()
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error("="*80)
            logger.error(f"[ERROR] Phase 2 failed: {e}")
            logger.error("="*80)
            logger.exception(e)
            raise
    
    def _play_audio_notification(self):
        """音声通知を再生"""
        audio_file = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
        if audio_file.exists():
            try:
                ps_cmd = f"""
                if (Test-Path '{audio_file}') {{
                    Add-Type -AssemblyName System.Windows.Forms
                    $player = New-Object System.Media.SoundPlayer '{audio_file}'
                    $player.PlaySync()
                    Write-Host '[OK] Audio notification played' -ForegroundColor Green
                }}
                """
                subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    cwd=str(PROJECT_ROOT),
                    check=False
                )
            except Exception as e:
                logger.warning(f"Failed to play audio: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Phase 2: Prepare Model B (SO(8)T Retraining)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/complete_automated_ab_pipeline.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Using default configuration...")
        config = {
            'model_b': {
                'base_model': 'Borea-Phi-3.5-mini-Instruct-Jp',
                'training_config': 'configs/train_so8t_borea_qlora.yaml',
                'web_scraping_data': 'data/web_scraping_cleaned.jsonl',
                'quadruple_thinking': True,
                'bayesian_optimization': {
                    'n_trials': 50,
                    'optimize_temperature': True,
                    'optimize_hyperparameters': True
                },
                'output_dir': 'D:/webdataset/gguf_models/so8t_borea_phi35_mini_q8_0'
            },
            'checkpoint': {
                'save_dir': 'D:/webdataset/checkpoints/complete_ab_pipeline'
            }
        }
    else:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # Phase 2実行
    preparer = Phase2ModelBPreparer(config)
    
    try:
        result = preparer.run()
        logger.info("Phase 2 completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("[WARNING] Phase 2 interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"[FAILED] Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())










