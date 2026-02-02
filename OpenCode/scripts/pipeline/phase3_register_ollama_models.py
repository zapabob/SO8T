#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Ollamaモデル登録

モデルA/Bをollamaに登録し、Modelfileを作成します。
最適化された温度・top_pパラメータを反映します。

Usage:
    python scripts/pipelines/phase3_register_ollama_models.py --config configs/complete_automated_ab_pipeline.yaml
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
from typing import Dict, Any, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase3_register_ollama_models.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase3OllamaRegistrar:
    """Phase 3: Ollamaモデル登録クラス"""
    
    def __init__(self, config: Dict[str, Any], phase1_result: Optional[Dict] = None, phase2_result: Optional[Dict] = None):
        """
        Args:
            config: 設定辞書
            phase1_result: Phase 1の結果（モデルAのパス等）
            phase2_result: Phase 2の結果（モデルBのパス、最適化結果等）
        """
        self.config = config
        self.ollama_config = config.get('ollama', {})
        
        # モデル名設定
        self.model_a_name = self.ollama_config.get('model_a_name', 'borea-phi35-mini-q8_0')
        self.model_b_name = self.ollama_config.get('model_b_name', 'so8t-borea-phi35-mini-q8_0')
        
        # Phase 1/2の結果からパスを取得
        if phase1_result:
            self.model_a_path = Path(phase1_result.get('q8_0_path', ''))
        else:
            model_a_config = config.get('model_a', {})
            output_dir = Path(model_a_config.get('output_dir', 'D:/webdataset/gguf_models/borea_phi35_mini_q8_0'))
            self.model_a_path = output_dir / "borea_phi35_mini_q8_0.gguf"
        
        if phase2_result:
            self.model_b_path = Path(phase2_result.get('q8_0_path', ''))
            self.optimization_results = phase2_result.get('optimization_results', {})
        else:
            model_b_config = config.get('model_b', {})
            output_dir = Path(model_b_config.get('output_dir', 'D:/webdataset/gguf_models/so8t_borea_phi35_mini_q8_0'))
            self.model_b_path = output_dir / "so8t_borea_phi35_mini_q8_0.gguf"
            self.optimization_results = {}
        
        # Modelfile出力ディレクトリ
        self.modelfiles_dir = PROJECT_ROOT / "modelfiles"
        self.modelfiles_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイント設定
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', 'D:/webdataset/checkpoints/complete_ab_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Phase 3: Ollama Model Registrar Initialized")
        logger.info("="*80)
        logger.info(f"Model A name: {self.model_a_name}")
        logger.info(f"Model A path: {self.model_a_path}")
        logger.info(f"Model B name: {self.model_b_name}")
        logger.info(f"Model B path: {self.model_b_path}")
        logger.info(f"Session ID: {self.session_id}")
    
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
            'phase': 'phase3',
            'model_a_name': self.model_a_name,
            'model_b_name': self.model_b_name,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"{self.session_id}_phase3_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def create_modelfile_a(self) -> Path:
        """モデルA用Modelfile作成"""
        logger.info("Creating Modelfile for Model A...")
        
        modelfile_path = self.modelfiles_dir / f"Modelfile-{self.model_a_name.replace(':', '-')}"
        
        # 最適化パラメータ（デフォルト値）
        temperature = 1.0
        top_p = 0.9
        
        modelfile_content = f"""FROM {self.model_a_path}

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM \"\"\"You are Borea-Phi-3.5-mini-Instruct-Jp, a Japanese instruction-following language model based on Phi-3.5-mini-Instruct. You provide accurate, helpful, and safe responses in Japanese.\"\"\"
"""
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"[OK] Modelfile created: {modelfile_path}")
        return modelfile_path
    
    def create_modelfile_b(self) -> Path:
        """モデルB用Modelfile作成（最適化パラメータ反映）"""
        logger.info("Creating Modelfile for Model B...")
        
        modelfile_path = self.modelfiles_dir / f"Modelfile-{self.model_b_name.replace(':', '-')}"
        
        # ベイズ最適化結果からパラメータを取得
        best_params = self.optimization_results.get('best_params', {})
        temperature = best_params.get('temperature', 0.8)
        top_p = 0.9  # デフォルト値
        
        modelfile_content = f"""FROM {self.model_b_path}

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM \"\"\"You are SO8T-Borea-Phi-3.5-mini-Instruct-Jp, a Japanese instruction-following language model with SO(8) Transformer architecture and quadruple thinking (Task/Safety/Policy/Final). You provide accurate, helpful, and safe responses in Japanese with advanced reasoning capabilities.

## Core Architecture

The SO8T model leverages the SO(8) group structure for advanced reasoning:

1. **Task Reasoning**: Analyzes the task and determines the approach
2. **Safety Reasoning**: Evaluates safety and ethical considerations
3. **Policy Reasoning**: Applies domain-specific policies and constraints
4. **Final Answer**: Provides the final response in Japanese

## Features

- Advanced reasoning with quadruple thinking
- Safety-aware response generation
- Domain-specific policy application
- High-quality Japanese language understanding
\"\"\"
"""
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"[OK] Modelfile created: {modelfile_path}")
        logger.info(f"Optimized temperature: {temperature}")
        return modelfile_path
    
    def register_model_a(self, modelfile_path: Path) -> bool:
        """モデルAをollamaに登録"""
        logger.info("="*80)
        logger.info("Registering Model A to Ollama")
        logger.info("="*80)
        
        if not self.model_a_path.exists():
            raise FileNotFoundError(f"Model A GGUF file not found: {self.model_a_path}")
        
        if not modelfile_path.exists():
            raise FileNotFoundError(f"Modelfile not found: {modelfile_path}")
        
        try:
            # 既存のモデルを削除（存在する場合）
            logger.info(f"Removing existing model (if exists): {self.model_a_name}")
            subprocess.run(
                ["ollama", "rm", self.model_a_name],
                capture_output=True,
                text=True,
                check=False
            )
            
            # 新しいモデルを作成
            logger.info(f"Creating model: {self.model_a_name}")
            cmd = [
                "ollama", "create",
                self.model_a_name,
                "-f", str(modelfile_path)
            ]
            
            logger.info(f"[REGISTER] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10分タイムアウト
            )
            
            logger.info(f"[OK] Model A registered successfully: {self.model_a_name}")
            logger.info(f"stdout: {result.stdout[-500:]}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] Model registration timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Model registration failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
    
    def register_model_b(self, modelfile_path: Path) -> bool:
        """モデルBをollamaに登録"""
        logger.info("="*80)
        logger.info("Registering Model B to Ollama")
        logger.info("="*80)
        
        if not self.model_b_path.exists():
            raise FileNotFoundError(f"Model B GGUF file not found: {self.model_b_path}")
        
        if not modelfile_path.exists():
            raise FileNotFoundError(f"Modelfile not found: {modelfile_path}")
        
        try:
            # 既存のモデルを削除（存在する場合）
            logger.info(f"Removing existing model (if exists): {self.model_b_name}")
            subprocess.run(
                ["ollama", "rm", self.model_b_name],
                capture_output=True,
                text=True,
                check=False
            )
            
            # 新しいモデルを作成
            logger.info(f"Creating model: {self.model_b_name}")
            cmd = [
                "ollama", "create",
                self.model_b_name,
                "-f", str(modelfile_path)
            ]
            
            logger.info(f"[REGISTER] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10分タイムアウト
            )
            
            logger.info(f"[OK] Model B registered successfully: {self.model_b_name}")
            logger.info(f"stdout: {result.stdout[-500:]}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] Model registration timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Model registration failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
    
    def verify_registration(self) -> Dict[str, bool]:
        """モデル登録の確認"""
        logger.info("Verifying model registration...")
        
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            
            registered_models = {}
            registered_models['model_a'] = self.model_a_name in result.stdout
            registered_models['model_b'] = self.model_b_name in result.stdout
            
            logger.info(f"Model A registered: {registered_models['model_a']}")
            logger.info(f"Model B registered: {registered_models['model_b']}")
            
            if registered_models['model_a'] and registered_models['model_b']:
                logger.info("[OK] Both models registered successfully")
            else:
                logger.warning("[WARNING] Some models are not registered")
            
            return registered_models
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Failed to verify registration: {e}")
            return {'model_a': False, 'model_b': False}
    
    def run(self, phase1_result: Optional[Dict] = None, phase2_result: Optional[Dict] = None) -> Dict[str, Any]:
        """Phase 3実行"""
        logger.info("="*80)
        logger.info("Starting Phase 3: Ollama Model Registration")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Phase 1/2の結果を更新
            if phase1_result:
                self.model_a_path = Path(phase1_result.get('q8_0_path', self.model_a_path))
            if phase2_result:
                self.model_b_path = Path(phase2_result.get('q8_0_path', self.model_b_path))
                self.optimization_results = phase2_result.get('optimization_results', {})
            
            # Modelfile作成
            modelfile_a_path = self.create_modelfile_a()
            modelfile_b_path = self.create_modelfile_b()
            self._save_checkpoint()
            
            # モデル登録
            model_a_registered = self.register_model_a(modelfile_a_path)
            self._save_checkpoint()
            
            model_b_registered = self.register_model_b(modelfile_b_path)
            self._save_checkpoint()
            
            # 登録確認
            verification = self.verify_registration()
            
            duration = time.time() - start_time
            
            result = {
                'status': 'completed' if (model_a_registered and model_b_registered) else 'partial',
                'duration': duration,
                'model_a_name': self.model_a_name,
                'model_a_registered': model_a_registered,
                'model_b_name': self.model_b_name,
                'model_b_registered': model_b_registered,
                'verification': verification,
                'modelfile_a_path': str(modelfile_a_path),
                'modelfile_b_path': str(modelfile_b_path),
                'session_id': self.session_id
            }
            
            logger.info("="*80)
            logger.info("[SUCCESS] Phase 3 completed!")
            logger.info("="*80)
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Model A: {self.model_a_name} ({'registered' if model_a_registered else 'failed'})")
            logger.info(f"Model B: {self.model_b_name} ({'registered' if model_b_registered else 'failed'})")
            
            # 音声通知
            self._play_audio_notification()
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error("="*80)
            logger.error(f"[ERROR] Phase 3 failed: {e}")
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
        description="Phase 3: Register Models to Ollama"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/complete_automated_ab_pipeline.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--phase1-result",
        type=str,
        default=None,
        help="Phase 1 result JSON file path"
    )
    parser.add_argument(
        "--phase2-result",
        type=str,
        default=None,
        help="Phase 2 result JSON file path"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Using default configuration...")
        config = {
            'ollama': {
                'model_a_name': 'borea-phi35-mini-q8_0',
                'model_b_name': 'so8t-borea-phi35-mini-q8_0'
            },
            'checkpoint': {
                'save_dir': 'D:/webdataset/checkpoints/complete_ab_pipeline'
            }
        }
    else:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # Phase 1/2の結果を読み込み
    phase1_result = None
    phase2_result = None
    
    if args.phase1_result:
        with open(args.phase1_result, 'r', encoding='utf-8') as f:
            phase1_result = json.load(f)
    
    if args.phase2_result:
        with open(args.phase2_result, 'r', encoding='utf-8') as f:
            phase2_result = json.load(f)
    
    # Phase 3実行
    registrar = Phase3OllamaRegistrar(config, phase1_result, phase2_result)
    
    try:
        result = registrar.run(phase1_result, phase2_result)
        logger.info("Phase 3 completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("[WARNING] Phase 3 interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"[FAILED] Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())



















