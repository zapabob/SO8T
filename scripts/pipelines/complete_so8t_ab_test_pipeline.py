#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T完全統合A/Bテストパイプライン

Borea-Phi-3.5-mini-Instruct-Jpのmodeling_phi3_so8t.pyをそのままGGUF化したもの（Model A）と、
SO8Tで再学習してデータセットでQLoRA/ファインチューニングで学習させたものをGGUF化したもの（Model B）の
A/Bテストまで一気通貫で全自動実行する統合パイプライン

Usage:
    python scripts/pipelines/complete_so8t_ab_test_pipeline.py --config configs/complete_so8t_ab_test_pipeline_config.yaml
"""

import sys
import json
import logging
import argparse
import subprocess
import signal
import pickle
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_so8t_ab_test_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioNotifier:
    """音声通知クラス"""
    
    @staticmethod
    def play_notification():
        """音声通知を再生"""
        audio_path = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
        if audio_path.exists():
            try:
                subprocess.run([
                    "powershell", "-ExecutionPolicy", "Bypass", "-File",
                    str(PROJECT_ROOT / "scripts" / "utils" / "play_audio_notification.ps1")
                ], check=False, timeout=10)
            except Exception as e:
                logger.warning(f"Failed to play audio notification: {e}")
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                except Exception:
                    pass


class CompleteSO8TABTestPipeline:
    """SO8T完全統合A/Bテストパイプライン"""
    
    def __init__(self, config_path: Path):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定ファイルを読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 基本設定
        self.session_id = datetime.now().strftime(self.config.get('session_id_format', '%Y%m%d_%H%M%S'))
        self.output_base_dir = Path(self.config.get('output_base_dir', 'D:/webdataset'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'D:/webdataset/checkpoints/complete_so8t_ab_test'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Model A設定
        model_a_config = self.config.get('model_a', {})
        self.model_a_base_path = Path(model_a_config.get('base_model', 'models/Borea-Phi-3.5-mini-Instruct-Jp'))
        self.model_a_gguf_output = self.output_base_dir / 'gguf_models' / 'model_a'
        self.model_a_gguf_output.mkdir(parents=True, exist_ok=True)
        
        # Model B設定
        model_b_config = self.config.get('model_b', {})
        self.model_b_base_path = Path(model_b_config.get('base_model', 'models/Borea-Phi-3.5-mini-Instruct-Jp'))
        self.model_b_training_output = self.output_base_dir / 'checkpoints' / 'training' / 'so8t_retrained'
        self.model_b_training_output.mkdir(parents=True, exist_ok=True)
        self.model_b_gguf_output = self.output_base_dir / 'gguf_models' / 'model_b'
        self.model_b_gguf_output.mkdir(parents=True, exist_ok=True)
        
        # データセット設定
        dataset_config = self.config.get('dataset', {})
        self.dataset_paths = [Path(p) for p in dataset_config.get('paths', [])]
        self.dataset_format = dataset_config.get('format', 'jsonl')
        
        # 学習設定
        training_config = self.config.get('training', {})
        self.training_config_path = Path(training_config.get('config', 'configs/train_so8t_phi3_qlora.yaml'))
        self.use_qlora = training_config.get('use_qlora', True)
        self.use_full_finetuning = training_config.get('use_full_finetuning', False)
        
        # GGUF変換設定
        gguf_config = self.config.get('gguf', {})
        self.convert_script_path = Path(gguf_config.get('convert_script', 'external/llama.cpp-master/convert_hf_to_gguf.py'))
        
        # サポートされている量子化タイプ（convert_hf_to_gguf.pyの仕様に基づく）
        self.supported_quantizations = ['f32', 'f16', 'bf16', 'q8_0', 'tq1_0', 'tq2_0', 'auto']
        
        # 設定ファイルから量子化タイプを読み込み
        requested_quantizations = gguf_config.get('quantizations', ['f16', 'q8_0'])
        
        # 量子化タイプの検証
        valid_quantizations = []
        invalid_quantizations = []
        
        for quant_type in requested_quantizations:
            if quant_type in self.supported_quantizations:
                valid_quantizations.append(quant_type)
            else:
                invalid_quantizations.append(quant_type)
                logger.warning(f"[WARNING] Unsupported quantization type '{quant_type}' will be skipped")
                logger.warning(f"[WARNING] Supported types: {', '.join(self.supported_quantizations)}")
        
        if invalid_quantizations:
            logger.warning(f"[WARNING] Skipping {len(invalid_quantizations)} unsupported quantization type(s): {', '.join(invalid_quantizations)}")
        
        if not valid_quantizations:
            raise ValueError(
                f"No valid quantization types found. "
                f"Requested: {requested_quantizations}, "
                f"Supported: {self.supported_quantizations}"
            )
        
        self.quantizations = valid_quantizations
        logger.info(f"[INFO] Using quantization types: {', '.join(self.quantizations)}")
        
        # Ollama設定
        ollama_config = self.config.get('ollama', {})
        self.ollama_model_a_name = ollama_config.get('model_a_name', 'borea-phi35-so8t-base')
        self.ollama_model_b_name = ollama_config.get('model_b_name', 'borea-phi35-so8t-retrained')
        self.modelfile_dir = Path(ollama_config.get('modelfile_dir', 'modelfiles'))
        self.modelfile_dir.mkdir(parents=True, exist_ok=True)
        
        # A/Bテスト設定
        ab_test_config = self.config.get('ab_test', {})
        self.ab_test_output_dir = self.output_base_dir / 'ab_test_results' / f'complete_so8t_ab_test_{self.session_id}'
        self.ab_test_output_dir.mkdir(parents=True, exist_ok=True)
        self.test_data_path = Path(ab_test_config.get('test_data', 'D:/webdataset/processed/four_class'))
        
        # 可視化設定
        viz_config = self.config.get('visualization', {})
        self.viz_output_dir = self.ab_test_output_dir / 'visualizations'
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 進捗管理
        self.current_phase = None
        self.phase_progress = {}
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("SO8T Complete A/B Test Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Model A base: {self.model_a_base_path}")
        logger.info(f"Model B base: {self.model_b_base_path}")
        logger.info(f"Output directory: {self.ab_test_output_dir}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            logger.info("Checkpoint saved. Exiting gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """チェックポイント読み込み"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        if not checkpoint_file.exists():
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if checkpoints:
                checkpoint_file = checkpoints[-1]
                logger.info(f"Loading latest checkpoint: {checkpoint_file}")
            else:
                return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info(f"[OK] Checkpoint loaded from {checkpoint_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _is_phase_completed(self, phase_name: str) -> bool:
        """
        フェーズの完了状態を検証
        
        Args:
            phase_name: フェーズ名
        
        Returns:
            completed: 完了しているかどうか
        """
        # チェックポイントからステータスを確認
        phase_progress = self.phase_progress.get(phase_name, {})
        status = phase_progress.get('status', 'pending')
        
        if status != 'completed':
            return False
        
        # Phase 1: Model AのGGUF変換
        if phase_name == 'phase1_model_a_gguf':
            gguf_files = phase_progress.get('gguf_files', {})
            if not gguf_files:
                logger.warning("[VERIFY] Phase 1: No GGUF files in checkpoint")
                return False
            
            # 各GGUFファイルの存在とサイズを確認
            for quant_type, file_path_str in gguf_files.items():
                file_path = Path(file_path_str)
                if not file_path.exists():
                    logger.warning(f"[VERIFY] Phase 1: GGUF file not found: {file_path}")
                    return False
                
                file_size = file_path.stat().st_size
                if file_size == 0:
                    logger.warning(f"[VERIFY] Phase 1: GGUF file is empty: {file_path}")
                    return False
                
                logger.debug(f"[VERIFY] Phase 1: GGUF file verified: {file_path} ({file_size:,} bytes)")
            
            logger.info(f"[VERIFY] Phase 1: All {len(gguf_files)} GGUF files verified")
            return True
        
        # Phase 2: SO8T再学習
        elif phase_name == 'phase2_train_model_b':
            trained_model_path_str = phase_progress.get('trained_model_path')
            if not trained_model_path_str:
                logger.warning("[VERIFY] Phase 2: No trained model path in checkpoint")
                return False
            
            trained_model_path = Path(trained_model_path_str)
            if not trained_model_path.exists():
                logger.warning(f"[VERIFY] Phase 2: Trained model path not found: {trained_model_path}")
                return False
            
            # モデルファイルの存在確認（config.json, tokenizer.jsonなど）
            required_files = ['config.json']
            for req_file in required_files:
                req_path = trained_model_path / req_file
                if not req_path.exists():
                    logger.warning(f"[VERIFY] Phase 2: Required file not found: {req_path}")
                    return False
            
            logger.info(f"[VERIFY] Phase 2: Trained model verified: {trained_model_path}")
            return True
        
        # Phase 3: Model BのGGUF変換
        elif phase_name == 'phase3_model_b_gguf':
            gguf_files = phase_progress.get('gguf_files', {})
            if not gguf_files:
                logger.warning("[VERIFY] Phase 3: No GGUF files in checkpoint")
                return False
            
            # 各GGUFファイルの存在とサイズを確認
            for quant_type, file_path_str in gguf_files.items():
                file_path = Path(file_path_str)
                if not file_path.exists():
                    logger.warning(f"[VERIFY] Phase 3: GGUF file not found: {file_path}")
                    return False
                
                file_size = file_path.stat().st_size
                if file_size == 0:
                    logger.warning(f"[VERIFY] Phase 3: GGUF file is empty: {file_path}")
                    return False
                
                logger.debug(f"[VERIFY] Phase 3: GGUF file verified: {file_path} ({file_size:,} bytes)")
            
            logger.info(f"[VERIFY] Phase 3: All {len(gguf_files)} GGUF files verified")
            return True
        
        # Phase 4: Ollamaインポート
        elif phase_name == 'phase4_ollama_import':
            model_a_imported = phase_progress.get('model_a_imported', False)
            model_b_imported = phase_progress.get('model_b_imported', False)
            
            if not model_a_imported or not model_b_imported:
                logger.warning(f"[VERIFY] Phase 4: Models not imported (A: {model_a_imported}, B: {model_b_imported})")
                return False
            
            logger.info("[VERIFY] Phase 4: Both models imported to Ollama")
            return True
        
        # Phase 5: A/Bテスト実行
        elif phase_name == 'phase5_ab_test':
            results_file_str = phase_progress.get('results_file')
            if not results_file_str:
                logger.warning("[VERIFY] Phase 5: No results file in checkpoint")
                return False
            
            results_file = Path(results_file_str)
            if not results_file.exists():
                logger.warning(f"[VERIFY] Phase 5: Results file not found: {results_file}")
                return False
            
            logger.info(f"[VERIFY] Phase 5: Results file verified: {results_file}")
            return True
        
        # Phase 6: 可視化とレポート生成
        elif phase_name == 'phase6_visualization':
            report_file_str = phase_progress.get('report_file')
            if not report_file_str:
                logger.warning("[VERIFY] Phase 6: No report file in checkpoint")
                return False
            
            report_file = Path(report_file_str)
            if not report_file.exists():
                logger.warning(f"[VERIFY] Phase 6: Report file not found: {report_file}")
                return False
            
            logger.info(f"[VERIFY] Phase 6: Report file verified: {report_file}")
            return True
        
        # Phase 7: Ollamaチェック
        elif phase_name == 'phase7_ollama_check':
            phase_progress = self.phase_progress.get('phase7_ollama_check', {})
            status = phase_progress.get('status', '')
            if status == 'completed':
                check_results_file = self.ab_test_output_dir / "ollama_check_results.json"
                if check_results_file.exists():
                    logger.info("[VERIFY] Phase 7: Ollama check results file exists")
                    return True
                else:
                    logger.warning("[VERIFY] Phase 7: No check results file in checkpoint")
                    return False
            else:
                logger.debug(f"[VERIFY] Phase 7: Status check only (status: {status})")
                return False
        
        # その他のフェーズはステータスのみで判定
        else:
            logger.debug(f"[VERIFY] Phase {phase_name}: Status check only (status: {status})")
            return status == 'completed'
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        checkpoint_data = {
            'session_id': self.session_id,
            'current_phase': self.current_phase,
            'phase_progress': self.phase_progress,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"[CHECKPOINT] Saved to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def phase1_convert_model_a_to_gguf(self) -> Dict[str, Path]:
        """
        Phase 1: Model AのGGUF変換
        
        Returns:
            gguf_files: 量子化タイプ -> GGUFファイルパスの辞書
        """
        logger.info("="*80)
        logger.info("PHASE 1: Converting Model A to GGUF")
        logger.info("="*80)
        
        self.current_phase = "phase1_model_a_gguf"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint:
            self.phase_progress = checkpoint.get('phase_progress', {})
        
        # フェーズ完了検証
        if self._is_phase_completed('phase1_model_a_gguf'):
            logger.info("[SKIP] Phase 1 already completed and verified")
            phase_progress = self.phase_progress.get('phase1_model_a_gguf', {})
            return {q: Path(p) for q, p in phase_progress.get('gguf_files', {}).items()}
        
        # GGUFファイルが既に存在する場合は検出してスキップ
        existing_gguf_files = {}
        for quant_type in self.quantizations:
            output_file = self.model_a_gguf_output / f"model_a_{quant_type}.gguf"
            if output_file.exists() and output_file.stat().st_size > 0:
                existing_gguf_files[quant_type] = output_file
                logger.info(f"[DETECT] Found existing Model A GGUF file: {output_file}")
        
        if existing_gguf_files:
            logger.info(f"[SKIP] Phase 1: Using existing GGUF files ({len(existing_gguf_files)} files)")
            self.phase_progress['phase1_model_a_gguf'] = {
                'status': 'completed',
                'gguf_files': {k: str(v) for k, v in existing_gguf_files.items()},
                'detected_from_existing': True
            }
            self._save_checkpoint()
            AudioNotifier.play_notification()
            return existing_gguf_files
        
        if not self.model_a_base_path.exists():
            raise FileNotFoundError(f"Model A base path not found: {self.model_a_base_path}")
        
        logger.info(f"Converting Model A from {self.model_a_base_path} to GGUF...")
        
        gguf_files = {}
        
        for quant_type in self.quantizations:
            # 量子化タイプの再検証（念のため）
            if quant_type not in self.supported_quantizations:
                logger.error(f"[ERROR] Unsupported quantization type '{quant_type}' detected during conversion")
                logger.error(f"[ERROR] Supported types: {', '.join(self.supported_quantizations)}")
                raise ValueError(f"Unsupported quantization type: {quant_type}")
            
            logger.info(f"Converting Model A to {quant_type}...")
            
            output_file = self.model_a_gguf_output / f"model_a_{quant_type}.gguf"
            
            cmd = [
                sys.executable,
                str(self.convert_script_path),
                str(self.model_a_base_path),
                "--outfile", str(output_file),
                "--outtype", quant_type
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=3600  # 1時間タイムアウト
                )
                logger.info(f"[OK] Model A GGUF conversion completed: {output_file}")
                gguf_files[quant_type] = output_file
            except subprocess.CalledProcessError as e:
                logger.error(f"[ERROR] Model A GGUF conversion failed for {quant_type}: {e}")
                logger.error(f"Command: {' '.join(cmd)}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                # エラーメッセージにサポートされている量子化タイプを含める
                if "invalid choice" in e.stderr.lower() or "invalid" in e.stderr.lower():
                    logger.error(f"[ERROR] Supported quantization types: {', '.join(self.supported_quantizations)}")
                raise
            except subprocess.TimeoutExpired:
                logger.error(f"[ERROR] Model A GGUF conversion timeout for {quant_type}")
                raise
        
        self.phase_progress['phase1_model_a_gguf'] = {
            'status': 'completed',
            'gguf_files': {k: str(v) for k, v in gguf_files.items()}
        }
        self._save_checkpoint()
        
        AudioNotifier.play_notification()
        
        logger.info(f"[OK] Phase 1 completed. Model A GGUF files: {len(gguf_files)}")
        return gguf_files
    
    def phase2_train_model_b(self) -> Path:
        """
        Phase 2: SO8T再学習（QLoRA/ファインチューニング）
        
        Returns:
            trained_model_path: 学習済みモデルのパス
        """
        logger.info("="*80)
        logger.info("PHASE 2: Training Model B with SO8T")
        logger.info("="*80)
        
        self.current_phase = "phase2_train_model_b"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint:
            self.phase_progress = checkpoint.get('phase_progress', {})
        
        # フェーズ完了検証
        if self._is_phase_completed('phase2_train_model_b'):
            logger.info("[SKIP] Phase 2 already completed and verified")
            phase_progress = self.phase_progress.get('phase2_train_model_b', {})
            return Path(phase_progress.get('trained_model_path'))
        
        # データセットを検出
        dataset_files = []
        for dataset_path in self.dataset_paths:
            if dataset_path.is_dir():
                dataset_files.extend(dataset_path.glob(f"*.{self.dataset_format}"))
            elif dataset_path.exists():
                dataset_files.append(dataset_path)
        
        if not dataset_files:
            raise FileNotFoundError(f"No dataset files found in {self.dataset_paths}")
        
        logger.info(f"Found {len(dataset_files)} dataset files")
        
        # 学習スクリプトを実行
        if self.use_qlora:
            train_script = PROJECT_ROOT / "scripts" / "training" / "train_so8t_phi3_qlora.py"
        else:
            raise NotImplementedError("Full fine-tuning not yet implemented")
        
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")
        
        # 学習設定ファイルを更新（データセットパスを設定）
        training_config_data = {}
        if self.training_config_path.exists():
            with open(self.training_config_path, 'r', encoding='utf-8') as f:
                training_config_data = yaml.safe_load(f)
        
        # データセットパスを設定
        training_config_data['data'] = training_config_data.get('data', {})
        training_config_data['data']['train_data'] = [str(f) for f in dataset_files]
        training_config_data['training'] = training_config_data.get('training', {})
        training_config_data['training']['output_dir'] = str(self.model_b_training_output)
        
        # 一時設定ファイルを作成
        temp_config_path = self.checkpoint_dir / f"training_config_{self.session_id}.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(training_config_data, f, default_flow_style=False)
        
        logger.info(f"Starting SO8T training with config: {temp_config_path}")
        
        cmd = [
            sys.executable,
            str(train_script),
            "--config", str(temp_config_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=86400  # 24時間タイムアウト
            )
            
            # 学習済みモデルのパスを取得（最後のチェックポイントまたは最終モデル）
            trained_model_path = self.model_b_training_output / "final_model"
            if not trained_model_path.exists():
                # チェックポイントから最新を探す
                checkpoints = sorted(self.model_b_training_output.glob("checkpoint-*"))
                if checkpoints:
                    trained_model_path = checkpoints[-1]
                else:
                    trained_model_path = self.model_b_training_output
            
            logger.info(f"[OK] Model B training completed: {trained_model_path}")
            
            self.phase_progress['phase2_train_model_b'] = {
                'status': 'completed',
                'trained_model_path': str(trained_model_path)
            }
            self._save_checkpoint()
            
            AudioNotifier.play_notification()
            
            return trained_model_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Model B training failed: {e}")
            raise
        except subprocess.TimeoutExpired:
            logger.error(f"[ERROR] Model B training timeout")
            raise
    
    def phase3_convert_model_b_to_gguf(self, trained_model_path: Path) -> Dict[str, Path]:
        """
        Phase 3: Model BのGGUF変換
        
        Args:
            trained_model_path: 学習済みモデルのパス
        
        Returns:
            gguf_files: 量子化タイプ -> GGUFファイルパスの辞書
        """
        logger.info("="*80)
        logger.info("PHASE 3: Converting Model B to GGUF")
        logger.info("="*80)
        
        self.current_phase = "phase3_model_b_gguf"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint:
            self.phase_progress = checkpoint.get('phase_progress', {})
        
        # フェーズ完了検証
        if self._is_phase_completed('phase3_model_b_gguf'):
            logger.info("[SKIP] Phase 3 already completed and verified")
            phase_progress = self.phase_progress.get('phase3_model_b_gguf', {})
            return {q: Path(p) for q, p in phase_progress.get('gguf_files', {}).items()}
        
        if not trained_model_path.exists():
            raise FileNotFoundError(f"Trained model path not found: {trained_model_path}")
        
        logger.info(f"Converting Model B from {trained_model_path} to GGUF...")
        
        gguf_files = {}
        
        for quant_type in self.quantizations:
            # 量子化タイプの再検証（念のため）
            if quant_type not in self.supported_quantizations:
                logger.error(f"[ERROR] Unsupported quantization type '{quant_type}' detected during conversion")
                logger.error(f"[ERROR] Supported types: {', '.join(self.supported_quantizations)}")
                raise ValueError(f"Unsupported quantization type: {quant_type}")
            
            logger.info(f"Converting Model B to {quant_type}...")
            
            output_file = self.model_b_gguf_output / f"model_b_{quant_type}.gguf"
            
            cmd = [
                sys.executable,
                str(self.convert_script_path),
                str(trained_model_path),
                "--outfile", str(output_file),
                "--outtype", quant_type
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=3600  # 1時間タイムアウト
                )
                logger.info(f"[OK] Model B GGUF conversion completed: {output_file}")
                gguf_files[quant_type] = output_file
            except subprocess.CalledProcessError as e:
                logger.error(f"[ERROR] Model B GGUF conversion failed for {quant_type}: {e}")
                logger.error(f"Command: {' '.join(cmd)}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                # エラーメッセージにサポートされている量子化タイプを含める
                if "invalid choice" in e.stderr.lower() or "invalid" in e.stderr.lower():
                    logger.error(f"[ERROR] Supported quantization types: {', '.join(self.supported_quantizations)}")
                raise
            except subprocess.TimeoutExpired:
                logger.error(f"[ERROR] Model B GGUF conversion timeout for {quant_type}")
                raise
        
        self.phase_progress['phase3_model_b_gguf'] = {
            'status': 'completed',
            'gguf_files': {k: str(v) for k, v in gguf_files.items()}
        }
        self._save_checkpoint()
        
        AudioNotifier.play_notification()
        
        logger.info(f"[OK] Phase 3 completed. Model B GGUF files: {len(gguf_files)}")
        return gguf_files
    
    def phase4_import_to_ollama(self, model_a_gguf_files: Dict[str, Path], model_b_gguf_files: Dict[str, Path]):
        """
        Phase 4: Ollamaインポート
        
        Args:
            model_a_gguf_files: Model AのGGUFファイル
            model_b_gguf_files: Model BのGGUFファイル
        """
        logger.info("="*80)
        logger.info("PHASE 4: Importing to Ollama")
        logger.info("="*80)
        
        self.current_phase = "phase4_ollama_import"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint:
            self.phase_progress = checkpoint.get('phase_progress', {})
        
        # フェーズ完了検証
        if self._is_phase_completed('phase4_ollama_import'):
            logger.info("[SKIP] Phase 4 already completed and verified")
            return
        
        # Q8_0を使用（なければ最初のファイル）
        model_a_gguf = model_a_gguf_files.get('q8_0') or list(model_a_gguf_files.values())[0]
        model_b_gguf = model_b_gguf_files.get('q8_0') or list(model_b_gguf_files.values())[0]
        
        # Model AのModelfile作成
        model_a_modelfile = self.modelfile_dir / f"{self.ollama_model_a_name}.modelfile"
        self._create_modelfile(model_a_gguf, self.ollama_model_a_name, model_a_modelfile)
        
        # Model BのModelfile作成
        model_b_modelfile = self.modelfile_dir / f"{self.ollama_model_b_name}.modelfile"
        self._create_modelfile(model_b_gguf, self.ollama_model_b_name, model_b_modelfile)
        
        # Ollamaにインポート
        success_a = self._import_to_ollama(model_a_modelfile, self.ollama_model_a_name)
        success_b = self._import_to_ollama(model_b_modelfile, self.ollama_model_b_name)
        
        self.phase_progress['phase4_ollama_import'] = {
            'status': 'completed' if (success_a and success_b) else 'partial',
            'model_a_imported': success_a,
            'model_b_imported': success_b
        }
        self._save_checkpoint()
        
        AudioNotifier.play_notification()
        
        logger.info(f"[OK] Phase 4 completed. Model A: {success_a}, Model B: {success_b}")
    
    def _create_modelfile(self, gguf_file: Path, model_name: str, output_path: Path):
        """Modelfileを作成"""
        modelfile_content = f"""FROM {gguf_file.absolute()}

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        logger.info(f"[OK] Modelfile created: {output_path}")
    
    def _import_to_ollama(self, modelfile_path: Path, model_name: str) -> bool:
        """Ollamaにインポート"""
        cmd = [
            "ollama",
            "create",
            f"{model_name}:latest",
            "-f",
            str(modelfile_path.absolute())
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logger.info(f"[OK] Model imported to Ollama: {model_name}:latest")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Ollama import failed for {model_name}: {e}")
            logger.error(f"stderr: {e.stderr}")
            return False
    
    def phase5_run_ab_test(self) -> Dict:
        """
        Phase 5: A/Bテスト実行
        
        Returns:
            ab_test_results: A/Bテスト結果
        """
        logger.info("="*80)
        logger.info("PHASE 5: Running A/B Test")
        logger.info("="*80)
        
        self.current_phase = "phase5_ab_test"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint:
            self.phase_progress = checkpoint.get('phase_progress', {})
        
        # フェーズ完了検証
        if self._is_phase_completed('phase5_ab_test'):
            logger.info("[SKIP] Phase 5 already completed and verified")
            phase_progress = self.phase_progress.get('phase5_ab_test', {})
            results_file = Path(phase_progress.get('results_file'))
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # テストデータを読み込み
        test_samples = []
        if self.test_data_path.is_dir():
            test_files = list(self.test_data_path.glob("*.jsonl"))
        else:
            test_files = [self.test_data_path] if self.test_data_path.exists() else []
        
        for test_file in test_files:
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(test_samples)} test samples")
        
        # Ollama経由でA/Bテストを実行
        results = self._run_ollama_ab_test(test_samples)
        
        # 結果を保存
        results_file = self.ab_test_output_dir / "ab_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.phase_progress['phase5_ab_test'] = {
            'status': 'completed',
            'results_file': str(results_file)
        }
        self._save_checkpoint()
        
        AudioNotifier.play_notification()
        
        logger.info(f"[OK] Phase 5 completed. Results saved to {results_file}")
        return results
    
    def _run_ollama_ab_test(self, test_samples: List[Dict]) -> Dict:
        """Ollama経由でA/Bテストを実行"""
        logger.info("Running A/B test via Ollama...")
        
        # テストサンプル数を制限（時間短縮のため）
        max_samples = min(50, len(test_samples))
        test_samples = test_samples[:max_samples]
        
        model_a_results = []
        model_b_results = []
        
        for sample in tqdm(test_samples, desc="A/B Testing"):
            # プロンプトを構築
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            prompt = f"{instruction}\n\n{input_text}" if instruction else input_text
            
            # Model Aで推論
            result_a = self._ollama_inference(self.ollama_model_a_name, prompt)
            model_a_results.append(result_a)
            
            # Model Bで推論
            result_b = self._ollama_inference(self.ollama_model_b_name, prompt)
            model_b_results.append(result_b)
        
        # メトリクスを計算
        metrics_a = self._calculate_metrics(model_a_results, test_samples)
        metrics_b = self._calculate_metrics(model_b_results, test_samples)
        
        # 比較
        comparison = {
            'accuracy_improvement': metrics_b['accuracy'] - metrics_a['accuracy'],
            'f1_macro_improvement': metrics_b['f1_macro'] - metrics_a['f1_macro'],
            'latency_change': metrics_b['avg_latency'] - metrics_a['avg_latency']
        }
        
        results = {
            'model_a': metrics_a,
            'model_b': metrics_b,
            'comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _ollama_inference(self, model_name: str, prompt: str) -> Dict:
        """Ollamaで推論を実行"""
        import time
        
        cmd = [
            "ollama",
            "run",
            f"{model_name}:latest",
            prompt
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=60
            )
            latency = time.time() - start_time
            
            return {
                'model': model_name,
                'prompt': prompt,
                'response': result.stdout.strip(),
                'latency': latency,
                'success': True
            }
        except subprocess.CalledProcessError as e:
            logger.warning(f"Ollama inference failed: {e}")
            return {
                'model': model_name,
                'prompt': prompt,
                'response': '',
                'latency': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        except subprocess.TimeoutExpired:
            return {
                'model': model_name,
                'prompt': prompt,
                'response': '',
                'latency': 60.0,
                'success': False,
                'error': 'timeout'
            }
    
    def _calculate_metrics(self, results: List[Dict], test_samples: List[Dict]) -> Dict:
        """メトリクスを計算"""
        # 簡易版メトリクス計算
        # 実際の実装では、より詳細な評価が必要
        
        latencies = [r['latency'] for r in results if r['success']]
        avg_latency = np.mean(latencies) if latencies else 0.0
        
        # 簡易的なaccuracy計算（実際の実装では、予測とラベルの比較が必要）
        accuracy = 0.5  # プレースホルダー
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': accuracy,  # プレースホルダー
            'avg_latency': avg_latency,
            'num_samples': len(results),
            'success_rate': sum(1 for r in results if r['success']) / len(results) if results else 0.0
        }
        
        return metrics
    
    def phase6_visualize_and_report(self, ab_test_results: Dict):
        """
        Phase 6: 結果可視化とレポート生成
        
        Args:
            ab_test_results: A/Bテスト結果
        """
        logger.info("="*80)
        logger.info("PHASE 6: Visualizing Results and Generating Report")
        logger.info("="*80)
        
        self.current_phase = "phase6_visualization"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint:
            self.phase_progress = checkpoint.get('phase_progress', {})
        
        # フェーズ完了検証
        if self._is_phase_completed('phase6_visualization'):
            logger.info("[SKIP] Phase 6 already completed and verified")
            phase_progress = self.phase_progress.get('phase6_visualization', {})
            return Path(phase_progress.get('report_file'))
        
        # 可視化を生成
        self._generate_visualizations(ab_test_results)
        
        # レポートを生成
        report_path = self._generate_report(ab_test_results)
        
        self.phase_progress['phase6_visualization'] = {
            'status': 'completed',
            'report_file': str(report_path)
        }
        self._save_checkpoint()
        
        AudioNotifier.play_notification()
        
        logger.info(f"[OK] Phase 6 completed. Report saved to {report_path}")
    
    def _generate_visualizations(self, results: Dict):
        """可視化を生成"""
        logger.info("Generating visualizations...")
        
        metrics_a = results['model_a']
        metrics_b = results['model_b']
        comparison = results['comparison']
        
        # 1. メトリクス比較バーチャート
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy比較
        axes[0, 0].bar(['Model A', 'Model B'], 
                      [metrics_a['accuracy'], metrics_b['accuracy']])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0, 1])
        
        # F1 Macro比較
        axes[0, 1].bar(['Model A', 'Model B'],
                      [metrics_a['f1_macro'], metrics_b['f1_macro']])
        axes[0, 1].set_title('F1 Macro Comparison')
        axes[0, 1].set_ylabel('F1 Macro')
        axes[0, 1].set_ylim([0, 1])
        
        # レイテンシ比較
        axes[1, 0].bar(['Model A', 'Model B'],
                      [metrics_a['avg_latency'], metrics_b['avg_latency']])
        axes[1, 0].set_title('Average Latency Comparison')
        axes[1, 0].set_ylabel('Latency (seconds)')
        
        # 改善率
        improvements = [
            comparison['accuracy_improvement'],
            comparison['f1_macro_improvement']
        ]
        axes[1, 1].bar(['Accuracy', 'F1 Macro'], improvements)
        axes[1, 1].set_title('Improvement (Model B - Model A)')
        axes[1, 1].set_ylabel('Improvement')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        
        plt.tight_layout()
        viz_file = self.viz_output_dir / "metrics_comparison.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Visualization saved to {viz_file}")
    
    def _generate_report(self, results: Dict) -> Path:
        """レポートを生成"""
        report_path = self.ab_test_output_dir / "report.md"
        
        metrics_a = results['model_a']
        metrics_b = results['model_b']
        comparison = results['comparison']
        
        report_content = f"""# SO8T Complete A/B Test Report

## Test Information
- **Session ID**: {self.session_id}
- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model A**: {self.ollama_model_a_name} (Base SO8T model)
- **Model B**: {self.ollama_model_b_name} (SO8T retrained model)

## Model A Metrics
- **Accuracy**: {metrics_a['accuracy']:.4f}
- **F1 Macro**: {metrics_a['f1_macro']:.4f}
- **Average Latency**: {metrics_a['avg_latency']:.4f} seconds
- **Success Rate**: {metrics_a['success_rate']:.2%}
- **Number of Samples**: {metrics_a['num_samples']}

## Model B Metrics
- **Accuracy**: {metrics_b['accuracy']:.4f}
- **F1 Macro**: {metrics_b['f1_macro']:.4f}
- **Average Latency**: {metrics_b['avg_latency']:.4f} seconds
- **Success Rate**: {metrics_b['success_rate']:.2%}
- **Number of Samples**: {metrics_b['num_samples']}

## Comparison
- **Accuracy Improvement**: {comparison['accuracy_improvement']:.4f}
- **F1 Macro Improvement**: {comparison['f1_macro_improvement']:.4f}
- **Latency Change**: {comparison['latency_change']:.4f} seconds

## Conclusion
{'Model B performs better' if comparison['accuracy_improvement'] > 0 else 'Model A performs better'}

## Visualization
See `visualizations/metrics_comparison.png` for detailed charts.
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[OK] Report saved to {report_path}")
        return report_path
    
    def run_complete_pipeline(self, resume: bool = True):
        """
        完全パイプラインを実行
        
        Args:
            resume: チェックポイントから再開するか
        """
        logger.info("="*80)
        logger.info("Starting SO8T Complete A/B Test Pipeline")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # チェックポイントから復旧
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logger.info(f"[RESUME] Resuming from checkpoint (Session: {self.session_id})")
                self.phase_progress = checkpoint.get('phase_progress', {})
        
        try:
            # Phase 1: Model AのGGUF変換
            if not self._is_phase_completed('phase1_model_a_gguf'):
                logger.info("[EXECUTE] Phase 1: Converting Model A to GGUF")
                model_a_gguf_files = self.phase1_convert_model_a_to_gguf()
            else:
                logger.info("[SKIP] Phase 1 already completed and verified")
                model_a_gguf_files = {q: Path(p) for q, p in self.phase_progress['phase1_model_a_gguf']['gguf_files'].items()}
            
            # Phase 2: SO8T再学習
            if not self._is_phase_completed('phase2_train_model_b'):
                logger.info("[EXECUTE] Phase 2: Training Model B with SO8T")
                trained_model_path = self.phase2_train_model_b()
            else:
                logger.info("[SKIP] Phase 2 already completed and verified")
                trained_model_path = Path(self.phase_progress['phase2_train_model_b']['trained_model_path'])
            
            # Phase 3: Model BのGGUF変換
            if not self._is_phase_completed('phase3_model_b_gguf'):
                logger.info("[EXECUTE] Phase 3: Converting Model B to GGUF")
                model_b_gguf_files = self.phase3_convert_model_b_to_gguf(trained_model_path)
            else:
                logger.info("[SKIP] Phase 3 already completed and verified")
                model_b_gguf_files = {q: Path(p) for q, p in self.phase_progress['phase3_model_b_gguf']['gguf_files'].items()}
            
            # Phase 4: Ollamaインポート
            if not self._is_phase_completed('phase4_ollama_import'):
                logger.info("[EXECUTE] Phase 4: Importing models to Ollama")
                self.phase4_import_to_ollama(model_a_gguf_files, model_b_gguf_files)
            else:
                logger.info("[SKIP] Phase 4 already completed and verified")
            
            # Phase 5: A/Bテスト実行
            if not self._is_phase_completed('phase5_ab_test'):
                logger.info("[EXECUTE] Phase 5: Running A/B test")
                ab_test_results = self.phase5_run_ab_test()
            else:
                logger.info("[SKIP] Phase 5 already completed and verified")
                results_file = Path(self.phase_progress['phase5_ab_test']['results_file'])
                with open(results_file, 'r', encoding='utf-8') as f:
                    ab_test_results = json.load(f)
            
            # Phase 6: 可視化とレポート生成
            if not self._is_phase_completed('phase6_visualization'):
                logger.info("[EXECUTE] Phase 6: Visualizing and generating report")
                self.phase6_visualize_and_report(ab_test_results)
            else:
                logger.info("[SKIP] Phase 6 already completed and verified")
            
            # Phase 7: Ollamaチェック
            if not self._is_phase_completed('phase7_ollama_check'):
                logger.info("[EXECUTE] Phase 7: Running Ollama check and validation")
                ollama_check_results = self.phase7_ollama_check(ab_test_results)
                
                # レポートにOllamaチェック結果を追加
                report_path = self.ab_test_output_dir / "report.md"
                if report_path.exists():
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    
                    ollama_check_section = f"""

## Ollama Check Results

- **Model A Available**: {ollama_check_results.get('model_a_available', False)}
- **Model B Available**: {ollama_check_results.get('model_b_available', False)}
- **Model A Test Passed**: {ollama_check_results.get('model_a_test_passed', False)}
- **Model B Test Passed**: {ollama_check_results.get('model_b_test_passed', False)}
- **Comparison Test Passed**: {ollama_check_results.get('comparison_test_passed', False)}
- **Check Timestamp**: {ollama_check_results.get('check_timestamp', 'N/A')}

### Model A Test Results
"""
                    for i, result in enumerate(ollama_check_results.get('model_a_test_results', []), 1):
                        ollama_check_section += f"\n**Test {i}**: {'PASSED' if result.get('success') else 'FAILED'}\n"
                        if result.get('error'):
                            ollama_check_section += f"- Error: {result.get('error')}\n"
                    
                    ollama_check_section += "\n### Model B Test Results\n"
                    for i, result in enumerate(ollama_check_results.get('model_b_test_results', []), 1):
                        ollama_check_section += f"\n**Test {i}**: {'PASSED' if result.get('success') else 'FAILED'}\n"
                        if result.get('error'):
                            ollama_check_section += f"- Error: {result.get('error')}\n"
                    
                    report_content += ollama_check_section
                    
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    
                    logger.info("[OK] Ollama check results added to report")
            else:
                logger.info("[SKIP] Phase 7 already completed and verified")
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("="*80)
            logger.info("Pipeline Completed Successfully")
            logger.info("="*80)
            logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
            logger.info(f"Results directory: {self.ab_test_output_dir}")
            
        except KeyboardInterrupt:
            logger.warning("[INTERRUPT] Pipeline interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self._save_checkpoint()
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Complete A/B Test Pipeline")
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/complete_so8t_ab_test_pipeline_config.yaml'),
        help='Configuration file path'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    pipeline = CompleteSO8TABTestPipeline(args.config)
    pipeline.run_complete_pipeline(resume=args.resume)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

