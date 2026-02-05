#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: モデルA準備（Borea-Phi-3.5-mini-Instruct-Jp → Q8_0 GGUF）

Borea-Phi-3.5-mini-Instruct-Jpをconvert_hf_to_gguf.pyでF16に変換し、
その後llama-quantizeでQ8_0に量子化します。

Usage:
    python scripts/pipelines/phase1_prepare_model_a.py --config configs/complete_automated_ab_pipeline.yaml
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
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase1_prepare_model_a.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase1ModelAPreparer:
    """Phase 1: モデルA準備クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.model_a_config = config.get('model_a', {})
        
        # パス設定
        base_path_str = self.model_a_config.get('base_path', 'Borea-Phi-3.5-mini-Instruct-Jp')
        # HuggingFaceモデルIDかローカルパスかを判定
        base_path_obj = Path(base_path_str)
        if base_path_obj.exists() and base_path_obj.is_dir():
            self.base_path = base_path_obj
            self.is_hf_model_id = False
        else:
            # HuggingFaceモデルIDとして扱う
            self.base_path = base_path_str
            self.is_hf_model_id = True
        
        self.output_dir = Path(self.model_a_config.get('output_dir', 'D:/webdataset/gguf_models/borea_phi35_mini_q8_0'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイント設定
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', 'D:/webdataset/checkpoints/complete_ab_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ツールパス
        self.convert_script = PROJECT_ROOT / "external" / "llama.cpp-master" / "convert_hf_to_gguf.py"
        self.quantize_tool = self._find_quantize_tool()
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Phase 1: Model A Preparer Initialized")
        logger.info("="*80)
        logger.info(f"Base model path: {self.base_path}")
        logger.info(f"Is HuggingFace model ID: {self.is_hf_model_id}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Session ID: {self.session_id}")
    
    def _find_quantize_tool(self) -> Path:
        """llama-quantizeツールを検索"""
        # 複数のパスを試行
        possible_paths = [
            PROJECT_ROOT / "external" / "llama.cpp-master" / "build" / "bin" / "Release" / "llama-quantize.exe",
            PROJECT_ROOT / "external" / "llama.cpp-master" / "build" / "bin" / "Release" / "llama-quantize",
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
            'phase': 'phase1',
            'base_path': str(self.base_path) if isinstance(self.base_path, Path) else self.base_path,
            'is_hf_model_id': self.is_hf_model_id,
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"{self.session_id}_phase1_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def validate_input(self) -> bool:
        """入力モデルの検証"""
        if self.is_hf_model_id:
            # HuggingFaceモデルIDの場合は検証をスキップ（convert_hf_to_gguf.pyが処理する）
            logger.info(f"[OK] Input model validated as HuggingFace model ID: {self.base_path}")
            logger.info(f"[INFO] Will use --remote option for conversion")
            return True
        else:
            # ローカルパスの場合
            if not self.base_path.exists():
                logger.error(f"Base model path does not exist: {self.base_path}")
                return False
            
            config_path = self.base_path / "config.json"
            if not config_path.exists():
                logger.error(f"config.json not found in: {self.base_path}")
                return False
            
            logger.info(f"[OK] Input model validated: {self.base_path}")
            return True
    
    def convert_to_f16(self) -> Path:
        """HFモデルをF16 GGUFに変換"""
        logger.info("="*80)
        logger.info("Step 1: Converting HF model to F16 GGUF")
        logger.info("="*80)
        
        if not self.convert_script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found: {self.convert_script}\n"
                f"Please ensure llama.cpp is cloned in external/llama.cpp-master/"
            )
        
        output_f16 = self.output_dir / "borea_phi35_mini_f16.gguf"
        
        # 既に存在する場合はサイズをチェック（不完全なファイルの可能性があるため）
        if output_f16.exists():
            file_size = output_f16.stat().st_size
            file_size_gb = file_size / (1024**3)
            logger.info(f"F16 GGUF already exists: {output_f16} ({file_size_gb:.2f} GB)")
            
            # ファイルサイズが異常に小さい場合は再変換
            # Phi-3.5-miniは約2-3GB程度になるはず
            if file_size_gb < 1.0:
                logger.warning(f"F16 file size is suspiciously small ({file_size_gb:.2f} GB), will re-convert")
                try:
                    output_f16.unlink()
                    logger.info("Corrupted F16 file removed successfully")
                except (PermissionError, OSError) as e:
                    logger.warning(f"Could not remove corrupted F16 file (may be in use): {e}")
                    # ファイルが使用中の場合は、一時ファイル名で変換してからリネーム
                    import time
                    temp_output = self.output_dir / f"borea_phi35_mini_f16_temp_{int(time.time())}.gguf"
                    logger.info(f"Will convert to temporary file: {temp_output}")
                    output_f16 = temp_output
            else:
                logger.info("Skipping conversion...")
                return output_f16
        
        # base_pathがPathオブジェクトの場合は絶対パスに変換
        if isinstance(self.base_path, Path):
            model_input = str(self.base_path.resolve())
        else:
            # 文字列の場合、まずPathとして確認してから絶対パスに変換
            path_obj = Path(self.base_path)
            if path_obj.exists():
                model_input = str(path_obj.resolve())
            else:
                model_input = self.base_path
        
        cmd = [
            sys.executable,
            str(self.convert_script),
            model_input,
            "--outfile", str(output_f16),
            "--outtype", "f16",
            "--model-name", "Borea-Phi-3.5-mini-Instruct-Jp"
        ]
        
        # HuggingFaceモデルIDの場合は--remoteオプションを追加
        if self.is_hf_model_id:
            cmd.append("--remote")
            logger.info(f"[INFO] Using --remote option for HuggingFace model ID")
        else:
            logger.info(f"[INFO] Using local model path")
        
        logger.info(f"[CONVERT] Running: {' '.join(cmd)}")
        
        try:
            # プログレスバー付きで実行
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
            output_lines = []
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line)
                    line_lower = line.lower()
                    
                    if "tensor" in line_lower or "layer" in line_lower:
                        pbar.update(1)
                    
                    if "successfully" in line_lower:
                        logger.info(f"  {line.strip()}")
                    
                    if "error" in line_lower or "warning" in line_lower:
                        logger.warning(f"  {line.strip()}")
            
            return_code = process.wait()
            pbar.close()
            
            # 出力ファイルが存在し、サイズが0より大きい場合は成功
            if output_f16.exists() and output_f16.stat().st_size > 0:
                file_size_gb = output_f16.stat().st_size / (1024**3)
                logger.info(f"[OK] F16 conversion completed: {output_f16} ({file_size_gb:.2f} GB)")
                logger.info(f"[INFO] Conversion process return code: {return_code}")
                
                # 一時ファイルの場合は、元のファイル名にリネーム
                final_output = self.output_dir / "borea_phi35_mini_f16.gguf"
                if output_f16 != final_output:
                    try:
                        # 元の破損ファイルを削除（可能な場合）
                        if final_output.exists():
                            try:
                                final_output.unlink()
                            except (PermissionError, OSError):
                                logger.warning(f"Could not remove old file, will overwrite: {final_output}")
                        
                        # 一時ファイルをリネーム
                        output_f16.rename(final_output)
                        logger.info(f"[OK] Renamed temporary file to: {final_output}")
                        return final_output
                    except Exception as e:
                        logger.warning(f"Could not rename temporary file: {e}")
                        logger.info(f"Using temporary file: {output_f16}")
                        return output_f16
                
                return output_f16
            else:
                error_output = '\n'.join(output_lines[-100:])
                logger.error(f"[ERROR] F16 conversion failed")
                logger.error(f"Process return code: {return_code}")
                logger.error(f"Error output (last 100 lines):\n{error_output}")
                raise RuntimeError("F16 conversion failed")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] F16 conversion failed: {e}")
            raise
    
    def quantize_to_q8_0(self, input_f16: Path) -> Path:
        """F16 GGUFをQ8_0に量子化"""
        logger.info("="*80)
        logger.info("Step 2: Quantizing F16 to Q8_0")
        logger.info("="*80)
        
        output_q8_0 = self.output_dir / "borea_phi35_mini_q8_0.gguf"
        
        # 既に存在する場合はスキップ
        if output_q8_0.exists():
            logger.info(f"Q8_0 GGUF already exists: {output_q8_0}")
            logger.info("Skipping quantization...")
            return output_q8_0
        
        cmd = [
            str(self.quantize_tool),
            str(input_f16),
            str(output_q8_0),
            "Q8_0"
        ]
        
        logger.info(f"[QUANTIZE] Running: {' '.join(cmd)}")
        
        try:
            # プログレスバー付きで実行
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
            output_lines = []
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line)
                    line_lower = line.lower()
                    
                    if "tensor" in line_lower or "layer" in line_lower or "%" in line:
                        pbar.update(1)
                        if "%" in line:
                            pbar.set_postfix_str(line.strip())
                    
                    if "successfully" in line_lower or "done" in line_lower:
                        logger.info(f"  {line.strip()}")
                    
                    if "error" in line_lower or "warning" in line_lower:
                        logger.warning(f"  {line.strip()}")
            
            process.wait()
            pbar.close()
            
            if output_q8_0.exists() and output_q8_0.stat().st_size > 0:
                file_size_gb = output_q8_0.stat().st_size / (1024**3)
                logger.info(f"[OK] Q8_0 quantization completed: {output_q8_0} ({file_size_gb:.2f} GB)")
                return output_q8_0
            else:
                error_output = '\n'.join(output_lines[-100:])
                logger.error(f"[ERROR] Q8_0 quantization failed")
                logger.error(f"Error output (last 100 lines):\n{error_output}")
                raise RuntimeError("Q8_0 quantization failed")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Q8_0 quantization failed: {e}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """Phase 1実行"""
        logger.info("="*80)
        logger.info("Starting Phase 1: Model A Preparation")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # 入力検証
            if not self.validate_input():
                raise ValueError("Input validation failed")
            
            # F16変換
            f16_path = self.convert_to_f16()
            
            # Q8_0量子化
            q8_0_path = self.quantize_to_q8_0(f16_path)
            
            # チェックポイント保存
            self._save_checkpoint()
            
            duration = time.time() - start_time
            
            result = {
                'status': 'completed',
                'duration': duration,
                'f16_path': str(f16_path),
                'q8_0_path': str(q8_0_path),
                'output_dir': str(self.output_dir),
                'session_id': self.session_id
            }
            
            logger.info("="*80)
            logger.info("[SUCCESS] Phase 1 completed!")
            logger.info("="*80)
            logger.info(f"Duration: {duration/60:.2f} minutes")
            logger.info(f"F16 model: {f16_path}")
            logger.info(f"Q8_0 model: {q8_0_path}")
            
            # 音声通知
            self._play_audio_notification()
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error("="*80)
            logger.error(f"[ERROR] Phase 1 failed: {e}")
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
        description="Phase 1: Prepare Model A (Borea-Phi-3.5-mini-Instruct-Jp → Q8_0 GGUF)"
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
            'model_a': {
                'base_path': 'models/Borea-Phi-3.5-mini-Instruct-Jp',
                'output_dir': 'D:/webdataset/gguf_models/borea_phi35_mini_q8_0'
            },
            'checkpoint': {
                'save_dir': 'D:/webdataset/checkpoints/complete_ab_pipeline'
            }
        }
    else:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # Phase 1実行
    preparer = Phase1ModelAPreparer(config)
    
    try:
        result = preparer.run()
        logger.info("Phase 1 completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("[WARNING] Phase 1 interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"[FAILED] Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

