#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5-mini-Instruct-Commonを直接GGUF変換するスクリプト

段階A: ベースモデルをGGUF形式に変換（F16）

Usage:
    python scripts/convert_borea_to_gguf.py --input Borea-Phi-3.5-mini-Instruct-Common --output models/borea_phi35_mini_base.gguf
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent


class BoreaGGUFConverter:
    """Borea-Phi-3.5-mini GGUF変換クラス"""
    
    def __init__(
        self,
        input_model_path: str,
        output_gguf_path: Optional[str] = None,
        outtype: str = "f16"
    ):
        """
        Args:
            input_model_path: 入力モデルパス（HF形式）
            output_gguf_path: 出力GGUFファイルパス
            outtype: 出力タイプ（f16, f32, bf16）
        """
        self.input_model_path = Path(input_model_path)
        self.outtype = outtype
        
        if output_gguf_path:
            self.output_gguf_path = Path(output_gguf_path)
        else:
            # デフォルト出力パス
            models_dir = PROJECT_ROOT / "models"
            models_dir.mkdir(exist_ok=True)
            self.output_gguf_path = models_dir / f"borea_phi35_mini_base_{outtype}.gguf"
        
        # convert_hf_to_gguf.pyのパス
        self.convert_script = PROJECT_ROOT / "external" / "llama.cpp-master" / "convert_hf_to_gguf.py"
        
        if not self.convert_script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found: {self.convert_script}\n"
                f"Please ensure llama.cpp is cloned in external/llama.cpp-master/"
            )
        
        logger.info(f"Input model: {self.input_model_path}")
        logger.info(f"Output GGUF: {self.output_gguf_path}")
        logger.info(f"Output type: {self.outtype}")
    
    def validate_input(self) -> bool:
        """入力モデルの検証"""
        if not self.input_model_path.exists():
            logger.error(f"Input model path does not exist: {self.input_model_path}")
            return False
        
        # config.jsonの存在確認
        config_path = self.input_model_path / "config.json"
        if not config_path.exists():
            logger.error(f"config.json not found in: {self.input_model_path}")
            return False
        
        logger.info(f"[OK] Input model validated: {self.input_model_path}")
        return True
    
    def convert(self) -> Path:
        """GGUF変換実行"""
        logger.info("="*80)
        logger.info("Borea-Phi-3.5-mini GGUF Conversion (Stage A)")
        logger.info("="*80)
        
        # 入力検証
        if not self.validate_input():
            raise ValueError("Input validation failed")
        
        # 出力ディレクトリ作成
        self.output_gguf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 変換コマンド構築
        cmd = [
            sys.executable,
            str(self.convert_script),
            str(self.input_model_path),
            "--outfile", str(self.output_gguf_path),
            "--outtype", self.outtype,
            "--model-name", "Borea-Phi-3.5-mini-Instruct-Common"
        ]
        
        logger.info(f"[CONVERT] Running conversion command...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # 変換実行
            result = subprocess.run(
                cmd,
                cwd=str(self.convert_script.parent),
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("[OK] GGUF conversion completed successfully")
            logger.info(f"Output file: {self.output_gguf_path}")
            
            # ファイルサイズ確認
            if self.output_gguf_path.exists():
                file_size_mb = self.output_gguf_path.stat().st_size / (1024 * 1024)
                logger.info(f"Output file size: {file_size_mb:.2f} MB")
            
            return self.output_gguf_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] GGUF conversion failed")
            logger.error(f"Return code: {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout[-1000:]}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr[-1000:]}")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Convert Borea-Phi-3.5-mini-Instruct-Common to GGUF format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Borea-Phi-3.5-mini-Instruct-Common",
        help="Input model path (HF format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GGUF file path (default: models/borea_phi35_mini_base_{outtype}.gguf)"
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="f16",
        choices=["f32", "f16", "bf16"],
        help="Output type (default: f16)"
    )
    
    args = parser.parse_args()
    
    # 変換実行
    converter = BoreaGGUFConverter(
        input_model_path=args.input,
        output_gguf_path=args.output,
        outtype=args.outtype
    )
    
    try:
        output_path = converter.convert()
        logger.info("="*80)
        logger.info(f"[SUCCESS] Conversion completed: {output_path}")
        logger.info("="*80)
        
        # 音声通知
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
        
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

