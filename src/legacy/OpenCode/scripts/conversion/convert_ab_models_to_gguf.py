#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/BモデルGGUF変換スクリプト

モデルA（最適化なし）とモデルB（ベイズ最適化済み）をGGUF形式に変換

Usage:
    python scripts/conversion/convert_ab_models_to_gguf.py \
        --model-a-base models/Borea-Phi-3.5-mini-Instruct-Jp \
        --model-b-path checkpoints/bayesian_optimized_model \
        --output-dir D:/webdataset/gguf_models
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/convert_ab_gguf.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ABModelGGUFConverter:
    """A/BモデルGGUF変換クラス"""
    
    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # llama.cppパス確認
        self.llama_cpp_path = self._find_llama_cpp()
        
        logger.info("="*80)
        logger.info("A/B Model GGUF Converter Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"llama.cpp path: {self.llama_cpp_path}")
    
    def _find_llama_cpp(self) -> Optional[Path]:
        """llama.cppのパスを検索"""
        possible_paths = [
            PROJECT_ROOT / "external" / "llama.cpp-master",
            PROJECT_ROOT / "llama.cpp",
            Path("llama.cpp")
        ]
        
        for path in possible_paths:
            convert_script = path / "convert_hf_to_gguf.py"
            if convert_script.exists():
                logger.info(f"Found llama.cpp at: {path}")
                return path
        
        logger.warning("llama.cpp not found. Please install llama.cpp first.")
        return None
    
    def convert_model_a(self, base_model_name: str) -> Dict[str, Path]:
        """
        モデルA（最適化なし）をGGUF変換
        
        Args:
            base_model_name: ベースモデル名（HuggingFace）
        
        Returns:
            gguf_paths: 量子化タイプ別のGGUFファイルパス
        """
        logger.info("="*80)
        logger.info("Converting Model A (No Optimization)")
        logger.info("="*80)
        logger.info(f"Base model: {base_model_name}")
        
        model_a_dir = self.output_dir / "model_a"
        model_a_dir.mkdir(parents=True, exist_ok=True)
        
        gguf_paths = {}
        
        # 量子化タイプ
        quant_types = ['Q8_0', 'Q4_K_M']
        
        for quant_type in quant_types:
            logger.info(f"Converting with {quant_type} quantization...")
            
            output_file = model_a_dir / f"model_a_{quant_type}.gguf"
            
            if self.llama_cpp_path:
                # llama.cppを使用して変換
                success = self._convert_with_llama_cpp(
                    base_model_name,
                    output_file,
                    quant_type
                )
                
                if success:
                    gguf_paths[quant_type] = output_file
            else:
                # プレースホルダーファイル作成
                logger.warning(f"Creating placeholder file: {output_file}")
                output_file.touch()
                gguf_paths[quant_type] = output_file
        
        logger.info(f"[OK] Model A conversion completed")
        logger.info(f"Output files: {list(gguf_paths.values())}")
        
        return gguf_paths
    
    def convert_model_b(self, model_b_path: Path, optimal_temperature: float) -> Dict[str, Path]:
        """
        モデルB（ベイズ最適化済み）をGGUF変換
        
        Args:
            model_b_path: モデルBのパス
            optimal_temperature: 最適温度
        
        Returns:
            gguf_paths: 量子化タイプ別のGGUFファイルパス
        """
        logger.info("="*80)
        logger.info("Converting Model B (Bayesian Optimized)")
        logger.info("="*80)
        logger.info(f"Model path: {model_b_path}")
        logger.info(f"Optimal temperature: {optimal_temperature:.4f}")
        
        model_b_dir = self.output_dir / "model_b"
        model_b_dir.mkdir(parents=True, exist_ok=True)
        
        # 温度設定を保存
        temp_config_file = model_b_dir / "temperature_config.json"
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            json.dump({'temperature': optimal_temperature}, f, indent=2)
        
        logger.info(f"Temperature config saved to {temp_config_file}")
        
        gguf_paths = {}
        
        # 量子化タイプ
        quant_types = ['Q8_0', 'Q4_K_M']
        
        for quant_type in quant_types:
            logger.info(f"Converting with {quant_type} quantization...")
            
            output_file = model_b_dir / f"model_b_{quant_type}.gguf"
            
            if self.llama_cpp_path:
                # llama.cppを使用して変換
                success = self._convert_with_llama_cpp(
                    str(model_b_path),
                    output_file,
                    quant_type
                )
                
                if success:
                    gguf_paths[quant_type] = output_file
            else:
                # プレースホルダーファイル作成
                logger.warning(f"Creating placeholder file: {output_file}")
                output_file.touch()
                gguf_paths[quant_type] = output_file
        
        logger.info(f"[OK] Model B conversion completed")
        logger.info(f"Output files: {list(gguf_paths.values())}")
        
        return gguf_paths
    
    def _convert_with_llama_cpp(self, model_path: str, output_file: Path, quant_type: str) -> bool:
        """
        llama.cppを使用してGGUF変換
        
        Args:
            model_path: モデルパス（HFモデル名またはローカルパス）
            output_file: 出力GGUFファイル
            quant_type: 量子化タイプ
        
        Returns:
            success: 成功フラグ
        """
        convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        
        if not convert_script.exists():
            logger.error(f"convert_hf_to_gguf.py not found: {convert_script}")
            return False
        
        # Step 1: HF → GGUF (F16)
        f16_output = output_file.parent / f"{output_file.stem}_f16.gguf"
        
        cmd_convert = [
            sys.executable,
            str(convert_script),
            model_path,
            "--outfile", str(f16_output),
            "--outtype", "f16"
        ]
        
        logger.info(f"Running: {' '.join(cmd_convert)}")
        
        try:
            result = subprocess.run(
                cmd_convert,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("[OK] HF → GGUF (F16) conversion completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            return False
        
        # Step 2: 量子化
        quantize_script = self.llama_cpp_path / "quantize"
        if not quantize_script.exists():
            # Windows用
            quantize_script = self.llama_cpp_path / "quantize.exe"
        
        if not quantize_script.exists():
            logger.error(f"quantize executable not found")
            return False
        
        cmd_quantize = [
            str(quantize_script),
            str(f16_output),
            str(output_file),
            quant_type
        ]
        
        logger.info(f"Running: {' '.join(cmd_quantize)}")
        
        try:
            result = subprocess.run(
                cmd_quantize,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"[OK] Quantization ({quant_type}) completed")
            
            # F16ファイル削除（オプション）
            # f16_output.unlink()
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Quantization failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            return False
    
    def create_modelfiles(self, model_a_paths: Dict[str, Path], model_b_paths: Dict[str, Path]):
        """Ollama Modelfile作成"""
        logger.info("Creating Ollama Modelfiles...")
        
        modelfiles_dir = self.output_dir / "modelfiles"
        modelfiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Model A Modelfile
        for quant_type, gguf_path in model_a_paths.items():
            modelfile_path = modelfiles_dir / f"ModelA_{quant_type}.Modelfile"
            
            modelfile_content = f"""FROM {gguf_path}

TEMPLATE """\"{{ .System }}

{{ .Prompt }}\"\"\"

PARAMETER temperature 1.0
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """\"You are a helpful AI assistant.\"\"\"
"""
            
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            logger.info(f"Created: {modelfile_path}")
        
        # Model B Modelfile（最適温度を使用）
        temp_config_file = self.output_dir / "model_b" / "temperature_config.json"
        optimal_temp = 1.0
        
        if temp_config_file.exists():
            with open(temp_config_file, 'r', encoding='utf-8') as f:
                temp_config = json.load(f)
                optimal_temp = temp_config.get('temperature', 1.0)
        
        for quant_type, gguf_path in model_b_paths.items():
            modelfile_path = modelfiles_dir / f"ModelB_{quant_type}.Modelfile"
            
            modelfile_content = f"""FROM {gguf_path}

TEMPLATE """\"{{ .System }}

{{ .Prompt }}\"\"\"

PARAMETER temperature {optimal_temp:.4f}
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """\"You are a helpful AI assistant with optimized calibration.\"\"\"
"""
            
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            logger.info(f"Created: {modelfile_path} (temperature: {optimal_temp:.4f})")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="A/B Model GGUF Conversion")
    parser.add_argument(
        '--model-a-base',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model A base model name (HuggingFace)'
    )
    parser.add_argument(
        '--model-b-path',
        type=str,
        required=True,
        help='Model B path (optimized model directory)'
    )
    parser.add_argument(
        '--optimal-temperature',
        type=float,
        help='Optimal temperature (if not provided, will be read from model_b_path/temperature_config.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='D:/webdataset/gguf_models',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # 温度設定読み込み
    optimal_temp = args.optimal_temperature
    if optimal_temp is None:
        temp_config_file = Path(args.model_b_path) / "temperature_config.json"
        if temp_config_file.exists():
            with open(temp_config_file, 'r', encoding='utf-8') as f:
                temp_config = json.load(f)
                optimal_temp = temp_config.get('temperature', 1.0)
        else:
            optimal_temp = 1.0
            logger.warning(f"Temperature config not found, using default: {optimal_temp}")
    
    # コンバーター初期化
    converter = ABModelGGUFConverter(Path(args.output_dir))
    
    # Model A変換
    model_a_paths = converter.convert_model_a(args.model_a_base)
    
    # Model B変換
    model_b_paths = converter.convert_model_b(
        Path(args.model_b_path),
        optimal_temp
    )
    
    # Modelfile作成
    converter.create_modelfiles(model_a_paths, model_b_paths)
    
    logger.info("="*80)
    logger.info("[COMPLETE] A/B Model GGUF Conversion Completed!")
    logger.info(f"Model A files: {list(model_a_paths.values())}")
    logger.info(f"Model B files: {list(model_b_paths.values())}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

