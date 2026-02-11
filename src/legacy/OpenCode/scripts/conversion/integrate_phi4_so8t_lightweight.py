#!/usr/bin/env python3
"""
Phi-4 SO8T統合（軽量版）
メモリ効率を重視した実装
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from so8t_core import SO8TRotationGate

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def integrate_so8t_config_only(
    model_path: str,
    output_path: str,
):
    """
    設定ファイルのみコピーしてSO8T統合情報を追加
    （メモリ節約のため、重みファイルは学習時に処理）
    
    Args:
        model_path: 元のPhi-4モデルパス
        output_path: SO8T統合後の保存先
    """
    logger.info(f"[STEP 1] Copying configuration from {model_path}")
    
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 設定ファイルをコピー
    config_files = [
        'config.json',
        'generation_config.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'added_tokens.json',
        'tokenizer.json',
        'vocab.json',
        'merges.txt',
    ]
    
    for config_file in config_files:
        src = model_path / config_file
        if src.exists():
            dst = output_path / config_file
            shutil.copy2(src, dst)
            logger.info(f"[COPY] {config_file}")
    
    # modeling_phi3.pyとconfiguration_phi3.pyもコピー
    for py_file in ['modeling_phi3.py', 'configuration_phi3.py']:
        src = model_path / py_file
        if src.exists():
            dst = output_path / py_file
            shutil.copy2(src, dst)
            logger.info(f"[COPY] {py_file}")
    
    # config.jsonを読み込んでSO8T情報を追加
    config_json = output_path / 'config.json'
    with open(config_json, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # SO8T統合フラグを追加
    config['so8t_integrated'] = True
    config['so8t_hidden_size'] = config['hidden_size']
    config['so8t_num_blocks'] = config['hidden_size'] // 8
    config['so8t_use_cayley'] = True
    config['so8t_orthogonal_reg'] = 1e-3
    
    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info("[STEP 2] Updated config.json with SO8T parameters")
    
    # 統合情報を保存
    integration_info = {
        'original_model': str(model_path),
        'integrated_layers': config['num_hidden_layers'],
        'total_layers': config['num_hidden_layers'],
        'hidden_size': config['hidden_size'],
        'num_blocks': config['hidden_size'] // 8,
        'so8t_parameters_per_layer': 64,
        'total_so8t_parameters': config['num_hidden_layers'] * 64,
        'note': 'Configuration-only integration. Actual weights will be integrated during training.',
    }
    
    with open(output_path / 'so8t_integration_info.json', 'w', encoding='utf-8') as f:
        json.dump(integration_info, f, indent=2, ensure_ascii=False)
    
    logger.info("[SUCCESS] Lightweight SO8T integration completed!")
    logger.info(f"Configuration saved to: {output_path}")
    logger.info("Note: Weight integration will happen during training")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Integrate SO8T (lightweight)")
    parser.add_argument("--model_path", type=str, default="Phi-4-mini-instruct", help="Phi-4 model path")
    parser.add_argument("--output_path", type=str, default="phi4_so8t_integrated", help="Output path")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Phi-4 SO8T Integration (Lightweight)")
    logger.info("=" * 70)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info("=" * 70)
    
    integrate_so8t_config_only(
        model_path=args.model_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()





