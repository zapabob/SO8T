#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重み凍結トレーニングのテストスクリプト

重み凍結の検証、学習可能パラメータ数の確認、勾配計算の検証を実行
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_frozen_weight_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FrozenWeightTester:
    """重み凍結トレーニングのテストクラス"""
    
    def __init__(self, model, config):
        """初期化"""
        self.model = model
        self.config = config
        logger.info("[INIT] FrozenWeightTester initialized")
    
    def test_weight_freezing(self) -> Dict[str, Any]:
        """重み凍結の検証"""
        logger.info("="*80)
        logger.info("TEST 1: Weight Freezing Verification")
        logger.info("="*80)
        
        freeze_base = self.config.get("model", {}).get("freeze_base_model", False)
        
        if not freeze_base:
            logger.warning("[WARNING] Weight freezing is disabled in config")
            return {"status": "skipped", "reason": "freeze_base_model is False"}
        
        # 学習可能パラメータと凍結パラメータをカウント
        trainable_params = []
        frozen_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.numel()))
            else:
                frozen_params.append((name, param.numel()))
        
        trainable_count = sum(p[1] for p in trainable_params)
        frozen_count = sum(p[1] for p in frozen_params)
        total_count = trainable_count + frozen_count
        
        logger.info(f"[RESULT] Trainable parameters: {trainable_count:,} ({100 * trainable_count / total_count:.2f}%)")
        logger.info(f"[RESULT] Frozen parameters: {frozen_count:,} ({100 * frozen_count / total_count:.2f}%)")
        
        # 学習可能パラメータの名前を確認
        trainable_keywords = ['lora', 'so8', 'rotation', 'alpha_gate', 'alpha', 'r_safe', 'r_cmd', 'soul']
        trainable_names = [name for name, _ in trainable_params]
        
        has_trainable_keywords = any(
            any(keyword in name.lower() for keyword in trainable_keywords)
            for name in trainable_names
        )
        
        if not has_trainable_keywords:
            logger.warning("[WARNING] No trainable parameters found with expected keywords")
        
        return {
            "status": "passed" if has_trainable_keywords else "failed",
            "trainable_count": trainable_count,
            "frozen_count": frozen_count,
            "trainable_ratio": trainable_count / total_count if total_count > 0 else 0.0,
            "has_trainable_keywords": has_trainable_keywords
        }
    
    def test_gradient_computation(self) -> Dict[str, Any]:
        """勾配計算の検証"""
        logger.info("="*80)
        logger.info("TEST 2: Gradient Computation Verification")
        logger.info("="*80)
        
        # ダミー入力を作成
        dummy_input = torch.randint(0, 1000, (1, 10)).to(next(self.model.parameters()).device)
        
        # フォワードパス
        try:
            outputs = self.model(dummy_input)
            
            # バックワードパス
            if isinstance(outputs, torch.Tensor):
                loss = outputs.mean()
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.mean()
            
            loss.backward()
            
            # 勾配が計算されているパラメータを確認
            params_with_grad = []
            params_without_grad = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        params_with_grad.append(name)
                    else:
                        params_without_grad.append(name)
            
            logger.info(f"[RESULT] Parameters with gradients: {len(params_with_grad)}")
            logger.info(f"[RESULT] Parameters without gradients: {len(params_without_grad)}")
            
            if params_without_grad:
                logger.warning(f"[WARNING] {len(params_without_grad)} trainable parameters have no gradients")
                logger.warning(f"[WARNING] First 5: {params_without_grad[:5]}")
            
            return {
                "status": "passed" if len(params_with_grad) > 0 else "failed",
                "params_with_grad": len(params_with_grad),
                "params_without_grad": len(params_without_grad)
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Gradient computation test failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_parameter_counts(self) -> Dict[str, Any]:
        """学習可能パラメータ数の確認"""
        logger.info("="*80)
        logger.info("TEST 3: Parameter Counts Verification")
        logger.info("="*80)
        
        # パラメータをカテゴリごとに分類
        categories = {
            'lora': [],
            'so8t': [],
            'alpha_gate': [],
            'soul': [],
            'base': []
        }
        
        for name, param in self.model.named_parameters():
            name_lower = name.lower()
            if 'lora' in name_lower:
                categories['lora'].append((name, param.numel(), param.requires_grad))
            elif any(kw in name_lower for kw in ['so8', 'rotation']):
                categories['so8t'].append((name, param.numel(), param.requires_grad))
            elif 'alpha_gate' in name_lower or 'alpha' in name_lower:
                categories['alpha_gate'].append((name, param.numel(), param.requires_grad))
            elif any(kw in name_lower for kw in ['r_safe', 'r_cmd', 'soul']):
                categories['soul'].append((name, param.numel(), param.requires_grad))
            else:
                categories['base'].append((name, param.numel(), param.requires_grad))
        
        # 結果をログ出力
        for category, params in categories.items():
            trainable_count = sum(p[1] for p in params if p[2])
            total_count = sum(p[1] for p in params)
            logger.info(f"[RESULT] {category.upper()}: {trainable_count:,} trainable / {total_count:,} total")
        
        return {
            "categories": {
                cat: {
                    "trainable": sum(p[1] for p in params if p[2]),
                    "total": sum(p[1] for p in params)
                }
                for cat, params in categories.items()
            }
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """すべてのテストを実行"""
        logger.info("="*80)
        logger.info("Running All Tests")
        logger.info("="*80)
        
        results = {}
        
        # Test 1: 重み凍結の検証
        results["weight_freezing"] = self.test_weight_freezing()
        
        # Test 2: 勾配計算の検証
        results["gradient_computation"] = self.test_gradient_computation()
        
        # Test 3: パラメータ数の確認
        results["parameter_counts"] = self.test_parameter_counts()
        
        # 総合結果
        all_passed = all(
            r.get("status") == "passed" or r.get("status") == "skipped"
            for r in results.values()
            if isinstance(r, dict) and "status" in r
        )
        
        results["overall"] = {
            "status": "passed" if all_passed else "failed",
            "all_tests_passed": all_passed
        }
        
        logger.info("="*80)
        logger.info(f"[RESULT] Overall status: {results['overall']['status']}")
        logger.info("="*80)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Test frozen weight training setup"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file path (YAML)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path to test"
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # モデル読み込み（簡易版）
    logger.info(f"[LOAD] Loading model from {args.model_path}...")
    # 実際のモデル読み込みは実装に依存するため、ここでは簡易版
    # 実際の使用時は、train_borea_phi35_so8t_thinking.pyのload_model_with_so8tを使用
    
    logger.info("[NOTE] Model loading is simplified in this test script")
    logger.info("[NOTE] For full testing, use the actual model loading from train_borea_phi35_so8t_thinking.py")
    
    # テスト実行（モデルが実際に読み込まれている場合）
    # tester = FrozenWeightTester(model, config)
    # results = tester.run_all_tests()
    
    logger.info("[COMPLETE] Test script completed (model loading needs to be implemented)")


if __name__ == "__main__":
    main()
