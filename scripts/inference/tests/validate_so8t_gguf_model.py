#!/usr/bin/env python3
"""
SO8T GGUFモデルの検証とテスト

変換されたSO8T GGUFモデルの動作確認とテストを実行
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import argparse
from tqdm import tqdm
import numpy as np
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# llama.cppのGGUFライブラリをパスに追加
llama_cpp_path = project_root / "external" / "llama.cpp-master"
sys.path.insert(0, str(llama_cpp_path / "gguf-py"))

try:
    import gguf
    from gguf import GGUFReader
except ImportError:
    print("[ERROR] GGUFライブラリが見つかりません。llama.cppのgguf-pyディレクトリを確認してください。")
    sys.exit(1)

# Transformers関連のインポート
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)


class SO8TGGUFValidator:
    """SO8T GGUFモデルの検証クラス"""
    
    def __init__(self, gguf_model_path: str, so8t_model_path: str):
        """
        Args:
            gguf_model_path: GGUFモデルパス
            so8t_model_path: SO8Tモデルパス（比較用）
        """
        self.gguf_model_path = Path(gguf_model_path)
        self.so8t_model_path = Path(so8t_model_path)
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用デバイス: {self.device}")
    
    def load_gguf_model(self) -> Tuple[GGUFReader, Dict[str, Any]]:
        """GGUFモデルを読み込み"""
        logger.info(f"GGUFモデルを読み込み中: {self.gguf_model_path}")
        
        try:
            # GGUFリーダーの作成
            reader = GGUFReader(self.gguf_model_path, "r")
            
            # メタデータの取得
            metadata = {}
            for key in reader.metadata.keys():
                metadata[key] = reader.metadata[key]
            
            logger.info("GGUFモデルの読み込み完了")
            return reader, metadata
            
        except Exception as e:
            logger.error(f"GGUFモデル読み込みエラー: {e}")
            raise
    
    def load_so8t_model(self) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """SO8Tモデルを読み込み（比較用）"""
        logger.info(f"SO8Tモデルを読み込み中: {self.so8t_model_path}")
        
        try:
            # 設定ファイルの読み込み
            config_path = self.so8t_model_path / "config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 重みファイルの読み込み
            weights_path = self.so8t_model_path / "model.safetensors"
            if weights_path.exists():
                from safetensors.torch import load_file
                weights = load_file(weights_path)
            else:
                weights_path = self.so8t_model_path / "pytorch_model.bin"
                weights = torch.load(weights_path, map_location=self.device)
            
            logger.info("SO8Tモデルの読み込み完了")
            return config, weights
            
        except Exception as e:
            logger.error(f"SO8Tモデル読み込みエラー: {e}")
            raise
    
    def validate_metadata(self, gguf_metadata: Dict[str, Any], so8t_config: Dict[str, Any]) -> Dict[str, bool]:
        """メタデータの検証"""
        logger.info("メタデータを検証中...")
        
        validation_results = {}
        
        # 基本構造の検証
        basic_fields = [
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "max_position_embeddings", "hidden_act", "rms_norm_eps"
        ]
        
        for field in basic_fields:
            gguf_key = f"so8t.{field}"
            if gguf_key in gguf_metadata:
                gguf_value = gguf_metadata[gguf_key]
                so8t_value = so8t_config.get(field)
                validation_results[f"metadata_{field}"] = gguf_value == so8t_value
                if not validation_results[f"metadata_{field}"]:
                    logger.warning(f"メタデータ不一致 ({field}): GGUF={gguf_value}, SO8T={so8t_value}")
            else:
                validation_results[f"metadata_{field}"] = False
                logger.warning(f"メタデータフィールドが見つかりません: {gguf_key}")
        
        # SO8T特有の設定の検証
        so8t_fields = [
            "rotation_dim", "safety_features", "pet_lambda",
            "safety_threshold", "group_structure"
        ]
        
        for field in so8t_fields:
            gguf_key = f"so8t.{field}"
            if gguf_key in gguf_metadata:
                gguf_value = gguf_metadata[gguf_key]
                so8t_value = so8t_config.get(field)
                validation_results[f"so8t_{field}"] = gguf_value == so8t_value
                if not validation_results[f"so8t_{field}"]:
                    logger.warning(f"SO8T設定不一致 ({field}): GGUF={gguf_value}, SO8T={so8t_value}")
            else:
                validation_results[f"so8t_{field}"] = False
                logger.warning(f"SO8T設定フィールドが見つかりません: {gguf_key}")
        
        logger.info("メタデータ検証完了")
        return validation_results
    
    def validate_tensors(self, gguf_reader: GGUFReader, so8t_weights: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """テンソルの検証"""
        logger.info("テンソルを検証中...")
        
        validation_results = {}
        
        # GGUFテンソルの取得
        gguf_tensors = {}
        for tensor_name in gguf_reader.tensor_names:
            tensor_data = gguf_reader.get_tensor_data(tensor_name)
            gguf_tensors[tensor_name] = torch.from_numpy(tensor_data)
        
        # テンソルの比較
        for so8t_name, so8t_tensor in tqdm(so8t_weights.items(), desc="テンソル検証"):
            # SO8Tテンソル名をGGUF形式に変換
            gguf_name = so8t_name.replace('.', '_')
            
            if gguf_name in gguf_tensors:
                gguf_tensor = gguf_tensors[gguf_name]
                
                # 形状の比較
                shape_match = so8t_tensor.shape == gguf_tensor.shape
                validation_results[f"tensor_{so8t_name}_shape"] = shape_match
                
                if not shape_match:
                    logger.warning(f"テンソル形状不一致 ({so8t_name}): SO8T={so8t_tensor.shape}, GGUF={gguf_tensor.shape}")
                    continue
                
                # 数値の比較（量子化の影響を考慮）
                try:
                    # テンソルを同じデバイスに移動
                    so8t_tensor = so8t_tensor.cpu()
                    gguf_tensor = gguf_tensor.cpu()
                    
                    # 数値型の統一
                    if so8t_tensor.dtype != gguf_tensor.dtype:
                        so8t_tensor = so8t_tensor.float()
                        gguf_tensor = gguf_tensor.float()
                    
                    # 相対誤差の計算
                    abs_diff = torch.abs(so8t_tensor - gguf_tensor)
                    rel_diff = abs_diff / (torch.abs(so8t_tensor) + 1e-8)
                    
                    max_abs_diff = torch.max(abs_diff).item()
                    max_rel_diff = torch.max(rel_diff).item()
                    mean_rel_diff = torch.mean(rel_diff).item()
                    
                    # 許容誤差の設定（量子化の影響を考慮）
                    tolerance_abs = 1e-3
                    tolerance_rel = 0.01  # 1%
                    
                    value_match = max_abs_diff < tolerance_abs and max_rel_diff < tolerance_rel
                    validation_results[f"tensor_{so8t_name}_values"] = value_match
                    
                    if not value_match:
                        logger.warning(f"テンソル値不一致 ({so8t_name}): max_abs_diff={max_abs_diff:.2e}, max_rel_diff={max_rel_diff:.2e}")
                    else:
                        logger.debug(f"テンソル一致 ({so8t_name}): max_abs_diff={max_abs_diff:.2e}, mean_rel_diff={mean_rel_diff:.2e}")
                
                except Exception as e:
                    logger.warning(f"テンソル比較エラー ({so8t_name}): {e}")
                    validation_results[f"tensor_{so8t_name}_values"] = False
            
            else:
                validation_results[f"tensor_{so8t_name}_exists"] = False
                logger.warning(f"GGUFテンソルが見つかりません: {gguf_name}")
        
        logger.info("テンソル検証完了")
        return validation_results
    
    def test_model_loading(self, gguf_reader: GGUFReader) -> Dict[str, bool]:
        """モデル読み込みテスト"""
        logger.info("モデル読み込みテストを実行中...")
        
        test_results = {}
        
        try:
            # テンソル数の確認
            tensor_count = len(gguf_reader.tensor_names)
            test_results["tensor_count"] = tensor_count > 0
            logger.info(f"テンソル数: {tensor_count}")
            
            # メタデータの確認
            metadata_count = len(gguf_reader.metadata)
            test_results["metadata_count"] = metadata_count > 0
            logger.info(f"メタデータ数: {metadata_count}")
            
            # 基本テンソルの存在確認
            essential_tensors = [
                "model_embed_tokens_weight",
                "model_norm_weight",
                "lm_head_weight"
            ]
            
            for tensor_name in essential_tensors:
                exists = tensor_name in gguf_reader.tensor_names
                test_results[f"tensor_{tensor_name}_exists"] = exists
                if not exists:
                    logger.warning(f"必須テンソルが見つかりません: {tensor_name}")
            
            # SO8T特有のテンソルの存在確認
            so8t_tensors = [
                "model_layers_0_so8_rotation_theta",
                "model_layers_0_non_commutative_R_safe_rotation_params",
                "model_layers_0_non_commutative_R_cmd_rotation_params"
            ]
            
            for tensor_name in so8t_tensors:
                exists = tensor_name in gguf_reader.tensor_names
                test_results[f"so8t_tensor_{tensor_name}_exists"] = exists
                if not exists:
                    logger.warning(f"SO8Tテンソルが見つかりません: {tensor_name}")
            
        except Exception as e:
            logger.error(f"モデル読み込みテストエラー: {e}")
            test_results["loading_error"] = False
        
        logger.info("モデル読み込みテスト完了")
        return test_results
    
    def test_so8_properties(self, gguf_reader: GGUFReader) -> Dict[str, bool]:
        """SO(8)群性質のテスト"""
        logger.info("SO(8)群性質をテスト中...")
        
        test_results = {}
        
        try:
            # 回転テンソルの取得
            rotation_tensors = [name for name in gguf_reader.tensor_names if "so8_rotation_theta" in name]
            
            if not rotation_tensors:
                logger.warning("回転テンソルが見つかりません")
                test_results["rotation_tensors_exist"] = False
                return test_results
            
            test_results["rotation_tensors_exist"] = True
            
            # 最初の回転テンソルでSO(8)群性質をテスト
            rotation_tensor_name = rotation_tensors[0]
            rotation_data = gguf_reader.get_tensor_data(rotation_tensor_name)
            rotation_tensor = torch.from_numpy(rotation_data)
            
            # 8次元回転行列の生成（簡易版）
            if len(rotation_tensor.shape) >= 2:
                # 最初の8x8ブロックを取得
                R = rotation_tensor[:8, :8] if rotation_tensor.shape[0] >= 8 and rotation_tensor.shape[1] >= 8 else rotation_tensor
                
                # 直交性のテスト: R^T @ R ≈ I
                R_T = R.transpose(-1, -2)
                identity_approx = torch.matmul(R_T, R)
                identity_true = torch.eye(min(8, R.shape[0]), min(8, R.shape[1]))
                
                orthogonality_error = torch.max(torch.abs(identity_approx - identity_true)).item()
                test_results["orthogonality"] = orthogonality_error < 1e-3
                
                if not test_results["orthogonality"]:
                    logger.warning(f"直交性エラー: {orthogonality_error:.2e}")
                
                # 行列式のテスト: det(R) ≈ 1
                if R.shape[0] == R.shape[1]:
                    det_R = torch.det(R).item()
                    det_error = abs(det_R - 1.0)
                    test_results["determinant"] = det_error < 1e-3
                    
                    if not test_results["determinant"]:
                        logger.warning(f"行列式エラー: det={det_R:.6f}, error={det_error:.2e}")
                
                logger.info(f"SO(8)群性質テスト完了: orthogonality_error={orthogonality_error:.2e}")
            
        except Exception as e:
            logger.error(f"SO(8)群性質テストエラー: {e}")
            test_results["so8_test_error"] = False
        
        return test_results
    
    def generate_validation_report(self, all_results: Dict[str, bool]) -> Dict[str, Any]:
        """検証レポートの生成"""
        logger.info("検証レポートを生成中...")
        
        # 結果の集計
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results.values() if result)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # カテゴリ別の集計
        categories = {
            "metadata": [k for k in all_results.keys() if k.startswith("metadata_")],
            "tensor": [k for k in all_results.keys() if k.startswith("tensor_")],
            "so8t": [k for k in all_results.keys() if k.startswith("so8t_")],
            "loading": [k for k in all_results.keys() if k.startswith("tensor_count") or k.startswith("metadata_count") or k.startswith("loading_error")],
            "so8_properties": [k for k in all_results.keys() if k.startswith("orthogonality") or k.startswith("determinant") or k.startswith("rotation_")]
        }
        
        category_results = {}
        for category, tests in categories.items():
            if tests:
                category_passed = sum(1 for test in tests if all_results.get(test, False))
                category_total = len(tests)
                category_results[category] = {
                    "passed": category_passed,
                    "total": category_total,
                    "success_rate": (category_passed / category_total) * 100 if category_total > 0 else 0
                }
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gguf_model_path": str(self.gguf_model_path),
            "so8t_model_path": str(self.so8t_model_path),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate
            },
            "category_results": category_results,
            "detailed_results": all_results,
            "status": "PASS" if success_rate >= 90 else "WARNING" if success_rate >= 70 else "FAIL"
        }
        
        logger.info("検証レポート生成完了")
        return report
    
    def validate(self) -> Dict[str, Any]:
        """検証の実行"""
        logger.info("=" * 80)
        logger.info("SO8T GGUFモデル検証開始")
        logger.info("=" * 80)
        
        all_results = {}
        
        try:
            # 1. GGUFモデルの読み込み
            gguf_reader, gguf_metadata = self.load_gguf_model()
            
            # 2. SO8Tモデルの読み込み
            so8t_config, so8t_weights = self.load_so8t_model()
            
            # 3. メタデータの検証
            metadata_results = self.validate_metadata(gguf_metadata, so8t_config)
            all_results.update(metadata_results)
            
            # 4. テンソルの検証
            tensor_results = self.validate_tensors(gguf_reader, so8t_weights)
            all_results.update(tensor_results)
            
            # 5. モデル読み込みテスト
            loading_results = self.test_model_loading(gguf_reader)
            all_results.update(loading_results)
            
            # 6. SO(8)群性質のテスト
            so8_results = self.test_so8_properties(gguf_reader)
            all_results.update(so8_results)
            
            # 7. 検証レポートの生成
            report = self.generate_validation_report(all_results)
            
            logger.info("=" * 80)
            logger.info("SO8T GGUFモデル検証完了")
            logger.info("=" * 80)
            logger.info(f"総テスト数: {report['summary']['total_tests']}")
            logger.info(f"成功: {report['summary']['passed_tests']}")
            logger.info(f"失敗: {report['summary']['failed_tests']}")
            logger.info(f"成功率: {report['summary']['success_rate']:.1f}%")
            logger.info(f"ステータス: {report['status']}")
            
            return report
            
        except Exception as e:
            logger.error(f"検証中にエラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='SO8T GGUFモデルの検証')
    parser.add_argument('--gguf-model', required=True, help='GGUFモデルパス')
    parser.add_argument('--so8t-model', required=True, help='SO8Tモデルパス（比較用）')
    parser.add_argument('--output-report', help='検証レポート出力パス')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ出力')
    
    args = parser.parse_args()
    
    # ログ設定
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 検証器の作成と実行
    validator = SO8TGGUFValidator(
        gguf_model_path=args.gguf_model,
        so8t_model_path=args.so8t_model
    )
    
    try:
        report = validator.validate()
        
        # レポートの保存
        if args.output_report:
            report_path = Path(args.output_report)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"検証レポートを保存しました: {report_path}")
        
        print(f"\n[SUCCESS] 検証が正常に完了しました")
        print(f"ステータス: {report['status']}")
        print(f"成功率: {report['summary']['success_rate']:.1f}%")
        
        return 0 if report['status'] == 'PASS' else 1
        
    except Exception as e:
        print(f"\n[ERROR] 検証中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
