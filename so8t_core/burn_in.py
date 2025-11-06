"""
SO8T焼き込み機構
学習済みSO8T回転行列を線形層に吸収
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class BurnInManager:
    """
    SO8T回転行列を線形層に焼き込むマネージャー
    
    主要機能:
    - 右掛け吸収: W_o' = W_o @ R
    - ブロック対角行列構築
    - 吸収前後の誤差検証
    - モデル全体への適用
    """
    
    def __init__(
        self,
        verify_error_threshold: float = 1e-5,
    ):
        """
        Args:
            verify_error_threshold: 誤差検証の閾値
        """
        self.verify_threshold = verify_error_threshold
        self.burn_in_log = []
    
    def build_block_diagonal_rotation(
        self,
        rotation_matrices: Tensor,
        hidden_size: int,
    ) -> Tensor:
        """
        ブロック対角回転行列を構築
        
        Args:
            rotation_matrices: [num_blocks, 8, 8] SO8T回転行列
            hidden_size: 隠れ層サイズ
        
        Returns:
            R: [hidden_size, hidden_size] ブロック対角回転行列
        """
        num_blocks = hidden_size // 8
        assert rotation_matrices.shape[0] == num_blocks
        
        # ブロック対角行列を初期化
        R = torch.zeros(hidden_size, hidden_size, device=rotation_matrices.device, dtype=rotation_matrices.dtype)
        
        # 各ブロックを配置
        for i in range(num_blocks):
            start_idx = i * 8
            end_idx = start_idx + 8
            R[start_idx:end_idx, start_idx:end_idx] = rotation_matrices[i]
        
        return R
    
    def burn_in_linear_layer(
        self,
        linear_layer: nn.Linear,
        rotation_gate,  # SO8TRotationGate
        layer_name: str = "unknown",
    ) -> Dict[str, float]:
        """
        線形層にSO8T回転を焼き込む
        W' = W @ R
        
        Args:
            linear_layer: 焼き込み対象の線形層
            rotation_gate: SO8TRotationGate インスタンス
            layer_name: レイヤー名（ログ用）
        
        Returns:
            metrics: 焼き込みメトリクス
        """
        with torch.no_grad():
            # 元の重みを保存
            W_original = linear_layer.weight.data.clone()
            
            # SO8T回転行列を取得
            if rotation_gate.use_cayley:
                rot_blocks = rotation_gate._cayley_rotation(rotation_gate.theta)
            else:
                rot_blocks = rotation_gate._exp_rotation(rotation_gate.theta)
            
            # ブロック対角回転行列を構築
            hidden_size = linear_layer.out_features
            R = self.build_block_diagonal_rotation(rot_blocks, hidden_size)
            
            # 右掛け吸収: W' = W @ R
            # linear_layer.weight.shape = [out_features, in_features]
            # R.shape = [in_features, in_features] (in_features == out_features の場合)
            
            if linear_layer.in_features == hidden_size:
                # 標準的なケース: W @ R
                W_new = torch.mm(W_original, R)
            else:
                # 入出力サイズが異なる場合（エラーチェック）
                raise ValueError(
                    f"Cannot burn in rotation: in_features ({linear_layer.in_features}) "
                    f"!= hidden_size ({hidden_size})"
                )
            
            # 重みを更新
            linear_layer.weight.data = W_new
            
            # 誤差検証
            # テスト入力を生成
            test_input = torch.randn(1, 16, hidden_size, device=W_original.device, dtype=W_original.dtype)
            
            # 焼き込み前の出力: linear(rotation_gate(x))
            with torch.enable_grad():
                rotated_input = rotation_gate(test_input)
                output_before = nn.functional.linear(rotated_input, W_original, linear_layer.bias)
            
            # 焼き込み後の出力: linear_burned(x)
            with torch.enable_grad():
                output_after = nn.functional.linear(test_input, W_new, linear_layer.bias)
            
            # 誤差計算
            max_error = torch.max(torch.abs(output_before - output_after)).item()
            mean_error = torch.mean(torch.abs(output_before - output_after)).item()
            
            metrics = {
                'layer_name': layer_name,
                'max_error': max_error,
                'mean_error': mean_error,
                'success': max_error < self.verify_threshold,
            }
            
            self.burn_in_log.append(metrics)
            
            logger.info(
                f"[BURN_IN] {layer_name}: "
                f"max_error={max_error:.2e}, mean_error={mean_error:.2e}, "
                f"success={metrics['success']}"
            )
            
            return metrics
    
    def burn_in_model(
        self,
        model: nn.Module,
        rotation_gate_mapping: Dict[str, any],  # layer_name -> SO8TRotationGate
        target_layer_pattern: str = "o_proj",
    ) -> Dict[str, any]:
        """
        モデル全体にSO8T焼き込みを適用
        
        Args:
            model: 対象モデル
            rotation_gate_mapping: レイヤー名 -> SO8TRotationGate のマッピング
            target_layer_pattern: 焼き込み対象レイヤーのパターン
        
        Returns:
            summary: 焼き込み結果サマリー
        """
        total_layers = 0
        successful_layers = 0
        failed_layers = []
        
        for name, module in model.named_modules():
            # 対象レイヤーのみ処理
            if target_layer_pattern in name and isinstance(module, nn.Linear):
                # 対応するSO8TRotationGateを取得
                rotation_gate = rotation_gate_mapping.get(name)
                
                if rotation_gate is None:
                    logger.warning(f"[BURN_IN] No rotation gate found for {name}, skipping")
                    continue
                
                # 焼き込み実行
                try:
                    metrics = self.burn_in_linear_layer(module, rotation_gate, name)
                    total_layers += 1
                    
                    if metrics['success']:
                        successful_layers += 1
                    else:
                        failed_layers.append(name)
                        logger.warning(
                            f"[BURN_IN] Failed verification for {name}: "
                            f"max_error={metrics['max_error']:.2e}"
                        )
                
                except Exception as e:
                    logger.error(f"[BURN_IN] Error burning in {name}: {e}")
                    failed_layers.append(name)
        
        summary = {
            'total_layers': total_layers,
            'successful_layers': successful_layers,
            'failed_layers': failed_layers,
            'success_rate': successful_layers / total_layers if total_layers > 0 else 0.0,
            'log': self.burn_in_log,
        }
        
        logger.info(
            f"[BURN_IN] Summary: {successful_layers}/{total_layers} layers successful "
            f"({summary['success_rate']*100:.1f}%)"
        )
        
        return summary
    
    def remove_rotation_gates(
        self,
        model: nn.Module,
        rotation_gate_attr: str = "so8t_gate",
    ) -> int:
        """
        焼き込み後、SO8TRotationGateを削除
        
        Args:
            model: 対象モデル
            rotation_gate_attr: SO8TRotationGateの属性名
        
        Returns:
            removed_count: 削除されたゲート数
        """
        removed_count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, rotation_gate_attr):
                delattr(module, rotation_gate_attr)
                removed_count += 1
                logger.info(f"[BURN_IN] Removed rotation gate from {name}")
        
        logger.info(f"[BURN_IN] Removed {removed_count} rotation gates")
        
        return removed_count
    
    def save_burn_in_report(
        self,
        output_path: Path,
        summary: Dict[str, any],
    ):
        """
        焼き込みレポートを保存
        
        Args:
            output_path: 出力パス
            summary: 焼き込み結果サマリー
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[BURN_IN] Report saved to {output_path}")


def burn_in_phi4_model(
    model: nn.Module,
    save_path: Path,
    verify_threshold: float = 1e-5,
) -> Dict[str, any]:
    """
    Phi-4モデルにSO8T焼き込みを適用
    
    Args:
        model: Phi-4 + SO8Tモデル
        save_path: 焼き込み済みモデルの保存先
        verify_threshold: 検証誤差の閾値
    
    Returns:
        summary: 焼き込み結果サマリー
    """
    manager = BurnInManager(verify_error_threshold=verify_threshold)
    
    # SO8TRotationGateのマッピングを構築
    rotation_gate_mapping = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'so8t_gate'):
            # o_projレイヤーに対応するSO8TRotationGate
            rotation_gate_mapping[name] = module.so8t_gate
    
    logger.info(f"[BURN_IN] Found {len(rotation_gate_mapping)} rotation gates")
    
    # 焼き込み実行
    summary = manager.burn_in_model(
        model=model,
        rotation_gate_mapping=rotation_gate_mapping,
        target_layer_pattern="o_proj",
    )
    
    # 回転ゲートを削除
    removed_count = manager.remove_rotation_gates(model)
    summary['removed_gates'] = removed_count
    
    # モデルを保存
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_path)
    logger.info(f"[BURN_IN] Model saved to {save_path}")
    
    # レポート保存
    manager.save_burn_in_report(
        output_path=save_path / "burn_in_report.json",
        summary=summary,
    )
    
    return summary


