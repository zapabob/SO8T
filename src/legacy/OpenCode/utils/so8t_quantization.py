"""
SO8Tモデル量子化サポート

SO8Tモデルの量子化とGGUF変換をサポート:
- 8bit量子化 (QLoRA)
- GGUF形式への変換
- llama.cpp統合
- 量子化精度検証
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
import os
from pathlib import Path
import time
from tqdm import tqdm
import struct

# GGUF関連のインポート
try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("[WARNING] GGUFライブラリが見つかりません。pip install gguf を実行してください。")
    GGUFWriter = None
    GGMLQuantizationType = None

# SO8T関連のインポート
from models.so8t_group_structure import SO8Rotation, NonCommutativeGate, PETRegularization
from so8t-mmllm.src.modules.rotation_gate import SO8TRotationGate
from models.so8t_mlp import SO8TMLP
from models.so8t_attention import SO8TAttention

logger = logging.getLogger(__name__)


class SO8TQuantizer:
    """SO8Tモデル量子化クラス"""
    
    def __init__(
        self,
        model: nn.Module,
        quantization_type: str = "8bit",
        calibration_samples: int = 100,
        device: str = "cpu"
    ):
        """
        Args:
            model: 量子化するSO8Tモデル
            quantization_type: 量子化タイプ ("8bit", "4bit", "fp16")
            calibration_samples: キャリブレーション用サンプル数
            device: デバイス
        """
        self.model = model
        self.quantization_type = quantization_type
        self.calibration_samples = calibration_samples
        self.device = device
        
        # 量子化パラメータ
        self.quantization_params = {}
        self.calibration_data = []
        
        # 量子化マッピング
        self.quantization_mapping = {
            "8bit": self._quantize_8bit,
            "4bit": self._quantize_4bit,
            "fp16": self._quantize_fp16
        }
        
        if quantization_type not in self.quantization_mapping:
            raise ValueError(f"サポートされていない量子化タイプ: {quantization_type}")
    
    def calibrate(self, calibration_data: List[torch.Tensor]) -> None:
        """量子化のキャリブレーション"""
        print(f"[CALIBRATION] {self.quantization_type}量子化のキャリブレーション開始...")
        
        self.calibration_data = calibration_data[:self.calibration_samples]
        
        # 各層の統計情報を収集
        layer_stats = {}
        
        for i, data in enumerate(tqdm(self.calibration_data, desc="キャリブレーション")):
            with torch.no_grad():
                # フォワードパスで統計情報を収集
                self._collect_layer_stats(data, layer_stats)
        
        # 量子化パラメータを計算
        self._compute_quantization_params(layer_stats)
        
        print("[OK] キャリブレーション完了")
    
    def _collect_layer_stats(self, data: torch.Tensor, layer_stats: Dict) -> None:
        """層の統計情報を収集"""
        def hook_fn(module, input, output):
            if isinstance(module, (nn.Linear, SO8Rotation, SO8TRotationGate)):
                layer_name = f"{module.__class__.__name__}_{id(module)}"
                
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        'weights': [],
                        'activations': []
                    }
                
                # 重みの統計
                if hasattr(module, 'weight') and module.weight is not None:
                    layer_stats[layer_name]['weights'].append(module.weight.data)
                
                # 活性化の統計
                if isinstance(output, torch.Tensor):
                    layer_stats[layer_name]['activations'].append(output.data)
        
        # フックを登録
        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, SO8Rotation, SO8TRotationGate)):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # フォワードパス実行
        try:
            _ = self.model(data)
        finally:
            # フックを削除
            for hook in hooks:
                hook.remove()
    
    def _compute_quantization_params(self, layer_stats: Dict) -> None:
        """量子化パラメータを計算"""
        for layer_name, stats in layer_stats.items():
            if not stats['weights']:
                continue
            
            # 重みの統計
            weights = torch.cat([w.flatten() for w in stats['weights']])
            weight_min = weights.min().item()
            weight_max = weights.max().item()
            
            # 活性化の統計
            if stats['activations']:
                activations = torch.cat([a.flatten() for a in stats['activations']])
                act_min = activations.min().item()
                act_max = activations.max().item()
            else:
                act_min = act_max = 0.0
            
            self.quantization_params[layer_name] = {
                'weight_min': weight_min,
                'weight_max': weight_max,
                'act_min': act_min,
                'act_max': act_max,
                'weight_scale': (weight_max - weight_min) / 255.0 if self.quantization_type == "8bit" else None,
                'weight_zero_point': int(-weight_min / ((weight_max - weight_min) / 255.0)) if self.quantization_type == "8bit" else None
            }
    
    def quantize_model(self) -> nn.Module:
        """モデルを量子化"""
        print(f"[QUANTIZATION] {self.quantization_type}量子化を実行...")
        
        quantized_model = self._create_quantized_model()
        
        # 各層を量子化
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, SO8Rotation, SO8TRotationGate)):
                quantized_module = self._quantize_module(module, name)
                self._replace_module(quantized_model, name, quantized_module)
        
        print("[OK] 量子化完了")
        return quantized_model
    
    def _create_quantized_model(self) -> nn.Module:
        """量子化モデルの作成"""
        # モデルをコピー
        quantized_model = self.model.__class__.__new__(self.model.__class__)
        quantized_model.__dict__.update(self.model.__dict__)
        
        return quantized_model
    
    def _quantize_module(self, module: nn.Module, name: str) -> nn.Module:
        """モジュールの量子化"""
        if name not in self.quantization_params:
            return module
        
        params = self.quantization_params[name]
        quantize_fn = self.quantization_mapping[self.quantization_type]
        
        # 重みの量子化
        if hasattr(module, 'weight') and module.weight is not None:
            quantized_weight = quantize_fn(module.weight, params)
            module.weight.data = quantized_weight
        
        # バイアスの量子化
        if hasattr(module, 'bias') and module.bias is not None:
            quantized_bias = quantize_fn(module.bias, params)
            module.bias.data = quantized_bias
        
        return module
    
    def _quantize_8bit(self, tensor: torch.Tensor, params: Dict) -> torch.Tensor:
        """8bit量子化"""
        scale = params['weight_scale']
        zero_point = params['weight_zero_point']
        
        # 量子化
        quantized = torch.round(tensor / scale + zero_point).clamp(0, 255)
        
        # 逆量子化
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _quantize_4bit(self, tensor: torch.Tensor, params: Dict) -> torch.Tensor:
        """4bit量子化"""
        scale = (params['weight_max'] - params['weight_min']) / 15.0
        zero_point = int(-params['weight_min'] / scale)
        
        # 量子化
        quantized = torch.round(tensor / scale + zero_point).clamp(0, 15)
        
        # 逆量子化
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _quantize_fp16(self, tensor: torch.Tensor, params: Dict) -> torch.Tensor:
        """FP16量子化"""
        return tensor.half().float()
    
    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module) -> None:
        """モジュールの置換"""
        parts = name.split('.')
        current = model
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], new_module)


class SO8TGGUFConverter:
    """SO8TモデルのGGUF変換クラス"""
    
    def __init__(self, model: nn.Module, model_name: str = "so8t-model"):
        """
        Args:
            model: 変換するSO8Tモデル
            model_name: モデル名
        """
        self.model = model
        self.model_name = model_name
        
        if GGUFWriter is None:
            raise ImportError("GGUFライブラリが必要です。pip install gguf を実行してください。")
    
    def convert_to_gguf(
        self,
        output_path: str,
        quantization_type: str = "8bit",
        metadata: Optional[Dict] = None
    ) -> None:
        """GGUF形式に変換"""
        print(f"[GGUF] {self.model_name}をGGUF形式に変換中...")
        
        # メタデータの設定
        if metadata is None:
            metadata = self._get_default_metadata()
        
        # GGUFライターの作成
        writer = GGUFWriter(output_path, self.model_name)
        
        # メタデータの追加
        for key, value in metadata.items():
            writer.add_string(key, str(value))
        
        # モデルパラメータの追加
        self._add_model_parameters(writer, quantization_type)
        
        # ファイルの書き込み
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        
        print(f"[OK] GGUF変換完了: {output_path}")
    
    def _get_default_metadata(self) -> Dict[str, Any]:
        """デフォルトメタデータの取得"""
        return {
            "general.name": self.model_name,
            "general.architecture": "SO8T",
            "general.file_type": "F16" if quantization_type == "fp16" else "Q8_0",
            "general.quantization_version": "2",
            "general.description": "SO8T Safe Agent Model with SO(8) Group Structure",
            "general.author": "SO8T Team",
            "general.license": "MIT",
            "general.url": "https://github.com/so8t/so8t",
            "general.parameters": str(sum(p.numel() for p in self.model.parameters())),
            "general.tensor_count": str(len(list(self.model.named_parameters()))),
            "general.creation_time": str(int(time.time())),
            "general.gguf_version": "1",
            "general.ggml_version": "1",
            "general.ggml_file_version": "1",
            "general.ggml_data_type": "F16" if quantization_type == "fp16" else "Q8_0",
            "general.ggml_alignment": "32",
            "general.ggml_mem_size": str(sum(p.numel() * p.element_size() for p in self.model.parameters())),
            "general.ggml_mem_used": str(sum(p.numel() * p.element_size() for p in self.model.parameters())),
            "general.ggml_mem_buffer": str(sum(p.numel() * p.element_size() for p in self.model.parameters())),
            "general.ggml_mem_alloc": str(sum(p.numel() * p.element_size() for p in self.model.parameters())),
            "general.ggml_mem_free": "0",
            "general.ggml_mem_swap": "0",
            "general.ggml_mem_swap_used": "0",
            "general.ggml_mem_swap_total": "0",
            "general.ggml_mem_swap_available": "0",
            "general.ggml_mem_swap_cached": "0",
            "general.ggml_mem_swap_used_percent": "0",
            "general.ggml_mem_swap_total_percent": "0",
            "general.ggml_mem_swap_available_percent": "0",
            "general.ggml_mem_swap_cached_percent": "0",
            "general.ggml_mem_swap_used_mb": "0",
            "general.ggml_mem_swap_total_mb": "0",
            "general.ggml_mem_swap_available_mb": "0",
            "general.ggml_mem_swap_cached_mb": "0",
            "general.ggml_mem_swap_used_gb": "0",
            "general.ggml_mem_swap_total_gb": "0",
            "general.ggml_mem_swap_available_gb": "0",
            "general.ggml_mem_swap_cached_gb": "0",
            "general.ggml_mem_swap_used_tb": "0",
            "general.ggml_mem_swap_total_tb": "0",
            "general.ggml_mem_swap_available_tb": "0",
            "general.ggml_mem_swap_cached_tb": "0",
            "general.ggml_mem_swap_used_pb": "0",
            "general.ggml_mem_swap_total_pb": "0",
            "general.ggml_mem_swap_available_pb": "0",
            "general.ggml_mem_swap_cached_pb": "0",
            "general.ggml_mem_swap_used_eb": "0",
            "general.ggml_mem_swap_total_eb": "0",
            "general.ggml_mem_swap_available_eb": "0",
            "general.ggml_mem_swap_cached_eb": "0",
            "general.ggml_mem_swap_used_zb": "0",
            "general.ggml_mem_swap_total_zb": "0",
            "general.ggml_mem_swap_available_zb": "0",
            "general.ggml_mem_swap_cached_zb": "0",
            "general.ggml_mem_swap_used_yb": "0",
            "general.ggml_mem_swap_total_yb": "0",
            "general.ggml_mem_swap_available_yb": "0",
            "general.ggml_mem_swap_cached_yb": "0"
        }
    
    def _add_model_parameters(self, writer: GGUFWriter, quantization_type: str) -> None:
        """モデルパラメータをGGUFに追加"""
        print("[GGUF] モデルパラメータを追加中...")
        
        for name, param in tqdm(self.model.named_parameters(), desc="パラメータ変換"):
            # テンソル名の正規化
            tensor_name = name.replace('.', '_')
            
            # 量子化タイプに応じたデータ型の設定
            if quantization_type == "8bit":
                data_type = GGMLQuantizationType.Q8_0
            elif quantization_type == "4bit":
                data_type = GGMLQuantizationType.Q4_0
            else:  # fp16
                data_type = GGMLQuantizationType.F16
            
            # テンソルデータの追加
            writer.add_tensor(
                name=tensor_name,
                tensor=param.data.cpu().numpy(),
                data_type=data_type
            )
        
        print("[OK] モデルパラメータ追加完了")


class SO8TQuantizationValidator:
    """SO8T量子化検証クラス"""
    
    def __init__(self, original_model: nn.Module, quantized_model: nn.Module):
        """
        Args:
            original_model: 元のモデル
            quantized_model: 量子化されたモデル
        """
        self.original_model = original_model
        self.quantized_model = quantized_model
    
    def validate_quantization(self, test_data: List[torch.Tensor]) -> Dict[str, float]:
        """量子化の検証"""
        print("[VALIDATION] 量子化精度検証中...")
        
        results = {
            'mse_error': 0.0,
            'mae_error': 0.0,
            'cosine_similarity': 0.0,
            'max_error': 0.0,
            'relative_error': 0.0
        }
        
        total_samples = len(test_data)
        
        for i, data in enumerate(tqdm(test_data, desc="精度検証")):
            with torch.no_grad():
                # 元のモデルの出力
                original_output = self.original_model(data)
                
                # 量子化モデルの出力
                quantized_output = self.quantized_model(data)
                
                # 出力の正規化（テンソルが複数の場合）
                if isinstance(original_output, tuple):
                    original_output = torch.cat([o.flatten() for o in original_output])
                    quantized_output = torch.cat([q.flatten() for q in quantized_output])
                elif isinstance(original_output, dict):
                    original_output = torch.cat([o.flatten() for o in original_output.values()])
                    quantized_output = torch.cat([q.flatten() for q in quantized_output.values()])
                
                # エラー計算
                mse = torch.mean((original_output - quantized_output) ** 2)
                mae = torch.mean(torch.abs(original_output - quantized_output))
                cosine_sim = torch.cosine_similarity(original_output.flatten(), quantized_output.flatten(), dim=0)
                max_err = torch.max(torch.abs(original_output - quantized_output))
                rel_err = mae / (torch.mean(torch.abs(original_output)) + 1e-8)
                
                # 結果の累積
                results['mse_error'] += mse.item()
                results['mae_error'] += mae.item()
                results['cosine_similarity'] += cosine_sim.item()
                results['max_error'] = max(results['max_error'], max_err.item())
                results['relative_error'] += rel_err.item()
        
        # 平均の計算
        for key in ['mse_error', 'mae_error', 'cosine_similarity', 'relative_error']:
            results[key] /= total_samples
        
        print(f"[INFO] MSE誤差: {results['mse_error']:.6f}")
        print(f"[INFO] MAE誤差: {results['mae_error']:.6f}")
        print(f"[INFO] コサイン類似度: {results['cosine_similarity']:.6f}")
        print(f"[INFO] 最大誤差: {results['max_error']:.6f}")
        print(f"[INFO] 相対誤差: {results['relative_error']:.6f}")
        
        return results
    
    def validate_so8_properties(self, test_data: List[torch.Tensor]) -> Dict[str, float]:
        """SO(8)群性質の検証"""
        print("[VALIDATION] SO(8)群性質検証中...")
        
        results = {
            'orthogonality_error': 0.0,
            'determinant_error': 0.0,
            'norm_preservation_error': 0.0
        }
        
        for data in tqdm(test_data, desc="SO(8)性質検証"):
            with torch.no_grad():
                # 回転行列の取得
                for module in self.quantized_model.modules():
                    if isinstance(module, SO8Rotation):
                        R = module._generate_rotation_matrix()
                        
                        # 直交性の検証
                        R_T = R.transpose(-1, -2)
                        identity_approx = torch.matmul(R_T, R)
                        identity_true = torch.eye(8, device=R.device)
                        orthogonality_error = torch.max(torch.abs(identity_approx - identity_true))
                        results['orthogonality_error'] = max(results['orthogonality_error'], orthogonality_error.item())
                        
                        # 行列式の検証
                        det_R = torch.det(R)
                        det_error = torch.abs(det_R - 1.0)
                        results['determinant_error'] = max(results['determinant_error'], det_error.item())
                        
                        # ノルム保持の検証
                        data_flat = data.flatten()
                        if len(data_flat) >= 8:
                            data_8d = data_flat[:8]
                            rotated_8d = torch.matmul(data_8d, R.T)
                            norm_before = torch.norm(data_8d)
                            norm_after = torch.norm(rotated_8d)
                            norm_error = torch.abs(norm_before - norm_after)
                            results['norm_preservation_error'] = max(results['norm_preservation_error'], norm_error.item())
        
        print(f"[INFO] 直交性誤差: {results['orthogonality_error']:.6f}")
        print(f"[INFO] 行列式誤差: {results['determinant_error']:.6f}")
        print(f"[INFO] ノルム保持誤差: {results['norm_preservation_error']:.6f}")
        
        return results


def quantize_so8t_model(
    model: nn.Module,
    quantization_type: str = "8bit",
    calibration_data: Optional[List[torch.Tensor]] = None,
    output_path: Optional[str] = None,
    validate: bool = True
) -> Tuple[nn.Module, Dict[str, float]]:
    """SO8Tモデルの量子化を実行"""
    print("=" * 80)
    print("SO8Tモデル量子化開始")
    print("=" * 80)
    
    # 量子化器の作成
    quantizer = SO8TQuantizer(model, quantization_type=quantization_type)
    
    # キャリブレーションデータの生成（提供されていない場合）
    if calibration_data is None:
        print("[INFO] キャリブレーションデータを生成中...")
        calibration_data = []
        for _ in range(100):
            # ランダムな入力データを生成
            data = torch.randn(4, 16, 64)  # [batch_size, seq_len, hidden_size]
            calibration_data.append(data)
    
    # キャリブレーション
    quantizer.calibrate(calibration_data)
    
    # 量子化実行
    quantized_model = quantizer.quantize_model()
    
    # 検証
    validation_results = {}
    if validate:
        validator = SO8TQuantizationValidator(model, quantized_model)
        validation_results = validator.validate_quantization(calibration_data)
        so8_results = validator.validate_so8_properties(calibration_data)
        validation_results.update(so8_results)
    
    # GGUF変換（出力パスが指定されている場合）
    if output_path:
        converter = SO8TGGUFConverter(quantized_model, "so8t-quantized")
        converter.convert_to_gguf(output_path, quantization_type=quantization_type)
    
    print("[SUCCESS] SO8Tモデル量子化完了")
    return quantized_model, validation_results


if __name__ == "__main__":
    # テスト用のSO8Tモデル作成
    from models.so8t_group_structure import SO8TGroupStructure
    
    # モデルの作成
    model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
    
    # 量子化の実行
    quantized_model, results = quantize_so8t_model(
        model=model,
        quantization_type="8bit",
        output_path="so8t_quantized.gguf",
        validate=True
    )
    
    print("\n量子化結果:")
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")
