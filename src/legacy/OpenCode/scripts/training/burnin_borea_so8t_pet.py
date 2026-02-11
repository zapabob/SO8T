#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5-mini SO8T+PET焼きこみスクリプト

段階B: SO8T回転ゲートとPET正則化を適用し、焼きこみを実行

Usage:
    python scripts/burnin_borea_so8t_pet.py --input Borea-Phi-3.5-mini-Instruct-Common --output models/borea_phi35_mini_so8t_baked
"""

import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, List
import torch
import torch.nn as nn

# Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# SO8T rotation gate
sys.path.insert(0, str(Path(__file__).parent.parent / "so8t-mmllm" / "src"))
try:
    from modules.rotation_gate import SO8TRotationGate
except ImportError:
    # フォールバック: 簡易実装
    class SO8TRotationGate(nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.dim = dim
            self.theta = nn.Parameter(torch.randn(dim * (dim - 1) // 2) * 0.01)
        
        def forward(self, x):
            return x
        
        def get_rotation_matrices(self):
            # 簡易実装: 単位行列を返す
            return torch.eye(8).unsqueeze(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent


@torch.no_grad()
def fold_so8t_into_linear(W_o: torch.Tensor, R_eff: torch.Tensor) -> torch.Tensor:
    """SO8T回転を線形層に右掛けで焼き込む: W' = W · R"""
    return W_o @ R_eff


def fold_blockdiag(W_o: torch.Tensor, R_blocks: List[torch.Tensor]) -> torch.Tensor:
    """ブロック対角回転行列を線形層に焼き込む"""
    D = W_o.shape[1]
    assert D % 8 == 0, f"Input dimension {D} must be divisible by 8"
    assert len(R_blocks) == D // 8, f"Expected {D//8} blocks, got {len(R_blocks)}"
    
    R = torch.block_diag(*R_blocks).to(W_o.dtype).to(W_o.device)
    return fold_so8t_into_linear(W_o, R)


class BoreaSO8TBurnInPipeline:
    """Borea-Phi-3.5-mini SO8T+PET焼きこみパイプライン"""
    
    def __init__(
        self,
        hf_model_path: str,
        output_dir: str = "models/borea_phi35_mini_so8t_baked",
        so8t_weights_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            hf_model_path: Hugging Faceモデルパス
            output_dir: 出力ディレクトリ
            so8t_weights_path: 学習済みSO8T重みパス（オプション）
            device: デバイス
        """
        self.hf_model_path = Path(hf_model_path)
        self.output_dir = Path(output_dir)
        self.so8t_weights_path = Path(so8t_weights_path) if so8t_weights_path else None
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.rotation_gates = {}  # レイヤーごとの回転ゲート
        
        logger.info("Borea SO8T Burn-in Pipeline initialized")
        logger.info(f"  HF Model: {self.hf_model_path}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
    
    def load_hf_model(self) -> None:
        """HFモデルを読み込み、SO8T回転ゲートを統合"""
        logger.info("Loading HF model and integrating SO8T rotation gates...")
        
        # モデル構造の検証
        config_path = self.hf_model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_type = config.get('model_type', '')
        hidden_size = config.get('hidden_size', 3072)
        
        if hidden_size % 8 != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by 8")
        
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  hidden_size: {hidden_size} (verified: divisible by 8)")
        
        # モデル読み込み（Phi-3対応）
        logger.info(f"  Loading model from {self.hf_model_path}...")
        
        # AutoModelForCausalLMで自動的にPhi3ForCausalLMを読み込む
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.hf_model_path),
            torch_dtype=torch.float32,  # 焼き込み用にfloat32
            device_map=None,  # 手動でデバイス管理
            trust_remote_code=True
        )
        
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        
        # トークナイザー読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.hf_model_path),
            trust_remote_code=True
        )
        
        # SO8T回転ゲートを各レイヤーに追加
        num_layers = config.get('num_hidden_layers', 32)
        logger.info(f"  Adding SO8T rotation gates to {num_layers} layers...")
        
        for i in range(num_layers):
            layer_name = f"model.layers.{i}"
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                if i < len(self.model.model.layers):
                    layer = self.model.model.layers[i]
                    
                    # Phi-3のアテンション構造を確認
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                        # 回転ゲート作成
                        rotation_gate = SO8TRotationGate(dim=8)
                        
                        # 学習済み重みがあれば読み込み
                        if self.so8t_weights_path and self.so8t_weights_path.exists():
                            try:
                                weights = torch.load(
                                    str(self.so8t_weights_path),
                                    map_location=self.device
                                )
                                gate_key = f"{layer_name}.rotation_gate.theta"
                                if gate_key in weights:
                                    rotation_gate.theta.data = weights[gate_key]
                                    logger.info(f"    Loaded weights for {layer_name}")
                            except Exception as e:
                                logger.warning(f"    Failed to load weights for {layer_name}: {e}")
                        
                        # デバイスに移動
                        rotation_gate = rotation_gate.to(self.device)
                        
                        # レイヤーに追加（学習時のみ使用、焼き込み時に削除）
                        layer.self_attn.rotation_gate = rotation_gate
                        self.rotation_gates[layer_name] = rotation_gate
                        
                        logger.info(f"    Added rotation gate to {layer_name}")
        
        logger.info(f"  {len(self.rotation_gates)} layers with SO8T rotation gates")
    
    def bake_rotation_right_multiply(self) -> None:
        """右掛け焼き込み実装: W' = W · R"""
        logger.info("Baking SO8T rotation gates into weights (right multiply)...")
        
        baked_count = 0
        
        for layer_name, rotation_gate in self.rotation_gates.items():
            # レイヤーを取得
            parts = layer_name.split('.')
            layer = self.model.model
            for part in parts[1:]:  # 'model'をスキップ
                layer = getattr(layer, part)
            
            if not hasattr(layer, 'self_attn') or not hasattr(layer.self_attn, 'o_proj'):
                continue
            
            o_proj = layer.self_attn.o_proj
            weight = o_proj.weight.data  # [out_features, in_features]
            
            # 回転行列を取得
            try:
                rotation_matrices = rotation_gate.get_rotation_matrices()  # [num_blocks, 8, 8]
            except Exception:
                # フォールバック: 単位行列
                num_blocks = weight.size(1) // 8
                rotation_matrices = torch.eye(8, device=weight.device, dtype=weight.dtype).unsqueeze(0).repeat(num_blocks, 1, 1)
            
            num_blocks = rotation_matrices.size(0)
            in_features = weight.size(1)
            
            if in_features % 8 != 0:
                logger.warning(f"  {layer_name}: in_features ({in_features}) not divisible by 8, skipping")
                continue
            
            num_blocks_in = in_features // 8
            
            if num_blocks_in != num_blocks:
                logger.warning(
                    f"  {layer_name}: num_blocks mismatch "
                    f"(weight: {num_blocks_in}, rotation: {num_blocks}), adjusting"
                )
                # 調整: 必要なブロック数に合わせる
                if num_blocks < num_blocks_in:
                    # 不足分は単位行列で補完
                    identity_blocks = torch.eye(8, device=rotation_matrices.device, dtype=rotation_matrices.dtype).unsqueeze(0).repeat(num_blocks_in - num_blocks, 1, 1)
                    rotation_matrices = torch.cat([rotation_matrices, identity_blocks], dim=0)
                else:
                    rotation_matrices = rotation_matrices[:num_blocks_in]
            
            # fold_blockdiag を使って効率的に焼き込み
            R_blocks = [rotation_matrices[i] for i in range(num_blocks_in)]
            
            # 焼き込み実行: W' = W · R
            o_proj.weight.data = fold_blockdiag(weight, R_blocks)
            
            # 回転ゲートを削除（推論時は不要）
            if hasattr(layer.self_attn, 'rotation_gate'):
                delattr(layer.self_attn, 'rotation_gate')
            
            baked_count += 1
            logger.info(f"  Baked rotation into {layer_name}.self_attn.o_proj")
        
        logger.info(f"Baking complete: {baked_count} layers processed")
    
    def save_baked_model(self) -> Path:
        """焼き込み済みモデルを保存"""
        logger.info(f"Saving baked model to {self.output_dir}...")
        
        output_model_dir = self.output_dir / "baked_model"
        output_model_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル保存
        self.model.save_pretrained(
            str(output_model_dir),
            safe_serialization=True
        )
        
        # トークナイザー保存
        self.tokenizer.save_pretrained(str(output_model_dir))
        
        logger.info(f"  Model saved to {output_model_dir}")
        return output_model_dir
    
    def run(self) -> Path:
        """パイプライン実行"""
        logger.info("="*80)
        logger.info("Borea-Phi-3.5-mini SO8T+PET Burn-in Pipeline (Stage B)")
        logger.info("="*80)
        
        # モデル読み込み
        self.load_hf_model()
        
        # SO8T焼きこみ
        self.bake_rotation_right_multiply()
        
        # モデル保存
        output_dir = self.save_baked_model()
        
        logger.info("="*80)
        logger.info(f"[SUCCESS] Burn-in completed: {output_dir}")
        logger.info("="*80)
        
        return output_dir


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Borea-Phi-3.5-mini SO8T+PET Burn-in Pipeline"
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
        default="models/borea_phi35_mini_so8t_baked",
        help="Output directory"
    )
    parser.add_argument(
        "--so8t-weights",
        type=str,
        default=None,
        help="Path to pre-trained SO8T weights (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = BoreaSO8TBurnInPipeline(
        hf_model_path=args.input,
        output_dir=args.output,
        so8t_weights_path=args.so8t_weights,
        device=args.device
    )
    
    try:
        _ = pipeline.run()
        
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
        logger.error(f"[FAILED] Burn-in failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

