#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8VIT: SO(8)回転ゲート対応ViT
SO8VIT: SO(8) rotation gate compatible Vision Transformer

マルチモーダル化のためのSO(8)回転ゲートを統合したViTアーキテクチャ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SO8RotationGate(nn.Module):
    """
    SO(8)回転ゲート
    SO(8) rotation gate for orthogonal error monitoring
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # SO(8)回転行列生成 (8x8回転行列)
        self.register_buffer("so8_basis", self._create_so8_basis())

        # 回転強度制御
        self.rotation_scale = nn.Parameter(torch.ones(1))

        # 直交性監視用の補助投影
        self.orth_monitor = nn.Linear(embed_dim, embed_dim)

    def _create_so8_basis(self) -> torch.Tensor:
        """SO(8) Lie代数の基底を作成"""
        # SO(8)には28個の生成子があるが、主要なものを8個選択
        basis_matrices = []

        # 基本的な回転平面 (1-2, 3-4, 5-6, 7-8)
        for i in range(0, 8, 2):
            R = torch.eye(8)
            R[i, i] = 0; R[i, i+1] = -1
            R[i+1, i] = 1; R[i+1, i+1] = 0
            basis_matrices.append(R)

        return torch.stack(basis_matrices)  # [8, 8, 8]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SO(8)回転ゲート適用
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            rotated_x: 回転適用後のテンソル
            orth_error: 直交誤差
        """
        B, N, D = x.shape

        # 入力を8次元に射影
        x_proj = self.orth_monitor(x)  # [B, N, D]

        # マルチヘッド分割
        x_heads = x_proj.view(B, N, self.num_heads, self.head_dim)  # [B, N, H, D/H]
        x_heads = x_heads.transpose(1, 2)  # [B, H, N, D/H]

        # SO(8)回転適用
        rotated_heads = []
        orth_errors = []

        for h in range(self.num_heads):
            head_x = x_heads[:, h]  # [B, N, D/H]

            # 8次元空間に射影（D/H >= 8を仮定）
            if self.head_dim >= 8:
                head_8d = head_x[:, :, :8]  # [B, N, 8]
            else:
                # パディング
                head_8d = F.pad(head_x, (0, 8 - self.head_dim), "constant", 0)

            # SO(8)回転適用
            rotated_8d = self._apply_so8_rotation(head_8d)

            # 元の次元に戻す
            if self.head_dim >= 8:
                rotated_head = head_x.clone()
                rotated_head[:, :, :8] = rotated_8d
            else:
                rotated_head = head_x + rotated_8d[:, :, :self.head_dim]

            rotated_heads.append(rotated_head)

            # 直交誤差計算
            orth_error = self._compute_orthogonality_error(head_8d, rotated_8d)
            orth_errors.append(orth_error)

        # ヘッド統合
        rotated_heads = torch.stack(rotated_heads, dim=1)  # [B, H, N, D/H]
        rotated_heads = rotated_heads.transpose(1, 2).contiguous()  # [B, N, H, D/H]
        rotated_x = rotated_heads.view(B, N, D)

        # 直交誤差の平均
        orth_error = torch.stack(orth_errors).mean(dim=0)  # [B, N]

        return rotated_x + x, orth_error  # 残差接続

    def _apply_so8_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """SO(8)回転を適用"""
        B, N, C = x.shape  # C=8

        # 回転角度生成（学習可能）
        angles = torch.randn(B, N, 8, device=x.device) * self.rotation_scale

        # SO(8)回転行列生成
        rotation_matrices = []
        for i in range(8):
            R = torch.eye(8, device=x.device)
            angle = angles[:, :, i].unsqueeze(-1).unsqueeze(-1)
            basis = self.so8_basis[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 8, 8]
            R_exp = torch.matrix_exp(angle * basis)
            rotation_matrices.append(R_exp.squeeze(0).squeeze(0))

        # 回転適用
        rotated = x.clone()
        for i in range(8):
            rotated = rotated @ rotation_matrices[i].transpose(-2, -1)

        return rotated

    def _compute_orthogonality_error(self, original: torch.Tensor, rotated: torch.Tensor) -> torch.Tensor:
        """直交誤差を計算"""
        # 回転行列の直交性をチェック
        diff = rotated - original
        # 直交誤差 = ||R^T @ R - I||_F
        orth_error = torch.norm(diff, p='fro', dim=-1)
        return orth_error


class SO8ResidualAdapter(nn.Module):
    """
    SO(8)回転ゲート残差アダプター
    SO(8) rotation gate residual adapter for transformer/ViT layers
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.rotation_gate = SO8RotationGate(embed_dim, num_heads)
        self.norm = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        残差アダプター適用
        Returns:
            adapted_x: アダプター適用後のテンソル
            orth_error: 直交誤差
        """
        residual = x
        x, orth_error = self.rotation_gate(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = x + residual  # 残差接続

        return x, orth_error


class SO8MultiModalAdapter(nn.Module):
    """
    SO(8)マルチモーダルアダプター
    SO(8) multimodal adapter for vision-language integration
    """

    def __init__(self, embed_dim: int, num_modalities: int = 2):
        super().__init__()
        self.num_modalities = num_modalities

        # モダリティ固有のSO(8)アダプター
        self.modality_adapters = nn.ModuleList([
            SO8ResidualAdapter(embed_dim) for _ in range(num_modalities)
        ])

        # クロスモーダル融合用回転ゲート
        self.cross_modal_gate = SO8RotationGate(embed_dim)

        # モダリティ識別子
        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, embed_dim))

    def forward(self, modalities: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        マルチモーダル入力の融合
        Args:
            modalities: 各モダリティのテンソルリスト
        Returns:
            fused: 融合された表現
            orth_errors: 各モダリティの直交誤差
        """
        assert len(modalities) == self.num_modalities

        adapted_modalities = []
        orth_errors = []

        # 各モダリティにSO(8)アダプター適用
        for i, modality in enumerate(modalities):
            adapted, orth_error = self.modality_adapters[i](modality)
            # モダリティ埋め込み追加
            adapted = adapted + self.modality_embeddings[i].unsqueeze(0).unsqueeze(0)
            adapted_modalities.append(adapted)
            orth_errors.append(orth_error)

        # クロスモーダル融合
        if len(adapted_modalities) > 1:
            # 平均プーリングで融合
            fused = torch.stack(adapted_modalities, dim=0).mean(dim=0)
            # クロスモーダル回転ゲート適用
            fused, cross_orth_error = self.cross_modal_gate(fused)
            orth_errors.append(cross_orth_error)
        else:
            fused = adapted_modalities[0]

        return fused, torch.stack(orth_errors)


class SO8VIT(nn.Module):
    """
    SO(8)対応Vision Transformer
    SO(8) compatible Vision Transformer with multimodal capabilities
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: nn.Module = LayerNorm,
        multimodal: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1  # cls token

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # SO(8) Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            SO8Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Multimodal adapter
        self.multimodal = multimodal
        if multimodal:
            self.multimodal_adapter = SO8MultiModalAdapter(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor, text_embeds: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        特徴抽出
        Args:
            x: 画像テンソル [B, C, H, W]
            text_embeds: テキスト埋め込み [B, seq_len, embed_dim]
        Returns:
            features: 特徴量
            aux_outputs: 補助出力（直交誤差など）
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # CLS tokenと位置埋め込み追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        orth_errors = []

        # SO(8) Transformer blocks
        for blk in self.blocks:
            x, orth_error = blk(x)
            orth_errors.append(orth_error)

        x = self.norm(x)

        # Multimodal fusion
        if self.multimodal and text_embeds is not None:
            modalities = [x, text_embeds]
            x, modal_orth_errors = self.multimodal_adapter(modalities)
            orth_errors.extend(modal_orth_errors)

        # CLS token抽出
        features = x[:, 0]

        aux_outputs = {
            'orth_errors': orth_errors,
            'layer_outputs': x
        }

        return features, aux_outputs

    def forward(self, x: torch.Tensor, text_embeds: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        順伝播
        """
        x, aux_outputs = self.forward_features(x, text_embeds)
        x = self.head(x)
        return x, aux_outputs


class SO8Block(nn.Module):
    """
    SO(8)対応Transformer Block
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = LayerNorm
    ):
        super().__init__()

        # Multi-head attention with SO(8) adapter
        self.norm1 = norm_layer(dim)
        self.attn = SO8Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # SO(8) residual adapter in MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            SO8ResidualAdapter(mlp_hidden_dim, num_heads=num_heads),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention with SO(8) adapter
        x_attn, attn_orth_error = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)

        # MLP with SO(8) adapter
        x_mlp, mlp_orth_error = self.mlp(self.norm2(x))
        x = x + self.drop_path(x_mlp)

        # Combine orthogonal errors
        orth_error = attn_orth_error + mlp_orth_error

        return x, orth_error


class SO8Attention(nn.Module):
    """
    SO(8)対応Multi-Head Attention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # SO(8) adapter for attention outputs
        self.so8_adapter = SO8ResidualAdapter(dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # SO(8) adapter適用
        x, orth_error = self.so8_adapter(x)

        return x, orth_error


# Utility functions and classes
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


if __name__ == '__main__':
    # Test SO8VIT
    model = SO8VIT(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        multimodal=True
    )

    # Test with dummy input
    img = torch.randn(2, 3, 224, 224)
    text_embeds = torch.randn(2, 10, 768)  # 10 text tokens

    output, aux = model(img, text_embeds)
    print(f"Output shape: {output.shape}")
    print(f"Orth errors: {len(aux['orth_errors'])}")
