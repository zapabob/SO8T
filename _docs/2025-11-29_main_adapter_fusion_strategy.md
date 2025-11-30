# Adapter Fusion戦略 - Phase 1→2移行計画

## 実装情報
- **日付**: 2025-11-29
- **Worktree**: main
- **機能名**: adapter_fusion_strategy
- **実装者**: AI Agent

## Gemini提案に基づくAdapter Fusion戦略

### 戦略概要

**目標**: RTX 3060 (12GB VRAM) 制約下で、Phase 1 (テキストオンリー) の学習成果をPhase 2 (マルチモーダル) に継承する。

**Geminiの示唆**: 「Adapter Fusion一択」
- Phase 1: SO(8)アダプタ学習 (テキスト理解の結晶)
- Phase 2: 視覚アダプタのみ新規学習 (画像→SO(8)空間変換)

### アーキテクチャ設計

#### Phase 1 (現在): Textual Singularity
```
[Phi-3.5-mini Base Model] ← 凍結
    ↓
[SO(8) Text Adapter] ← 学習対象 (メイン)
    ↓
[四重推論 + PPO]
```

**学習内容**:
- SO(8)回転ゲートによるテキスト特徴量の幾何学的変換
- 四重推論 (Observation/Deduction/Abduction/Integration)
- PPO報酬最適化 (構造・同型性・安定性)

#### Phase 2 (未来): Multimodal Expansion
```
[Phi-3.5-mini Base Model] ← 凍結
    ↓
[SO(8) Text Adapter] ← Phase 1成果を凍結
    ↓
[Visual Projection Adapter] ← 新規学習 (最小VRAM)
    ↓
[Adapter Fusion Layer] ← 軽量統合
    ↓
[四重推論 + PPO]
```

**新規学習対象**:
- Visual Projection Adapterのみ
- SO(8) Text Adapterは凍結
- Base Modelは凍結

### 技術的実装詳細

#### 1. Visual Projection Adapter設計
```python
class VisualProjectionAdapter(nn.Module):
    """画像特徴量 → SO(8)空間射影アダプタ"""

    def __init__(self, vision_dim=768, so8_dim=4096, hidden_dim=1024):
        super().__init__()

        # 軽量な射影ネットワーク
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, so8_dim),
            nn.LayerNorm(so8_dim)
        )

        # SO(8)空間への適応
        self.so8_adapter = SO8RotationGate(
            hidden_size=so8_dim,
            num_rotations=8
        )

    def forward(self, vision_features):
        """画像特徴量をSO(8)空間に射影"""
        # 1. 次元射影
        projected = self.projection(vision_features)

        # 2. SO(8)幾何学的変換
        adapted = self.so8_adapter(projected)

        return adapted
```

#### 2. Adapter Fusion Layer
```python
class AdapterFusionLayer(nn.Module):
    """テキストアダプタと視覚アダプタの融合"""

    def __init__(self, so8_dim=4096, fusion_method="weighted_sum"):
        super().__init__()
        self.fusion_method = fusion_method

        if fusion_method == "weighted_sum":
            self.fusion_weights = nn.Parameter(torch.ones(2))  # [text_weight, vision_weight]
        elif fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=so8_dim,
                num_heads=8,
                batch_first=True
            )

    def forward(self, text_features, vision_features):
        """モーダル融合"""
        if self.fusion_method == "weighted_sum":
            # 重み付き和
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = weights[0] * text_features + weights[1] * vision_features

        elif self.fusion_method == "attention":
            # アテンション融合
            combined = torch.stack([text_features, vision_features], dim=1)
            fused, _ = self.fusion_attention(
                combined, combined, combined
            )
            fused = fused.mean(dim=1)  # 平均プーリング

        return fused
```

#### 3. メモリ最適化戦略

**VRAM消費削減**:
- Base Model: 凍結 (0 VRAM増加)
- SO(8) Text Adapter: 凍結 (0 VRAM増加)
- Visual Projection Adapter: 軽量 (~10Mパラメータ)
- Fusion Layer: 最小限 (~1Mパラメータ)

**総VRAM消費**: ~50MB (既存12GB + 微小増加)

### 学習戦略

#### Phase 2 学習フロー
1. **Vision Encoder準備**: SigLIP/SigLIP-2 または CLIP の軽量版
2. **SO(8) Text Adapter凍結**: Phase 1学習済みパラメータを使用
3. **Visual Adapter学習**: 画像+テキストペアで教師あり学習
4. **Fusion Layer学習**: 多モーダルタスクでファインチューニング

#### データ要件
- **教師データ**: 画像キャプション+説明文ペア
- **評価データ**: VQA, 画像理解タスク
- **サイズ**: 最小限 (10K-50Kサンプル)

### 理論的根拠

**Geminiの示唆を反映**:
- **「翻訳機」アプローチ**: 視覚情報を「賢者の脳」が理解できるSO(8)空間に変換
- **モジュール式進化**: Phase 1の知性をPhase 2で拡張
- **VRAM効率**: 最小限のパラメータ更新で最大限の機能拡張

### 実装スケジュール

#### Phase 2 準備フェーズ (今すぐ)
- [ ] Visual Projection Adapterクラス実装
- [ ] Adapter Fusion Layerクラス実装
- [ ] モジュラーアーキテクチャ統合テスト

#### Phase 2 学習フェーズ (Phase 1完了後)
- [ ] SigLIP/CLIP統合
- [ ] 多モーダルデータセット準備
- [ ] Visual Adapter学習スクリプト
- [ ] Fusion評価と最適化

### リスク評価と緩和策

**リスク1: モーダルギャップ**
- **問題**: テキストSO(8)空間 vs 視覚SO(8)空間 の不整合
- **対策**: 共通のSO(8)幾何学的制約を両アダプタに適用

**リスク2: VRAM超過**
- **問題**: 予期せぬメモリ消費
- **対策**: Gradient Checkpointing + 厳格なパラメータ凍結

**リスク3: 学習不安定性**
- **問題**: 多モーダル融合の不安定性
- **対策**: 小さな学習レート + 段階的融合強度増加

### 結論

このAdapter Fusion戦略により、RTX 3060の制約を最大限に活かしつつ、Phase 1のテキスト知性をPhase 2のマルチモーダル知性へと進化させることができる。

Geminiの示唆通り、「『目』から入った情報を、『賢者の脳』が理解できる形式に変換する『翻訳機』」というアプローチが、理論的・実装的な両面で最適解である。
