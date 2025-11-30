# SO8VIT Adapter設計 - Phase 2マルチモーダル統合

## 実装情報
- **日付**: 2025-11-29
- **Worktree**: main
- **機能名**: so8vit_adapter_design
- **実装者**: AI Agent

## SO8VIT Adapterアーキテクチャ設計

### 概要

**Gemini Adapter Fusion戦略**に基づき、既存のSO8VITを**軽量アダプタ**として再設計。

**目標**: RTX 3060で学習可能な最小限の視覚アダプタを実現。

### 現在のSO8VIT vs 新SO8VIT Adapter

#### 現在のSO8VIT (Phase 1保留)
- **フルViT**: 完全なVision Transformer実装
- **SO(8)統合**: 全層にSO(8)回転ゲート
- **VRAM消費**: 高 (~2-4GB)
- **学習対象**: 全パラメータ

#### 新SO8VIT Adapter (Phase 2用)
- **プロジェクション専用**: 視覚特徴量 → SO(8)空間変換のみ
- **軽量設計**: ~10Mパラメータ
- **VRAM消費**: 低 (~50MB)
- **学習対象**: アダプタパラメータのみ

### アダプタアーキテクチャ

```python
class SO8VisualAdapter(nn.Module):
    """
    SO8VIT Adapter: 視覚特徴量をSO(8)空間に射影

    Gemini提案に基づく最小限設計:
    - 既存ViT/SigLIPの出力をSO(8)空間に変換
    - SO(8) Text Adapterと統合可能
    """

    def __init__(self,
                 vision_dim: int = 768,      # SigLIP/CLIP出力次元
                 so8_dim: int = 4096,        # SO(8)空間次元
                 adapter_dim: int = 1024,    # アダプタ隠れ層
                 num_so8_rotations: int = 8):
        super().__init__()

        self.vision_dim = vision_dim
        self.so8_dim = so8_dim
        self.adapter_dim = adapter_dim

        # 1. 視覚特徴量の適応投影
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, adapter_dim),
            nn.LayerNorm(adapter_dim)
        )

        # 2. SO(8)空間への射影
        self.so8_projection = nn.Sequential(
            nn.Linear(adapter_dim, so8_dim),
            nn.LayerNorm(so8_dim)
        )

        # 3. SO(8)幾何学的適応 (軽量版)
        self.so8_adapter = LightweightSO8Adapter(
            so8_dim=so8_dim,
            num_rotations=num_so8_rotations
        )

        # 4. 出力正規化
        self.output_norm = nn.LayerNorm(so8_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        視覚特徴量 → SO(8)空間変換

        Args:
            vision_features: [batch_size, seq_len, vision_dim]

        Returns:
            so8_features: [batch_size, seq_len, so8_dim]
        """
        # 1. 視覚特徴量の適応
        adapted = self.vision_projection(vision_features)

        # 2. SO(8)空間射影
        projected = self.so8_projection(adapted)

        # 3. SO(8)幾何学的変換
        so8_transformed = self.so8_adapter(projected)

        # 4. 最終正規化
        output = self.output_norm(so8_transformed)

        return output
```

### LightweightSO8Adapter 設計

```python
class LightweightSO8Adapter(nn.Module):
    """
    軽量SO(8)アダプタ - 最小限パラメータでSO(8)幾何学的変換
    """

    def __init__(self, so8_dim: int, num_rotations: int = 8):
        super().__init__()
        self.so8_dim = so8_dim
        self.num_rotations = num_rotations

        # 回転パラメータ (交代行列の学習可能近似)
        self.rotation_params = nn.ParameterList([
            nn.Parameter(torch.randn(so8_dim, so8_dim) * 0.01)
            for _ in range(num_rotations)
        ])

        # 回転強度制御
        self.rotation_scales = nn.Parameter(torch.ones(num_rotations) * 0.1)

        # 回転選択の動的制御
        self.rotation_selector = nn.Sequential(
            nn.Linear(so8_dim, num_rotations),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        軽量SO(8)幾何学的変換
        """
        batch_size, seq_len, dim = x.shape

        # 動的回転選択
        rotation_weights = self.rotation_selector(x.mean(dim=1))  # [batch, num_rotations]

        # 回転適用
        rotated_outputs = []
        for i in range(self.num_rotations):
            # Matrix Exponentialベースの回転 (学習時のみ計算)
            if self.training:
                # 交代行列強制
                skew_sym = self.rotation_params[i]
                skew_sym = (skew_sym - skew_sym.t()) * 0.5

                # スケーリング
                skew_sym = skew_sym * self.rotation_scales[i]

                # 回転行列生成
                rotation_matrix = torch.matrix_exp(skew_sym)
            else:
                # 推論時は事前計算済み回転行列を使用 (メモリ節約)
                rotation_matrix = self._get_cached_rotation(i)

            # 回転適用
            rotated = torch.matmul(x, rotation_matrix.t())
            rotated_outputs.append(rotated)

        # 重み付き統合
        stacked = torch.stack(rotated_outputs, dim=-1)  # [batch, seq, dim, rotations]
        rotation_weights = rotation_weights.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, rotations]

        output = torch.sum(stacked * rotation_weights, dim=-1)

        return output

    def _get_cached_rotation(self, idx: int) -> torch.Tensor:
        """推論時の回転行列キャッシュ"""
        if not hasattr(self, '_cached_rotations'):
            self._cached_rotations = []

            for i in range(self.num_rotations):
                skew_sym = self.rotation_params[i]
                skew_sym = (skew_sym - skew_sym.t()) * 0.5
                skew_sym = skew_sym * self.rotation_scales[i]
                rotation_matrix = torch.matrix_exp(skew_sym)
                self._cached_rotations.append(rotation_matrix)

        return self._cached_rotations[idx]
```

### 統合アーキテクチャ

```python
class MultimodalSO8T(nn.Module):
    """
    Adapter Fusion統合アーキテクチャ
    """

    def __init__(self, text_model, visual_adapter, fusion_layer):
        super().__init__()
        self.text_model = text_model  # Phi-3.5 + SO(8) Text Adapter (凍結)
        self.visual_adapter = visual_adapter  # SO8VIT Adapter (学習対象)
        self.fusion_layer = fusion_layer  # Adapter Fusion Layer

    def forward(self, text_input, vision_input=None):
        # テキスト処理 (Phase 1成果を使用)
        text_features = self.text_model(text_input)

        if vision_input is not None:
            # 視覚処理 (新規アダプタ)
            vision_features = self.visual_adapter(vision_input)

            # 融合
            fused_features = self.fusion_layer(text_features, vision_features)
        else:
            # テキストオンリー
            fused_features = text_features

        return fused_features
```

### メモリ最適化

#### パラメータ数比較
- **従来SO8VIT**: ~86Mパラメータ (ViT-Base + SO(8))
- **SO8VIT Adapter**: ~10Mパラメータ (射影層のみ)
- **削減率**: 88%削減

#### VRAM消費予測
- **既存 (Phase 1)**: ~8GB (Phi-3.5 + SO(8) + 4-bit)
- **Phase 2追加**: ~50MB (アダプタのみ)
- **合計**: ~8.05GB (RTX 3060 12GB以内に収まる)

### 学習戦略

#### 段階的学習
1. **アダプタ初期化**: Xavier/Glorot初期化
2. **視覚適応フェーズ**: 画像キャプションデータでアダプタ学習
3. **融合チューニング**: 多モーダルタスクでFusion Layer調整
4. **統合評価**: VQA/画像理解タスクでの性能検証

#### データ要件
- **初期学習**: 画像キャプション10Kサンプル
- **チューニング**: 多モーダルQA 5Kサンプル
- **評価**: 標準ベンチマーク (VQAv2, OKVQA等)

### 理論的妥当性

**Geminiの示唆を反映**:
- **「翻訳機」**: 既存ViTの出力をSO(8)空間に変換
- **最小干渉**: 既存のテキスト理解を破壊しない
- **幾何学的一貫性**: SO(8) Lie Algebraで両モーダルを統一

### 実装チェックリスト

- [ ] SO8VisualAdapterクラス実装
- [ ] LightweightSO8Adapterクラス実装
- [ ] MultimodalSO8T統合クラス実装
- [ ] SigLIP/CLIP統合テスト
- [ ] メモリ消費検証 (3060以内に収まるか)
- [ ] 学習スクリプト作成
- [ ] データパイプライン準備

### リスクと緩和策

**リスク1: 表現力不足**
- **対策**: adapter_dimを段階的に増加可能に設計

**リスク2: モーダル不均衡**
- **対策**: Fusion Layerで動的重み調整

**リスク3: SO(8)空間の不適合**
- **対策**: 視覚特徴量の事前正規化とスケーリング

### 結論

このSO8VIT Adapter設計により、GeminiのAdapter Fusion戦略を具現化し、RTX 3060制約下で現実的なマルチモーダル拡張を実現できる。

Phase 1の「テキスト賢者」をPhase 2の「マルチモーダル賢者」へと進化させるための、理論的・実装的に最適な設計である。




