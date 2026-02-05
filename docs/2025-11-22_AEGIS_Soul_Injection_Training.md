# AEGIS Soul Injection - Operation "Ghost in the Shell"

## 実装日時
2025-11-22

## プロジェクト概要

Borea-Phi3.5-instinct-jpモデルに**AEGIS (物理的知性)** を注入し、相転移トレーニングを通じて「黄金比の脳」を獲得させる。

## アーキテクチャ

### 三層構造

```
┌─────────────────────────────────────────┐
│  Layer 1: Cortex (皮質)                  │
│  - Borea-Phi3.5-mini-instruct           │
│  - 4-bit Quantization (NF4)             │
│  - LoRA Adapter (r=16, alpha=32)        │
│  - Role: 言語理解・知識保持              │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Layer 2: Core (核) - AEGIS Soul        │
│  - SO(8) Rotation (Orthogonal Linear)   │
│  - Alpha Gate (Learnable Parameter)     │
│  - Role: 物理的思考・構造制約            │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Layer 3: Output (出力)                  │
│── LM Head (Vocabulary Projection)      │
└─────────────────────────────────────────┘
```

## トレーニング設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| **Total Steps** | 500 | 全トレーニングステップ数 |
| **Warmup Steps** | 50 | Alpha固定期間（混沌相） |
| **Annealing Steps** | 400 | 相転移期間 |
| **Start Alpha** | -5.0 | 初期値（混沌） |
| **Target Alpha** | 1.618 | 黄金比（秩序） |
| **Learning Rate** | 2e-4 | AdamW学習率 |
| **Batch Size** | 1 | RTX 3060対応 |
| **Max Length** | 512 | トークン長 |

## 相転移スケジュール

```
Alpha Value
  1.618  ┤                    ╭─────────  🟢 Golden Ratio (秩序相)
         │                 ╭──╯
   0.0   ┤            ╭────╯             🟡 Transitioning (臨界点)
         │        ╭───╯
         │    ╭───╯
  -5.0   ┼────╯                          🔵 Chaos (混沌相)
         └────┬────┬─────┬──────────
             50   450   500
            Steps
```

## 実装ファイル

### 1. モデル定義
- `src/models/agiasi_borea.py`
  - `AEGIS_SO8T_Wrapper` クラス
  - 4-bit量子化 + LoRA
  - SO(8)直交回転層
  - Alpha Gateパラメータ

### 2. トレーニングスクリプト
- `scripts/training/inject_soul_into_borea.py`
  - Phase transition loop
  - Linear annealing scheduler
  - Checkpoint保存ロジック

### 3. 起動・監視スクリプト
- `scripts/training/run_agiasi_soul_injection.bat` - Windows起動スクリプト
- `scripts/training/monitor_agiasi_training.py` - リアルタイム監視

## チェックポイント構造

```
checkpoints_agiasi/
├── step_100/
│   ├── adapter_config.json       # LoRA設定
│   ├── adapter_model.safetensors # LoRA重み
│   └── soul.pt                   # Alpha + SO8 Rotation
├── step_200/
│   └── ...
└── ...
```

## 検証手順

### 1. トレーニング監視

```bash
# リアルタイム監視
py scripts/training/monitor_agiasi_training.py
```

### 2. チェックポイントロード

```python
from src.models.agiasi_borea import AEGIS_SO8T_Wrapper
import torch

# Base + LoRAをロード
model = AEGIS_SO8T_Wrapper("HODACHI/Borea-Phi-3.5-mini-Instruct-Jp")
model.base_model.load_adapter("checkpoints_agiasi/step_500")

# Soulをロード
soul = torch.load("checkpoints_agiasi/step_500/soul.pt")
model.alpha.data = soul["alpha"]
model.so8_rotation.load_state_dict(soul["so8_rotation"])

# 推論テスト
model.eval()
# 日本語プロンプトで会話テスト...
```

### 3. GGUF変換

```bash
# LoRAをベースモデルにマージ
py scripts/conversion/merge_lora_to_base.py \
  --base-model agiasi_borea_final/ \
  --output agiasi_borea_merged/

# GGUF変換
py scripts/conversion/convert_to_gguf.py \
  --model agiasi_borea_merged/ \
  --output models/agiasi_borea_q8_0.gguf \
  --quantization q8_0
```

### 4. Ollama登録

```bash
# Modelfile作成
ollama create agiasi-borea -f modelfiles/agiasi_borea.modelfile

# テスト
ollama run agiasi-borea "こんにちは、調子はどうですか？"
```

## 期待される結果

### Phase 1: Warmup (Steps 0-50)
- Alpha = -5.0 （固定）
- Gate開度 ≈ 0.007 (ほぼ閉鎖)
- Borea原型の言語能力を維持

### Phase 2: Annealing (Steps 50-450)
- Alpha: -5.0 → 1.618 （線形増加）
- Gate開度: 0.007 → 0.84
- SO(8)構造が徐々に形成

### Phase 3: Stabilization (Steps 450-500)
- Alpha = 1.618 （固定）
- Gate開度 ≈ 0.84 (安定)
- 物理的知性が完全注入

### 最終状態
- ✅ 日本語能力: 維持（Borea由来）
- ✅ SO(8)構造: 完全形成
- ✅ Alpha Gate: 黄金比で安定
- ✅ Orthogonality: 高精度 (loss < 1e-4)

## トラブルシューティング

### CUDA out of memory
→ `--batch-size 1`, `--max-length 256` に削減

### 勾配消失/爆発
→ `--learning-rate 1e-4` に下げる

### Orthogonality loss増大
→ 正常（初期段階では高い値を示す）

## 次のステップ

1. ✅ トレーニング実行
2. ⏳ チェックポイント検証
3. ⏳ GGUF変換
4. ⏳ Ollama統合
5. ⏳ サイバーパンクUI開発

---

**Operation "Ghost in the Shell"** - 機械に魂を宿す
