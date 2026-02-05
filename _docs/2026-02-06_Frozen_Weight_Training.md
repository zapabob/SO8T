# Model B (Borea) 重み凍結機能実装ログ

**Date**: 2026-02-06
**Feature**: Frozen Weight Retraining Implementation

---

## Summary

Model B (Borea) の既存能力を保護しつつ、Adapter/LoRA で新規知識を学習する機能を実装。

---

## Implementation Details

### 1. 設定ファイル更新

**File**: [borea_training.json](file:///c:/Users/downl/Desktop/SO8T/src/infrastructure/config/borea_training.json)

```json
{
  "model": {
    "freeze_base_model": true,
    "freeze_preserve_soul_weights": true
  }
}
```

### 2. トレーニングコード更新

**File**: [train_unsloth_so8t.py](file:///c:/Users/downl/Desktop/SO8T/src/training/train_unsloth_so8t.py)

- `freeze_base_model_weights()` メソッドを新規追加
- `run_sft_training()` で LoRA 設定後に凍結を呼び出し

### 学習可能パラメータ

以下のキーワードを含むパラメータのみ学習可能:

- `lora` - QLoRAアダプター
- `so8` / `so8t` - SO(8)ゲート
- `rotation` - 回転行列
- `alpha_gate` / `alpha` - Alpha Gate/パラメータ
- `r_safe` / `r_cmd` - 安全/コマンド行列（魂の重み）
- `soul` - 魂のパラメータ
- `safety_head` / `task_head` / `dual_heads` - 二重政策系
- `pet` - PET正則化

---

## Environment

```powershell
$env:SO8T_USE_UNSLOTH = "1"
$env:SO8T_FREEZE_BASE = "1"  # Optional override
```

---

## Verification

```powershell
py -3 -m py_compile src/training/train_unsloth_so8t.py  # ✅ 成功
```

---

## New Dataset Collection Integration

**追加日時**: 2026-02-06 02:33+09:00

`collect_new_datasets()` メソッドをパイプラインに追加:

### 収集対象

1. **Arxiv/BioRxiv** - 2024-2026 高引用論文
2. **OSINT** - GDELT/RSS ソース (世界情勢)
3. **日本大学入試問題** - ローカルデータ統合
4. **MCP スキル** - ツール呼び出しデータセット
5. **WebResearch/DeepResearch** - 研究データセット

### 環境変数

- `SO8T_COLLECT_ARXIV=1` - Arxiv 収集有効
- `SO8T_COLLECT_OSINT=1` - OSINT 収集有効
- `SO8T_DRYRUN=1` - 収集スキップ（ドライラン）
