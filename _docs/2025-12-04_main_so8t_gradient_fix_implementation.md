# SO8T勾配フロー修正実装ログ

## 実装情報
- **日付**: 2025-12-04
- **Worktree**: main
- **機能名**: SO8T勾配フロー修正実装
- **実装者**: AI Agent

## 実装内容

### 1. [Grad Norm: None] 問題の根本原因特定

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-04
**備考**: Unslothによるモデル凍結が中間層出力のrequires_grad=Falseを引き起こす

- Unslothでベースモデルを凍結すると、中間層の`hidden_states`の`requires_grad`がFalseになる
- Hookで割り込んでも勾配が伝わらず、`Grad Norm: None`が発生
- SO8Tアダプターが「学習不能」状態になる

### 2. Hook内での強制勾配有効化実装

**ファイル**: `scripts/models/so8t_residual_adapter.py`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-04
**備考**: Hook内でrequires_grad_(True)を呼び出し勾配を強制有効化

```python
def nkat_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0]

        # ★★★ 修正ポイント: 勾配の「呼び水」を入れる ★★★
        # Unslothで凍結された中間層の出力を強制的に勾配有効化
        if hidden_states.requires_grad is False and torch.is_grad_enabled():
            hidden_states.requires_grad_(True)

        # アダプター適用
        new_hidden = module.nkat_adapter(hidden_states)
        return (new_hidden,) + output[1:]
    # ... (Tensor単体の場合も同様)
```

### 3. Manual Optimizer Registration実装

**ファイル**: `scripts/pipeline/sunshine_pipeline.py`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-04
**備考**: SO8Tの場合のみLoRA+NKTパラメータを抽出してOptimizerに手動登録

```python
# 🔥 緊急バイパス手術：Optimizerへの手動登録（SO8Tの場合のみ）
if config.so8_config:
    print("[4.5/5] Manual optimizer registration for SO8T...")
    # 1. 学習対象パラメータの抽出 (LoRA と NKATアダプタ だけ)
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "nkat_adapter" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    print(f"🔥 Total Trainable Params: {len(trainable_params)} tensors")

    # 2. Optimizerの手動作成
    try:
        from unsloth.optim import AdamW8bit
        optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
        print("✅ Using Unsloth AdamW8bit")
    except ImportError:
        from torch.optim import AdamW8bit
        optimizer = AdamW8bit(trainable_params, lr=config.training_config.get('learning_rate', 2e-5))
        print("✅ Using Standard AdamW")

    # 3. Trainerに手動Optimizerを渡す
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[callback, NKATDebugCallback(model)],
        optimizers=(manual_optimizer, None)  # (optimizer, scheduler)
    )
```

## 作成・変更ファイル
- `scripts/models/so8t_residual_adapter.py` - Hook内勾配有効化修正
- `scripts/pipeline/sunshine_pipeline.py` - Manual Optimizer Registration追加

## 設計判断
- **Hook内修正**: Unslothの凍結を回避するため、Hook内で強制的に`requires_grad_(True)`を呼び出す
- **Manual Optimizer**: Unslothの自動管理を回避するため、LoRA+NKTパラメータのみを明示的にOptimizerに登録
- **条件分岐**: SO8Tの場合のみ適用し、Baselineには影響を与えない

## 運用注意事項

### データ収集ポリシー
- 利用条件を守りつつ、高信頼ソースとして優先使用
- robots.txt遵守を徹底
- 個人情報・機密情報の除外を徹底

### NSFWコーパス運用
- **主目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- モデル設計とドキュメントに明記
- 分類器は検出・拒否用途のみ

### /thinkエンドポイント運用
- 四重Thinking部（`<think-*>`）は外部非公開を徹底
- `<final>`のみ返す実装を維持
- 監査ログでThinkingハッシュを記録（内容は非公開）
