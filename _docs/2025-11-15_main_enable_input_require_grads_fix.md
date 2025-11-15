# enable_input_require_grads 修正実装ログ

## 実装情報
- **日付**: 2025-11-15
- **Worktree**: main
- **機能名**: enable_input_require_grads fix for hidden_states requires_grad issue
- **実装者**: AI Agent

## 実装内容

### 問題分析
`hidden_states`が`requires_grad=False`になる問題が発生。8-bit量子化とQLoRAを使用する場合、`prepare_model_for_kbit_training`を呼び出した後に`enable_input_require_grads`を呼び出す必要があるが、これが欠けていた。

### 実装項目

#### 1. メインのトレーニングセットアップに`enable_input_require_grads`を追加

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: `prepare_model_for_kbit_training`の直後に`enable_input_require_grads`を呼び出すように修正。ImportErrorの場合は手動で`requires_grad=True`を設定するフォールバックを実装。

**変更内容**:
- Lines 790-802: `enable_input_require_grads`の呼び出しを追加
- ImportError時のフォールバック処理を実装

#### 2. SO8Tモデル読み込み時に`enable_input_require_grads`を追加

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: `load_model_with_so8t`関数内で、SO8Tモデルに量子化を適用した後に`enable_input_require_grads`を呼び出すように修正。

**変更内容**:
- Lines 584-595: SO8Tモデル読み込み時の`enable_input_require_grads`呼び出しを追加
- ImportError時のフォールバック処理を実装

#### 3. QLoRAセットアップ後のモデルパラメータ検証

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: `get_peft_model`の後に、学習可能パラメータ数をログ出力し、`enable_input_require_grads`を再度呼び出すように修正。モデルが訓練モードであることも確認。

**変更内容**:
- Lines 834-854: 学習可能パラメータ数の検証とログ出力を追加
- `enable_input_require_grads`の再呼び出しを追加
- モデルを訓練モードに設定する処理を追加

#### 4. `compute_loss`でのフォールバック機構追加

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: `hidden_states.requires_grad=False`の場合に、モデルの状態を修正するフォールバック機構を追加。訓練モードの確認、`enable_input_require_grads`の再呼び出し、学習可能パラメータの検証を実装。

**変更内容**:
- Lines 352-374: モデル状態修正のフォールバック機構を追加
- 訓練モードの確認と設定
- `enable_input_require_grads`の再呼び出し
- 学習可能パラメータの検証と手動有効化

## 作成・変更ファイル
- `scripts/training/train_borea_phi35_so8t_thinking.py`

## 設計判断
1. **ImportError時のフォールバック**: `enable_input_require_grads`が利用できない場合でも、手動で`requires_grad=True`を設定することで互換性を確保
2. **複数箇所での呼び出し**: モデル読み込み時、QLoRA適用後、エラー発生時の3箇所で`enable_input_require_grads`を呼び出すことで、確実に勾配計算が有効になるように設計
3. **学習可能パラメータ数の検証**: QLoRA適用後に学習可能パラメータ数をログ出力することで、設定が正しいことを確認可能

## テスト結果
- リンターエラー: なし
- 実装完了: すべての修正を適用済み

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

