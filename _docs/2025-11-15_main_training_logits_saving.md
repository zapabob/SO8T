# 学習時logits保存機能実装ログ

## 実装情報
- **日付**: 2025-11-15
- **Worktree**: main
- **機能名**: Training logits saving functionality
- **実装者**: AI Agent

## 実装内容

### 問題分析
学習中のモデルの出力logitsを保存して、後で分析できるようにする機能が必要でした。

### 実装項目

#### 1. 設定ファイルにlogits保存設定を追加

**ファイル**: `configs/train_borea_phi35_so8t_thinking_rtx3060.yaml`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 学習設定セクションにlogits保存関連の設定を追加。

**変更内容**:
- Lines 92-96: logits保存設定を追加
  - `save_logits`: logits保存を有効化するフラグ
  - `save_logits_steps`: logits保存間隔（ステップ数）
  - `save_logits_dir`: logits保存先ディレクトリ（output_dir相対）
  - `save_logits_max_files`: 最大保存ファイル数（古いファイルを自動削除）

#### 2. SO8TPETTrainerクラスにlogits保存機能を追加

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: `SO8TPETTrainer`クラスにlogits保存機能を追加。`__init__`メソッドで設定を受け取り、`_save_logits`メソッドで実際の保存処理を実装。

**変更内容**:
- Lines 289-308: `__init__`メソッドにlogits保存関連のパラメータを追加
  - `save_logits`: logits保存を有効化するフラグ
  - `save_logits_steps`: logits保存間隔
  - `save_logits_dir`: logits保存先ディレクトリ
  - `save_logits_max_files`: 最大保存ファイル数
  - `saved_logits_files`: 保存済みファイルリスト（古いファイル削除用）
  - logits保存ディレクトリの作成

#### 3. compute_lossメソッドにlogits保存処理を追加

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: `compute_loss`メソッドの最後に、定期的にlogitsを保存する処理を追加。

**変更内容**:
- Lines 479-490: logits保存処理を追加
  - `save_logits_steps`ごとにlogitsを保存
  - エラーハンドリングを実装

#### 4. _save_logitsメソッドの実装

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: logitsを保存する専用メソッドを実装。メモリ効率を考慮してCPUに移動してから保存。

**変更内容**:
- Lines 494-542: `_save_logits`メソッドを実装
  - logitsとlabelsをCPUに移動
  - ファイル名を生成（`logits_step_{step}_epoch_{epoch}_{timestamp}.pt`）
  - 保存データを準備（logits, labels, step, epoch, timestamp, shape, dtype等）
  - PyTorch形式で保存
  - 古いファイルを自動削除（最大ファイル数制限）

#### 5. SO8TPETTrainerインスタンス化時に設定を渡す

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: `SO8TPETTrainer`のインスタンス化時に、設定ファイルから読み込んだlogits保存設定を渡すように修正。

**変更内容**:
- Lines 1058-1075: logits保存設定を取得して`SO8TPETTrainer`に渡す
  - 設定ファイルから`save_logits`, `save_logits_steps`, `save_logits_dir`, `save_logits_max_files`を取得
  - `SO8TPETTrainer`のインスタンス化時にこれらの設定を渡す

## 作成・変更ファイル
- `configs/train_borea_phi35_so8t_thinking_rtx3060.yaml`
- `scripts/training/train_borea_phi35_so8t_thinking.py`

## 設計判断
1. **メモリ効率**: logitsをCPUに移動してから保存することで、GPUメモリを節約
2. **自動ファイル管理**: 最大保存ファイル数を設定し、古いファイルを自動削除することで、ディスク容量を管理
3. **柔軟な設定**: 設定ファイルで保存間隔や保存先を変更可能
4. **エラーハンドリング**: logits保存に失敗しても学習が継続できるようにエラーハンドリングを実装

## 保存形式
- **ファイル形式**: PyTorch形式（`.pt`）
- **ファイル名**: `logits_step_{step}_epoch_{epoch}_{timestamp}.pt`
- **保存内容**:
  - `logits`: モデルの出力logits（CPU上）
  - `labels`: ラベル（存在する場合、CPU上）
  - `step`: 現在のステップ数
  - `epoch`: 現在のエポック数
  - `timestamp`: 保存日時
  - `logits_shape`: logitsの形状
  - `logits_dtype`: logitsのデータ型
  - `labels_shape`: labelsの形状（存在する場合）

## テスト結果
- リンターエラー: 警告のみ（実装には影響なし）
- 実装完了: すべての修正を適用済み

## 使用方法

### 設定ファイルでの有効化
```yaml
training:
  # Logits保存設定
  save_logits: true              # logits保存を有効化
  save_logits_steps: 100         # logits保存間隔（ステップ数）
  save_logits_dir: "logits"      # logits保存先ディレクトリ（output_dir相対）
  save_logits_max_files: 10      # 最大保存ファイル数（古いファイルを自動削除）
```

### 保存先ディレクトリ
- デフォルト: `{output_dir}/logits/`
- 例: `D:/webdataset/checkpoints/training/borea_phi35_so8t_thinking_rtx3060/logits/`

### 保存されたlogitsの読み込み
```python
import torch

# logitsファイルを読み込み
logits_data = torch.load("logits_step_100_epoch_0_20251115_120000.pt")

# logitsとlabelsを取得
logits = logits_data['logits']
labels = logits_data.get('labels', None)

# メタデータを取得
step = logits_data['step']
epoch = logits_data['epoch']
timestamp = logits_data['timestamp']
```

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

