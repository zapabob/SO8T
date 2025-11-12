# Borea-Phi-3.5-mini-Instruct-Jp SO8T再学習再開ログ

## 実装情報
- **日付**: 2025-11-08
- **Worktree**: main
- **機能名**: Borea-Phi-3.5-mini-Instruct-Jp SO8T再学習再開
- **実装者**: AI Agent

## 実装内容

### 1. 再学習再開

**実装状況**: [実装済み]  
**動作確認**: [実行中]  
**確認日時**: 2025-11-08  
**備考**: 最新のログを参考にBorea-Phi-3.5-mini-Instruct-JpのSO8T再学習を再開

#### 実行コマンド
```bash
py -3 scripts/training/retrain_borea_phi35_with_so8t.py \
    --base-model models/Borea-Phi-3.5-mini-Instruct-Jp \
    --dataset D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl \
    --output D:/webdataset/checkpoints/so8t_retrained_borea_phi35 \
    --config configs/retrain_borea_phi35_so8t_config.yaml
```

#### 設定内容
- **ベースモデル**: `models/Borea-Phi-3.5-mini-Instruct-Jp`
- **データセット**: `D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl`
- **出力ディレクトリ**: `D:/webdataset/checkpoints/so8t_retrained_borea_phi35`
- **設定ファイル**: `configs/retrain_borea_phi35_so8t_config.yaml`

#### 学習設定
- **エポック数**: 3
- **バッチサイズ**: 1 (per device)
- **勾配累積**: 16
- **学習率**: 2.0e-4
- **LoRA r**: 64
- **LoRA alpha**: 128
- **チェックポイント間隔**: 5分（300秒）
- **最大チェックポイント数**: 10

#### 特徴
- QLoRA 8bit学習対応
- 四重推論形式対応（use_quadruple_thinking）
- 電源断リカバリー機能（5分間隔チェックポイント）
- パフォーマンスプロファイリング機能
- 自動データセット分割（train/val/test: 80/10/10）

## 作成・変更ファイル
- 再学習スクリプト実行: `scripts/training/retrain_borea_phi35_with_so8t.py`
- 設定ファイル: `configs/retrain_borea_phi35_so8t_config.yaml`

## 設計判断

1. **ベースモデル**: `models/Borea-Phi-3.5-mini-Instruct-Jp`（リポジトリ整理後のパス）を使用
2. **データセット**: `D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl`を使用
3. **学習方式**: QLoRA 8bit学習を使用し、メモリ効率を重視
4. **四重推論**: 四重推論形式（think-task/think-safety/think-policy/final）に対応
5. **電源断リカバリー**: 5分間隔でチェックポイントを自動保存

## 進捗確認方法

### ログファイル確認
```bash
# 最新のログを確認
Get-Content logs\retrain_borea_phi35_so8t.log -Tail 50
```

### チェックポイント確認
```bash
# 最新のチェックポイントを確認
Get-ChildItem D:\webdataset\checkpoints\so8t_retrained_borea_phi35\checkpoints -Recurse -Filter "*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### プロセス確認
```bash
# Pythonプロセスを確認
Get-Process python | Where-Object { $_.CommandLine -like "*retrain_borea_phi35*" }
```

## 出力ファイル

### チェックポイント
- `D:/webdataset/checkpoints/so8t_retrained_borea_phi35/checkpoints/checkpoint_*.pt`
- 5分間隔で自動保存（最大10個）

### パフォーマンスレポート
- `D:/webdataset/checkpoints/so8t_retrained_borea_phi35/performance_report.json`
- メモリ使用量、GPUメモリ使用量、学習速度の記録

### 最終モデル
- `D:/webdataset/checkpoints/so8t_retrained_borea_phi35/final_model/`
- 学習完了後に保存

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

## 次のステップ

1. **進捗監視**: 再学習の進捗を定期的に確認
2. **チェックポイント確認**: チェックポイントが正常に保存されているか確認
3. **パフォーマンス確認**: パフォーマンスレポートを確認
4. **学習完了確認**: 学習完了後に最終モデルを確認
5. **評価実行**: 再学習済みモデルを評価
6. **A/Bテスト実行**: 元のモデルと再学習済みモデルを比較

## 参考資料

- `_docs/2025-11-08_main_Borea-Phi-3.5-mini-Instruct-Jp_SO8T再学習実装.md`: 再学習実装ログ
- `configs/retrain_borea_phi35_so8t_config.yaml`: 設定ファイル
- `scripts/training/retrain_borea_phi35_with_so8t.py`: 再学習スクリプト










