# SO8T Dataset Documentation

## データプロベナンス（Data Provenance）

### 主要データソース

#### 1. TFMC重要度行列データセット
- **データセット名**: `TFMC/imatrix-dataset-for-japanese-llm`
- **提供元**: [Hugging Face](https://huggingface.co/datasets/TFMC/imatrix-dataset-for-japanese-llm)
- **用途**: 量子化最適化のための重要度学習
- **ライセンス**: MIT License
- **サイズ**: 日本語LLM向け重要度行列データ
- **最終アクセス日**: 2025年11月

#### 2. 独自生成データセット

##### 安全性学習データセット (`so8t_safety_dataset.jsonl`)
- **生成元**: 倫理的ジレンマシナリオと安全応答パターンの合成生成
- **サイズ**: 10,000+ サンプル
- **特徴**:
  - 倫理的ジレンマケース
  - 安全性の高い応答パターン
  - 多様な文化的文脈
- **生成方法**: Phi-3.5ベースの拡張生成 + 品質フィルタリング

##### 日本語複雑推論データセット (`japanese_complex_dataset_enhanced.jsonl`)
- **生成元**: 数学・科学・倫理的問題の日本語訳生成
- **サイズ**: 5,000+ サンプル
- **特徴**:
  - 高度な推論タスク
  - 学術的・専門的内容
  - 日本語特有の表現バリエーション
- **生成方法**: 多言語モデルによる翻訳 + 専門家レビュー

##### SO(8)思考制御トレーニングデータ (`so8t_thinking_phi35_weighted_train.jsonl`)
- **生成元**: Phi-3.5ベースの思考プロセス拡張
- **サイズ**: 15,000+ サンプル
- **特徴**:
  - 四重推論タグ付き（`<think-logic>`, `<think-ethics>`, `<think-practical>`, `<think-creative>`）
  - SO(8)幾何学的制約を考慮した思考構造
  - 段階的推論プロセス
- **生成方法**: Phi-3.5拡張 + SO(8)幾何学的制約適用

## データ前処理パイプライン

### 前処理スクリプト
```bash
# データクリーニング
python scripts/data_preprocessing/clean_dataset.py

# 品質チェック
python scripts/data_preprocessing/validate_dataset.py

# SO(8)適応変換
python scripts/data_preprocessing/apply_so8t_transform.py
```

### 品質基準
- **NSFWフィルタリング**: 安全学習目的のみ使用（生成目的禁止）
- **言語品質**: 日本語・英語の両言語対応
- **多様性確保**: ドメイン偏在の排除
- **一貫性チェック**: 四重推論構造の完全性検証

## データディレクトリ構造

```
data/
├── cleaned/           # クリーニング済みデータ
├── processed/         # 前処理済みデータ
├── splits/           # 学習/検証/テスト分割
├── synthetic/        # 合成生成データ
├── validated/        # 品質検証済みデータ
├── japanese_complex_dataset_enhanced.jsonl    # 日本語複雑推論
├── so8t_safety_dataset.jsonl                  # 安全性学習
├── so8t_thinking_phi35_weighted_train.jsonl   # SO(8)思考制御
└── README.md         # このファイル
```

## 再現性確保

### 乱数シード
- **データ生成**: `seed=42`
- **データ分割**: `seed=123`
- **品質検証**: `seed=456`

### バージョン管理
- **データセットバージョン**: v1.0.0 (2025-11-24)
- **前処理バージョン**: v1.2.0
- **品質基準バージョン**: v1.1.0

## 倫理的考慮事項

### NSFWコンテンツ
- **使用目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- **フィルタリング**: 自動分類器による検出・除外
- **監査**: 完全な使用ログ記録

### プライバシー保護
- **個人情報**: 一切含まれない
- **生成元**: 合成データのみ
- **トレーサビリティ**: 完全な生成履歴保持

## 貢献ガイドライン

### データ追加時の要件
1. **出典明記**: Hugging Faceまたは生成方法の詳細記述
2. **品質検証**: 自動テストを通す
3. **ドキュメント更新**: このREADME.mdの更新
4. **倫理審査**: NSFW判定とプライバシー確認

---

**最終更新**: 2025年11月24日
**担当**: AI Agent
