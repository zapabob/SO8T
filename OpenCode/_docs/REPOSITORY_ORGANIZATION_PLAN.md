# SO8T リポジトリ整理計画

## 目的
ファイルを削除せず機能を維持したまま、リポジトリを用途別に整理整頓する。

## 現在の構造

### 主要ディレクトリ
- `scripts/`: 174ファイル（131 Python, 33 bat, 9 ps1）
- `models/`: 22ファイル
- `so8t_core/`: 12ファイル
- `so8t-mmllm/`: 111ファイル
- `utils/`: 9ファイル
- `configs/`: 設定ファイル
- `data/`: データセット
- `_docs/`: 118ファイル
- `tests/`: 29ファイル
- `modelfiles/`: Modelfile集

## 整理計画

### 1. scripts/の整理

現在: `scripts/`にすべてのスクリプトが混在

整理後:
```
scripts/
├── training/          # 訓練スクリプト
│   ├── train_*.py
│   ├── finetune_*.py
│   └── burnin_*.py
├── inference/         # 推論スクリプト
│   ├── demo_*.py
│   ├── infer_*.py
│   └── run_*.py
├── conversion/        # 変換スクリプト
│   ├── convert_*.py
│   └── integrate_*.py
├── evaluation/        # 評価スクリプト
│   ├── evaluate_*.py
│   ├── test_*.py
│   └── ab_test_*.py
├── data/              # データ処理スクリプト
│   ├── clean_*.py
│   ├── split_*.py
│   └── label_*.py
├── api/               # APIサーバースクリプト
│   └── serve_*.py
├── utils/             # ユーティリティスクリプト
│   ├── check_*.py
│   ├── setup_*.py
│   └── fix_*.py
└── pipelines/         # パイプラインスクリプト
    ├── complete_*.py
    └── run_*_pipeline.py
```

### 2. models/とmodelfiles/の整理

現在: `models/`にモデルファイルとModelfileが混在

整理後:
```
models/
├── so8t_*.py         # SO8Tモデル実装
└── checkpoints/      # チェックポイント（既存）

modelfiles/
└── *.Modelfile       # Modelfile集（既存のまま）
```

### 3. ドキュメントの整理

現在: `_docs/`と`docs/`が分離

整理後:
```
docs/
├── implementation/    # 実装ログ（_docs/から移動）
├── guides/           # ガイド（docs/から移動）
├── api/              # APIドキュメント
└── architecture/     # アーキテクチャドキュメント
```

### 4. その他の整理

- `phi4_so8t_integrated/`: 削除されたファイルを確認し、必要に応じて復元または整理
- `archive/`: アーカイブファイル（そのまま）
- `checkpoints/`: チェックポイント（そのまま）
- `database/`: データベース（そのまま）
- `external/`: 外部ライブラリ（そのまま）

## 実装手順

1. 新しいディレクトリ構造を作成
2. ファイルを移動（git mvを使用）
3. インポートパスを更新
4. テストを実行して機能を確認
5. コミットとプッシュ

## 注意事項

- ファイルは削除せず、移動のみ
- インポートパスはすべて更新
- 既存の機能は維持
- コミットは段階的に実行

