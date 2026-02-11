# Wikipedia Chromium クローラー実装ログ

## 実装日時
2025-11-07

## 概要
Chromium（Playwright）を使用してWikipedia日本語・英語をクロールし、指定ドメイン（防衛、航空宇宙、半導体、精密機器、インフラ、運輸）の知識を収集。自動ラベル付け（ALLOW/ESCALATION/DENY）を行い、/thinkingモデルの教師データとして偏りのないデータセットを生成するシステムを実装。

## 実装ファイル

### 1. `scripts/data/wikipedia_domain_keywords.json`
- ドメイン別キーワード定義
- 6ドメイン（defense, aerospace, semiconductor, precision, infrastructure, transport）
- 各ドメインの日本語・英語キーワード
- ラベル付けルール定義
- バランス調整設定

### 2. `scripts/data/wikipedia_chromium_crawler.py`
- ChromiumベースのWikipediaクローラー
- Playwrightを使用
- ドメイン別キーワードベースの分類
- 自動ラベル付け機能統合
- チェックポイント機能
- バランス調整機能

### 3. `scripts/data/auto_labeler_thinking.py`
- /thinkingモデル用の自動ラベル付け
- ドメイン別ラベル付けルール
- バランス調整機能（ドメイン別、ラベル別、言語別）
- thinkingフィールド生成

### 4. `scripts/data/test_wikipedia_crawler.py`
- クローラーのテストスクリプト

## 実装詳細

### 1. ドメイン別キーワード定義

6つのドメインを定義:
- **defense（防衛）**: 防衛、軍事、安全保障、国防、自衛隊、ミサイル、領土、サイバー戦、PKO、災害派遣
- **aerospace（航空宇宙）**: 航空、宇宙、JAXA、ロケット、衛星、国際宇宙ステーション、航空機
- **semiconductor（半導体）**: 半導体、チップ、集積回路、トランジスタ、CPU、GPU、メモリ
- **precision（精密機器）**: 精密機器、計測機器、センサー、光学的機器、医療機器
- **infrastructure（インフラ）**: インフラ、インフラストラクチャ、電力、水道、ガス、通信、道路、橋梁
- **transport（運輸）**: 運輸、輸送、交通、物流、鉄道、自動車、船舶、航空輸送

各ドメインに日本語・英語キーワードを定義。

### 2. クローラー機能

#### 主な機能
- Chromium（Playwright）を使用した非同期クローリング
- robots.txt遵守
- レート制限（1秒/リクエスト）
- タイムアウト処理（30秒）
- 最大深度: 3階層
- チェックポイント機能（3分間隔、最大5個）

#### クロールプロセス
1. シードURL生成（ドメイン別キーワードからWikipedia URL生成）
2. ページアクセス（Playwright）
3. HTML解析（BeautifulSoup）
4. テキスト抽出（メインコンテンツのみ）
5. ドメイン分類（キーワードマッチング）
6. 自動ラベル付け
7. サンプル保存

### 3. 自動ラベル付け機能

#### ラベル分類ルール
- **ALLOW（33%）**: 公開情報、一般的な知識、教育目的
- **ESCALATION（34%）**: 専門判断が必要、詳細情報、計画情報
- **DENY（33%）**: 機密情報、個人情報、開示禁止情報

#### ドメイン別分布
- 公開情報（60%）→ ALLOW
- 訓練計画等（25%）→ ESCALATION
- 装備詳細等（10%）→ DENY
- 作戦詳細等（5%）→ DENY

#### 分類方法
1. キーワードマッチング
2. パターンマッチング（正規表現）
3. ドメイン別分布に基づくサンプリング

### 4. バランス調整機能

#### 調整項目
1. **ドメイン別**: 各ドメイン均等（16.67%ずつ）
2. **ラベル別**: ALLOW 33%、ESCALATION 34%、DENY 33%
3. **言語別**: 日本語 70%、英語 30%

#### 実装
- `balance_by_domain()`: ドメイン別バランス調整
- `balance_dataset()`: ラベル別バランス調整
- `balance_by_language()`: 言語別バランス調整
- `balance_complete()`: 完全なバランス調整

### 5. データ形式

#### /thinkingモデル用データ形式
```json
{
  "instruction": "ユーザークエリ",
  "input": "",
  "output": "モデル応答",
  "thinking": "<think>内部推論プロセス</think>",
  "safety_judgment": "ALLOW|ESCALATION|DENY",
  "confidence": 0.95,
  "domain": "defense|aerospace|semiconductor|precision|infrastructure|transport",
  "language": "ja|en",
  "so8_group_state": "stable",
  "pet_regularization": 0.1,
  "self_verification": "passed"
}
```

### 6. データ保存

#### 保存先
- `D:\webdataset`

#### ファイル構成
- `wikipedia_defense_ja.jsonl`
- `wikipedia_defense_en.jsonl`
- `wikipedia_aerospace_ja.jsonl`
- ...（各ドメイン×言語）
- `wikipedia_all_samples.jsonl`（統合ファイル）
- `wikipedia_statistics.json`（統計情報）

## 使用方法

### 基本使用
```bash
python scripts/data/wikipedia_chromium_crawler.py --output D:\webdataset --target 1000
```

### テスト実行
```bash
python scripts/data/test_wikipedia_crawler.py
```

## 注意事項

### 実装済み
- Playwright/Seleniumのインストール確認
- ドメイン別キーワード定義ファイル作成
- ChromiumベースのWikipediaクローラー実装
- 自動ラベル付け機能実装
- バランス調整機能実装
- D:\webdatasetへの保存機能実装
- テスト実行と動作確認

### 動作確認事項
- クローラーは正常に動作
- ドメイン分類は正常に動作
- 自動ラベル付けは正常に動作
- バランス調整は正常に動作

### 今後の改善点
- より詳細なエラーハンドリング
- ページ読み込み失敗時のリトライ機能
- より高度なテキスト抽出
- より正確なドメイン分類（機械学習ベース）

## 技術スタック

- Python 3.x
- Playwright (Chromium)
- BeautifulSoup (HTML解析)
- asyncio (非同期処理)
- tqdm (進捗表示)

## 参考資料

- `_docs/appendix/dataset_specifications.md`: データセット仕様詳細
- `scripts/data/label_four_class_dataset.py`: 四値分類ラベル付けスクリプト
- `scripts/data/simple_web_crawler.py`: 簡易Webクローラー

---

**実装完了**

