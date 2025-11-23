# Webクロール機能ガイド

## 概要

`collect_japanese_data.py`にWebクロール機能を追加しました。HuggingFaceデータセットに加えて、日本の政府・公共サイトから日本語データを直接収集できます。

## 機能

### 1. 倫理的クロール
- ✅ **robots.txt遵守**: 自動確認、禁止URLはスキップ
- ✅ **レート制限**: 1秒間隔（設定可能）
- ✅ **User-Agent設定**: 研究目的を明示
- ✅ **タイムアウト**: 10秒（サーバー負荷軽減）

### 2. 対象サイト（7ドメイン）

| ドメイン | サイト | URL |
|---------|--------|-----|
| **防衛** | 防衛省 | https://www.mod.go.jp/ |
| **金融** | 金融庁、日本銀行 | https://www.fsa.go.jp/, https://www.boj.or.jp/ |
| **医療** | 厚生労働省 | https://www.mhlw.go.jp/ |
| **航空宇宙** | JAXA | https://www.jaxa.jp/ |
| **運輸** | 国土交通省 | https://www.mlit.go.jp/ |
| **ビジネス** | 経済産業省 | https://www.meti.go.jp/ |
| **一般** | 首相官邸、内閣官房 | https://www.kantei.go.jp/, https://www.cas.go.jp/ |

### 3. クロール設定

```python
CRAWL_CONFIG = {
    "max_depth": 3,  # 最大深度（トップページから3階層まで）
    "delay": 1.0,  # リクエスト間隔1秒
    "timeout": 10,  # タイムアウト10秒
    "max_pages_per_domain": 100,  # ドメインあたり最大100ページ
    "user_agent": "SO8T-DataCollector/1.0 (Research Purpose)"
}
```

## 使用方法

### 基本使用（HF + Web）

```bash
cd C:\Users\downl\Desktop\SO8T\so8t-mmllm

# HFデータセット（50%）+ Webクロール（50%）
python scripts/data/collect_japanese_data.py --target 10000
```

### Webクロールのみ

```bash
# HFをスキップ、Webクロールのみ
python scripts/data/collect_japanese_data.py --target 5000 --crawl-only
```

### HFのみ（Webクロール無効）

```bash
# Webクロールを無効化
python scripts/data/collect_japanese_data.py --target 10000 --no-web-crawl
```

### カスタム設定

```bash
# ターゲット数、ワーカー数指定
python scripts/data/collect_japanese_data.py \
  --target 50000 \
  --workers 8 \
  --web-crawl
```

## 出力

### データファイル

```
data/collected/
├── japanese_collected_defense.jsonl
├── japanese_collected_aerospace.jsonl
├── japanese_collected_transport.jsonl
├── japanese_collected_medical.jsonl
├── japanese_collected_finance.jsonl
├── japanese_collected_business.jsonl
├── japanese_collected_general.jsonl
└── collection_stats_YYYYMMDD_HHMMSS.json
```

### レポート

```
_docs/YYYY-MM-DD_data_collection_report.md
```

### サンプル形式

```json
{
  "text": "防衛省は...",
  "url": "https://www.mod.go.jp/...",
  "domain": "defense",
  "source": "web_crawl",
  "quality_score": 0.85,
  "timestamp": 1730936400.0
}
```

## 倫理的配慮

### 1. 法的遵守

- ✅ **著作権法**: 公開情報のみ収集、私的利用範囲
- ✅ **利用規約**: 各サイトのTerms of Service確認
- ✅ **robots.txt**: 完全遵守
- ✅ **個人情報保護法**: 個人情報は収集しない

### 2. 技術的配慮

- ✅ **レート制限**: 1秒間隔（サーバー負荷軽減）
- ✅ **User-Agent**: 研究目的を明示
- ✅ **タイムアウト**: 10秒（長時間接続回避）
- ✅ **同一ドメインのみ**: 外部サイトへの自動遷移なし

### 3. データ利用

- ✅ **研究目的**: AI学習データとして利用
- ✅ **非営利**: 学術研究・PoC用途
- ✅ **データクレジット**: 出所を記録

## 推奨収集量

### PoC用（テスト）

```bash
# 1,000サンプル（約10-30分）
python scripts/data/collect_japanese_data.py --target 1000 --crawl-only
```

### 中規模（開発）

```bash
# 10,000サンプル（約1-3時間）
python scripts/data/collect_japanese_data.py --target 10000
```

### 大規模（本番）

```bash
# 100,000サンプル（約6-12時間）
python scripts/data/collect_japanese_data.py --target 100000 --workers 8
```

## トラブルシューティング

### 1. robots.txt読み込み失敗

```
[WARNING] robots.txt読み込み失敗: ...
→ 対策: ネットワーク接続確認、URLの正確性確認
```

### 2. 403 Forbidden

```
[ERROR] リクエスト失敗: 403 Client Error
→ 対策: User-Agentを調整、レート制限を緩める（delay増加）
```

### 3. タイムアウト

```
[ERROR] リクエスト失敗: timeout
→ 対策: --timeoutを増やす（デフォルト10秒→30秒）
```

### 4. メモリ不足

```
[ERROR] MemoryError
→ 対策: --targetを減らす、チェックポイント機能を活用
```

## 依存関係

### 必須

```bash
pip install beautifulsoup4>=4.12.0
pip install lxml>=4.9.0
pip install html5lib>=1.1
pip install requests>=2.31.0
pip install datasets>=2.14.0
```

### または一括インストール

```bash
cd C:\Users\downl\Desktop\SO8T
pip install -r requirements.txt
```

## 高度な使用例

### カスタムURL追加

```python
# collect_japanese_data.py の WEB_CRAWL_SOURCES に追加

WEB_CRAWL_SOURCES = {
    "defense": [
        "https://www.mod.go.jp/",
        "https://www.mod.go.jp/j/approach/",
        "https://YOUR_CUSTOM_URL/",  # カスタムURL追加
    ],
    # ...
}
```

### クロール設定カスタマイズ

```python
# CRAWL_CONFIG を編集

CRAWL_CONFIG = {
    "max_depth": 5,  # より深くクロール
    "delay": 0.5,  # より速くクロール（注意！）
    "timeout": 30,  # より長く待つ
    "max_pages_per_domain": 500,  # より多くページ収集
}
```

## パフォーマンス

### 推定速度

- **1ページ**: ~2秒（1秒delay + 1秒処理）
- **100ページ**: ~3-5分
- **1,000ページ**: ~30-50分
- **10,000ページ**: ~5-8時間

### ボトルネック

1. **ネットワーク速度**: インターネット接続に依存
2. **サーバー応答**: 対象サイトの速度に依存
3. **レート制限**: 1秒delay（倫理的制約）

## セキュリティ

### 安全なクロール

- ✅ **HTTPSのみ**: 可能な限りHTTPS接続
- ✅ **証明書検証**: SSL証明書の検証
- ✅ **リダイレクト制限**: 無限ループ防止
- ✅ **ファイルサイズ制限**: 大規模ファイルスキップ

### プライバシー

- ✅ **個人情報除外**: 公開情報のみ収集
- ✅ **Cookie非使用**: セッション管理のみ
- ✅ **トラッキング回避**: UTMパラメータ除外

## ライセンス

本クローラーは研究・教育目的で使用してください。商用利用は各サイトの利用規約を確認してください。

## サポート

問題が発生した場合は、以下を確認してください：

1. ログファイル: `logs/data_collection_*.log`
2. セッションファイル: `data/checkpoints/session.json`
3. チェックポイント: `data/checkpoints/checkpoint_*.pkl`

---

**注意**: 本機能は倫理的・法的配慮を最優先しています。robots.txt違反、過度なアクセス、個人情報収集などは行いません。

