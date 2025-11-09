# Cursorブラウザ並列タブスクレイピング実装 実装ログ

## 実装情報
- **日付**: 2025-11-09
- **Worktree**: main
- **機能名**: Cursorブラウザ並列タブスクレイピング実装
- **実装者**: AI Agent

## 実装内容

### 1. 並列タブスクレイピングスクリプトの作成

**ファイル**: `scripts/data/cursor_parallel_tab_scraping.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: Cursorブラウザを使って10個のタブを並列処理し、各タブで10ページずつ計100ページをスクレイピングする機能を実装

#### 主要機能
- `CursorParallelTabScraper`クラスの実装
- タブごとの並列処理（asyncio.gather）
- 人間を模倣した動作（マウス移動、スクロール、待機、ホバー）
- ボット検知チェックと回避
- ページ遷移失敗時のリカバリー処理（ブラウザバックと別ページへの遷移）

#### 実装詳細
- **並列タブ処理**: 10個のタブを並列実行し、各タブで10ページずつスクレイピング
- **人間模倣動作**: `enhanced_human_behavior()`メソッドで高度な人間模倣動作を実装
  - ランダムなマウス軌跡（3-6回）
  - キーボード入力のシミュレート（Tabキー）
  - ウィンドウフォーカスのシミュレート
  - スクロールの不規則な動き
  - ページ要素への複数回ホバー
- **ボット検知チェック**: `detect_bot_checks()`メソッドで以下のチェックを実装
  - CAPTCHA検出
  - アクセス拒否検出
  - Cloudflare検出
  - ボット検知検出
  - レート制限検出
- **リカバリー処理**: `handle_check_failure()`メソッドでチェック失敗時の処理を実装
  - ブラウザバック実行
  - 別のページへの遷移
  - URLキューからの代替URL取得

### 2. 人間を模倣した動作の強化

**ファイル**: `scripts/data/cursor_parallel_tab_scraping.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 既存の`parallel_deep_research_scraping.py`と`human_like_web_scraping.py`を参考に実装

#### 実装詳細
- **マウス移動**: `human_like_mouse_move()`メソッドで滑らかなマウス軌跡を実装
- **人間模倣動作**: `enhanced_human_behavior()`メソッドで以下の動作を実装
  - ランダムなマウス軌跡（複雑な動き）
  - キーボード入力のシミュレート
  - ウィンドウフォーカスのシミュレート
  - スクロールの不規則な動き
  - ページ要素への複数回ホバー

### 3. ボット検知チェックと回避

**ファイル**: `scripts/data/cursor_parallel_tab_scraping.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 既存の`parallel_deep_research_scraping.py`の`detect_background_checks()`を参考に実装

#### 実装詳細
- **ボット検知チェック**: `detect_bot_checks()`メソッドで以下のチェックを実装
  - CAPTCHA検出（recaptcha, hcaptcha, turnstile等）
  - アクセス拒否検出（403, blocked等）
  - Cloudflare検出（checking your browser等）
  - ボット検知検出（verify you are human等）
  - レート制限検出（429, too many requests等）
- **リカバリー処理**: `handle_check_failure()`メソッドでチェック失敗時の処理を実装
  - ブラウザバック実行
  - URLキューからの代替URL取得
  - 別のページへの遷移

### 4. タブごとの並列処理実装

**ファイル**: `scripts/data/cursor_parallel_tab_scraping.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: asyncio.gatherを使用して10個のタブを並列処理

#### 実装詳細
- **並列タブ処理**: `scrape_with_parallel_tabs()`メソッドで10個のタブを並列実行
- **タブごとのスクレイピング**: `scrape_tab()`メソッドで各タブで10ページずつスクレイピング
- **URLキュー管理**: 各タブで独立したURLキューを管理
- **リンク抽出**: `extract_links()`メソッドでページからリンクを抽出してURLキューに追加
- **エラーハンドリング**: 連続失敗時のリカバリー処理を実装

### 5. パイプラインへの統合

**ファイル**: `scripts/pipelines/unified_master_pipeline.py` (修正)  
**ファイル**: `configs/unified_master_pipeline_config.yaml` (修正)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: Phase 1に並列タブスクレイピングオプションを追加

#### 実装詳細
- **設定ファイル追加**: `configs/unified_master_pipeline_config.yaml`に以下の設定を追加
  - `use_parallel_tabs`: 並列タブスクレイピングを使用するか（デフォルト: false）
  - `num_tabs`: タブ数（デフォルト: 10）
  - `pages_per_tab`: タブあたりのページ数（デフォルト: 10）
  - `total_pages`: 総ページ数（デフォルト: 100）
- **パイプライン統合**: `scripts/pipelines/unified_master_pipeline.py`の`phase1_parallel_scraping()`メソッドを修正
  - `use_parallel_tabs`がtrueの場合、`cursor_parallel_tab_scraping.py`を実行
  - 既存のSO8T/thinkingモデル統制スクレイピングと統合

## 作成・変更ファイル
- `scripts/data/cursor_parallel_tab_scraping.py` (新規作成)
- `scripts/pipelines/unified_master_pipeline.py` (修正)
- `configs/unified_master_pipeline_config.yaml` (修正)

## 設計判断
- **並列処理**: asyncio.gatherを使用して10個のタブを並列実行することで、効率的なスクレイピングを実現
- **人間模倣動作**: 既存の実装を参考に、高度な人間模倣動作を実装してボット検知を回避
- **リカバリー処理**: ボット検知チェック失敗時にブラウザバックと別ページへの遷移を実装して、スクレイピングの継続性を確保
- **パイプライン統合**: Phase 1に並列タブスクレイピングオプションを追加し、既存のスクレイピングと統合

## テスト結果
- 未実施（実装完了後、テストが必要）

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

### 並列タブスクレイピング運用
- **リモートデバッグポート**: 各タブで異なるポートを使用（base_port + tab_index）
- **メモリ使用量**: 10タブ並列処理のため、メモリ使用量に注意
- **ボット検知回避**: 適切な待機時間を設定してボット検知を回避
- **エラーハンドリング**: 連続失敗時のリカバリー処理を実装
