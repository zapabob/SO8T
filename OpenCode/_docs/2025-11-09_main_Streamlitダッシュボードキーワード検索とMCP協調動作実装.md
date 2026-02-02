# Streamlitダッシュボードキーワード検索とMCP協調動作実装ログ

## 実装情報
- **日付**: 2025-11-09
- **Worktree**: main
- **機能名**: Streamlitダッシュボードキーワード検索とMCP協調動作
- **実装者**: AI Agent

## 実装内容

### 1. キーワード共有メカニズムの実装

**ファイル**: `scripts/utils/keyword_coordinator.py`（新規作成）

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: キーワードキュー管理クラスを実装

- キーワードキュー管理クラス`KeywordCoordinator`を実装
- JSONファイルベースの共有メモリによるキーワード状態管理
- キーワード状態: `pending`, `assigned`, `processing`, `completed`, `failed`
- キーワードの割り当て状態を追跡
- キーワードの完了状態を管理
- ロック機構による並行アクセス制御（Windows対応）
- タイムアウト処理（割り当て後一定時間経過で再割り当て可能）
- リトライ機能（最大3回まで自動リトライ）

### 2. MCPサーバーを介したブラウザ間協調通信の実装

**ファイル**: `scripts/data/browser_coordinator.py`（新規作成）

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 中央コーディネータークラスを実装

- 中央コーディネータークラス`BrowserCoordinator`を実装
- MCPサーバーを介したブラウザ間メッセージング（フォールバック: 共有メモリ）
- キーワード割り当てロジック（重複回避）
- ブラウザ状態の共有（処理中キーワード、完了キーワード）
- ハートビート機能によるブラウザ生存確認
- メッセージタイプ: `keyword_assignment`, `keyword_completion`, `heartbeat`, `status_update`
- ブロードキャスト機能（全ブラウザへの通知）
- メッセージの順序保証と重複排除

### 3. Streamlitダッシュボードにキーワード入力機能を追加

**ファイル**: `scripts/dashboard/unified_scraping_monitoring_dashboard.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: キーワード入力フィールドと状態表示を追加

- キーワード入力フィールドを追加（複数キーワード対応、カンマ区切り）
- キーワード送信ボタンを追加
- 入力されたキーワードを共有メモリ（JSONファイル）に保存
- 現在処理中のキーワード一覧を表示
- キーワードごとの進捗状況を表示（総キーワード数、待機中、処理中、完了、失敗）
- キーワード状態テーブル（キーワード、状態、ブラウザID、追加時刻、割り当て時刻）

### 4. SO8TChromeDevDaemonManagerへの協調機能統合

**ファイル**: `scripts/data/so8t_chromedev_daemon_manager.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 協調機能とジャンル分類を統合

- `BrowserCoordinator`を統合
- キーワードキューからキーワードを取得
- キーワード割り当て状態をMCPサーバー経由で共有
- キーワード完了時に状態を更新
- 他のブラウザの状態を監視して重複を回避
- 協調機能の初期化と停止処理を追加

### 5. ジャンル分類の統合（DataLabeler + SO8T）

**ファイル**: `scripts/data/so8t_chromedev_daemon_manager.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: DataLabelerとSO8T分類器を統合

- `DataLabeler`によるキーワードベース分類（高速）
- SO8Tモデルによる分類（高精度、オプション）
- 分類結果をメタデータに追加
- 分類結果をMCPサーバー経由で共有（オプション）
- ジャンル分類の初期化処理を追加

### 6. パイプラインへのキーワード入力統合

**ファイル**: `scripts/pipelines/unified_master_pipeline.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: キーワードキューからキーワードを読み込み

- キーワードキューからキーワードを読み込み
- キーワードをURL生成に反映
- キーワードごとの進捗を追跡
- キーワード完了時にキューを更新
- Phase 1の`_phase1_so8t_chromedev_daemon_scraping`メソッドに統合

**ファイル**: `configs/unified_master_pipeline_config.yaml`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: キーワード協調動作設定を追加

- `keyword_coordination`セクションを追加
- `phase1_parallel_scraping`セクションに`keyword_coordination`設定を追加
- 設定項目: `enabled`, `keyword_queue_file`, `assignment_timeout`, `heartbeat_interval`, `mcp_coordination`

## 作成・変更ファイル

### 新規作成ファイル
- `scripts/utils/keyword_coordinator.py`: キーワード共有メカニズム
- `scripts/data/browser_coordinator.py`: MCPサーバーを介したブラウザ間協調通信

### 変更ファイル
- `scripts/dashboard/unified_scraping_monitoring_dashboard.py`: キーワード入力機能追加
- `scripts/data/so8t_chromedev_daemon_manager.py`: 協調機能とジャンル分類統合
- `scripts/pipelines/unified_master_pipeline.py`: キーワードキュー統合
- `configs/unified_master_pipeline_config.yaml`: キーワード協調動作設定追加

## 設計判断

### キーワード共有メカニズム
- JSONファイルベースの共有メモリを採用（`D:/webdataset/checkpoints/keyword_queue.json`）
- Windows環境でのファイルロック対応（`fcntl`はWindowsではスキップ）
- キーワード状態管理: `pending` → `assigned` → `processing` → `completed`/`failed`
- タイムアウト処理: 割り当て後1時間経過で自動的に`pending`に戻す
- リトライ機能: 最大3回まで自動リトライ

### MCPサーバー協調通信
- MCPサーバーを介したメッセージング（フォールバック: 共有メモリ）
- 共有メモリファイル: `D:/webdataset/checkpoints/browser_coordination_state.json`
- ハートビート間隔: 30秒
- メッセージ履歴: 最新100件のみ保持

### 重複回避ロジック
- キーワードレベルでの重複回避
- キーワード割り当て時の排他制御
- 他のブラウザが処理中のキーワードを確認してから割り当て
- タイムアウト処理とデッドロック回避

### ジャンル分類
- `DataLabeler`によるキーワードベース分類（高速、デフォルト）
- SO8Tモデルによる分類（高精度、オプション）
- 分類結果のキャッシュ（将来実装予定）
- 分類結果のMCPサーバー経由での共有（オプション）

## テスト結果

未実施（実装完了後、動作確認が必要）

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

### キーワード協調動作運用
- Streamlitダッシュボードからキーワードを入力すると、自動的にキューに追加される
- 各ブラウザはキューからキーワードを取得し、重複を回避しながら処理する
- キーワード完了時に自動的にキューが更新される
- MCPサーバーが利用できない場合は、共有メモリ（JSONファイル）で協調動作する

## 今後の改善点

1. キーワード分類結果のキャッシュ機能
2. キーワードごとの進捗状況の詳細追跡
3. キーワード優先度の設定機能
4. キーワード自動生成機能（既存キーワードから関連キーワードを生成）
5. キーワード統計情報の可視化（Streamlitダッシュボード）

