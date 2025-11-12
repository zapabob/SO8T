# 完全バックグラウンド並列DeepResearch Webスクレイピングパイプラインマネージャー実装

## 実装情報
- **日付**: 2025-11-08
- **Worktree**: main
- **機能名**: 完全バックグラウンド並列DeepResearch Webスクレイピングパイプラインマネージャー
- **実装者**: AI Agent

## 実装内容

### 1. 並列パイプラインマネージャー

**ファイル**: `scripts/data/parallel_pipeline_manager.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 10個のDeepResearch webスクレイピングパイプラインを完全バックグラウンドで並列実行するマネージャー。

- `ParallelPipelineManager`クラスを実装
- 10個のインスタンスを並列起動（各インスタンスは10個のブラウザを並列実行、合計100個のブラウザ）
- 完全バックグラウンド実行（デーモンモード）
- 各インスタンスは異なる出力ディレクトリ、リモートデバッグポート、ログファイルを使用
- 自動再起動機能（失敗したインスタンスを自動的に再起動）
- インスタンス状態監視（実行中、停止、失敗を監視）
- 状態ファイル保存（`parallel_pipeline_status.json`）

### 2. 起動バッチスクリプト

**ファイル**: `scripts/data/run_parallel_pipeline_manager.bat`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 並列パイプラインマネージャーを起動するバッチスクリプト。

- 10個のインスタンスを完全バックグラウンドで起動
- デーモンモードで実行
- 状態確認とログ確認のコマンドを表示

### 3. 停止バッチスクリプト

**ファイル**: `scripts/data/stop_parallel_pipeline_manager.bat`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 並列パイプラインマネージャーとすべてのインスタンスを停止するバッチスクリプト。

- PIDファイルから各インスタンスのプロセスを停止
- マネージャープロセスを停止
- クリーンアップ処理

### 4. 電源投入時自動起動設定

**ファイル**: `scripts/data/parallel_pipeline_manager.py` (変更)、`scripts/data/parallel_pipeline_manager_autostart.bat` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: 2025-11-09  
**備考**: Windowsタスクスケジューラに自動実行タスクを登録する機能を実装完了。`master_automated_pipeline.py`の実装を参考に、`setup_auto_start()`関数と`--setup`/`--run`引数を追加。Windowsタスクスケジューラの`/tr`オプションの261文字制限を回避するため、バッチファイル経由で実行する方式を採用。タスクスケジューラ実行時の作業ディレクトリ問題を解決するため、バッチファイル内で絶対パスを使用するように修正。

- `check_admin_privileges()`関数を追加（399-412行目）：管理者権限チェック
- `setup_auto_start()`関数を追加（415-493行目）：Windowsタスクスケジューラ登録
- `--setup`引数でタスクスケジューラ登録を実行（527-535行目）
- `--run`引数でタスクスケジューラから呼び出された場合の処理を追加（537-568行目）
- タスクスケジューラ用バッチファイル: `parallel_pipeline_manager_autostart.bat`（261文字制限回避のため、絶対パスで実行）
- タスク名: `SO8T-ParallelPipelineManager-AutoStart`
- トリガー: システム起動時 (`/sc onstart`)
- 遅延: 60秒（システム起動後、他のサービスが起動してから実行）
- 実行ユーザー: 現在のユーザー（`/ru`オプションを省略することでアクセス権限の問題を回避）
- デーモンモード: `--run`時は常にデーモンモードで起動
- **重要**: システム起動時のタスク作成には管理者権限が必要です。管理者権限で実行してください。

### 5. セットアップスクリプト（バッチ版）

**ファイル**: `scripts/data/setup_parallel_pipeline_manager.bat` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 並列パイプラインマネージャーの自動起動設定を行うバッチスクリプト。

- 管理者権限チェック
- `parallel_pipeline_manager.py --setup`を実行
- セットアップ完了確認

### 6. セットアップスクリプト（Python版）

**ファイル**: `scripts/data/setup_parallel_pipeline_manager.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 並列パイプラインマネージャーの自動起動設定を行うPythonスクリプト。

- 依存関係チェック
- タスクスケジューラ登録
- セットアップ確認

### 7. 統合セットアップスクリプト

**ファイル**: `scripts/pipelines/setup_all_auto_start.bat` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: `master_automated_pipeline`と`parallel_pipeline_manager`の両方を一括セットアップする統合スクリプト。

- `master_automated_pipeline`のセットアップ
- `parallel_pipeline_manager`のセットアップ
- 両方のタスク登録確認

## 作成・変更ファイル
- `scripts/data/parallel_pipeline_manager.py`（変更：自動起動機能追加）
- `scripts/data/parallel_pipeline_manager_autostart.bat`（新規作成：タスクスケジューラ用バッチファイル、261文字制限回避）
- `scripts/data/run_parallel_pipeline_manager.bat`（新規作成）
- `scripts/data/stop_parallel_pipeline_manager.bat`（新規作成）
- `scripts/data/setup_parallel_pipeline_manager.bat`（新規作成）
- `scripts/data/setup_parallel_pipeline_manager.ps1`（新規作成）
- `scripts/data/setup_parallel_pipeline_manager.py`（新規作成）
- `scripts/pipelines/setup_all_auto_start.bat`（新規作成）
- `scripts/pipelines/setup_all_auto_start.ps1`（新規作成）

## 設計判断

### 並列実行アーキテクチャ
- **10個のインスタンス**: 各インスタンスが独立して動作し、リソース競合を回避
- **各インスタンス10個のブラウザ**: 合計100個のブラウザで大規模スクレイピングを実現
- **異なる出力ディレクトリ**: 各インスタンスのデータを分離し、管理を容易に
- **異なるリモートデバッグポート**: ポート競合を回避（9222-9231）

### 完全バックグラウンド実行
- **デーモンモード**: Windowsの`CREATE_NEW_CONSOLE`フラグを使用して完全バックグラウンド実行
- **独立したログファイル**: 各インスタンスが独自のログファイルを持つ
- **PIDファイル管理**: プロセスIDを保存し、後で停止可能に

### 自動再起動機能
- **失敗検出**: プロセス状態を定期的にチェック
- **自動再起動**: 失敗したインスタンスを自動的に再起動
- **最大再起動回数**: 10回まで再起動を試行（無限ループを防止）

### リソース管理
- **インスタンスあたりのメモリ制限**: 8GB（デフォルト）
- **インスタンスあたりのCPU制限**: 80%（デフォルト）
- **動的リソース管理**: 各インスタンス内でリソースを動的に管理

## テスト結果
- 未実施（実装完了後、統合テストが必要）

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

### 並列実行運用
- **10個のインスタンス**: 各インスタンスは独立して動作
- **合計100個のブラウザ**: 大規模スクレイピングを実現
- **リソース管理**: 各インスタンスのメモリ・CPU使用量を監視
- **自動再起動**: 失敗したインスタンスを自動的に再起動
- **状態監視**: `parallel_pipeline_status.json`で状態を確認可能

### 起動・停止方法
- **起動**: `scripts/data/run_parallel_pipeline_manager.bat`を実行
- **停止**: `scripts/data/stop_parallel_pipeline_manager.bat`を実行
- **状態確認**: `logs/parallel_pipeline_status.json`を確認
- **ログ確認**: `logs/parallel_pipeline_manager.log`と`logs/parallel_instance_*.log`を確認

### 電源投入時自動起動設定
- **個別セットアップ**: `scripts/data/setup_parallel_pipeline_manager.bat`を管理者権限で実行
- **統合セットアップ**: `scripts/pipelines/setup_all_auto_start.bat`を管理者権限で実行（両方のパイプラインを一括セットアップ）
- **タスク名**: `SO8T-ParallelPipelineManager-AutoStart`
- **トリガー**: システム起動時（60秒遅延）
- **確認方法**: `schtasks /query /tn "SO8T-ParallelPipelineManager-AutoStart"`で確認

## 次のステップ
1. 統合テストの実施（10個のインスタンスの並列起動・動作確認）
2. リソース使用量の監視・最適化
3. 自動再起動機能のテスト
4. 大規模データでの動作確認
5. パフォーマンステスト（100個のブラウザでの動作確認）
6. 電源投入時自動起動の動作確認（システム再起動後の自動起動テスト）

