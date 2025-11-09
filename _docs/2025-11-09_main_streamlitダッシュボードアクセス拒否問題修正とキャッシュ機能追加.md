# Streamlitダッシュボードアクセス拒否問題修正とキャッシュ機能追加 実装ログ

## 実装情報
- **日付**: 2025-11-09
- **Worktree**: main
- **機能名**: Streamlitダッシュボードアクセス拒否問題修正とキャッシュ機能追加
- **実装者**: AI Agent

## 概要

Streamlitダッシュボードがアクセス拒否で見られない問題を修正し、外部アクセスを許可する設定を追加しました。また、サンプル数カウントと品質スコア計算のキャッシュ機能、動的閾値設定機能を追加しました。

## 実装内容

### 1. 設定ファイルへのダッシュボード設定追加

**ファイル**: `configs/unified_master_pipeline_config.yaml`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-09  
**備考**: なし

- Streamlitダッシュボード設定セクションを追加
- ホストとポートを設定可能にする
- デフォルトで`0.0.0.0`（すべてのインターフェース）を許可

**変更内容**:
```yaml
# Streamlitダッシュボード設定
dashboard:
  enabled: true
  host: "0.0.0.0"  # 外部アクセスを許可（0.0.0.0 = すべてのインターフェース）
  port: 8501
```

### 2. unified_master_pipeline.pyのstart_streamlit_dashboard()メソッド修正

**ファイル**: `scripts/pipelines/unified_master_pipeline.py`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-09  
**備考**: なし

- `--server.address 0.0.0.0`を追加して外部アクセスを許可
- 設定ファイルからホストとポートを読み込む機能を追加
- ログメッセージに外部アクセス用のURLを追加

**変更内容**:
- 設定からホストとポートを取得する処理を追加
- `--server.address`パラメータを追加
- 外部アクセス用のURLをログに出力

### 3. バッチファイルの修正

**ファイル**: 
- `scripts/dashboard/run_scraping_dashboard.bat`
- `scripts/dashboard/run_unified_dashboard.bat`
- `scripts/dashboard/run_unified_pipeline_dashboard.bat`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-09  
**備考**: なし

- `--server.address localhost`を`--server.address 0.0.0.0`に変更
- 外部アクセスを許可する設定に変更

**変更内容**:
```batch
REM Streamlitダッシュボードを起動
"%PYTHON%" -m streamlit run scripts\dashboard\so8t_scraping_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### 4. .streamlit/config.tomlの作成

**ファイル**: `.streamlit/config.toml`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-09  
**備考**: オプション機能

- Streamlitのデフォルト設定ファイルを作成
- 外部アクセスを許可する設定を追加

**変更内容**:
```toml
[server]
address = "0.0.0.0"
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true
```

### 5. サンプル数カウントのキャッシュ機能追加

**ファイル**: `scripts/pipelines/unified_master_pipeline.py`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-09  
**備考**: キャッシュ有効期限は5分間

- `_count_current_samples()`メソッドの結果をチェックポイントに保存
- 再起動時にチェックポイントからサンプル数を読み込んで再利用
- キャッシュの有効期限を設定（5分間）
- キャッシュが古い場合は再カウントを実行

**実装内容**:
- チェックポイントからキャッシュを読み込む処理を追加
- キャッシュの有効期限チェックを実装
- カウント結果をキャッシュに保存する処理を追加

### 6. 動的閾値設定機能の追加

**ファイル**: 
- `scripts/pipelines/unified_master_pipeline.py`
- `configs/unified_master_pipeline_config.yaml`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-09  
**備考**: デフォルトでは無効（`dynamic_threshold: false`）

- データセットの種類や品質に応じて、動的に最小サンプル数を調整する機能
- 設定ファイルでデータセットタイプごとの閾値を設定可能にする
- データセットの品質スコアに基づいて閾値を調整

**実装内容**:
- `_get_dynamic_threshold()`メソッドを追加
- `_analyze_dataset_type()`メソッドを追加
- Phase 3のデータセット量チェックで動的閾値を使用

**設定内容**:
```yaml
phase3_ab_test:
  dynamic_threshold: false  # 動的閾値設定を有効化（デフォルト: false）
  threshold_by_dataset_type:
    nsfw_detection: 30000
    general: 50000
    high_quality: 30000
    medium_quality: 50000
    low_quality: 100000
```

### 7. 品質スコア計算のキャッシュ機能追加

**ファイル**: `scripts/pipelines/unified_master_pipeline.py`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-09  
**備考**: キャッシュ有効期限は10分間、データセットが更新された場合は自動的に無効化

- `_calculate_quality_score()`メソッドの結果をチェックポイントに保存
- 再起動時にチェックポイントから品質スコアを読み込んで再利用
- キャッシュの有効期限を設定（10分間、データセットが更新された場合は無効化）
- キャッシュが古い場合は再計算を実行

**実装内容**:
- `_calculate_quality_score()`メソッドにキャッシュ機能を追加
- `_calculate_dataset_hash()`メソッドを追加（データセット変更検出用）
- `_compute_quality_metrics()`メソッドを追加（品質スコア計算）

**品質スコア計算ロジック**:
- 平均テキスト長（50-10000文字が適切）
- エンコーディングエラー率
- 総合的な品質スコア（0.0-1.0）

## 作成・変更ファイル

### 変更ファイル

1. **設定ファイル**:
   - `configs/unified_master_pipeline_config.yaml`: ダッシュボード設定と動的閾値設定を追加

2. **パイプラインスクリプト**:
   - `scripts/pipelines/unified_master_pipeline.py`: 
     - `start_streamlit_dashboard()`メソッドの修正
     - `_count_current_samples()`メソッドにキャッシュ機能追加
     - `_get_dynamic_threshold()`メソッド追加
     - `_analyze_dataset_type()`メソッド追加
     - `_calculate_quality_score()`メソッド追加
     - `_calculate_dataset_hash()`メソッド追加
     - `_compute_quality_metrics()`メソッド追加

3. **バッチファイル**:
   - `scripts/dashboard/run_scraping_dashboard.bat`: `--server.address 0.0.0.0`に変更
   - `scripts/dashboard/run_unified_dashboard.bat`: `--server.address 0.0.0.0`に変更
   - `scripts/dashboard/run_unified_pipeline_dashboard.bat`: `--server.address 0.0.0.0`に変更

### 新規作成ファイル

1. **Streamlit設定ファイル**:
   - `.streamlit/config.toml`: Streamlitのデフォルト設定ファイル

## 設計判断

### 1. 外部アクセス許可の設定

**理由**:
- `0.0.0.0`を指定することで、すべてのネットワークインターフェースでアクセス可能になる
- 同一ネットワーク内の他のデバイスからもアクセスできるようにするため

**利点**:
- リモートアクセスが可能になる
- モバイルデバイスからのアクセスが可能になる
- 開発・運用の柔軟性が向上する

**注意事項**:
- セキュリティ上の理由から、本番環境では適切なファイアウォール設定が必要
- Windowsファイアウォールでポートが許可されていることを確認

### 2. キャッシュ機能の実装

**理由**:
- サンプル数カウントと品質スコア計算は時間がかかる可能性がある
- チェックポイントに保存することで、再起動時にも再利用できる

**利点**:
- パフォーマンスの向上
- 再起動時の状態保持
- 不要な再計算の回避

**注意事項**:
- キャッシュの有効期限を適切に設定（サンプル数: 5分、品質スコア: 10分）
- データセットが更新された場合は自動的にキャッシュを無効化

### 3. 動的閾値設定機能

**理由**:
- データセットの種類や品質に応じて、適切な最小サンプル数を設定できる
- 高品質なデータセットの場合は、より少ないサンプル数でA/Bテストを実行できる

**利点**:
- データセットの特性に応じた柔軟な設定
- 効率的なパイプライン運用
- 品質に基づいた適切な判断

**注意事項**:
- デフォルトでは無効（`dynamic_threshold: false`）
- 有効化する場合は、適切な閾値を設定する必要がある

## テスト結果

### 実装完了項目

- [x] 設定ファイルへのダッシュボード設定追加
- [x] unified_master_pipeline.pyのstart_streamlit_dashboard()メソッド修正
- [x] バッチファイルの修正
- [x] .streamlit/config.tomlの作成
- [x] サンプル数カウントのキャッシュ機能追加
- [x] 動的閾値設定機能の追加
- [x] 品質スコア計算のキャッシュ機能追加

### リンターエラー

- エラーなし（`read_lints`で確認済み）

## 今後の拡張予定

1. **キャッシュ設定の外部化**:
   - キャッシュの有効期限を設定ファイルから読み込む機能

2. **品質スコア計算の改善**:
   - より詳細な品質メトリクスの計算
   - 重複率、ラベル品質などの追加

3. **セキュリティ強化**:
   - 認証機能の追加
   - HTTPS対応

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

### Streamlitダッシュボード運用
- 外部アクセスを許可する場合は、適切なファイアウォール設定が必要
- Windowsファイアウォールでポート8501（または設定したポート）を許可
- セキュリティ上の理由から、本番環境では認証機能の追加を検討

### キャッシュ運用
- サンプル数カウントのキャッシュは5分間有効
- 品質スコア計算のキャッシュは10分間有効
- データセットが更新された場合は自動的にキャッシュが無効化される

### 動的閾値設定運用
- デフォルトでは無効（`dynamic_threshold: false`）
- 有効化する場合は、適切な閾値を設定ファイルで設定
- データセットタイプと品質スコアに基づいて自動的に調整される

## 参考資料

- 実装計画: `.cursor/plans/so8t-web.plan.md`
- 関連ドキュメント: `_docs/2025-11-09_main_a/bテスト条件付きスキップ機能追加.md`

---

**実装完了日時**: 2025-11-09  
**Worktree**: main  
**実装者**: AI Agent
