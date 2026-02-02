# 新セッション用メタプロンプト

## コンテキスト

**Moonshot Pipeline v3.0** 監視システムを開発中。

### 現在の作業ディレクトリ
```
C:\Users\downl\Desktop\SO8T\OpenCode
```

### 開発中のシステム
SO(8)四重推論訓練パイプライン用の包括的監視システム

## 完了した作業

### 1. 監視システム基盤実装
- YAML設定ファイル（`config/monitoring_config.yaml`）
- 設定ローダー（環境変数置換対応）
- 指標収集モジュール（全指標対応）
- LINE通知モジュール（詳細フォーマットB対応）
- 訓練モニター本体

### 2. テスト実装
- ドライランテストスイート
- Pytestユニットテスト（全モジュール）

### 3. 起動スクリプト
- メインエントリポイント（`training_monitor.py`）
- 各種CLIオプション対応

## 現在のステータス

| テストスイート | 結果 |
|---------------|------|
| Config | ✅ 2/2 パス |
| LINE Notifier | ✅ 3/3 パス |
| Metrics Collector | ⚠️ 1/4 パス（修正中）|
| Training Monitor | ⚠️ 0/4 パス（修正中）|

## 問題点と修正が必要な部分

### 1. ETACalculatorの`_last_step_step`変数
`metrics_collector.py`で`_last_step_time`と`_last_step_step`を区別する必要がある

### 2. Training Monitorテストのインポート
`training_monitor_core.py`とテスト間のインポート修正

## 新セッションでの優先タスク

1. **テスト修正** - 全テストをパスさせる
   - `scripts/monitoring/modules/dry_run_suite.py`を実行
   - 失敗しているテストを修正

2. **再テスト** - 修正後、ドライランを再実行
   ```bash
   cd OpenCode && py -3 scripts/monitoring/modules/dry_run_suite.py
   ```

3. **フルパイプライン実行** - 訓練モニターの完全テスト
   ```bash
   py -3 scripts/monitoring/training_monitor.py --skip-training
   ```

## 参照すべきファイル

### 設定
- `config/monitoring_config.yaml` - 監視設定
- `scripts/monitoring/modules/config_loader.py` - 設定ローダー

### コアモジュール
- `scripts/monitoring/modules/metrics_collector.py` - 指標収集
- `scripts/monitoring/modules/line_notifier.py` - LINE通知
- `scripts/monitoring/modules/training_monitor_core.py` - 訓練モニター

### テスト
- `scripts/monitoring/modules/dry_run_suite.py` - ドライランテスト
- `tests/monitoring/` - ユニットテスト

### ドキュメント
- `_docs/2026-02-03_monitoring_system_v3_implementation.md` - 本実装記録
- `docs/OpenCode_METAPROMPT.md` - ワークツリー指針
- `docs/implementation_log_v3.md` - v3実装ログ

## 新セッション開始時の質問

1. `scripts/monitoring/modules/dry_run_suite.py` を実行し、現在の結果を確認してください
2. 失敗しているテストの原因を特定してください
3. 修正を実施し、再度テストを実行してください
4. 全テストがパスするまで繰り返しください

## 技術的メモ

### ETACalculator修正済みコード
```python
# updateメソッド内の修正
if self._last_step_time is not None and step > self._last_step_step:
    steps_completed = step - self._last_step_step
    # ...

# __init__に追加
self._last_step_step: int = 0

# resetに追加
self._last_step_step = 0
```

### training_monitor_core.pyのインポート修正
```python
# get_configをconfig_loaderから参照
config = config_loader.get_config(config_path)
```

## 環境のセットアップ

```powershell
# Conda環境確認
conda info --envs

# UV環境確認
uv --version

# Pythonバージョン確認
python --version
```

## 重要なファイルパス

| ファイル | パス |
|---------|------|
| プロジェクトルート | `C:\Users\downl\Desktop\SO8T\OpenCode` |
| 監視設定 | `C:\Users\downl\Desktop\SO8T\OpenCode\config\monitoring_config.yaml` |
| 監視モジュール | `C:\Users\downl\Desktop\SO8T\OpenCode\scripts\monitoring\modules\` |
| テスト | `C:\Users\downl\Desktop\SO8T\OpenCode\tests\monitoring\` |
| 実装記録 | `C:\Users\downl\Desktop\SO8T\OpenCode\_docs\2026-02-03_monitoring_system_v3_implementation.md` |

## コマンド一覧

```bash
# ドライランテスト実行
cd OpenCode && py -3 scripts/monitoring/modules/dry_run_suite.py

# モニター起動（訓練スキップ）
py -3 scripts/monitoring/training_monitor.py --skip-training

# ステータス確認
py -3 scripts/monitoring/training_monitor.py --status

# LINE接続テスト
py -3 scripts/monitoring/training_monitor.py --test-line --line-token=YOUR_TOKEN

# Pytest実行
cd OpenCode && py -3 -m pytest tests/monitoring/ -v
```
