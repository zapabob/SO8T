# 🤖 AEGIS完全自動無人化システム

## 概要

AEGIS (Autonomous Expert Guided Intelligence System) は、SO8Tプロジェクトの完全自動無人運用を実現するシステムです。電源投入時から学習完了まで、全てのプロセスが自動的に実行され、チェックポイントによる中断復旧機能が備わっています。

## 🚀 特徴

### ✅ 完全自動化
- **電源投入時自動起動**: Windows起動と同時にトレーニング開始
- **無人運用**: 人間の介入不要で24時間365日稼働
- **自己復旧**: クラッシュや電源断からの自動再開

### ✅ 包括的チェックポイント
- **RLPO学習**: 3分間隔でチェックポイント保存（5個ローリングストック）
- **GGUF変換**: 変換進捗を追跡し、中断から再開可能
- **全プロセス**: データセット作成、評価、レポート生成全てに適用

### ✅ システム監視
- **リアルタイム監視**: CPU、メモリ、GPU、ネットワーク状態を常時監視
- **自動復旧**: プロセス停止時の自動再起動
- **健康診断**: システム状態の定期レポート生成

## 🛠️ インストール

### 1. フルセットアップ実行
```batch
# 管理者権限で実行
.\setup_full_automation.bat
```

このコマンドで以下の設定が自動的に行われます：
- Windowsタスクスケジューラへの登録（起動時 + 毎日午前2時）
- システム監視デーモンのスタートアップ登録
- 全てのコンポーネントのテスト

### 2. 個別設定（オプション）

#### Windowsタスクスケジューラのみ設定
```powershell
powershell -ExecutionPolicy Bypass -File "setup_autonomous_system.ps1" -Install
```

#### システム監視デーモンのみ設定
```batch
python scripts/utils/system_monitor.py --daemon
```

## 🎯 使用方法

### 自動運用（推奨）
システムが自動的に以下のサイクルを繰り返します：
1. **電源投入時**: 自動起動
2. **環境チェック**: Python, PyTorch, CUDA検証
3. **データセット更新**: 不足データの自動作成
4. **RLPO学習**: SO8Tアダプター付き学習（3分間隔チェックポイント）
5. **GGUF変換**: 学習済みモデルのGGUF変換（進捗追跡）
6. **評価実行**: 変換済みモデルの性能評価
7. **レポート生成**: 学習結果の自動レポート作成
8. **待機**: 次回の電源投入またはスケジュール起動まで待機

### 手動制御

#### 即時実行
```batch
# 完全パイプライン実行
.\auto_aegis_pipeline.bat
```

#### 個別タスク実行
```batch
# RLPO学習のみ
python scripts/utils/task_manager.py rlpo --max_steps 10000

# GGUF変換のみ
python scripts/utils/task_manager.py gguf --model_path "checkpoints/rlpo_science_nsfw_automated/final_model"

# サンシャインパイプライン（比較実験）
python scripts/utils/task_manager.py sunshine --skip_baseline

# データセット作成
python scripts/utils/task_manager.py data --dataset_type science
```

#### システム監視
```batch
# 現在のシステム状態確認
python scripts/utils/system_monitor.py --status

# 失敗サービスの再起動
python scripts/utils/system_monitor.py --restart-services

# バックグラウンド監視開始
python scripts/utils/system_monitor.py --daemon
```

## 📊 チェックポイントシステム

### RLPO学習チェックポイント
- **保存間隔**: 3分ごと
- **保持数**: 最新5個
- **保存場所**: `checkpoints/rlpo_science_nsfw_automated/rolling_checkpoints/`
- **自動再開**: 中断時の最新チェックポイントから再開

### GGUF変換チェックポイント
- **保存タイミング**: 変換進捗更新時（約3分間隔）
- **追跡情報**: 変換ステージ、進捗率、処理済みテンソル数
- **保存場所**: `D:/webdataset/gguf_models/{model_name}/gguf_conversion_checkpoint.json`

### 全プロセスチェックポイント
- **タスクマネージャー**: すべてのタスクに汎用チェックポイント適用
- **デコレータ使用**: `@with_checkpointing` で任意の関数に適用可能
- **コンテキストマネージャ**: `checkpoint_context()` で適用

## 🔧 設定カスタマイズ

### チェックポイント間隔変更
```python
# scripts/utils/checkpoint_manager.py
save_interval_sec=180  # 秒単位（デフォルト: 3分）
```

### 保持チェックポイント数変更
```python
max_keep=5  # デフォルト: 5個
```

### スケジュール変更
```powershell
# setup_autonomous_system.ps1
$triggers += New-ScheduledTaskTrigger -Daily -At "02:00"  # 毎日午前2時
```

## 📈 監視とログ

### ログファイル
- `auto_training.log`: メイン実行ログ
- `system_monitor.log`: システム監視ログ
- `boot_history.log`: 起動履歴
- `system_status.log`: システム状態レポート

### リアルタイム監視
```batch
# システム状態表示
python scripts/utils/system_monitor.py --status

# 実行中プロセス確認
tasklist | findstr python
tasklist | findstr ollama
```

## 🚨 トラブルシューティング

### 自動起動しない場合
```batch
# タスクスケジューラ確認
powershell -ExecutionPolicy Bypass -File "setup_autonomous_system.ps1" -Test

# 手動起動テスト
.\auto_aegis_pipeline.bat
```

### チェックポイントが保存されない場合
```batch
# ディスク容量確認
dir | find "bytes free"

# 権限確認
icacls checkpoints
```

### システム監視が動作しない場合
```batch
# Pythonプロセス確認
tasklist | findstr python

# 監視デーモン再起動
python scripts/utils/system_monitor.py --daemon
```

## 🔄 アンインストール

### 完全アンインストール
```powershell
# タスクスケジューラから削除
powershell -ExecutionPolicy Bypass -File "setup_autonomous_system.ps1" -Uninstall

# スタートアップから削除
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\AEGIS_System_Monitor.lnk"
```

### 部分アンインストール
- タスクスケジューラのみ: `setup_autonomous_system.ps1 -Uninstall`
- 監視デーモンのみ: スタートアップショートカット削除

## 📋 システム要件

- **OS**: Windows 10/11 (管理者権限)
- **Python**: 3.8+ (PyTorch, Transformers, PEFT, TRL)
- **GPU**: NVIDIA GPU (CUDA 12.0+)
- **ストレージ**: 最低100GB (モデル保存用)
- **RAM**: 最低16GB

## 🎯 次のステップ

1. **拡張タスク**: 新しいタスクを `task_manager.py` に追加
2. **カスタム監視**: `system_monitor.py` を拡張して追加メトリクス監視
3. **リモート監視**: クラウドベースの監視システム統合
4. **マルチノード**: 分散学習環境での自動運用

## 📞 サポート

問題が発生した場合:
1. `system_monitor.log` と `auto_training.log` を確認
2. `python scripts/utils/system_monitor.py --status` を実行
3. ログを添付してIssueを作成

---

**🚀 AEGIS - 真の自律型AIシステムがここに誕生しました！**


