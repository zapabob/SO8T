# 付録D: チェックポイント戦略詳細

## チェックポイント設計思想

### 目的
1. **電源断リカバリー**: 予期せぬ停電・システム障害からの復旧
2. **実験追跡**: 学習過程の完全記録
3. **ベストモデル保存**: 最良性能モデルの保持
4. **ディスク容量管理**: 古いチェックポイント自動削除

## チェックポイント設定

### 基本設定

```yaml
checkpoint_config:
  interval: 180           # 3分（秒）
  max_keep: 5             # 最大保持数
  rotation: FIFO          # 削除方式（先入先出）
  compression: false      # 圧縮（無効、速度優先）
  async_save: true        # 非同期保存
```

### 保存トリガー

```python
# 時間ベース
if time.time() - last_checkpoint_time >= checkpoint_interval:
    save_checkpoint()

# ステップベース（代替案）
if global_step % checkpoint_steps == 0:
    save_checkpoint()

# 損失改善時（ベストモデル）
if val_loss < best_loss:
    save_best_model()
```

## チェックポイント構造

### ファイル形式

**命名規則**:
```
checkpoint_{session_id}_{global_step}.pt

例: checkpoint_20251106_120000_37500.pt
     ├─ セッションID: 20251106_120000
     └─ グローバルステップ: 37500
```

### 保存内容

```python
checkpoint = {
    # モデル状態
    'model_state_dict': model.state_dict(),
    
    # オプティマイザ状態
    'optimizer_state_dict': optimizer.state_dict(),
    
    # スケジューラ状態
    'scheduler_state_dict': scheduler.state_dict(),
    
    # 学習状態
    'epoch': current_epoch,
    'global_step': global_step,
    'best_loss': best_loss,
    
    # セッション情報
    'session_id': session_id,
    'timestamp': datetime.now().isoformat(),
    
    # 学習統計
    'training_stats': {
        'loss_history': loss_history,
        'pet_contribution': pet_contribution,
        'learning_rates': lr_history,
        'gpu_memory': gpu_memory_history
    },
    
    # SWA状態（75%以降）
    'swa_state': {
        'swa_model': swa_model_state,
        'swa_n': swa_n,
        'swa_active': swa_active
    },
    
    # ランダム状態（再現性）
    'random_state': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all()
    },
    
    # メタデータ
    'metadata': {
        'hostname': os.environ.get('COMPUTERNAME'),
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__
    }
}
```

### ファイルサイズ

```
model_state_dict: ~7GB（8bit圧縮後）
optimizer_state_dict: ~7GB（8bit状態）
その他: ~1GB
合計: ~15GB/チェックポイント

5個保持: ~75GB
```

## セッション管理

### セッションファイル

**場所**: `checkpoints/training/training_session.json`

**内容**:
```json
{
  "session_id": "20251106_120000",
  "start_time": 1699257600.0,
  "current_epoch": 2,
  "current_step": 25000,
  "total_steps": 37500,
  "best_loss": 0.245,
  "checkpoints": [
    "checkpoints/training/checkpoint_20251106_120000_30000.pt",
    "checkpoints/training/checkpoint_20251106_120000_32500.pt",
    "checkpoints/training/checkpoint_20251106_120000_35000.pt",
    "checkpoints/training/checkpoint_20251106_120000_37500.pt"
  ],
  "last_checkpoint": 1699270800.0
}
```

### 復旧プロトコル

```python
def recover_training():
    """
    学習復旧手順
    """
    # 1. セッションファイル読み込み
    if not session_file.exists():
        print("[INFO] No previous session, starting fresh")
        return None
    
    with open(session_file, 'r') as f:
        session = json.load(f)
    
    # 2. 最新チェックポイント取得
    latest_checkpoint = session['checkpoints'][-1]
    
    if not Path(latest_checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {latest_checkpoint}")
        return None
    
    # 3. チェックポイントロード
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # 4. モデル状態復元
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 5. 学習状態復元
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['global_step']
    best_loss = checkpoint['best_loss']
    
    # 6. ランダム状態復元（再現性）
    random.setstate(checkpoint['random_state']['python'])
    np.random.set_state(checkpoint['random_state']['numpy'])
    torch.set_rng_state(checkpoint['random_state']['torch'])
    torch.cuda.set_rng_state_all(checkpoint['random_state']['torch_cuda'])
    
    print(f"[OK] Resumed from epoch {start_epoch}, step {start_step}")
    
    return {
        'start_epoch': start_epoch,
        'start_step': start_step,
        'best_loss': best_loss
    }
```

## チェックポイントローテーション

### FIFO（First-In-First-Out）

```python
class CheckpointRotation:
    """
    チェックポイントローテーション管理
    """
    
    def __init__(self, max_keep: int = 5):
        self.max_keep = max_keep
        self.checkpoints = deque(maxlen=max_keep)
    
    def add_checkpoint(self, checkpoint_path: Path):
        """
        新規チェックポイント追加
        """
        # dequeに追加（自動的に古いものが削除される）
        if len(self.checkpoints) >= self.max_keep:
            # 最古のチェックポイントを削除
            oldest = self.checkpoints[0]
            if Path(oldest).exists():
                Path(oldest).unlink()
                print(f"[CLEAN] Removed old checkpoint: {oldest}")
        
        self.checkpoints.append(str(checkpoint_path))
        print(f"[SAVE] Checkpoint saved: {checkpoint_path}")
    
    def get_latest(self) -> Optional[Path]:
        """最新チェックポイント取得"""
        if self.checkpoints:
            return Path(self.checkpoints[-1])
        return None
```

## 非同期保存

### 実装

```python
import threading
import queue

class AsyncCheckpointSaver:
    """
    非同期チェックポイント保存
    学習を停止せずにバックグラウンドで保存
    """
    
    def __init__(self):
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(
            target=self._save_worker,
            daemon=True
        )
        self.save_thread.start()
    
    def _save_worker(self):
        """保存ワーカースレッド"""
        while True:
            try:
                checkpoint_data, save_path = self.save_queue.get(timeout=1)
                
                # ディスクに保存
                torch.save(checkpoint_data, save_path)
                print(f"[ASYNC] Saved: {save_path}")
                
                self.save_queue.task_done()
            
            except queue.Empty:
                continue
    
    def save_async(self, checkpoint_data: Dict, save_path: Path):
        """
        非同期保存キュー投入
        """
        # ディープコピー（学習継続のため）
        checkpoint_copy = {
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in checkpoint_data.items()
        }
        
        self.save_queue.put((checkpoint_copy, save_path))
        print(f"[ASYNC] Queued for save: {save_path}")
```

**利点**:
- 学習中断なし（保存時間 ~30秒を節約）
- スループット向上（~5%）

**注意点**:
- メモリ一時的に2倍使用（コピーのため）
- ディスクI/O帯域に注意

## ベストモデル保存

### 戦略

```python
def save_if_best(val_loss: float, best_loss: float, model, tokenizer):
    """
    最良モデル保存
    """
    if val_loss < best_loss:
        best_loss = val_loss
        
        save_path = output_dir / "best_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # モデル保存（HuggingFace形式）
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # メタデータ保存
        metadata = {
            'best_loss': best_loss,
            'epoch': current_epoch,
            'step': global_step,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path / 'best_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[BEST] New best model saved (loss: {val_loss:.4f})")
    
    return best_loss
```

## エラーハンドリング

### チェックポイント破損検出

```python
def verify_checkpoint(checkpoint_path: Path) -> bool:
    """
    チェックポイント整合性確認
    """
    try:
        # ロード試行
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 必須キー確認
        required_keys = [
            'model_state_dict',
            'optimizer_state_dict',
            'epoch',
            'global_step'
        ]
        
        for key in required_keys:
            if key not in checkpoint:
                print(f"[ERROR] Missing key: {key}")
                return False
        
        # state_dict整合性
        if not isinstance(checkpoint['model_state_dict'], dict):
            print("[ERROR] Invalid model_state_dict")
            return False
        
        print(f"[OK] Checkpoint verified: {checkpoint_path}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Checkpoint verification failed: {e}")
        return False
```

### 自動修復

```python
def auto_repair_checkpoints():
    """
    破損チェックポイント自動修復
    """
    checkpoint_dir = Path("checkpoints/training")
    
    for ckpt_file in checkpoint_dir.glob("checkpoint_*.pt"):
        if not verify_checkpoint(ckpt_file):
            print(f"[REPAIR] Removing corrupted: {ckpt_file}")
            ckpt_file.unlink()
    
    # セッションファイル再構築
    rebuild_session_file()
```

## バックアップ戦略

### ローカルバックアップ

```python
def backup_checkpoint(checkpoint_path: Path, backup_dir: Path):
    """
    チェックポイントバックアップ
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    backup_path = backup_dir / checkpoint_path.name
    
    # コピー
    shutil.copy2(checkpoint_path, backup_path)
    
    print(f"[BACKUP] Copied to {backup_path}")
```

### クラウドバックアップ（オプション、暗号化）

```python
def encrypt_and_upload(checkpoint_path: Path, cloud_bucket: str):
    """
    暗号化してクラウドにバックアップ
    """
    from cryptography.fernet import Fernet
    
    # 暗号化キー（環境変数から）
    key = os.getenv('BACKUP_ENCRYPTION_KEY')
    fernet = Fernet(key)
    
    # ファイル読み込み
    with open(checkpoint_path, 'rb') as f:
        data = f.read()
    
    # 暗号化
    encrypted = fernet.encrypt(data)
    
    # クラウドアップロード（S3/Azure/GCS等）
    upload_to_cloud(encrypted, cloud_bucket)
    
    print(f"[CLOUD] Encrypted backup uploaded")
```

## 緊急保存機構

### シグナルハンドラー

```python
import signal

class EmergencySaver:
    """
    緊急保存システム
    """
    
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # シグナルハンドラー登録
        signal.signal(signal.SIGINT, self.emergency_handler)
        signal.signal(signal.SIGTERM, self.emergency_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self.emergency_handler)
    
    def emergency_handler(self, signum, frame):
        """
        緊急保存ハンドラー
        """
        print(f"\n[EMERGENCY] Signal {signum} received")
        print("[SAVE] Emergency checkpoint saving...")
        
        emergency_path = Path("checkpoints/emergency_latest.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'signal': signum
        }
        
        torch.save(checkpoint, emergency_path)
        
        print(f"[OK] Emergency checkpoint saved: {emergency_path}")
        print("[EXIT] Exiting gracefully")
        
        sys.exit(0)
```

### Windows電源イベント対応

```powershell
# PowerShellで電源イベント監視
Register-WmiEvent -Query "SELECT * FROM Win32_PowerManagementEvent" -Action {
    Write-Host "[POWER] Power event detected, triggering emergency save"
    
    # Pythonプロセスに緊急保存シグナル送信
    $pythonPid = Get-Content "logs\training.pid"
    Stop-Process -Id $pythonPid -PassThru
}
```

## チェックポイント検証

### 整合性チェック

```python
def validate_checkpoint_integrity(checkpoint_path: Path) -> Dict[str, bool]:
    """
    チェックポイント完全性検証
    """
    checks = {
        'file_exists': False,
        'loadable': False,
        'has_model': False,
        'has_optimizer': False,
        'state_dict_valid': False
    }
    
    # ファイル存在確認
    checks['file_exists'] = checkpoint_path.exists()
    if not checks['file_exists']:
        return checks
    
    try:
        # ロード確認
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checks['loadable'] = True
        
        # 必須キー確認
        checks['has_model'] = 'model_state_dict' in checkpoint
        checks['has_optimizer'] = 'optimizer_state_dict' in checkpoint
        
        # state_dict妥当性
        if checks['has_model']:
            state_dict = checkpoint['model_state_dict']
            checks['state_dict_valid'] = (
                isinstance(state_dict, dict) and
                len(state_dict) > 0 and
                all(isinstance(v, torch.Tensor) for v in state_dict.values())
            )
    
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
    
    return checks
```

## 復旧テスト

### 定期復旧テスト

```python
def test_checkpoint_recovery():
    """
    チェックポイント復旧テスト（毎週実行推奨）
    """
    print("[TEST] Checkpoint recovery test")
    
    # 最新チェックポイント取得
    checkpoint_dir = Path("checkpoints/training")
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    
    if not checkpoints:
        print("[ERROR] No checkpoints found")
        return False
    
    latest = checkpoints[-1]
    
    # 復旧試行
    try:
        checkpoint = torch.load(latest, map_location='cpu')
        
        # ダミーモデルで復元テスト
        test_model = create_model()
        test_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 前向き計算テスト
        test_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = test_model(test_input)
        
        print(f"[OK] Recovery test passed: {latest}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Recovery test failed: {e}")
        return False
```

## パフォーマンス最適化

### 保存時間削減

**手法1: 選択的保存**
```python
# オプティマイザ状態を一部のチェックポイントのみ保存
if global_step % (checkpoint_interval * 5) == 0:
    # 完全保存
    save_full_checkpoint()
else:
    # モデルのみ保存（軽量）
    save_model_only()
```

**手法2: 圧縮保存**
```python
import zipfile

def save_compressed(checkpoint: Dict, save_path: Path):
    # 一時保存
    temp_path = save_path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    
    # ZIP圧縮
    with zipfile.ZipFile(save_path.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(temp_path, temp_path.name)
    
    # 一時ファイル削除
    temp_path.unlink()
    
    print(f"[COMPRESS] Saved compressed: {save_path}")
```

**圧縮率**: 15GB → 8GB（~50%削減）

### ロード時間削減

**手法: mmap（メモリマップ）**
```python
checkpoint = torch.load(
    checkpoint_path,
    map_location='cpu',
    mmap=True  # メモリマップ使用
)
```

**効果**:
- ロード時間: 60秒 → 10秒（6倍高速化）
- メモリピーク削減

## チェックポイント統計

### 保存頻度分析

```
学習時間: 33時間 = 1,980分
チェックポイント間隔: 3分
総保存回数: 1,980 / 3 = 660回

保持数: 5個
削除回数: 660 - 5 = 655回

総書き込み容量: 660 × 15GB = 9.9TB
実使用容量: 5 × 15GB = 75GB
```

### I/O負荷

```
書き込み速度: 15GB / 30秒 = 500MB/s
NVMe SSD推奨（3000MB/s以上）
```

## トラブルシューティング

### 問題1: 保存に時間がかかりすぎる

**症状**: チェックポイント保存に5分以上

**原因**: HDD使用、断片化

**対策**:
- SSD使用（NVMe推奨）
- 非同期保存有効化
- 圧縮保存検討

### 問題2: ディスク容量不足

**症状**: No space left on device

**対策**:
```bash
# 古いチェックポイント手動削除
Remove-Item checkpoints\training\checkpoint_*.pt -Exclude checkpoint_*_latest*.pt

# ログローテーション
Remove-Item logs\*.log -OlderThan (Get-Date).AddDays(-7)
```

### 問題3: 復旧失敗

**症状**: チェックポイントロードでエラー

**対策**:
```python
# 複数チェックポイント試行
checkpoints = sorted(glob("checkpoint_*.pt"), reverse=True)

for ckpt in checkpoints:
    try:
        load_checkpoint(ckpt)
        print(f"[OK] Recovered from {ckpt}")
        break
    except:
        print(f"[ERROR] Failed to load {ckpt}, trying next...")
        continue
```

---

**付録D終了**
