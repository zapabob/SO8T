# WSL2環境での依存関係インストール手順

## 正しいコマンド

**注意**: コマンドは `wsl2` ではなく `wsl` です。

## 方法1: 手動でWSL2に入って実行（推奨）

1. **PowerShellまたはコマンドプロンプトを開く**

2. **WSL2を起動**:
   ```powershell
   wsl
   ```

3. **プロジェクトディレクトリに移動**:
   ```bash
   cd /mnt/c/Users/downl/Desktop/SO8T
   ```

4. **インストールスクリプトを実行**:
   ```bash
   bash scripts/utils/setup/install_all_dependencies_wsl2.sh
   ```

## 方法2: バッチファイルを実行

PowerShellまたはコマンドプロンプトから:

```powershell
scripts\utils\setup\run_install_wsl2.bat
```

## 方法3: 直接コマンドで実行

```powershell
wsl bash scripts/utils/setup/install_all_dependencies_wsl2.sh
```

## WSL2が起動しない場合

1. **WSL2を再起動**:
   ```powershell
   wsl --shutdown
   ```
   数秒待ってから再度 `wsl` を実行

2. **WSL2の状態を確認**:
   ```powershell
   wsl --list --verbose
   ```

3. **WSL2がインストールされているか確認**:
   ```powershell
   wsl --status
   ```

## インストール内容

- pip, uv
- PyTorch with CUDA 12.1
- すべての依存関係 (requirements.txt)
- flash-attention 2.5.0

## 予想される時間

30-60分（システムによって異なります）

