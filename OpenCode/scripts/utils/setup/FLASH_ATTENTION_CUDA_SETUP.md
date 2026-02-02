# Flash Attention 2.5.0 インストール手順（WSL2 + CUDA_HOME設定）

## 概要

Flash Attention 2.5.0をWSL2環境にインストールするには、CUDA toolkitが必要です。PyTorchはCUDAライブラリをバンドルしていますが、flash-attentionのコンパイルには`nvcc`（CUDAコンパイラ）が必要です。

## 前提条件

- WSL2がインストールされている
- NVIDIAドライバーがWindows側にインストールされている
- PyTorch with CUDA 12.1がインストールされている
- sudo権限がある

## 方法1: CUDA Toolkitを自動インストール（推奨）

### ステップ1: CUDA Toolkitをインストール

```bash
wsl
sudo bash scripts/utils/setup/install_cuda_toolkit_wsl2.sh
```

このスクリプトは以下を実行します：
- CUDA repositoryの追加
- CUDA toolkit 12.1のインストール（~2GB、10-20分）
- 環境変数の設定（`/etc/profile.d/cuda.sh`）

### ステップ2: 環境変数を読み込む

```bash
source /etc/profile.d/cuda.sh
```

または、新しいターミナルを開く（自動的に読み込まれます）

### ステップ3: Flash Attentionをインストール

```bash
cd /mnt/c/Users/downl/Desktop/SO8T
bash scripts/utils/setup/install_flash_attn_wsl2_with_cuda.sh
```

## 方法2: 手動でCUDA_HOMEを設定（CUDA toolkitが既にインストールされている場合）

### ステップ1: CUDAの場所を確認

```bash
# 一般的な場所を確認
ls -la /usr/local/cuda*
ls -la /usr/local/cuda-12.1
ls -la /usr/local/cuda-12.0

# nvccの場所を確認
which nvcc
```

### ステップ2: CUDA_HOMEを設定

```bash
# CUDA toolkitが見つかった場合
export CUDA_HOME=/usr/local/cuda-12.1  # または実際のパス
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### ステップ3: Flash Attentionをインストール

```bash
cd /mnt/c/Users/downl/Desktop/SO8T
bash scripts/utils/setup/install_flash_attn_wsl2_with_cuda.sh
```

## 方法3: 手動でCUDA Toolkitをインストール

### ステップ1: CUDA repositoryを追加

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

### ステップ2: CUDA Toolkitをインストール

```bash
sudo apt-get install -y cuda-toolkit-12-1
```

### ステップ3: 環境変数を設定

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### ステップ4: Flash Attentionをインストール

```bash
cd /mnt/c/Users/downl/Desktop/SO8T
bash scripts/utils/setup/install_flash_attn_wsl2_with_cuda.sh
```

## トラブルシューティング

### 問題1: "CUDA toolkit not found"

**原因**: CUDA toolkitがインストールされていない

**解決策**: 方法1または方法3でCUDA toolkitをインストール

### 問題2: "nvcc not found"

**原因**: CUDA_HOMEが正しく設定されていない、またはPATHに追加されていない

**解決策**:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version  # 確認
```

### 問題3: "Permission denied" during installation

**原因**: sudo権限が必要

**解決策**: `sudo`を使用してインストールスクリプトを実行

### 問題4: "Insufficient memory" during compilation

**原因**: メモリ不足

**解決策**:
```bash
export MAX_JOBS=2  # 並列ジョブ数を減らす
bash scripts/utils/setup/install_flash_attn_wsl2_with_cuda.sh
```

### 問題5: "gcc/g++ not found"

**原因**: ビルドツールがインストールされていない

**解決策**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

## インストールの確認

```bash
python3 -c "from flash_attn import flash_attn_func; print('Flash Attention installed successfully!')"
```

## 注意事項

- CUDA toolkitのインストールには約2GBのディスク容量が必要です
- インストールには10-20分かかる場合があります
- flash-attentionのコンパイルには10-30分かかる場合があります
- メモリが不足する場合は、`MAX_JOBS`環境変数を設定して並列ジョブ数を制限してください

## 参考リンク

- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [WSL2 CUDA Setup Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

