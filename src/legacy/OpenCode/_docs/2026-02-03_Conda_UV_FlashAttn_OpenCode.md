# Conda + uv + FlashAttention (CUDA12.8) 手順

- Worktree: OpenCode
- Timestamp: 2026-02-03T05:58:50.454955

## 前提
- Python 3.12.9
- CUDA 12.8
- RTX 3060

## Conda 環境作成
conda create -n so8t-py312 python=3.12.9 -y
conda activate so8t-py312

## CUDA/Toolchain
setx CUDA_HOME C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

## uv で依存をロックしてインストール
uv pip install -r requirements.txt
uv pip sync uv.lock

## FlashAttention（Windows）
uv pip install flash-attn

## 注意
- flash-attn はWindowsで失敗しやすいので、失敗時はログを_logsへ残す
- conda環境内で python -c import torch; print(torch.version.cuda) を確認

## 検証
python -c import torch, flash_attn; print(torch.__version__, torch.version.cuda)
