# -*- coding: utf-8 -*-
"""
Common CUDA Utilities
共通CUDAユーティリティ
"""

import torch
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def check_cuda_availability() -> bool:
    """
    CUDA利用可能性を確認

    Returns:
        CUDAが利用可能かどうか
    """
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            logger.info(f"CUDA is available: {device_count} device(s)")
            logger.info(f"Current device: {current_device} ({device_name})")

            # GPUメモリ情報
            memory_info = get_gpu_memory_info()
            if memory_info:
                logger.info(f"GPU memory: {memory_info['used']:.1f}GB / {memory_info['total']:.1f}GB")

            return True
        else:
            logger.warning("CUDA is not available")
            return False

    except Exception as e:
        logger.error(f"CUDA check failed: {e}")
        return False


def get_optimal_device() -> torch.device:
    """
    最適なデバイスを取得

    Returns:
        PyTorchデバイス
    """
    if torch.cuda.is_available():
        # 最もメモリの多いGPUを選択
        max_memory = 0
        best_device = 0

        for i in range(torch.cuda.device_count()):
            memory_info = get_gpu_memory_info(i)
            if memory_info and memory_info['free'] > max_memory:
                max_memory = memory_info['free']
                best_device = i

        device = torch.device(f'cuda:{best_device}')
        logger.info(f"Selected optimal device: {device}")
        return device
    else:
        logger.info("Using CPU device")
        return torch.device('cpu')


def get_gpu_memory_info(device: int = 0) -> Optional[Dict[str, float]]:
    """
    GPUメモリ情報を取得

    Args:
        device: GPUデバイスID

    Returns:
        メモリ情報辞書またはNone
    """
    try:
        if torch.cuda.is_available() and device < torch.cuda.device_count():
            torch.cuda.synchronize(device)  # 同期

            # メモリ情報取得
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB

            return {
                'total': total_memory,
                'allocated': allocated_memory,
                'reserved': reserved_memory,
                'free': total_memory - reserved_memory
            }
        else:
            return None

    except Exception as e:
        logger.error(f"Failed to get GPU memory info: {e}")
        return None


def optimize_cuda_settings() -> Dict[str, Any]:
    """
    CUDA設定を最適化

    Returns:
        最適化設定
    """
    settings = {}

    if torch.cuda.is_available():
        # cuDNN最適化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # メモリ効率設定
        settings['benchmark'] = True
        settings['deterministic'] = False

        logger.info("CUDA settings optimized for performance")
    else:
        logger.warning("CUDA not available, cannot optimize settings")

    return settings


def clear_cuda_cache() -> None:
    """CUDAキャッシュをクリア"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")
    else:
        logger.warning("CUDA not available, cannot clear cache")


def get_cuda_version() -> Optional[str]:
    """
    CUDAバージョンを取得

    Returns:
        CUDAバージョン文字列またはNone
    """
    try:
        return torch.version.cuda
    except Exception:
        return None


def check_cudnn_version() -> Optional[str]:
    """
    cuDNNバージョンを取得

    Returns:
        cuDNNバージョン文字列またはNone
    """
    try:
        return torch.backends.cudnn.version()
    except Exception:
        return None


def diagnose_cuda_setup() -> Dict[str, Any]:
    """
    CUDAセットアップの診断

    Returns:
        診断結果
    """
    diagnosis = {
        'cuda_available': check_cuda_availability(),
        'cuda_version': get_cuda_version(),
        'cudnn_version': check_cudnn_version(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'devices': []
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory': get_gpu_memory_info(i)
            }
            diagnosis['devices'].append(device_info)

    return diagnosis
