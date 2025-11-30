#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
SO8T WebDataset Access Helper
H:\from_D\webdatasetへのアクセスを支援するヘルパーモジュール

著者: AI Agent (峯岸亮ボブにゃん理論実装)
日付: 2025-11-30
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, List
import warnings

class WebDatasetAccessor:
    """H:\from_D\webdatasetへのアクセスを管理するクラス"""

    def __init__(self, external_path: str = r"H:\from_D\webdataset"):
        """
        WebDatasetアクセッサーの初期化

        Args:
            external_path: 外部ストレージのパス（デフォルト: H:\from_D\webdataset）
        """
        self.external_path = Path(external_path)
        self.local_symlink = Path("webdataset")

        # 外部ストレージの存在確認
        if not self.external_path.exists():
            warnings.warn(f"External webdataset path does not exist: {self.external_path}")
            self.external_available = False
        else:
            self.external_available = True
            print(f"External webdataset found at: {self.external_path}")

    def get_path(self, subdir: str = "", use_symlink: bool = True) -> Path:
        """
        webdataset内のパスを取得

        Args:
            subdir: サブディレクトリパス
            use_symlink: シンボリックリンクを使用するかどうか

        Returns:
            Path: アクセス可能なパス
        """
        if use_symlink and self.local_symlink.exists() and self.local_symlink.is_symlink():
            # シンボリックリンクが存在する場合
            return self.local_symlink / subdir
        elif self.external_available:
            # 外部ストレージに直接アクセス
            return self.external_path / subdir
        else:
            # フォールバック: ローカルのwebdatasetディレクトリ
            return Path("webdataset") / subdir

    def ensure_external_access(self) -> bool:
        """外部ストレージへのアクセスが可能か確認"""
        return self.external_available

    def list_datasets(self, category: Optional[str] = None) -> List[str]:
        """
        利用可能なデータセットをリストアップ

        Args:
            category: カテゴリでフィルタリング（例: 'datasets', 'models'）

        Returns:
            List[str]: データセット名のリスト
        """
        datasets = []

        try:
            if category:
                datasets_path = self.get_path(category)
            else:
                datasets_path = self.get_path("datasets")

            if datasets_path.exists():
                datasets = [d.name for d in datasets_path.iterdir() if d.is_dir()]
        except Exception as e:
            print(f"Error listing datasets: {e}")

        return sorted(datasets)

    def get_dataset_path(self, dataset_name: str) -> Path:
        """データセットのパスを取得"""
        return self.get_path(f"datasets/{dataset_name}")

    def get_checkpoint_path(self, checkpoint_type: str = "training", run_name: str = "") -> Path:
        """チェックポイントのパスを取得"""
        if run_name:
            return self.get_path(f"checkpoints/{checkpoint_type}/{run_name}")
        else:
            return self.get_path(f"checkpoints/{checkpoint_type}")

    def get_model_path(self, model_name: str = "", model_type: str = "final") -> Path:
        """モデル格納パスを取得"""
        if model_name:
            return self.get_path(f"models/{model_type}/{model_name}")
        else:
            return self.get_path(f"models/{model_type}")

    def get_gguf_path(self, model_name: str = "") -> Path:
        """GGUFモデル格納パスを取得"""
        if model_name:
            return self.get_path(f"gguf_models/{model_name}")
        else:
            return self.get_path("gguf_models")

    def create_checkpoint_dir(self, checkpoint_type: str = "training", run_name: str = "") -> Path:
        """チェックポイントディレクトリを作成"""
        checkpoint_path = self.get_checkpoint_path(checkpoint_type, run_name)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        return checkpoint_path

    def create_model_dir(self, model_name: str, model_type: str = "final") -> Path:
        """モデルディレクトリを作成"""
        model_path = self.get_model_path(model_name, model_type)
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    def create_gguf_dir(self, model_name: str) -> Path:
        """GGUFディレクトリを作成"""
        gguf_path = self.get_gguf_path(model_name)
        gguf_path.mkdir(parents=True, exist_ok=True)
        return gguf_path

# グローバルインスタンス
_accessor = None

def get_webdataset_accessor() -> WebDatasetAccessor:
    """WebDatasetアクセッサーのグローバルインスタンスを取得"""
    global _accessor
    if _accessor is None:
        _accessor = WebDatasetAccessor()
    return _accessor

def get_path(subdir: str = "", use_symlink: bool = True) -> Path:
    """webdataset内のパスを取得（便利関数）"""
    accessor = get_webdataset_accessor()
    return accessor.get_path(subdir, use_symlink)

def get_dataset_path(dataset_name: str) -> Path:
    """データセットのパスを取得（便利関数）"""
    accessor = get_webdataset_accessor()
    return accessor.get_dataset_path(dataset_name)

def get_checkpoint_path(checkpoint_type: str = "training", run_name: str = "") -> Path:
    """チェックポイントのパスを取得（便利関数）"""
    accessor = get_webdataset_accessor()
    return accessor.get_checkpoint_path(checkpoint_type, run_name)

def get_model_path(model_name: str = "", model_type: str = "final") -> Path:
    """モデル格納パスを取得（便利関数）"""
    accessor = get_webdataset_accessor()
    return accessor.get_model_path(model_name, model_type)

def get_gguf_path(model_name: str = "") -> Path:
    """GGUFモデル格納パスを取得（便利関数）"""
    accessor = get_webdataset_accessor()
    return accessor.get_gguf_path(model_name)

def list_available_datasets(category: Optional[str] = None) -> List[str]:
    """利用可能なデータセットをリストアップ（便利関数）"""
    accessor = get_webdataset_accessor()
    return accessor.list_datasets(category)

# テスト関数
def test_access():
    """アクセス機能をテスト"""
    print("Testing WebDataset access...")

    accessor = get_webdataset_accessor()
    print(f"External storage available: {accessor.ensure_external_access()}")

    # パス取得テスト
    print(f"Dataset path: {get_dataset_path('test_dataset')}")
    print(f"Checkpoint path: {get_checkpoint_path('training', 'test_run')}")
    print(f"Model path: {get_model_path('test_model')}")
    print(f"GGUF path: {get_gguf_path('test_model')}")

    # データセットリストテスト
    datasets = list_available_datasets()
    print(f"Available datasets: {datasets}")

    print("WebDataset access test completed!")

if __name__ == "__main__":
    test_access()
