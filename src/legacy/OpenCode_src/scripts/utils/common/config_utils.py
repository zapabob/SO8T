# -*- coding: utf-8 -*-
"""
Common Configuration Utilities
共通設定ユーティリティ
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path], config_type: Optional[str] = None) -> Dict[str, Any]:
    """
    設定ファイルを読み込み

    Args:
        config_path: 設定ファイルパス
        config_type: 設定ファイルタイプ（yaml/json/auto）

    Returns:
        設定データの辞書
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_type is None:
        config_type = config_path.suffix.lower()

    try:
        if config_type in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_type == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 自動検出
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                try:
                    config = json.loads(content)
                except json.JSONDecodeError:
                    config = yaml.safe_load(content)

        logger.info(f"Loaded config from {config_path}")
        return config if config else {}

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: Union[str, Path], config_type: Optional[str] = None) -> None:
    """
    設定ファイルを保存

    Args:
        config: 設定データ
        config_path: 保存先パス
        config_type: 設定ファイルタイプ
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_type is None:
        config_type = config_path.suffix.lower()

    try:
        if config_type in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif config_type == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            # JSONとして保存
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved config to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    複数の設定をマージ

    Args:
        *configs: マージする設定辞書

    Returns:
        マージ済み設定
    """
    result = {}

    for config in configs:
        if config:
            _deep_merge(result, config)

    return result


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    辞書を深くマージ（インプレース）

    Args:
        target: マージ先
        source: マージ元
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """
    設定の妥当性検証

    Args:
        config: 検証する設定
        schema: 検証スキーマ（オプション）

    Returns:
        妥当性
    """
    if not isinstance(config, dict):
        logger.error("Config must be a dictionary")
        return False

    # 基本的な検証
    required_keys = ['model', 'training'] if schema is None else schema.get('required', [])

    for key in required_keys:
        if key not in config:
            logger.error(f"Required config key missing: {key}")
            return False

    return True


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    設定から値を安全に取得（ドット区切りパス対応）

    Args:
        config: 設定辞書
        key_path: キーへのパス（例: "training.batch_size"）
        default: デフォルト値

    Returns:
        設定値またはデフォルト値
    """
    keys = key_path.split('.')
    current = config

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default
