# -*- coding: utf-8 -*-
"""
Config Loader Utility for Moonshot Pipeline
Robustly loads configuration files from multiple search paths.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .path_resolver import PathResolver

logger = logging.getLogger(__name__)

class ConfigLoader:
    @staticmethod
    def load_json(config_name: str, required: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load a JSON config file by name (e.g., 'dataset.json').
        Searches in:
        1. <project_root>/config/
        2. <project_root>/src/config/
        3. <project_root>/src/infrastructure/config/
        """
        root = PathResolver.get_project_root()
        search_paths = [
            root / "config" / config_name,
            root / "src" / "config" / config_name,
            root / "src" / "infrastructure" / "config" / config_name,
        ]

        for path in search_paths:
            if path.exists():
                try:
                    logger.info(f"[ConfigLoader] Loading config from: {path}")
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"[ConfigLoader] Failed to parse {path}: {e}")
                    if required:
                        raise e
                    return None

        error_msg = f"Configuration file '{config_name}' not found in search paths: {[str(p) for p in search_paths]}"
        logger.error(f"[ConfigLoader] {error_msg}")
        
        if required:
            raise FileNotFoundError(error_msg)
        
        return None
