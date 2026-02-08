# -*- coding: utf-8 -*-
"""
RTX 3060 Optimized Dataset Pipeline for Moonshot
Handles memory-efficient dataset loading and preparation for AEGIS v3.0.
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from datasets import load_dataset, Dataset, concatenate_datasets
except ImportError:
    # Basic fallback if datasets is not available
    class Dataset: pass
    def load_dataset(*args, **kwargs): return None
    def concatenate_datasets(*args, **kwargs): return None

logger = logging.getLogger(__name__)

class RTX3060DatasetPipeline:
    def __init__(self):
        try:
            from src.utils.path_resolver import PathResolver
            self.project_root = PathResolver.get_project_root()
        except ImportError:
            self.project_root = Path.cwd()
            
        self.memory_limits = {
            'max_dataset_size': 50_000_000,  # 50M samples max
            'chunk_size': 10_000,            # Process in chunks
            'cache_efficient': True,         # Minimize memory usage
            'streaming': True                # Don't load everything at once
        }
        self.moonshot_base_dir = self.project_root / "data" / "moonshot"

    def _load_mcp_skills_hf_datasets(self) -> Optional[Any]:
        """Load MCP skills integration dataset from HuggingFace"""
        try:
            logger.info("[RTX3060] Loading MCP skills dataset from HF")
            # Typical repos for MCP/API/Skill training
            dataset = load_dataset("mcp-archive/mcp-skills-v1", split="train")
            return dataset
        except Exception as e:
            logger.warning(f"[RTX3060] Failed to load MCP skills from HF: {e}")
            return None

    def _load_api_skill_calling_hf_datasets(self) -> Optional[Any]:
        """Load API skill calling dataset from HuggingFace"""
        try:
            logger.info("[RTX3060] Loading API skill calling dataset from HF")
            dataset = load_dataset("mcp-archive/api-calling-v1", split="train")
            return dataset
        except Exception as e:
            logger.warning(f"[RTX3060] Failed to load API skill calling from HF: {e}")
            return None

    def _download_moonshot_dataset(self, dataset_name: str) -> Optional[Any]:
        """Download or load a Moonshot specific dataset"""
        try:
            logger.info(f"[RTX3060] Resolving Moonshot dataset: {dataset_name}")
            
            # 1. Check local directory
            local_path = self.moonshot_base_dir / f"{dataset_name}.jsonl"
            if local_path.exists():
                logger.info(f"[RTX3060] Loading local Moonshot dataset: {local_path}")
                return load_dataset('json', data_files=str(local_path), split='train')
            
            # 2. Check alternative local extensions
            for ext in ['.json', '.csv', '.parquet']:
                local_path = self.moonshot_base_dir / f"{dataset_name}{ext}"
                if local_path.exists():
                    logger.info(f"[RTX3060] Loading local Moonshot dataset: {local_path}")
                    fmt = ext.replace('.', '')
                    if fmt == 'jsonl': fmt = 'json'
                    return load_dataset(fmt, data_files=str(local_path), split='train')

            # 3. Fallback to HF (simulated/plausible naming)
            logger.info(f"[RTX3060] Local dataset {dataset_name} not found. Searching HF...")
            hf_repo = f"moonshot-data/{dataset_name}"
            return load_dataset(hf_repo, split='train')
            
        except Exception as e:
            logger.warning(f"[RTX3060] Failed to download/load Moonshot dataset {dataset_name}: {e}")
            return None

    def get_capabilities_dataset(self) -> Optional[Any]:
        """Get a combined dataset for agent capabilities (MCP/API/Skill)"""
        datasets = []
        
        mcp = self._load_mcp_skills_hf_datasets()
        if mcp: datasets.append(mcp)
        
        api = self._load_api_skill_calling_hf_datasets()
        if api: datasets.append(api)
        
        if not datasets:
            return None
            
        return concatenate_datasets(datasets)
