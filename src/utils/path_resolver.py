# -*- coding: utf-8 -*-
"""
Path Resolver Utility for Moonshot Pipeline
Ensures reliable path resolution regardless of CWD or execution context.
"""
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PathResolver:
    _project_root = None

    @classmethod
    def get_project_root(cls) -> Path:
        """
        Dynamically determine the project root directory.
        Strategy:
        1. Look for 'src' directory in parents of this file.
        2. Look for markers like '.git' or 'requirements.txt'.
        """
        if cls._project_root:
            return cls._project_root

        # Start from the current file's location
        current_path = Path(__file__).resolve()
        
        # Traverse up looking for "src" or root markers
        for parent in current_path.parents:
            if (parent / "src").exists() and (parent / "src").is_dir():
                cls._project_root = parent
                return parent
            if (parent / ".git").exists():
                cls._project_root = parent
                return parent
        
        # Fallback: assume CWD if nothing else works (warning issued)
        # This might happen in some packaged environments
        cwd = Path.cwd()
        logger.warning(f"[PathResolver] Could not detect project root from structure. Using CWD: {cwd}")
        cls._project_root = cwd
        return cwd

    @classmethod
    def resolve(cls, relative_path: str) -> Path:
        """Resolve a path relative to the project root."""
        return cls.get_project_root() / relative_path
