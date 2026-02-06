# -*- coding: utf-8 -*-
"""
Execution Guards Utility for Moonshot Pipeline
Centralizes platform capability checks and safety guards.
"""
import platform
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

class ExecutionGuards:
    @staticmethod
    def get_safe_num_proc() -> int:
        """
        Determines safe multiprocessing count for dataset processing.
        Windows has issues with dill/multiprocessing serialization in Unsloth.
        """
        if platform.system() == "Windows":
            # Windows: Single process is safest for Unsloth/datasets interactions
            logger.debug("[Guard] Windows detected: dataset_num_proc set to 1")
            return 1
        
        # Linux/Mac: Use available CPUs but cap reasonable limits
        cpu_count = os.cpu_count() or 2
        return min(cpu_count, 4)  # Cap at 4 to prevent OOM
    
    @staticmethod
    def check_unsloth_availability() -> Tuple[bool, str]:
        """
        Check if Unsloth is available and functional.
        Returns: (is_available, error_message)
        """
        try:
            import unsloth
            return True, ""
        except ImportError as e:
            msg = f"Unsloth import failed: {e}. Install with pip install unsloth[colab-new]"
            return False, msg
        except Exception as e:
            # Catch other potential loading errors (e.g., CUDA missing)
            msg = f"Unsloth loading error: {e}"
            return False, msg
