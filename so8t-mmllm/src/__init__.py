"""
SO8T×マルチモーダルLLM（ローカル）
SO(8)群回転ゲート + PET正則化 + OCR要約 + SQLite監査

RTX3060 12GB環境用の安全エージェント実装
"""

__version__ = "1.0.0"
__author__ = "SO8T Team"
__description__ = "SO(8)群Transformer with Multimodal Safety Agent"

# コアモジュールのインポート
from .modules.rotation_gate import SO8TRotationGate, apply_block_rotation
from .losses.pet import PETLoss, pet_penalty
from .training.qlora import QLoRATrainer
from .io.ocr_summary import OCRSummaryProcessor
from .audit.sqlite_logger import SQLiteAuditLogger

__all__ = [
    "SO8TRotationGate",
    "apply_block_rotation", 
    "PETLoss",
    "pet_penalty",
    "QLoRATrainer",
    "OCRSummaryProcessor",
    "SQLiteAuditLogger",
]
