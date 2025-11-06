"""
SO8T×マルチモーダルLLM（ローカル）
SO(8)群回転ゲート + PET正則化 + OCR要約 + SQLite監査

RTX3060 12GB環境用の安全エージェント実装
"""

__version__ = "1.0.0"
__author__ = "SO8T Team"
__description__ = "SO(8)群Transformer with Multimodal Safety Agent"

# コアモジュールのインポート（エラー回避版）
try:
    from .modules.rotation_gate import SO8TRotationGate, apply_block_rotation
except ImportError:
    SO8TRotationGate = None
    apply_block_rotation = None

try:
    from .losses.pet import PETLoss, pet_penalty
except ImportError:
    PETLoss = None
    pet_penalty = None

try:
    from .training.qlora import QLoRATrainer
except ImportError:
    print("[WARNING] QLoRATrainer import failed, using standard training")
    QLoRATrainer = None

try:
    from .io.ocr_summary import OCRSummaryProcessor
except ImportError:
    OCRSummaryProcessor = None

try:
    from .audit.sqlite_logger import SQLiteAuditLogger
except ImportError:
    SQLiteAuditLogger = None

__all__ = [
    "SO8TRotationGate",
    "apply_block_rotation", 
    "PETLoss",
    "pet_penalty",
    "QLoRATrainer",
    "OCRSummaryProcessor",
    "SQLiteAuditLogger",
]
