"""
Minimal OCR helper built on pytesseract for text-only auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    text: str
    confidence: float


def extract_text(image_path: Path) -> OCRResult:
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    text = " ".join(word for word in data["text"] if word.strip())
    conf = [float(c) for c in data["conf"] if c != "-1"]
    confidence = sum(conf) / max(len(conf), 1) if conf else 0.0
    return OCRResult(text=text, confidence=confidence)
