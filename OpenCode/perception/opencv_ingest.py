"""
OpenCV based ingestion utilities that summarise frames into structured observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import cv2
import numpy as np


@dataclass
class FrameSummary:
    frame_id: int
    mean_rgb: tuple[float, float, float]
    motion_score: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "frame_id": self.frame_id,
                "mean_rgb": self.mean_rgb,
                "motion_score": self.motion_score,
            }
        )


def summarize_video(path: Path, stride: int = 15) -> Iterator[FrameSummary]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {path}")

    prev_frame: Optional[np.ndarray] = None
    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % stride != 0:
                frame_id += 1
                continue

            frame_float = frame.astype(np.float32) / 255.0
            mean_rgb = frame_float.mean(axis=(0, 1))

            motion_score = 0.0
            if prev_frame is not None:
                diff = cv2.absdiff(frame_float, prev_frame)
                motion_score = float(diff.mean())
            prev_frame = frame_float

            yield FrameSummary(frame_id=frame_id, mean_rgb=tuple(mean_rgb.tolist()), motion_score=motion_score)
            frame_id += 1
    finally:
        cap.release()
