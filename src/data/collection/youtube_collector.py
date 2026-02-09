# -*- coding: utf-8 -*-
"""
YouTube Video Collector with SO8ViT Orthogonal Transformations

Downloads videos and extracts frames using:
- SO(n) orthogonal transformations (det=1) for rotation/scale invariance
- Robust frame extraction resistant to camera motion
"""

from __future__ import annotations

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
import threading
import queue
import time

import torch
import torch.nn as nn
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    download_dir: str = "data/youtube_videos"
    cache_dir: str = "data/youtube_cache"
    frame_dir: str = "data/youtube_frames"
    max_videos: int = 100
    num_frames_per_video: int = 8
    video_quality: str = "best"
    audio_quality: str = "best"
    yt_dlp_path: str = "yt-dlp"
    image_size: int = 224
    use_so8_transform: bool = True


class SO8OrthogonalTransform(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        self.rotation_matrix = nn.Parameter(torch.eye(dim))
        self.scale = nn.Parameter(torch.ones(dim))
        self._ensure_special_orthogonal()

    def _ensure_special_orthogonal(self) -> None:
        with torch.no_grad():
            U, S, V = torch.svd(self.rotation_matrix)
            det = torch.det(U @ V.T)
            V[:, -1] *= det
            self.rotation_matrix.data = U @ V.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.rotation_matrix / (self.rotation_matrix.norm(dim=1, keepdim=True) + 1e-8)
        scaled = x * self.scale
        return Q @ scaled

    def transform_2d(self, image: torch.Tensor) -> torch.Tensor:
        matrix = self.rotation_matrix[:2, :2].detach()
        h, w = image.shape[1:]
        coords = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=matrix.dtype),
            torch.arange(w, dtype=matrix.dtype),
            indexing='ij'
        )).reshape(2, -1)
        transformed_coords = matrix @ coords
        grid = transformed_coords.reshape(2, h, w).flip(0)
        grid = grid / (h - 1) * 2 - 1
        return nn.functional.grid_sample(image.unsqueeze(0), grid.unsqueeze(0), align_corners=True).squeeze(0)

    def get_scale_rotation_invariant(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        normalized = x / (norm + 1e-8)
        return self.forward(normalized)


class SO8ViTFrameExtractor(nn.Module):
    def __init__(self, config: Optional[VideoConfig] = None):
        super().__init__()
        self.config = config or VideoConfig()
        self.ortho_transform = SO8OrthogonalTransform(dim=768)
        self._setup_feature_extractor()

    def _setup_feature_extractor(self) -> None:
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPImageProcessor()
            self.hidden_size = self.encoder.config.hidden_size
        except ImportError:
            self.encoder = None
            self.hidden_size = 768

    def forward(self, frames: List[torch.Tensor]) -> torch.Tensor:
        if self.encoder is not None:
            inputs = self.processor(images=[f.squeeze().permute(1, 2, 0).numpy() for f in frames], return_tensors="pt")
            outputs = self.encoder(**inputs)
            features = outputs.last_hidden_state
        else:
            features = torch.stack([self._simple_extract(f) for f in frames])
        ortho_features = self.ortho_transform(features)
        return ortho_features

    def _simple_extract(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.mean(dim=[1, 2])

    def extract_robust_frames(
        self,
        video_path: str,
        num_frames: int = 8,
        motion_threshold: float = 0.1,
    ) -> List[torch.Tensor]:
        frames = self._extract_keyframes(video_path, num_frames, motion_threshold)
        features = self.forward(frames)
        return features

    def _extract_keyframes(
        self,
        video_path: str,
        num_frames: int,
        motion_threshold: float,
    ) -> List[torch.Tensor]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 10
        frame_indices = np.linspace(0, total_frames - 1, min(num_frames * 2, total_frames), dtype=int)
        frames = []
        prev_frame = None
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb).float() / 255.0
            tensor = tensor.permute(2, 0, 1)
            if prev_frame is not None:
                motion = (tensor - prev_frame).abs().mean().item()
                if motion < motion_threshold and len(frames) >= num_frames:
                    continue
            prev_frame = tensor
            frames.append(tensor)
            if len(frames) >= num_frames:
                break
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1].clone() if frames else torch.zeros(3, 224, 224))
        return frames[:num_frames]


class YouTubeCollector:
    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.frame_extractor = SO8ViTFrameExtractor(self.config)
        self.download_dir = Path(self.config.download_dir)
        self.cache_dir = Path(self.config.cache_dir)
        self.frame_dir = Path(self.config.frame_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_queue: queue.Queue = queue.Queue()
        self._download_process: Optional[subprocess.Popen] = None

    def search_and_download(
        self,
        queries: List[str],
        max_videos: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        max_videos = max_videos or self.config.max_videos
        downloaded = []
        for query in queries:
            cmd = [
                self.config.yt_dlp_path,
                f"ytsearch{(max_videos // len(queries))}:{query}",
                "-o", str(self.download_dir / "%(id)s.%(ext)s),
                "--download-annotations",
                "--write-info-json",
                "--no-playlist",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    for video_file in self.download_dir.glob("*.mp4"):
                        downloaded.append({
                            "path": str(video_file),
                            "query": query,
                            "title": video_file.stem,
                        })
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout downloading for query: {query}")
            except FileNotFoundError:
                logger.warning("yt-dlp not found, skipping download")
                break
        return downloaded

    def extract_all_frames(
        self,
        videos: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        extracted = []
        for video in videos:
            video_path = video["path"]
            frame_output_dir = self.frame_dir / Path(video_path).stem
            frame_output_dir.mkdir(exist_ok=True)
            try:
                features = self.frame_extractor.extract_robust_frames(
                    video_path,
                    num_frames=self.config.num_frames_per_video,
                )
                feature_path = frame_output_dir / "features.pt"
                torch.save(features, feature_path)
                metadata = {
                    "video_path": video_path,
                    "feature_path": str(feature_path),
                    "num_frames": len(features),
                    "query": video.get("query", ""),
                }
                extracted.append(metadata)
                logger.info(f"Extracted {len(features)} frames from {Path(video_path).stem}")
            except Exception as e:
                logger.error(f"Failed to extract frames from {video_path}: {e}")
        return extracted

    def create_video_dataset(
        self,
        videos: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        dataset = []
        for video in videos:
            frame_path = self.frame_dir / Path(video["path"]).stem / "features.pt"
            if frame_path.exists():
                features = torch.load(frame_path)
                dataset.append({
                    "video_id": Path(video["path"]).stem,
                    "query": video.get("query", ""),
                    "features": features,
                    "source": "youtube",
                })
        return dataset


class YouTubeDataPipeline:
    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.collector = YouTubeCollector(self.config)
        self.ortho_transform = SO8OrthogonalTransform(dim=768)

    def run(
        self,
        queries: List[str],
        max_videos: int = 100,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Starting YouTube data pipeline for {len(queries)} queries")
        videos = self.collector.search_and_download(queries, max_videos)
        extracted = self.collector.extract_all_frames(videos)
        dataset = self.collector.create_video_dataset(extracted)
        logger.info(f"Created dataset with {len(dataset)} videos")
        return dataset

    def get_orthogonal_invariant_features(
        self,
        video_features: torch.Tensor,
    ) -> torch.Tensor:
        normalized = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-8)
        return self.ortho_transform(normalized)


def main():
    logging.basicConfig(level=logging.INFO)
    config = VideoConfig(
        max_videos=10,
        num_frames_per_video=8,
    )
    pipeline = YouTubeDataPipeline(config)
    queries = [
        "machine learning tutorial",
        "AI research lecture",
        "programming tutorial",
    ]
    dataset = pipeline.run(queries, max_videos=10)
    logger.info(f"Collected {len(dataset)} video samples")


if __name__ == "__main__":
    main()
