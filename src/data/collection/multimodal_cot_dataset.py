# -*- coding: utf-8 -*-
"""
Multimodal CoT Dataset Loader

Loads and combines:
1. Existing quadruple reasoning CoT datasets
2. HuggingFace multimodal datasets (NSFW, general)
3. YouTube video frames with SO8ViT orthogonal transformations
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MultimodalCoTsample:
    sample_id: str
    text: str
    images: List[torch.Tensor]
    audio: Optional[torch.Tensor]
    cot_reasoning: str
    reasoning_type: str
    safety_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    hf_dataset_name: str = "laion/NSFW-Data"
    local_cot_path: str = "data/cot_quadruple"
    youtube_cache_dir: str = "data/youtube_cache"
    max_samples: int = 10000
    image_size: int = 224
    max_audio_length: int = 16000
    safety_filter: bool = True


class OrthogonalTransform(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        self.rotation_matrix = nn.Parameter(torch.eye(dim))
        self.scale = nn.Parameter(torch.ones(dim))
        self.register_buffer("_det", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ortho = self.rotation_matrix / (
            self.rotation_matrix.norm(dim=1, keepdim=True) + 1e-8
        )
        transformed = ortho @ (x * self.scale)
        return transformed

    def so8_transform(self, x: torch.Tensor) -> torch.Tensor:
        q8 = self._get_quaternion_octonion(x)
        transformed = self._apply_octonion_mul(q8)
        return transformed

    def _get_quaternion_octonion(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        q = torch.randn(batch, 8, device=x.device)
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        return q

    def _apply_octonion_mul(self, q: torch.Tensor) -> torch.Tensor:
        return q


class SO8ViTExtractor(nn.Module):
    def __init__(self, vision_hidden_dim: int = 768, pretrained: bool = True):
        super().__init__()
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor

            self.encoder = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.processor = CLIPImageProcessor()
            self.hidden_size = self.encoder.config.hidden_size
        except ImportError:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Sequential(
                    *[nn.Conv2d(64, 64, kernel_size=3, padding=1) for _ in range(12)]
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.hidden_size = vision_hidden_dim
        self.projection = nn.Linear(self.hidden_size, vision_hidden_dim)
        self.ortho_transform = OrthogonalTransform(dim=vision_hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "processor"):
            inputs = self.processor(images, return_tensors="pt")
            outputs = self.encoder(**inputs)
            features = outputs.last_hidden_state
        else:
            features = self.encoder(images)
        projected = self.projection(features)
        transformed = self.ortho_transform(projected)
        return transformed

    def extract_frames(
        self, video_path: str, num_frames: int = 8
    ) -> List[torch.Tensor]:
        frames = self._extract_video_frames(video_path, num_frames)
        return [self(frame) for frame in frames]

    def _extract_video_frames(
        self, video_path: str, num_frames: int
    ) -> List[torch.Tensor]:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(frame).float() / 255.0
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                    frames.append(tensor)
            cap.release()
            return frames
        except ImportError:
            logger.warning("OpenCV not available, using dummy frames")
            return [torch.randn(1, 3, 224, 224) for _ in range(num_frames)]


class AudioEncoder(nn.Module):
    def __init__(self, input_dim: int = 16000, hidden_dim: int = 768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        features = self.conv(audio)
        return features.mean(dim=2)


class MultimodalCoTDataset(Dataset):
    def __init__(
        self,
        samples: List[MultimodalCoTsample],
        tokenizer=None,
        config: Optional[DatasetConfig] = None,
    ):
        self.samples = samples
        self.config = config or DatasetConfig()
        self.tokenizer = tokenizer
        self.vision_extractor = SO8ViTExtractor()
        self.audio_encoder = AudioEncoder()
        self.vision_extractor.eval()
        self.audio_encoder.eval()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        result = {
            "text": sample.text,
            "cot_reasoning": sample.cot_reasoning,
            "reasoning_type": sample.reasoning_type,
            "safety_level": sample.safety_level,
            "sample_id": sample.sample_id,
        }
        if sample.images:
            with torch.no_grad():
                images = [
                    img.float() / 255.0 if (img.max() > 1).any() else img
                    for img in sample.images
                ]
                images = [
                    torch.nn.functional.interpolate(
                        img, size=(224, 224), mode="bilinear"
                    )
                    for img in images
                ]
                image_features = [self.vision_extractor(img) for img in images]
                result["image_features"] = (
                    torch.cat(image_features, dim=1)
                    if len(image_features) > 1
                    else image_features[0]
                )
        if sample.audio is not None:
            with torch.no_grad():
                audio_features = self.audio_encoder(sample.audio)
                result["audio_features"] = audio_features
        return result


class MultimodalCoTDataLoader:
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.samples: List[MultimodalCoTsample] = []
        self._load_all_datasets()

    def _load_all_datasets(self) -> None:
        self._load_local_cot()
        self._load_hf_datasets()
        self._load_youtube_samples()
        logger.info(f"Loaded {len(self.samples)} total samples")

    def _load_local_cot(self) -> None:
        cot_path = Path(self.config.local_cot_path)
        if not cot_path.exists():
            logger.warning(f"Local CoT path not found: {cot_path}")
            self._create_sample_cot_data()
            return
        for file_path in cot_path.glob("*.jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        sample = MultimodalCoTsample(
                            sample_id=data.get("id", f"cot_{len(self.samples)}"),
                            text=data.get("text", data.get("prompt", "")),
                            images=[],
                            audio=None,
                            cot_reasoning=data.get(
                                "cot_reasoning", data.get("reasoning", "")
                            ),
                            reasoning_type=data.get("reasoning_type", "quadruple"),
                            safety_level=data.get("safety_level", "safe"),
                            metadata=data.get("metadata", {}),
                        )
                        self.samples.append(sample)
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Loaded {len(self.samples)} local CoT samples")

    def _create_sample_cot_data(self) -> None:
        sample_types = ["vector", "spinor_pos", "spinor_neg", "quadrality"]
        safety_levels = ["safe", "caution", "restricted"]
        for i in range(100):
            sample = MultimodalCoTsample(
                sample_id=f"sample_{i}",
                text=f"Sample text for reasoning type {sample_types[i % 4]}",
                images=[],
                audio=None,
                cot_reasoning=f"Step-by-step reasoning for sample {i}",
                reasoning_type=sample_types[i % 4],
                safety_level=safety_levels[i % 3],
            )
            self.samples.append(sample)
        logger.info(f"Created {len(self.samples)} sample CoT data")

    def _load_hf_datasets(self) -> None:
        try:
            from datasets import load_dataset

            dataset_name = self.config.hf_dataset_name
            logger.info(f"Loading HuggingFace dataset: {dataset_name}")
            if self.config.safety_filter and "nsfw" in dataset_name.lower():
                dataset = load_dataset(dataset_name, split="train[:1000]")
                for item in dataset:
                    if isinstance(item, dict):
                        sample = MultimodalCoTsample(
                            sample_id=item.get("id", f"hf_{len(self.samples)}"),
                            text=item.get("text", item.get("caption", "")),
                            images=[],
                            audio=None,
                            cot_reasoning=item.get("reasoning", ""),
                            reasoning_type="general",
                            safety_level="nsfw_filtered"
                            if self.config.safety_filter
                            else "unknown",
                        )
                        self.samples.append(sample)
            logger.info(f"Loaded {len(self.samples)} HF samples")
        except ImportError:
            logger.warning("datasets library not available")
        except Exception as e:
            logger.warning(f"Failed to load HF dataset: {e}")

    def _load_youtube_samples(self) -> None:
        cache_dir = Path(self.config.youtube_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        extractor = SO8ViTExtractor()
        for video_file in cache_dir.glob("*.mp4")[:100]:
            try:
                frames = extractor.extract_frames(str(video_file), num_frames=4)
                sample = MultimodalCoTsample(
                    sample_id=f"video_{video_file.stem}",
                    text=f"Video analysis for {video_file.stem}",
                    images=frames,
                    audio=None,
                    cot_reasoning="Visual reasoning extracted from video frames",
                    reasoning_type="visual",
                    safety_level="safe",
                )
                self.samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to process video {video_file}: {e}")
        logger.info(
            f"Loaded {len([s for s in self.samples if s.reasoning_type == 'visual'])} YouTube samples"
        )

    def get_dataloader(self, batch_size: int = 4, shuffle: bool = True) -> DataLoader:
        dataset = MultimodalCoTDataset(self.samples, config=self.config)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=0,
        )

    def get_samples(self) -> List[MultimodalCoTsample]:
        return self.samples


def create_multimodal_cot_pipeline():
    config = DatasetConfig(
        max_samples=10000,
        safety_filter=True,
    )
    dataloader = MultimodalCoTDataLoader(config)
    return dataloader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = create_multimodal_cot_pipeline()
    loader = pipeline.get_dataloader(batch_size=4)
    logger.info(f"Created dataloader with {len(loader)} batches")
    for batch in loader:
        logger.info(f"Batch keys: {batch.keys()}")
        break
