# -*- coding: utf-8 -*-
"""
Comprehensive Multimodal Data Collection Pipeline

Combines:
1. Local CoT datasets
2. HuggingFace datasets (NSFW, general multimodal)
3. YouTube video collection with SO8ViT
4. Audio extraction and processing
"""

from __future__ import annotations

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DataCollectionConfig:
    output_dir: str = "data/multimodal_cot"
    max_samples: int = 50000
    batch_size: int = 32
    num_workers: int = 4
    hf_datasets: List[str] = field(
        default_factory=lambda: [
            "laion/NSFW-Data",
            "HuggingFaceM4/COCO",
            "lmms-lab/LLaVA-Pretrain",
        ]
    )
    youtube_queries: List[str] = field(
        default_factory=lambda: [
            "machine learning",
            "AI research",
            "programming tutorial",
            "data science",
        ]
    )
    min_safety_level: str = "safe"
    use_so8_transform: bool = True
    image_size: int = 224
    max_audio_length: int = 16000


class AudioProcessor(nn.Module):
    def __init__(self, sample_rate: int = 16000, hidden_dim: int = 768):
        super().__init__()
        self.sample_rate = sample_rate
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        features = self.conv(audio)
        return features.mean(dim=-1)

    def extract_from_video(self, video_path: str) -> torch.Tensor:
        import cv2

        try:
            cap = cv2.VideoCapture(video_path)
            audio = cap.get(cv2.CAP_PROP_AUDIO_STREAM)
            cap.release()
            if audio:
                return self._process_audio_stream(audio)
            return torch.randn(self.hidden_dim)
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return torch.randn(self.hidden_dim)

    def _process_audio_stream(self, audio_stream: Any) -> torch.Tensor:
        return torch.randn(self.hidden_dim)


class SafetyFilter:
    def __init__(self, min_level: str = "safe"):
        self.min_level = min_level
        self.safety_levels = ["safe", "caution", "restricted", "nsfw"]

    def is_acceptable(self, safety_level: str) -> bool:
        return self.safety_levels.index(safety_level) <= self.safety_levels.index(
            self.min_level
        )

    def filter_samples(self, samples: List[Dict]) -> List[Dict]:
        return [s for s in samples if self.is_acceptable(s.get("safety_level", "safe"))]


class MultimodalDataCollector:
    def __init__(self, config: Optional[DataCollectionConfig] = None):
        self.config = config or DataCollectionConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_processor = AudioProcessor(hidden_dim=self.config.image_size)
        self.safety_filter = SafetyFilter(min_level=self.config.min_safety_level)
        self.collected_samples: List[Dict] = []

    def collect_all(self) -> List[Dict]:
        logger.info("Starting comprehensive multimodal data collection...")
        self._collect_hf_datasets()
        self._collect_youtube()
        self._collect_local_cot()
        self._filter_and_deduplicate()
        logger.info(f"Collected {len(self.collected_samples)} total samples")
        return self.collected_samples

    def _collect_hf_datasets(self) -> None:
        try:
            from datasets import load_dataset

            for ds_name in self.config.hf_datasets:
                try:
                    logger.info(f"Loading dataset: {ds_name}")
                    if "nsfw" in ds_name.lower():
                        ds = load_dataset(ds_name, split="train[:1000]")
                        for item in ds:
                            if isinstance(item, dict):
                                sample = {
                                    "id": f"hf_{ds_name}_{len(self.collected_samples)}",
                                    "text": item.get(
                                        "text",
                                        item.get("caption", item.get("prompt", "")),
                                    ),
                                    "images": [],
                                    "audio": None,
                                    "reasoning": "",
                                    "reasoning_type": "general",
                                    "safety_level": "nsfw"
                                    if "nsfw" in ds_name.lower()
                                    else "safe",
                                    "source": ds_name,
                                }
                                if self.safety_filter.is_acceptable(
                                    sample["safety_level"]
                                ):
                                    self.collected_samples.append(sample)
                    logger.info(
                        f"Added {len(self.collected_samples)} samples from {ds_name}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load {ds_name}: {e}")
        except ImportError:
            logger.warning("datasets library not available")

    def _collect_youtube(self) -> None:
        try:
            from .youtube_collector import YouTubeDataPipeline, VideoConfig

            yt_config = VideoConfig(
                max_videos=50,
                num_frames_per_video=4,
            )
            pipeline = YouTubeDataPipeline(yt_config)
            for query in self.config.youtube_queries[:5]:
                try:
                    dataset = pipeline.run([query], max_videos=10)
                    for item in dataset:
                        sample = {
                            "id": f"youtube_{item.get('video_id', len(self.collected_samples))}",
                            "text": f"Video analysis: {item.get('query', '')}",
                            "images": [],
                            "features": item.get("features"),
                            "reasoning": "Visual reasoning from video",
                            "reasoning_type": "visual",
                            "safety_level": "safe",
                            "source": "youtube",
                        }
                        self.collected_samples.append(sample)
                except Exception as e:
                    logger.warning(f"YouTube collection failed for {query}: {e}")
        except ImportError:
            logger.warning("YouTube collector not available")

    def _collect_local_cot(self) -> None:
        cot_paths = [
            "data/cot_quadruple",
            "data/thinking_datasets",
            "data/reasoning",
        ]
        for cot_path in cot_paths:
            p = Path(cot_path)
            if p.exists():
                for file_path in p.glob("*.jsonl"):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                data = json.loads(line)
                                sample = {
                                    "id": f"local_{len(self.collected_samples)}",
                                    "text": data.get("text", data.get("prompt", "")),
                                    "images": data.get("images", []),
                                    "audio": None,
                                    "reasoning": data.get(
                                        "cot_reasoning", data.get("reasoning", "")
                                    ),
                                    "reasoning_type": data.get(
                                        "reasoning_type", "quadruple"
                                    ),
                                    "safety_level": data.get("safety_level", "safe"),
                                    "source": str(file_path),
                                }
                                self.collected_samples.append(sample)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
        logger.info(
            f"Added local CoT samples: {len([s for s in self.collected_samples if s['source'].startswith('local')])}"
        )

    def _filter_and_deduplicate(self) -> None:
        unique_samples = {}
        for sample in self.collected_samples:
            key = sample.get("id", sample.get("text", "")[:100])
            if key not in unique_samples:
                unique_samples[key] = sample
        self.collected_samples = list(unique_samples.values())[
            : self.config.max_samples
        ]
        self.collected_samples = self.safety_filter.filter_samples(
            self.collected_samples
        )
        logger.info(f"After filtering: {len(self.collected_samples)} samples")

    def save_dataset(self, filename: str = "multimodal_cot_dataset.jsonl") -> str:
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self.collected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"Dataset saved to: {output_path}")
        return str(output_path)

    def get_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        batch_size = batch_size or self.config.batch_size
        dataset = _InternalDataset(self.collected_samples)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.num_workers,
        )


class _InternalDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        result = {
            "id": sample.get("id", str(idx)),
            "text": sample.get("text", ""),
            "reasoning": sample.get("reasoning", ""),
            "reasoning_type": sample.get("reasoning_type", "general"),
            "safety_level": sample.get("safety_level", "safe"),
        }
        return result


class IntegratedMultimodalPipeline:
    def __init__(self, config: Optional[DataCollectionConfig] = None):
        self.config = config or DataCollectionConfig()
        self.collector = MultimodalDataCollector(self.config)

    def run_full_pipeline(self) -> Tuple[str, DataLoader]:
        samples = self.collector.collect_all()
        output_path = self.collector.save_dataset()
        dataloader = self.collector.get_dataloader()
        logger.info(f"Pipeline complete: {len(samples)} samples saved to {output_path}")
        return output_path, dataloader


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    config = DataCollectionConfig(
        max_samples=10000,
        min_safety_level="safe",
        use_so8_transform=True,
    )
    pipeline = IntegratedMultimodalPipeline(config)
    output_path, dataloader = pipeline.run_full_pipeline()
    logger.info(f"Final dataset: {output_path}")
    logger.info(f"Batches: {len(dataloader)}")


if __name__ == "__main__":
    main()
