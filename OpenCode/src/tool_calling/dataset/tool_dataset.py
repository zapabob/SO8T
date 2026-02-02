"""Tool calling dataset definition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from torch.utils.data import Dataset


@dataclass
class ToolDatasetExample:
    """Single tool-calling example."""

    prompt: str
    tool_name: str
    arguments: dict
    expected_output: dict


class ToolDataset(Dataset):
    """Dataset wrapper for tool-calling examples."""

    def __init__(self, examples: List[ToolDatasetExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ToolDatasetExample:
        return self.examples[idx]
