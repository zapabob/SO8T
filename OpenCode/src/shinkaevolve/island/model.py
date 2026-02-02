"""Island-level model structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..core.evolution import Individual


@dataclass
class IslandPopulation:
    """Container for island population."""

    island_id: str
    individuals: List[Individual] = field(default_factory=list)

    def add(self, individual: Individual) -> None:
        individual.island_id = self.island_id
        self.individuals.append(individual)
