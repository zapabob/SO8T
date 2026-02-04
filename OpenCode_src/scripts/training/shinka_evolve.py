#!/usr/bin/env python3
"""
ShinkaEvolve - lightweight evolutionary search for reasoning responses.

Implements:
- island populations
- mutation for code/reasoning variation
- simple fitness scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random


@dataclass
class ShinkaEvolveConfig:
    population_size: int = 8
    island_count: int = 2
    generations: int = 2
    mutation_rate: float = 0.25
    migration_rate: float = 0.2


class ShinkaEvolveEngine:
    def __init__(self, config: ShinkaEvolveConfig) -> None:
        self.config = config

    def _mutate(self, response: str) -> str:
        """Simple mutation: append or tweak reasoning token."""
        tokens = [
            "Therefore,",
            "In summary,",
            "Consider this:",
            "Step-by-step:",
            "To verify:",
        ]
        if random.random() < 0.5:
            return f"{random.choice(tokens)} {response}"
        return response + " [mut]"

    def _score(self, response: str) -> float:
        """Fitness: favor concise reasoning markers."""
        score = min(len(response) / 200, 1.0)
        bonus = 0.2 if any(t in response.lower() for t in ["therefore", "step"]) else 0
        return score + bonus

    def _evolve_island(self, population: List[str]) -> List[str]:
        """Evolve a single island."""
        for _ in range(self.config.generations):
            scored = sorted(population, key=self._score, reverse=True)
            survivors = scored[: max(2, len(scored) // 2)]
            population = survivors[:]
            while len(population) < self.config.population_size:
                parent = random.choice(survivors)
                child = self._mutate(parent) if random.random() < self.config.mutation_rate else parent
                population.append(child)
        return population

    def evolve(self, prompts: List[str], responses: List[str]) -> List[str]:
        """Evolve responses for each prompt (independent islands)."""
        evolved = []
        for response in responses:
            # Seed population
            population = [response] + [self._mutate(response) for _ in range(self.config.population_size - 1)]
            # Split into islands
            islands = [
                population[i :: self.config.island_count]
                for i in range(self.config.island_count)
            ]
            islands = [self._evolve_island(island) for island in islands]

            # Migration: move top candidates across islands
            if self.config.island_count > 1:
                migrants = [max(island, key=self._score) for island in islands]
                for i, island in enumerate(islands):
                    if random.random() < self.config.migration_rate:
                        island.append(migrants[(i + 1) % len(migrants)])

            # Select best overall
            merged = [cand for island in islands for cand in island]
            best = max(merged, key=self._score)
            evolved.append(best)
        return evolved
