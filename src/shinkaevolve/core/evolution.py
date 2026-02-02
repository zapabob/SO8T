"""
Evolutionary program search module for ShinkaEvolve.
References:
    Lange et al. (2025) arXiv:2509.19349
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import random

from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Individual:
    """Evolutionary individual."""

    code: str
    fitness: float = 0.0
    metadata: dict = field(default_factory=dict)
    generation: int = 0
    island_id: str = "default"


@dataclass
class PopulationConfig:
    """Population configuration."""

    population_size: int = 50
    elite_ratio: float = 0.2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    max_generations: int = 100


class EvolutionEngine:
    """Evolutionary algorithm engine."""

    def __init__(self, config: Optional[PopulationConfig] = None) -> None:
        self.config = config or PopulationConfig()
        self.population: list[Individual] = []
        self.generation = 0

    def initialize(self, initial_code: str, *, num_individuals: Optional[int] = None) -> None:
        """Initialize population with clones of initial code."""
        num = num_individuals or self.config.population_size
        for i in range(num):
            individual = Individual(
                code=initial_code,
                fitness=0.0,
                generation=0,
                island_id=f"island_{i % 4}",
            )
            self.population.append(individual)
        logger.info("Initialized population: %s", num)

    def evolve(self) -> None:
        """Run one generation."""
        self.generation += 1
        self._evaluate()
        self._select()
        self._reproduce()
        logger.info("Generation %s complete", self.generation)

    def _evaluate(self) -> None:
        for individual in self.population:
            if individual.fitness == 0.0:
                individual.fitness = self._fitness_function(individual.code)

    def _fitness_function(self, code: str) -> float:
        return 0.0

    def _select(self) -> None:
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        self.elites = sorted_pop[:elite_count]

    def _reproduce(self) -> None:
        new_population = self.elites.copy()
        while len(new_population) < self.config.population_size:
            parent = random.choice(self.elites)
            if random.random() < self.config.mutation_rate:
                child_code = self._mutate(parent.code)
            else:
                child_code = self._crossover(parent.code)
            child = Individual(
                code=child_code,
                fitness=0.0,
                generation=self.generation,
                island_id=parent.island_id,
            )
            new_population.append(child)
        self.population = new_population[: self.config.population_size]

    def _mutate(self, code: str) -> str:
        return code

    def _crossover(self, code: str) -> str:
        return code

    @property
    def best_individual(self) -> Optional[Individual]:
        if not self.population:
            return None
        return max(self.population, key=lambda x: x.fitness)


class IslandModel:
    """Island model for migration across sub-populations."""

    MIGRATION_INTERVAL = 10

    def __init__(
        self,
        num_islands: int = 4,
        migration_interval: int = MIGRATION_INTERVAL,
        migration_rate: float = 0.1,
    ) -> None:
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.islands: dict[str, list[Individual]] = {}
        self.generation = 0
        for i in range(num_islands):
            self.islands[f"island_{i}"] = []

    def add_individual(self, individual: Individual, island_id: str) -> None:
        if island_id not in self.islands:
            raise ValueError(f"Invalid island id: {island_id}")
        individual.island_id = island_id
        self.islands[island_id].append(individual)

    def migrate(self) -> None:
        for island_id in self.islands:
            num_migrate = int(len(self.islands[island_id]) * self.migration_rate)
            if num_migrate == 0:
                continue
            migrants = random.sample(
                self.islands[island_id],
                min(num_migrate, len(self.islands[island_id])),
            )
            for migrant in migrants:
                target_island = random.choice([i for i in self.islands if i != island_id])
                self.islands[target_island].append(migrant)
                self.islands[island_id].remove(migrant)
        logger.info("Migration complete")

    def evolve_all(self) -> None:
        for island_id in self.islands:
            engine = EvolutionEngine()
            engine.population = self.islands[island_id]
            engine.generation = self.generation
            engine.evolve()
            self.islands[island_id] = engine.population
        self.generation += 1
        if self.generation % self.migration_interval == 0:
            self.migrate()

    @property
    def best_individual(self) -> Optional[Individual]:
        all_individuals = sum(self.islands.values(), [])
        if not all_individuals:
            return None
        return max(all_individuals, key=lambda x: x.fitness)
