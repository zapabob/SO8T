import random
import logging
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Genome:
    code: str
    fitness: float
    novelty: float
    metadata: Dict[str, Any]

class EvolutionaryOptimizer:
    """
    Implements program evolution logic inspired by ShinkaEvolve.
    Key features: Mutation operators (LLM), Parent sampling, Novelty filtering.
    """
    def __init__(self, population_size: int = 20):
        self.population: List[Genome] = []
        self.population_size = population_size
        self.generation = 0

    def initialize_population(self, seed_code: str):
        """Initializes the population with mutations of the seed code."""
        logger.info(f"[EVO] Initializing population with seed code.")
        for i in range(self.population_size):
            # In a real system, this calls an LLM to generate diverse starting points
            self.population.append(Genome(
                code=f"{seed_code}\n# Init variation {i}",
                fitness=0.0,
                novelty=1.0,
                metadata={"origin": "seed"}
            ))

    def evolve(self, generations: int = 5):
        """Execution loop for evolution."""
        for g in range(generations):
            self.generation += 1
            logger.info(f"[EVO] Generation {self.generation}")
            
            # 1. Evaluation (Fitness & Novelty)
            self._evaluate_population()
            
            # 2. Parent Selection (Fitness- and Novelty-aware)
            parents = self._sample_parents()
            
            # 3. Mutation (LLM-driven)
            offspring = self._generate_offspring(parents)
            
            # 4. Replacement (Survival of the fittest/most novel)
            self.population = self._select_survivors(self.population + offspring)
            
            best = max(self.population, key=lambda x: x.fitness)
            logger.info(f"[EVO] Best fitness: {best.fitness}")

    def _evaluate_population(self):
        """Simulates evaluation of models/programs."""
        for genome in self.population:
            if genome.fitness == 0.0:
                # Simulated fitness score (higher is better)
                genome.fitness = random.uniform(0.1, 0.95)
                # Simulated novelty score
                genome.novelty = random.uniform(0.1, 1.0)

    def _sample_parents(self) -> List[Genome]:
        """Adaptive parent sampling."""
        # Weighted sample based on fitness + novelty
        weights = [g.fitness + 0.2 * g.novelty for g in self.population]
        return random.choices(self.population, weights=weights, k=self.population_size // 2)

    def _generate_offspring(self, parents: List[Genome]) -> List[Genome]:
        """Produces offspring via mutations."""
        offspring = []
        for p in parents:
            # LLM mutation simulation
            new_code = f"{p.code}\n# Mutation in Gen {self.generation}"
            offspring.append(Genome(
                code=new_code,
                fitness=0.0,
                novelty=1.0,
                metadata={"parent_fitness": p.fitness}
            ))
        return offspring

    def _select_survivors(self, candidates: List[Genome]) -> List[Genome]:
        """Survival selection."""
        # Sort by fitness descending, keep top population_size
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        return candidates[:self.population_size]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evo = EvolutionaryOptimizer(population_size=10)
    evo.initialize_population("def solve(): pass")
    evo.evolve(generations=3)
