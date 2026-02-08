import sys
from pathlib import Path
import logging

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "scripts" / "autonomous_research"))

from autonomous_researcher import AutonomousResearcher
from evolutionary_optimizer import EvolutionaryOptimizer

def test_research_foundation():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestResearch")
    logger.info("Starting Autonomous Research Foundation Test")

    # 1. Test Research Loop
    researcher = AutonomousResearcher(project_root)
    log = researcher.run_research_cycle("Differential Privacy in Distributed LLM training", max_iterations=2)
    assert len(log["iterations"]) > 0
    logger.info("[OK] Research cycle completed and logged.")

    # 2. Test Evolutionary Optimizer
    evo = EvolutionaryOptimizer(population_size=5)
    evo.initialize_population("def train_step(): ...")
    evo.evolve(generations=2)
    assert len(evo.population) == 5
    logger.info("[OK] Evolutionary optimization loop verified.")

    logger.info("ALL RESEARCH FOUNDATION TESTS PASSED.")

if __name__ == "__main__":
    test_research_foundation()
