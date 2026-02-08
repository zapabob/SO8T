# -*- coding: utf-8 -*-
"""
Functional tests for Evolved Shinka Pipeline components
"""

import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print("[TEST] Functional tests for Evolved Shinka Pipeline")
print("=" * 50)

print("\n[TEST 1] World Events 2024-2026")
from data.world_events_2024_2026 import WorldEvents2024_2026

we = WorldEvents2024_2026()
stats = we.get_statistics()
print(f"  Total events: {stats['total_events']}")
print(f"  Scientific events: {stats['scientific_events']}")
print(f"  Categories: {list(stats['by_category'].keys())[:5]}...")

if stats["total_events"] > 0:
    print("[OK] WorldEvents2024_2026 works")
else:
    print("[NG] No events loaded")

print("\n[TEST 2] Ebbinghaus Freeze")
from optimization.ebbinghaus_freeze import EbbinghausFreeze, FreezeConfig

config = FreezeConfig(protection_domains=["science", "math"])
ef = EbbinghausFreeze(config=config)

mem_id = ef.add_memory("Test knowledge", "science", importance_score=0.8)
print(f"  Added memory: {mem_id}")
retention = ef.memory_nodes[mem_id].get_retention()
print(f"  Retention: {retention:.3f}")

stats = ef.get_statistics()
print(f"  Total memories: {stats['total_memories']}")

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False, encoding="utf-8"
) as f:
    temp_path = f.name

ef.save_state(temp_path)
ef2 = EbbinghausFreeze(config=config)
ef2.load_state(temp_path)
os.unlink(temp_path)

if ef2.get_statistics()["total_memories"] > 0:
    print("[OK] Ebbinghaus Freeze save/load works")
else:
    print("[NG] Save/load failed")

print("\n[TEST 3] Statistical Cleansing 95%")
from evaluation.llm_judge_95 import StatisticalCleansing95, CleansingConfig

samples = [
    {"content": f"Sample {i}", "fitness": 0.5 + (i * 0.1) if i < 5 else 0.1}
    for i in range(10)
]

cleansing = StatisticalCleansing95(CleansingConfig())
cleansed, stats = cleansing.cleanse(samples, "fitness")

print(
    f"  Original: {stats['original_count']}, Kept: {stats['kept_count']}, Removed: {stats['removed_count']}"
)
print(f"  Removal rate: {stats.get('removal_rate', 0):.2%}")

if stats["kept_count"] > 0 and stats["removed_count"] > 0:
    print("[OK] Statistical cleansing works")
else:
    print("[NG] Cleansing did not remove outliers")

print("\n[TEST 4] ShinkaNEAT Engine Classes")
from evolution.shinka_neat_engine import Individual, ReasoningNode, EvolutionConfig

ind = Individual(
    genome=[ReasoningNode(id=0, content="Test", node_type="observation")], domain="test"
)
print(f"  Individual ID: {ind.id}")
print(f"  Genome length: {len(ind.genome)}")
print(f"  Fitness: {ind.fitness}")

config = EvolutionConfig(population_size=8, generations=3)
print(f"  Config population: {config.population_size}")
print(f"  Config generations: {config.generations}")

print("[OK] ShinkaNEAT classes work")

print("\n[TEST 5] Quadruple VSSI Generator Classes")
from data.evolutionary.quadruple_vssi_generator import (
    QuadrupleReasoning,
    VSSIDataSample,
)

reasoning = QuadrupleReasoning(
    think_task="Test task",
    think_analysis="Test analysis",
    think_safety="Test safety",
    think_policy="Test policy",
)

sample = VSSIDataSample(
    id="test_001",
    topic="Test Topic",
    domain="test",
    instruction="Test instruction",
    quadruple_reasoning=reasoning,
    final_output="Test output",
)

dict_output = sample.to_dict()
print(f"  Sample ID: {dict_output['id']}")
print(f"  Has quadruple_reasoning: {'quadruple_reasoning' in dict_output}")

if "quadruple_reasoning" in dict_output:
    print("[OK] Quadruple VSSI classes work")
else:
    print("[NG] Quadruple VSSI classes failed")

print("\n[TEST 6] Pipeline Config")
from infrastructure.pipeline.evolved_shinka_pipeline import (
    PipelineConfig,
    PipelineState,
)

config = PipelineConfig(skip_evolution=False, skip_quadruple=True)
print(f"  Skip evolution: {config.skip_evolution}")
print(f"  Skip quadruple: {config.skip_quadruple}")

state = PipelineState()
print(f"  Initial phase: {state.current_phase}")
print(f"  Completed: {state.is_completed}")

print("[OK] Pipeline config works")

print("\n" + "=" * 50)
print("[TEST] All functional tests completed!")
print("=" * 50)
