# -*- coding: utf-8 -*-
"""
Test imports for Evolved Shinka Pipeline components
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print("[TEST] Testing imports...")

try:
    from optimization.ebbinghaus_freeze import (
        EbbinghausFreeze,
        FreezeConfig,
        MemoryNode,
    )

    print("[OK] ebbinghaus_freeze.py imported")
except Exception as e:
    print(f"[NG] ebbinghaus_freeze.py: {e}")

try:
    from data.world_events_2024_2026 import WorldEvents2024_2026, WorldEvent

    print("[OK] world_events_2024_2026.py imported")
except Exception as e:
    print(f"[NG] world_events_2024_2026.py: {e}")

try:
    from evaluation.llm_judge_95 import (
        OllamaJudgeClient,
        StatisticalCleansing95,
        CleansingConfig,
    )

    print("[OK] llm_judge_95.py imported")
except Exception as e:
    print(f"[NG] llm_judge_95.py: {e}")

try:
    from evolution.shinka_neat_engine import (
        ShinkaNEATPipeline,
        NEATReasoningEngine,
        ShinkaEvolveEngine,
        OllamaClient,
        Individual,
        ReasoningNode,
        EvolutionConfig,
    )

    print("[OK] shinka_neat_engine.py imported")
except Exception as e:
    print(f"[NG] shinka_neat_engine.py: {e}")

try:
    from data.evolutionary.quadruple_vssi_generator import (
        QuadrupleVSSIGenerator,
        OllamaQuadrupleGenerator,
        VSSIDataSample,
        QuadrupleReasoning,
    )

    print("[OK] quadruple_vssi_generator.py imported")
except Exception as e:
    print(f"[NG] quadruple_vssi_generator.py: {e}")

try:
    from infrastructure.pipeline.evolved_shinka_pipeline import (
        EvolvedShinkaPipeline,
        PipelineConfig,
        PipelineState,
    )

    print("[OK] evolved_shinka_pipeline.py imported")
except Exception as e:
    print(f"[NG] evolved_shinka_pipeline.py: {e}")

print("\n[TEST] All imports completed!")
