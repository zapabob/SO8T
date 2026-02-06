#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moonshot Pipeline Launcher (2025 E026)
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

# --- Environment setup (Windows safe defaults) ---
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LC_ALL", "C")

from src.infrastructure.pipeline.integrated_moonshot_pipeline_2025_2026 import (
    IntegratedMoonshotPipeline2025_2026,
)
from src.utils.startup_manager import StartupManager

from src.utils.safe_logger import SafeLogger

logger = SafeLogger.setup_logger(
    "moonshot_pipeline",
    log_file=Path("moonshot_pipeline_2025_2026.log")
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Moonshot pipeline launcher (2025 E026)")
    parser.add_argument(
        "--use-existing-datasets",
        action="store_true",
        default=True,
        help="Use existing datasets (default: true)",
    )
    parser.add_argument(
        "--collect-new-data",
        action="store_true",
        help="Collect new datasets via HF CLI (overrides --use-existing-datasets)",
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="Moonshot",
        help="Name of the pipeline run (e.g., 'Moonshot', 'Sunset')",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List discovered datasets and exit",
    )
    parser.add_argument(
        "--grape-variant",
        default=os.getenv("SO8T_GRAPE_VARIANT", "multiplicative"),
        help="GRAPE variant: multiplicative/additive/hybrid (default: env or multiplicative)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip heavy model loading/conversion steps (smoke test mode)",
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Use Unsloth training for SFT/GRPO phases",
    )
    parser.add_argument(
        "--mcp-api-skill",
        action="store_true",
        help="Prioritize MCP/API/Skill datasets during Unsloth training",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Recover Unsloth training from the latest checkpoint",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        help="Override training config path for Unsloth",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=os.getenv("SO8T_BASE_MODEL"),
        help="Override base model (e.g., 'unsloth/Qwen2.5-7B-Instruct')",
    )
    parser.add_argument(
        "--subagent-strategy",
        default=os.getenv("SO8T_SUBAGENT_STRATEGY", "single_best"),
        choices=["single_best", "parallel", "sequential"],
        help="Subagent routing strategy",
    )
    parser.add_argument(
        "--subagent-schedule",
        action="store_true",
        help="Generate subagent schedule at pipeline start",
    )
    parser.add_argument(
        "--dynamic-dispatch",
        action="store_true",
        help="Generate dynamic dispatch schedule via tools/dynamic_skill_dispatcher.py",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip Phase 4 multi-domain enrichment",
    )
    parser.add_argument(
        "--skip-reward-strategy",
        action="store_true",
        help="Skip Phase 5 quadrality reward strategy",
    )
    parser.add_argument(
        "--enable-mhc",
        action="store_true",
        help="Enable mHC manifold projection integration",
    )
    parser.add_argument(
        "--enable-so8",
        action="store_true",
        help="Enable SO8 residual adapter injection",
    )
    parser.add_argument(
        "--so8-mode",
        default=os.getenv("SO8T_SO8_MODE", "mlp_only"),
        choices=["mlp_only", "full_layer"],
        help="SO8 adapter injection mode",
    )
    parser.add_argument(
        "--mhc-targets",
        default=os.getenv("SO8T_MHC_TARGETS", "o_proj,down_proj,up_proj,gate_proj"),
        help="Comma-separated module name fragments for mHC projection",
    )
    parser.add_argument(
        "--mhc-blend",
        default=os.getenv("SO8T_MHC_BLEND", "0.1"),
        help="Blend factor for mHC projection",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"{args.pipeline_name} Pipeline 2025–2026")
    logger.info("Start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)

    # Export pipeline name to env
    os.environ["SO8T_PIPELINE_NAME"] = args.pipeline_name

    os.environ["SO8T_GRAPE_VARIANT"] = args.grape_variant
    if args.dry_run:
        os.environ["SO8T_DRYRUN"] = "1"
    if args.use_unsloth:
        os.environ["SO8T_USE_UNSLOTH"] = "1"
    if args.mcp_api_skill:
        os.environ["SO8T_MCP_API_SKILL"] = "1"
    if args.recover:
        os.environ["SO8T_RECOVER"] = "1"
    if args.training_config:
        os.environ["SO8T_TRAINING_CONFIG"] = args.training_config
    if args.base_model:
        os.environ["SO8T_BASE_MODEL"] = args.base_model
    if args.subagent_strategy:
        os.environ["SO8T_SUBAGENT_STRATEGY"] = args.subagent_strategy
    if args.subagent_schedule:
        os.environ["SO8T_SUBAGENT_SCHEDULE"] = "1"
    if args.dynamic_dispatch:
        os.environ["SO8T_DYNAMIC_DISPATCH"] = "1"
    if args.skip_enrichment:
        os.environ["SO8T_ENRICH_ACADEMIC"] = "0"
        os.environ["SO8T_ENRICH_POP"] = "0"
        os.environ["SO8T_ENRICH_WORLD"] = "0"
        os.environ["SO8T_ENRICH_PHARMA"] = "0"
    if args.skip_reward_strategy:
        os.environ["SO8T_REWARD_STRATEGY"] = "0"
    if args.enable_mhc:
        os.environ["SO8T_MHC_ENABLE"] = "1"
    if args.enable_so8:
        os.environ["SO8T_SO8_ENABLE"] = "1"
    if args.so8_mode:
        os.environ["SO8T_SO8_MODE"] = args.so8_mode
    if args.mhc_targets:
        os.environ["SO8T_MHC_TARGETS"] = args.mhc_targets
    if args.mhc_blend:
        os.environ["SO8T_MHC_BLEND"] = str(args.mhc_blend)

    pipeline = IntegratedMoonshotPipeline2025_2026()
    startup = StartupManager(Path(__file__))

    if args.list_datasets:
        datasets = pipeline.discover_existing_datasets()
        print("\n=== Discovered datasets ===")
        for cat, files in datasets.items():
            print(f"\n[{cat}]")
            for f in files:
                print(f"  - {f}")
        return

    startup_enabled = os.getenv("SO8T_STARTUP_REGISTER", "1") == "1"
    if startup_enabled:
        # Register startup auto-resume
        startup_args = ["--use-existing-datasets"]
        if args.collect_new_data:
            startup_args = ["--collect-new-data"]
        if args.grape_variant:
            startup_args.extend(["--grape-variant", args.grape_variant])
        if args.use_unsloth:
            startup_args.append("--use-unsloth")
        if args.mcp_api_skill:
            startup_args.append("--mcp-api-skill")
        if args.recover:
            startup_args.append("--recover")
        if args.training_config:
            startup_args.extend(["--training-config", args.training_config])
        if args.subagent_strategy:
            startup_args.extend(["--subagent-strategy", args.subagent_strategy])
        if args.subagent_schedule:
            startup_args.append("--subagent-schedule")
        if args.dynamic_dispatch:
            startup_args.append("--dynamic-dispatch")
        if args.skip_enrichment:
            startup_args.append("--skip-enrichment")
        if args.skip_reward_strategy:
            startup_args.append("--skip-reward-strategy")
        if args.enable_mhc:
            startup_args.append("--enable-mhc")
        if args.enable_so8:
            startup_args.append("--enable-so8")
        if args.so8_mode:
            startup_args.extend(["--so8-mode", args.so8_mode])
        if args.mhc_targets:
            startup_args.extend(["--mhc-targets", args.mhc_targets])
        if args.mhc_blend:
            startup_args.extend(["--mhc-blend", str(args.mhc_blend)])
        startup.register(extra_args=startup_args)

    use_existing = args.use_existing_datasets and not args.collect_new_data
    pipeline.execute_full_pipeline(use_existing_datasets=use_existing)

    # Unregister after successful completion
    if startup_enabled:
        startup.unregister()


if __name__ == "__main__":
    main()

