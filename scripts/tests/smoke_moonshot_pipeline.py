#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for integrated Moonshot pipeline.

Runs the pipeline with SO8T_DRYRUN=1 to avoid heavy model downloads and
conversion steps. Intended for quick verification after implementation changes.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.integrated_moonshot_pipeline_2025_2026 import (
    IntegratedMoonshotPipeline2025_2026,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Moonshot pipeline")
    parser.add_argument(
        "--grape-variant",
        default=os.getenv("SO8T_GRAPE_VARIANT", "multiplicative"),
        help="GRAPE variant to simulate (multiplicative/additive/hybrid)",
    )
    args = parser.parse_args()

    os.environ["SO8T_DRYRUN"] = "1"
    os.environ["SO8T_GRAPE_VARIANT"] = args.grape_variant

    logger.info("Starting smoke test at %s", datetime.now().isoformat())
    pipeline = IntegratedMoonshotPipeline2025_2026()
    pipeline.execute_full_pipeline(use_existing_datasets=True)
    logger.info("Smoke test completed")


if __name__ == "__main__":
    main()

