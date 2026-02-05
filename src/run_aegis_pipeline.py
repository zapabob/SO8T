#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS-v3.0 全自動パイプライン オーケストレーター

Phase 4 (高度データ拡充) から Phase 6 (統計的ベンチマーク) までを
一気通貫で実行する統合パイプラインです。

Usage:
    python run_aegis_pipeline.py [--phase 4|5|6|all] [--resume]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "aegis_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_phase4() -> Path:
    """Phase 4: 高度データ拡充"""
    logger.info("=" * 60)
    logger.info("PHASE 4: Advanced Data Enrichment")
    logger.info("=" * 60)
    
    from src.data.phase4_data_enrichment_pipeline import Phase4DataEnrichmentPipeline
    
    pipeline = Phase4DataEnrichmentPipeline()
    output_path = pipeline.run()
    
    return output_path


def run_phase5(dataset_path: Path = None, resume: bool = True) -> bool:
    """Phase 5: 全自動再学習"""
    logger.info("=" * 60)
    logger.info("PHASE 5: Auto-Retraining (Borea -> AEGIS-v3.0)")
    logger.info("=" * 60)
    
    from src.training.phase5_auto_retraining_pipeline import Phase5AutoRetrainingPipeline
    
    pipeline = Phase5AutoRetrainingPipeline(dataset_path=dataset_path)
    success = pipeline.run(resume=resume)
    
    return success


def run_phase6() -> bool:
    """Phase 6: 統計的ベンチマーク (A/B/C test)"""
    logger.info("=" * 60)
    logger.info("PHASE 6: Statistical Benchmarking (ANOVA, Cohen's d)")
    logger.info("=" * 60)
    
    from src.evaluation.phase6_statistical_benchmark import Phase6StatisticalBenchmark
    
    pipeline = Phase6StatisticalBenchmark()
    output_path = pipeline.run()
    
    logger.info(f"Phase 6 results: {output_path}")
    return True



def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS-v3.0 Full Pipeline Orchestrator")
    parser.add_argument("--phase", type=str, default="all", choices=["4", "5", "6", "all"],
                        help="Which phase to run (4, 5, 6, or all)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AEGIS-v3.0 Full Pipeline Orchestrator")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 60)
    
    dataset_path = None
    
    if args.phase in ["4", "all"]:
        dataset_path = run_phase4()
    
    if args.phase in ["5", "all"]:
        success = run_phase5(dataset_path=dataset_path, resume=args.resume)
        if not success:
            logger.error("Phase 5 failed. Stopping pipeline.")
            sys.exit(1)
    
    if args.phase in ["6", "all"]:
        run_phase6()
    
    logger.info("=" * 60)
    logger.info("AEGIS-v3.0 Pipeline Complete!")
    logger.info(f"End time: {datetime.now().isoformat()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
