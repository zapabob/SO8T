#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Data Cleansing Utility (95% Significance Level)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class StatisticalDataCleanser:
    def __init__(self, significance_level: float = 0.95):
        self.significance_level = significance_level
        # For 95% significance (two-tailed), z-score is ~1.96
        self.z_threshold = 1.96 if significance_level == 0.95 else 2.58 # 99% is 2.58

    def calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics."""
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values)
        }

    def cleanse_dataset(self, input_path: Path, output_path: Path, metric_key: str = "text_length") -> Dict[str, Any]:
        """
        Cleanse dataset using z-score filtering for the specified metric.
        Default metric is text length.
        """
        logger.info(f"[CLEANSE] Reading dataset from {input_path}")
        samples = []
        metrics = []

        if not input_path.exists():
            logger.error(f"Input path {input_path} does not exist.")
            return {"status": "error", "message": "Input path missing"}

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    # Calculate metric
                    if metric_key == "text_length":
                        val = len(sample.get("text", ""))
                    elif metric_key == "complexity":
                        words = sample.get("text", "").split()
                        val = len(set(words)) / len(words) if words else 0
                    else:
                        val = sample.get(metric_key, 0)
                    
                    sample["_temp_metric"] = val
                    samples.append(sample)
                    metrics.append(val)
                except Exception as e:
                    logger.warning(f"Failed to parse line: {e}")

        if not metrics:
            return {"status": "error", "message": "No metrics found"}

        stats = self.calculate_stats(metrics)
        mu = stats["mean"]
        sigma = stats["std"]

        # 95% significance level filtering
        lower_bound = mu - self.z_threshold * sigma
        upper_bound = mu + self.z_threshold * sigma

        logger.info(f"[CLEANSE] Stats: mu={mu:.2f}, sigma={sigma:.2f}. Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

        cleansed_samples = []
        outliers_count = 0
        for s in samples:
            val = s["_temp_metric"]
            if lower_bound <= val <= upper_bound:
                del s["_temp_metric"]
                cleansed_samples.append(s)
            else:
                outliers_count += 1

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in cleansed_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        results = {
            "status": "success",
            "total_before": len(samples),
            "total_after": len(cleansed_samples),
            "outliers_removed": outliers_count,
            "stats": stats,
            "bounds": {"lower": lower_bound, "upper": upper_bound}
        }
        logger.info(f"[CLEANSE] Completed. Removed {outliers_count} outliers. Saved to {output_path}")
        return results

if __name__ == "__main__":
    # Test logic
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        for i in range(100):
            text = "a" * (100 if i < 95 else 1000) # Outliers
            tmp.write(json.dumps({"text": text}) + "\n")
        tmp_path = Path(tmp.name)
    
    out_path = tmp_path.with_name("cleansed_test.jsonl")
    cleanser = StatisticalDataCleanser()
    res = cleanser.cleanse_dataset(tmp_path, out_path)
    print(json.dumps(res, indent=2))
