#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama-based Synthetic Data Generation for SO8T
================================================
Generates high-quality synthetic data using Borea-phi3.5-instinct-jp on Ollama (CPU).
Includes logic for data cleansing and evolutionary expansion.
"""

import json
import logging
import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaSyntheticGenerator:
    def __init__(self, model_name: str = "Borea-phi3.5-instinct-jp", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = f"{base_url}/api/generate"
        self.output_dir = Path("data/synthetic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_sample(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generates a single sample from Ollama."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(self.base_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None

    def cleanse_data(self, text: str) -> bool:
        """
        Cleanses the generated data.
        Returns True if the data passes quality checks.
        """
        # SO8T Quadruple Inference tags check
        required_tags = ["<think-task>", "<think-analysis>", "<think-safety>", "<think-policy>"]
        has_all_tags = all(tag in text for tag in required_tags)
        
        # Basic length and content checks
        is_long_enough = len(text) > 200
        
        return has_all_tags and is_long_enough

    def evolutionary_expand(self, seeds: List[str], iterations: int = 1):
        """
        Expands the dataset evolutionarily by using previous high-quality samples as seeds.
        """
        logger.info(f"Starting evolutionary expansion for {iterations} iterations...")
        current_seeds = seeds
        
        for i in range(iterations):
            new_samples = []
            for seed in current_seeds:
                prompt = f"Based on this example of high-quality reasoning, generate a new complex problem and solve it using the same SO8T quadruple inference structure.\n\nExample:\n{seed}"
                
                output = self.generate_sample(prompt)
                if output and self.cleanse_data(output):
                    new_samples.append(output)
                    self.save_sample(output, f"evolved_v{i}_{len(new_samples)}")
            
            if not new_samples:
                logger.warning(f"No high-quality samples generated in iteration {i}")
                break
                
            current_seeds = new_samples
            logger.info(f"Iteration {i} completed with {len(new_samples)} new samples.")

    def save_sample(self, text: str, name: str):
        filepath = self.output_dir / f"{name}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            json.dump({"text": text, "metadata": {"generator": "OllamaSyntheticGenerator", "timestamp": time.time()}}, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    generator = OllamaSyntheticGenerator()
    # Simple test seed
    test_seed = """<think-task>Analyze the impact of PET on residual networks.</think-task>
<think-analysis>PET regularization provides a way to maintain smoothness across inference passes.</think-analysis>
<think-safety>Ensure the gradients do not explode during backprop.</think-safety>
<think-policy>Follow the SO8T reasoning framework strictly.</think-policy>
The implementation shows that second-order differences stabilize the gate coefficients alpha."""
    
    # generator.evolutionary_expand([test_seed], iterations=1)
    logger.info("OllamaSyntheticGenerator initialized.")
