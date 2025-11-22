#!/usr/bin/env python3
"""
NKAT-SO8T Model Evaluation Script
検証データでのパフォーマンス確認と評価レポート生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datetime import datetime
import time
import gc

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.nkat_wrapper import NKAT_SO8T_Adapter
from scripts.training.train_nkat_so8t_adapter import NKATSO8TTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationDataset(Dataset):
    """評価用データセット"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')

        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze(),  # For language modeling
            'text': text,
            'category': item.get('category', 'unknown')
        }


class ModelEvaluator:
    """NKAT-SO8Tモデル評価クラス"""

    def __init__(self, model_path: str, test_data_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-3.5-mini-instruct",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model()

        # Setup evaluation dataset
        self.test_dataset = EvaluationDataset(test_data_path, self.tokenizer)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=4,  # Smaller batch for evaluation
            shuffle=False,
            collate_fn=self._collate_fn
        )

        # Metrics
        self.metrics = {}

    def _load_model(self):
        """Load the trained NKAT-SO8T model"""
        try:
            # Try to load from checkpoint
            if (self.model_path / "pytorch_model.bin").exists() or \
               any((self.model_path / f"pytorch_model-{i}.bin").exists()
                   for i in range(10)):

                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/phi-3.5-mini-instruct",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )

                # Load NKAT-SO8T adapter
                nkat_adapter = NKAT_SO8T_Adapter(
                    hidden_size=base_model.config.hidden_size,
                    num_blocks=8  # SO(8) blocks
                )

                # Load adapter weights if available
                adapter_path = self.model_path / "nkat_adapter.pt"
                if adapter_path.exists():
                    adapter_state = torch.load(adapter_path, map_location='cpu')
                    nkat_adapter.load_state_dict(adapter_state)
                    logger.info("Loaded NKAT-SO8T adapter weights")

                # Attach adapter to model
                base_model.nkat_adapter = nkat_adapter

                return base_model

            else:
                logger.warning(f"No model checkpoint found at {self.model_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def _collate_fn(self, batch):
        """Collate function for evaluation"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'texts': [item['text'] for item in batch],
            'categories': [item['category'] for item in batch]
        }

    def evaluate(self):
        """メイン評価関数"""
        if self.model is None:
            logger.error("Model not loaded, skipping evaluation")
            return {}

        logger.info("Starting NKAT-SO8T model evaluation...")
        self.model.eval()

        all_losses = []
        all_perplexities = []
        category_metrics = {}
        sample_results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                try:
                    # Move to device
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)
                    labels = batch['labels'].to(self.model.device)

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    # Calculate loss
                    loss = outputs.loss.item()
                    all_losses.append(loss)

                    # Calculate perplexity
                    perplexity = torch.exp(torch.tensor(loss)).item()
                    all_perplexities.append(perplexity)

                    # Store sample results
                    for i in range(len(batch['texts'])):
                        sample_results.append({
                            'text': batch['texts'][i][:100] + '...' if len(batch['texts'][i]) > 100 else batch['texts'][i],
                            'category': batch['categories'][i],
                            'loss': loss,
                            'perplexity': perplexity
                        })

                    # Category-wise metrics
                    for cat in batch['categories']:
                        if cat not in category_metrics:
                            category_metrics[cat] = []
                        category_metrics[cat].append(loss)

                    if batch_idx % 10 == 0:
                        logger.info(f"Evaluated {batch_idx + 1}/{len(self.test_loader)} batches")

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue

        # Calculate final metrics
        self.metrics = {
            'total_samples': len(self.test_dataset),
            'average_loss': np.mean(all_losses) if all_losses else float('inf'),
            'average_perplexity': np.mean(all_perplexities) if all_perplexities else float('inf'),
            'loss_std': np.std(all_losses) if all_losses else 0,
            'perplexity_std': np.std(all_perplexities) if all_perplexities else 0,
            'min_loss': min(all_losses) if all_losses else float('inf'),
            'max_loss': max(all_losses) if all_losses else float('inf'),
        }

        # Category-wise metrics
        self.metrics['category_metrics'] = {}
        for cat, losses in category_metrics.items():
            self.metrics['category_metrics'][cat] = {
                'count': len(losses),
                'avg_loss': np.mean(losses),
                'avg_perplexity': np.mean([np.exp(l) for l in losses]),
            }

        # NKAT-SO8T specific evaluation
        self._evaluate_nkat_specific_metrics()

        # Save results
        self._save_results(sample_results)

        logger.info("Evaluation completed!")
        logger.info(f"Average Loss: {self.metrics['average_loss']:.4f}")
        logger.info(f"Average Perplexity: {self.metrics['average_perplexity']:.2f}")

        return self.metrics

    def _evaluate_nkat_specific_metrics(self):
        """NKAT-SO8T固有のメトリクス評価"""
        logger.info("Evaluating NKAT-SO8T specific metrics...")

        # Check if adapter is attached
        if not hasattr(self.model, 'nkat_adapter'):
            logger.warning("No NKAT adapter found in model")
            self.metrics['nkat_metrics'] = {'adapter_present': False}
            return

        adapter = self.model.nkat_adapter

        # Alpha gate analysis
        alpha_gates = []
        for layer_idx in range(len(adapter.layers)):
            if hasattr(adapter.layers[layer_idx], 'alpha_gate'):
                alpha_val = adapter.layers[layer_idx].alpha_gate.alpha.item()
                gate_activation = torch.sigmoid(torch.tensor(alpha_val)).item()
                alpha_gates.append({
                    'layer': layer_idx,
                    'alpha_value': alpha_val,
                    'gate_activation': gate_activation
                })

        # Geometric consistency check (simplified)
        geometric_consistency = 0.0
        if hasattr(adapter, 'so8t_layer'):
            # Check if SO(8) transformations are properly learned
            with torch.no_grad():
                test_input = torch.randn(1, adapter.hidden_size, device=self.model.device)
                try:
                    output = adapter(test_input)
                    # Basic sanity check
                    geometric_consistency = 1.0 if output.shape == test_input.shape else 0.0
                except:
                    geometric_consistency = 0.0

        self.metrics['nkat_metrics'] = {
            'adapter_present': True,
            'alpha_gates': alpha_gates,
            'avg_gate_activation': np.mean([g['gate_activation'] for g in alpha_gates]) if alpha_gates else 0,
            'geometric_consistency': geometric_consistency,
            'phase_transition_achieved': any(g['gate_activation'] > 0.1 for g in alpha_gates) if alpha_gates else False
        }

    def _save_results(self, sample_results):
        """結果を保存"""
        # Save metrics
        metrics_file = self.output_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        # Save sample results
        samples_file = self.output_dir / "sample_results.jsonl"
        with open(samples_file, 'w', encoding='utf-8') as f:
            for result in sample_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # Generate report
        report_file = self.output_dir / "evaluation_report.md"
        self._generate_report(report_file)

        logger.info(f"Results saved to {self.output_dir}")

    def _generate_report(self, report_path):
        """評価レポート生成"""
        report = f"""# NKAT-SO8T Model Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- **Model Path**: {self.model_path}
- **Test Data**: {self.test_data_path}
- **Total Samples**: {self.metrics.get('total_samples', 0)}

## Performance Metrics

### Overall Performance
- **Average Loss**: {self.metrics.get('average_loss', 0):.4f}
- **Average Perplexity**: {self.metrics.get('average_perplexity', 0):.2f}
- **Loss Std**: {self.metrics.get('loss_std', 0):.4f}
- **Perplexity Std**: {self.metrics.get('perplexity_std', 0):.4f}

### Range Analysis
- **Min Loss**: {self.metrics.get('min_loss', 0):.4f}
- **Max Loss**: {self.metrics.get('max_loss', 0):.4f}

## Category-wise Performance

"""
        for cat, metrics in self.metrics.get('category_metrics', {}).items():
            report += f"### {cat.title()}\n"
            report += f"- **Count**: {metrics['count']}\n"
            report += f"- **Avg Loss**: {metrics['avg_loss']:.4f}\n"
            report += f"- **Avg Perplexity**: {metrics['avg_perplexity']:.2f}\n\n"

        # NKAT-specific metrics
        nkat = self.metrics.get('nkat_metrics', {})
        report += "## NKAT-SO8T Specific Metrics\n\n"
        report += f"- **Adapter Present**: {'✅ Yes' if nkat.get('adapter_present', False) else '❌ No'}\n"

        if nkat.get('adapter_present', False):
            report += f"- **Average Gate Activation**: {nkat.get('avg_gate_activation', 0):.4f}\n"
            report += f"- **Phase Transition Achieved**: {'✅ Yes' if nkat.get('phase_transition_achieved', False) else '❌ No'}\n"
            report += f"- **Geometric Consistency**: {nkat.get('geometric_consistency', 0):.4f}\n\n"

            report += "### Alpha Gates Status\n"
            for gate in nkat.get('alpha_gates', []):
                report += f"- **Layer {gate['layer']}**: α={gate['alpha_value']:.4f}, activation={gate['gate_activation']:.4f}\n"

        report += "\n## Evaluation Summary\n\n"
        avg_loss = self.metrics.get('average_loss', float('inf'))
        if avg_loss < 2.0:
            performance = "Excellent"
        elif avg_loss < 3.0:
            performance = "Good"
        elif avg_loss < 4.0:
            performance = "Fair"
        else:
            performance = "Needs Improvement"

        report += f"- **Overall Performance**: {performance}\n"
        report += f"- **NKAT Integration**: {'✅ Successful' if nkat.get('phase_transition_achieved', False) else '❌ Not Achieved'}\n"
        report += f"- **Ready for Production**: {'✅ Yes' if avg_loss < 3.0 and nkat.get('phase_transition_achieved', False) else '❌ No'}\n"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NKAT-SO8T Model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--test-data", type=str, required=True,
                       help="Path to test/validation data")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.test_data, args.output_dir)

    # Run evaluation
    metrics = evaluator.evaluate()

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Loss: {metrics.get('average_loss', 0):.4f}")
    print(f"Average Perplexity: {metrics.get('average_perplexity', 0):.4f}")

    nkat_metrics = metrics.get('nkat_metrics', {})
    if nkat_metrics.get('adapter_present'):
        print(f"Phase Transition: {'Achieved' if nkat_metrics.get('phase_transition_achieved') else 'Not Achieved'}")
        print(f"Average Gate Activation: {nkat_metrics.get('avg_gate_activation', 0):.4f}")
    else:
        print("NKAT Adapter: Not Found")

    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
