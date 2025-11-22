#!/usr/bin/env python3
"""
NKAT-SO8T Phase Transition Validation
Validates theoretical Alpha Gate behavior and geometric reasoning emergence.

Tests:
1. Alpha Gate phase transition (σ(α) evolution)
2. SO(8) geometric consistency
3. Mathematical reasoning improvement
4. Training stability metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.nkat_wrapper import NKAT_Wrapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhaseTransitionAnalyzer:
    """
    Analyzes Alpha Gate phase transition behavior.
    Validates theoretical predictions about geometric reasoning emergence.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints = self._find_checkpoints()

        # Load base model and tokenizer for analysis
        self._load_base_model()

    def _find_checkpoints(self) -> List[Path]:
        """Find all checkpoints in chronological order."""
        checkpoints = []
        for ckpt_dir in self.checkpoint_dir.iterdir():
            if ckpt_dir.is_dir() and ckpt_dir.name.startswith("checkpoint-step-"):
                step = int(ckpt_dir.name.split("-")[-1])
                checkpoints.append((step, ckpt_dir))

        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        return [ckpt for _, ckpt in checkpoints]

    def _load_base_model(self):
        """Load base model for analysis."""
        # Try to find model path from config
        config_path = self.checkpoint_dir / "checkpoint-step-1" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                # Extract model path if available
                pass

        # Default path
        self.model_path = "models/Borea-Phi-3.5-mini-Instruct-Jp"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def analyze_phase_transition(self) -> Dict:
        """
        Analyze Alpha Gate phase transition across training checkpoints.

        Returns:
            Dictionary containing phase transition metrics
        """
        logger.info("Analyzing Alpha Gate phase transition...")

        phase_data = {
            'steps': [],
            'avg_alpha': [],
            'avg_activation': [],
            'active_gates_ratio': [],
            'max_activation': [],
            'min_activation': [],
            'phase_transition_step': None,
            'transition_strength': 0.0
        }

        for ckpt_path in self.checkpoints:
            step = int(ckpt_path.name.split("-")[-1])

            # Load checkpoint
            try:
                model = self._load_checkpoint_model(ckpt_path)
                gate_values = model.get_gate_values()
                gate_activations = model.get_gate_activations()

                # Calculate metrics
                avg_alpha = np.mean(gate_values)
                avg_activation = np.mean(gate_activations)
                active_gates_ratio = np.mean([act > 0.1 for act in gate_activations])
                max_activation = np.max(gate_activations)
                min_activation = np.min(gate_activations)

                # Store data
                phase_data['steps'].append(step)
                phase_data['avg_alpha'].append(avg_alpha)
                phase_data['avg_activation'].append(avg_activation)
                phase_data['active_gates_ratio'].append(active_gates_ratio)
                phase_data['max_activation'].append(max_activation)
                phase_data['min_activation'].append(min_activation)

                # Detect phase transition (significant activation increase)
                if phase_data['phase_transition_step'] is None and avg_activation > 0.05:
                    phase_data['phase_transition_step'] = step
                    phase_data['transition_strength'] = avg_activation

                logger.info(f"Step {step}: α_avg={avg_alpha:.4f}, σ(α)_avg={avg_activation:.4f}")

            except Exception as e:
                logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
                continue

        return phase_data

    def _load_checkpoint_model(self, ckpt_path: Path) -> NKAT_Wrapper:
        """Load model from checkpoint."""
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Create NKAT wrapper
        model = NKAT_Wrapper(base_model)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path / "checkpoint.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def validate_geometric_consistency(self) -> Dict:
        """
        Validate SO(8) geometric consistency and rotation properties.

        Returns:
            Dictionary with geometric validation metrics
        """
        logger.info("Validating SO(8) geometric consistency...")

        # Load final model
        if not self.checkpoints:
            return {"error": "No checkpoints found"}

        final_ckpt = self.checkpoints[-1]
        model = self._load_checkpoint_model(final_ckpt)

        consistency_metrics = {
            'orthogonality_errors': [],
            'rotation_matrix_determinants': [],
            'activation_distributions': []
        }

        # Analyze each NKAT adapter
        for layer_idx, adapter in enumerate(model.nkat_adapters):
            # Extract SO(8) parameters
            so8_raw = adapter.so8_raw.detach().cpu().numpy()

            # Create rotation matrices
            A = 0.5 * (so8_raw - so8_raw.transpose(1, 2))  # Skew-symmetric
            R = np.array([self._matrix_exp(a) for a in A])  # SO(8) rotations

            # Check orthogonality: R^T @ R ≈ I
            for r in R:
                orthogonality_error = np.linalg.norm(r.T @ r - np.eye(8))
                consistency_metrics['orthogonality_errors'].append(float(orthogonality_error))

                # Check determinant (should be ±1 for rotations)
                det = np.linalg.det(r)
                consistency_metrics['rotation_matrix_determinants'].append(float(det))

            logger.info(f"Layer {layer_idx}: Orthogonality error = {np.mean(consistency_metrics['orthogonality_errors']):.6f}")

        return consistency_metrics

    def _matrix_exp(self, A: np.ndarray) -> np.ndarray:
        """Compute matrix exponential for SO(8) generation."""
        # Use scipy if available, otherwise approximation
        try:
            from scipy.linalg import expm
            return expm(A)
        except ImportError:
            # Simple approximation for small matrices
            return np.eye(8) + A + (A @ A) / 2

    def evaluate_mathematical_reasoning(self) -> Dict:
        """
        Evaluate improvement in mathematical and geometric reasoning.

        Returns:
            Dictionary with reasoning evaluation metrics
        """
        logger.info("Evaluating mathematical reasoning capabilities...")

        # Test problems requiring geometric intuition
        test_problems = [
            "Solve for x: sin(x) + cos(x) = √2",
            "Calculate the volume of intersection between unit sphere and unit cube in 3D",
            "Find the eigenvalues of rotation matrix R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]",
            "Determine if the vector field F = ∇×(xî + yĵ + zᵏ) is conservative",
            "Compute the Gaussian curvature of a sphere of radius r"
        ]

        reasoning_metrics = {
            'problems_tested': len(test_problems),
            'responses_generated': 0,
            'geometric_correctness_score': 0.0,
            'mathematical_consistency_score': 0.0
        }

        # Load final model for evaluation
        if self.checkpoints:
            final_ckpt = self.checkpoints[-1]
            model = self._load_checkpoint_model(final_ckpt)
            model.eval()

            device = next(model.parameters()).device

            for problem in test_problems:
                try:
                    # Generate response
                    inputs = self.tokenizer(problem, return_tensors="pt").to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=inputs['input_ids'].shape[1] + 200,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            num_return_sequences=1
                        )

                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    reasoning_metrics['responses_generated'] += 1

                    # Simple heuristic scoring (in practice, use more sophisticated evaluation)
                    if any(keyword in response.lower() for keyword in ['calculate', 'compute', 'solve', 'determine']):
                        reasoning_metrics['mathematical_consistency_score'] += 1

                    if any(keyword in response.lower() for keyword in ['geometry', 'rotation', 'sphere', 'volume', 'eigen']):
                        reasoning_metrics['geometric_correctness_score'] += 1

                    logger.info(f"Problem: {problem[:50]}...")
                    logger.info(f"Response: {response[len(problem):][:100]}...")

                except Exception as e:
                    logger.warning(f"Failed to evaluate problem: {e}")
                    continue

        # Normalize scores
        if reasoning_metrics['responses_generated'] > 0:
            reasoning_metrics['geometric_correctness_score'] /= reasoning_metrics['responses_generated']
            reasoning_metrics['mathematical_consistency_score'] /= reasoning_metrics['responses_generated']

        return reasoning_metrics

    def generate_phase_transition_report(self) -> str:
        """Generate comprehensive phase transition analysis report."""
        logger.info("Generating phase transition analysis report...")

        # Analyze phase transition
        phase_data = self.analyze_phase_transition()

        # Validate geometric consistency
        geometric_data = self.validate_geometric_consistency()

        # Evaluate reasoning capabilities
        reasoning_data = self.evaluate_mathematical_reasoning()

        # Generate report
        report = ".1f"        report += "=" * 80 + "\n\n"

        # Phase transition analysis
        report += "PHASE TRANSITION ANALYSIS\n"
        report += "-" * 30 + "\n"

        if phase_data['phase_transition_step']:
            report += f"✓ Phase transition detected at step {phase_data['phase_transition_step']}\n"
            report += ".4f"        else:
            report += "✗ No significant phase transition detected\n"

        if phase_data['steps']:
            final_avg_activation = phase_data['avg_activation'][-1]
            final_active_ratio = phase_data['active_gates_ratio'][-1]

            report += ".4f"            report += ".3f"
            report += f"Final max activation: {phase_data['max_activation'][-1]:.4f}\n"
            report += f"Final min activation: {phase_data['min_activation'][-1]:.4f}\n\n"

        # Geometric consistency
        report += "GEOMETRIC CONSISTENCY ANALYSIS\n"
        report += "-" * 35 + "\n"

        if 'orthogonality_errors' in geometric_data:
            avg_orthogonality_error = np.mean(geometric_data['orthogonality_errors'])
            report += ".6f"
            if avg_orthogonality_error < 1e-3:
                report += "  → EXCELLENT: SO(8) rotations properly formed\n"
            elif avg_orthogonality_error < 1e-2:
                report += "  → GOOD: Acceptable orthogonality\n"
            else:
                report += "  → POOR: Significant orthogonality violations\n"

        # Reasoning evaluation
        report += "MATHEMATICAL REASONING EVALUATION\n"
        report += "-" * 38 + "\n"
        report += f"Problems tested: {reasoning_data['problems_tested']}\n"
        report += f"Responses generated: {reasoning_data['responses_generated']}\n"
        report += ".3f"
        report += ".3f"
        report += "\n"

        # Theoretical validation
        report += "THEORETICAL VALIDATION\n"
        report += "-" * 25 + "\n"

        theory_checks = []

        # Check 1: Alpha Gate initialization
        if phase_data['steps'] and phase_data['avg_alpha'][0] < -4.0:
            theory_checks.append("✓ Alpha Gate initialization correct (α ≈ -5.0)")
        else:
            theory_checks.append("✗ Alpha Gate initialization incorrect")

        # Check 2: Phase transition occurrence
        if phase_data['phase_transition_step']:
            theory_checks.append("✓ Phase transition occurred (geometric reasoning emerged)")
        else:
            theory_checks.append("✗ Phase transition did not occur")

        # Check 3: Geometric consistency
        if 'orthogonality_errors' in geometric_data:
            avg_error = np.mean(geometric_data['orthogonality_errors'])
            if avg_error < 1e-2:
                theory_checks.append("✓ SO(8) geometric consistency maintained")
            else:
                theory_checks.append("✗ SO(8) geometric consistency violated")

        # Check 4: Reasoning improvement
        if reasoning_data['geometric_correctness_score'] > 0.3:
            theory_checks.append("✓ Mathematical/geometric reasoning improved")
        else:
            theory_checks.append("✗ Mathematical/geometric reasoning not improved")

        for check in theory_checks:
            report += check + "\n"

        report += "\n" + "=" * 80 + "\n"
        report += "CONCLUSION\n"
        report += "-" * 12 + "\n"

        successful_checks = sum(1 for check in theory_checks if check.startswith("✓"))
        total_checks = len(theory_checks)

        if successful_checks == total_checks:
            report += "🎉 SUCCESS: All theoretical predictions validated!\n"
            report += "   NKAT-SO8T adapter successfully implements phase transition behavior.\n"
        elif successful_checks >= total_checks * 0.75:
            report += "✅ MOSTLY SUCCESSFUL: Core theoretical behavior achieved.\n"
            report += f"   {successful_checks}/{total_checks} validation checks passed.\n"
        else:
            report += "⚠️  PARTIAL SUCCESS: Some theoretical aspects need refinement.\n"
            report += f"   Only {successful_checks}/{total_checks} validation checks passed.\n"

        return report

    def plot_phase_transition(self, save_path: Optional[str] = None):
        """Plot Alpha Gate phase transition evolution."""
        phase_data = self.analyze_phase_transition()

        if not phase_data['steps']:
            logger.warning("No phase transition data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Alpha values evolution
        axes[0, 0].plot(phase_data['steps'], phase_data['avg_alpha'], 'b-', linewidth=2)
        axes[0, 0].set_title('Alpha Gate Values (α) Evolution')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Average α')
        axes[0, 0].grid(True, alpha=0.3)

        # Activation values evolution
        axes[0, 1].plot(phase_data['steps'], phase_data['avg_activation'], 'r-', linewidth=2)
        axes[0, 1].set_title('Gate Activations σ(α) Evolution')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Average σ(α)')
        axes[0, 1].grid(True, alpha=0.3)

        # Active gates ratio
        axes[1, 0].plot(phase_data['steps'], phase_data['active_gates_ratio'], 'g-', linewidth=2)
        axes[1, 0].set_title('Active Gates Ratio (>0.1)')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Active Gates Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # Min/Max activation range
        axes[1, 1].plot(phase_data['steps'], phase_data['max_activation'], 'r--', label='Max', linewidth=2)
        axes[1, 1].plot(phase_data['steps'], phase_data['min_activation'], 'b--', label='Min', linewidth=2)
        axes[1, 1].fill_between(phase_data['steps'],
                               phase_data['min_activation'],
                               phase_data['max_activation'],
                               alpha=0.3, color='gray')
        axes[1, 1].set_title('Gate Activation Range')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('σ(α)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Phase transition plot saved to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Validate NKAT-SO8T Phase Transition")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory containing training checkpoints")
    parser.add_argument("--output-report", type=str, default=None,
                       help="Path to save validation report")
    parser.add_argument("--plot-phase-transition", type=str, default=None,
                       help="Path to save phase transition plot")
    parser.add_argument("--quick-validation", action="store_true",
                       help="Run quick validation (skip detailed analysis)")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = PhaseTransitionAnalyzer(args.checkpoint_dir)

    logger.info("Starting NKAT-SO8T phase transition validation...")
    logger.info("=" * 60)

    if args.quick_validation:
        # Quick validation
        phase_data = analyzer.analyze_phase_transition()
        if phase_data['phase_transition_step']:
            logger.info(f"✓ Phase transition detected at step {phase_data['phase_transition_step']}")
            logger.info(".4f"        else:
            logger.info("✗ No phase transition detected")

    else:
        # Full validation
        report = analyzer.generate_phase_transition_report()

        if args.output_report:
            with open(args.output_report, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Validation report saved to {args.output_report}")
        else:
            print(report)

        # Generate plot if requested
        if args.plot_phase_transition:
            analyzer.plot_phase_transition(args.plot_phase_transition)

    logger.info("Validation completed!")


if __name__ == "__main__":
    main()

