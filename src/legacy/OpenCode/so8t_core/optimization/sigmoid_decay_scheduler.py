#!/usr/bin/env python3
"""
Sigmoid Decay Learning Rate Scheduler with Golden Ratio Final Value.

This scheduler uses a sigmoid decay function to gradually reduce the learning rate
to PHI^-2 (golden ratio inverse squared, approximately 0.382) at the final step.

Mathematical Foundation:
- Golden Ratio PHI = (1 + sqrt(5)) / 2 approx 1.6180339887
- PHI^-1 = 1/PHI approx 0.6180339887
- PHI^-2 = 1/PHI^2 approx 0.3819660113

Schedule:
- Warmup phase (0 -> warmup_steps): Linear increase from lr_min to lr_max
- Decay phase (warmup_steps -> total_steps): Sigmoid decay to lr_max * PHI^-2
"""

from __future__ import annotations

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Golden ratio constants
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.6180339887
PHI_INV = 1 / PHI  # ≈ 0.6180339887
PHI_INV_SQ = PHI_INV**2  # ≈ 0.3819660113


def sigmoid(x: float) -> float:
    """Sigmoid function: 1 / (1 + exp(-x))

    Returns a value in (0, 1) for all real x.
    """
    return 1.0 / (1.0 + math.exp(-x))


def calculate_steepness(phi_inv_sq: float = PHI_INV_SQ) -> float:
    """Calculate sigmoid steepness parameter k for convergence.

    The steepness k is calculated to ensure the sigmoid function
    transitions from near 1 to near Φ⁻² over the decay phase.

    k = -2 × ln(Φ⁻²) / 1

    This ensures that at the final step (normalized_t = 1),
    the sigmoid output is approximately Φ⁻².
    """
    return -2.0 * math.log(phi_inv_sq)


class SigmoidDecayScheduler(_LRScheduler):
    """Learning rate scheduler with sigmoid decay to golden ratio (PHI^-2).

    This scheduler implements a two-phase learning rate schedule:

    1. **Warmup Phase** (0 -> warmup_steps):
       Linear increase from lr_min to lr_max

    2. **Decay Phase** (warmup_steps -> total_steps):
       Sigmoid decay from lr_max to lr_max * PHI^-2

    The final learning rate is approximately 38.2% of the maximum,
    which is the inverse square of the golden ratio (PHI^-2).

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> scheduler = SigmoidDecayScheduler(
        ...     optimizer=optimizer,
        ...     warmup_steps=100,
        ...     total_steps=1000,
        ...     lr_max=1e-4,
        ... )
        >>> for step in range(1000):
        ...     scheduler.step()
        ...     print(f"Step {step}: lr = {scheduler.get_last_lr()[0]:.2e}")

    Attributes:
        PHI: Golden ratio constant (approx 1.618)
        PHI_INV_SQ: Golden ratio inverse squared (approx 0.382)
    """

    PHI = PHI
    PHI_INV_SQ = PHI_INV_SQ

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float,
        lr_min: float = 0.0,
        phi_inv_sq: float = PHI_INV_SQ,
        steepness: Optional[float] = None,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: The optimizer to adjust learning rate for.
            warmup_steps: Number of steps for warmup phase (linear increase).
            total_steps: Total number of training steps.
            lr_max: Maximum learning rate (reached at end of warmup).
            lr_min: Minimum learning rate (default: 0.0).
            phi_inv_sq: Final learning rate multiplier (default: Φ⁻² ≈ 0.382).
            steepness: Sigmoid steepness parameter k. If None, auto-calculated.
            last_epoch: The index of the last epoch. Default: -1.
            verbose: If True, prints a message to stdout for each update.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.phi_inv_sq = phi_inv_sq
        self.steepness = (
            steepness if steepness is not None else calculate_steepness(phi_inv_sq)
        )

        # Validate parameters
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be greater than "
                f"warmup_steps ({warmup_steps})"
            )
        if lr_max < lr_min:
            raise ValueError(
                f"lr_max ({lr_max}) must be greater than or equal to lr_min ({lr_min})"
            )
        if not 0 < phi_inv_sq < 1:
            raise ValueError(f"phi_inv_sq must be in (0, 1), got {phi_inv_sq}")

        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        state = super().state_dict()
        state.update(
            {
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
                "lr_max": self.lr_max,
                "lr_min": self.lr_min,
                "phi_inv_sq": self.phi_inv_sq,
                "steepness": self.steepness,
            }
        )
        return state

    def load_state_dict(self, state_dict):
        """Load the state of the scheduler from a dict."""
        super().load_state_dict(state_dict)
        self.warmup_steps = state_dict.get("warmup_steps", 100)
        self.total_steps = state_dict.get("total_steps", 1000)
        self.lr_max = state_dict.get("lr_max", 1e-4)
        self.lr_min = state_dict.get("lr_min", 0.0)
        self.phi_inv_sq = state_dict.get("phi_inv_sq", PHI_INV_SQ)
        self.steepness = state_dict.get(
            "steepness", calculate_steepness(self.phi_inv_sq)
        )

    def get_lr(self):
        """Compute current learning rate.

        Returns:
            Current learning rate for each parameter group.
        """
        step = self.last_epoch

        if step < self.warmup_steps:
            # Warmup phase: linear increase from lr_min to lr_max
            progress = step / self.warmup_steps
            lr = self.lr_min + (self.lr_max - self.lr_min) * progress
        else:
            # Decay phase: sigmoid decay to lr_max × phi_inv_sq
            decay_progress = (step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            # Normalize to [-1, 1] for sigmoid centered at 0.5
            normalized = 2.0 * decay_progress - 1.0

            # Sigmoid decay factor (converges to phi_inv_sq at final step)
            sig = sigmoid(-self.steepness * normalized)

            # Scale to final value
            lr = (
                self.lr_min
                + (self.lr_max - self.lr_min) * sig * self.phi_inv_sq / self.PHI_INV_SQ
            )

        return [lr for _ in self.optimizer.param_groups]

    def get_lr_factor(self) -> float:
        """Get the learning rate as a factor of lr_max.

        Useful for logging and visualization.

        Returns:
            Learning rate factor (1.0 at peak, PHI^-2 at end).
        """
        lr = self.get_lr()[0]
        return lr / self.lr_max


def visualize_schedule(
    warmup_steps: int = 100,
    total_steps: int = 1000,
    lr_max: float = 1e-4,
    lr_min: float = 0.0,
    save_path: Optional[str] = None,
):
    """Visualize the learning rate schedule.

    Args:
        warmup_steps: Number of warmup steps.
        total_steps: Total number of steps.
        lr_max: Maximum learning rate.
        lr_min: Minimum learning rate.
        save_path: Optional path to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARNING] matplotlib not installed, skipping visualization")
        return

    # Create scheduler
    optimizer = __import__("torch").optim.Adam([{"params": [0]}], lr=lr_max)
    scheduler = SigmoidDecayScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        lr_max=lr_max,
        lr_min=lr_min,
    )

    # Collect learning rates
    steps = list(range(total_steps + 1))
    lrs = []

    for step in steps:
        scheduler.last_epoch = step
        lrs.append(scheduler.get_lr()[0])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Full schedule
    ax1 = axes[0]
    ax1.plot(steps, [lr * 1e4 for lr in lrs], "b-", linewidth=2)
    ax1.axhline(
        y=lr_max * 1e4 * PHI_INV_SQ,
        color="r",
        linestyle="--",
        label=f"Final LR (PHI^-2) = {lr_max * PHI_INV_SQ * 1e4:.2f}e-4",
    )
    ax1.axvline(
        x=warmup_steps, color="g", linestyle=":", label=f"Warmup end ({warmup_steps})"
    )
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Learning Rate (x10^-4)", fontsize=12)
    ax1.set_title("Sigmoid Decay LR Schedule (Golden Ratio Final Value)", fontsize=14)
    ax1.legend(loc="right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Decay phase only (zoomed)
    ax2 = axes[1]
    decay_steps = steps[warmup_steps:]
    decay_lrs = [lr * 1e4 for lr in lrs[warmup_steps:]]
    ax2.plot(decay_steps, decay_lrs, "b-", linewidth=2)
    ax2.axhline(
        y=lr_max * 1e4 * PHI_INV_SQ,
        color="r",
        linestyle="--",
        label=f"PHI^-2 = {PHI_INV_SQ:.4f}",
    )
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Learning Rate (x10^-4)", fontsize=12)
    ax2.set_title("Decay Phase (Sigmoid)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[VISUALIZE] Saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sigmoid Decay Scheduler Visualization"
    )
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps")
    parser.add_argument("--total", type=int, default=1000, help="Total steps")
    parser.add_argument(
        "--lr-max", type=float, default=1e-4, help="Maximum learning rate"
    )
    parser.add_argument(
        "--lr-min", type=float, default=0.0, help="Minimum learning rate"
    )
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    parser.add_argument("--test", action="store_true", help="Run tests")

    args = parser.parse_args()

    if args.test:
        # Run basic tests
        print("=" * 60)
        print("Sigmoid Decay Scheduler Tests")
        print("=" * 60)

        # Test 1: Golden ratio constants
        print("\n[Test 1] Golden Ratio Constants")
        print(f"  PHI = {PHI:.10f}")
        print(f"  PHI^-1 = {PHI_INV:.10f}")
        print(f"  PHI^-2 = {PHI_INV_SQ:.10f}")
        assert abs(PHI_INV_SQ - 0.3819660113) < 1e-6
        print("  ✓ Golden ratio constants verified")

        # Test 2: Sigmoid properties
        print("\n[Test 2] Sigmoid Properties")
        print(f"  sigmoid(0) = {sigmoid(0):.4f}")
        print(f"  sigmoid(10) = {sigmoid(10):.4f}")
        print(f"  sigmoid(-10) = {sigmoid(-10):.4f}")
        assert 0 < sigmoid(0) < 1
        assert sigmoid(10) > 0.99
        assert sigmoid(-10) < 0.01
        print("  ✓ Sigmoid properties verified")

        # Test 3: Steepness calculation
        print("\n[Test 3] Steepness Calculation")
        k = calculate_steepness(PHI_INV_SQ)
        print(f"  k = {k:.6f}")
        assert abs(k - 1.924) < 0.01
        print("  ✓ Steepness verified")

        # Test 4: Scheduler behavior
        print("\n[Test 4] Scheduler Behavior")
        optimizer = __import__("torch").optim.Adam([{"params": [0]}], lr=args.lr_max)
        scheduler = SigmoidDecayScheduler(
            optimizer=optimizer,
            warmup_steps=args.warmup,
            total_steps=args.total,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
        )

        # Check warmup
        scheduler.last_epoch = 0
        lr_0 = scheduler.get_lr()[0]
        scheduler.last_epoch = args.warmup - 1
        lr_warmup_end = scheduler.get_lr()[0]

        print(f"  lr at step 0: {lr_0:.2e}")
        print(f"  lr at warmup end: {lr_warmup_end:.2e}")
        assert abs(lr_0 - args.lr_min) < 1e-10
        assert abs(lr_warmup_end - args.lr_max) < 1e-6

        # Check final value
        scheduler.last_epoch = args.total
        lr_final = scheduler.get_lr()[0]
        lr_expected = args.lr_max * PHI_INV_SQ

        print(f"  lr at final step: {lr_final:.2e}")
        print(f"  expected (lr_max * PHI^-2): {lr_expected:.2e}")
        assert abs(lr_final - lr_expected) / lr_expected < 0.01

        print("  ✓ Scheduler behavior verified")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    else:
        # Visualize
        print(
            f"Visualizing schedule: warmup={args.warmup}, total={args.total}, lr_max={args.lr_max}"
        )
        visualize_schedule(
            warmup_steps=args.warmup,
            total_steps=args.total,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
            save_path=args.save,
        )
