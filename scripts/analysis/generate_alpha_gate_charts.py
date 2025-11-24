#!/usr/bin/env python3
"""
Alpha Gate Training Loss Chart Generator
物理的Phase Transitionを示すLoss曲線を生成
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_training_log(log_path: Path) -> dict:
    """Load alpha gate training log"""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_loss_chart(data: dict, output_path: Path):
    """Create Loss vs Steps chart with Phase Transition highlight"""
    loss_history = data['loss_history']

    steps = [entry['step'] for entry in loss_history]
    losses = [entry['loss'] for entry in loss_history]
    alphas = [entry['alpha'] for entry in loss_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Loss chart with phase transition
    ax1.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
    ax1.axvline(x=data['phase_transition_analysis']['cliff_point'],
                color='r', linestyle='--', alpha=0.7,
                label=f'Phase Transition (Step {data["phase_transition_analysis"]["cliff_point"]})')
    ax1.fill_betweenx([min(losses), max(losses)],
                      data['phase_transition_analysis']['cliff_point'],
                      max(steps), alpha=0.1, color='red',
                      label='Post-Transition Region')

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('SO(8) Alpha Gate Training: Loss Curve with Phase Transition',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Alpha value progression
    ax2.plot(steps, alphas, 'g-', linewidth=2, label='Alpha Gate Value')
    ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7,
                label='Critical Threshold (α=0.5)')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Alpha Gate Value')
    ax2.set_title('Alpha Gate Annealing Schedule',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[CHART] Alpha Gate training chart saved: {output_path}")

def create_combined_analysis_chart(data: dict, output_path: Path):
    """Create comprehensive analysis chart"""
    loss_history = data['loss_history']

    steps = np.array([entry['step'] for entry in loss_history])
    losses = np.array([entry['loss'] for entry in loss_history])
    alphas = np.array([entry['alpha'] for entry in loss_history])

    # Calculate moving averages (smaller window for demonstration data)
    window_size = 10  # Reduced window size
    if len(losses) > window_size:
        loss_ma = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        alpha_ma = np.convolve(alphas, np.ones(window_size)/window_size, mode='valid')
        steps_ma = steps[window_size-1:]
    else:
        # Skip moving average if data is too small
        loss_ma = losses
        alpha_ma = alphas
        steps_ma = steps

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Loss curve with moving average
    ax1.plot(steps, losses, 'b-', alpha=0.5, label='Raw Loss')
    ax1.plot(steps_ma, loss_ma, 'b-', linewidth=2, label='Moving Average (50 steps)')
    ax1.axvline(x=data['phase_transition_analysis']['cliff_point'],
                color='r', linestyle='--', alpha=0.7, label='Phase Transition')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve Analysis', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Alpha progression with sigmoid fit
    ax2.plot(steps, alphas, 'g-', linewidth=2, label='Alpha Values')
    # Fit sigmoid curve
    from scipy.optimize import curve_fit
    def sigmoid_func(x, a, b, c, d):
        return a / (1 + np.exp(-b * (x - c))) + d

    try:
        popt, _ = curve_fit(sigmoid_func, steps, alphas, p0=[0.8, 0.01, 500, 0.1])
        fitted_alpha = sigmoid_func(steps, *popt)
        ax2.plot(steps, fitted_alpha, 'r--', alpha=0.7, label='Sigmoid Fit')
    except:
        pass

    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Alpha Gate Value')
    ax2.set_title('Alpha Gate Annealing', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Loss gradient (rate of change)
    loss_gradient = np.gradient(losses, steps)
    ax3.plot(steps, loss_gradient, 'purple', linewidth=2, label='Loss Gradient')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axvline(x=data['phase_transition_analysis']['cliff_point'],
                color='r', linestyle='--', alpha=0.7, label='Phase Transition')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Loss Gradient (dLoss/dStep)')
    ax3.set_title('Loss Convergence Analysis', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Phase transition zoom-in
    cliff_point = data['phase_transition_analysis']['cliff_point']
    zoom_start = max(0, cliff_point - 100)
    zoom_end = min(len(steps), cliff_point + 100)

    ax4.plot(steps[zoom_start:zoom_end], losses[zoom_start:zoom_end],
             'b-', linewidth=2, label='Loss around Transition')
    ax4.axvline(x=cliff_point, color='r', linestyle='--', alpha=0.7,
                label='Phase Transition Point')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Loss')
    ax4.set_title('Phase Transition Zoom (Steps 200-400)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('SO(8) Alpha Gate Training Analysis: Physical Phase Transition Demonstration',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[CHART] Comprehensive analysis chart saved: {output_path}")

def main():
    """Generate alpha gate training charts"""
    log_path = Path("logs/alpha_gate_training_log.json")
    charts_dir = Path("logs/charts")
    charts_dir.mkdir(exist_ok=True)

    if not log_path.exists():
        print(f"[ERROR] Training log not found: {log_path}")
        return

    print("[CHART] Loading alpha gate training log...")
    data = load_training_log(log_path)

    # Create loss chart
    loss_chart_path = charts_dir / "alpha_gate_loss_curve.png"
    create_loss_chart(data, loss_chart_path)

    # Create comprehensive analysis chart
    analysis_chart_path = charts_dir / "alpha_gate_comprehensive_analysis.png"
    create_combined_analysis_chart(data, analysis_chart_path)

    # Save summary text
    summary_path = charts_dir / "training_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("SO(8) Alpha Gate Training Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {data['training_metadata']['model']}\n")
        f.write(f"Training Duration: {data['training_metadata']['total_steps']} steps\n")
        f.write(f"Alpha Range: {data['training_metadata']['alpha_initial']} → {data['training_metadata']['alpha_final']}\n")
        f.write(f"Final Loss: {data['final_metrics']['final_loss']:.4f}\n")
        f.write(f"Phase Transition: Step {data['phase_transition_analysis']['cliff_point']}\n")
        f.write(f"Transition Ratio: {data['phase_transition_analysis']['transition_ratio']:.2f}\n\n")
        f.write("Physical Interpretation:\n")
        f.write("- Alpha Gate represents geometric constraint strength\n")
        f.write("- Phase transition occurs when constraint becomes dominant\n")
        f.write("- Loss drop indicates successful SO(8) structure integration\n")

    print(f"[SUMMARY] Training summary saved: {summary_path}")
    print("[SUCCESS] All alpha gate training charts generated!")

if __name__ == "__main__":
    main()
