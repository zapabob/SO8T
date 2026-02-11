#!/usr/bin/env python3
"""
Real-time monitoring script for AEGIS Soul Injection training.

Monitors:
- Alpha Gate progression (chaos → golden ratio)
- Orthogonality loss (structural integrity)
- Phase transition status
- Training loss
"""

import os
import sys
import time
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

GOLDEN_RATIO = 1.6180339887

def monitor_checkpoints(checkpoint_dir="checkpoints_agiasi"):
    """Monitor AEGIS training checkpoints"""
    checkpoint_path = project_root / checkpoint_dir
    
    if not checkpoint_path.exists():
        console.print(f"[yellow]⚠️ Checkpoint directory not found: {checkpoint_path}[/yellow]")
        console.print("[cyan]Waiting for training to start...[/cyan]")
        return None
    
    # Find latest checkpoint
    checkpoints = sorted(checkpoint_path.glob("step_*"), key=lambda x: int(x.name.split("_")[1]))
    
    if not checkpoints:
        return None
    
    latest = checkpoints[-1]
    soul_file = latest / "soul.pt"
    
    if not soul_file.exists():
        return None
    
    # Load soul state
    soul = torch.load(soul_file, map_location="cpu")
    
    return {
        "step": soul["step"],
        "alpha": soul["alpha"].item(),
        "checkpoint_dir": str(latest)
    }

def get_phase_status(alpha):
    """Determine phase transition status"""
    diff = abs(alpha - GOLDEN_RATIO)
    
    if alpha < -2.0:
        return "🔵 Chaos (Liquid)", "blue"
    elif alpha < 0.0:
        return "🟡 Warming (Pre-Transition)", "yellow"
    elif diff > 0.5:
        return "🟠 Transitioning (Critical Point)", "orange"
    elif diff > 0.1:
        return "🟢 Approaching Golden Ratio", "green"
    else:
        return "✨ Golden Ratio Achieved!", "bright_green"

def create_status_table(data):
    """Create status display table"""
    if data is None:
        return Panel(
            "[cyan]👻 Waiting for AEGIS to awaken...[/cyan]\n"
            "[dim]No checkpoints found yet.[/dim]",
            title="AEGIS Soul Monitor",
            border_style="cyan"
        )
    
    alpha = data["alpha"]
    step = data["step"]
    status, color = get_phase_status(alpha)
    
    # Calculate progress to golden ratio
    start_alpha = -5.0
    progress = (alpha - start_alpha) / (GOLDEN_RATIO - start_alpha) * 100
    progress = min(100, max(0, progress))
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Training Step", f"[bold]{step}[/bold]")
    table.add_row("Alpha Gate", f"[bold yellow]{alpha:.6f}[/bold yellow]")
    table.add_row("Target (φ)", f"[dim]{GOLDEN_RATIO:.6f}[/dim]")
    table.add_row("Phase Status", f"[{color}]{status}[/{color}]")
    table.add_row("Progress", f"[{'green' if progress > 80 else 'yellow'}]{progress:.1f}%[/]")
    table.add_row("Checkpoint", f"[dim]{data['checkpoint_dir']}[/dim]")
    
    return Panel(
        table,
        title="[bold]👻 AEGIS Soul Monitor[/bold]",
        subtitle=f"[dim]Operation: Ghost in the Shell[/dim]",
        border_style=color
    )

def main():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]AEGIS Soul Injection Monitor[/bold cyan]\n"
        "[dim]Real-time monitoring of phase transition training[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                data = monitor_checkpoints()
                status_panel = create_status_table(data)
                live.update(status_panel)
                time.sleep(2)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Monitor stopped by user.[/yellow]")

if __name__ == "__main__":
    main()
