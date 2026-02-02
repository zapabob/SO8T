#!/usr/bin/env python3
"""Pipeline Monitor - Real-time monitoring with SQLite-based progress tracking.

Displays pipeline status, system resources, and logs. Integrates with
pipeline_progress_store.py for persistent progress tracking and resumability.
"""

import time
import json
import os
import psutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = PROJECT_ROOT / "integrated_moonshot_pipeline_2025_2026.log"
INNER_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "latest_checkpoint.json"
OUTER_CHECKPOINT_DIR = PROJECT_ROOT / "data" / "collected_2025_2026"

try:
    from .pipeline_progress_store import (
        get_run_status,
        get_latest_checkpoint,
        get_latest_rolling_checkpoint,
        get_all_runs,
        init_db,
        SQL_STORE_AVAILABLE,
    )
except ImportError:
    SQL_STORE_AVAILABLE = False


def get_latest_outer_checkpoint():
    """Get the latest outer pipeline checkpoint."""
    if not OUTER_CHECKPOINT_DIR.exists():
        return None

    checkpoints = list(OUTER_CHECKPOINT_DIR.glob("pipeline_checkpoint_*.json"))
    if not checkpoints:
        return None

    try:
        latest = max(checkpoints, key=os.path.getmtime)
        with open(latest, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def get_inner_checkpoint():
    """Get the latest inner pipeline checkpoint."""
    if not INNER_CHECKPOINT.exists():
        return None
    try:
        with open(INNER_CHECKPOINT, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def get_log_tail(n=10):
    """Get the last n lines of the log file."""
    if not LOG_FILE.exists():
        return ["Log file not found."]

    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            return [l.strip() for l in lines[-n:]]
    except Exception as e:
        return [f"Error reading log: {e}"]


def make_layout():
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=10),
    )
    layout["main"].split_row(
        Layout(name="status", ratio=1), Layout(name="system", ratio=1)
    )
    return layout


def generate_status_panel():
    outer_cp = get_latest_outer_checkpoint()
    inner_cp = get_inner_checkpoint()

    sql_status = ""
    sql_checkpoint = None
    if SQL_STORE_AVAILABLE:
        try:
            runs = get_all_runs()
            if runs:
                latest_run = runs[0]
                run_id = latest_run.get("run_id", "N/A")
                status = latest_run.get("status", "N/A")
                sql_status = f"[bold yellow]{status}[/]"
                sql_checkpoint = get_latest_rolling_checkpoint(run_id)
        except Exception:
            sql_status = "[red]DB Error[/]"

    table = Table(show_header=False, expand=True, box=None)

    phase = "Initializing..."
    timestamp = "N/A"
    details = ""

    if sql_checkpoint:
        phase = f"Rolling: {sql_checkpoint.get('phase', 'Unknown')}"
        timestamp = sql_checkpoint.get("timestamp", "N/A")
        metrics = sql_checkpoint.get("metrics", {})
        if metrics:
            details = f"Metrics: {json.dumps(metrics, indent=2)[:100]}"
    elif inner_cp:
        phase = f"Inner: {inner_cp.get('current_phase', 'Unknown')}"
        timestamp = inner_cp.get("timestamp", "N/A")
        model_state = inner_cp.get("model_state", {})
        details = f"Model Phase: {model_state.get('phase', 'N/A')}\nResume Attempts: {inner_cp.get('resume_attempt_count', 0)}"
    elif outer_cp:
        phase = f"Outer: {outer_cp.get('phase', 'Unknown')}"
        timestamp = outer_cp.get("timestamp", "N/A")
        data = outer_cp.get("data", {})
        details = f"Data keys: {', '.join(data.keys())}"

    table.add_row("Phase", f"[bold cyan]{phase}[/]")
    table.add_row("Last Update", f"[green]{timestamp}[/]")
    table.add_row("Details", details)
    if SQL_STORE_AVAILABLE:
        table.add_row("SQL Status", sql_status)

    return Panel(table, title="Pipeline Status", border_style="blue")


def generate_system_panel():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage(str(PROJECT_ROOT)).percent

    table = Table(show_header=False, expand=True, box=None)
    table.add_row("CPU Usage", f"{cpu}%")
    table.add_row("Memory Usage", f"{mem}%")
    table.add_row("Disk Usage", f"{disk}%")

    return Panel(table, title="System Stats", border_style="red")


def generate_log_panel():
    logs = get_log_tail(8)
    text = Text("\n".join(logs), style="dim white")
    return Panel(text, title="Log Output", border_style="grey70")


def main():
    layout = make_layout()
    layout["header"].update(
        Panel(
            Text(
                "Moonshot Pipeline Monitor (SQL-Integrated)",
                justify="center",
                style="bold magenta",
            )
        )
    )

    if SQL_STORE_AVAILABLE:
        try:
            init_db()
        except Exception as e:
            print(f"Warning: Could not initialize SQL store: {e}")

    with Live(layout, refresh_per_second=1):
        while True:
            layout["status"].update(generate_status_panel())
            layout["system"].update(generate_system_panel())
            layout["footer"].update(generate_log_panel())
            time.sleep(1)


if __name__ == "__main__":
    main()
