import time
import json
import os
import psutil
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = PROJECT_ROOT / "integrated_moonshot_pipeline_2025_2026.log"
INNER_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "latest_checkpoint.json"
OUTER_CHECKPOINT_DIR = PROJECT_ROOT / "data" / "collected_2025_2026"

def get_latest_outer_checkpoint():
    """Get the latest outer pipeline checkpoint."""
    if not OUTER_CHECKPOINT_DIR.exists():
        return None
    
    checkpoints = list(OUTER_CHECKPOINT_DIR.glob("pipeline_checkpoint_*.json"))
    if not checkpoints:
        return None
    
    try:
        latest = max(checkpoints, key=os.path.getmtime)
        with open(latest, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def get_inner_checkpoint():
    """Get the latest inner pipeline checkpoint."""
    if not INNER_CHECKPOINT.exists():
        return None
    try:
        with open(INNER_CHECKPOINT, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def get_log_tail(n=10):
    """Get the last n lines of the log file."""
    if not LOG_FILE.exists():
        return ["Log file not found."]
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            # Simple tail implementation
            lines = f.readlines()
            return [l.strip() for l in lines[-n:]]
    except Exception as e:
        return [f"Error reading log: {e}"]

def make_layout():
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=10)
    )
    layout["main"].split_row(
        Layout(name="status", ratio=1),
        Layout(name="system", ratio=1)
    )
    return layout

def generate_status_panel():
    outer_cp = get_latest_outer_checkpoint()
    inner_cp = get_inner_checkpoint()
    
    table = Table(show_header=False, expand=True, box=None)
    
    phase = "Initializing..."
    timestamp = "N/A"
    details = ""
    
    if inner_cp:
        phase = f"Inner: {inner_cp.get('current_phase', 'Unknown')}"
        timestamp = inner_cp.get('timestamp', 'N/A')
        model_state = inner_cp.get('model_state', {})
        details = f"Model Phase: {model_state.get('phase', 'N/A')}\nResume Attempts: {inner_cp.get('resume_attempt_count', 0)}"
    elif outer_cp:
        phase = f"Outer: {outer_cp.get('phase', 'Unknown')}"
        timestamp = outer_cp.get('timestamp', 'N/A')
        data = outer_cp.get('data', {})
        details = f"Data keys: {', '.join(data.keys())}"
        
    table.add_row("Phase", f"[bold cyan]{phase}[/]")
    table.add_row("Last Update", f"[green]{timestamp}[/]")
    table.add_row("Details", details)
    
    return Panel(table, title="Pipeline Status", border_style="blue")

def generate_system_panel():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    
    table = Table(show_header=False, expand=True, box=None)
    table.add_row("CPU Usage", f"{cpu}%")
    table.add_row("Memory Usage", f"{mem}%")
    
    return Panel(table, title="System Stats", border_style="red")

def generate_log_panel():
    logs = get_log_tail(8)
    text = Text("\n".join(logs), style="dim white")
    return Panel(text, title="Log Output", border_style="grey70")

def main():
    layout = make_layout()
    layout["header"].update(Panel(Text("Moonshot Pipeline Monitor", justify="center", style="bold magenta")))
    
    with Live(layout, refresh_per_second=1):
        while True:
            layout["status"].update(generate_status_panel())
            layout["system"].update(generate_system_panel())
            layout["footer"].update(generate_log_panel())
            time.sleep(1)

if __name__ == "__main__":
    main()
