#!/usr/bin/env python3
"""
Boot-time wrapper for Moonshot Pipeline v3.0 with power failure protection.

Features:
- Rolling checkpoints every 3 minutes (5 snapshots kept)
- Automatic resume from last checkpoint on power failure
- Power-on auto-start via Windows Task Scheduler
- SQL-based progress tracking
- Tqdm-style progress reporting (simple English)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import uuid
import os
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = PROJECT_ROOT / "run_moonshot_full_pipeline.py"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "boot_pipeline_launcher.log"
CHECKPOINT_SOURCE = PROJECT_ROOT / "checkpoints" / "latest_checkpoint.json"
ROLLING_DIR = PROJECT_ROOT / "checkpoints" / "rolling_snapshots"
ROLLING_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_INTERVAL_SECONDS = 180
MAX_ROLLING_CHECKPOINTS = 5

try:
    from .pipeline_progress_store import (
        init_db,
        record_run,
        record_checkpoint,
        log_progress,
        complete_run,
        fail_run,
        get_run_status,
        get_latest_rolling_checkpoint_any,
        get_current_run_id,
    )

    SQL_STORE_AVAILABLE = True
except ImportError:
    SQL_STORE_AVAILABLE = False


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("boot_launcher")
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def show_status() -> None:
    """Print startup + pipeline status."""
    print("=" * 60)
    print("Moonshot Pipeline v3.0 - Boot Launcher Status")
    print("=" * 60)

    # Task Scheduler status
    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-ScheduledTask -TaskName 'MoonshotPipelineV3*' 2>$null | Select-Object -ExpandProperty TaskName",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print("Task Scheduler:", "OK" if result.stdout.strip() else "Not set")
    except Exception:
        print("Task Scheduler: Unknown")

    # Last run status
    status_path = LOG_DIR / "last_run_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            print("Last Run:", status.get("timestamp", "N/A"))
            print("Completed:", status.get("completed", False))
            print("Phase:", status.get("phase", "N/A"))
        except Exception:
            print("Last Run: <failed to read>")
    else:
        print("Last Run: Not found")

    # Rolling checkpoint
    latest = None
    if SQL_STORE_AVAILABLE:
        try:
            latest = get_latest_rolling_checkpoint_any()
        except Exception:
            latest = None
    if latest:
        print("Rolling Checkpoint:", latest.get("checkpoint_path"))
    else:
        snapshots = sorted(ROLLING_DIR.glob("rolling_checkpoint_*.json"))
        print("Rolling Checkpoint:", snapshots[-1].name if snapshots else "None")


class RollingCheckpointManager(threading.Thread):
    def __init__(
        self, interval: int, keep: int, logger: logging.Logger, run_id: str
    ) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.keep = keep
        self.logger = logger
        self.run_id = run_id
        self._stop_event = threading.Event()

    def run(self) -> None:
        self.logger.info(
            "RollingCheckpointManager: interval=%ds, keep=%d snapshots",
            self.interval,
            self.keep,
        )
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval):
                return
            self.capture()

    def stop(self) -> None:
        self._stop_event.set()

    def capture(self) -> None:
        if not CHECKPOINT_SOURCE.exists():
            self.logger.debug("No source checkpoint at %s", CHECKPOINT_SOURCE)
            return

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dest = ROLLING_DIR / f"rolling_checkpoint_{timestamp}.json"

        try:
            shutil.copy2(CHECKPOINT_SOURCE, dest)
            self.logger.info("[CHECKPOINT] Captured: %s", dest.name)
            self._trim()

            if SQL_STORE_AVAILABLE:
                record_checkpoint(
                    self.run_id,
                    "rolling_snapshot",
                    step=None,
                    checkpoint_path=str(dest),
                    is_rolling=True,
                )
        except Exception as exc:
            self.logger.error("Failed to capture rolling checkpoint: %s", exc)

    def _trim(self) -> None:
        snapshots = sorted(
            ROLLING_DIR.glob("rolling_checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(snapshots) > self.keep:
            stale = snapshots.pop(0)
            try:
                stale.unlink()
                self.logger.info("[CLEANUP] Removed stale: %s", stale.name)
            except Exception as exc:
                self.logger.warning("Failed to remove stale checkpoint: %s", exc)


class ProgressReporter(threading.Thread):
    BAR_LENGTH = 10

    def __init__(self, logger: logging.Logger, interval: float = 60.0) -> None:
        super().__init__(daemon=True)
        self.logger = logger
        self.interval = interval
        self._stop_event = threading.Event()
        self._state = 0

    def run(self) -> None:
        self.logger.info(
            "ProgressReporter: simple English tqdm-style every %ds", self.interval
        )
        while not self._stop_event.is_set():
            bar = self._render_bar(self._state)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.logger.info("[PROGRESS] %s | %s | Pipeline running", bar, timestamp)
            self._state = (self._state + 1) % (self.BAR_LENGTH + 1)
            if self._stop_event.wait(self.interval):
                return

    def stop(self) -> None:
        self._stop_event.set()

    def _render_bar(self, filled: int) -> str:
        filled = min(self.BAR_LENGTH, max(0, filled))
        buffer = StringIO()
        bar = tqdm(
            total=self.BAR_LENGTH,
            file=buffer,
            bar_format="|{bar}|",
            ncols=40,
            leave=False,
        )
        bar.n = filled
        bar.refresh()
        bar.close()
        text = buffer.getvalue().strip()
        buffer.close()
        clean = text.replace("\r", "").replace("\n", "")
        return clean or "[----------]"


class PowerFailureRecovery:
    """Handles resume from power failure."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.resume_file = PROJECT_ROOT / "logs" / "last_run_status.json"

    def get_last_status(self) -> Optional[Dict[str, Any]]:
        """Get the last run status for recovery."""
        if self.resume_file.exists():
            try:
                with open(self.resume_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning("Failed to read resume status: %s", e)
        return None

    def save_status(
        self, run_id: str, phase: str, checkpoint_path: Optional[str] = None
    ) -> None:
        """Save current run status for power failure recovery."""
        status = {
            "run_id": run_id,
            "phase": phase,
            "checkpoint_path": checkpoint_path,
            "last_update": datetime.now().isoformat(),
            "python_pid": os.getpid(),
        }
        try:
            with open(self.resume_file, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning("Failed to save resume status: %s", e)

    def get_latest_rolling_checkpoint(self) -> Optional[Path]:
        """Get the latest rolling checkpoint for resume."""
        if SQL_STORE_AVAILABLE:
            cp = get_latest_rolling_checkpoint_any()
            if cp and cp.get("checkpoint_path"):
                return Path(cp["checkpoint_path"])
        return None

    def check_and_restore(self) -> bool:
        """Check if recovery is needed and restore if possible."""
        status = self.get_last_status()
        if not status:
            self.logger.info("No previous run found - starting fresh")
            return False

        last_update = status.get("last_update", "")
        phase = status.get("phase", "unknown")

        self.logger.info(
            "[RECOVERY] Previous run detected: phase=%s, last_update=%s",
            phase,
            last_update,
        )

        rolling_cp = self.get_latest_rolling_checkpoint()
        if rolling_cp and rolling_cp.exists():
            self.logger.info("[RECOVERY] Found rolling checkpoint: %s", rolling_cp.name)

            restore_dest = CHECKPOINT_SOURCE
            shutil.copy2(rolling_cp, restore_dest)
            self.logger.info(
                "[RECOVERY] Restored checkpoint from: %s", restore_dest.name
            )
            return True

        return False


class PipelineRunner:
    def __init__(self, logger: logging.Logger, run_id: str) -> None:
        self.logger = logger
        self.run_id = run_id
        self.recovery = PowerFailureRecovery(logger)

    def run(self) -> int:
        if not PIPELINE_SCRIPT.exists():
            self.logger.error("[ERROR] Pipeline script missing: %s", PIPELINE_SCRIPT)
            if SQL_STORE_AVAILABLE:
                log_progress(
                    self.run_id, "error", f"Pipeline script missing: {PIPELINE_SCRIPT}"
                )
                fail_run(self.run_id, "Pipeline script missing")
            return 1

        self.logger.info("[PIPELINE] Checking for power failure recovery...")
        restored = self.recovery.check_and_restore()
        if restored:
            self.logger.info("[PIPELINE] Resumed from checkpoint - continuing training")
            if SQL_STORE_AVAILABLE:
                log_progress(self.run_id, "resumed", "Resumed from rolling checkpoint")
        else:
            self.logger.info("[PIPELINE] Starting fresh - no recovery needed")

        cmd = [sys.executable, str(PIPELINE_SCRIPT), "--use-existing-datasets"]
        self.logger.info("[PIPELINE] Launching: %s", " ".join(cmd))

        if SQL_STORE_AVAILABLE:
            log_progress(self.run_id, "launch", f"Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self.recovery.save_status(self.run_id, "running")

        return_code = 0
        try:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                clean_line = line.strip()
                if clean_line:
                    self.logger.info("[PIPELINE] %s", clean_line)
                if process.poll() is not None and not line:
                    break
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
            self.recovery.save_status(
                self.run_id, "completed" if return_code == 0 else "failed"
            )

        return return_code


def setup_windows_startup() -> bool:
    """Set up Windows Task Scheduler for auto-start on power-on."""
    script_path = Path(__file__).resolve()
    python_exe = sys.executable
    task_name = "MoonshotPipelineV3_AutoResume"
    reuse_task_name = "SO8T-AutoResume"
    reuse_script = PROJECT_ROOT / "scripts" / "utils" / "auto_resume_startup.bat"

    # Prefer reusing existing SO8T auto-resume task if present
    try:
        existing = subprocess.run(
            [
                "powershell",
                "-Command",
                f"Get-ScheduledTask -TaskName '{reuse_task_name}' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty TaskName",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if existing.stdout.strip():
            print(f"[STARTUP] Reusing existing task: {reuse_task_name}")
            return True
    except Exception:
        pass

    if reuse_script.exists():
        powershell_script = f'''
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c \\"{reuse_script}\\"" -WorkingDirectory "{PROJECT_ROOT}"
$Trigger = New-ScheduledTaskTrigger -AtStartup -RandomDelay "00:01:00"
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries $true -DontStopIfGoingOnBatteries $true -RunOnlyIfNetworkAvailable $true
$Principal = New-ScheduledTaskPrincipal -UserId "NT AUTHORITY\\SYSTEM" -RunLevel "Highest"

try {{
    Get-ScheduledTask -TaskName "{reuse_task_name}" -ErrorAction Stop | Unregister-ScheduledTask -Confirm:$false -ErrorAction Stop
    Start-Sleep -Seconds 2
}} catch {{ }}

Register-ScheduledTask -TaskName "{reuse_task_name}" -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force
Write-Host "Task \\"{reuse_task_name}\\" registered successfully"
'''
    else:
        powershell_script = f'''
$Action = New-ScheduledTaskAction -Execute "{python_exe}" -Argument "-FullPath \\"{script_path}\\" --use-existing-datasets" -WorkingDirectory "{PROJECT_ROOT}"
$Trigger = New-ScheduledTaskTrigger -AtStartup -RandomDelay "00:01:00"
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries $true -DontStopIfGoingOnBatteries $true -RunOnlyIfNetworkAvailable $true
$Principal = New-ScheduledTaskPrincipal -UserId "NT AUTHORITY\\SYSTEM" -RunLevel "Highest"

try {{
    Get-ScheduledTask -TaskName "{task_name}" -ErrorAction Stop | Unregister-ScheduledTask -Confirm:$false -ErrorAction Stop
    Start-Sleep -Seconds 2
}} catch {{ }}

Register-ScheduledTask -TaskName "{task_name}" -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force
Write-Host "Task \\"{task_name}\\" registered successfully"
'''

    try:
        result = subprocess.run(
            ["powershell", "-Command", powershell_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("[STARTUP] Windows Task Scheduler: OK")
            if reuse_script.exists():
                print(f"[STARTUP] Task Name: {reuse_task_name}")
            else:
                print(f"[STARTUP] Task Name: {task_name}")
            print(f"[STARTUP] Trigger: System startup (1 min delay)")
            return True
        else:
            print(f"[STARTUP] Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[STARTUP] Error: {e}")
        return False


def remove_windows_startup() -> bool:
    """Remove Windows Task Scheduler entry."""
    task_name = "MoonshotPipelineV3_AutoResume"

    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                f'Get-ScheduledTask -TaskName "{task_name}" -ErrorAction Stop | Unregister-ScheduledTask -Confirm:$false',
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(f"[STARTUP] Removed task: {task_name}")
        return True
    except Exception as e:
        print(f"[STARTUP] Remove failed (may not exist): {e}")
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Moonshot Pipeline v3.0 Boot Launcher")
    parser.add_argument(
        "--setup-startup", action="store_true", help="Setup Windows auto-start"
    )
    parser.add_argument(
        "--remove-startup", action="store_true", help="Remove Windows auto-start"
    )
    parser.add_argument("--status", action="store_true", help="Check startup status")
    parser.add_argument("--use-existing-datasets", action="store_true", default=True)

    args = parser.parse_args()

    if args.setup_startup:
        setup_windows_startup()
        return

    if args.remove_startup:
        remove_windows_startup()
        return

    if args.status:
        show_status()
        return

    logger = setup_logger()

    logger.info("=" * 60)
    logger.info("Moonshot Pipeline v3.0 - Boot Launcher")
    logger.info("=" * 60)
    logger.info("[CONFIG] Checkpoint interval: %ds", CHECKPOINT_INTERVAL_SECONDS)
    logger.info("[CONFIG] Max rolling checkpoints: %d", MAX_ROLLING_CHECKPOINTS)
    logger.info("[CONFIG] Power failure recovery: enabled")
    logger.info("=" * 60)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    if SQL_STORE_AVAILABLE:
        init_db()
        record_run(run_id, git_commit_hash=_get_git_commit())
        logger.info("[SQL] Progress tracking initialized: %s", run_id)
    else:
        logger.warning("[SQL] Not available - using file logging only")

    checkpoint_manager = RollingCheckpointManager(
        CHECKPOINT_INTERVAL_SECONDS, MAX_ROLLING_CHECKPOINTS, logger, run_id
    )
    progress_reporter = ProgressReporter(logger)
    runner = PipelineRunner(logger, run_id)

    checkpoint_manager.start()
    progress_reporter.start()

    try:
        if SQL_STORE_AVAILABLE:
            log_progress(run_id, "started", "Boot launcher started")

        return_code = runner.run()

        if return_code != 0:
            logger.error("[EXIT] Pipeline exited with code: %s", return_code)
            if SQL_STORE_AVAILABLE:
                log_progress(run_id, "error", f"Exit code: {return_code}")
                fail_run(run_id, f"Exit code: {return_code}")
        else:
            logger.info("[EXIT] Pipeline finished cleanly")
            if SQL_STORE_AVAILABLE:
                log_progress(run_id, "completed", "Pipeline finished")
                complete_run(run_id)

    except Exception as exc:
        logger.exception("[ERROR] Unhandled exception: %s", exc)
        if SQL_STORE_AVAILABLE:
            log_progress(run_id, "error", str(exc))
            fail_run(run_id, str(exc))
    finally:
        checkpoint_manager.stop()
        progress_reporter.stop()
        checkpoint_manager.join(timeout=5)
        progress_reporter.join(timeout=5)


def _get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()[:40] if result.stdout else "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
