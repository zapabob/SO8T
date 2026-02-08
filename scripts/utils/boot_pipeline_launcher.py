#!/usr/bin/env python3
"""Boot-time wrapper that keeps the moonshot retraining pipeline alive with checkpointing and progress logs.

This launcher is designed for Python 3.12+ (UV node compatibility) and emulates a simplified tqdm-style
progress bar while the main pipeline runs. It also maintains a rolling stack of three checkpoints every five minutes.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from io import StringIO
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = PROJECT_ROOT / "run_moonshot_pipeline_2025_2026.py"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "boot_pipeline_launcher.log"
CHECKPOINT_SOURCE = PROJECT_ROOT / "checkpoints" / "latest_checkpoint.json"
ROLLING_DIR = PROJECT_ROOT / "checkpoints" / "rolling_snapshots"
ROLLING_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_INTERVAL_SECONDS = 300
MAX_ROLLING_CHECKPOINTS = 3


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("boot_launcher")
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class RollingCheckpointManager(threading.Thread):
    def __init__(self, interval: int, keep: int, logger: logging.Logger) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.keep = keep
        self.logger = logger
        self._stop_event = threading.Event()

    def run(self) -> None:
        self.logger.info("RollingCheckpointManager started, every %s seconds, keep %s snapshots.", self.interval, self.keep)
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
            self.logger.info("Captured rolling checkpoint %s", dest.name)
            self._trim()
        except Exception as exc:
            self.logger.error("Failed to capture rolling checkpoint: %s", exc)

    def _trim(self) -> None:
        snapshots = sorted(ROLLING_DIR.glob("rolling_checkpoint_*.json"), key=lambda p: p.stat().st_mtime)
        while len(snapshots) > self.keep:
            stale = snapshots.pop(0)
            try:
                stale.unlink()
                self.logger.info("Removed stale rolling checkpoint %s", stale.name)
            except Exception as exc:
                self.logger.warning("Failed to remove stale checkpoint %s: %s", stale.name, exc)


class ProgressReporter(threading.Thread):
    BAR_LENGTH = 10

    def __init__(self, logger: logging.Logger, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.logger = logger
        self.interval = interval
        self._stop_event = threading.Event()
        self._state = 0

    def run(self) -> None:
        self.logger.info("ProgressReporter started (simple English, tqdm-style).")
        while not self._stop_event.is_set():
            bar = self._render_bar(self._state)
            self.logger.info("progress: %s pipeline running", bar)
            self._state = (self._state + 1) % (self.BAR_LENGTH + 1)
            if self._stop_event.wait(self.interval):
                return

    def stop(self) -> None:
        self._stop_event.set()

    def _render_bar(self, filled: int) -> str:
        filled = min(self.BAR_LENGTH, max(0, filled))
        buffer = StringIO()
        bar = tqdm(total=self.BAR_LENGTH, file=buffer, bar_format="|{bar}|", ncols=40, leave=False)
        bar.n = filled
        bar.refresh()
        bar.close()
        text = buffer.getvalue().strip()
        buffer.close()
        clean = text.replace("\r", "").replace("\n", "")
        return clean or "[----------]"


class PipelineRunner:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def run(self) -> int:
        if not PIPELINE_SCRIPT.exists():
            self.logger.error("Pipeline script missing: %s", PIPELINE_SCRIPT)
            return 1
        cmd = [sys.executable, str(PIPELINE_SCRIPT), "--use-existing-datasets"]
        self.logger.info("Launching pipeline: %s", cmd)
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        return_code = 0
        try:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                clean_line = line.strip()
                if clean_line:
                    self.logger.info("pipeline output: %s", clean_line)
                if process.poll() is not None and not line:
                    break
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
        return return_code


def main() -> None:
    logger = setup_logger()
    checkpoint_manager = RollingCheckpointManager(CHECKPOINT_INTERVAL_SECONDS, MAX_ROLLING_CHECKPOINTS, logger)
    progress_reporter = ProgressReporter(logger)
    runner = PipelineRunner(logger)

    checkpoint_manager.start()
    progress_reporter.start()

    try:
        return_code = runner.run()
        if return_code != 0:
            logger.error("Pipeline exited with code %s", return_code)
        else:
            logger.info("Pipeline finished cleanly.")
    except Exception as exc:
        logger.exception("Unhandled exception in boot launcher: %s", exc)
    finally:
        checkpoint_manager.stop()
        progress_reporter.stop()
        checkpoint_manager.join(timeout=5)
        progress_reporter.join(timeout=5)


if __name__ == "__main__":
    main()
