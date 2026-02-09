from __future__ import annotations

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import sys


@dataclass
class ProgressConfig:
    log_file: str = "logs\\training.log"
    console_output: bool = True
    log_interval: int = 10
    tqdm_position: int = 0
    tqdm_leave: bool = False


class TrainingProgressTracker:
    def __init__(
        self,
        total_steps: int,
        desc: str = "Training",
        config: Optional[ProgressConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            from tqdm import tqdm

            self.tqdm_available = True
            self.pbar = tqdm(
                total=total_steps,
                desc=desc,
                position=config.tqdm_position if config else 0,
                leave=config.tqdm_leave if config else False,
            )
        except ImportError:
            self.tqdm_available = False
            self.pbar = None
        self.config = config or ProgressConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._setup_logging()
        self.current_step = 0
        self.total_steps = total_steps
        self.start_time = datetime.now()
        self.metrics_history: List[Dict[str, Any]] = []
        self.checkpoint_info: List[Dict[str, Any]] = []

    def _setup_logging(self) -> None:
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        console_handler = (
            logging.StreamHandler(sys.stdout)
            if self.config.console_output
            else logging.NullHandler()
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)

    def update(
        self,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.current_step = step if step > self.current_step else self.current_step + 1
        metrics = metrics or {}
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = (
            (elapsed / self.current_step) * (self.total_steps - self.current_step)
            if self.current_step > 0
            else 0
        )
        display_metrics = {
            **metrics,
            "step": self.current_step,
            "eta": f"{eta / 60:.1f}m",
        }
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix(display_metrics)
        if step % self.config.log_interval == 0:
            self.logger.info(f"Step {step}: {metrics}")
        if checkpoint_info:
            self.checkpoint_info.append(checkpoint_info)
            self.logger.info(f"Checkpoint saved: {checkpoint_info}")
        self.metrics_history.append(
            {
                "step": step,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        self.logger.info(f"Epoch {epoch} completed: {metrics}")

    def log_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        self.logger.info(f"Checkpoint saved: {checkpoint_info}")
        self.checkpoint_info.append(checkpoint_info)

    def log_eval(self, eval_name: str, metrics: Dict[str, float]) -> None:
        self.logger.info(f"Evaluation ({eval_name}): {metrics}")

    def log_hf_upload(self, repo_id: str, file_type: str, success: bool) -> None:
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"HF Upload {status}: repo={repo_id}, type={file_type}")

    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None) -> None:
        msg = f"ERROR: {error}"
        if context:
            msg += f", context={context}"
        self.logger.error(msg)

    def get_summary(self) -> Dict[str, Any]:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        avg_metrics = {}
        if self.metrics_history:
            for key in self.metrics_history[0]["metrics"].keys():
                values = [
                    m["metrics"].get(key, 0)
                    for m in self.metrics_history
                    if key in m["metrics"]
                ]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
        return {
            "total_steps": self.total_steps,
            "completed_steps": self.current_step,
            "progress_percent": (self.current_step / self.total_steps) * 100
            if self.total_steps > 0
            else 0,
            "elapsed_seconds": elapsed,
            "average_metrics": avg_metrics,
            "checkpoints_saved": len(self.checkpoint_info),
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
        }

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
        summary = self.get_summary()
        self.logger.info(f"Training completed: {summary}")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self) -> "TrainingProgressTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
