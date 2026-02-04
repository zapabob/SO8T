#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Pipeline Monitor - Power-on Auto Start System
電源投入時に自動起動し、パイプラインを監視してエラーまたはHF完了時に停止・通知
"""

import os
import sys
import time
import logging
import subprocess
import threading
import signal
from pathlib import Path
from datetime import datetime
import json

class SO8TPipelineMonitor:
    """SO8T Pipeline Monitor for automatic power-on execution"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.log_dir = self.project_root / "logs"
        self.checkpoint_dir = Path("D:/webdataset/checkpoints/ppo_so8t")
        self.hf_model_dir = Path("D:/webdataset/models/final/so8t_ppo_final")

        # Create log directory
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Pipeline process
        self.pipeline_process = None
        self.monitoring_active = True

        # Exit codes
        self.EXIT_SUCCESS = 0      # HF model completed
        self.EXIT_ERROR = 1        # Error occurred
        self.EXIT_USER_STOP = 2    # User requested stop

    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"so8t_pipeline_monitor_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SO8T Pipeline Monitor started - Log: {log_file}")

    def start_pipeline(self):
        """Start the SO8T PPO training pipeline"""
        try:
            self.logger.info("Starting SO8T PPO training pipeline...")

            # Set environment
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{self.project_root};{self.project_root}/so8t-mmllm/src"
            env['ATTN_IMPLEMENTATION'] = 'eager'

            # Start pipeline process
            cmd = [
                sys.executable,
                str(self.project_root / "scripts/training/train_aegis_v2_ppo_so8t.py")
            ]

            self.pipeline_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                universal_newlines=True
            )

            self.logger.info(f"Pipeline process started with PID: {self.pipeline_process.pid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            return False

    def check_pipeline_status(self):
        """Check if pipeline process is still running"""
        if self.pipeline_process is None:
            return False

        return self.pipeline_process.poll() is None

    def check_for_errors(self, log_content):
        """Check log content for error patterns"""
        error_patterns = [
            "ERROR",
            "CRITICAL",
            "Exception:",
            "Traceback",
            "Failed to",
            "CUDA error",
            "Out of memory",
            "AssertionError",
            "ValueError",
            "RuntimeError",
            "ImportError"
        ]

        # Exclude known non-error messages
        exclude_patterns = [
            "oneDNN custom operations are on",  # TensorFlow warning
            "Unsloth not available",  # Expected fallback message
            "falling back to bitsandbytes",  # Expected fallback message
            "FutureWarning",  # Python warnings
            "NOTE: Redirects are currently not supported",  # PyTorch warning
            "tensorflow/core/util/port.cc",  # TensorFlow internal message
            "You may see slightly different numerical results due to floating-point round-off errors",  # TensorFlow numerical warning
            "TF_ENABLE_ONEDNN_OPTS=0",  # TensorFlow environment variable
            "pynvml package is deprecated",  # PyTorch warning
            "Will patch your computer to enable 2x faster free finetuning",  # Unsloth message
        ]

        # Check if content contains error patterns but not excluded patterns
        content_lower = log_content.lower()
        for pattern in error_patterns:
            if pattern.lower() in content_lower:
                # Check if this is an excluded message
                for exclude in exclude_patterns:
                    if exclude.lower() in content_lower:
                        return False  # This is an excluded warning, not a real error
                return True  # This is a real error
        return False

    def check_hf_completion(self, log_content):
        """Check if HF model has been completed and uploaded"""
        completion_patterns = [
            "Successfully uploaded to HuggingFace",
            "HF upload completed",
            "Model uploaded to HF",
            "HuggingFace upload successful",
            "Final model saved and uploaded"
        ]

        for pattern in completion_patterns:
            if pattern.lower() in log_content.lower():
                return True
        return False

    def monitor_pipeline(self):
        """Monitor the pipeline execution"""
        self.logger.info("Starting pipeline monitoring...")

        log_buffer = []
        error_detected = False
        hf_completed = False

        try:
            while self.monitoring_active and self.check_pipeline_status():
                # Read pipeline output
                if self.pipeline_process.stdout:
                    line = self.pipeline_process.stdout.readline()
                    if line:
                        print(line.strip())  # Real-time output
                        log_buffer.append(line.strip())

                        # Check for errors
                        if not error_detected and self.check_for_errors(line):
                            self.logger.warning(f"Error detected in pipeline output: {line.strip()}")
                            error_detected = True

                        # Check for HF completion
                        if not hf_completed and self.check_hf_completion(line):
                            self.logger.info(f"HF model completion detected: {line.strip()}")
                            hf_completed = True
                            break  # Stop monitoring on completion

                # Check every 5 seconds
                time.sleep(5)

            # Wait for process to finish
            if self.pipeline_process:
                self.pipeline_process.wait()

        except KeyboardInterrupt:
            self.logger.info("Monitor interrupted by user")
            return self.EXIT_USER_STOP
        except Exception as e:
            self.logger.error(f"Error during monitoring: {e}")
            return self.EXIT_ERROR

        # Determine exit reason
        if hf_completed:
            self.logger.info("Pipeline completed successfully - HF model uploaded")
            return self.EXIT_SUCCESS
        elif error_detected:
            self.logger.error("Pipeline stopped due to error detection")
            return self.EXIT_ERROR
        else:
            self.logger.info("Pipeline monitoring completed")
            return self.EXIT_SUCCESS

    def stop_pipeline(self):
        """Stop the pipeline process gracefully"""
        if self.pipeline_process and self.check_pipeline_status():
            self.logger.info("Stopping pipeline process...")

            try:
                # Try graceful termination first
                self.pipeline_process.terminate()
                time.sleep(10)  # Wait 10 seconds

                # Force kill if still running
                if self.check_pipeline_status():
                    self.logger.warning("Force killing pipeline process...")
                    self.pipeline_process.kill()

                self.logger.info("Pipeline process stopped")
            except Exception as e:
                self.logger.error(f"Error stopping pipeline: {e}")

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.logger.info(f"Received signal {signum}, stopping monitor...")
        self.monitoring_active = False
        self.stop_pipeline()

    def run(self):
        """Main execution method"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Start pipeline
            if not self.start_pipeline():
                self.logger.error("Failed to start pipeline")
                return self.EXIT_ERROR

            # Monitor pipeline
            result = self.monitor_pipeline()

            # Ensure pipeline is stopped
            self.stop_pipeline()

            return result

        except Exception as e:
            self.logger.error(f"Unexpected error in monitor: {e}")
            self.stop_pipeline()
            return self.EXIT_ERROR


def main():
    """Main entry point"""
    monitor = SO8TPipelineMonitor()
    exit_code = monitor.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
