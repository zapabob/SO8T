#!/usr/bin/env python3
"""
Simple test script to verify tqdm and logging functionality
"""

import time
import logging
import sys
import os
from datetime import datetime
from tqdm import tqdm
import psutil
import GPUtil

# Set console encoding to UTF-8 for Windows compatibility
if os.name == 'nt':
    import codecs
    try:
        codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp65001' else None)
    except:
        pass

# Setup logging
log_filename = f"test_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='w')
    ]
)
logger = logging.getLogger(__name__)

class ProgressMonitor:
    """Progress monitoring with tqdm and system metrics"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

        # Setup tqdm progress bar (simple format for Windows compatibility)
        self.pbar = tqdm(
            total=total_steps,
            desc="Test Progress",
            unit="step",
            ncols=80,
            ascii=True,
            bar_format='{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed} | {rate_fmt} | {postfix}'
        )

    def update(self, step: int, loss: float = None):
        """Update progress bar"""
        self.current_step = step

        # Get system metrics
        gpu_info = self._get_gpu_info()
        memory_info = self._get_memory_info()

        # Update progress bar
        postfix = {
            'loss': '.4f' if loss is not None else 'N/A',
            'gpu': f"{gpu_info['utilization']:.1f}%",
            'mem': f"{gpu_info['memory_used']:.0f}MB",
            'temp': f"{gpu_info['temperature']:.0f}°C",
            'cpu': f"{memory_info['cpu_percent']:.1f}%"
        }

        self.pbar.set_postfix(postfix)
        self.pbar.update(1)

        # Log every 10 steps
        if step % 10 == 0:
            logger.info(f"Step {step}/{self.total_steps} | Loss: {loss:.4f} | "
                       f"GPU: {gpu_info['utilization']:.1f}% | Memory: {gpu_info['memory_used']:.0f}MB | "
                       f"Temperature: {gpu_info['temperature']:.0f}°C")

    def _get_gpu_info(self):
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'temperature': gpu.temperature
                }
        except Exception as e:
            logger.warning(f"GPU info error: {e}")
        return {'utilization': 0, 'memory_used': 0, 'temperature': 0}

    def _get_memory_info(self):
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': memory.percent
        }

    def close(self):
        """Close progress bar"""
        self.pbar.close()

def simulate_training():
    """Simulate training with progress monitoring"""
    logger.info("Starting progress monitor test")
    logger.info(f"Log file: {log_filename}")

    monitor = ProgressMonitor(total_steps=50)

    for step in range(1, 51):
        # Simulate training step
        loss = 2.0 * (0.99 ** step) + 0.1 * (0.5 - time.time() % 1)  # Decreasing loss with noise
        time.sleep(0.5)  # Simulate computation time

        monitor.update(step, loss)

        # Simulate occasional longer steps
        if step % 15 == 0:
            time.sleep(1.0)

    monitor.close()
    logger.info("Progress monitor test completed!")

if __name__ == "__main__":
    try:
        simulate_training()
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


