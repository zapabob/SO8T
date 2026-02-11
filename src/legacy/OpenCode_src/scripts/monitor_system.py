#!/usr/bin/env python3
"""
RTX 3060 System Monitor
System Monitoring Script
"""

import psutil
import GPUtil
import time
import json
from datetime import datetime
from pathlib import Path

class RTX3060Monitor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(exist_ok=True)

    def get_system_stats(self):
        """Get system statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_percent': psutil.virtual_memory().percent
            }
        }

        # GPU statistics (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats['gpu'] = {
                    'name': gpu.name,
                    'usage_percent': gpu.load * 100,
                    'memory_used_gb': gpu.memoryUsed / 1024,
                    'memory_total_gb': gpu.memoryTotal / 1024,
                    'memory_percent': gpu.memoryUtil * 100,
                    'temperature_c': gpu.temperature
                }
            else:
                stats['gpu'] = {'error': 'No GPU detected'}
        except:
            stats['gpu'] = {'error': 'GPU monitoring failed'}

        return stats

    def log_stats(self, stats):
        """Log statistics to file"""
        log_file = self.log_dir / f"system_monitor_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(stats, ensure_ascii=False) + '\n')

    def monitor_loop(self, interval=30, duration=None):
        """Continuous monitoring loop"""
        print("[MONITOR] RTX 3060 System Monitoring Started")
        print(f"Interval: {interval} seconds")
        if duration:
            print(f"Duration: {duration} seconds")
        print("=" * 50)

        start_time = time.time()
        count = 0

        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break

                stats = self.get_system_stats()
                self.log_stats(stats)

                # Console display
                cpu_usage = stats['cpu']['usage_percent']
                cpu_mem = stats['cpu']['memory_percent']
                gpu_info = stats.get('gpu', {})

                print(f"[{count:3d}] CPU: {cpu_usage:5.1f}% | RAM: {cpu_mem:5.1f}%", end='')

                if 'usage_percent' in gpu_info:
                    gpu_usage = gpu_info['usage_percent']
                    gpu_mem = gpu_info['memory_percent']
                    gpu_temp = gpu_info.get('temperature_c', 'N/A')
                    print(f" | GPU: {gpu_usage:5.1f}% | VRAM: {gpu_mem:5.1f}% | Temp: {gpu_temp}C")
                else:
                    print(" | GPU: N/A")

                count += 1
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[STOP] Monitoring stopped")

        print("=" * 50)
        print(f"Monitoring completed: {count} samples collected")

    def show_current_stats(self):
        """Show current statistics"""
        stats = self.get_system_stats()

        print("[STATS] Current System Statistics")
        print("=" * 30)

        cpu = stats['cpu']
        print("CPU:")
        print(".1f")
        print(".1f")
        print(".1f")

        gpu = stats.get('gpu', {})
        if 'usage_percent' in gpu:
            print("\nGPU:")
            print(".1f")
            print(".1f")
            print(".1f")
            print(".1f")
        else:
            print("\nGPU: Not detected")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='RTX 3060 System Monitor')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval (seconds)')
    parser.add_argument('--duration', type=int, help='Monitoring duration (seconds)')
    parser.add_argument('--current', action='store_true', help='Show current stats only')

    args = parser.parse_args()

    monitor = RTX3060Monitor()

    if args.current:
        monitor.show_current_stats()
    else:
        monitor.monitor_loop(interval=args.interval, duration=args.duration)

if __name__ == "__main__":
    main()
