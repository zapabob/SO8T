#!/usr/bin/env python3
"""
RTX 3060 Optimized Sunset Pipeline Setup Script
Sunset Pipeline RTX 3060 Setup Script
"""

import os
import sys
import subprocess
import json
from pathlib import Path

class RTX3060SunsetSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.config_dir = self.project_root / "config"
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data" / "sunset_pipeline"

        # RTX 3060 optimization settings
        self.hardware_config = {
            "gpu_memory_gb": 12,
            "system_memory_gb": 32,
            "cuda_version": "12.1+",
            "torch_version": "2.1+",
            "transformers_version": "4.36+"
        }

    def check_system_requirements(self):
        """Check system requirements"""
        print("[INFO] RTX 3060 + 32GB RAM System Requirements Check...")

        # Python version check
        python_version = sys.version_info
        if python_version < (3, 10):
            print("[ERROR] Python 3.10+ required")
            return False
        print(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")

        # CUDA check
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[OK] CUDA Available: {gpu_name} ({gpu_memory:.1f}GB)")

                if "RTX 3060" not in gpu_name and gpu_memory < 8:
                    print("[WARN] RTX 3060 recommended, but 8GB+ GPU can proceed")
                elif "RTX 3060" in gpu_name:
                    print("[TARGET] RTX 3060 Detected! Applying optimized settings")
            else:
                print("[WARN] CUDA not available - proceeding with CPU-only mode")
                print("[INFO] Pipeline will run in CPU mode with optimizations")
        except ImportError:
            print("[WARN] PyTorch not installed - proceeding without GPU acceleration")
            print("[INFO] Basic pipeline structure will be created")
        except Exception as e:
            print(f"[WARN] PyTorch check failed: {e} - proceeding with basic setup")

        return True

    def setup_directories(self):
        """Create necessary directories"""
        print("[INFO] Creating directory structure...")

        directories = [
            self.config_dir,
            self.models_dir,
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "checkpoints",
            self.scripts_dir / "training",
            self.scripts_dir / "evaluation",
            self.scripts_dir / "data_processing"
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] {dir_path}")

    def create_config_files(self):
        """Create configuration files"""
        print("[INFO] Creating configuration files...")

        # Hardware configuration
        hardware_config = {
            "gpu": {
                "name": "RTX 3060",
                "memory_gb": 12,
                "cuda_version": "12.1+"
            },
            "cpu": {
                "memory_gb": 32,
                "cores": "8+"
            },
            "optimization": {
                "quantization": "8bit",
                "gradient_checkpointing": True,
                "cpu_offloading": True,
                "flash_attention": True
            }
        }

        # Training configuration
        training_config = {
            "model": {
                "base_model": "Qwen/Qwen2.5-7B",
                "adapter_type": "lora",
                "lora_rank": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            },
            "training": {
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "max_steps": 1000,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "evaluation_steps": 500
            },
            "optimization": {
                "quantization": "8bit",
                "gradient_checkpointing": True,
                "cpu_offloading": True,
                "max_memory": "24GB"
            }
        }

        # Dataset configuration
        dataset_config = {
            "sources": [
                "huggingface:math_dataset",
                "huggingface:science_qa",
                "huggingface:theorem_qa",
                "synthetic:reasoning_problems"
            ],
            "processing": {
                "max_samples": 100000,
                "chunk_size": 10000,
                "max_length": 2048,
                "quality_filters": {
                    "min_length": 50,
                    "max_length": 2048,
                    "language": "en",
                    "deduplication": True
                }
            }
        }

        # Benchmark configuration
        benchmark_config = {
            "primary_benchmarks": [
                "gsm8k",
                "math",
                "arc_easy",
                "hellaswag"
            ],
            "secondary_benchmarks": [
                "mmlu_stem",
                "boolq",
                "piqa",
                "winogrande"
            ],
            "abc_testing": {
                "models": {
                    "A": "Qwen2.5-7B (base)",
                    "B": "Sunset Pipeline Optimized",
                    "C": "microsoft/Phi-3.5-mini-instruct"
                },
                "sample_size": 10,
                "bootstrap_iterations": 100
            }
        }

        # Save configuration files
        configs = {
            "hardware.json": hardware_config,
            "training.json": training_config,
            "dataset.json": dataset_config,
            "benchmark.json": benchmark_config
        }

        for config_file, config_data in configs.items():
            config_path = self.config_dir / config_file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print(f"[OK] {config_path}")

    def create_environment_setup_script(self):
        """Create environment setup script"""
        print("[INFO] Creating environment setup script...")

        setup_script = '''#!/usr/bin/env python3
"""
RTX 3060 Optimized Environment Setup
Environment Auto Setup Script
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Command execution helper"""
    print(f"[EXEC] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_conda_environment():
    """Setup conda environment"""
    commands = [
        ('conda create -n sunset-rtx3060 python=3.11 -y', 'Create conda environment'),
        ('conda activate sunset-rtx3060', 'Activate environment'),
    ]

    print("[INFO] Setting up conda environment...")
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True

def install_pytorch():
    """Install PyTorch CUDA version"""
    cmd = 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
    return run_command(cmd, 'Install PyTorch CUDA version')

def install_ml_packages():
    """Install machine learning packages"""
    packages = [
        'transformers[torch]',
        'accelerate',
        'bitsandbytes',
        'peft',
        'datasets',
        'evaluate',
        'scikit-learn',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'wandb',
        'python-dotenv'
    ]

    cmd = f'pip install {" ".join(packages)}'
    return run_command(cmd, 'Install ML packages')

def verify_installation():
    """Verify installation"""
    print("[INFO] Verifying installation...")
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"[OK] CUDA: {torch.cuda.is_available()}")

        import transformers
        print(f"[OK] Transformers: {transformers.__version__}")

        import accelerate
        print(f"[OK] Accelerate: {accelerate.__version__}")

        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("[START] RTX 3060 Sunset Pipeline Environment Setup")
    print("=" * 50)

    success = True

    # Optional conda environment setup
    if input("Create conda environment? (y/n): ").lower() == 'y':
        success &= setup_conda_environment()

    # Install PyTorch
    success &= install_pytorch()

    # Install ML packages
    success &= install_ml_packages()

    # Verify installation
    success &= verify_installation()

    if success:
        print("=" * 50)
        print("[SUCCESS] Setup completed!")
        print("Activate environment with: conda activate sunset-rtx3060")
        print("=" * 50)
    else:
        print("=" * 50)
        print("[ERROR] Setup failed")
        print("Check logs and try again")
        print("=" * 50)

if __name__ == "__main__":
    main()
'''

        setup_script_path = self.scripts_dir / "setup_environment.py"
        with open(setup_script_path, 'w', encoding='utf-8') as f:
            f.write(setup_script)
        print(f"[OK] {setup_script_path}")

    def create_main_pipeline_script(self):
        """Create main pipeline script"""
        print("[INFO] Creating main pipeline script...")

        pipeline_script = '''#!/usr/bin/env python3
"""
RTX 3060 Optimized Sunset Pipeline Main Script
Sunset Pipeline Main Execution Script
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

class SunsetPipelineRTX3060:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.scripts_dir = self.project_root / "scripts"

        # Load configuration files
        self.load_configs()

    def load_configs(self):
        """Load configuration files"""
        config_files = ['hardware.json', 'training.json', 'dataset.json', 'benchmark.json']

        self.configs = {}
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.configs[config_file.replace('.json', '')] = json.load(f)
            else:
                print(f"[WARN] Config file not found: {config_file}")

    def run_data_pipeline(self):
        """Run data pipeline"""
        print("[INFO] Starting data pipeline...")
        data_script = self.scripts_dir / "data_processing" / "dataset_pipeline.py"
        if data_script.exists():
            os.system(f"python {data_script}")
        else:
            print("[ERROR] Data pipeline script not found")

    def run_model_training(self):
        """Run model training"""
        print("[INFO] Starting model training...")
        training_script = self.scripts_dir / "training" / "train_quadrality_model.py"
        if training_script.exists():
            os.system(f"python {training_script}")
        else:
            print("[ERROR] Training script not found")

    def run_evaluation(self):
        """Run evaluation"""
        print("[INFO] Starting evaluation...")
        eval_script = self.scripts_dir / "evaluation" / "run_benchmarks.py"
        if eval_script.exists():
            os.system(f"python {eval_script}")
        else:
            print("[ERROR] Evaluation script not found")

    def run_abc_testing(self):
        """Run ABC testing"""
        print("[INFO] Starting ABC testing...")
        abc_script = self.scripts_dir / "evaluation" / "abc_testing.py"
        if abc_script.exists():
            os.system(f"python {abc_script}")
        else:
            print("[ERROR] ABC testing script not found")

    def run_full_pipeline(self):
        """Run full pipeline"""
        print("[START] Sunset Pipeline RTX 3060 Full Execution")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        try:
            # Phase 1: Data preparation
            print("\\n[PHASE 1] Data Pipeline")
            self.run_data_pipeline()

            # Phase 2: Model training
            print("\\n[PHASE 2] Model Training")
            self.run_model_training()

            # Phase 3: Evaluation
            print("\\n[PHASE 3] Evaluation")
            self.run_evaluation()

            # Phase 4: ABC testing
            print("\\n[PHASE 4] ABC Testing")
            self.run_abc_testing()

            print("\\n" + "=" * 60)
            print("[SUCCESS] Sunset Pipeline Execution Completed!")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

        except Exception as e:
            print(f"[ERROR] Error occurred: {e}")
            return False

        return True

def main():
    parser = argparse.ArgumentParser(description='RTX 3060 Sunset Pipeline')
    parser.add_argument('--phase', choices=['data', 'training', 'evaluation', 'abc', 'full'],
                       default='full', help='Phase to execute')
    parser.add_argument('--config', help='Configuration directory')

    args = parser.parse_args()

    pipeline = SunsetPipelineRTX3060()

    if args.config:
        pipeline.config_dir = Path(args.config)

    if args.phase == 'data':
        pipeline.run_data_pipeline()
    elif args.phase == 'training':
        pipeline.run_model_training()
    elif args.phase == 'evaluation':
        pipeline.run_evaluation()
    elif args.phase == 'abc':
        pipeline.run_abc_testing()
    elif args.phase == 'full':
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
'''

        pipeline_script_path = self.scripts_dir / "run_sunset_pipeline.py"
        with open(pipeline_script_path, 'w', encoding='utf-8') as f:
            f.write(pipeline_script)
        print(f"[OK] {pipeline_script_path}")

    def create_monitoring_script(self):
        """Create monitoring script"""
        print("[INFO] Creating monitoring script...")

        monitor_script = '''#!/usr/bin/env python3
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
            f.write(json.dumps(stats, ensure_ascii=False) + '\\n')

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
            print("\\n[STOP] Monitoring stopped")

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
            print("\\nGPU:")
            print(".1f")
            print(".1f")
            print(".1f")
            print(".1f")
        else:
            print("\\nGPU: Not detected")

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
'''

        monitor_script_path = self.scripts_dir / "monitor_system.py"
        with open(monitor_script_path, 'w', encoding='utf-8') as f:
            f.write(monitor_script)
        print(f"[OK] {monitor_script_path}")

    def setup_complete(self):
        """Setup completion message"""
        print("\\n" + "=" * 60)
        print("[SUCCESS] RTX 3060 Sunset Pipeline Setup Completed!")
        print("=" * 60)
        print("\\n[GUIDE] Next steps:")
        print("1. Environment setup: python scripts/setup_environment.py")
        print("2. Pipeline execution: python scripts/run_sunset_pipeline.py")
        print("3. System monitoring: python scripts/monitor_system.py --current")
        print("\\n[CONFIG] Configuration files:")
        for config_file in ['hardware.json', 'training.json', 'dataset.json', 'benchmark.json']:
            print(f"   - config/{config_file}")
        print("\\n[LOGS] Log directory: logs/")
        print("\\n[READY] Ready to start Sunset Pipeline!")
        print("=" * 60)

def main():
    """Main setup function"""
    print("[START] RTX 3060 Sunset Pipeline Setup")
    print("=" * 60)

    setup = RTX3060SunsetSetup()

    # System requirements check
    if not setup.check_system_requirements():
        print("[ERROR] System requirements not met")
        return False

    # Directory creation
    setup.setup_directories()

    # Configuration file creation
    setup.create_config_files()

    # Script creation
    setup.create_environment_setup_script()
    setup.create_main_pipeline_script()
    setup.create_monitoring_script()

    # Completion message
    setup.setup_complete()

    return True

if __name__ == "__main__":
    main()