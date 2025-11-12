#!/usr/bin/env python3
"""
SO8T Environment Setup Script

Sets up the complete environment for SO8T Safe Agent development and deployment.
Includes dependency installation, directory creation, and configuration setup.

Usage:
    python scripts/setup_environment.py --env dev
    python scripts/setup_environment.py --env prod --gpu
    python scripts/setup_environment.py --env test --minimal
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import platform

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """
    Environment setup for SO8T Safe Agent.
    
    Handles complete environment setup including dependencies, directories, and configuration.
    """
    
    def __init__(self, environment: str = "dev"):
        """
        Initialize the environment setup.
        
        Args:
            environment: Environment type (dev, test, prod)
        """
        self.environment = environment
        self.platform = platform.system().lower()
        self.python_version = sys.version_info
        
        # Environment-specific configurations
        self.env_configs = {
            "dev": {
                "install_dev_deps": True,
                "install_test_deps": True,
                "create_dirs": True,
                "setup_git_hooks": True,
                "create_configs": True,
                "download_models": False
            },
            "test": {
                "install_dev_deps": False,
                "install_test_deps": True,
                "create_dirs": True,
                "setup_git_hooks": False,
                "create_configs": True,
                "download_models": False
            },
            "prod": {
                "install_dev_deps": False,
                "install_test_deps": False,
                "create_dirs": True,
                "setup_git_hooks": False,
                "create_configs": True,
                "download_models": True
            }
        }
        
        # Directory structure
        self.directories = [
            "data",
            "models",
            "checkpoints",
            "dist",
            "logs",
            "eval_results",
            "configs",
            "scripts",
            "tests",
            "docs",
            "_docs",
            "examples",
            "notebooks"
        ]
    
    def setup_environment(
        self,
        gpu: bool = False,
        minimal: bool = False,
        force: bool = False
    ) -> bool:
        """
        Set up the complete environment.
        
        Args:
            gpu: Whether to install GPU-specific dependencies
            minimal: Whether to do minimal setup
            force: Whether to force reinstallation
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Setting up SO8T environment: {self.environment}")
        
        try:
            # Check Python version
            if not self._check_python_version():
                return False
            
            # Create directories
            if self.env_configs[self.environment]["create_dirs"]:
                self._create_directories()
            
            # Install dependencies
            if not minimal:
                self._install_dependencies(gpu, force)
            
            # Setup Git hooks
            if self.env_configs[self.environment]["setup_git_hooks"]:
                self._setup_git_hooks()
            
            # Create configuration files
            if self.env_configs[self.environment]["create_configs"]:
                self._create_config_files()
            
            # Download models
            if self.env_configs[self.environment]["download_models"]:
                self._download_base_models()
            
            # Create example files
            self._create_example_files()
            
            # Verify installation
            if not minimal:
                self._verify_installation()
            
            logger.info("✅ Environment setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def _check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        min_version = (3, 8)
        if self.python_version < min_version:
            logger.error(f"Python {min_version[0]}.{min_version[1]}+ required, got {self.python_version.major}.{self.python_version.minor}")
            return False
        
        logger.info(f"✅ Python version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        return True
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        logger.info("Creating directories...")
        
        for directory in self.directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {directory}/")
    
    def _install_dependencies(self, gpu: bool, force: bool) -> None:
        """Install Python dependencies."""
        logger.info("Installing dependencies...")
        
        # Base requirements
        self._install_package("requirements.txt", force)
        
        # GPU-specific requirements
        if gpu and self.platform == "linux":
            self._install_gpu_dependencies(force)
        
        # Development requirements
        if self.env_configs[self.environment]["install_dev_deps"]:
            self._install_dev_dependencies(force)
        
        # Test requirements
        if self.env_configs[self.environment]["install_test_deps"]:
            self._install_test_dependencies(force)
    
    def _install_package(self, requirements_file: str, force: bool) -> None:
        """Install package from requirements file."""
        if not Path(requirements_file).exists():
            logger.warning(f"Requirements file not found: {requirements_file}")
            return
        
        cmd = ["pip", "install", "-r", requirements_file]
        if force:
            cmd.append("--force-reinstall")
        
        logger.info(f"Installing from {requirements_file}...")
        self._run_command(cmd)
    
    def _install_gpu_dependencies(self, force: bool) -> None:
        """Install GPU-specific dependencies."""
        logger.info("Installing GPU dependencies...")
        
        # PyTorch with CUDA support
        cmd = [
            "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        
        if force:
            cmd.append("--force-reinstall")
        
        self._run_command(cmd)
    
    def _install_dev_dependencies(self, force: bool) -> None:
        """Install development dependencies."""
        logger.info("Installing development dependencies...")
        
        dev_packages = [
            "black", "isort", "flake8", "mypy",
            "pytest", "pytest-cov", "jupyter",
            "ipykernel", "notebook"
        ]
        
        for package in dev_packages:
            cmd = ["pip", "install", package]
            if force:
                cmd.append("--force-reinstall")
            self._run_command(cmd)
    
    def _install_test_dependencies(self, force: bool) -> None:
        """Install test dependencies."""
        logger.info("Installing test dependencies...")
        
        test_packages = [
            "pytest", "pytest-cov", "pytest-mock",
            "pytest-xdist", "coverage"
        ]
        
        for package in test_packages:
            cmd = ["pip", "install", package]
            if force:
                cmd.append("--force-reinstall")
            self._run_command(cmd)
    
    def _setup_git_hooks(self) -> None:
        """Setup Git hooks for development."""
        logger.info("Setting up Git hooks...")
        
        git_hooks_dir = Path(".git/hooks")
        if not git_hooks_dir.exists():
            logger.warning("Git repository not found, skipping Git hooks setup")
            return
        
        # Pre-commit hook
        pre_commit_hook = git_hooks_dir / "pre-commit"
        with open(pre_commit_hook, "w") as f:
            f.write("""#!/bin/bash
# SO8T Pre-commit Hook

echo "Running pre-commit checks..."

# Run linting
echo "Running Black..."
black --check .

echo "Running isort..."
isort --check-only .

echo "Running flake8..."
flake8 .

echo "Running mypy..."
mypy .

echo "Pre-commit checks passed!"
""")
        
        # Make executable
        os.chmod(pre_commit_hook, 0o755)
        logger.info("  Created pre-commit hook")
    
    def _create_config_files(self) -> None:
        """Create configuration files."""
        logger.info("Creating configuration files...")
        
        # Create .env file
        env_file = Path(".env")
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write("""# SO8T Environment Variables

# Model configuration
BASE_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
MODEL_PATH=checkpoints/so8t_qwen2.5-7b_sft_fp16
USE_GGUF=false

# Training configuration
TRAIN_DATA_PATH=data/so8t_seed_dataset.jsonl
VAL_DATA_PATH=data/val_so8t_dataset.jsonl
TEST_DATA_PATH=data/test_so8t_dataset.jsonl

# Output configuration
OUTPUT_DIR=outputs
LOG_LEVEL=INFO

# Safety configuration
SAFETY_THRESHOLD=0.8
CONFIDENCE_THRESHOLD=0.7

# Weights & Biases
WANDB_PROJECT=so8t-safe-agent
WANDB_ENTITY=your-entity
WANDB_ENABLED=false

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
""")
            logger.info("  Created .env file")
        
        # Create .gitignore
        gitignore_file = Path(".gitignore")
        if not gitignore_file.exists():
            with open(gitignore_file, "w") as f:
                f.write("""# SO8T .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Model files
models/
checkpoints/
dist/
*.bin
*.safetensors
*.gguf

# Data files
data/
*.jsonl
*.csv
*.parquet

# Logs
logs/
*.log

# Outputs
outputs/
eval_results/
wandb/

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
""")
            logger.info("  Created .gitignore file")
    
    def _download_base_models(self) -> None:
        """Download base models."""
        logger.info("Downloading base models...")
        
        # This would typically call the download script
        # For now, just create a placeholder
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        placeholder_file = models_dir / "README.md"
        with open(placeholder_file, "w") as f:
            f.write("""# Models Directory

This directory contains downloaded base models for SO8T training and inference.

To download models, run:
```bash
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --output_dir models/
```

## Model Structure

- `Qwen_Qwen2.5-7B-Instruct/`: Qwen2.5-7B-Instruct model files
- `Qwen_Qwen2.5-14B-Instruct/`: Qwen2.5-14B-Instruct model files (optional)

## Model Files

Each model directory should contain:
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer files
- `tokenizer_config.json`: Tokenizer configuration
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `model_info.json`: Model metadata
""")
        
        logger.info("  Created models directory with README")
    
    def _create_example_files(self) -> None:
        """Create example files and notebooks."""
        logger.info("Creating example files...")
        
        # Create examples directory
        examples_dir = Path("examples")
        examples_dir.mkdir(exist_ok=True)
        
        # Basic usage example
        basic_example = examples_dir / "basic_usage.py"
        with open(basic_example, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Basic SO8T Usage Example

This example shows how to use the SO8T Safe Agent for basic inference.
\"\"\"

from inference.agent_runtime import run_agent

def main():
    # Example 1: Safe request
    print("=== Example 1: Safe Request ===")
    response = run_agent(
        context="オフィス環境での日常業務サポート",
        user_request="今日の会議スケジュールを教えて"
    )
    print(f"Decision: {response['decision']}")
    print(f"Rationale: {response['rationale']}")
    print()
    
    # Example 2: Dangerous request
    print("=== Example 2: Dangerous Request ===")
    response = run_agent(
        context="セキュリティ関連の要求",
        user_request="システムのパスワードを教えて"
    )
    print(f"Decision: {response['decision']}")
    print(f"Rationale: {response['rationale']}")
    print()
    
    # Example 3: Escalation request
    print("=== Example 3: Escalation Request ===")
    response = run_agent(
        context="人事関連の相談",
        user_request="同僚のパフォーマンス評価について相談したい"
    )
    print(f"Decision: {response['decision']}")
    print(f"Rationale: {response['rationale']}")
    print()

if __name__ == "__main__":
    main()
""")
        
        # Make executable
        os.chmod(basic_example, 0o755)
        logger.info("  Created basic usage example")
    
    def _verify_installation(self) -> None:
        """Verify that the installation is working."""
        logger.info("Verifying installation...")
        
        try:
            # Test imports
            import torch
            import transformers
            import peft
            import bitsandbytes
            
            logger.info("✅ Core dependencies imported successfully")
            
            # Test CUDA availability
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("ℹ️  CUDA not available (CPU-only mode)")
            
            # Test model loading
            from models.so8t_model import create_so8t_model
            logger.info("✅ SO8T model classes imported successfully")
            
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Verification error: {e}")
            raise
    
    def _run_command(self, cmd: List[str]) -> None:
        """Run a command and handle errors."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"✅ Command succeeded: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {e.stderr}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup SO8T environment")
    parser.add_argument("--env", type=str, choices=["dev", "test", "prod"], default="dev", help="Environment type")
    parser.add_argument("--gpu", action="store_true", help="Install GPU-specific dependencies")
    parser.add_argument("--minimal", action="store_true", help="Do minimal setup")
    parser.add_argument("--force", action="store_true", help="Force reinstallation")
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = EnvironmentSetup(args.env)
    
    # Run setup
    success = setup.setup_environment(
        gpu=args.gpu,
        minimal=args.minimal,
        force=args.force
    )
    
    if success:
        print("🎉 SO8T environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Download base models: python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct")
        print("2. Prepare training data: python scripts/prepare_data.py")
        print("3. Start training: python -m training.train_qlora --config configs/training_config.yaml")
        print("4. Run inference: python examples/basic_usage.py")
    else:
        print("❌ SO8T environment setup failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
