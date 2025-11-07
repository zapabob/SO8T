# Scripts Directory

This directory contains organized scripts for the SO8T project.

## Directory Structure

### api/
API server scripts
- `serve_fastapi.py`: FastAPI service for SO8T inference
- `serve_think_api.py`: SO8T /think endpoint implementation

### utils/
Utility scripts for setup, debugging, and maintenance
- `check_memory.py`: Memory usage checker
- `debug_*.py`: Debugging scripts
- `setup_*.py`, `setup_*.bat`, `setup_*.ps1`: Setup scripts
- `fix_*.ps1`: Fix scripts

### data/
Data processing scripts
- `clean_*.py`: Data cleaning scripts
- `split_*.py`: Dataset splitting scripts
- `label_*.py`: Data labeling scripts
- `collect_*.py`: Data collection scripts

### pipelines/
Complete pipeline scripts
- `complete_*.py`: Complete pipeline implementations
- `run_*_pipeline.py`: Pipeline execution scripts
- `so8t_*_pipeline.py`: SO8T-specific pipelines

### training/
Training scripts
- `train_*.py`: Training scripts
- `finetune_*.py`: Fine-tuning scripts
- `burnin_*.py`, `burn_in_*.py`: Burn-in scripts

### conversion/
Model conversion scripts
- `convert_*.py`: Model conversion scripts
- `integrate_*.py`: Integration scripts

### evaluation/
Evaluation scripts
- `evaluate_*.py`: Evaluation scripts
- `ab_test_*.py`: A/B testing scripts
- `compare_*.py`: Comparison scripts

### inference/
Inference and demo scripts
- `demo_*.py`: Demo scripts
- `test_*.py`: Test scripts
- `infer_*.py`: Inference scripts

## Usage

Scripts can be run from the project root:

```bash
# API server
python scripts/api/serve_think_api.py

# Training
python scripts/training/train_so8t_transformer.py

# Evaluation
python scripts/evaluation/evaluate_comprehensive.py

# Data processing
python scripts/data/clean_japanese_dataset.py
```

## Import Paths

When importing modules from scripts, use the full path:

```python
from scripts.api.serve_think_api import app
from scripts.training.train_so8t_transformer import train
from scripts.data.clean_japanese_dataset import clean_data
```

Or add the scripts directory to the Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
```

