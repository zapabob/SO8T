# webdataset Directory

This directory is a symbolic link (junction) to `D:\webdataset`.

## Purpose

Due to limited space on the C: drive, all model checkpoints, GGUF files, training data, and other large outputs are stored on the D: drive.

## Directory Structure

```
D:\webdataset\
├── datasets\             # Large datasets (downloaded from HuggingFace)
│   ├── wikipedia_ja\     # Wikipedia Japanese dataset
│   ├── cc100_ja\         # CC-100 Japanese corpus
│   └── ...
├── processed\            # Processed data from pipeline
│   ├── web_crawled\      # Web scraped data
│   ├── cleaned\          # Cleaned data
│   ├── labeled\          # Labeled data
│   ├── thinking\         # Thinking format data
│   └── four_class\       # Four-class classified data
├── gguf_models\          # GGUF converted models
├── checkpoints\          # Training and fine-tuning checkpoints
│   ├── training\         # Training checkpoints
│   ├── finetuning\       # Fine-tuning checkpoints
│   └── pipeline\         # Pipeline checkpoints
├── weights\              # Model weights
└── models\               # Final models
    └── final\
```

## Symlink Setup

This directory is created as a symlink using:

```powershell
# Create symlink (run as Administrator)
cmd /c mklink /D webdataset D:\webdataset
```

Or using PowerShell (run as Administrator):

```powershell
New-Item -ItemType SymbolicLink -Path "webdataset" -Target "D:\webdataset"
```

Or create a junction (does not require Administrator):

```powershell
cmd /c mklink /J webdataset D:\webdataset
```

## Git Configuration

- Directory structure is tracked by Git (via .gitkeep files)
- Large files (`.pt`, `.gguf`, `.jsonl`, etc.) are excluded via `.gitignore`
- Metadata files (`.md`, `.txt`, `.json` configs) are tracked

## Usage

All scripts should save outputs to `D:\webdataset` or use the `webdataset/` symlink:

```python
from pathlib import Path

# Option 1: Use absolute path
output_dir = Path(r"D:\webdataset\gguf_models") / model_name

# Option 2: Use symlink (relative to repo root)
output_dir = Path("webdataset/gguf_models") / model_name
```

## Downloading Large Datasets

Use the download script to download large datasets to `D:\webdataset\datasets`:

```bash
# List available datasets
python scripts/data/download_large_datasets.py --list

# Download default datasets (wikipedia_ja, cc100_ja)
python scripts/data/download_large_datasets.py --dataset all

# Download specific dataset
python scripts/data/download_large_datasets.py --dataset wikipedia_ja

# Download with sample limit
python scripts/data/download_large_datasets.py --dataset wikipedia_ja --max-samples 10000

# Download custom HuggingFace dataset
python scripts/data/download_large_datasets.py --custom-dataset "dataset_name" --config "config_name"
```

## Download and Start Production Pipeline

Download datasets and start production pipeline in one command:

```bash
# Download datasets and start pipeline
python scripts/pipelines/download_and_start_production.py

# Only download datasets
python scripts/pipelines/download_and_start_production.py --download-only

# Skip download, start pipeline directly
python scripts/pipelines/download_and_start_production.py --skip-download
```

## Notes

- The symlink must be created on each machine that clones the repository
- Windows requires Administrator privileges to create symlinks (use junction as alternative)
- If symlink creation fails, scripts will fall back to `D:\webdataset` directly
- All large files are stored on D: drive to save C: drive space
