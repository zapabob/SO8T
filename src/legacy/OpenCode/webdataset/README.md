# webdataset Directory

This directory provides access to external dataset storage at `H:\from_D\webdataset`.

Due to limited space on the C: drive, all model checkpoints, GGUF files, training data, and other large outputs are stored on the H: drive external storage.

## Purpose

Due to limited space on the C: drive, all model checkpoints, GGUF files, training data, and other large outputs are stored on the D: drive.

## Directory Structure

```
H:\from_D\webdataset\
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
├── models\               # Final models
│   └── final\
├── aegis_v2.0\          # AEGIS v2.0 datasets and checkpoints
├── benchmarks\          # Benchmark datasets and results
├── coding_dataset\      # Coding-related datasets
├── nsfw_detection_dataset\  # NSFW detection datasets
└── phi35_integrated\    # Phi-3.5 integrated datasets
```

## External Storage Integration

The webdataset directory provides access to external storage at `H:\from_D\webdataset`.

### Symlink Setup (Optional)

If you want to create a local symlink for convenience:

```powershell
# Create symlink (run as Administrator)
cmd /c mklink /D webdataset H:\from_D\webdataset
```

Or using PowerShell (run as Administrator):

```powershell
New-Item -ItemType SymbolicLink -Path "webdataset" -Target "H:\from_D\webdataset"
```

Or create a junction (does not require Administrator):

```powershell
cmd /c mklink /J webdataset H:\from_D\webdataset
```

### Direct Access

Scripts can access the external storage directly:

```python
from pathlib import Path

# Direct access to external storage
external_dataset = Path(r"H:\from_D\webdataset\datasets")
external_checkpoints = Path(r"H:\from_D\webdataset\checkpoints")
```

## Git Configuration

- Directory structure is tracked by Git (via .gitkeep files)
- Large files (`.pt`, `.gguf`, `.jsonl`, etc.) are excluded via `.gitignore`
- Metadata files (`.md`, `.txt`, `.json` configs) are tracked

## Usage

All scripts should save outputs to `H:\from_D\webdataset` or use direct path access:

```python
from pathlib import Path

# Option 1: Use absolute path to external storage
output_dir = Path(r"H:\from_D\webdataset\gguf_models") / model_name

# Option 2: Use symlink if created (relative to repo root)
output_dir = Path("webdataset/gguf_models") / model_name

# Option 3: Check if symlink exists, fallback to direct path
def get_webdataset_path(subdir: str = "") -> Path:
    """Get webdataset path with fallback"""
    base_path = Path("webdataset")
    if base_path.exists() and base_path.is_symlink():
        return base_path / subdir
    else:
        return Path(r"H:\from_D\webdataset") / subdir
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

- External storage is located at `H:\from_D\webdataset` on the local machine
- Symlink creation is optional for convenience
- Scripts automatically fall back to direct path `H:\from_D\webdataset` if symlink doesn't exist
- All large files are stored on external H: drive to save C: drive space
- Repository tracks directory structure and metadata, but excludes large data files via .gitignore
