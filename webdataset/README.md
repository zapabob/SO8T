# webdataset Directory

This directory is a symbolic link (junction) to `D:\webdataset`.

## Purpose

Due to limited space on the C: drive, all model checkpoints, GGUF files, training data, and other large outputs are stored on the D: drive.

## Directory Structure

```
D:\webdataset\
├── gguf_models\          # GGUF converted models
├── checkpoints\          # Training and fine-tuning checkpoints
│   ├── training\         # Training checkpoints
│   └── finetuning\       # Fine-tuning checkpoints
├── weights\              # Model weights
├── models\               # Final models
│   └── final\
└── wikipedia_*.jsonl     # Wikipedia crawled data
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

## Git Configuration

- Directory structure is tracked by Git
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

## Notes

- The symlink must be created on each machine that clones the repository
- Windows requires Administrator privileges to create symlinks
- If symlink creation fails, scripts will fall back to `D:\webdataset` directly

