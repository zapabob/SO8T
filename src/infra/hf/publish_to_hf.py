"""Publish artifacts to Hugging Face Hub using CLI."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Publish artifacts to Hugging Face Hub')
    parser.add_argument('--repo', required=True)
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--repo-type', default='model')
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(artifact_dir)

    cmd = [
        'huggingface-cli', 'upload',
        args.repo,
        str(artifact_dir),
        '--repo-type', args.repo_type,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


if __name__ == '__main__':
    main()
