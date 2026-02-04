#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch GRAPE (Group Representational Position Encoding) repository.

Clones: https://github.com/model-architectures/GRAPE
"""

import argparse
import logging
import subprocess
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

GRAPE_REPO = "https://github.com/zhuang-li/GRAPE.git"


def run(cmd, cwd=None):
    logger.info(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch GRAPE repo")
    parser.add_argument("--dest", type=str, default="external/GRAPE", help="Destination directory")
    parser.add_argument("--update", action="store_true", help="Pull if repo already exists")
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.exists() and (dest / ".git").exists():
        if args.update:
            run(["git", "fetch", "--all"], cwd=dest)
            run(["git", "pull", "--ff-only"], cwd=dest)
        logger.info(f"[OK] GRAPE repo exists at {dest}")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", GRAPE_REPO, str(dest)])
    logger.info(f"[OK] GRAPE repo cloned to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
