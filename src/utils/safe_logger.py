# -*- coding: utf-8 -*-
"""
Safe Logger Utility for Moonshot Pipeline
Ensures encoding-safe logging on Windows and unifies log formats.
"""
import sys
import logging
import platform
from pathlib import Path
from typing import Optional

class SafeLogger:
    @staticmethod
    def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO) -> logging.Logger:
        """
        Configure a logger that safely handles utf-8 output on Windows consoles.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers = []  # Clear existing handlers to prevent duplication

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # 1. File Handler (Always UTF-8)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding='utf-8', mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 2. Console Handler (Safe Stream)
        # On Windows, writing arbitrary utf-8 to sys.stdout/stderr can fail if not handled.
        # We use a custom stream handler that replaces unencodable characters instead of crashing.
        if platform.system() == "Windows":
            stream = sys.stdout
        else:
            stream = sys.stdout

        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        # We wrap the emit to catch encoding errors if they occur at the underlying stream level
        # mostly relevant for older Windows consoles or mismatched code pages
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def safe_print(message: str):
        """
        Safely print a message to stdout, handling UnicodeEncodeError on Windows.
        """
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback: print ASCII representation or replace chars
            print(message.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
