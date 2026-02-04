#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim for Borea retraining.
Delegates to src.training.borea_adapter_pipeline.
"""
from src.training.borea_adapter_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
