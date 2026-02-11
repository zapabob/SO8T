#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tqdm progress helpers.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback

T = TypeVar("T")


def progress(iterable: Iterable[T], *, desc: Optional[str] = None, total: Optional[int] = None) -> Iterator[T]:
    """
    Wrap an iterable with tqdm if available.
    """
    if tqdm is None:
        return iter(iterable)
    return tqdm(iterable, desc=desc, total=total)

