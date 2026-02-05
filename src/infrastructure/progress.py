"""Progress helpers with tqdm fallback."""
from __future__ import annotations

from contextlib import contextmanager

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@contextmanager
def progress_bar(total: int, desc: str, unit: str = "step"):
    if tqdm is None:
        yield None
        return
    with tqdm(total=total, desc=desc, unit=unit) as bar:
        yield bar
