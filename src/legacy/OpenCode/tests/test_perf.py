import time

import torch

from so8t_core.attention_so8 import apply_so8_rotation


def test_rotation_runtime_budget():
    x = torch.randn(2, 128, 256)
    theta = torch.randn(32, 8, 8)
    start = time.time()
    _ = apply_so8_rotation(x, theta)
    elapsed = time.time() - start
    assert elapsed < 0.1
