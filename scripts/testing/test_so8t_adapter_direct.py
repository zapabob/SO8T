#!/usr/bin/env python3
"""
SO8TAdapter直接テスト

bitsandbytes依存を避けてSO8TAdapter単体をテスト
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接インポートしてbitsandbytes依存を避ける
sys.path.insert(0, str(project_root / "so8t" / "core"))
from so8t_adapter import SO8TAdapter

def test_adapter_basic():
    """基本機能テスト"""
    print("Testing SO8TAdapter basic functionality...")

    # アダプタ作成
    adapter = SO8TAdapter(hidden_size=64, so8_dim=8, init_strength=0.1, use_matrix_exp=True)
    print("✓ Adapter created successfully")

    # パラメータ確認
    print(f"Projection weight shape: {adapter.proj.weight.shape}")
    print(f"A_params shape: {adapter.A_params.shape}")
    print(f"Strength initial value: {adapter.strength.item()}")

    # ダミー入力
    batch_size, seq_len, hidden_size = 2, 10, 64
    h = torch.randn(batch_size, seq_len, hidden_size)
    alpha = 0.5

    # 順伝播
    h_out = adapter(h, alpha)
    print(f"Input shape: {h.shape}")
    print(f"Output shape: {h_out.shape}")
    print("✓ Forward pass successful")

    # 出力差分確認
    diff = (h_out - h).abs().mean().item()
    print(f"Mean output difference: {diff:.6f}")
    print("✓ Output difference is reasonable")

def test_skew_symmetric():
    """skew-symmetric行列構築テスト"""
    print("\nTesting skew-symmetric matrix construction...")

    adapter = SO8TAdapter(hidden_size=64, so8_dim=8, init_strength=0.1)

    # skew-symmetric行列を取得
    A = adapter._build_skew_symmetric_matrix()
    print(f"A shape: {A.shape}")

    # skew-symmetric性確認: A^T = -A
    A_transpose = A.t()
    diff = (A + A_transpose).abs().max().item()
    print(f"Skew-symmetric check (should be ~0): {diff:.2e}")

    if diff < 1e-6:
        print("✓ Matrix is skew-symmetric")
    else:
        print("✗ Matrix is not skew-symmetric")

def test_rotation_matrix():
    """回転行列テスト"""
    print("\nTesting rotation matrix computation...")

    adapter = SO8TAdapter(hidden_size=64, so8_dim=8, init_strength=0.1)

    alphas = [0.0, 0.1, 0.5, 1.0]

    for alpha in alphas:
        R = adapter._compute_rotation_matrix(alpha)
        print(f"Alpha={alpha}: R shape={R.shape}")

        # 直交性チェック
        I = torch.eye(8, dtype=R.dtype, device=R.device)
        orth_error = torch.norm(R.t() @ R - I).item()
        print(f"  Orthogonality error: {orth_error:.2e}")

        # 行列式チェック
        det = torch.det(R).item()
        det_error = abs(det - 1.0)
        print(f"  Determinant error: {det_error:.2e}")

        if orth_error < 1e-5 and det_error < 1e-5:
            print("  ✓ Good rotation matrix")
        elif alpha == 0.0 and orth_error < 1e-10:
            print("  ✓ Identity matrix (alpha=0)")
        else:
            print("  ⚠ Suboptimal rotation matrix")

def test_gradient_flow():
    """勾配フロー検証"""
    print("\nTesting gradient flow...")

    adapter = SO8TAdapter(hidden_size=64, so8_dim=8, init_strength=0.1)

    # ダミー入力
    h = torch.randn(2, 10, 64, requires_grad=True)
    alpha = 0.5

    # 順伝播
    h_out = adapter(h, alpha)

    # 損失計算
    loss = h_out.sum()

    # 逆伝播
    loss.backward()

    # 勾配確認
    strength_grad = adapter.strength.grad
    proj_grad = adapter.proj.weight.grad
    A_grad = adapter.A_params.grad

    print(f"Strength gradient: {strength_grad.item() if strength_grad is not None else None}")
    print(f"Projection gradient norm: {proj_grad.norm().item() if proj_grad is not None else None}")
    print(f"A_params gradient norm: {A_grad.norm().item() if A_grad is not None else None}")

    if (strength_grad is not None and abs(strength_grad.item()) > 0 and
        proj_grad is not None and proj_grad.norm().item() > 0 and
        A_grad is not None and A_grad.norm().item() > 0):
        print("✓ All gradients flow correctly")
    else:
        print("✗ Some gradients are missing or zero")

def test_zero_strength():
    """λ=0での恒等性テスト"""
    print("\nTesting zero strength (λ=0)...")

    adapter = SO8TAdapter(hidden_size=64, so8_dim=8, init_strength=0.0)

    # ダミー入力
    h = torch.randn(2, 10, 64)
    alpha = 0.5

    # 出力計算
    h_out = adapter(h, alpha)

    # 差分確認
    diff = (h_out - h).abs().max().item()
    print(f"Max difference with λ=0: {diff:.2e}")

    if diff < 1e-6:
        print("✓ λ=0 gives identity transformation")
    else:
        print("✗ λ=0 does not give identity transformation")

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("SO8TAdapter Direct Test")
    print("=" * 60)

    try:
        test_adapter_basic()
        test_skew_symmetric()
        test_rotation_matrix()
        test_gradient_flow()
        test_zero_strength()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
