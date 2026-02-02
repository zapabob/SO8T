"""
KromHC関連エラーの定義モジュール。

Errors:
    KromHCError: KromHC関連エラーの基底クラス
    ConvergenceError: 収束失敗時のエラー
    ModelDimensionError: モデル次元の不整合エラー
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KromHCError(Exception):
    """KromHC関連エラーの基底クラス。

    Attributes:
        message: エラーメッセージ
        details: 追加の詳細情報（辞書）
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConvergenceError(KromHCError):
    """Sinkhorn反復が収束しなかった場合のエラー。

    Attributes:
        max_iter: 最大反復回数
        final_error: 最終誤差
        matrix_shape: 行列の形状
    """

    def __init__(
        self,
        max_iter: int,
        final_error: float,
        *,
        matrix_shape: Optional[tuple[int, int]] = None,
    ) -> None:
        self.max_iter = max_iter
        self.final_error = final_error
        self.matrix_shape = matrix_shape

        message = (
            f"Sinkhorn収束失敗: max_iter={max_iter}, final_error={final_error:.2e}"
        )
        if matrix_shape:
            message += f", matrix_shape={matrix_shape}"

        details = {
            "max_iter": max_iter,
            "final_error": final_error,
            "matrix_shape": matrix_shape,
        }
        super().__init__(message, details=details)

        logger.warning(
            f"Sinkhorn収束失敗: max_iter={max_iter}, final_error={final_error:.2e}"
        )


class ModelDimensionError(KromHCError):
    """モデル次元の不整合エラー。

    Attributes:
        expected: 期待される次元
        actual: 実際の次元
        param_name: パラメータ名
    """

    def __init__(
        self,
        expected: int,
        actual: int,
        *,
        param_name: Optional[str] = None,
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.param_name = param_name

        message = f"モデル次元不一致: expected={expected}, actual={actual}"
        if param_name:
            message += f" (param={param_name})"

        details = {
            "expected": expected,
            "actual": actual,
            "param_name": param_name,
        }
        super().__init__(message, details=details)


class MatrixConstraintError(KromHCError):
    """行列制約違反エラー。

    Attributes:
        constraint: 違反した制約名
        matrix_sum: 行列の和
        expected_sum: 期待される和
    """

    def __init__(
        self,
        constraint: str,
        matrix_sum: float,
        expected_sum: float,
    ) -> None:
        self.constraint = constraint
        self.matrix_sum = matrix_sum
        self.expected_sum = expected_sum

        message = (
            f"行列制約違反: {constraint}, sum={matrix_sum:.4f}, expected={expected_sum}"
        )
        details = {
            "constraint": constraint,
            "matrix_sum": matrix_sum,
            "expected_sum": expected_sum,
        }
        super().__init__(message, details=details)


def handle_kromhc_errors(func):
    """KromHCエラーを処理するデコレータ。

    Args:
        func: デコレートする関数

    Returns:
        エラーハンドリング付きの関数
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConvergenceError:
            raise
        except ModelDimensionError:
            raise
        except KromHCError:
            raise
        except Exception as e:
            logger.error(f"予期エラー: {e}")
            raise KromHCError(f"予期エラー: {e}") from e

    return wrapper
