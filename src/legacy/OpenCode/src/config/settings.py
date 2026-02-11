"""
設定管理モジュール。

Features:
    - 型安全な設定クラス
    - YAML/JSONサポート（オプション）
    - ハードウェア制約対応
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class HardwareConfig:
    """ハードウェア制約設定。

    Attributes:
        gpu_memory_gb: GPUメモリ（GB）
        cpu_cores: CPUコア数
        ram_gb: RAM（GB）
        max_batch_size: 最大バッチサイズ
        max_sequence_length: 最大シーケンス長
        use_4bit_quantization: 4-bit量子化を使用するか
        gradient_checkpointing: 勾配チェックポインティング
    """

    gpu_memory_gb: float = 12.0
    cpu_cores: int = 6
    ram_gb: int = 32
    max_batch_size: int = 4
    max_sequence_length: int = 2048
    use_4bit_quantization: bool = True
    gradient_checkpointing: bool = True

    @property
    def is_gpu_limited(self) -> bool:
        """GPUメモリ制約があるか返す。"""
        return self.gpu_memory_gb < 16.0

    @property
    def recommended_batch_size(self) -> int:
        """推奨バッチサイズを返す。"""
        if self.is_gpu_limited:
            return min(self.max_batch_size, 2)
        return self.max_batch_size


@dataclass
class KromHCConfig:
    """KromHC設定。

    Attributes:
        n_streams: 残差ストリーム数
        hidden_dim: 隠れ次元
        factor_dim: 因子行列の次元
        learnable_projection: 射影を学習可能にするか
        sinkhorn_iter: Sinkhorn反復回数
    """

    n_streams: int = 4
    hidden_dim: int = 256
    factor_dim: Optional[int] = None
    learnable_projection: bool = False
    sinkhorn_iter: int = 50

    def __post_init__(self) -> None:
        if self.factor_dim is None:
            self.factor_dim = self.n_streams


@dataclass
class DGPOConfig:
    """DGPO設定。

    Attributes:
        group_size: グループサイズ
        learning_rate: 学習率
        entropy_coef: エントロピー係数
        clip_epsilon: クリッピング係数
        difficulty_aware: 難易度認識を有効化するか
        reformulation_aspects: 問題再構成の側面
    """

    group_size: int = 8
    learning_rate: float = 1e-4
    entropy_coef: float = 0.01
    clip_epsilon: float = 0.2
    difficulty_aware: bool = True
    reformulation_aspects: list[str] = field(
        default_factory=lambda: ["constraint", "reasoning", "numeric"]
    )


@dataclass
class BenchmarkConfig:
    """ベンチマーク設定。

    Attributes:
        n_shots: 評価時のN-shot設定
        batch_size: バッチサイズ
        output_dir: 出力ディレクトリ
        save_results: 結果を保存するか
    """

    n_shots: list[int] = field(default_factory=lambda: [0, 1, 5])
    batch_size: int = 8
    output_dir: Path = Path("benchmark_results")
    save_results: bool = True


@dataclass
class Settings:
    """アプリケーション設定。

    Attributes:
        hardware: ハードウェア設定
        kromhc: KromHC設定
        dgrpo: DGPO設定
        benchmark: ベンチマーク設定
        log_level: ログレベル
        workdir: ワークディレクトリ
    """

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    kromhc: KromHCConfig = field(default_factory=KromHCConfig)
    dgrpo: DGPOConfig = field(default_factory=DGPOConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    log_level: str = "INFO"
    workdir: Path = Path(".")


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """設定を読み込む。

    Args:
        config_path: 設定ファイルパス

    Returns:
        設定オブジェクト
    """
    if config_path is None:
        return Settings()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    if config_path.suffix == ".yaml":
        return _load_yaml_settings(config_path)
    elif config_path.suffix == ".json":
        return _load_json_settings(config_path)
    else:
        raise ValueError(f"未対応のファイル形式: {config_path.suffix}")


def _load_yaml_settings(path: Path) -> Settings:
    """YAML設定ファイルを読み込む。"""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Settings(**data)
    except ImportError:
        raise ImportError("YAMLサポートにはPyYAMLが必要です")


def _load_json_settings(path: Path) -> Settings:
    """JSON設定ファイルを読み込む。"""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Settings(**data)
