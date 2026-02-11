"""
ベイズ最適化の使用例

Optunaベースのベイズ最適化でハイパーパラメータを自動調整する例
"""

import torch
from transformers import AutoTokenizer
from datasets import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.safety_aware_so8t import SafetyAwareSO8TConfig
from src.optimization.bayesian_optimizer import BayesianHyperparameterOptimizer


def create_dummy_dataset(num_samples: int = 100):
    """
    ダミーデータセットを作成（実際の使用では実データを使用）
    
    Args:
        num_samples: サンプル数
    
    Returns:
        データセット
    """
    # ダミーデータを作成
    data = {
        "input_ids": torch.randint(0, 1000, (num_samples, 32)),
        "attention_mask": torch.ones(num_samples, 32),
        "labels": torch.randint(0, 1000, (num_samples, 32)),
        "safety_labels": torch.randint(0, 3, (num_samples,)),
        "is_easy_case": torch.rand(num_samples) > 0.5,
        "is_danger_case": torch.rand(num_samples) > 0.1,
    }
    
    return Dataset.from_dict(data)


def main():
    """メイン関数"""
    base_name = "Qwen/Qwen2-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("[INFO] Creating dummy datasets...")
    train_dataset = create_dummy_dataset(num_samples=50)
    val_dataset = create_dummy_dataset(num_samples=20)
    
    print("[INFO] Initializing Bayesian optimizer...")
    optimizer = BayesianHyperparameterOptimizer(
        base_model_name=base_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        n_trials=10,  # 実際の使用では50-100以上推奨
        n_jobs=1,
        study_name="so8t_hyperparameter_optimization",
    )
    
    print("[INFO] Starting optimization...")
    study = optimizer.optimize()
    
    print(f"\n[RESULT] Best value: {study.best_value:.4f}")
    print(f"[RESULT] Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    
    # 最適な設定を取得
    best_config = optimizer.get_best_config()
    print(f"\n[RESULT] Best config created: {best_config}")
    
    # 最適な設定を保存
    output_path = "configs/optimized_safety_aware_so8t.json"
    print(f"[INFO] Saving best config to {output_path}...")
    optimizer.save_best_config(output_path)
    
    # 可視化
    output_dir = "optimization_results"
    print(f"[INFO] Generating visualizations in {output_dir}...")
    optimizer.visualize(output_dir)
    
    print("\n[OK] Optimization completed!")


if __name__ == "__main__":
    main()

