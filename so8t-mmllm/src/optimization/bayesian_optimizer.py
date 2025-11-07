"""
ベイズ最適化によるハイパーパラメータ調整

Optunaベースのベイズ最適化でSafetyAwareSO8TConfigのハイパーパラメータを自動調整する。
"""

from typing import Dict, Any, Optional, List, Callable
import json
import os
from pathlib import Path

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

from ..models.safety_aware_so8t import SafetyAwareSO8TConfig, SafetyAwareSO8TModel
from transformers import AutoTokenizer


class BayesianHyperparameterOptimizer:
    """
    ベイズ最適化によるハイパーパラメータ調整
    
    OptunaベースのTPE（Tree-structured Parzen Estimator）を使用して
    SafetyAwareSO8TConfigのハイパーパラメータを最適化する。
    """
    
    def __init__(
        self,
        base_model_name: str,
        train_dataset: Any,
        val_dataset: Any,
        device: str = "cuda",
        n_trials: int = 50,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        """
        Args:
            base_model_name: ベースモデル名
            train_dataset: 訓練データセット
            val_dataset: 検証データセット
            device: デバイス
            n_trials: トライアル数
            n_jobs: 並列実行数
            study_name: スタディ名
            storage: ストレージ（SQLite等）
        """
        self.base_model_name = base_model_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name or "so8t_hyperparameter_optimization"
        
        # Optunaスタディの作成
        if storage:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                load_if_exists=True,
                direction="maximize",  # 安全性メトリクスを最大化
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            )
        else:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            )
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        目的関数（Optunaトライアル）
        
        Args:
            trial: Optunaトライアル
        
        Returns:
            目的関数値（安全性メトリクス）
        """
        # ハイパーパラメータの提案
        pet_lambda = trial.suggest_float("pet_lambda", 0.0, 1.0, log=True)
        alpha_safety = trial.suggest_float("alpha_safety", 0.5, 5.0)
        beta_danger_penalty = trial.suggest_float("beta_danger_penalty", 1.0, 20.0)
        gamma_safe_allow_reward = trial.suggest_float("gamma_safe_allow_reward", 0.1, 3.0)
        delta_escalate_penalty = trial.suggest_float("delta_escalate_penalty", 0.1, 2.0)
        safety_conf_threshold = trial.suggest_float("safety_conf_threshold", 0.5, 0.95)
        
        # 幾何学的制約の重み
        mu_norm = trial.suggest_float("mu_norm", 0.0, 0.1, log=True)
        nu_orth = trial.suggest_float("nu_orth", 0.0, 0.1, log=True)
        rho_iso = trial.suggest_float("rho_iso", 0.0, 0.1, log=True)
        
        # 設定を作成
        config = SafetyAwareSO8TConfig(
            pet_lambda=pet_lambda,
            alpha_safety=alpha_safety,
            beta_danger_penalty=beta_danger_penalty,
            gamma_safe_allow_reward=gamma_safe_allow_reward,
            delta_escalate_penalty=delta_escalate_penalty,
            safety_conf_threshold=safety_conf_threshold,
            mu_norm=mu_norm,
            nu_orth=nu_orth,
            rho_iso=rho_iso,
        )
        
        # モデルを作成
        model = SafetyAwareSO8TModel(
            base_model_name_or_path=self.base_model_name,
            so8t_config=config,
        ).to(self.device)
        
        # 簡易訓練（実際にはより詳細な訓練ループが必要）
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        
        # 訓練ループ（簡易版）
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
        )
        
        for epoch in range(1):  # 簡易版なので1エポックのみ
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # 早期終了の判定
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # 検証セットで評価
        model.eval()
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
        )
        
        total_refuse_recall = 0.0
        total_escalate_recall = 0.0
        total_f1 = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                safety_logits = outputs["safety_logits"]
                safety_probs = safety_logits.softmax(dim=-1)
                pred = safety_probs.argmax(dim=-1)
                
                if "safety_labels" in batch:
                    labels = batch["safety_labels"]
                    
                    # REFUSE再現率（最重要）
                    refuse_mask = labels == 2
                    if refuse_mask.any():
                        refuse_recall = (pred[refuse_mask] == 2).float().mean()
                        total_refuse_recall += refuse_recall.item() * refuse_mask.sum().item()
                    
                    # ESCALATE再現率
                    escalate_mask = labels == 1
                    if escalate_mask.any():
                        escalate_recall = (pred[escalate_mask] == 1).float().mean()
                        total_escalate_recall += escalate_recall.item() * escalate_mask.sum().item()
                    
                    # F1スコア（簡易版）
                    correct = (pred == labels).float().mean()
                    total_f1 += correct.item()
                    total_samples += labels.size(0)
        
        # 安全性メトリクスを計算
        if total_samples > 0:
            avg_refuse_recall = total_refuse_recall / total_samples if total_samples > 0 else 0.0
            avg_escalate_recall = total_escalate_recall / total_samples if total_samples > 0 else 0.0
            avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
            
            # 目的関数: REFUSE再現率を最重要視
            objective_value = (
                0.5 * avg_refuse_recall +
                0.3 * avg_escalate_recall +
                0.2 * avg_f1
            )
        else:
            objective_value = 0.0
        
        return objective_value
    
    def optimize(self) -> optuna.Study:
        """
        最適化を実行
        
        Returns:
            Optunaスタディ
        """
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
        )
        
        return self.study
    
    def get_best_config(self) -> SafetyAwareSO8TConfig:
        """
        最適なハイパーパラメータを取得
        
        Returns:
            最適化されたSafetyAwareSO8TConfig
        """
        best_params = self.study.best_params
        
        return SafetyAwareSO8TConfig(
            pet_lambda=best_params["pet_lambda"],
            alpha_safety=best_params["alpha_safety"],
            beta_danger_penalty=best_params["beta_danger_penalty"],
            gamma_safe_allow_reward=best_params["gamma_safe_allow_reward"],
            delta_escalate_penalty=best_params["delta_escalate_penalty"],
            safety_conf_threshold=best_params["safety_conf_threshold"],
            mu_norm=best_params["mu_norm"],
            nu_orth=best_params["nu_orth"],
            rho_iso=best_params["rho_iso"],
        )
    
    def save_best_config(self, output_path: str):
        """
        最適なハイパーパラメータをJSON形式で保存
        
        Args:
            output_path: 出力パス
        """
        best_config = self.get_best_config()
        
        # dataclassを辞書に変換
        config_dict = {
            "pet_lambda": best_config.pet_lambda,
            "alpha_safety": best_config.alpha_safety,
            "beta_danger_penalty": best_config.beta_danger_penalty,
            "gamma_safe_allow_reward": best_config.gamma_safe_allow_reward,
            "delta_escalate_penalty": best_config.delta_escalate_penalty,
            "safety_conf_threshold": best_config.safety_conf_threshold,
            "mu_norm": best_config.mu_norm,
            "nu_orth": best_config.nu_orth,
            "rho_iso": best_config.rho_iso,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
        }
        
        # JSON形式で保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def visualize(self, output_dir: str):
        """
        最適化過程を可視化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 最適化履歴
        fig = plot_optimization_history(self.study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        
        # パラメータ重要度
        fig = plot_param_importances(self.study)
        fig.write_html(str(output_dir / "param_importances.html"))
        
        # パラレル座標
        fig = plot_parallel_coordinate(self.study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))


