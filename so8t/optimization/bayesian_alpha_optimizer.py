#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Alpha Optimizer for SO8T Thinking Model

αの値をシグモイド関数として区間[0,1]でベイズ最適化
α=0: 統計的（元のモデルから変わらない）
α=1: 幾何的、かつ物理的
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class GaussianProcessRegressor:
    """ガウス過程回帰の実装（ベイズ最適化用）"""

    def __init__(self, length_scale: float = 1.0, noise_variance: float = 1e-6):
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """GPモデルの学習"""
        self.X_train = X
        self.y_train = y

        # カーネル行列計算
        K = self._rbf_kernel(X, X) + self.noise_variance * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """予測（平均と分散）"""
        if self.X_train is None:
            raise ValueError("Model not fitted")

        K_star = self._rbf_kernel(X_test, self.X_train)
        K_star_star = self._rbf_kernel(X_test, X_test)

        # 予測平均
        mu = K_star @ self.K_inv @ self.y_train

        # 予測分散
        var = K_star_star - K_star @ self.K_inv @ K_star.T
        var = np.diag(var)  # 分散のみ抽出

        return mu.flatten(), var

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBFカーネル"""
        X1 = X1.reshape(-1, 1) if X1.ndim == 1 else X1
        X2 = X2.reshape(-1, 1) if X2.ndim == 1 else X2

        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                np.sum(X2**2, axis=1) - 2 * X1 @ X2.T

        return np.exp(-0.5 * sqdist / (self.length_scale ** 2))


class BayesianAlphaOptimizer:
    """
    ベイズ最適化によるα値最適化

    α ∈ [0,1] をシグモイド関数で制御
    α=0: 統計的（元のモデルから変わらない）
    α=1: 幾何的、かつ物理的
    """

    def __init__(self, alpha_bounds: Tuple[float, float] = (0.0, 1.0),
                 n_initial_points: int = 5, n_iterations: int = 25,
                 exploration_weight: float = 1.0):
        self.alpha_bounds = alpha_bounds
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight

        # GPモデル
        self.gp = GaussianProcessRegressor(length_scale=0.1)

        # 最適化履歴
        self.alpha_history = []
        self.score_history = []
        self.best_alpha = None
        self.best_score = float('-inf')

        logger.info("Bayesian Alpha Optimizer initialized:"        logger.info(f"  Alpha bounds: {alpha_bounds}")
        logger.info(f"  Initial points: {n_initial_points}")
        logger.info(f"  Iterations: {n_iterations}")

    def optimize(self, objective_function: Callable[[float], float],
                n_evaluations: int = None) -> Dict[str, Any]:
        """
        ベイズ最適化実行

        Args:
            objective_function: 目的関数 f(α) -> score
            n_evaluations: 評価回数（Noneの場合は自動決定）

        Returns:
            最適化結果
        """
        if n_evaluations is None:
            n_evaluations = self.n_initial_points + self.n_iterations

        logger.info(f"Starting Bayesian optimization with {n_evaluations} evaluations...")

        # 初期点の評価
        initial_alphas = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1],
                                   self.n_initial_points)

        for alpha in initial_alphas:
            score = objective_function(alpha)
            self._update_history(alpha, score)

        # ベイズ最適化ループ
        for iteration in range(self.n_iterations):
            # GPモデル更新
            self._update_gp_model()

            # 次の評価点を提案
            next_alpha = self._acquire_next_point()

            # 評価
            score = objective_function(next_alpha)
            self._update_history(next_alpha, score)

            logger.info(f"Iteration {iteration + 1}/{self.n_iterations}: "
                       f"α = {next_alpha:.4f}, score = {score:.4f}, "
                       f"best = {self.best_score:.4f}")

        # 最終結果
        result = {
            'best_alpha': self.best_alpha,
            'best_score': self.best_score,
            'alpha_history': self.alpha_history,
            'score_history': self.score_history,
            'optimization_path': self._get_optimization_path(),
            'convergence_analysis': self._analyze_convergence()
        }

        logger.info("Bayesian optimization completed!"        logger.info(f"Best α: {self.best_alpha:.4f} (score: {self.best_score:.4f})")

        return result

    def _update_history(self, alpha: float, score: float):
        """履歴更新"""
        self.alpha_history.append(alpha)
        self.score_history.append(score)

        if score > self.best_score:
            self.best_score = score
            self.best_alpha = alpha

    def _update_gp_model(self):
        """GPモデルの更新"""
        X = np.array(self.alpha_history).reshape(-1, 1)
        y = np.array(self.score_history)

        self.gp.fit(X, y)

    def _acquire_next_point(self) -> float:
        """次の評価点を獲得（Upper Confidence Bound）"""

        def acquisition_function(alpha: float) -> float:
            """UCB獲得関数"""
            alpha = np.array([alpha])
            mu, var = self.gp.predict(alpha)

            # UCB = μ + κσ （κは探索パラメータ）
            kappa = self.exploration_weight * np.sqrt(np.log(len(self.alpha_history) + 1))
            ucb = mu[0] + kappa * np.sqrt(var[0])

            return -ucb  # 最小化なので負の値

        # 獲得関数を最適化
        result = minimize_scalar(
            acquisition_function,
            bounds=self.alpha_bounds,
            method='bounded'
        )

        return float(result.x)

    def _get_optimization_path(self) -> Dict[str, Any]:
        """最適化パスの分析"""
        alphas = np.array(self.alpha_history)
        scores = np.array(self.score_history)

        return {
            'initial_exploration': self.n_initial_points,
            'total_evaluations': len(alphas),
            'alpha_range': [float(alphas.min()), float(alphas.max())],
            'score_range': [float(scores.min()), float(scores.max())],
            'improvement_rate': self._calculate_improvement_rate(scores),
            'exploration_exploitation_ratio': self._calculate_ee_ratio(alphas, scores)
        }

    def _calculate_improvement_rate(self, scores: np.ndarray) -> float:
        """改善率計算"""
        if len(scores) < 2:
            return 0.0

        improvements = []
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:
                improvements.append(scores[i] - scores[i-1])

        return np.mean(improvements) if improvements else 0.0

    def _calculate_ee_ratio(self, alphas: np.ndarray, scores: np.ndarray) -> float:
        """探索/活用比率計算"""
        if len(alphas) < self.n_initial_points + 5:
            return 1.0

        # 初期探索フェーズのスコア分散
        exploration_scores = scores[:self.n_initial_points]
        exploration_var = np.var(exploration_scores) if len(exploration_scores) > 1 else 0

        # 最適化フェーズのスコア分散
        optimization_scores = scores[self.n_initial_points:]
        optimization_var = np.var(optimization_scores) if len(optimization_scores) > 1 else 0

        # 探索/活用比率（分散比）
        if optimization_var == 0:
            return float('inf')
        return exploration_var / optimization_var

    def _analyze_convergence(self) -> Dict[str, Any]:
        """収束分析"""
        if len(self.score_history) < 5:
            return {'converged': False, 'iterations_to_convergence': None}

        # 移動平均で収束判定
        window_size = min(5, len(self.score_history) // 2)
        recent_scores = self.score_history[-window_size:]
        earlier_scores = self.score_history[-2*window_size:-window_size]

        if len(earlier_scores) == 0:
            return {'converged': False, 'iterations_to_convergence': None}

        # 改善が停滞しているかチェック
        improvement = np.mean(recent_scores) - np.mean(earlier_scores)
        threshold = 0.01 * abs(np.mean(self.score_history))  # 1%以内の改善

        converged = abs(improvement) < threshold

        # 収束までの反復回数推定
        iterations_to_convergence = None
        if converged:
            # 改善がthreshold以下になった最初の反復
            for i in range(window_size, len(self.score_history)):
                window_scores = self.score_history[i-window_size:i]
                prev_window = self.score_history[i-2*window_size:i-window_size]
                if abs(np.mean(window_scores) - np.mean(prev_window)) < threshold:
                    iterations_to_convergence = i
                    break

        return {
            'converged': converged,
            'iterations_to_convergence': iterations_to_convergence,
            'final_improvement_rate': improvement,
            'convergence_threshold': threshold
        }


class AlphaOptimizationEvaluator:
    """
    α最適化評価器

    異なるα値でのモデル性能を評価
    """

    def __init__(self, model, tokenizer, test_dataset,
                 benchmark_configs: List[Dict[str, Any]]):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.benchmark_configs = benchmark_configs

    def evaluate_alpha(self, alpha: float) -> float:
        """
        指定α値での評価

        Args:
            alpha: 評価するα値

        Returns:
            総合スコア
        """
        logger.info(f"Evaluating α = {alpha:.4f}...")

        # α値をモデルに設定
        self._set_model_alpha(alpha)

        # ベンチマーク評価
        benchmark_results = {}
        total_score = 0.0

        for benchmark_config in self.benchmark_configs:
            benchmark_name = benchmark_config['name']
            logger.info(f"Running {benchmark_name}...")

            try:
                score = self._run_benchmark(benchmark_config)
                benchmark_results[benchmark_name] = score
                total_score += score * benchmark_config.get('weight', 1.0)

                logger.info(f"  {benchmark_name}: {score:.4f}")

            except Exception as e:
                logger.error(f"  {benchmark_name} failed: {e}")
                benchmark_results[benchmark_name] = 0.0

        # 幾何学的/統計的バランス評価
        balance_score = self._evaluate_geometric_statistical_balance(alpha)
        total_score += balance_score * 0.2  # 20%の重み

        logger.info(f"  Balance score: {balance_score:.4f}")
        logger.info(f"  Total score: {total_score:.4f}")

        return total_score

    def _set_model_alpha(self, alpha: float):
        """モデルにα値を設定"""
        # SO8ViTアダプターのαを設定
        if hasattr(self.model, 'so8vit_adapter'):
            # シグモイド適用で[0,1]に制限
            sigmoid_alpha = torch.sigmoid(torch.tensor(alpha)).item()
            self.model.so8vit_adapter.thinking_alpha.data = torch.tensor(sigmoid_alpha)

    def _run_benchmark(self, benchmark_config: Dict[str, Any]) -> float:
        """ベンチマーク実行"""
        benchmark_type = benchmark_config.get('type', 'generation')

        if benchmark_type == 'generation':
            return self._run_generation_benchmark(benchmark_config)
        elif benchmark_type == 'classification':
            return self._run_classification_benchmark(benchmark_config)
        elif benchmark_type == 'reasoning':
            return self._run_reasoning_benchmark(benchmark_config)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    def _run_generation_benchmark(self, config: Dict[str, Any]) -> float:
        """生成ベンチマーク（本番実装）"""
        import evaluate

        metric = config.get('metric', 'bleu')
        eval_metric = evaluate.load(metric)
        total_score = 0.0
        count = 0

        num_samples = config.get("num_samples", 50)
        target_field = config.get("target_field", "reference")
        input_field = config.get("input_field", "text")
        max_prompt_length = config.get("max_prompt_length", 256)
        gen_kwargs = config.get("generation_kwargs", {
            "max_new_tokens": 50,
            "do_sample": False,
            "temperature": 1.0,
            "num_return_sequences": 1
        })

        sampled_dataset = self.test_dataset[:min(num_samples, len(self.test_dataset))]
        for sample in sampled_dataset:
            if input_field not in sample or target_field not in sample:
                continue
            try:
                prompt = sample[input_field][:max_prompt_length]
                target = sample[target_field]

                inputs = self.tokenizer(
                    prompt, truncation=True, max_length=max_prompt_length, return_tensors="pt"
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                
                generated_text = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )

                # Optionally: try different evaluation metrics
                # Not all metrics accept the same arguments.
                # Many take predictions=[...], references=[...]
                # We'll handle BLEU, ROUGE, METEOR, etc.

                # BLEU, METEOR, ROUGE etc. expect list inputs:
                metric_result = eval_metric.compute(
                    predictions=[generated_text], references=[target]
                )

                # Most metrics return a dict. Try to get the main score.
                if metric == "bleu":
                    score = metric_result.get("bleu", 0.0)
                elif metric == "rouge":
                    score = metric_result.get("rougeL", 0.0)
                elif metric == "meteor":
                    score = metric_result.get("meteor", 0.0)
                else:
                    # fallback: take first value found
                    score = next(iter(metric_result.values()))

                total_score += score
                count += 1

            except Exception as e:
                logger.warning(f"Generation benchmark failed: {e}")
                continue

        return total_score / count if count > 0 else 0.0

        for sample in self.test_dataset[:min(10, len(self.test_dataset))]:
            try:
                inputs = self.tokenizer(sample['text'][:100], return_tensors='pt',
                                      truncation=True, max_length=512)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 50,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7
                    )

                # 生成品質の簡易評価（長さベース）
                generated_length = outputs.shape[1] - inputs['input_ids'].shape[1]
                score = min(generated_length / 50.0, 1.0)  # 最大50トークンで1.0

                total_score += score
                count += 1

            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                continue

        return total_score / count if count > 0 else 0.0

    def _run_classification_benchmark(self, config: Dict[str, Any]) -> float:
        """分類ベンチマーク"""
        # 簡易実装
        correct = 0
        total = 0

        for sample in self.test_dataset[:min(50, len(self.test_dataset))]:
            if 'label' not in sample:
                continue

            try:
                # 簡易分類（実際にはより適切な分類タスクが必要）
                text = sample.get('phi35_thinking', sample.get('text', ''))
                inputs = self.tokenizer(text[:200], return_tensors='pt', truncation=True)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    prediction = outputs.logits.argmax().item()

                # ラベルマッチングの簡易評価
                expected_label = sample['label']
                if isinstance(expected_label, str):
                    # 文字列ラベルの場合の簡易マッチング
                    prediction_correct = 1 if prediction % 4 == hash(expected_label) % 4 else 0
                else:
                    prediction_correct = 1 if prediction == expected_label else 0

                correct += prediction_correct
                total += 1

            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                continue

        return correct / total if total > 0 else 0.0

    def _run_reasoning_benchmark(self, config: Dict[str, Any]) -> float:
        """推論ベンチマーク"""
        # 簡易実装（数学的推論タスク）
        correct = 0
        total = 0

        math_problems = [
            ("2 + 2 = ?", "4"),
            ("10 - 3 = ?", "7"),
            ("5 * 6 = ?", "30")
        ]

        for problem, expected in math_problems:
            try:
                prompt = f"解いてください: {problem}"
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 20,
                        do_sample=False
                    )

                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = generated.replace(prompt, '').strip()

                # 簡易正解判定
                is_correct = expected in generated or generated.strip() == expected
                correct += 1 if is_correct else 0
                total += 1

            except Exception as e:
                logger.warning(f"Reasoning failed: {e}")
                continue

        return correct / total if total > 0 else 0.0

    def _evaluate_geometric_statistical_balance(self, alpha: float) -> float:
        """幾何学的/統計的バランス評価"""
        # α=0（統計的）とα=1（幾何的）のバランスを評価
        sigmoid_alpha = 1 / (1 + math.exp(-alpha))  # シグモイド適用

        # 理想的なバランスは中間値（0.5）付近
        balance_score = 1.0 - abs(sigmoid_alpha - 0.5) * 2  # [0,1]の範囲

        # 直交誤差などの幾何学的制約も考慮
        if hasattr(self.model, 'so8vit_adapter'):
            adapter = self.model.so8vit_adapter
            if hasattr(adapter, 'thinking_stats'):
                # 直交誤差の平均
                orthogonal_errors = adapter.thinking_stats[:, 1]  # rotation_error
                geometric_penalty = orthogonal_errors.mean().item()
                balance_score *= max(0.1, 1.0 - geometric_penalty)

        return balance_score


def create_bayesian_optimizer(n_iterations: int = 25) -> BayesianAlphaOptimizer:
    """ベイズ最適化器の作成"""
    return BayesianAlphaOptimizer(
        alpha_bounds=(0.0, 1.0),
        n_initial_points=5,
        n_iterations=n_iterations,
        exploration_weight=1.0
    )


def create_evaluation_benchmarks() -> List[Dict[str, Any]]:
    """評価ベンチマーク設定"""
    return [
        {
            'name': 'generation_quality',
            'type': 'generation',
            'weight': 0.4,
            'description': 'テキスト生成品質評価'
        },
        {
            'name': 'classification_accuracy',
            'type': 'classification',
            'weight': 0.3,
            'description': '分類タスク精度評価'
        },
        {
            'name': 'reasoning_correctness',
            'type': 'reasoning',
            'weight': 0.3,
            'description': '推論タスク正確性評価'
        }
    ]
