#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
温度較正実装（Temperature Calibration）
- ECE（Expected Calibration Error）最小化
- Held-out検証セット
- 最適温度T探索（grid search）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CalibrationResult:
    """較正結果"""
    optimal_temperature: float
    ece_before: float
    ece_after: float
    brier_score_before: float
    brier_score_after: float
    accuracy: float
    confidence_mean: float
    calibration_curve: Dict[str, List[float]]


class ExpectedCalibrationError:
    """ECE（Expected Calibration Error）計算器"""
    
    def __init__(self, n_bins: int = 15):
        """
        Args:
            n_bins: ビン数（デフォルト15）
        """
        self.n_bins = n_bins
    
    def compute(self, 
                confidences: np.ndarray,
                predictions: np.ndarray,
                targets: np.ndarray) -> Tuple[float, Dict[str, List[float]]]:
        """
        ECE計算
        
        Args:
            confidences: 予測確信度 [n_samples]
            predictions: 予測クラス [n_samples]
            targets: 正解クラス [n_samples]
        
        Returns:
            ece: Expected Calibration Error
            curve_data: 較正曲線データ
        """
        # ビン境界
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        accuracies = []
        confidences_mean = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # ビン内サンプル選択
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # ビン内精度
                accuracy_in_bin = np.mean(predictions[in_bin] == targets[in_bin])
                # ビン内平均確信度
                conf_in_bin = np.mean(confidences[in_bin])
                
                # ECE寄与
                ece += np.abs(accuracy_in_bin - conf_in_bin) * prop_in_bin
                
                accuracies.append(accuracy_in_bin)
                confidences_mean.append(conf_in_bin)
                bin_counts.append(np.sum(in_bin))
            else:
                accuracies.append(0.0)
                confidences_mean.append(0.0)
                bin_counts.append(0)
        
        curve_data = {
            'accuracies': accuracies,
            'confidences': confidences_mean,
            'bin_counts': bin_counts,
            'bin_boundaries': bin_boundaries.tolist()
        }
        
        return ece, curve_data


class BrierScore:
    """Brier Score計算器"""
    
    @staticmethod
    def compute(probabilities: np.ndarray, targets: np.ndarray) -> float:
        """
        Brier Score計算
        
        Args:
            probabilities: 予測確率分布 [n_samples, n_classes]
            targets: 正解クラス [n_samples]
        
        Returns:
            brier_score: Brier Score
        """
        n_samples, n_classes = probabilities.shape
        
        # One-hot targets
        targets_one_hot = np.zeros((n_samples, n_classes))
        targets_one_hot[np.arange(n_samples), targets] = 1
        
        # Brier score
        brier = np.mean(np.sum((probabilities - targets_one_hot) ** 2, axis=1))
        
        return brier


class TemperatureScaling(nn.Module):
    """Temperature Scaling モジュール"""
    
    def __init__(self, initial_temperature: float = 1.0):
        """
        Args:
            initial_temperature: 初期温度（デフォルト1.0）
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        温度スケーリング適用
        
        Args:
            logits: [batch, n_classes]
        
        Returns:
            scaled_logits: [batch, n_classes]
        """
        return logits / self.temperature
    
    def get_temperature(self) -> float:
        """現在の温度取得"""
        return self.temperature.item()


class TemperatureCalibrator:
    """温度較正器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 n_bins: int = 15):
        """
        Args:
            model: 較正対象モデル
            device: デバイス
            n_bins: ECE計算のビン数
        """
        self.model = model
        self.device = device
        self.n_bins = n_bins
        
        self.temperature_scaler = TemperatureScaling().to(device)
        self.ece_calculator = ExpectedCalibrationError(n_bins=n_bins)
        self.brier_calculator = BrierScore()
    
    def collect_logits_and_labels(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        検証データからロジットとラベル収集
        
        Args:
            dataloader: PyTorchデータローダー
        
        Returns:
            logits: [n_samples, n_classes]
            labels: [n_samples]
        """
        print("[COLLECT] Collecting logits and labels...")
        
        self.model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting"):
                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(inputs)
                logits = outputs.logits
                
                # 最後のトークンのロジットのみ使用（生成タスクの場合）
                # または全トークンを使用（分類タスクの場合）
                all_logits.append(logits[:, -1, :].cpu().numpy())
                all_labels.append(labels[:, -1].cpu().numpy())
        
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"[OK] Collected {len(labels):,} samples")
        return logits, labels
    
    def compute_metrics(self, logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        較正前メトリクス計算
        
        Args:
            logits: [n_samples, n_classes]
            labels: [n_samples]
        
        Returns:
            metrics: メトリクス辞書
        """
        # Softmax確率
        probabilities = softmax(logits, axis=1)
        
        # 予測
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        # 精度
        accuracy = np.mean(predictions == labels)
        
        # ECE
        ece, curve_data = self.ece_calculator.compute(confidences, predictions, labels)
        
        # Brier Score
        brier = self.brier_calculator.compute(probabilities, labels)
        
        metrics = {
            'accuracy': accuracy,
            'ece': ece,
            'brier_score': brier,
            'confidence_mean': np.mean(confidences),
            'curve_data': curve_data
        }
        
        return metrics
    
    def grid_search_temperature(self,
                                 logits: np.ndarray,
                                 labels: np.ndarray,
                                 temp_range: Tuple[float, float] = (0.5, 3.0),
                                 n_temps: int = 50) -> float:
        """
        Grid searchで最適温度探索
        
        Args:
            logits: [n_samples, n_classes]
            labels: [n_samples]
            temp_range: 温度探索範囲（min, max）
            n_temps: 温度候補数
        
        Returns:
            optimal_temperature: 最適温度
        """
        print(f"\n[SEARCH] Grid search for optimal temperature...")
        print(f"Temperature range: {temp_range[0]:.2f} - {temp_range[1]:.2f}")
        print(f"Grid points: {n_temps}")
        
        temperatures = np.linspace(temp_range[0], temp_range[1], n_temps)
        best_ece = float('inf')
        best_temp = 1.0
        
        for temp in tqdm(temperatures, desc="Grid search"):
            # 温度スケーリング適用
            scaled_logits = logits / temp
            probs = softmax(scaled_logits, axis=1)
            
            # 予測
            predictions = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            
            # ECE計算
            ece, _ = self.ece_calculator.compute(confidences, predictions, labels)
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        print(f"[OK] Optimal temperature: {best_temp:.4f} (ECE: {best_ece:.6f})")
        return best_temp
    
    def optimize_temperature(self,
                             logits: torch.Tensor,
                             labels: torch.Tensor,
                             n_epochs: int = 100,
                             lr: float = 0.01) -> float:
        """
        勾配降下法で温度最適化
        
        Args:
            logits: [n_samples, n_classes]
            labels: [n_samples]
            n_epochs: エポック数
            lr: 学習率
        
        Returns:
            optimal_temperature: 最適温度
        """
        print(f"\n[OPTIMIZE] Optimizing temperature with gradient descent...")
        
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        
        optimizer = torch.optim.LBFGS([self.temperature_scaler.temperature], lr=lr, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.temperature_scaler(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
        
        for epoch in range(n_epochs):
            optimizer.step(closure)
        
        optimal_temp = self.temperature_scaler.get_temperature()
        print(f"[OK] Optimized temperature: {optimal_temp:.4f}")
        
        return optimal_temp
    
    def calibrate(self,
                  validation_dataloader,
                  method: str = 'grid_search') -> CalibrationResult:
        """
        温度較正実行
        
        Args:
            validation_dataloader: 検証データローダー
            method: 'grid_search' or 'gradient_descent'
        
        Returns:
            calibration_result: 較正結果
        """
        print(f"\n{'='*60}")
        print(f"[START] Temperature Calibration")
        print(f"Method: {method}")
        print(f"{'='*60}\n")
        
        # ロジット・ラベル収集
        logits, labels = self.collect_logits_and_labels(validation_dataloader)
        
        # 較正前メトリクス
        print("\n[METRICS] Computing pre-calibration metrics...")
        metrics_before = self.compute_metrics(logits, labels)
        print(f"Accuracy: {metrics_before['accuracy']:.4f}")
        print(f"ECE: {metrics_before['ece']:.6f}")
        print(f"Brier Score: {metrics_before['brier_score']:.6f}")
        print(f"Mean Confidence: {metrics_before['confidence_mean']:.4f}")
        
        # 最適温度探索
        if method == 'grid_search':
            optimal_temp = self.grid_search_temperature(logits, labels)
        else:
            logits_tensor = torch.from_numpy(logits).float()
            labels_tensor = torch.from_numpy(labels).long()
            optimal_temp = self.optimize_temperature(logits_tensor, labels_tensor)
        
        # 較正後メトリクス
        print("\n[METRICS] Computing post-calibration metrics...")
        scaled_logits = logits / optimal_temp
        metrics_after = self.compute_metrics(scaled_logits, labels)
        print(f"Accuracy: {metrics_after['accuracy']:.4f}")
        print(f"ECE: {metrics_after['ece']:.6f}")
        print(f"Brier Score: {metrics_after['brier_score']:.6f}")
        print(f"Mean Confidence: {metrics_after['confidence_mean']:.4f}")
        
        # 改善率計算
        ece_improvement = (metrics_before['ece'] - metrics_after['ece']) / metrics_before['ece'] * 100
        brier_improvement = (metrics_before['brier_score'] - metrics_after['brier_score']) / metrics_before['brier_score'] * 100
        
        print(f"\n[IMPROVEMENT]")
        print(f"ECE reduction: {ece_improvement:.2f}%")
        print(f"Brier reduction: {brier_improvement:.2f}%")
        
        # 結果構築
        result = CalibrationResult(
            optimal_temperature=optimal_temp,
            ece_before=metrics_before['ece'],
            ece_after=metrics_after['ece'],
            brier_score_before=metrics_before['brier_score'],
            brier_score_after=metrics_after['brier_score'],
            accuracy=metrics_after['accuracy'],
            confidence_mean=metrics_after['confidence_mean'],
            calibration_curve=metrics_after['curve_data']
        )
        
        print(f"\n{'='*60}")
        print(f"[OK] Temperature calibration completed!")
        print(f"Optimal temperature: {optimal_temp:.4f}")
        print(f"{'='*60}\n")
        
        return result
    
    def save_calibrated_temperature(self, temperature: float, output_path: str):
        """較正温度保存"""
        config = {
            'temperature': temperature,
            'calibration_time': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
            'n_bins': self.n_bins
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[OK] Calibrated temperature saved to {output_path}")


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """NumPy softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def visualize_calibration_curve(result: CalibrationResult, save_path: str = None):
    """較正曲線可視化（オプション）"""
    try:
        import matplotlib.pyplot as plt
        
        curve = result.calibration_curve
        confidences = curve['confidences']
        accuracies = curve['accuracies']
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(confidences, accuracies, 'o-', label='Model calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration Curve (ECE: {result.ece_after:.4f})')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Calibration curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    except ImportError:
        print("[WARNING] matplotlib not available, skipping visualization")


# [OK] テスト用
def test_calibration():
    """較正テスト"""
    print("\n[TEST] Testing temperature calibration...")
    
    # ダミーデータ
    n_samples = 1000
    n_classes = 100
    
    # 過確信なロジット生成
    logits = np.random.randn(n_samples, n_classes) * 3.0  # 過確信
    labels = np.random.randint(0, n_classes, n_samples)
    
    # ECE計算
    ece_calc = ExpectedCalibrationError(n_bins=15)
    probs = softmax(logits, axis=1)
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    ece_before, _ = ece_calc.compute(confidences, predictions, labels)
    print(f"ECE before: {ece_before:.6f}")
    
    # 温度スケーリング（T=2.0で較正）
    optimal_temp = 2.0
    scaled_logits = logits / optimal_temp
    probs_after = softmax(scaled_logits, axis=1)
    predictions_after = np.argmax(probs_after, axis=1)
    confidences_after = np.max(probs_after, axis=1)
    
    ece_after, _ = ece_calc.compute(confidences_after, predictions_after, labels)
    print(f"ECE after (T={optimal_temp}): {ece_after:.6f}")
    print(f"Improvement: {(ece_before - ece_after) / ece_before * 100:.2f}%")
    
    print("\n[OK] Calibration test passed!")


if __name__ == "__main__":
    test_calibration()
