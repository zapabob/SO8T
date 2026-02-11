"""
SO8T Temperature Scaling Calibration

温度スケーリングによる較正（ECE/Brier計算）を実装する。
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from tqdm import tqdm

# Transformers
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """温度スケーリングモジュール"""
    
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """温度でスケーリング"""
        return logits / self.temperature


def calculate_ece(
    predictions: np.ndarray,
    confidences: np.ndarray,
    num_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Expected Calibration Error (ECE) を計算
    
    Args:
        predictions: 予測クラス [N]
        confidences: 予測確信度 [N]
        num_bins: ビン数
    
    Returns:
        (ECE値, 詳細統計)
    """
    # ビンに分割
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # このビンに含まれるサンプル
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            # ビン内の精度
            accuracy_in_bin = predictions[in_bin].mean()
            # ビン内の平均確信度
            avg_confidence_in_bin = confidences[in_bin].mean()
            # 重み付き誤差
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_stats.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'prop_in_bin': prop_in_bin / len(confidences),
                'accuracy': accuracy_in_bin,
                'avg_confidence': avg_confidence_in_bin,
                'error': abs(avg_confidence_in_bin - accuracy_in_bin)
            })
        else:
            bin_stats.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'prop_in_bin': 0.0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'error': 0.0
            })
    
    return ece, {'bins': bin_stats}


def calculate_brier_score(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int
) -> float:
    """
    Brier Score を計算
    
    Args:
        predictions: 予測クラス [N]
        probabilities: 予測確率 [N, num_classes]
        num_classes: クラス数
    
    Returns:
        Brier Score
    """
    # ワンホットエンコーディング
    true_labels = np.zeros((len(predictions), num_classes))
    true_labels[np.arange(len(predictions)), predictions] = 1.0
    
    # Brier Score: mean((predicted_prob - true_label)^2)
    brier = np.mean(np.sum((probabilities - true_labels) ** 2, axis=1))
    
    return brier


def calculate_nll(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Negative Log Likelihood を計算
    
    Args:
        logits: ロジット [N, num_classes]
        labels: 正解ラベル [N]
    
    Returns:
        NLL値
    """
    criterion = nn.CrossEntropyLoss()
    nll = criterion(logits, labels).item()
    return nll


class SO8TCalibrator:
    """SO8T較正器"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: モデル
            tokenizer: トークナイザー
            device: デバイス
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.eval()
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        
        self.temperature_scaler = None
        self.optimal_temp = 1.0
    
    def prepare_validation_data(
        self,
        validation_texts: List[str],
        max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        検証データを準備
        
        Args:
            validation_texts: 検証テキストリスト
            max_length: 最大長
        
        Returns:
            (input_ids, attention_mask)
        """
        inputs = self.tokenizer(
            validation_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        return input_ids, attention_mask
    
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        ロジットを取得
        
        Args:
            input_ids: 入力ID
            attention_mask: アテンションマスク
        
        Returns:
            ロジット [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
        
        return logits
    
    def optimize_temperature_nll(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        initial_temp: float = 1.0
    ) -> float:
        """
        NLLを最小化する温度を最適化
        
        Args:
            logits: ロジット [N, num_classes]
            labels: 正解ラベル [N]
            initial_temp: 初期温度
        
        Returns:
            最適温度
        """
        def objective(temp):
            temp_val = np.clip(temp[0], 0.1, 10.0)  # 温度は0.1-10.0の範囲
            scaled_logits = logits / temp_val
            nll = calculate_nll(scaled_logits, labels)
            return nll
        
        result = minimize(
            objective,
            x0=[initial_temp],
            method='L-BFGS-B',
            bounds=[(0.1, 10.0)]
        )
        
        optimal_temp = np.clip(result.x[0], 0.1, 10.0)
        return optimal_temp
    
    def optimize_temperature_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        initial_temp: float = 1.0
    ) -> float:
        """
        ECEを最小化する温度を最適化
        
        Args:
            logits: ロジット [N, num_classes]
            labels: 正解ラベル [N]
            initial_temp: 初期温度
        
        Returns:
            最適温度
        """
        def objective(temp):
            temp_val = np.clip(temp[0], 0.1, 10.0)
            scaled_logits = logits / temp_val
            probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()
            
            predictions = np.argmax(probs, axis=-1)
            confidences = np.max(probs, axis=-1)
            
            ece, _ = calculate_ece(
                predictions,
                confidences,
                num_bins=10
            )
            return ece
        
        result = minimize(
            objective,
            x0=[initial_temp],
            method='L-BFGS-B',
            bounds=[(0.1, 10.0)]
        )
        
        optimal_temp = np.clip(result.x[0], 0.1, 10.0)
        return optimal_temp
    
    def calibrate(
        self,
        validation_texts: List[str],
        validation_labels: Optional[List[int]] = None,
        optimization_method: str = "ece",
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        較正を実行
        
        Args:
            validation_texts: 検証テキストリスト
            validation_labels: 検証ラベル（オプション、生成タスクの場合は不要）
            optimization_method: 最適化方法 ("ece" or "nll")
            batch_size: バッチサイズ
        
        Returns:
            較正結果
        """
        logger.info("Starting calibration...")
        
        # 検証データ準備
        input_ids, attention_mask = self.prepare_validation_data(validation_texts)
        
        # ロジット取得
        logger.info("  Computing logits...")
        all_logits = []
        all_labels = []
        
        for i in tqdm(range(0, len(validation_texts), batch_size)):
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_mask = attention_mask[i:i+batch_size]
            
            batch_logits = self.get_logits(batch_input_ids, batch_attention_mask)
            
            # 最後のトークンのロジットを使用
            last_token_logits = batch_logits[:, -1, :]  # [batch_size, vocab_size]
            all_logits.append(last_token_logits.cpu())
            
            if validation_labels is not None:
                batch_labels = validation_labels[i:i+batch_size]
                all_labels.extend(batch_labels)
        
        logits = torch.cat(all_logits, dim=0)  # [N, vocab_size]
        
        # 較正前のメトリクス計算
        logger.info("  Computing pre-calibration metrics...")
        probs_before = torch.softmax(logits, dim=-1).numpy()
        predictions_before = np.argmax(probs_before, axis=-1)
        confidences_before = np.max(probs_before, axis=-1)
        
        ece_before, ece_details_before = calculate_ece(
            predictions_before,
            confidences_before,
            num_bins=10
        )
        
        vocab_size = logits.size(1)
        if validation_labels:
            labels_tensor = torch.tensor(all_labels)
            brier_before = calculate_brier_score(
                predictions_before,
                probs_before,
                vocab_size
            )
            nll_before = calculate_nll(logits, labels_tensor)
        else:
            # 生成タスクの場合は簡易的な評価
            brier_before = 0.0
            nll_before = 0.0
        
        logger.info(f"  Pre-calibration:")
        logger.info(f"    ECE: {ece_before:.6f}")
        logger.info(f"    Brier Score: {brier_before:.6f}")
        logger.info(f"    NLL: {nll_before:.6f}")
        
        # 温度最適化
        logger.info(f"  Optimizing temperature (method: {optimization_method})...")
        
        if validation_labels:
            labels_tensor = labels_tensor.to(self.device)
            logits_gpu = logits.to(self.device)
            
            if optimization_method == "nll":
                optimal_temp = self.optimize_temperature_nll(
                    logits_gpu,
                    labels_tensor,
                    initial_temp=1.0
                )
            else:  # ece
                optimal_temp = self.optimize_temperature_ece(
                    logits_gpu,
                    labels_tensor,
                    initial_temp=1.0
                )
        else:
            # ラベルがない場合はECEのみで最適化
            optimal_temp = self.optimize_temperature_ece(
                logits.to(self.device),
                torch.zeros(logits.size(0), dtype=torch.long).to(self.device),
                initial_temp=1.0
            )
        
        self.optimal_temp = optimal_temp
        
        logger.info(f"  Optimal temperature: {optimal_temp:.6f}")
        
        # 較正後のメトリクス計算
        logger.info("  Computing post-calibration metrics...")
        scaled_logits = logits / optimal_temp
        probs_after = torch.softmax(scaled_logits, dim=-1).numpy()
        predictions_after = np.argmax(probs_after, axis=-1)
        confidences_after = np.max(probs_after, axis=-1)
        
        ece_after, ece_details_after = calculate_ece(
            predictions_after,
            confidences_after,
            num_bins=10
        )
        
        if validation_labels:
            brier_after = calculate_brier_score(
                predictions_after,
                probs_after,
                vocab_size
            )
            nll_after = calculate_nll(scaled_logits, labels_tensor)
        else:
            brier_after = 0.0
            nll_after = 0.0
        
        logger.info(f"  Post-calibration:")
        logger.info(f"    ECE: {ece_after:.6f}")
        logger.info(f"    Brier Score: {brier_after:.6f}")
        logger.info(f"    NLL: {nll_after:.6f}")
        
        # 改善率計算
        ece_improvement = ((ece_before - ece_after) / (ece_before + 1e-8)) * 100
        brier_improvement = ((brier_before - brier_after) / (brier_before + 1e-8)) * 100
        
        logger.info(f"  Improvement:")
        logger.info(f"    ECE: {ece_improvement:.2f}%")
        logger.info(f"    Brier Score: {brier_improvement:.2f}%")
        
        results = {
            'optimal_temperature': optimal_temp,
            'ece_before': ece_before,
            'ece_after': ece_after,
            'ece_improvement': ece_improvement,
            'brier_before': brier_before,
            'brier_after': brier_after,
            'brier_improvement': brier_improvement,
            'nll_before': nll_before,
            'nll_after': nll_after,
            'ece_details_before': ece_details_before,
            'ece_details_after': ece_details_after
        }
        
        return results
    
    def save_calibration_report(
        self,
        results: Dict,
        output_path: Path
    ) -> None:
        """較正レポートを保存"""
        logger.info(f"Saving calibration report to {output_path}...")
        
        report = {
            'calibration_results': {
                'optimal_temperature': float(results['optimal_temperature']),
                'ece': {
                    'before': float(results['ece_before']),
                    'after': float(results['ece_after']),
                    'improvement_percent': float(results['ece_improvement'])
                },
                'brier_score': {
                    'before': float(results['brier_before']),
                    'after': float(results['brier_after']),
                    'improvement_percent': float(results['brier_improvement'])
                },
                'nll': {
                    'before': float(results['nll_before']),
                    'after': float(results['nll_after'])
                }
            },
            'ece_details': {
                'before': results['ece_details_before'],
                'after': results['ece_details_after']
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Report saved")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SO8T Calibration")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        required=True,
        help="Path to validation data (JSON file with 'texts' and optionally 'labels')"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="_docs/so8t_burnin_calibration_report.json",
        help="Output report path"
    )
    parser.add_argument(
        "--optimization-method",
        type=str,
        default="ece",
        choices=["ece", "nll"],
        help="Optimization method"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # モデル読み込み
    logger.info("Loading model from %s...", args.model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # 検証データ読み込み
    logger.info("Loading validation data from %s...", args.validation_data)
    with open(args.validation_data, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    
    validation_texts = validation_data.get('texts', [])
    validation_labels = validation_data.get('labels', None)
    
    if not validation_texts:
        raise ValueError("validation_data must contain 'texts' field")
    
    logger.info("Loaded %d validation samples", len(validation_texts))
    
    # 較正実行
    calibrator = SO8TCalibrator(model, tokenizer, device=args.device)
    results = calibrator.calibrate(
        validation_texts,
        validation_labels,
        optimization_method=args.optimization_method,
        batch_size=args.batch_size
    )
    
    # レポート保存
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save_calibration_report(results, output_path)
    
    logger.info("Calibration complete!")
    logger.info("Optimal temperature: %.6f", results['optimal_temperature'])
    logger.info("ECE improvement: %.2f%%", results['ece_improvement'])
    logger.info("Brier improvement: %.2f%%", results['brier_improvement'])


if __name__ == "__main__":
    main()

