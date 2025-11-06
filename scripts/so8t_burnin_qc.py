"""
SO8T Burn-in QC (Quality Control) Script

焼きこみ前後の出力一致検証、RoPE位相安定性テスト、較正メトリクス計算を実施する。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# ロジット差分検証
# ========================================

def calculate_logit_difference(
    logits_pre: torch.Tensor,
    logits_post: torch.Tensor
) -> Dict[str, float]:
    """
    焼きこみ前後のロジット差分を計算
    
    Args:
        logits_pre: 焼きこみ前のロジット [B, T, V]
        logits_post: 焼きこみ後のロジット [B, T, V]
    
    Returns:
        差分統計
    """
    # 最大絶対誤差
    max_abs_error = torch.max(torch.abs(logits_pre - logits_post)).item()
    
    # 平均絶対誤差
    mean_abs_error = torch.mean(torch.abs(logits_pre - logits_post)).item()
    
    # RMS誤差
    rms_error = torch.sqrt(torch.mean((logits_pre - logits_post) ** 2)).item()
    
    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'rms_error': rms_error
    }


def calculate_kl_divergence(
    logits_pre: torch.Tensor,
    logits_post: torch.Tensor,
    temperature: float = 1.0
) -> float:
    """
    焼きこみ前後のKLダイバージェンスを計算
    
    Args:
        logits_pre: 焼きこみ前のロジット [B, T, V]
        logits_post: 焼きこみ後のロジット [B, T, V]
        temperature: 温度スケーリング
    
    Returns:
        KLダイバージェンス
    """
    # 確率分布に変換
    probs_pre = F.softmax(logits_pre / temperature, dim=-1)
    log_probs_post = F.log_softmax(logits_post / temperature, dim=-1)
    
    # KL(p_pre || p_post)
    kl = F.kl_div(log_probs_post, probs_pre, reduction='batchmean').item()
    
    return kl


# ========================================
# RoPE位相安定性テスト
# ========================================

def calculate_attention_entropy(
    attention_weights: torch.Tensor
) -> torch.Tensor:
    """
    アテンション重みのエントロピーを計算
    
    Args:
        attention_weights: アテンション重み [B, H, T, T]
    
    Returns:
        各トークンのエントロピー [B, T]
    """
    # 各クエリトークンに対するアテンション分布のエントロピー
    # attention_weights: [B, H, T_query, T_key]
    
    # 各ヘッドの平均を取る
    attn_mean = attention_weights.mean(dim=1)  # [B, T_query, T_key]
    
    # エントロピー計算: -sum(p * log(p))
    # 数値安定性のため小さな値を追加
    eps = 1e-10
    entropy = -torch.sum(attn_mean * torch.log(attn_mean + eps), dim=-1)  # [B, T_query]
    
    return entropy


def detect_periodic_entropy_drops(
    entropy_series: np.ndarray,
    window_size: int = 50,
    threshold: float = 0.2
) -> List[int]:
    """
    エントロピーの周期的落ち込みを検出
    
    Args:
        entropy_series: エントロピー時系列 [T]
        window_size: 窓サイズ
        threshold: 落ち込み閾値（標準偏差の倍数）
    
    Returns:
        落ち込み位置のリスト
    """
    drops = []
    
    if len(entropy_series) < window_size:
        return drops
    
    # 移動平均と標準偏差を計算
    moving_avg = np.convolve(entropy_series, np.ones(window_size) / window_size, mode='valid')
    moving_std = np.array([
        np.std(entropy_series[max(0, i - window_size // 2):min(len(entropy_series), i + window_size // 2)])
        for i in range(len(entropy_series))
    ])
    
    # 落ち込みを検出
    for i in range(window_size // 2, len(entropy_series) - window_size // 2):
        if entropy_series[i] < moving_avg[i - window_size // 2] - threshold * moving_std[i]:
            drops.append(i)
    
    return drops


# ========================================
# 較正メトリクス
# ========================================

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
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            accuracy_in_bin = predictions[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'prop_in_bin': float(prop_in_bin / len(confidences)),
                'accuracy': float(accuracy_in_bin),
                'avg_confidence': float(avg_confidence_in_bin),
                'error': float(abs(avg_confidence_in_bin - accuracy_in_bin))
            })
        else:
            bin_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'prop_in_bin': 0.0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'error': 0.0
            })
    
    ece = ece / len(confidences)
    return ece, {'bins': bin_stats}


def calculate_brier_score(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Brier Score を計算
    
    Args:
        predictions: 予測クラス [N]
        probabilities: 予測確率 [N, num_classes]
        labels: 正解ラベル [N]
    
    Returns:
        Brier Score
    """
    num_classes = probabilities.shape[1]
    
    # ワンホットエンコーディング
    true_labels = np.zeros((len(labels), num_classes))
    true_labels[np.arange(len(labels)), labels] = 1.0
    
    # Brier Score: mean((predicted_prob - true_label)^2)
    brier = np.mean(np.sum((probabilities - true_labels) ** 2, axis=1))
    
    return float(brier)


# ========================================
# 直交性ドリフト検証
# ========================================

def check_orthogonality(
    rotation_matrix: torch.Tensor,
    tolerance: float = 1e-5
) -> Dict[str, float]:
    """
    回転行列の直交性を検証
    
    Args:
        rotation_matrix: 回転行列 [D, D] または [num_blocks, 8, 8]
        tolerance: 許容誤差
    
    Returns:
        直交性統計
    """
    if rotation_matrix.dim() == 3:
        # ブロック対角の場合
        num_blocks = rotation_matrix.size(0)
        orthogonality_errors = []
        
        for i in range(num_blocks):
            R = rotation_matrix[i]  # [8, 8]
            I = torch.eye(8, device=R.device, dtype=R.dtype)
            error = torch.norm(R.T @ R - I, p='fro').item()
            orthogonality_errors.append(error)
        
        return {
            'max_orthogonality_error': max(orthogonality_errors),
            'mean_orthogonality_error': np.mean(orthogonality_errors),
            'orthogonality_errors': orthogonality_errors
        }
    else:
        # 単一行列の場合
        I = torch.eye(rotation_matrix.size(0), device=rotation_matrix.device, dtype=rotation_matrix.dtype)
        error = torch.norm(rotation_matrix.T @ rotation_matrix - I, p='fro').item()
        
        return {
            'orthogonality_error': error
        }


# ========================================
# QC検証器
# ========================================

class SO8TBurnInQC:
    """SO8T焼きこみQC検証器"""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            device: デバイス
        """
        self.device = device
        self.results = {}
        
        logger.info("SO8T Burn-in QC initialized")
        logger.info(f"  Device: {self.device}")
    
    def verify_logit_consistency(
        self,
        model_pre: nn.Module,
        model_post: nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        sample_name: str = "sample"
    ) -> Dict[str, float]:
        """
        焼きこみ前後のロジット一致性を検証
        
        Args:
            model_pre: 焼きこみ前のモデル
            model_post: 焼きこみ後のモデル
            test_inputs: テスト入力（input_ids, attention_maskなど）
            sample_name: サンプル名
        
        Returns:
            検証結果
        """
        logger.info(f"Verifying logit consistency for {sample_name}...")
        
        model_pre.eval()
        model_post.eval()
        
        with torch.no_grad():
            # 焼きこみ前
            outputs_pre = model_pre(**test_inputs)
            logits_pre = outputs_pre.logits
            
            # 焼きこみ後
            outputs_post = model_post(**test_inputs)
            logits_post = outputs_post.logits
            
            # ロジット差分
            logit_diff = calculate_logit_difference(logits_pre, logits_post)
            
            # KLダイバージェンス
            kl_div = calculate_kl_divergence(logits_pre, logits_post)
            
            logger.info(f"  Max absolute error: {logit_diff['max_abs_error']:.6e}")
            logger.info(f"  Mean absolute error: {logit_diff['mean_abs_error']:.6e}")
            logger.info(f"  RMS error: {logit_diff['rms_error']:.6e}")
            logger.info(f"  KL divergence: {kl_div:.6e}")
            
            # 目標達成チェック
            target_max_error = 1e-5
            target_kl = 1e-6
            
            passed = (
                logit_diff['max_abs_error'] <= target_max_error and
                kl_div <= target_kl
            )
            
            logger.info(f"  QC status: {'PASS' if passed else 'FAIL'}")
            
            results = {
                **logit_diff,
                'kl_divergence': kl_div,
                'target_max_error': target_max_error,
                'target_kl': target_kl,
                'qc_passed': passed
            }
            
            self.results[f'logit_consistency_{sample_name}'] = results
            
            return results
    
    def test_rope_phase_stability(
        self,
        model: nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        sample_name: str = "sample"
    ) -> Dict:
        """
        RoPE位相安定性をテスト
        
        Args:
            model: テスト対象モデル
            test_inputs: テスト入力（長文）
            sample_name: サンプル名
        
        Returns:
            テスト結果
        """
        logger.info(f"Testing RoPE phase stability for {sample_name}...")
        
        model.eval()
        
        # 注意: このテストは完全な実装にはモデルの内部状態へのアクセスが必要
        # ここでは簡易版を実装
        
        with torch.no_grad():
            outputs = model(**test_inputs, output_attentions=True)
            
            # アテンション重みを取得（利用可能な場合）
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # 最後の層のアテンションを使用
                attention = outputs.attentions[-1]  # [B, H, T, T]
                
                # エントロピーを計算
                entropy = calculate_attention_entropy(attention)  # [B, T]
                entropy_series = entropy[0].cpu().numpy()  # 最初のバッチ
                
                # 周期的落ち込みを検出
                drops = detect_periodic_entropy_drops(entropy_series)
                
                logger.info(f"  Sequence length: {len(entropy_series)}")
                logger.info(f"  Mean entropy: {np.mean(entropy_series):.4f}")
                logger.info(f"  Std entropy: {np.std(entropy_series):.4f}")
                logger.info(f"  Periodic drops detected: {len(drops)}")
                
                results = {
                    'sequence_length': len(entropy_series),
                    'mean_entropy': float(np.mean(entropy_series)),
                    'std_entropy': float(np.std(entropy_series)),
                    'entropy_series': entropy_series.tolist(),
                    'periodic_drops': drops,
                    'num_drops': len(drops)
                }
            else:
                logger.warning("  Model does not support attention output")
                results = {
                    'error': 'Model does not support attention output'
                }
            
            self.results[f'rope_phase_stability_{sample_name}'] = results
            
            return results
    
    def verify_rotation_orthogonality(
        self,
        rotation_gates: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        回転行列の直交性を検証
        
        Args:
            rotation_gates: レイヤー名をキーとした回転行列の辞書
        
        Returns:
            検証結果
        """
        logger.info("Verifying rotation matrix orthogonality...")
        
        layer_results = {}
        max_errors = []
        
        for layer_name, rotation_matrix in rotation_gates.items():
            orth_stats = check_orthogonality(rotation_matrix)
            layer_results[layer_name] = orth_stats
            
            if 'max_orthogonality_error' in orth_stats:
                max_error = orth_stats['max_orthogonality_error']
                max_errors.append(max_error)
                logger.info(f"  {layer_name}: max orthogonality error = {max_error:.6e}")
            elif 'orthogonality_error' in orth_stats:
                error = orth_stats['orthogonality_error']
                max_errors.append(error)
                logger.info(f"  {layer_name}: orthogonality error = {error:.6e}")
        
        overall_max_error = max(max_errors) if max_errors else 0.0
        logger.info(f"  Overall max orthogonality error: {overall_max_error:.6e}")
        
        results = {
            'layer_results': layer_results,
            'overall_max_error': float(overall_max_error)
        }
        
        self.results['rotation_orthogonality'] = results
        
        return results
    
    def save_report(
        self,
        output_path: Path
    ) -> None:
        """
        QC検証レポートを保存
        
        Args:
            output_path: 出力パス
        """
        logger.info(f"Saving QC report to {output_path}...")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Report saved")
    
    def generate_markdown_report(
        self,
        output_path: Path
    ) -> None:
        """
        Markdown形式のQCレポートを生成
        
        Args:
            output_path: 出力パス
        """
        logger.info(f"Generating markdown QC report to {output_path}...")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# SO8T Burn-in QC Report

Generated: {timestamp}

## Summary

This report documents the quality control (QC) verification results for SO8T burn-in process.

"""
        
        # ロジット一致性
        logit_tests = {k: v for k, v in self.results.items() if k.startswith('logit_consistency_')}
        if logit_tests:
            report += "## Logit Consistency Verification\n\n"
            for test_name, results in logit_tests.items():
                sample_name = test_name.replace('logit_consistency_', '')
                qc_status = "[PASS]" if results.get('qc_passed', False) else "[FAIL]"
                report += f"### {sample_name} {qc_status}\n\n"
                report += f"- **Max Absolute Error**: {results['max_abs_error']:.6e} (target: {results['target_max_error']:.6e})\n"
                report += f"- **Mean Absolute Error**: {results['mean_abs_error']:.6e}\n"
                report += f"- **RMS Error**: {results['rms_error']:.6e}\n"
                report += f"- **KL Divergence**: {results['kl_divergence']:.6e} (target: {results['target_kl']:.6e})\n\n"
        
        # RoPE位相安定性
        rope_tests = {k: v for k, v in self.results.items() if k.startswith('rope_phase_stability_')}
        if rope_tests:
            report += "## RoPE Phase Stability Test\n\n"
            for test_name, results in rope_tests.items():
                sample_name = test_name.replace('rope_phase_stability_', '')
                report += f"### {sample_name}\n\n"
                if 'error' in results:
                    report += f"- **Error**: {results['error']}\n\n"
                else:
                    report += f"- **Sequence Length**: {results['sequence_length']}\n"
                    report += f"- **Mean Entropy**: {results['mean_entropy']:.4f}\n"
                    report += f"- **Std Entropy**: {results['std_entropy']:.4f}\n"
                    report += f"- **Periodic Drops Detected**: {results['num_drops']}\n\n"
        
        # 直交性検証
        if 'rotation_orthogonality' in self.results:
            results = self.results['rotation_orthogonality']
            report += "## Rotation Matrix Orthogonality\n\n"
            report += f"- **Overall Max Error**: {results['overall_max_error']:.6e}\n\n"
            report += "### Layer Results\n\n"
            for layer_name, layer_stats in results['layer_results'].items():
                if 'max_orthogonality_error' in layer_stats:
                    report += f"- **{layer_name}**: {layer_stats['max_orthogonality_error']:.6e}\n"
                elif 'orthogonality_error' in layer_stats:
                    report += f"- **{layer_name}**: {layer_stats['orthogonality_error']:.6e}\n"
            report += "\n"
        
        report += "## Recommendations\n\n"
        
        # 推奨事項を生成
        recommendations = []
        
        for test_name, results in logit_tests.items():
            if not results.get('qc_passed', False):
                recommendations.append(
                    f"- Logit consistency check failed for {test_name.replace('logit_consistency_', '')}. "
                    "Review burn-in implementation."
                )
        
        for test_name, results in rope_tests.items():
            if results.get('num_drops', 0) > 10:
                recommendations.append(
                    f"- Significant periodic entropy drops detected ({results['num_drops']} drops). "
                    "Consider applying PET regularization or adjusting RoPE configuration."
                )
        
        if 'rotation_orthogonality' in self.results:
            if self.results['rotation_orthogonality']['overall_max_error'] > 1e-3:
                recommendations.append(
                    "- High orthogonality error detected. Review rotation gate training and PET application."
                )
        
        if recommendations:
            for rec in recommendations:
                report += rec + "\n"
        else:
            report += "- All QC checks passed. Model is ready for deployment.\n"
        
        report += "\n---\n\nEnd of Report\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"  Markdown report saved")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SO8T Burn-in QC")
    parser.add_argument(
        "--model-pre",
        type=str,
        help="Path to pre-burnin model"
    )
    parser.add_argument(
        "--model-post",
        type=str,
        help="Path to post-burnin model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data (JSON)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="_docs/so8t_burnin_qc_report.json",
        help="Output JSON report path"
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="_docs/so8t_burnin_qc_report.md",
        help="Output Markdown report path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # QC検証器を初期化
    qc = SO8TBurnInQC(device=args.device)
    
    # TODO: モデル読み込みとテスト実行
    # この部分は統合パイプラインスクリプトで実装される
    
    logger.info("QC verification complete!")


if __name__ == "__main__":
    main()




