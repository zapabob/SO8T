"""
SO8T Long Text Regression Test

長文でのRoPE位相ドリフト、発振（ギザつき）、エントロピー安定性を検証する。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # バックエンドをAggに設定
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# 長文テストケース
# ========================================

LONG_TEXT_TEST_CASES = [
    {
        "name": "scientific_explanation",
        "prompt": """Provide a comprehensive explanation of the Standard Model of particle physics. 
Include detailed discussions of:
1. The fundamental fermions (quarks and leptons) and their properties
2. The gauge bosons (photon, W, Z, gluons) and their role in mediating forces
3. The Higgs mechanism and mass generation
4. Quantum chromodynamics (QCD) and the strong force
5. Electroweak unification and symmetry breaking
6. Known limitations of the Standard Model
7. Experimental evidence supporting each component
8. Mathematical formalism using group theory (SU(3) × SU(2) × U(1))
9. Open questions and beyond-Standard-Model physics
10. Connections to cosmology and astrophysics

Explain each point with mathematical rigor while maintaining clarity for advanced undergraduate level.""",
        "expected_length": 2048,
        "category": "theoretical_physics"
    },
    {
        "name": "mathematical_proof",
        "prompt": """Prove Fermat's Last Theorem using a step-by-step approach. While the full Wiles proof is extensive, provide:

1. Historical context and why the theorem was important
2. The statement of the theorem: x^n + y^n = z^n has no non-zero integer solutions for n > 2
3. Proof for n=3 (Euler's proof) in detail
4. Proof for n=4 (Fermat's infinite descent) in detail
5. Overview of the Taniyama-Shimura conjecture
6. How elliptic curves relate to Fermat's Last Theorem
7. The concept of modular forms and their connection
8. Galois representations and their role
9. Key steps in Wiles' proof strategy
10. Significance of the proof in modern mathematics
11. Related problems and generalizations
12. Applications in cryptography and coding theory

Include specific mathematical notation, lemmas, and corollaries where appropriate.""",
        "expected_length": 2048,
        "category": "pure_mathematics"
    },
    {
        "name": "technical_implementation",
        "prompt": """Design and explain a complete implementation of a transformer-based language model with the following requirements:

1. Architecture Overview:
   - Multi-head self-attention mechanism with 12 heads
   - Hidden dimension of 768
   - 12 transformer layers
   - Feedforward dimension of 3072
   - Layer normalization and residual connections
   - Positional encoding (both absolute and relative)

2. Attention Mechanism Details:
   - Scaled dot-product attention formula
   - Query, Key, Value projections
   - Attention mask implementation
   - Dropout for regularization
   - Attention pattern analysis

3. Position Encoding:
   - Sinusoidal position encoding derivation
   - Learned position embeddings
   - Rotary Position Embedding (RoPE) implementation
   - ALiBi (Attention with Linear Biases)
   - Comparison of methods

4. Training Strategy:
   - Loss function (cross-entropy)
   - Optimization algorithm (AdamW)
   - Learning rate scheduling (cosine with warmup)
   - Gradient clipping
   - Mixed precision training

5. Advanced Techniques:
   - Flash Attention for efficiency
   - Gradient checkpointing
   - Dynamic batching
   - KV cache for inference
   - Quantization strategies

6. Implementation Code:
   - PyTorch implementation of core components
   - Forward pass logic
   - Backward pass considerations
   - Memory optimization techniques

7. Evaluation Metrics:
   - Perplexity calculation
   - BLEU score for generation
   - Token accuracy
   - Convergence analysis

Include pseudocode, mathematical formulations, and optimization considerations.""",
        "expected_length": 2560,
        "category": "machine_learning"
    }
]


# ========================================
# 発振検出
# ========================================

def detect_oscillation(
    logits: torch.Tensor,
    window_size: int = 10,
    threshold: float = 2.0
) -> Dict:
    """
    ロジット分布の発振（ギザつき）を検出
    
    Args:
        logits: ロジット系列 [T, V]
        window_size: 窓サイズ
        threshold: 閾値（標準偏差の倍数）
    
    Returns:
        発振検出結果
    """
    # 各トークンの最大ロジット値の時系列
    max_logits = logits.max(dim=-1).values.cpu().numpy()
    
    # 一次差分（変化率）
    diff = np.diff(max_logits)
    
    # 二次差分（加速度）
    diff2 = np.diff(diff)
    
    # 発振の指標: 二次差分の絶対値の平均
    oscillation_index = np.mean(np.abs(diff2))
    
    # 移動標準偏差
    moving_std = np.array([
        np.std(max_logits[max(0, i - window_size):min(len(max_logits), i + window_size)])
        for i in range(len(max_logits))
    ])
    
    # 異常な変動を検出
    high_variance_points = np.where(moving_std > threshold * np.mean(moving_std))[0]
    
    return {
        'oscillation_index': float(oscillation_index),
        'mean_max_logit': float(np.mean(max_logits)),
        'std_max_logit': float(np.std(max_logits)),
        'max_diff': float(np.max(np.abs(diff))),
        'max_diff2': float(np.max(np.abs(diff2))),
        'num_high_variance_points': len(high_variance_points),
        'high_variance_ratio': float(len(high_variance_points) / len(max_logits)),
        'max_logits_series': max_logits.tolist(),
        'diff_series': diff.tolist(),
        'diff2_series': diff2.tolist()
    }


def visualize_oscillation(
    oscillation_data: Dict,
    output_path: Path,
    title: str = "Logit Oscillation Analysis"
):
    """
    発振データを可視化
    
    Args:
        oscillation_data: 発振検出結果
        output_path: 出力パス
        title: グラフタイトル
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 最大ロジット値の時系列
    ax1 = axes[0]
    max_logits = oscillation_data['max_logits_series']
    ax1.plot(max_logits, label='Max Logits', linewidth=1)
    ax1.set_ylabel('Max Logit Value')
    ax1.set_title(f'{title} - Max Logits Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 一次差分
    ax2 = axes[1]
    diff = oscillation_data['diff_series']
    ax2.plot(diff, label='First Difference', color='orange', linewidth=1)
    ax2.set_ylabel('First Difference')
    ax2.set_title('Change Rate (First Derivative)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 二次差分
    ax3 = axes[2]
    diff2 = oscillation_data['diff2_series']
    ax3.plot(diff2, label='Second Difference', color='red', linewidth=1)
    ax3.set_ylabel('Second Difference')
    ax3.set_xlabel('Token Position')
    ax3.set_title('Acceleration (Second Derivative)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ========================================
# エントロピー分析
# ========================================

def analyze_entropy_stability(
    logits: torch.Tensor
) -> Dict:
    """
    エントロピーの安定性を分析
    
    Args:
        logits: ロジット系列 [T, V]
    
    Returns:
        エントロピー分析結果
    """
    # 確率分布に変換
    probs = torch.softmax(logits, dim=-1)
    
    # エントロピーを計算
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    entropy_series = entropy.cpu().numpy()
    
    # 統計量
    mean_entropy = np.mean(entropy_series)
    std_entropy = np.std(entropy_series)
    min_entropy = np.min(entropy_series)
    max_entropy = np.max(entropy_series)
    
    # エントロピーの変動係数（CV）
    cv = std_entropy / (mean_entropy + 1e-10)
    
    # 時系列での変化を検出
    entropy_diff = np.abs(np.diff(entropy_series))
    mean_change = np.mean(entropy_diff)
    max_change = np.max(entropy_diff)
    
    return {
        'mean_entropy': float(mean_entropy),
        'std_entropy': float(std_entropy),
        'min_entropy': float(min_entropy),
        'max_entropy': float(max_entropy),
        'cv_entropy': float(cv),
        'mean_entropy_change': float(mean_change),
        'max_entropy_change': float(max_change),
        'entropy_series': entropy_series.tolist()
    }


def visualize_entropy(
    entropy_data: Dict,
    output_path: Path,
    title: str = "Entropy Stability Analysis"
):
    """
    エントロピーデータを可視化
    
    Args:
        entropy_data: エントロピー分析結果
        output_path: 出力パス
        title: グラフタイトル
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # エントロピー時系列
    ax1 = axes[0]
    entropy_series = entropy_data['entropy_series']
    ax1.plot(entropy_series, label='Entropy', linewidth=1)
    ax1.axhline(y=entropy_data['mean_entropy'], color='r', linestyle='--', 
                label=f"Mean: {entropy_data['mean_entropy']:.3f}", alpha=0.7)
    ax1.fill_between(
        range(len(entropy_series)),
        entropy_data['mean_entropy'] - entropy_data['std_entropy'],
        entropy_data['mean_entropy'] + entropy_data['std_entropy'],
        alpha=0.2, color='r', label=f"±1 Std: {entropy_data['std_entropy']:.3f}"
    )
    ax1.set_ylabel('Entropy')
    ax1.set_title(f'{title} - Entropy Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # エントロピー変化率
    ax2 = axes[1]
    entropy_diff = np.abs(np.diff(entropy_series))
    ax2.plot(entropy_diff, label='Absolute Change', color='orange', linewidth=1)
    ax2.axhline(y=entropy_data['mean_entropy_change'], color='r', linestyle='--',
                label=f"Mean Change: {entropy_data['mean_entropy_change']:.3f}", alpha=0.7)
    ax2.set_ylabel('Absolute Entropy Change')
    ax2.set_xlabel('Token Position')
    ax2.set_title('Entropy Change Rate')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ========================================
# 長文回帰テスト
# ========================================

class SO8TLongTextRegressionTest:
    """SO8T長文回帰テスト"""
    
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
        
        logger.info("SO8T Long Text Regression Test initialized")
        logger.info(f"  Device: {self.device}")
    
    def run_test(
        self,
        model: nn.Module,
        tokenizer,
        test_case: Dict,
        max_new_tokens: int = 512
    ) -> Dict:
        """
        長文テストケースを実行
        
        Args:
            model: テスト対象モデル
            tokenizer: トークナイザー
            test_case: テストケース
            max_new_tokens: 最大生成トークン数
        
        Returns:
            テスト結果
        """
        test_name = test_case['name']
        prompt = test_case['prompt']
        
        logger.info(f"Running long text test: {test_name}")
        logger.info(f"  Prompt length: {len(prompt)} chars")
        
        model.eval()
        
        # 入力をトークン化
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False  # 長文を切り詰めない
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        input_length = input_ids.size(1)
        logger.info(f"  Input tokens: {input_length}")
        
        # 生成実行
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # 決定的生成
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
                )
                
                generated_ids = outputs.sequences
                scores = outputs.scores  # List of [B, V] tensors
                
                # スコアをロジットに変換
                logits = torch.stack(scores, dim=1)  # [B, T_gen, V]
                logits = logits[0]  # 最初のバッチ [T_gen, V]
                
                logger.info(f"  Generated tokens: {logits.size(0)}")
                
                # 発振検出
                oscillation_results = detect_oscillation(logits)
                logger.info(f"  Oscillation index: {oscillation_results['oscillation_index']:.6f}")
                logger.info(f"  High variance ratio: {oscillation_results['high_variance_ratio']:.3f}")
                
                # エントロピー分析
                entropy_results = analyze_entropy_stability(logits)
                logger.info(f"  Mean entropy: {entropy_results['mean_entropy']:.4f}")
                logger.info(f"  CV entropy: {entropy_results['cv_entropy']:.4f}")
                
                # 生成テキストをデコード
                generated_text = tokenizer.decode(
                    generated_ids[0][input_length:],
                    skip_special_tokens=True
                )
                
                results = {
                    'test_name': test_name,
                    'input_length': input_length,
                    'generated_length': logits.size(0),
                    'total_length': input_length + logits.size(0),
                    'oscillation': oscillation_results,
                    'entropy': entropy_results,
                    'generated_text': generated_text,
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"  Test failed: {e}")
                results = {
                    'test_name': test_name,
                    'error': str(e),
                    'success': False
                }
        
        self.results[test_name] = results
        return results
    
    def run_comparison(
        self,
        model_pre: nn.Module,
        model_post: nn.Module,
        tokenizer,
        test_case: Dict,
        max_new_tokens: int = 512
    ) -> Dict:
        """
        焼きこみ前後の比較テストを実行
        
        Args:
            model_pre: 焼きこみ前のモデル
            model_post: 焼きこみ後のモデル
            tokenizer: トークナイザー
            test_case: テストケース
            max_new_tokens: 最大生成トークン数
        
        Returns:
            比較結果
        """
        test_name = test_case['name']
        logger.info(f"Running comparison test: {test_name}")
        
        # 焼きこみ前
        logger.info("  Testing pre-burnin model...")
        results_pre = self.run_test(model_pre, tokenizer, test_case, max_new_tokens)
        
        # 焼きこみ後
        logger.info("  Testing post-burnin model...")
        results_post = self.run_test(model_post, tokenizer, test_case, max_new_tokens)
        
        # 比較分析
        comparison = {
            'test_name': test_name,
            'pre': results_pre,
            'post': results_post
        }
        
        if results_pre['success'] and results_post['success']:
            # 発振の比較
            osc_diff = (
                results_post['oscillation']['oscillation_index'] -
                results_pre['oscillation']['oscillation_index']
            )
            
            # エントロピーの比較
            entropy_diff = (
                results_post['entropy']['cv_entropy'] -
                results_pre['entropy']['cv_entropy']
            )
            
            comparison['oscillation_difference'] = osc_diff
            comparison['entropy_cv_difference'] = entropy_diff
            
            logger.info(f"  Oscillation difference: {osc_diff:+.6f}")
            logger.info(f"  Entropy CV difference: {entropy_diff:+.6f}")
        
        self.results[f'comparison_{test_name}'] = comparison
        return comparison
    
    def visualize_results(
        self,
        output_dir: Path
    ):
        """
        テスト結果を可視化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        logger.info(f"Visualizing results to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for test_name, results in self.results.items():
            if test_name.startswith('comparison_'):
                continue
            
            if not results.get('success', False):
                continue
            
            # 発振の可視化
            osc_path = output_dir / f"{test_name}_oscillation.png"
            visualize_oscillation(
                results['oscillation'],
                osc_path,
                title=f"{test_name} - Oscillation"
            )
            logger.info(f"  Saved: {osc_path}")
            
            # エントロピーの可視化
            entropy_path = output_dir / f"{test_name}_entropy.png"
            visualize_entropy(
                results['entropy'],
                entropy_path,
                title=f"{test_name} - Entropy"
            )
            logger.info(f"  Saved: {entropy_path}")
    
    def save_report(
        self,
        output_path: Path
    ):
        """
        テストレポートを保存
        
        Args:
            output_path: 出力パス
        """
        logger.info(f"Saving test report to {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 結果をJSON形式で保存（generated_textは除外）
        results_for_json = {}
        for test_name, results in self.results.items():
            if isinstance(results, dict):
                results_copy = results.copy()
                if 'generated_text' in results_copy:
                    # テキストは長すぎるので最初の500文字のみ保存
                    results_copy['generated_text_preview'] = results_copy['generated_text'][:500]
                    del results_copy['generated_text']
                results_for_json[test_name] = results_copy
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        
        logger.info("  Report saved")
    
    def generate_markdown_report(
        self,
        output_path: Path
    ):
        """
        Markdown形式のレポートを生成
        
        Args:
            output_path: 出力パス
        """
        logger.info(f"Generating markdown report to {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# SO8T Long Text Regression Test Report

Generated: {timestamp}

## Summary

This report documents the long text regression test results for SO8T model.

"""
        
        # テスト結果
        for test_name, results in self.results.items():
            if test_name.startswith('comparison_'):
                continue
            
            if not results.get('success', False):
                report += f"### {test_name} [FAILED]\n\n"
                report += f"- **Error**: {results.get('error', 'Unknown error')}\n\n"
                continue
            
            report += f"### {test_name}\n\n"
            report += f"- **Input Length**: {results['input_length']} tokens\n"
            report += f"- **Generated Length**: {results['generated_length']} tokens\n"
            report += f"- **Total Length**: {results['total_length']} tokens\n\n"
            
            # 発振統計
            osc = results['oscillation']
            report += "#### Oscillation Analysis\n\n"
            report += f"- **Oscillation Index**: {osc['oscillation_index']:.6f}\n"
            report += f"- **High Variance Ratio**: {osc['high_variance_ratio']:.3%}\n"
            report += f"- **Max Difference**: {osc['max_diff']:.6f}\n"
            report += f"- **Max 2nd Difference**: {osc['max_diff2']:.6f}\n\n"
            
            # エントロピー統計
            ent = results['entropy']
            report += "#### Entropy Stability\n\n"
            report += f"- **Mean Entropy**: {ent['mean_entropy']:.4f}\n"
            report += f"- **Std Entropy**: {ent['std_entropy']:.4f}\n"
            report += f"- **CV (Coefficient of Variation)**: {ent['cv_entropy']:.4f}\n"
            report += f"- **Mean Entropy Change**: {ent['mean_entropy_change']:.4f}\n"
            report += f"- **Max Entropy Change**: {ent['max_entropy_change']:.4f}\n\n"
        
        # 比較テスト
        comparisons = {k: v for k, v in self.results.items() if k.startswith('comparison_')}
        if comparisons:
            report += "## Pre/Post Burn-in Comparison\n\n"
            for comp_name, comp_results in comparisons.items():
                test_name = comp_results['test_name']
                report += f"### {test_name}\n\n"
                
                if 'oscillation_difference' in comp_results:
                    report += f"- **Oscillation Index Difference**: {comp_results['oscillation_difference']:+.6f}\n"
                    report += f"- **Entropy CV Difference**: {comp_results['entropy_cv_difference']:+.6f}\n\n"
        
        report += "## Recommendations\n\n"
        
        # 推奨事項を生成
        recommendations = []
        
        for test_name, results in self.results.items():
            if test_name.startswith('comparison_') or not results.get('success', False):
                continue
            
            osc = results['oscillation']
            ent = results['entropy']
            
            if osc['oscillation_index'] > 0.1:
                recommendations.append(
                    f"- High oscillation detected in {test_name} (index: {osc['oscillation_index']:.6f}). "
                    "Consider applying PET regularization or reducing learning rate."
                )
            
            if ent['cv_entropy'] > 0.5:
                recommendations.append(
                    f"- High entropy variation in {test_name} (CV: {ent['cv_entropy']:.4f}). "
                    "Model predictions may be unstable for long sequences."
                )
            
            if osc['high_variance_ratio'] > 0.2:
                recommendations.append(
                    f"- Significant high-variance points in {test_name} ({osc['high_variance_ratio']:.1%}). "
                    "Check for RoPE phase drift or numerical instability."
                )
        
        if recommendations:
            for rec in recommendations:
                report += rec + "\n"
        else:
            report += "- All long text tests passed. Model is stable for extended sequences.\n"
        
        report += "\n---\n\nEnd of Report\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("  Markdown report saved")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SO8T Long Text Regression Test")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="_docs/longtext_regression",
        help="Output directory"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # テスト実行器を初期化
    tester = SO8TLongTextRegressionTest(device=args.device)
    
    # TODO: モデルとトークナイザーの読み込み
    # この部分は統合パイプラインスクリプトで実装される
    
    logger.info("Long text regression test complete!")


if __name__ == "__main__":
    main()







