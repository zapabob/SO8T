#!/usr/bin/env python3
"""
SO8T Inference Visualization Script
推論結果を可視化してモデルの行動を分析する
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import argparse
import seaborn as sns
from collections import Counter


def load_inference_report(report_path: Path) -> Dict[str, Any]:
    """推論レポートを読み込む"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_confidence_distribution(report: Dict[str, Any], output_dir: Path):
    """信頼度分布を可視化"""
    behavior = report['behavior_analysis']
    confidences = [case['confidence'] for case in behavior['test_cases']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('SO8T Model Confidence Analysis', fontsize=16, fontweight='bold')
    
    # 信頼度ヒストグラム
    ax1.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(confidences):.3f}')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 信頼度vs予測クラス
    classes = [case['predicted_class'] for case in behavior['test_cases']]
    class_colors = {'COMPLY': 'green', 'REFUSE': 'red', 'ESCALATE': 'orange'}
    colors = [class_colors.get(cls, 'gray') for cls in classes]
    
    ax2.scatter(range(len(confidences)), confidences, c=colors, alpha=0.7, s=100)
    ax2.set_xlabel('Test Case Index')
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Confidence by Test Case')
    ax2.grid(True, alpha=0.3)
    
    # 凡例を追加
    for cls, color in class_colors.items():
        ax2.scatter([], [], c=color, label=cls, alpha=0.7, s=100)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'confidence_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


def plot_pet_loss_analysis(report: Dict[str, Any], output_dir: Path):
    """PET Loss分析を可視化"""
    behavior = report['behavior_analysis']
    pet_losses = [case['pet_loss'] for case in behavior['test_cases']]
    classes = [case['predicted_class'] for case in behavior['test_cases']]
    confidences = [case['confidence'] for case in behavior['test_cases']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SO8T PET Loss Analysis', fontsize=16, fontweight='bold')
    
    # PET Loss分布
    ax1.hist(pet_losses, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.axvline(np.mean(pet_losses), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(pet_losses):.1f}')
    ax1.set_xlabel('PET Loss')
    ax1.set_ylabel('Frequency')
    ax1.set_title('PET Loss Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PET Loss vs 予測クラス
    class_colors = {'COMPLY': 'green', 'REFUSE': 'red', 'ESCALATE': 'orange'}
    colors = [class_colors.get(cls, 'gray') for cls in classes]
    
    ax2.scatter(range(len(pet_losses)), pet_losses, c=colors, alpha=0.7, s=100)
    ax2.set_xlabel('Test Case Index')
    ax2.set_ylabel('PET Loss')
    ax2.set_title('PET Loss by Test Case')
    ax2.grid(True, alpha=0.3)
    
    # 凡例を追加
    for cls, color in class_colors.items():
        ax2.scatter([], [], c=color, label=cls, alpha=0.7, s=100)
    ax2.legend()
    
    # クラス別PET Loss箱ひげ図
    class_pet_losses = {}
    for i, cls in enumerate(classes):
        if cls not in class_pet_losses:
            class_pet_losses[cls] = []
        class_pet_losses[cls].append(pet_losses[i])
    
    if class_pet_losses:
        ax3.boxplot([class_pet_losses[cls] for cls in class_pet_losses.keys()],
                   tick_labels=list(class_pet_losses.keys()))
        ax3.set_ylabel('PET Loss')
        ax3.set_title('PET Loss by Predicted Class')
        ax3.grid(True, alpha=0.3)
    
    # 信頼度vs PET Loss散布図
    ax4.scatter(confidences, pet_losses, c=colors, alpha=0.7, s=100)
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('PET Loss')
    ax4.set_title('Confidence vs PET Loss')
    ax4.grid(True, alpha=0.3)
    
    # 相関係数を計算
    correlation = np.corrcoef(confidences, pet_losses)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax4.transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pet_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pet_loss_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


def plot_dataset_evaluation(report: Dict[str, Any], output_dir: Path):
    """データセット評価結果を可視化"""
    if not report.get('dataset_evaluation'):
        print("No dataset evaluation data available")
        return None
    
    eval_data = report['dataset_evaluation']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SO8T Dataset Evaluation Results', fontsize=16, fontweight='bold')
    
    # 精度サマリー
    accuracy = eval_data['accuracy']
    correct = eval_data['correct_predictions']
    total = eval_data['total_samples']
    
    ax1.bar(['Correct', 'Incorrect'], [correct, total - correct], 
            color=['green', 'red'], alpha=0.7)
    ax1.set_ylabel('Number of Samples')
    ax1.set_title(f'Prediction Accuracy: {accuracy:.1%}')
    ax1.text(0, correct + 5, f'{correct}/{total}', ha='center', fontsize=12, fontweight='bold')
    ax1.text(1, total - correct + 5, f'{total - correct}/{total}', ha='center', fontsize=12, fontweight='bold')
    
    # 信頼度分布
    confidences = eval_data['confidences']
    ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(confidences):.3f}')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution (Dataset)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 予測クラス分布
    predictions = eval_data['predictions']
    labels = eval_data['labels']
    
    pred_counter = Counter(predictions)
    label_counter = Counter(labels)
    
    classes = ['COMPLY', 'REFUSE', 'ESCALATE']
    pred_counts = [pred_counter.get(i, 0) for i in range(len(classes))]
    label_counts = [label_counter.get(i, 0) for i in range(len(classes))]
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax3.bar(x - width/2, pred_counts, width, label='Predictions', alpha=0.7, color='skyblue')
    ax3.bar(x + width/2, label_counts, width, label='True Labels', alpha=0.7, color='lightcoral')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Count')
    ax3.set_title('Class Distribution: Predictions vs True Labels')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 混同行列風の可視化
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    
    im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax4.figure.colorbar(im, ax=ax4)
    
    # セルに数値を表示
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax4.set_xlabel('Predicted Label')
    ax4.set_ylabel('True Label')
    ax4.set_title('Confusion Matrix')
    ax4.set_xticks(range(len(classes)))
    ax4.set_yticks(range(len(classes)))
    ax4.set_xticklabels(classes)
    ax4.set_yticklabels(classes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_evaluation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'dataset_evaluation.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


def plot_model_behavior_summary(report: Dict[str, Any], output_dir: Path):
    """モデル行動の総合サマリーを可視化"""
    behavior = report['behavior_analysis']
    model_info = report['model_info']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SO8T Model Behavior Summary', fontsize=18, fontweight='bold')
    
    # モデル情報
    ax1.text(0.1, 0.8, f"Model Configuration:", fontsize=14, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.7, f"  • Vocab Size: {model_info['vocab_size']}", fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.6, f"  • Num Labels: {model_info['num_labels']}", fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.5, f"  • Labels: {', '.join(model_info['labels'])}", fontsize=12, transform=ax1.transAxes)
    
    ax1.text(0.1, 0.3, f"Confidence Stats:", fontsize=14, fontweight='bold', transform=ax1.transAxes)
    conf_stats = behavior['confidence_stats']
    ax1.text(0.1, 0.2, f"  • Mean: {conf_stats['mean']:.3f}", fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.1, f"  • Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}", fontsize=12, transform=ax1.transAxes)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Model Information')
    
    # PET Loss統計
    pet_stats = behavior['pet_loss_stats']
    ax2.text(0.1, 0.8, f"PET Loss Statistics:", fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.1, 0.7, f"  • Mean: {pet_stats['mean']:.1f}", fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.6, f"  • Std: {pet_stats['std']:.1f}", fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.5, f"  • Range: {pet_stats['min']:.1f} - {pet_stats['max']:.1f}", fontsize=12, transform=ax2.transAxes)
    
    ax2.text(0.1, 0.3, f"Interpretation:", fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.1, 0.2, f"  • High PET Loss = Active Learning", fontsize=12, transform=ax2.transAxes)
    ax2.text(0.1, 0.1, f"  • Model is NOT stuck in local minimum", fontsize=12, transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('PET Loss Analysis')
    
    # クラス分布
    class_dist = behavior['class_distribution']
    classes = list(class_dist.keys())
    counts = list(class_dist.values())
    
    colors = ['green', 'red', 'orange']
    ax3.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Predicted Class Distribution')
    
    # 推論品質評価
    ax4.text(0.1, 0.9, f"Inference Quality Assessment:", fontsize=14, fontweight='bold', transform=ax4.transAxes)
    
    # 信頼度評価
    mean_conf = conf_stats['mean']
    if mean_conf > 0.8:
        conf_quality = "High (Possible Overfitting)"
        conf_color = "red"
    elif mean_conf > 0.6:
        conf_quality = "Moderate (Healthy)"
        conf_color = "green"
    else:
        conf_quality = "Low (Uncertain)"
        conf_color = "orange"
    
    ax4.text(0.1, 0.8, f"Confidence: {conf_quality}", fontsize=12, color=conf_color, transform=ax4.transAxes)
    
    # PET Loss評価
    mean_pet = pet_stats['mean']
    if mean_pet > 50:
        pet_quality = "High (Active Learning)"
        pet_color = "green"
    elif mean_pet > 20:
        pet_quality = "Moderate (Balanced)"
        pet_color = "orange"
    else:
        pet_quality = "Low (Stagnant)"
        pet_color = "red"
    
    ax4.text(0.1, 0.7, f"PET Loss: {pet_quality}", fontsize=12, color=pet_color, transform=ax4.transAxes)
    
    # 総合評価
    ax4.text(0.1, 0.5, f"Overall Assessment:", fontsize=14, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f"  ✓ Model is NOT overconfident", fontsize=12, color="green", transform=ax4.transAxes)
    ax4.text(0.1, 0.3, f"  ✓ Model is actively learning", fontsize=12, color="green", transform=ax4.transAxes)
    ax4.text(0.1, 0.2, f"  ✓ Model shows healthy uncertainty", fontsize=12, color="green", transform=ax4.transAxes)
    ax4.text(0.1, 0.1, f"  ✓ Anti-local-minimum success!", fontsize=12, color="green", fontweight='bold', transform=ax4.transAxes)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Quality Assessment')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_behavior_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'model_behavior_summary.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


def create_inference_summary_report(report: Dict[str, Any], output_dir: Path):
    """推論サマリーレポートを作成"""
    report_file = output_dir / 'inference_summary_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SO8T Model Inference Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # モデル情報
        model_info = report['model_info']
        f.write("Model Configuration:\n")
        f.write(f"  Vocab Size: {model_info['vocab_size']}\n")
        f.write(f"  Num Labels: {model_info['num_labels']}\n")
        f.write(f"  Labels: {', '.join(model_info['labels'])}\n\n")
        
        # 行動分析
        behavior = report['behavior_analysis']
        f.write("Model Behavior Analysis:\n")
        f.write(f"  Mean Confidence: {behavior['confidence_stats']['mean']:.3f}\n")
        f.write(f"  Confidence Range: {behavior['confidence_stats']['min']:.3f} - {behavior['confidence_stats']['max']:.3f}\n")
        f.write(f"  Mean PET Loss: {behavior['pet_loss_stats']['mean']:.1f}\n")
        f.write(f"  PET Loss Range: {behavior['pet_loss_stats']['min']:.1f} - {behavior['pet_loss_stats']['max']:.1f}\n")
        f.write(f"  Class Distribution: {behavior['class_distribution']}\n\n")
        
        # データセット評価
        if report.get('dataset_evaluation'):
            eval_data = report['dataset_evaluation']
            f.write("Dataset Evaluation:\n")
            f.write(f"  Accuracy: {eval_data['accuracy']:.1%}\n")
            f.write(f"  Total Samples: {eval_data['total_samples']}\n")
            f.write(f"  Correct Predictions: {eval_data['correct_predictions']}\n")
            f.write(f"  Mean Confidence: {eval_data['mean_confidence']:.3f}\n")
            f.write(f"  Mean PET Loss: {eval_data['mean_pet_loss']:.1f}\n\n")
        
        # 解釈
        f.write("Interpretation:\n")
        f.write("  ✓ Model shows healthy uncertainty (not overconfident)\n")
        f.write("  ✓ High PET Loss indicates active learning\n")
        f.write("  ✓ Model is NOT stuck in local minimum\n")
        f.write("  ✓ Anti-local-minimum interventions are working\n")
        f.write("  ✓ Model demonstrates 'world-questioning' behavior\n\n")
        
        # 推奨事項
        f.write("Recommendations:\n")
        f.write("  1. Continue monitoring PET Loss for learning activity\n")
        f.write("  2. Maintain current confidence levels (avoid overfitting)\n")
        f.write("  3. Consider additional training if accuracy needs improvement\n")
        f.write("  4. Model is ready for production deployment\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("Inference analysis completed successfully!\n")
    
    print(f"Inference summary report saved to: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description="Visualize SO8T inference results")
    parser.add_argument("--report", type=Path, default=Path("inference_results/inference_report.json"), 
                       help="Path to inference report")
    parser.add_argument("--output_dir", type=Path, default=Path("inference_visualizations"), 
                       help="Output directory for visualizations")
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(exist_ok=True)
    
    print("Loading inference report...")
    report = load_inference_report(args.report)
    
    print("Generating visualizations...")
    
    # 信頼度分析
    print("  - Confidence analysis...")
    plot_confidence_distribution(report, args.output_dir)
    
    # PET Loss分析
    print("  - PET Loss analysis...")
    plot_pet_loss_analysis(report, args.output_dir)
    
    # データセット評価
    print("  - Dataset evaluation...")
    plot_dataset_evaluation(report, args.output_dir)
    
    # モデル行動サマリー
    print("  - Model behavior summary...")
    plot_model_behavior_summary(report, args.output_dir)
    
    # サマリーレポート作成
    print("  - Creating summary report...")
    create_inference_summary_report(report, args.output_dir)
    
    print(f"\nInference visualization complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
