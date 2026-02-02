#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習メトリクス記録・可視化・PoCレポート生成モジュール

Hugging Faceや業務提携先へのPoC提出用の学習曲線と指標を保存
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# 日本語フォント設定（Windows対応）
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except Exception:
    pass

sns.set_style("whitegrid")
sns.set_palette("husl")


class TrainingMetricsRecorder:
    """学習メトリクス記録・可視化・PoCレポート生成クラス"""
    
    def __init__(
        self,
        output_dir: Path,
        model_name: str = "so8t_model",
        save_interval: int = 10,
        save_plots: bool = True,
        save_csv: bool = True,
        save_json: bool = True
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            model_name: モデル名
            save_interval: メトリクス保存間隔（ステップ数）
            save_plots: グラフを保存するか
            save_csv: CSVを保存するか
            save_json: JSONを保存するか
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.save_interval = save_interval
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.save_json = save_json
        
        # ディレクトリ作成
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "poc_reports"
        
        for dir_path in [self.metrics_dir, self.plots_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # メトリクス履歴
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        # ファイルパス
        self.metrics_json_path = self.metrics_dir / "training_metrics.json"
        self.metrics_csv_path = self.metrics_dir / "training_metrics.csv"
        self.plots_pdf_path = self.plots_dir / "training_curves.pdf"
        
        logger.info(f"[METRICS] Metrics recorder initialized: {self.output_dir}")
    
    def record_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        learning_rate: float,
        pet_loss: Optional[float] = None,
        so8t_loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        perplexity: Optional[float] = None,
        grad_norm: Optional[float] = None,
        **kwargs
    ):
        """
        ステップごとのメトリクスを記録
        
        Args:
            step: ステップ数
            epoch: エポック数
            loss: 損失
            learning_rate: 学習率
            pet_loss: PET損失（オプション）
            so8t_loss: SO8T損失（オプション）
            accuracy: 精度（オプション）
            perplexity: パープレキシティ（オプション）
            grad_norm: 勾配ノルム（オプション）
            **kwargs: その他のメトリクス
        """
        timestamp = datetime.now().isoformat()
        elapsed_time = time.time() - self.start_time
        
        metric_entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': timestamp,
            'elapsed_time_seconds': elapsed_time,
            'loss': loss,
            'learning_rate': learning_rate,
        }
        
        # オプションメトリクス
        if pet_loss is not None:
            metric_entry['pet_loss'] = pet_loss
        if so8t_loss is not None:
            metric_entry['so8t_loss'] = so8t_loss
        if accuracy is not None:
            metric_entry['accuracy'] = accuracy
        if perplexity is not None:
            metric_entry['perplexity'] = perplexity
        if grad_norm is not None:
            metric_entry['grad_norm'] = grad_norm
        
        # その他のメトリクス
        metric_entry.update(kwargs)
        
        self.metrics_history.append(metric_entry)
        
        # 定期的に保存
        if step % self.save_interval == 0:
            self._save_metrics()
            if self.save_plots:
                self._update_plots()
    
    def _save_metrics(self):
        """メトリクスをファイルに保存"""
        # JSON保存
        if self.save_json:
            with open(self.metrics_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        
        # CSV保存
        if self.save_csv and self.metrics_history:
            with open(self.metrics_csv_path, 'w', encoding='utf-8', newline='') as f:
                if self.metrics_history:
                    writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].keys())
                    writer.writeheader()
                    writer.writerows(self.metrics_history)
    
    def _update_plots(self):
        """学習曲線を更新"""
        if not self.metrics_history:
            return
        
        try:
            # データ準備
            steps = [m['step'] for m in self.metrics_history]
            losses = [m['loss'] for m in self.metrics_history]
            learning_rates = [m.get('learning_rate', 0) for m in self.metrics_history]
            
            # 複数サブプロット
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{self.model_name} - Training Curves', fontsize=16, fontweight='bold')
            
            # Loss曲線
            axes[0, 0].plot(steps, losses, 'b-', linewidth=2, label='Total Loss')
            if any('pet_loss' in m for m in self.metrics_history):
                pet_losses = [m.get('pet_loss', 0) for m in self.metrics_history]
                axes[0, 0].plot(steps, pet_losses, 'r--', linewidth=1.5, label='PET Loss', alpha=0.7)
            if any('so8t_loss' in m for m in self.metrics_history):
                so8t_losses = [m.get('so8t_loss', 0) for m in self.metrics_history]
                axes[0, 0].plot(steps, so8t_losses, 'g--', linewidth=1.5, label='SO8T Loss', alpha=0.7)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Learning Rate曲線
            axes[0, 1].plot(steps, learning_rates, 'purple', linewidth=2)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Accuracy曲線（存在する場合）
            if any('accuracy' in m for m in self.metrics_history):
                accuracies = [m.get('accuracy', 0) for m in self.metrics_history]
                axes[1, 0].plot(steps, accuracies, 'green', linewidth=2)
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].set_title('Training Accuracy')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Accuracy data not available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Training Accuracy')
            
            # Perplexity曲線（存在する場合）
            if any('perplexity' in m for m in self.metrics_history):
                perplexities = [m.get('perplexity', 0) for m in self.metrics_history]
                axes[1, 1].plot(steps, perplexities, 'orange', linewidth=2)
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Perplexity')
                axes[1, 1].set_title('Perplexity')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Perplexity data not available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Perplexity')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"[METRICS] Failed to update plots: {e}")
    
    def generate_poc_report(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        final_metrics: Optional[Dict[str, float]] = None
    ) -> Path:
        """
        PoC提出用レポートを生成
        
        Args:
            model_config: モデル設定
            training_config: 学習設定
            final_metrics: 最終メトリクス（オプション）
        
        Returns:
            レポートファイルパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"poc_report_{timestamp}.json"
        
        # 最終メトリクスを計算
        if not final_metrics and self.metrics_history:
            final_metrics = self._calculate_final_metrics()
        
        # レポートデータ
        report = {
            'model_info': {
                'name': self.model_name,
                'config': model_config,
            },
            'training_info': {
                'config': training_config,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_time_seconds': time.time() - self.start_time,
                'total_steps': len(self.metrics_history),
            },
            'metrics': {
                'history': self.metrics_history,
                'final': final_metrics or {},
                'summary': self._calculate_summary_metrics(),
            },
            'files': {
                'metrics_json': str(self.metrics_json_path),
                'metrics_csv': str(self.metrics_csv_path),
                'plots_png': str(self.plots_dir / "training_curves.png"),
            }
        }
        
        # JSONレポート保存
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # CSVサマリーも生成
        summary_csv_path = self.reports_dir / f"poc_summary_{timestamp}.csv"
        self._save_summary_csv(summary_csv_path, report)
        
        logger.info(f"[METRICS] PoC report generated: {report_path}")
        
        return report_path
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """最終メトリクスを計算"""
        if not self.metrics_history:
            return {}
        
        last_metrics = self.metrics_history[-1]
        final = {
            'final_loss': last_metrics.get('loss', 0.0),
            'final_learning_rate': last_metrics.get('learning_rate', 0.0),
        }
        
        if 'accuracy' in last_metrics:
            final['final_accuracy'] = last_metrics['accuracy']
        if 'perplexity' in last_metrics:
            final['final_perplexity'] = last_metrics['perplexity']
        
        return final
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """サマリーメトリクスを計算"""
        if not self.metrics_history:
            return {}
        
        losses = [m['loss'] for m in self.metrics_history]
        
        summary = {
            'loss': {
                'min': float(np.min(losses)),
                'max': float(np.max(losses)),
                'mean': float(np.mean(losses)),
                'std': float(np.std(losses)),
                'final': float(losses[-1]),
            }
        }
        
        if any('accuracy' in m for m in self.metrics_history):
            accuracies = [m.get('accuracy', 0) for m in self.metrics_history]
            summary['accuracy'] = {
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'final': float(accuracies[-1]) if accuracies else 0.0,
            }
        
        if any('perplexity' in m for m in self.metrics_history):
            perplexities = [m.get('perplexity', 0) for m in self.metrics_history]
            summary['perplexity'] = {
                'min': float(np.min(perplexities)),
                'max': float(np.max(perplexities)),
                'mean': float(np.mean(perplexities)),
                'std': float(np.std(perplexities)),
                'final': float(perplexities[-1]) if perplexities else 0.0,
            }
        
        return summary
    
    def _save_summary_csv(self, path: Path, report: Dict[str, Any]):
        """サマリーCSVを保存"""
        try:
            summary = report['metrics']['summary']
            rows = []
            
            for metric_name, metric_values in summary.items():
                for stat_name, stat_value in metric_values.items():
                    rows.append({
                        'metric': metric_name,
                        'statistic': stat_name,
                        'value': stat_value
                    })
            
            if rows:
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['metric', 'statistic', 'value'])
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as e:
            logger.warning(f"[METRICS] Failed to save summary CSV: {e}")

