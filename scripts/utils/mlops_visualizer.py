#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLOps/LLMOps可視化統合モジュール

MLflow + W&B + TensorBoardを統合した可視化機能を提供
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow統合
MLFLOW_AVAILABLE = False
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("MLflow not available")

# W&B統合
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    logger.warning("W&B not available")

# TensorBoard統合（無効化されているためインポートをスキップ）
TENSORBOARD_AVAILABLE = False
SummaryWriter = None
# TensorBoardは無効化されているため、インポートエラーを抑制
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_AVAILABLE = True
# except ImportError:
#     logger.warning("TensorBoard not available")


class MLOpsVisualizer:
    """MLOps/LLMOps可視化統合クラス"""
    
    def __init__(self, config: Dict[str, Any], session_id: str = None):
        """
        Args:
            config: 設定辞書
            session_id: セッションID
        """
        self.config = config
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # MLflow設定
        self.mlflow_config = config.get('mlflow', {})
        self.mlflow_enabled = self.mlflow_config.get('enabled', True) and MLFLOW_AVAILABLE
        self.mlflow_tracking_uri = self.mlflow_config.get('tracking_uri', 'file:./mlruns')
        self.mlflow_experiment_name = self.mlflow_config.get('experiment_name', 'SO8T_Production')
        
        # W&B設定
        self.wandb_config = config.get('wandb', {})
        self.wandb_enabled = self.wandb_config.get('enabled', True) and WANDB_AVAILABLE
        self.wandb_project = self.wandb_config.get('project', 'so8t-production')
        self.wandb_entity = self.wandb_config.get('entity', None)
        
        # TensorBoard設定
        self.tensorboard_config = config.get('tensorboard', {})
        self.tensorboard_enabled = self.tensorboard_config.get('enabled', True) and TENSORBOARD_AVAILABLE
        self.tensorboard_log_dir = Path(self.tensorboard_config.get('log_dir', 'runs'))
        
        # 初期化
        self._initialize_mlflow()
        self._initialize_wandb()
        self._initialize_tensorboard()
        
        logger.info("="*80)
        logger.info("MLOps Visualizer Initialized")
        logger.info("="*80)
        logger.info(f"MLflow: {'Enabled' if self.mlflow_enabled else 'Disabled'}")
        logger.info(f"W&B: {'Enabled' if self.wandb_enabled else 'Disabled'}")
        logger.info(f"TensorBoard: {'Enabled' if self.tensorboard_enabled else 'Disabled'}")
        logger.info("="*80)
    
    def _initialize_mlflow(self):
        """MLflow初期化"""
        if not self.mlflow_enabled:
            return
        
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_experiment_name)
            logger.info(f"[MLFLOW] Initialized: {self.mlflow_tracking_uri}")
        except Exception as e:
            logger.warning(f"[MLFLOW] Initialization failed: {e}")
            self.mlflow_enabled = False
    
    def _initialize_wandb(self):
        """W&B初期化"""
        if not self.wandb_enabled:
            return
        
        try:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=f"SO8T_Production_{self.session_id}",
                config=self.config,
                reinit=True
            )
            logger.info(f"[WANDB] Initialized: {self.wandb_project}")
        except Exception as e:
            logger.warning(f"[WANDB] Initialization failed: {e}")
            self.wandb_enabled = False
    
    def _initialize_tensorboard(self):
        """TensorBoard初期化"""
        if not self.tensorboard_enabled:
            self.tensorboard_writer = None
            return
        
        try:
            self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(
                log_dir=str(self.tensorboard_log_dir / self.session_id)
            )
            logger.info(f"[TENSORBOARD] Initialized: {self.tensorboard_log_dir}")
        except Exception as e:
            logger.warning(f"[TENSORBOARD] Initialization failed: {e}")
            self.tensorboard_enabled = False
            self.tensorboard_writer = None
    
    def log_phase_start(self, phase_name: str, phase_config: Dict[str, Any] = None):
        """フェーズ開始をログ"""
        logger.info(f"[MLOPS] Phase started: {phase_name}")
        
        if self.mlflow_enabled:
            try:
                mlflow.log_param(f"phase_{phase_name}_started", True)
                if phase_config:
                    for key, value in phase_config.items():
                        mlflow.log_param(f"phase_{phase_name}_{key}", value)
            except Exception as e:
                logger.warning(f"[MLFLOW] Failed to log phase start: {e}")
        
        if self.wandb_enabled:
            try:
                wandb.log({f"phase/{phase_name}/started": 1})
            except Exception as e:
                logger.warning(f"[WANDB] Failed to log phase start: {e}")
    
    def log_phase_complete(self, phase_name: str, phase_metrics: Dict[str, Any]):
        """フェーズ完了をログ"""
        logger.info(f"[MLOPS] Phase completed: {phase_name}")
        
        if self.mlflow_enabled:
            try:
                mlflow.log_metrics({
                    f"phase_{phase_name}_{k}": v 
                    for k, v in phase_metrics.items() 
                    if isinstance(v, (int, float))
                })
                mlflow.log_param(f"phase_{phase_name}_completed", True)
            except Exception as e:
                logger.warning(f"[MLFLOW] Failed to log phase complete: {e}")
        
        if self.wandb_enabled:
            try:
                wandb.log({
                    f"phase/{phase_name}/{k}": v 
                    for k, v in phase_metrics.items()
                })
            except Exception as e:
                logger.warning(f"[WANDB] Failed to log phase complete: {e}")
        
        if self.tensorboard_enabled and self.tensorboard_writer is not None:
            try:
                for key, value in phase_metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(
                            f"phase/{phase_name}/{key}", 
                            value, 
                            self._get_step()
                        )
            except Exception as e:
                logger.warning(f"[TENSORBOARD] Failed to log phase complete: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """メトリクスをログ"""
        if self.mlflow_enabled:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"[MLFLOW] Failed to log metrics: {e}")
        
        if self.wandb_enabled:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"[WANDB] Failed to log metrics: {e}")
        
        if self.tensorboard_enabled and self.tensorboard_writer is not None:
            try:
                step = step or self._get_step()
                for key, value in metrics.items():
                    self.tensorboard_writer.add_scalar(key, value, step)
            except Exception as e:
                logger.warning(f"[TENSORBOARD] Failed to log metrics: {e}")
    
    def log_artifacts(self, artifact_path: Path, artifact_name: str = None):
        """アーティファクトをログ"""
        if not artifact_path.exists():
            logger.warning(f"[MLOPS] Artifact not found: {artifact_path}")
            return
        
        artifact_name = artifact_name or artifact_path.name
        
        if self.mlflow_enabled:
            try:
                mlflow.log_artifacts(str(artifact_path.parent), artifact_name)
            except Exception as e:
                logger.warning(f"[MLFLOW] Failed to log artifact: {e}")
        
        if self.wandb_enabled:
            try:
                wandb.log_artifact(str(artifact_path))
            except Exception as e:
                logger.warning(f"[WANDB] Failed to log artifact: {e}")
    
    def log_classification_results(self, classification_stats: Dict[str, int], phase_name: str = "four_class"):
        """分類結果をログ"""
        total = classification_stats.get('total', 0)
        if total == 0:
            return
        
        metrics = {}
        for label in ['ALLOW', 'ESCALATION', 'DENY', 'REFUSE']:
            count = classification_stats.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0
            metrics[f"{phase_name}/{label}_count"] = count
            metrics[f"{phase_name}/{label}_percentage"] = percentage
        
        self.log_metrics(metrics)
    
    def log_nsfw_stats(self, nsfw_stats: Dict[str, Any], phase_name: str = "nsfw_detection"):
        """NSFW統計をログ"""
        metrics = {
            f"{phase_name}/total_checked": nsfw_stats.get('total_checked', 0),
            f"{phase_name}/nsfw_detected": nsfw_stats.get('nsfw_detected', 0),
            f"{phase_name}/safe": nsfw_stats.get('safe', 0),
        }
        
        # ラベル別統計
        by_label = nsfw_stats.get('by_label', {})
        for label, count in by_label.items():
            metrics[f"{phase_name}/label/{label}"] = count
        
        self.log_metrics(metrics)
    
    def create_summary_dashboard(self, all_phase_metrics: Dict[str, Dict[str, Any]], output_path: Path):
        """統合ダッシュボード生成"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import pandas as pd
            
            logger.info("[MLOPS] Creating summary dashboard...")
            
            # データ準備
            dashboard_data = {
                'phase': [],
                'metric': [],
                'value': []
            }
            
            for phase_name, phase_metrics in all_phase_metrics.items():
                for metric_name, metric_value in phase_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        dashboard_data['phase'].append(phase_name)
                        dashboard_data['metric'].append(metric_name)
                        dashboard_data['value'].append(metric_value)
            
            df = pd.DataFrame(dashboard_data)
            
            # ダッシュボード生成
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('SO8T Production Pipeline Dashboard', fontsize=16)
            
            # 1. フェーズ別メトリクス
            if not df.empty:
                phase_metrics = df.groupby('phase')['value'].sum()
                axes[0, 0].bar(phase_metrics.index, phase_metrics.values)
                axes[0, 0].set_title('Metrics by Phase')
                axes[0, 0].set_xlabel('Phase')
                axes[0, 0].set_ylabel('Total Value')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 分類分布（四値分類）
            if 'four_class_classification' in all_phase_metrics:
                stats = all_phase_metrics['four_class_classification'].get('stats', {})
                labels = ['ALLOW', 'ESCALATION', 'DENY', 'REFUSE']
                counts = [stats.get(label, 0) for label in labels]
                axes[0, 1].bar(labels, counts, color=['green', 'orange', 'yellow', 'red'])
                axes[0, 1].set_title('Four Class Classification Distribution')
                axes[0, 1].set_ylabel('Count')
            
            # 3. NSFW検知統計
            if 'web_scraping' in all_phase_metrics:
                nsfw_stats = all_phase_metrics['web_scraping'].get('nsfw_stats', {})
                if nsfw_stats:
                    by_label = nsfw_stats.get('by_label', {})
                    if by_label:
                        axes[1, 0].bar(by_label.keys(), by_label.values())
                        axes[1, 0].set_title('NSFW Detection by Label')
                        axes[1, 0].set_ylabel('Count')
                        axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. サンプル数推移
            sample_counts = {}
            for phase_name, phase_metrics in all_phase_metrics.items():
                samples = phase_metrics.get('samples', 0)
                if samples > 0:
                    sample_counts[phase_name] = samples
            
            if sample_counts:
                axes[1, 1].bar(sample_counts.keys(), sample_counts.values())
                axes[1, 1].set_title('Sample Counts by Phase')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[MLOPS] Dashboard saved to {output_path}")
            
            # アーティファクトとしてログ
            self.log_artifacts(output_path, "dashboard")
            
        except ImportError:
            logger.warning("[MLOPS] matplotlib/pandas not available, skipping dashboard")
        except Exception as e:
            logger.warning(f"[MLOPS] Failed to create dashboard: {e}")
    
    def _get_step(self) -> int:
        """現在のステップを取得（簡易実装）"""
        return int(datetime.now().timestamp())
    
    def close(self):
        """リソースをクローズ"""
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception:
                pass
        
        if self.tensorboard_enabled and self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.close()
            except Exception:
                pass
        
        logger.info("[MLOPS] Visualizer closed")


def main():
    """テスト用メイン関数"""
    config = {
        'mlflow': {'enabled': True},
        'wandb': {'enabled': True, 'project': 'so8t-test'},
        'tensorboard': {'enabled': True}
    }
    
    visualizer = MLOpsVisualizer(config)
    
    # テストメトリクス
    visualizer.log_metrics({'test_metric': 0.95})
    visualizer.log_classification_results({
        'ALLOW': 100,
        'ESCALATION': 20,
        'DENY': 10,
        'REFUSE': 5,
        'total': 135
    })
    
    visualizer.close()


if __name__ == '__main__':
    main()

