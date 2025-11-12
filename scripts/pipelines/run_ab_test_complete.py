#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/Bテスト統合実行スクリプト

全パイプラインの統合実行:
1. モデルA（ベースライン）評価
2. モデルB作成（焼きこみ→事後学習→ファインチューニング→温度較正）
3. モデルB評価
4. A/Bテスト比較
5. 可視化
6. レポート生成

Usage:
    python scripts/run_ab_test_complete.py --config configs/ab_test_borea_phi35.yaml
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class ABTestCompletePipeline:
    """A/Bテスト統合パイプライン"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config = self._load_config(config_path)
        self.base_model_path = self.config['model']['base_model_path']
        self.test_data_path = self.config['evaluation']['test_data']
        self.output_dir = Path(self.config['evaluation']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("A/B Test Complete Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Base model: {self.base_model_path}")
        logger.info(f"Test data: {self.test_data_path}")
        logger.info(f"Output dir: {self.output_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def step1_evaluate_model_a(self) -> Path:
        """
        ステップ1: モデルA（ベースライン）評価
        
        Returns:
            metrics_path: 評価結果のパス
        """
        logger.info("="*80)
        logger.info("STEP 1: Evaluating Model A (Baseline)")
        logger.info("="*80)
        
        metrics_path = self.output_dir / "metrics_model_a.json"
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_model_a_baseline.py"),
            "--model", self.base_model_path,
            "--test", self.test_data_path,
            "--output", str(metrics_path),
            "--batch-size", str(self.config['evaluation']['batch_size']),
            "--max-new-tokens", str(self.config['evaluation'].get('max_new_tokens', 50))
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            raise RuntimeError(f"Model A evaluation failed with return code {result.returncode}")
        
        logger.info(f"[OK] Step 1 completed. Metrics saved to {metrics_path}")
        return metrics_path
    
    def step2_create_model_b(self, skip_steps: Optional[list] = None) -> Path:
        """
        ステップ2: モデルB作成
        
        Args:
            skip_steps: スキップするステップのリスト
        
        Returns:
            model_b_path: モデルBの最終パス
        """
        logger.info("="*80)
        logger.info("STEP 2: Creating Model B")
        logger.info("="*80)
        
        config_path = self.config.get('pipeline_config', 'configs/ab_test_borea_phi35.yaml')
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "create_model_b_pipeline.py"),
            "--config", config_path
        ]
        
        if skip_steps:
            cmd.extend(["--skip-steps"] + skip_steps)
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            raise RuntimeError(f"Model B creation failed with return code {result.returncode}")
        
        # モデルBの最終パスを取得
        model_b_path = Path(self.config['output']['base_dir']) / "calibrated" / "final_model"
        
        logger.info(f"[OK] Step 2 completed. Model B saved to {model_b_path}")
        return model_b_path
    
    def step3_evaluate_model_b(self, model_b_path: Path) -> Path:
        """
        ステップ3: モデルB評価
        
        Args:
            model_b_path: モデルBのパス
        
        Returns:
            metrics_path: 評価結果のパス
        """
        logger.info("="*80)
        logger.info("STEP 3: Evaluating Model B")
        logger.info("="*80)
        
        metrics_path = self.output_dir / "metrics_model_b.json"
        
        # モデルBは分類モデルなので、evaluate_four_class.pyを使用
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_four_class.py"),
            "--model", str(model_b_path),
            "--test", self.test_data_path,
            "--output", str(metrics_path),
            "--batch-size", str(self.config['evaluation']['batch_size'])
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            raise RuntimeError(f"Model B evaluation failed with return code {result.returncode}")
        
        logger.info(f"[OK] Step 3 completed. Metrics saved to {metrics_path}")
        return metrics_path
    
    def step4_ab_test_comparison(self, metrics_a_path: Path, metrics_b_path: Path, model_b_path: Path) -> Path:
        """
        ステップ4: A/Bテスト比較
        
        Args:
            metrics_a_path: モデルAのメトリクスパス
            metrics_b_path: モデルBのメトリクスパス
            model_b_path: モデルBのパス
        
        Returns:
            comparison_path: 比較結果のパス
        """
        logger.info("="*80)
        logger.info("STEP 4: A/B Test Comparison")
        logger.info("="*80)
        
        comparison_path = self.output_dir / "comparison_report.json"
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ab_test_borea_phi35.py"),
            "--model-a", self.base_model_path,
            "--model-b", str(model_b_path),
            "--test", self.test_data_path,
            "--output", str(comparison_path),
            "--batch-size", str(self.config['evaluation']['batch_size']),
            "--max-new-tokens", str(self.config['evaluation'].get('max_new_tokens', 50))
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            raise RuntimeError(f"A/B test comparison failed with return code {result.returncode}")
        
        logger.info(f"[OK] Step 4 completed. Comparison report saved to {comparison_path}")
        return comparison_path
    
    def step5_visualization(self, metrics_a_path: Path, metrics_b_path: Path):
        """
        ステップ5: 可視化
        
        Args:
            metrics_a_path: モデルAのメトリクスパス
            metrics_b_path: モデルBのメトリクスパス
        """
        logger.info("="*80)
        logger.info("STEP 5: Visualization")
        logger.info("="*80)
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "visualize_ab_test_training_curves.py"),
            "--metrics-a", str(metrics_a_path),
            "--metrics-b", str(metrics_b_path),
            "--output-dir", str(self.output_dir)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            logger.warning(f"Visualization failed with return code {result.returncode}, continuing...")
        else:
            logger.info(f"[OK] Step 5 completed. Visualizations saved to {self.output_dir}")
    
    def step6_generate_report(self, comparison_path: Path) -> Path:
        """
        ステップ6: レポート生成
        
        Args:
            comparison_path: 比較結果のパス
        
        Returns:
            report_path: レポートのパス
        """
        logger.info("="*80)
        logger.info("STEP 6: Generating Final Report")
        logger.info("="*80)
        
        # 比較結果を読み込み
        with open(comparison_path, 'r', encoding='utf-8') as f:
            comparison = json.load(f)
        
        # レポート生成
        report_path = self.output_dir / "ab_test_report.md"
        
        report = f"""# Borea-Phi-3.5-mini-Instruct-Common A/B Test Report

## 実施日時
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## テスト概要

### モデルA（ベースライン）
- **モデル**: {self.base_model_path}
- **説明**: 元のBorea-Phi-3.5-mini-Instruct-Commonモデル（そのまま）

### モデルB（処理済み）
- **モデル**: {comparison['model_b']['path']}
- **説明**: 焼きこみ→事後学習→ファインチューニング→温度較正を適用

## 評価結果

### モデルA（ベースライン）

| メトリクス | 値 |
|---------|-----|
| Accuracy | {comparison['model_a']['metrics']['accuracy']:.4f} |
| F1 Macro | {comparison['model_a']['metrics']['f1_macro']:.4f} |
| F1 ALLOW | {comparison['model_a']['metrics']['f1_allow']:.4f} |
| F1 ESCALATION | {comparison['model_a']['metrics']['f1_escalation']:.4f} |
| F1 DENY | {comparison['model_a']['metrics']['f1_deny']:.4f} |
| F1 REFUSE | {comparison['model_a']['metrics']['f1_refuse']:.4f} |
| False Positive Rate | {comparison['model_a']['metrics']['false_positive_rate']:.4f} |

### モデルB（処理済み）

| メトリクス | 値 |
|---------|-----|
| Accuracy | {comparison['model_b']['metrics']['accuracy']:.4f} |
| F1 Macro | {comparison['model_b']['metrics']['f1_macro']:.4f} |
| F1 ALLOW | {comparison['model_b']['metrics']['f1_allow']:.4f} |
| F1 ESCALATION | {comparison['model_b']['metrics']['f1_escalation']:.4f} |
| F1 DENY | {comparison['model_b']['metrics']['f1_deny']:.4f} |
| F1 REFUSE | {comparison['model_b']['metrics']['f1_refuse']:.4f} |
| False Positive Rate | {comparison['model_b']['metrics']['false_positive_rate']:.4f} |

## 改善結果

| メトリクス | 改善率 |
|---------|--------|
| Accuracy | {comparison['comparison']['improvements']['accuracy_improvement_pct']:.2f}% |
| F1 Macro | {comparison['comparison']['improvements']['f1_macro_improvement_pct']:.2f}% |
| False Positive Rate | {comparison['comparison']['improvements']['false_positive_rate_improvement_pct']:.2f}% |

## 結論

モデルBはモデルAと比較して、以下の改善を示しました：

- **Accuracy**: {comparison['comparison']['improvements']['accuracy_improvement_pct']:+.2f}%
- **F1 Macro**: {comparison['comparison']['improvements']['f1_macro_improvement_pct']:+.2f}%
- **False Positive Rate**: {comparison['comparison']['improvements']['false_positive_rate_improvement_pct']:+.2f}%

## 詳細結果

詳細な結果は以下のファイルを参照してください：

- 比較レポート: `{comparison_path}`
- モデルAメトリクス: `{self.output_dir / "metrics_model_a.json"}`
- モデルBメトリクス: `{self.output_dir / "metrics_model_b.json"}`
- 可視化結果: `{self.output_dir}`

"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"[OK] Step 6 completed. Report saved to {report_path}")
        return report_path
    
    def run_complete_pipeline(self, skip_model_b_creation: bool = False, skip_steps: Optional[list] = None):
        """
        パイプライン全体を実行
        
        Args:
            skip_model_b_creation: モデルB作成をスキップするか
            skip_steps: モデルB作成でスキップするステップのリスト
        """
        logger.info("="*80)
        logger.info("Starting A/B Test Complete Pipeline")
        logger.info("="*80)
        
        try:
            # Step 1: モデルA評価
            metrics_a_path = self.step1_evaluate_model_a()
            
            # Step 2: モデルB作成
            if not skip_model_b_creation:
                model_b_path = self.step2_create_model_b(skip_steps=skip_steps)
            else:
                logger.info("[SKIP] Model B creation")
                model_b_path = Path(self.config['output']['base_dir']) / "calibrated" / "final_model"
                if not model_b_path.exists():
                    raise FileNotFoundError(f"Model B not found at {model_b_path}. Please create it first or remove --skip-model-b-creation flag.")
            
            # Step 3: モデルB評価
            metrics_b_path = self.step3_evaluate_model_b(model_b_path)
            
            # Step 4: A/Bテスト比較
            comparison_path = self.step4_ab_test_comparison(metrics_a_path, metrics_b_path, model_b_path)
            
            # Step 5: 可視化
            self.step5_visualization(metrics_a_path, metrics_b_path)
            
            # Step 6: レポート生成
            report_path = self.step6_generate_report(comparison_path)
            
            logger.info("="*80)
            logger.info("[SUCCESS] A/B Test Complete Pipeline Finished!")
            logger.info("="*80)
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"Final report: {report_path}")
            logger.info("="*80)
            
            return report_path
            
        except Exception as e:
            logger.error("="*80)
            logger.error(f"[ERROR] Pipeline failed: {e}")
            logger.error("="*80)
            logger.exception(e)
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="A/B Test Complete Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ab_test_borea_phi35.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--skip-model-b-creation",
        action="store_true",
        help="Skip Model B creation (use existing model)"
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        nargs='+',
        help="Steps to skip in Model B creation (e.g., --skip-steps step1 step2)"
    )
    
    args = parser.parse_args()
    
    # パイプライン初期化
    pipeline = ABTestCompletePipeline(config_path=args.config)
    
    # パイプライン実行
    report_path = pipeline.run_complete_pipeline(
        skip_model_b_creation=args.skip_model_b_creation,
        skip_steps=args.skip_steps
    )
    
    logger.info(f"A/B test completed successfully. Report: {report_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())








