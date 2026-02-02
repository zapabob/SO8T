#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning & A/Bテスト統合パイプライン

収集・加工済みデータでHugging Faceモデルをfine-tuningし、
SO8Tで既存のTransformerモデルと置き換えてA/Bテストを実行
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/finetune_and_ab_test_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FinetuneAndABTestPipeline:
    """Fine-tuning & A/Bテスト統合パイプライン"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.output_base_dir = Path(config.get("output_dir", "D:/webdataset/finetuned_models"))
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("Fine-tuning & A/B Test Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_base_dir}")
    
    def step1_convert_data(self) -> Path:
        """ステップ1: データ変換"""
        logger.info("="*80)
        logger.info("STEP 1: Convert Four Class Data to Hugging Face Dataset")
        logger.info("="*80)
        
        from scripts.training.convert_four_class_to_hf_dataset import FourClassToHFDatasetConverter
        
        input_path = Path(self.config["data"]["input_path"])
        output_dir = self.output_base_dir / "hf_datasets"
        base_model = self.config["model"]["base_model"]
        
        converter = FourClassToHFDatasetConverter(base_model_name=base_model)
        output_path = converter.convert(
            input_path=input_path,
            output_dir=output_dir,
            format_type=self.config["data"].get("format_type", "instruction"),
            max_length=self.config["data"].get("max_length", 2048),
            split=True,
            tokenize=True
        )
        
        logger.info(f"[OK] Step 1 completed. Dataset saved to {output_path}")
        return output_path
    
    def step2_finetune(self, dataset_path: Path) -> Path:
        """ステップ2: Fine-tuning"""
        logger.info("="*80)
        logger.info("STEP 2: Fine-tune Hugging Face Model")
        logger.info("="*80)
        
        from scripts.training.finetune_hf_model_with_processed_data import HFModelFinetuner
        
        base_model = self.config["model"]["base_model"]
        output_dir = self.output_base_dir / "finetuned_models" / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        finetuner = HFModelFinetuner(
            base_model_name=base_model,
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=self.config.get("training", {})
        )
        
        final_model_dir = finetuner.train()
        
        logger.info(f"[OK] Step 2 completed. Fine-tuned model saved to {final_model_dir}")
        return final_model_dir
    
    def step3_replace_so8t(self, finetuned_model_path: Path) -> Path:
        """ステップ3: SO8Tモデル置き換え"""
        logger.info("="*80)
        logger.info("STEP 3: Replace SO8T Model with Fine-tuned Model")
        logger.info("="*80)
        
        from scripts.training.replace_so8t_with_finetuned_hf import SO8TModelReplacer
        
        output_dir = self.output_base_dir / "so8t_replaced_models" / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        replacer = SO8TModelReplacer(
            finetuned_model_path=finetuned_model_path,
            output_dir=output_dir
        )
        
        result = replacer.replace()
        
        logger.info(f"[OK] Step 3 completed. Replacement result: {result}")
        return Path(result["so8t_model_path"])
    
    def step4_ab_test(self, finetuned_model_path: Path) -> Dict:
        """ステップ4: A/Bテスト"""
        logger.info("="*80)
        logger.info("STEP 4: A/B Test")
        logger.info("="*80)
        
        import torch
        from scripts.evaluation.ab_test_so8t_vs_finetuned_hf import ABTestEvaluator
        
        test_data_path = Path(self.config["test"]["test_data_path"])
        output_dir = self.output_base_dir / "ab_test_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        model_a_path = self.config.get("test", {}).get("model_a_path")
        
        evaluator = ABTestEvaluator(output_dir)
        results = evaluator.run_ab_test(
            model_a_path=Path(model_a_path) if model_a_path else None,
            model_b_path=finetuned_model_path,
            test_data_path=test_data_path,
            device=device
        )
        
        logger.info(f"[OK] Step 4 completed. A/B test results saved to {output_dir}")
        return results
    
    def run_pipeline(self):
        """パイプライン実行"""
        logger.info("="*80)
        logger.info("Starting Fine-tuning & A/B Test Pipeline")
        logger.info("="*80)
        
        try:
            # ステップ1: データ変換
            dataset_path = self.step1_convert_data()
            
            # ステップ2: Fine-tuning
            finetuned_model_path = self.step2_finetune(dataset_path)
            
            # ステップ3: SO8Tモデル置き換え
            so8t_model_path = self.step3_replace_so8t(finetuned_model_path)
            
            # ステップ4: A/Bテスト
            ab_test_results = self.step4_ab_test(finetuned_model_path)
            
            # 最終結果
            final_results = {
                "dataset_path": str(dataset_path),
                "finetuned_model_path": str(finetuned_model_path),
                "so8t_model_path": str(so8t_model_path),
                "ab_test_results": ab_test_results,
                "timestamp": datetime.now().isoformat()
            }
            
            # 結果保存
            results_path = self.output_base_dir / "pipeline_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            logger.info("="*80)
            logger.info("[COMPLETE] Pipeline completed successfully!")
            logger.info(f"Results saved to: {results_path}")
            logger.info("="*80)
            
            return final_results
            
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """メイン関数"""
    import yaml
    
    parser = argparse.ArgumentParser(description="Fine-tuning & A/B Test Pipeline")
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Configuration file path (YAML)'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # パイプライン実行
    pipeline = FinetuneAndABTestPipeline(config)
    results = pipeline.run_pipeline()
    
    logger.info("="*80)
    logger.info("[COMPLETE] All steps completed!")
    logger.info("="*80)


if __name__ == '__main__':
    main()

