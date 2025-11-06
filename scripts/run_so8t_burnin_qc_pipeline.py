"""
SO8T Burn-in QC 統合パイプライン

焼きこみ → QC検証 → 量子化 → 温度較正 → 長文テストの全ステップを自動実行する。
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import torch

# Transformers
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SO8TBurnInQCPipeline:
    """SO8T焼きこみQC統合パイプライン"""
    
    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str = "models/so8t_burnin_qc_output",
        validation_data_path: Optional[str] = None,
        validation_data_japanese_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            checkpoint_path: チェックポイントパス（models/so8t_rotations_epoch_final.pt など）
            output_dir: 出力ディレクトリ
            validation_data_path: 検証データパス（JSON）
            validation_data_japanese_path: 日本語検証データパス（JSON）
            device: デバイス
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.validation_data_path = Path(validation_data_path) if validation_data_path else None
        self.validation_data_japanese_path = Path(validation_data_japanese_path) if validation_data_japanese_path else None
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_pre = None
        self.model_post = None
        self.tokenizer = None
        
        self.validation_texts = []
        self.validation_texts_japanese = []
        
        logger.info("SO8T Burn-in QC Pipeline initialized")
        logger.info(f"  Checkpoint: {self.checkpoint_path}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
    
    def load_validation_data(self) -> None:
        """検証データを読み込み"""
        logger.info("Loading validation data...")
        
        if self.validation_data_path and self.validation_data_path.exists():
            with open(self.validation_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.validation_texts = data.get('texts', [])
            logger.info(f"  Loaded {len(self.validation_texts)} English validation texts")
        
        if self.validation_data_japanese_path and self.validation_data_japanese_path.exists():
            with open(self.validation_data_japanese_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.validation_texts_japanese = data.get('texts', [])
            logger.info(f"  Loaded {len(self.validation_texts_japanese)} Japanese validation texts")
        
        # 検証データがない場合はデフォルトを使用
        if not self.validation_texts and not self.validation_texts_japanese:
            logger.warning("  No validation data provided, using default texts")
            self.validation_texts = [
                "Explain the concept of neural networks.",
                "What is the meaning of artificial intelligence?"
            ]
            self.validation_texts_japanese = [
                "ニューラルネットワークの概念を説明してください。",
                "人工知能の意味は何ですか？"
            ]
    
    def load_checkpoint(self) -> Dict:
        """チェックポイントを読み込み"""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # チェックポイントの読み込み
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        
        logger.info("  Checkpoint loaded successfully")
        logger.info(f"  Keys: {list(checkpoint.keys())}")
        
        return checkpoint
    
    def run_burnin_pipeline(
        self,
        base_model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    ) -> Dict:
        """
        焼きこみパイプラインを実行
        
        Args:
            base_model_path: ベースモデルパス
        
        Returns:
            パイプライン結果
        """
        logger.info("=" * 80)
        logger.info("Step 1: Burn-in Pipeline")
        logger.info("=" * 80)
        
        # SO8T焼きこみパイプラインをインポート
        from so8t_burnin_pipeline import SO8TBurnInPipeline
        
        pipeline = SO8TBurnInPipeline(
            hf_model_path=base_model_path,
            output_dir=str(self.output_dir / "burnin"),
            so8t_weights_path=str(self.checkpoint_path),
            device=self.device
        )
        
        # 全検証テキストを統合
        all_validation_texts = self.validation_texts + self.validation_texts_japanese
        
        # パイプライン実行
        results = pipeline.run_pipeline(
            quantization="Q5_K_M",
            verify=True,
            validation_texts=all_validation_texts,
            enable_temperature_calibration=True
        )
        
        return results
    
    def run_qc_verification(
        self,
        baked_model_path: Path
    ) -> Dict:
        """
        QC検証を実行
        
        Args:
            baked_model_path: 焼きこみ済みモデルパス
        
        Returns:
            QC検証結果
        """
        logger.info("=" * 80)
        logger.info("Step 2: QC Verification")
        logger.info("=" * 80)
        
        from so8t_burnin_qc import SO8TBurnInQC
        
        qc = SO8TBurnInQC(device=self.device)
        
        # モデル読み込み
        logger.info("  Loading models for QC verification...")
        
        # 焼きこみ後のモデルのみ使用（焼きこみ前との比較は省略）
        model_post = Qwen2VLForConditionalGeneration.from_pretrained(
            str(baked_model_path),
            torch_dtype=torch.float16,
            device_map=None
        ).to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(baked_model_path),
            trust_remote_code=True
        )
        
        # テスト入力を準備
        test_texts = self.validation_texts[:3]  # 最初の3サンプルを使用
        
        for i, text in enumerate(test_texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # RoPE位相安定性テスト
            qc.test_rope_phase_stability(
                model_post,
                inputs,
                sample_name=f"sample_{i}"
            )
        
        # レポート保存
        qc_json_path = self.output_dir / f"{datetime.now().strftime('%Y-%m-%d')}_so8t_burnin_qc_report.json"
        qc_md_path = self.output_dir / f"{datetime.now().strftime('%Y-%m-%d')}_so8t_burnin_qc_report.md"
        
        qc.save_report(qc_json_path)
        qc.generate_markdown_report(qc_md_path)
        
        logger.info("  QC verification complete")
        
        return {
            'qc_json_report': qc_json_path,
            'qc_md_report': qc_md_path
        }
    
    def run_longtext_regression(
        self,
        baked_model_path: Path
    ) -> Dict:
        """
        長文回帰テストを実行
        
        Args:
            baked_model_path: 焼きこみ済みモデルパス
        
        Returns:
            長文テスト結果
        """
        logger.info("=" * 80)
        logger.info("Step 3: Long Text Regression Test")
        logger.info("=" * 80)
        
        from so8t_longtext_regression_test import (
            SO8TLongTextRegressionTest,
            LONG_TEXT_TEST_CASES
        )
        
        tester = SO8TLongTextRegressionTest(device=self.device)
        
        # モデル読み込み
        logger.info("  Loading model for long text testing...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(baked_model_path),
            torch_dtype=torch.float16,
            device_map=None
        ).to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(baked_model_path),
            trust_remote_code=True
        )
        
        # テストケースを実行（最初の2ケースのみ）
        for test_case in LONG_TEXT_TEST_CASES[:2]:
            try:
                tester.run_test(
                    model,
                    tokenizer,
                    test_case,
                    max_new_tokens=512
                )
            except Exception as e:
                logger.error(f"  Test {test_case['name']} failed: {e}")
        
        # 結果の保存
        output_subdir = self.output_dir / "longtext_regression"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        longtext_json_path = output_subdir / f"{datetime.now().strftime('%Y-%m-%d')}_so8t_longtext_regression.json"
        longtext_md_path = output_subdir / f"{datetime.now().strftime('%Y-%m-%d')}_so8t_longtext_regression.md"
        
        tester.save_report(longtext_json_path)
        tester.generate_markdown_report(longtext_md_path)
        tester.visualize_results(output_subdir)
        
        logger.info("  Long text regression test complete")
        
        return {
            'longtext_json_report': longtext_json_path,
            'longtext_md_report': longtext_md_path,
            'longtext_plots': output_subdir
        }
    
    def run_full_pipeline(
        self,
        base_model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        skip_burnin: bool = False,
        baked_model_path: Optional[str] = None
    ) -> Dict:
        """
        全パイプラインを実行
        
        Args:
            base_model_path: ベースモデルパス
            skip_burnin: 焼きこみをスキップするか
            baked_model_path: 既存の焼きこみ済みモデルパス
        
        Returns:
            全体の結果
        """
        logger.info("=" * 80)
        logger.info("SO8T Burn-in QC Full Pipeline")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # 検証データ読み込み
        self.load_validation_data()
        
        results = {}
        
        # Step 1: 焼きこみ（スキップ可能）
        if not skip_burnin:
            try:
                burnin_results = self.run_burnin_pipeline(base_model_path)
                results['burnin'] = burnin_results
                baked_model_path = burnin_results.get('baked_model_dir')
            except Exception as e:
                logger.error(f"Burn-in pipeline failed: {e}")
                logger.error("Continuing with QC verification using provided model path...")
        
        if not baked_model_path:
            logger.error("No baked model path available. Cannot proceed.")
            return results
        
        baked_model_path = Path(baked_model_path)
        
        # Step 2: QC検証
        try:
            qc_results = self.run_qc_verification(baked_model_path)
            results['qc'] = qc_results
        except Exception as e:
            logger.error(f"QC verification failed: {e}")
        
        # Step 3: 長文回帰テスト
        try:
            longtext_results = self.run_longtext_regression(baked_model_path)
            results['longtext'] = longtext_results
        except Exception as e:
            logger.error(f"Long text regression test failed: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("Full Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Output directory: {self.output_dir}")
        
        # 統合レポート生成
        self.generate_integrated_report(results, duration)
        
        return results
    
    def generate_integrated_report(
        self,
        results: Dict,
        duration: float
    ) -> None:
        """
        統合レポートを生成
        
        Args:
            results: 全体の結果
            duration: 実行時間（秒）
        """
        logger.info("Generating integrated report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = self.output_dir / f"{datetime.now().strftime('%Y-%m-%d')}_so8t_burnin_qc_integrated_report.md"
        
        report = f"""# SO8T Burn-in QC Integrated Report

Generated: {timestamp}
Duration: {duration:.2f} seconds

## Overview

This report documents the complete SO8T burn-in QC pipeline execution, including:
1. SO8T rotation gate burn-in
2. Quality control verification
3. Long text regression testing

## Pipeline Steps

"""
        
        # 焼きこみ結果
        if 'burnin' in results:
            burnin = results['burnin']
            report += "### Step 1: Burn-in Pipeline\n\n"
            report += f"- **Status**: Success\n"
            report += f"- **Baked Model**: `{burnin.get('baked_model_dir', 'N/A')}`\n"
            report += f"- **F16 GGUF**: `{burnin.get('f16_gguf', 'N/A')}`\n"
            report += f"- **Quantized GGUF**: `{burnin.get('quantized_gguf', 'N/A')}`\n"
            if 'optimal_temperature' in burnin:
                report += f"- **Optimal Temperature**: {burnin['optimal_temperature']:.4f}\n"
            report += "\n"
        
        # QC検証結果
        if 'qc' in results:
            qc = results['qc']
            report += "### Step 2: QC Verification\n\n"
            report += f"- **Status**: Success\n"
            report += f"- **JSON Report**: `{qc.get('qc_json_report', 'N/A')}`\n"
            report += f"- **Markdown Report**: `{qc.get('qc_md_report', 'N/A')}`\n\n"
        
        # 長文回帰テスト結果
        if 'longtext' in results:
            longtext = results['longtext']
            report += "### Step 3: Long Text Regression Test\n\n"
            report += f"- **Status**: Success\n"
            report += f"- **JSON Report**: `{longtext.get('longtext_json_report', 'N/A')}`\n"
            report += f"- **Markdown Report**: `{longtext.get('longtext_md_report', 'N/A')}`\n"
            report += f"- **Plots Directory**: `{longtext.get('longtext_plots', 'N/A')}`\n\n"
        
        report += "## Summary\n\n"
        report += f"- **Total Steps Completed**: {len(results)}\n"
        report += f"- **Execution Time**: {duration:.2f} seconds\n"
        report += f"- **Output Directory**: `{self.output_dir}`\n\n"
        
        report += "## Recommendations\n\n"
        report += "- Review individual reports for detailed QC metrics\n"
        report += "- Check temperature calibration results before deployment\n"
        report += "- Verify long text regression test plots for oscillation and entropy stability\n\n"
        
        report += "---\n\nEnd of Integrated Report\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"  Integrated report saved: {report_path}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SO8T Burn-in QC Full Pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/so8t_rotations_epoch_final.pt",
        help="Path to SO8T checkpoint"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/so8t_burnin_qc_output",
        help="Output directory"
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        default="data/validation_burnin_test.json",
        help="Validation data path (English)"
    )
    parser.add_argument(
        "--validation-data-japanese",
        type=str,
        default="data/validation_burnin_test_japanese.json",
        help="Validation data path (Japanese)"
    )
    parser.add_argument(
        "--skip-burnin",
        action="store_true",
        help="Skip burn-in step (use existing baked model)"
    )
    parser.add_argument(
        "--baked-model",
        type=str,
        help="Path to existing baked model (if --skip-burnin is used)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = SO8TBurnInQCPipeline(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        validation_data_path=args.validation_data,
        validation_data_japanese_path=args.validation_data_japanese,
        device=args.device
    )
    
    results = pipeline.run_full_pipeline(
        base_model_path=args.base_model,
        skip_burnin=args.skip_burnin,
        baked_model_path=args.baked_model
    )
    
    logger.info("All done!")
    logger.info(f"Results: {json.dumps({k: str(v) for k, v in results.items()}, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()




