#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/Bテスト完全自動パイプライン

A/Bテスト実行 → 勝者モデル判定 → GGUF変換 → Ollamaインポート → 日本語パフォーマンステスト
"""

import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_ab_test_post_processing_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioNotifier:
    """音声通知クラス"""
    
    @staticmethod
    def play_notification():
        """音声通知を再生"""
        audio_path = Path(".cursor/marisa_owattaze.wav")
        if audio_path.exists():
            try:
                import subprocess
                subprocess.run([
                    "powershell", "-Command",
                    f"Add-Type -AssemblyName System.Windows.Forms; "
                    f"[System.Media.SoundPlayer]::new('{audio_path.absolute()}').Play(); "
                    f"Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green"
                ], check=False)
            except Exception as e:
                logger.warning(f"Failed to play audio notification: {e}")
        else:
            logger.warning(f"Audio file not found: {audio_path}")


class WinnerModelDeterminer:
    """勝者モデル判定クラス"""
    
    @staticmethod
    def determine_winner(ab_test_results_path: Path) -> Tuple[str, Dict]:
        """
        A/Bテスト結果から勝者モデルを判定
        
        Args:
            ab_test_results_path: A/Bテスト結果JSONファイルパス
        
        Returns:
            (winner_name, winner_metrics): 勝者モデル名とメトリクス
        """
        logger.info("="*80)
        logger.info("Determining Winner Model")
        logger.info("="*80)
        
        with open(ab_test_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        model_a_metrics = results.get('model_a', {})
        model_b_metrics = results.get('model_b', {})
        comparison = results.get('comparison', {})
        
        # 主要メトリクスで比較
        accuracy_a = model_a_metrics.get('accuracy', 0)
        accuracy_b = model_b_metrics.get('accuracy', 0)
        f1_macro_a = model_a_metrics.get('f1_macro', 0)
        f1_macro_b = model_b_metrics.get('f1_macro', 0)
        
        # 勝者判定（accuracy + F1 macroの平均）
        score_a = (accuracy_a + f1_macro_a) / 2
        score_b = (accuracy_b + f1_macro_b) / 2
        
        if score_b > score_a:
            winner_name = "SO8T Retrained"
            winner_metrics = model_b_metrics
            logger.info(f"[WINNER] SO8T Retrained Model (Score: {score_b:.4f} vs {score_a:.4f})")
        else:
            winner_name = "Original"
            winner_metrics = model_a_metrics
            logger.info(f"[WINNER] Original Model (Score: {score_a:.4f} vs {score_b:.4f})")
        
        logger.info(f"Accuracy: {accuracy_a:.4f} vs {accuracy_b:.4f}")
        logger.info(f"F1 Macro: {f1_macro_a:.4f} vs {f1_macro_b:.4f}")
        
        return winner_name, winner_metrics


class GGUFConverter:
    """GGUF変換クラス"""
    
    def __init__(self, convert_script_path: Path):
        """
        Args:
            convert_script_path: convert_hf_to_gguf.pyのパス
        """
        self.convert_script_path = Path(convert_script_path)
        
        if not self.convert_script_path.exists():
            raise FileNotFoundError(f"GGUF conversion script not found: {convert_script_path}")
    
    def convert_to_gguf(
        self,
        model_dir: Path,
        output_dir: Path,
        model_name: str,
        quantizations: list = ['f16', 'q8_0']
    ) -> Dict[str, Path]:
        """
        Hugging FaceモデルをGGUF形式に変換
        
        Args:
            model_dir: HFモデルディレクトリ
            output_dir: 出力ディレクトリ
            model_name: モデル名
            quantizations: 量子化タイプリスト
        
        Returns:
            gguf_files: 量子化タイプ -> GGUFファイルパスの辞書
        """
        logger.info("="*80)
        logger.info("Converting to GGUF Format")
        logger.info("="*80)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_files = {}
        
        for quant_type in quantizations:
            logger.info(f"Converting to {quant_type}...")
            
            output_file = output_dir / f"{model_name}_{quant_type}.gguf"
            
            cmd = [
                sys.executable,
                str(self.convert_script_path),
                str(model_dir),
                "--outfile", str(output_file),
                "--outtype", quant_type
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                logger.info(f"[OK] GGUF conversion completed: {output_file}")
                gguf_files[quant_type] = output_file
            except subprocess.CalledProcessError as e:
                logger.error(f"[ERROR] GGUF conversion failed for {quant_type}: {e}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                raise
        
        return gguf_files


class OllamaImporter:
    """Ollamaインポートクラス"""
    
    @staticmethod
    def create_modelfile(gguf_file: Path, model_name: str, output_path: Path) -> Path:
        """
        Modelfileを作成
        
        Args:
            gguf_file: GGUFファイルパス
            model_name: モデル名
            output_path: Modelfile出力パス
        
        Returns:
            modelfile_path: Modelfileパス
        """
        modelfile_content = f"""FROM {gguf_file.absolute()}

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"[OK] Modelfile created: {output_path}")
        return output_path
    
    @staticmethod
    def import_to_ollama(modelfile_path: Path, model_name: str) -> bool:
        """
        GGUFモデルをOllamaにインポート
        
        Args:
            modelfile_path: Modelfileパス
            model_name: Ollamaモデル名
        
        Returns:
            success: 成功フラグ
        """
        logger.info("="*80)
        logger.info("Importing to Ollama")
        logger.info("="*80)
        
        cmd = [
            "ollama",
            "create",
            f"{model_name}:latest",
            "-f",
            str(modelfile_path.absolute())
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logger.info(f"[OK] Model imported to Ollama: {model_name}:latest")
            logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Ollama import failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False


class CompleteABTestPostProcessingPipeline:
    """A/Bテスト完全自動パイプライン"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.base_model_path = Path(config['ab_test']['base_model'])
        self.retrained_model_path = Path(config['ab_test']['retrained_model'])
        self.test_data_path = Path(config['ab_test']['test_data'])
        self.ab_test_output_dir = Path(config['ab_test']['output_dir'])
        self.gguf_output_dir = Path(config['gguf']['output_dir'])
        self.convert_script_path = Path(config['gguf']['convert_script'])
        self.ollama_model_name = config['ollama']['model_name']
        self.japanese_test_output_dir = Path(config['japanese_test']['output_dir'])
        
        logger.info("="*80)
        logger.info("Complete A/B Test Post-Processing Pipeline Initialized")
        logger.info("="*80)
    
    def step1_run_ab_test(self) -> Path:
        """ステップ1: A/Bテスト実行"""
        logger.info("="*80)
        logger.info("STEP 1: Running A/B Test")
        logger.info("="*80)
        
        # A/Bテストスクリプトをsubprocessで実行
        ab_test_script = PROJECT_ROOT / "scripts" / "evaluation" / "ab_test_borea_phi35_original_vs_so8t.py"
        
        cmd = [
            sys.executable,
            str(ab_test_script),
            "--base-model", str(self.base_model_path),
            "--retrained-model", str(self.retrained_model_path),
            "--test-data", str(self.test_data_path),
            "--output-dir", str(self.ab_test_output_dir),
            "--device", self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logger.info("[OK] A/B test completed")
            logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] A/B test failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
        
        results_path = self.ab_test_output_dir / "ab_test_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"A/B test results not found: {results_path}")
        
        logger.info(f"[OK] A/B test results: {results_path}")
        
        AudioNotifier.play_notification()
        return results_path
    
    def step2_determine_winner(self, ab_test_results_path: Path) -> Tuple[str, Path]:
        """ステップ2: 勝者モデル判定"""
        logger.info("="*80)
        logger.info("STEP 2: Determining Winner Model")
        logger.info("="*80)
        
        winner_name, winner_metrics = WinnerModelDeterminer.determine_winner(ab_test_results_path)
        
        # 勝者モデルのパスを決定
        if winner_name == "SO8T Retrained":
            winner_model_path = self.retrained_model_path
        else:
            winner_model_path = self.base_model_path
        
        logger.info(f"[OK] Winner model: {winner_name} ({winner_model_path})")
        
        AudioNotifier.play_notification()
        return winner_name, winner_model_path
    
    def step3_convert_to_gguf(self, winner_model_path: Path, winner_name: str) -> Dict[str, Path]:
        """ステップ3: GGUF変換"""
        logger.info("="*80)
        logger.info("STEP 3: Converting to GGUF Format")
        logger.info("="*80)
        
        converter = GGUFConverter(self.convert_script_path)
        
        # モデル名を生成（ファイル名から）
        model_name = winner_name.lower().replace(' ', '_')
        output_dir = self.gguf_output_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gguf_files = converter.convert_to_gguf(
            model_dir=winner_model_path,
            output_dir=output_dir,
            model_name=model_name,
            quantizations=['f16', 'q8_0']
        )
        
        logger.info(f"[OK] GGUF conversion completed. Files: {list(gguf_files.keys())}")
        
        AudioNotifier.play_notification()
        return gguf_files
    
    def step4_import_to_ollama(self, gguf_files: Dict[str, Path], model_name: str) -> bool:
        """ステップ4: Ollamaインポート"""
        logger.info("="*80)
        logger.info("STEP 4: Importing to Ollama")
        logger.info("="*80)
        
        # Q8_0を使用（なければf16）
        gguf_file = gguf_files.get('q8_0') or gguf_files.get('f16')
        if not gguf_file:
            logger.error("[ERROR] No GGUF file available for Ollama import")
            return False
        
        # Modelfile作成
        modelfile_dir = Path("modelfiles")
        modelfile_path = modelfile_dir / f"{model_name}.modelfile"
        OllamaImporter.create_modelfile(gguf_file, model_name, modelfile_path)
        
        # Ollamaインポート
        success = OllamaImporter.import_to_ollama(modelfile_path, model_name)
        
        if success:
            logger.info(f"[OK] Model imported to Ollama: {model_name}:latest")
        else:
            logger.error(f"[ERROR] Failed to import model to Ollama")
        
        AudioNotifier.play_notification()
        return success
    
    def step5_run_japanese_performance_test(self, model_name: str) -> Path:
        """ステップ5: 日本語パフォーマンステスト実行"""
        logger.info("="*80)
        logger.info("STEP 5: Running Japanese Performance Test")
        logger.info("="*80)
        
        from scripts.testing.japanese_llm_performance_test import JapaneseLLMPerformanceTester
        
        tester = JapaneseLLMPerformanceTester(
            model_name=f"{model_name}:latest",
            output_dir=self.japanese_test_output_dir
        )
        
        results_path = tester.run_all_tests()
        
        logger.info(f"[OK] Japanese performance test completed. Results: {results_path}")
        
        AudioNotifier.play_notification()
        return results_path
    
    def run_complete_pipeline(self) -> Dict:
        """完全パイプライン実行"""
        logger.info("="*80)
        logger.info("Starting Complete A/B Test Post-Processing Pipeline")
        logger.info("="*80)
        
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # Step 1: A/Bテスト実行
            ab_test_results_path = self.step1_run_ab_test()
            pipeline_results['steps']['ab_test'] = {
                'status': 'completed',
                'results_path': str(ab_test_results_path)
            }
            
            # Step 2: 勝者モデル判定
            winner_name, winner_model_path = self.step2_determine_winner(ab_test_results_path)
            pipeline_results['steps']['winner_determination'] = {
                'status': 'completed',
                'winner_name': winner_name,
                'winner_model_path': str(winner_model_path)
            }
            
            # Step 3: GGUF変換
            gguf_files = self.step3_convert_to_gguf(winner_model_path, winner_name)
            pipeline_results['steps']['gguf_conversion'] = {
                'status': 'completed',
                'gguf_files': {k: str(v) for k, v in gguf_files.items()}
            }
            
            # Step 4: Ollamaインポート
            model_name = winner_name.lower().replace(' ', '_')
            ollama_success = self.step4_import_to_ollama(gguf_files, model_name)
            pipeline_results['steps']['ollama_import'] = {
                'status': 'completed' if ollama_success else 'failed',
                'model_name': f"{model_name}:latest"
            }
            
            # Step 5: 日本語パフォーマンステスト
            if ollama_success:
                japanese_test_results_path = self.step5_run_japanese_performance_test(model_name)
                pipeline_results['steps']['japanese_performance_test'] = {
                    'status': 'completed',
                    'results_path': str(japanese_test_results_path)
                }
            else:
                logger.warning("[SKIP] Japanese performance test skipped (Ollama import failed)")
                pipeline_results['steps']['japanese_performance_test'] = {
                    'status': 'skipped',
                    'reason': 'Ollama import failed'
                }
            
            # パイプライン結果保存
            results_path = self.ab_test_output_dir / "complete_pipeline_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
            
            logger.info("="*80)
            logger.info("[COMPLETE] Complete Pipeline Finished!")
            logger.info("="*80)
            logger.info(f"Pipeline results: {results_path}")
            
            AudioNotifier.play_notification()
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}", exc_info=True)
            pipeline_results['error'] = str(e)
            pipeline_results['status'] = 'failed'
            
            # エラー時も結果を保存
            results_path = self.ab_test_output_dir / "complete_pipeline_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
            
            AudioNotifier.play_notification()
            raise


def main():
    """メイン関数"""
    import yaml
    import torch
    
    parser = argparse.ArgumentParser(description="Complete A/B Test Post-Processing Pipeline")
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/complete_ab_test_post_processing_config.yaml'),
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # パイプライン実行
    pipeline = CompleteABTestPostProcessingPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    logger.info("="*80)
    logger.info("[SUCCESS] All steps completed!")
    logger.info("="*80)


if __name__ == '__main__':
    main()

