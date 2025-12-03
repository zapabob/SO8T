#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLPO学習の完全自動化動作チェック
科学・数学SFT + NKAT理論 + 薬物NSFWデータ の統合テスト

実行方法:
python run_rlpo_science_nsfw_automated_test.py

特徴:
- データセット整合性チェック
- モデルロードテスト
- NKATアダプター統合テスト
- ミニトレーニング実行
- 結果評価とレポート生成
"""

import os
import json
import torch
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RLPOTestSuite:
    """RLPO学習の完全自動化テストスイート"""

    def __init__(self):
        self.results = {}
        self.test_dir = Path("test_results") / f"rlpo_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def run_all_tests(self):
        """全テスト実行"""
        logger.info("🚀 Starting RLPO Automated Test Suite...")

        try:
            # 1. 環境チェック
            self.test_environment()

            # 2. データセットチェック
            self.test_datasets()

            # 3. モデルロードテスト
            self.test_model_loading()

            # 4. NKATアダプター統合テスト
            self.test_nkat_integration()

            # 5. ミニトレーニング実行
            self.test_mini_training()

            # 6. 結果レポート生成
            self.generate_report()

            logger.info("✅ All tests completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            self.results['final_status'] = 'FAILED'
            self.generate_report()
            return False

    def test_environment(self):
        """環境チェック"""
        logger.info("🔍 Testing environment...")

        results = {}

        # Pythonバージョン
        results['python_version'] = sys.version

        # PyTorch
        results['torch_version'] = torch.__version__
        results['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results['cuda_version'] = torch.version.cuda
            results['gpu_count'] = torch.cuda.device_count()
            results['gpu_name'] = torch.cuda.get_device_name(0)

        # 必要なモジュール
        required_modules = ['transformers', 'peft', 'datasets', 'accelerate']
        for module in required_modules:
            try:
                __import__(module)
                results[f'{module}_available'] = True
            except ImportError:
                results[f'{module}_available'] = False

        # NKATモジュール
        try:
            from scripts.models.so8t_residual_adapter import SO8ResidualAdapter
            results['nkat_available'] = True
        except ImportError:
            results['nkat_available'] = False

        self.results['environment'] = results
        logger.info("✅ Environment test completed")

    def test_datasets(self):
        """データセットチェック"""
        logger.info("📊 Testing datasets...")

        results = {}

        # 科学データセット
        science_path = Path("data/science_reasoning_dataset_final.jsonl")
        if science_path.exists():
            with open(science_path, 'r', encoding='utf-8') as f:
                science_count = sum(1 for line in f if line.strip())
            results['science_dataset'] = {'exists': True, 'count': science_count}
        else:
            results['science_dataset'] = {'exists': False, 'count': 0}

        # NSFW薬物データセット
        nsfw_path = Path("data/nsfw_drug_detection/nsfw_drug_mixed_dataset.jsonl")
        if nsfw_path.exists():
            with open(nsfw_path, 'r', encoding='utf-8') as f:
                nsfw_count = sum(1 for line in f if line.strip())
            results['nsfw_dataset'] = {'exists': True, 'count': nsfw_count}
        else:
            results['nsfw_dataset'] = {'exists': False, 'count': 0}

        self.results['datasets'] = results
        logger.info("✅ Dataset test completed")

    def test_model_loading(self):
        """モデルロードテスト"""
        logger.info("🤖 Testing model loading...")

        results = {}

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_name = "microsoft/phi-3.5-mini-instruct"

            # トークナイザーロード
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            results['tokenizer_loaded'] = True

            # モデルロード（最小限）
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True
            )
            results['model_loaded'] = True
            results['model_params'] = sum(p.numel() for p in model.parameters())

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Model loading failed: {e}")

        self.results['model_loading'] = results
        logger.info("✅ Model loading test completed")

    def test_nkat_integration(self):
        """NKATアダプター統合テスト"""
        logger.info("🧬 Testing NKAT integration...")

        results = {}

        try:
            from scripts.models.so8t_residual_adapter import SO8ResidualAdapter, NKATMLPWrapper

            # アダプター作成テスト
            adapter = SO8ResidualAdapter(hidden_size=1024)
            results['adapter_created'] = True

            # 統計取得テスト
            stats = adapter.get_adapter_stats()
            results['adapter_stats'] = stats
            results['stats_keys'] = list(stats.keys())

            # フォワードテスト
            x = torch.randn(2, 10, 1024)
            out = adapter(x)
            results['forward_pass'] = True
            results['output_shape'] = list(out.shape)

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"NKAT integration failed: {e}")

        self.results['nkat_integration'] = results
        logger.info("✅ NKAT integration test completed")

    def test_mini_training(self):
        """ミニトレーニング実行"""
        logger.info("🎯 Running mini training test...")

        results = {}

        try:
            # ミニトレーニングスクリプト実行
            cmd = [
                sys.executable,
                "scripts/training/rlpo_science_nsfw_automated.py",
                "--max_steps", "10",  # 最小ステップ
                "--batch_size", "1",
                "--output_dir", str(self.test_dir / "mini_training")
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分タイムアウト
            )

            results['return_code'] = result.returncode
            results['stdout'] = result.stdout[-1000:]  # 最後1000文字
            results['stderr'] = result.stderr[-1000:]

            if result.returncode == 0:
                results['success'] = True
                logger.info("✅ Mini training completed successfully")
            else:
                results['success'] = False
                logger.error(f"❌ Mini training failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            results['error'] = 'Timeout'
            logger.error("❌ Mini training timed out")
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"❌ Mini training failed: {e}")

        self.results['mini_training'] = results

    def generate_report(self):
        """結果レポート生成"""
        logger.info("📋 Generating test report...")

        report_path = self.test_dir / "rlpo_test_report.json"

        # 最終ステータス判定
        all_passed = True
        for test_name, test_results in self.results.items():
            if test_name != 'final_status':
                if isinstance(test_results, dict) and 'error' in test_results:
                    all_passed = False
                    break
                elif test_name == 'mini_training' and not test_results.get('success', False):
                    all_passed = False
                    break

        self.results['final_status'] = 'PASSED' if all_passed else 'FAILED'

        # JSONレポート保存
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # コンソールレポート
        print("\n" + "="*60)
        print("🎯 RLPO AUTOMATED TEST REPORT")
        print("="*60)
        print(f"Test Directory: {self.test_dir}")
        print(f"Final Status: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        print(f"Report saved to: {report_path}")
        print("="*60)

        # 詳細表示
        for test_name, results in self.results.items():
            if test_name != 'final_status':
                status = "✅" if self._is_test_passed(test_name, results) else "❌"
                print(f"{status} {test_name.replace('_', ' ').title()}")

        print("="*60)

    def _is_test_passed(self, test_name, results):
        """テスト成功判定"""
        if not isinstance(results, dict):
            return True

        if 'error' in results:
            return False

        if test_name == 'mini_training':
            return results.get('success', False)

        if test_name == 'datasets':
            return (results.get('science_dataset', {}).get('exists', False) and
                   results.get('nsfw_dataset', {}).get('exists', False))

        return True


def main():
    """メイン実行関数"""
    print("🚀 RLPO Science + NSFW + NKAT Automated Test Suite")
    print("=" * 60)

    test_suite = RLPOTestSuite()
    success = test_suite.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()



