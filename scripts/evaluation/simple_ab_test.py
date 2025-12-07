#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプルなAEGIS v2.1 A/Bテストスクリプト
基本的な評価のみ実施
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleABTester:
    """シンプルA/Bテストクラス"""

    def __init__(self):
        self.base_model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        self.aegis_model_path = "H:/from_D/webdataset/models/final/aegis_v21_sft_hf"
        self.results_dir = Path("results/ab_test_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # テストプロンプト（ELYZA-100スタイル）
        self.test_prompts = [
            "日本の首都はどこですか？",
            "1 + 1 = ？",
            "量子力学について簡単に説明してください。",
            "次の文を英語に翻訳してください：「今日は良い天気ですね。」",
            "AIの将来についてどう思いますか？",
            "日本の伝統的な食べ物は何ですか？",
            "地球温暖化の原因は何ですか？",
            "プログラミング言語Pythonの特徴は何ですか？"
        ]

    def load_models(self):
        """モデル読み込み"""
        print("[LOAD] Loading models...")

        # Base model
        print("Loading base model...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # AEGIS model
        print("Loading AEGIS model...")
        self.aegis_tokenizer = AutoTokenizer.from_pretrained(self.aegis_model_path, local_files_only=True)
        # trust_remote_code=Falseで読み込み（modelingファイルがなくても動作）
        self.aegis_model = AutoModelForCausalLM.from_pretrained(
            self.aegis_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=False
        )

        print("[OK] Models loaded successfully")

    def generate_response(self, model, tokenizer, prompt, max_length=100):
        """応答生成"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # プロンプト部分を除去
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def evaluate_responses(self):
        """応答評価"""
        print("[EVAL] Evaluating responses...")

        results = {
            "base_model": {},
            "aegis_model": {}
        }

        for i, prompt in enumerate(self.test_prompts):
            print(f"Evaluating prompt {i+1}/{len(self.test_prompts)}: {prompt[:50]}...")

            # Base model
            try:
                base_response = self.generate_response(self.base_model, self.base_tokenizer, prompt)
                results["base_model"][f"prompt_{i}"] = {
                    "prompt": prompt,
                    "response": base_response,
                    "response_length": len(base_response)
                }
            except Exception as e:
                print(f"[ERROR] Base model failed on prompt {i}: {e}")
                results["base_model"][f"prompt_{i}"] = {
                    "prompt": prompt,
                    "response": "",
                    "response_length": 0,
                    "error": str(e)
                }

            # AEGIS model
            try:
                aegis_response = self.generate_response(self.aegis_model, self.aegis_tokenizer, prompt)
                results["aegis_model"][f"prompt_{i}"] = {
                    "prompt": prompt,
                    "response": aegis_response,
                    "response_length": len(aegis_response)
                }
            except Exception as e:
                print(f"[ERROR] AEGIS model failed on prompt {i}: {e}")
                results["aegis_model"][f"prompt_{i}"] = {
                    "prompt": prompt,
                    "response": "",
                    "response_length": 0,
                    "error": str(e)
                }

        return results

    def analyze_results(self, results):
        """結果分析"""
        print("[ANALYSIS] Analyzing results...")

        # 応答長の統計
        base_lengths = [r["response_length"] for r in results["base_model"].values() if r["response_length"] > 0]
        aegis_lengths = [r["response_length"] for r in results["aegis_model"].values() if r["response_length"] > 0]

        analysis = {
            "base_model": {
                "mean_length": float(np.mean(base_lengths)) if base_lengths else 0,
                "std_length": float(np.std(base_lengths)) if base_lengths else 0,
                "max_length": max(base_lengths) if base_lengths else 0,
                "min_length": min(base_lengths) if base_lengths else 0,
                "valid_responses": len(base_lengths)
            },
            "aegis_model": {
                "mean_length": float(np.mean(aegis_lengths)) if aegis_lengths else 0,
                "std_length": float(np.std(aegis_lengths)) if aegis_lengths else 0,
                "max_length": max(aegis_lengths) if aegis_lengths else 0,
                "min_length": min(aegis_lengths) if aegis_lengths else 0,
                "valid_responses": len(aegis_lengths)
            }
        }

        # 統計的有意性テスト
        if len(base_lengths) > 1 and len(aegis_lengths) > 1:
            try:
                t_stat, p_value = stats.ttest_ind(base_lengths, aegis_lengths, equal_var=False)
                analysis["statistics"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }
            except:
                analysis["statistics"] = {"error": "Could not perform statistical test"}

        return analysis

    def create_comparison_report(self, results, analysis):
        """比較レポート作成"""
        print("[REPORT] Creating comparison report...")

        report = f"""# AEGIS v2.1 vs Base Model Comparison Report

## Overview
This report compares the performance of AEGIS v2.1 (SO(8) optimized Phi-3.5) against the base model (Borea-Phi-3.5-mini-Instruct-Jp) on {len(self.test_prompts)} test prompts.

## Models Compared
- **Base Model**: {self.base_model_name}
- **AEGIS v2.1**: SO(8) optimized model with fine-tuning on 50,000 samples

## Response Length Statistics

### Base Model
- Mean Length: {analysis['base_model']['mean_length']:.1f} characters
- Std Deviation: {analysis['base_model']['std_length']:.1f}
- Max Length: {analysis['base_model']['max_length']}
- Min Length: {analysis['base_model']['min_length']}
- Valid Responses: {analysis['base_model']['valid_responses']}/{len(self.test_prompts)}

### AEGIS v2.1
- Mean Length: {analysis['aegis_model']['mean_length']:.1f} characters
- Std Deviation: {analysis['aegis_model']['std_length']:.1f}
- Max Length: {analysis['aegis_model']['max_length']}
- Min Length: {analysis['aegis_model']['min_length']}
- Valid Responses: {analysis['aegis_model']['valid_responses']}/{len(self.test_prompts)}

## Statistical Analysis
"""

        if "statistics" in analysis:
            stats_info = analysis["statistics"]
            if "error" not in stats_info:
                report += f"""- t-statistic: {stats_info['t_statistic']:.3f}
- p-value: {stats_info['p_value']:.4f}
- Significant difference: {'Yes' if stats_info['significant'] else 'No'} (p < 0.05)
"""

        report += f"""
## Sample Responses

### Prompt 1: {self.test_prompts[0]}
**Base Model:** {results['base_model']['prompt_0']['response'][:200]}{'...' if len(results['base_model']['prompt_0']['response']) > 200 else ''}

**AEGIS v2.1:** {results['aegis_model']['prompt_0']['response'][:200]}{'...' if len(results['aegis_model']['prompt_0']['response']) > 200 else ''}

### Prompt 2: {self.test_prompts[1]}
**Base Model:** {results['base_model']['prompt_1']['response'][:200]}{'...' if len(results['base_model']['prompt_1']['response']) > 200 else ''}

**AEGIS v2.1:** {results['aegis_model']['prompt_1']['response'][:200]}{'...' if len(results['aegis_model']['prompt_1']['response']) > 200 else ''}

## Conclusion
"""

        base_mean = analysis['base_model']['mean_length']
        aegis_mean = analysis['aegis_model']['mean_length']

        if aegis_mean > base_mean:
            report += f"AEGIS v2.1 generates longer responses on average ({aegis_mean:.1f} vs {base_mean:.1f} characters)."
        else:
            report += f"AEGIS v2.1 generates shorter responses on average ({aegis_mean:.1f} vs {base_mean:.1f} characters)."

        if "statistics" in analysis and "significant" in analysis["statistics"]:
            if analysis["statistics"]["significant"]:
                report += " The difference is statistically significant."
            else:
                report += " The difference is not statistically significant."

        return report

    def create_visualizations(self, results, analysis):
        """可視化作成"""
        print("[VISUAL] Creating visualizations...")

        # スタイル設定
        plt.style.use('default')
        sns.set_palette("husl")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 応答長比較
        models = ['Base Model', 'AEGIS v2.1']
        means = [analysis['base_model']['mean_length'], analysis['aegis_model']['mean_length']]
        stds = [analysis['base_model']['std_length'], analysis['aegis_model']['std_length']]

        bars = axes[0].bar(models, means, yerr=stds, capsize=5, alpha=0.8, color=['skyblue', 'lightcoral'])
        axes[0].set_title('Response Length Comparison', fontweight='bold')
        axes[0].set_ylabel('Mean Response Length (characters)')
        axes[0].grid(True, alpha=0.3)

        # 値ラベル追加
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + stds[models.index(bar.get_label() or 'Base Model')],
                       '.1f', ha='center', va='bottom', fontweight='bold')

        # プロンプト別比較
        prompt_labels = [f'P{i+1}' for i in range(len(self.test_prompts))]
        base_lengths = [results['base_model'][f'prompt_{i}']['response_length'] for i in range(len(self.test_prompts))]
        aegis_lengths = [results['aegis_model'][f'prompt_{i}']['response_length'] for i in range(len(self.test_prompts))]

        x = np.arange(len(prompt_labels))
        width = 0.35

        axes[1].bar(x - width/2, base_lengths, width, label='Base Model', alpha=0.8, color='skyblue')
        axes[1].bar(x + width/2, aegis_lengths, width, label='AEGIS v2.1', alpha=0.8, color='lightcoral')

        axes[1].set_title('Response Length by Prompt', fontweight='bold')
        axes[1].set_ylabel('Response Length (characters)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(prompt_labels, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.results_dir / "simple_ab_test_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[VISUAL] Plot saved to {plot_path}")
        plt.close()

    def run_test(self):
        """テスト実行"""
        print("[START] Simple A/B Test for AEGIS v2.1")
        print("=" * 50)

        try:
            # モデル読み込み
            self.load_models()

            # 応答生成
            results = self.evaluate_responses()

            # 結果分析
            analysis = self.analyze_results(results)

            # 可視化
            self.create_visualizations(results, analysis)

            # レポート作成
            report = self.create_comparison_report(results, analysis)

            # 保存
            with open(self.results_dir / "simple_ab_test_results.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "results": results,
                    "analysis": analysis
                }, f, indent=2, ensure_ascii=False)

            with open(self.results_dir / "simple_ab_test_report.md", 'w', encoding='utf-8') as f:
                f.write(report)

            print("\n[SUCCESS] Simple A/B test completed!")
            print(f"Results saved to: {self.results_dir}")

            # HFパッケージ作成
            self.create_hf_package(results, analysis)

            return results, analysis

        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def create_hf_package(self, results, analysis):
        """HFアップロード用パッケージ作成"""
        print("[HF-PACKAGE] Creating HF submission package")

        package_dir = Path("hf_upload_package")
        package_dir.mkdir(exist_ok=True)

        # 評価サマリー
        summary = {
            "model_name": "AEGIS v2.1",
            "base_model": self.base_model_name,
            "evaluation_type": "Simple A/B Test",
            "test_prompts": len(self.test_prompts),
            "response_comparison": {
                "base_model_avg_length": analysis['base_model']['mean_length'],
                "aegis_model_avg_length": analysis['aegis_model']['mean_length'],
                "length_difference": analysis['aegis_model']['mean_length'] - analysis['base_model']['mean_length']
            }
        }

        if "statistics" in analysis:
            summary["statistics"] = analysis["statistics"]

        # パッケージ保存
        with open(package_dir / "simple_evaluation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # グラフコピー
        import shutil
        plot_src = self.results_dir / "simple_ab_test_comparison.png"
        if plot_src.exists():
            shutil.copy2(plot_src, package_dir / "evaluation_plot.png")

        # レポートコピー
        shutil.copy2(self.results_dir / "simple_ab_test_report.md", package_dir / "SIMPLE_EVALUATION_REPORT.md")

        print(f"[HF-PACKAGE] Package created at: {package_dir}")
        print("Contents:")
        for item in package_dir.glob("*"):
            print(f"  - {item.name}")

def main():
    """メイン実行"""
    tester = SimpleABTester()
    results, analysis = tester.run_test()

    if results and analysis:
        print("\n🎵 Playing completion notification...")
        os.system('powershell -ExecutionPolicy Bypass -File "scripts/utils/play_audio_notification.ps1"')

if __name__ == "__main__":
    main()