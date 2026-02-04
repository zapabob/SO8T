#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run AEGIS-phi3.5-v2.0 Training Pipeline

データセット生成からHFモデル統合トレーニングまでの一括実行スクリプト
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import warnings

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class AEGISTrainingPipeline:
    """AEGISトレーニングパイプライン"""

    def __init__(self, args):
        self.args = args
        self.project_root = PROJECT_ROOT

        # パス設定
        self.dataset_script = self.project_root / "scripts" / "data" / "enhance_nobel_fields_datasets.py"
        self.training_script = self.project_root / "scripts" / "training" / "train_nobel_fields_hf_integration.py"
        self.audio_notification = self.project_root / ".cursor" / "marisa_owattaze.wav"

        # 出力ディレクトリ
        self.dataset_dir = self.project_root / "data" / "aegis_phi35_v2_datasets"
        self.model_dir = self.project_root / "outputs" / "aegis_phi35_v2_integrated"

    def play_audio_notification(self, message: str = "処理完了"):
        """オーディオ通知を再生"""
        try:
            import winsound
            if self.audio_notification.exists():
                # PowerShellでオーディオ再生
                ps_command = f"""
                Write-Host "[AUDIO] {message} - 通知再生中..." -ForegroundColor Green
                $audioFile = "{self.audio_notification}"
                if (Test-Path $audioFile) {{
                    try {{
                        Add-Type -AssemblyName System.Windows.Forms
                        $player = New-Object System.Media.SoundPlayer $audioFile
                        $player.PlaySync()
                        Write-Host "[OK] オーディオ通知再生成功" -ForegroundColor Green
                    }} catch {{
                        Write-Host "[WARNING] オーディオ再生失敗: $($_.Exception.Message)" -ForegroundColor Yellow
                        [System.Console]::Beep(1000, 500)
                    }}
                }} else {{
                    Write-Host "[WARNING] オーディオファイルが見つかりません" -ForegroundColor Yellow
                    [System.Console]::Beep(1000, 500)
                }}
                """
                subprocess.run(["powershell", "-Command", ps_command], check=False)
            else:
                winsound.Beep(1000, 500)
        except Exception as e:
            print(f"[WARNING] オーディオ通知失敗: {e}")
            try:
                import winsound
                winsound.Beep(800, 1000)
            except:
                pass

    def run_command(self, command: list, description: str, cwd=None):
        """コマンド実行"""
        print(f"\n=== {description} ===")
        print(f"コマンド: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=False,  # リアルタイム出力
                text=True,
                encoding='utf-8'
            )

            if result.returncode == 0:
                print(f"[OK] {description} 完了")
                self.play_audio_notification(f"{description}完了")
                return True
            else:
                print(f"[ERROR] {description} 失敗 (コード: {result.returncode})")
                return False

        except Exception as e:
            print(f"[ERROR] {description} 実行エラー: {e}")
            return False

    def check_prerequisites(self):
        """前提条件チェック"""
        print("=== 前提条件チェック ===")

        # Pythonスクリプト存在チェック
        required_scripts = [
            self.dataset_script,
            self.training_script
        ]

        for script in required_scripts:
            if not script.exists():
                print(f"[ERROR] スクリプトが見つかりません: {script}")
                return False
            print(f"[OK] {script.name}")

        # データディレクトリ作成
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        print("[OK] 前提条件チェック完了")
        return True

    def generate_datasets(self):
        """データセット生成"""
        command = [
            sys.executable, "-m", "src.data.enhance_nobel_fields_datasets"
        ]

        return self.run_command(command, "AEGISデータセット生成")

    def train_model(self):
        """モデルトレーニング"""
        command = [
            sys.executable, "-m", "src.training.train_nobel_fields_hf_integration",
            "--model_name", self.args.model_name,
            "--output_dir", str(self.model_dir),
            "--epochs", str(self.args.epochs),
            "--batch_size", str(self.args.batch_size),
            "--learning_rate", str(self.args.learning_rate),
            "--enable_mathematical_reasoning", str(self.args.enable_mathematical_reasoning).lower(),
            "--reasoning_format", self.args.reasoning_format
        ]

        if self.args.use_4bit:
            command.append("--use_4bit")

        if self.args.test_after_training:
            command.append("--test_after_training")

        return self.run_command(command, "AEGISモデルトレーニング")

    def create_training_summary(self):
        """トレーニングサマリー作成"""
        print("\n=== トレーニングサマリー作成 ===")

        summary = {
            "model_name": self.args.model_name,
            "training_completed": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_dir),
            "model_path": str(self.model_dir),
            "training_config": {
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "learning_rate": self.args.learning_rate,
                "use_4bit": self.args.use_4bit,
                "mathematical_reasoning": self.args.enable_mathematical_reasoning,
                "reasoning_format": self.args.reasoning_format
            },
            "theories_integrated": [
                "URT (Unified Representation Theorem)",
                "NC-KART★ (Non-Commutative Kolmogorov-Arnold Theory)",
                "SO(8) Enhanced Adapter",
                "Quadruple Thinking Engine"
            ]
        }

        summary_file = self.model_dir / "aegis_phi35_v2_training_summary.json"
        try:
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[OK] トレーニングサマリー保存: {summary_file}")
        except Exception as e:
            print(f"[WARNING] サマリー保存失敗: {e}")

    def run_pipeline(self):
        """パイプライン実行"""
        print("🎯 AEGIS-phi3.5-v2.0 トレーニングパイプライン開始")
        print("=" * 60)
        print(f"モデル名: {self.args.model_name}")
        print(f"データセット: {self.dataset_dir}")
        print(f"出力先: {self.model_dir}")
        print(f"エポック数: {self.args.epochs}")
        print(f"バッチサイズ: {self.args.batch_size}")
        print("=" * 60)

        # 前提条件チェック
        if not self.check_prerequisites():
            print("[ERROR] 前提条件チェック失敗")
            return False

        # ステップ1: データセット生成
        if not self.generate_datasets():
            print("[ERROR] データセット生成失敗")
            return False

        # ステップ2: モデルトレーニング
        if not self.train_model():
            print("[ERROR] モデルトレーニング失敗")
            return False

        # ステップ3: サマリー作成
        self.create_training_summary()

        print("\n🎉 AEGIS-phi3.5-v2.0 パイプライン完了！")
        print("高度知能AIシステムの統合が完了しました。")
        print(f"モデル保存先: {self.model_dir}")

        # 最終オーディオ通知
        self.play_audio_notification("AEGISトレーニング完了")

        return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Run AEGIS-phi3.5-v2.0 Training Pipeline")
    parser.add_argument("--model_name", type=str, default="AEGIS-phi3.5-v2.0",
                       help="モデル名")
    parser.add_argument("--epochs", type=int, default=3,
                       help="トレーニングエポック数")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="バッチサイズ")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="学習率")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="4bit量子化を使用")
    parser.add_argument("--enable_mathematical_reasoning", action="store_true", default=True,
                       help="数学推論機能を有効化")
    parser.add_argument("--reasoning_format", type=str, default="nobel_fields",
                       choices=["standard", "nobel_fields"],
                       help="推論フォーマット")
    parser.add_argument("--test_after_training", action="store_true", default=True,
                       help="トレーニング後にテスト実行")
    parser.add_argument("--skip_dataset_generation", action="store_true", default=False,
                       help="データセット生成をスキップ")

    args = parser.parse_args()

    # パイプライン実行
    pipeline = AEGISTrainingPipeline(args)

    success = pipeline.run_pipeline()

    if success:
        print("\n✅ AEGIS-phi3.5-v2.0 トレーニング成功！")
        print("HFモデルにノーベル賞・フィールズ賞級の推論機能が統合されました。")
    else:
        print("\n❌ AEGISトレーニング失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
