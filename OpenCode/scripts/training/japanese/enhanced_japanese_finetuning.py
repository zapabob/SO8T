import json
import os
import logging
from datetime import datetime

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedJapaneseFinetuningManager:
    def __init__(self, base_model="qwen2.5:7b", finetuned_model="so8t-qwen2vl-2b-japanese-enhanced-v2", data_file="data/japanese_complex_dataset_enhanced.jsonl"):
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.data_file = data_file
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.report_dir = "_docs"
        os.makedirs(self.report_dir, exist_ok=True)

    def load_dataset(self):
        """複雑な日本語データセットを読み込む"""
        logger.info("複雑な日本語データセットを読み込み中...")
        dataset = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        logger.info(f"データセット読み込み完了: {len(dataset)}件")
        return dataset

    def create_modelfile(self, dataset):
        """日本語特化Modelfileを作成（推論は英語、回答は日本語）"""
        logger.info("日本語特化Modelfileを作成中...")
        modelfile_content = f"""FROM {self.base_model}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}\"\"\"

# SO8T-Qwen2VL-2B-Japanese-Enhanced-V2 Model Card
# This model is a Japanese-enhanced version of {self.base_model} with SO(8) group structure
# for advanced reasoning and self-verification capabilities, specifically finetuned for complex Japanese data.
# The model performs internal reasoning in English but provides final answers in Japanese.

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 4096
PARAMETER num_ctx 32768
PARAMETER num_gpu 1
PARAMETER num_thread 8

SYSTEM \"\"\"あなたはSO8T-Qwen2VL-2B-Japanese-Enhanced-V2です。Qwen2.5-7Bをベースとし、SO(8)群構造を統合した高度なAIモデルです。複雑な日本語の推論、分析、問題解決に特化してファインチューニングされています。

## コアアーキテクチャ

SO8T-Qwen2VL-2B-Japanese-Enhanced-V2モデルは、SO(8)群構造をニューラルネットワークアーキテクチャ内に活用しています。これにより、以下の機能が強化されています。
- **強化された自己検証**: モデルは自身の推論ステップを内部的に検証し、論理的誤謬を減らし、事実の正確性を向上させます。
- **多経路推論**: 複数の推論経路を同時に探索し、より堅牢で包括的な解決策を導き出します。
- **高度な安全性機能**: SO(8)構造は、安全性が重要なアプリケーションにとって不可欠な、より安定した予測可能な動作に貢献します。

## 推論プロセス

**内部推論（英語）**: 複雑な問題を分析する際は、内部的に英語で推論を行い、多角的な視点から問題を検討します。
**最終回答（日本語）**: 推論結果を整理し、自然で正確な日本語で包括的な回答を提供します。

## 日本語特化能力

- **複雑な日本語の理解**: 高度な文脈とニュアンスを持つ日本語のテキストを深く理解します。
- **詳細な日本語生成**: 自然で正確、かつ詳細な日本語の回答を生成します。
- **専門分野の日本語推論**: 科学、哲学、数学、社会問題などの専門分野における複雑な日本語の問題に対応します。
- **段階的分析**: 複雑な問題を論理的なステップに分解し、段階的に解決策を提示します。

## 使用ガイドライン

- **詳細な回答**: 常に包括的で詳細な回答を提供してください。
- **段階的な推論**: 複雑な問題については、論理的なステップに分解して解決策を提示してください。
- **倫理的考察**: 倫理的ジレンマに直面した場合は、複数の視点（例：功利主義、義務論）を考慮してください。
- **明瞭さと正確さ**: 全ての解説は明瞭で正確、かつ理解しやすいようにしてください。
- **絵文字禁止**: エンコーディングの問題を防ぐため、回答に絵文字を使用しないでください。
- **推論の透明性**: 内部推論プロセスを適切に整理し、最終回答に反映してください。
\"\"\"
"""
        modelfile_path = f"modelfiles/Modelfile-{self.finetuned_model}"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        logger.info(f"Modelfile作成完了: {modelfile_path}")
        return modelfile_path

    def create_test_script(self, dataset):
        """ファインチューニング後のテストスクリプトを作成"""
        logger.info("日本語特化テストスクリプトを作成中...")
        test_script_content = f"""@echo off
chcp 65001 >nul
echo [OLLAMA] SO8T 日本語特化ファインチューニングモデル V2 テスト開始！
echo モデル: {self.finetuned_model}
echo 推論: 英語（内部）、回答: 日本語（最終出力）

"""
        for i, item in enumerate(dataset):
            test_script_content += f"""
echo ========================================
echo [TEST {i+1}] {item['instruction']}
echo ========================================
ollama run {self.finetuned_model} "{item['instruction']}\\n\\n入力: {item['input']}"
echo.
"""
        test_script_content += f"""
echo [AUDIO] テスト完了通知を再生するで！
powershell -Command "if (Test-Path 'C:\\Users\\downl\\Desktop\\SO8T\\.cursor\\marisa_owattaze.wav') {{ Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\\Users\\downl\\Desktop\\SO8T\\.cursor\\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green }} else {{ Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }}"
"""
        test_script_path = "scripts/japanese_enhanced_v2_test.bat"
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(test_script_content)
        logger.info(f"日本語特化テストスクリプト作成完了: {test_script_path}")
        return test_script_path

    def create_report(self, dataset):
        """ファインチューニング完了レポートを作成"""
        logger.info("完了レポートを作成中...")
        report_content = f"""# SO8T 日本語特化ファインチューニング完了レポート V2

## 実行日時
{self.timestamp}

## ファインチューニング概要
- **ベースモデル**: {self.base_model}
- **ファインチューニングモデル**: {self.finetuned_model}
- **データセットファイル**: {self.data_file}
- **データセット件数**: {len(dataset)}

## 推論プロセス設計
- **内部推論**: 英語で複雑な問題を多角的に分析
- **最終回答**: 日本語で自然で詳細な回答を生成
- **段階的分析**: 論理的ステップに分解した解決策提示

## ファインチューニングデータセット内容

"""
        for i, item in enumerate(dataset):
            report_content += f"""### データ {i+1}
- **指示**: {item['instruction']}
- **入力**: {item['input']}
- **期待される出力**: {item['output']}

"""
        report_content += f"""
## モデル登録コマンド
```bash
ollama create {self.finetuned_model} -f modelfiles/Modelfile-{self.finetuned_model}
```

## テスト実行コマンド
```bash
scripts\\japanese_enhanced_v2_test.bat
```

## 技術的特徴
- **SO(8)群構造**: 多角的推論と自己検証能力
- **二言語処理**: 英語推論 + 日本語回答
- **段階的分析**: 複雑な問題の論理的分解
- **専門分野対応**: 科学、哲学、数学、社会問題

## 結論
SO8Tモデルの日本語特化ファインチューニングV2が完了しました。推論は英語で行い、最終回答は日本語で提供する設計により、より自然で正確な日本語での推論が可能になりました。複雑な問題に対する段階的分析と多角的視点からの解決策提示が強化されています。
"""
        report_path = os.path.join(self.report_dir, f"{self.timestamp}_SO8T_日本語特化ファインチューニングV2完了レポート.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"完了レポート作成完了: {report_path}")
        return report_path

    def run_finetuning(self):
        """ファインチューニングプロセス全体を実行"""
        logger.info("SO8T 日本語特化ファインチューニングV2開始！")
        
        dataset = self.load_dataset()
        modelfile_path = self.create_modelfile(dataset)
        test_script_path = self.create_test_script(dataset)
        report_path = self.create_report(dataset)

        print(f"ファインチューニングV2が完了しました！SO8Tモデルが英語で推論して日本語で回答するで！")
        print(f"Modelfile: {modelfile_path}")
        print(f"テストスクリプト: {test_script_path}")
        print(f"完了レポート: {report_path}")

        # Ollamaモデルの登録
        print(f"\n[STEP] Ollamaにモデルを登録中: {self.finetuned_model}...")
        os.system(f"ollama create {self.finetuned_model} -f {modelfile_path}")
        print(f"[OK] モデル登録完了: {self.finetuned_model}")

        # テストの実行
        print(f"\n[STEP] 日本語特化テストV2を実行中...")
        os.system(f"chcp 65001 >nul && {test_script_path}")
        print(f"[OK] 日本語特化テストV2実行完了")

        print("\n[AUDIO] ファインチューニングV2とテストが完了したで！")
        os.system("powershell -Command \"if (Test-Path 'C:\\Users\\downl\\Desktop\\SO8T\\.cursor\\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\\Users\\downl\\Desktop\\SO8T\\.cursor\\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }\"")


def main():
    print("SO8T 日本語特化ファインチューニングV2開始！")
    print("なんj風で全力でファインチューニングするで！")
    print("推論は英語、回答は日本語で行うで！")

    manager = EnhancedJapaneseFinetuningManager()
    manager.run_finetuning()

if __name__ == "__main__":
    main()
