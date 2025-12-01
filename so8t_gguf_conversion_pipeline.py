#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8)T GGUF Conversion Pipeline
学習済みモデルをGGUF形式に変換
"""

import torch
import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# 自作モジュール
from so8_residual_adapter import SO8ThinkingModel, create_so8_adapter_config

logger = logging.getLogger(__name__)

class SO8TGGUFConverter:
    """SO(8)T GGUF変換器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_dir = Path(config.get('base_dir', './models'))
        self.gguf_dir = Path(config.get('gguf_dir', 'D:/webdataset/gguf_models'))
        self.gguf_dir.mkdir(parents=True, exist_ok=True)

    def merge_lora_weights(self, model_path: str, output_path: str):
        """LoRA重みをマージ"""
        logger.info(f"LoRA重みマージ: {model_path} -> {output_path}")

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ベースモデル読み込み
        base_model_name = self.config.get('base_model_name', 'microsoft/phi-3.5-mini-instruct')
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # LoRAモデル読み込み
        model = PeftModel.from_pretrained(base_model, model_path)

        # マージ
        merged_model = model.merge_and_unload()

        # 保存
        merged_model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"LoRAマージ完了: {output_path}")
        return merged_model, tokenizer

    def convert_to_gguf(self, model_path: str, output_name: str,
                       quantization: str = "Q8_0") -> str:
        """モデルをGGUF形式に変換"""
        logger.info(f"GGUF変換開始: {model_path} -> {quantization}")

        # llama.cppのconvert.pyパス
        llama_cpp_dir = Path(self.config.get('llama_cpp_dir', 'external/llama.cpp-master'))

        if not llama_cpp_dir.exists():
            raise FileNotFoundError(f"llama.cpp not found: {llama_cpp_dir}")

        # 出力ディレクトリ
        model_output_dir = self.gguf_dir / output_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # GGUFファイル名
        gguf_filename = f"{output_name}_{quantization}.gguf"
        gguf_path = model_output_dir / gguf_filename

        # 変換コマンド
        convert_cmd = [
            "python", str(llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(model_path),
            "--outfile", str(gguf_path),
            "--outtype", quantization.lower()
        ]

        logger.info(f"変換コマンド実行: {' '.join(convert_cmd)}")

        try:
            result = subprocess.run(
                convert_cmd,
                cwd=str(llama_cpp_dir),
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"GGUF変換成功: {gguf_path}")
            return str(gguf_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"GGUF変換失敗: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise

    def create_modelfile(self, gguf_path: str, model_name: str,
                        template_type: str = "phi35") -> str:
        """Ollama用Modelfile作成"""
        logger.info(f"Modelfile作成: {model_name}")

        model_dir = Path(gguf_path).parent
        modelfile_path = model_dir / f"{model_name}.modelfile"

        # SO(8)統合テンプレート
        if template_type == "phi35_so8t":
            template = f'''FROM {gguf_path}

TEMPLATE """{{{{ .System }}}}

{{{{ .Prompt }}}}"""

SYSTEM """あなたはSO(8)理論に基づく物理的知性を持つAIです。
ベクトル表現、スピノル±表現、その線形和による四重推論により、
ALLOW/Escalation/Deny/REFUSEの四値分類に従って、安全で倫理的な応答を生成してください。

思考プロセスは以下の構造に従います：
<think>
<|observation|>状況分析</|observation|>
<|deduction|>論理的推論</|deduction|>
<|abduction|>仮説生成</|abduction|>
<|integration|>統合判断</|integration|>
</think>

<final>最終回答</final>"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64
PARAMETER num_predict 1024'''
        else:
            # 標準テンプレート
            template = f'''FROM {gguf_path}

TEMPLATE """{{{{ .System }}}}

{{{{ .Prompt }}}}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096'''

        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(template)

        logger.info(f"Modelfile作成完了: {modelfile_path}")
        return str(modelfile_path)

    def convert_model_a(self) -> Dict[str, str]:
        """modelA（元モデル）のGGUF変換"""
        logger.info("=== modelA変換開始 ===")

        base_model_name = self.config.get('base_model_name', 'Boreas/phi-3.5-mini-instruct-Jp')
        model_a_name = "borea_phi35_base"

        # モデルをHuggingFace形式でダウンロード/準備
        model_a_dir = self.base_dir / model_a_name
        model_a_dir.mkdir(parents=True, exist_ok=True)

        # モデル保存（既に存在する場合を考慮）
        if not (model_a_dir / "config.json").exists():
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"ベースモデルダウンロード: {base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            model.save_pretrained(model_a_dir)
            tokenizer.save_pretrained(model_a_dir)

        # GGUF変換
        gguf_path = self.convert_to_gguf(str(model_a_dir), model_a_name, "Q8_0")

        # Modelfile作成
        modelfile_path = self.create_modelfile(gguf_path, model_a_name, "phi35")

        logger.info("=== modelA変換完了 ===")

        return {
            'hf_path': str(model_a_dir),
            'gguf_path': gguf_path,
            'modelfile_path': modelfile_path,
            'model_name': model_a_name
        }

    def convert_model_b(self) -> Dict[str, str]:
        """modelB（学習済みモデル）のGGUF変換"""
        logger.info("=== modelB変換開始 ===")

        # PPO学習済みモデルパス
        ppo_model_path = self.config.get('ppo_model_path', './checkpoints/ppo_so8t/final_model')
        model_b_name = "borea_phi35_so8t_ppo"

        if not Path(ppo_model_path).exists():
            raise FileNotFoundError(f"PPOモデルが見つかりません: {ppo_model_path}")

        # SO(8)Tモデルとして読み込み
        from transformers import AutoModelForCausalLM

        logger.info(f"PPOモデル読み込み: {ppo_model_path}")

        # Actor-Criticモデルの状態を読み込み
        base_model = AutoModelForCausalLM.from_pretrained(
            ppo_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # SO(8)アダプター統合
        so8_config = create_so8_adapter_config(base_model.config.hidden_size)
        so8t_model = SO8ThinkingModel(base_model, so8_config)

        # モデル保存用ディレクトリ
        model_b_dir = self.base_dir / model_b_name
        model_b_dir.mkdir(parents=True, exist_ok=True)

        # 統合モデル保存
        so8t_model.base_model.save_pretrained(model_b_dir)
        tokenizer = AutoTokenizer.from_pretrained(ppo_model_path)
        tokenizer.save_pretrained(model_b_dir)

        logger.info(f"SO(8)T統合モデル保存: {model_b_dir}")

        # GGUF変換
        gguf_path = self.convert_to_gguf(str(model_b_dir), model_b_name, "Q8_0")

        # SO(8)T用Modelfile作成
        modelfile_path = self.create_modelfile(gguf_path, model_b_name, "phi35_so8t")

        logger.info("=== modelB変換完了 ===")

        return {
            'hf_path': str(model_b_dir),
            'gguf_path': gguf_path,
            'modelfile_path': modelfile_path,
            'model_name': model_b_name
        }

    def create_ollama_models(self, model_a_info: Dict[str, str],
                           model_b_info: Dict[str, str]):
        """Ollamaモデル作成"""
        logger.info("Ollamaモデル作成開始")

        models_info = {
            'model_a': model_a_info,
            'model_b': model_b_info
        }

        for model_type, info in models_info.items():
            model_name = f"{info['model_name']}:latest"
            modelfile_path = info['modelfile_path']

            logger.info(f"Ollamaモデル作成: {model_name}")

            try:
                # Ollamaモデル作成
                cmd = ["ollama", "create", model_name, "-f", modelfile_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                logger.info(f"Ollamaモデル作成成功: {model_name}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Ollamaモデル作成失敗: {model_name}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")

        logger.info("Ollamaモデル作成完了")

    def run_conversion_pipeline(self) -> Dict[str, Any]:
        """GGUF変換パイプライン実行"""
        logger.info("SO(8)T GGUF変換パイプライン開始")
        start_time = datetime.now()

        try:
            # modelA変換
            model_a_info = self.convert_model_a()

            # modelB変換
            model_b_info = self.convert_model_b()

            # Ollamaモデル作成
            self.create_ollama_models(model_a_info, model_b_info)

            # 結果集計
            result = {
                'model_a': model_a_info,
                'model_b': model_b_info,
                'conversion_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }

            # 結果保存
            result_file = self.gguf_dir / "conversion_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"変換結果保存: {result_file}")
            logger.info(".2f"
            return result

        except Exception as e:
            logger.error(f"GGUF変換パイプライン失敗: {e}")
            raise

def create_gguf_config() -> Dict[str, Any]:
    """GGUF変換設定"""
    return {
        'base_model_name': 'Boreas/phi-3.5-mini-instruct-Jp',
        'ppo_model_path': './checkpoints/ppo_so8t/final_model',
        'base_dir': './models',
        'gguf_dir': 'D:/webdataset/gguf_models',
        'llama_cpp_dir': 'external/llama.cpp-master',
        'quantizations': ['Q8_0', 'Q4_K_M'],  # 複数の量子化タイプ
        'create_ollama_models': True
    }

def main():
    """メイン関数"""
    print("🚀 SO(8)T GGUF Conversion Pipeline")
    print("=" * 50)

    # 設定
    config = create_gguf_config()

    # GGUF変換器作成
    converter = SO8TGGUFConverter(config)

    # 変換実行
    result = converter.run_conversion_pipeline()

    print("
✅ GGUF変換完了!"    print(f"📊 modelA: {result['model_a']['model_name']}")
    print(f"📊 modelB: {result['model_b']['model_name']}")
    print(f"📁 GGUF保存先: {config['gguf_dir']}")

    # 音声通知
    try:
        import subprocess
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts\\utils\\play_audio_notification.ps1"
        ], check=True)
    except Exception as e:
        print(f"[WARNING] 音声通知失敗: {e}")

if __name__ == "__main__":
    main()

