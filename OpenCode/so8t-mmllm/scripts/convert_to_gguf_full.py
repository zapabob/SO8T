#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3モデルGGUF変換・Ollama配備パイプライン
- ①ベースPhi-4-mini-instruct（GGUF化のみ）
- ②SO8T統合版（焼きこみ前、GGUF化）
- ③ファインチューニング後（焼きこみ後、GGUF化）
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm


# [OK] 変換設定
MODELS_CONFIG = {
    "base_phi4": {
        "name": "phi4-base",
        "source": "microsoft/phi-4-mini-instruct",
        "apply_burnin": False,
        "quantizations": ["Q8_0", "Q4_K_M"]
    },
    "so8t_integrated": {
        "name": "phi4-so8t",
        "source": "phi4_so8t_integrated",  # SO8T統合済みモデルパス
        "apply_burnin": False,
        "quantizations": ["Q8_0", "Q4_K_M"]
    },
    "finetuned": {
        "name": "phi4-so8t-ja-finetuned",
        "source": "outputs/so8t_ja_finetuned/final_model",
        "apply_burnin": True,
        "quantizations": ["Q8_0", "Q4_K_M"]
    }
}

LLAMA_CPP_PATH = Path("external/llama.cpp-master")
OUTPUT_DIR = Path("outputs/gguf_models")
MODELFILES_DIR = Path("modelfiles")


@dataclass
class ConversionResult:
    """変換結果"""
    model_name: str
    gguf_path: Path
    quantization: str
    file_size_mb: float
    success: bool
    error_message: Optional[str] = None


class SO8TBurnIn:
    """SO8T焼きこみ適用"""
    
    @staticmethod
    def apply(model_path: Path, output_path: Path) -> bool:
        """
        焼きこみ適用
        
        Args:
            model_path: 入力モデルパス
            output_path: 出力モデルパス
        
        Returns:
            success: 成功フラグ
        """
        print(f"\n[BURNIN] Applying SO8T burn-in...")
        print(f"Input: {model_path}")
        print(f"Output: {output_path}")
        
        try:
            # burn_in.pyを使用
            sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
            from burn_in import apply_rotation_burnin
            
            # モデルロード
            print("[LOAD] Loading model...")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # 焼きこみ適用
            print("[APPLY] Applying burn-in...")
            model_burnin = apply_rotation_burnin(model)
            
            # 保存
            print("[SAVE] Saving burned-in model...")
            output_path.mkdir(parents=True, exist_ok=True)
            model_burnin.save_pretrained(str(output_path))
            
            # Tokenizerもコピー
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            tokenizer.save_pretrained(str(output_path))
            
            print("[OK] Burn-in applied successfully")
            return True
        
        except Exception as e:
            print(f"[ERROR] Burn-in failed: {e}")
            return False


class GGUFConverter:
    """GGUF変換器"""
    
    def __init__(self, llama_cpp_path: Path):
        self.llama_cpp_path = llama_cpp_path
        self.convert_script = llama_cpp_path / "convert-hf-to-gguf.py"
        self.quantize_bin = llama_cpp_path / "llama-quantize.exe"
        
        # Windows環境確認
        if not self.quantize_bin.exists():
            self.quantize_bin = llama_cpp_path / "llama-quantize"
        
        if not self.convert_script.exists():
            raise FileNotFoundError(f"convert-hf-to-gguf.py not found: {self.convert_script}")
    
    def convert_to_gguf(self, model_path: Path, output_path: Path) -> bool:
        """
        HuggingFaceモデル→GGUF変換
        
        Args:
            model_path: HuggingFaceモデルパス
            output_path: GGUF出力パス
        
        Returns:
            success: 成功フラグ
        """
        print(f"\n[CONVERT] Converting to GGUF...")
        print(f"Model: {model_path}")
        print(f"Output: {output_path}")
        
        try:
            cmd = [
                "py", "-3",
                str(self.convert_script),
                str(model_path),
                "--outfile", str(output_path),
                "--outtype", "f16"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print("[OK] GGUF conversion successful")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] GGUF conversion failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False
    
    def quantize(self, input_gguf: Path, output_gguf: Path, quant_type: str) -> bool:
        """
        GGUF量子化
        
        Args:
            input_gguf: 入力GGUFパス
            output_gguf: 出力GGUFパス
            quant_type: 量子化タイプ（Q8_0, Q4_K_M等）
        
        Returns:
            success: 成功フラグ
        """
        print(f"\n[QUANTIZE] Quantizing to {quant_type}...")
        print(f"Input: {input_gguf}")
        print(f"Output: {output_gguf}")
        
        try:
            cmd = [
                str(self.quantize_bin),
                str(input_gguf),
                str(output_gguf),
                quant_type
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print("[OK] Quantization successful")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Quantization failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False


class OllamaDeployer:
    """Ollama配備器"""
    
    @staticmethod
    def create_modelfile(model_name: str, gguf_path: Path, output_path: Path, 
                         temperature: float = 0.7):
        """
        Modelfile生成
        
        Args:
            model_name: モデル名
            gguf_path: GGUFファイルパス
            output_path: Modelfile出力パス
            temperature: 温度パラメータ
        """
        modelfile_content = f"""FROM {gguf_path}

# Temperature setting
PARAMETER temperature {temperature}

# System prompt for Japanese
SYSTEM \"\"\"
あなたは日本語に特化したAIアシスタントです。
防衛・航空宇宙・運輸分野の専門知識を持ち、安全性を最優先します。
不明確な要求や危険な質問には、適切にエスカレーションまたは拒否します。
\"\"\"

# Template
TEMPLATE \"\"\"{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
\"\"\"
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"[OK] Modelfile created: {output_path}")
    
    @staticmethod
    def deploy_to_ollama(model_name: str, modelfile_path: Path) -> bool:
        """
        Ollamaにモデル配備
        
        Args:
            model_name: モデル名
            modelfile_path: Modelfileパス
        
        Returns:
            success: 成功フラグ
        """
        print(f"\n[DEPLOY] Deploying to Ollama...")
        print(f"Model: {model_name}")
        print(f"Modelfile: {modelfile_path}")
        
        try:
            cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"[OK] Model deployed: {model_name}")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Deployment failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False


class ModelConversionPipeline:
    """モデル変換パイプライン"""
    
    def __init__(self):
        self.converter = GGUFConverter(LLAMA_CPP_PATH)
        self.results: List[ConversionResult] = []
    
    def process_model(self, model_key: str, model_config: Dict) -> List[ConversionResult]:
        """
        単一モデルの処理
        
        Args:
            model_key: モデルキー
            model_config: モデル設定
        
        Returns:
            results: 変換結果リスト
        """
        print(f"\n{'='*60}")
        print(f"[PROCESS] Model: {model_config['name']}")
        print(f"{'='*60}\n")
        
        results = []
        model_name = model_config["name"]
        source_path = Path(model_config["source"])
        
        # 焼きこみ適用（必要な場合）
        if model_config["apply_burnin"]:
            burnin_path = OUTPUT_DIR / f"{model_name}_burnin"
            if not SO8TBurnIn.apply(source_path, burnin_path):
                print(f"[ERROR] Burn-in failed for {model_name}")
                return results
            source_path = burnin_path
        
        # GGUF変換（F16）
        f16_gguf = OUTPUT_DIR / f"{model_name}_f16.gguf"
        if not self.converter.convert_to_gguf(source_path, f16_gguf):
            print(f"[ERROR] GGUF conversion failed for {model_name}")
            return results
        
        # 量子化
        for quant_type in model_config["quantizations"]:
            quant_gguf = OUTPUT_DIR / f"{model_name}_{quant_type}.gguf"
            
            if self.converter.quantize(f16_gguf, quant_gguf, quant_type):
                file_size = quant_gguf.stat().st_size / (1024 * 1024)  # MB
                
                result = ConversionResult(
                    model_name=f"{model_name}_{quant_type}",
                    gguf_path=quant_gguf,
                    quantization=quant_type,
                    file_size_mb=file_size,
                    success=True
                )
                
                results.append(result)
                
                # Ollama配備
                modelfile_path = MODELFILES_DIR / f"Modelfile-{model_name}-{quant_type}"
                MODELFILES_DIR.mkdir(parents=True, exist_ok=True)
                
                OllamaDeployer.create_modelfile(
                    model_name=f"{model_name}_{quant_type}",
                    gguf_path=quant_gguf,
                    output_path=modelfile_path
                )
                
                OllamaDeployer.deploy_to_ollama(
                    model_name=f"so8t-{model_name}-{quant_type.lower()}",
                    modelfile_path=modelfile_path
                )
            else:
                result = ConversionResult(
                    model_name=f"{model_name}_{quant_type}",
                    gguf_path=quant_gguf,
                    quantization=quant_type,
                    file_size_mb=0.0,
                    success=False,
                    error_message="Quantization failed"
                )
                results.append(result)
        
        # F16削除（容量節約）
        if f16_gguf.exists():
            f16_gguf.unlink()
            print(f"[CLEAN] Removed intermediate F16 file")
        
        return results
    
    def process_all(self):
        """全モデル処理"""
        print(f"\n{'='*60}")
        print(f"[START] Model Conversion Pipeline")
        print(f"Models: {len(MODELS_CONFIG)}")
        print(f"{'='*60}\n")
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        for model_key, model_config in MODELS_CONFIG.items():
            results = self.process_model(model_key, model_config)
            self.results.extend(results)
        
        self._generate_report()
        
        print(f"\n{'='*60}")
        print(f"[OK] Conversion pipeline completed!")
        print(f"Total models: {len([r for r in self.results if r.success])}")
        print(f"{'='*60}\n")
    
    def _generate_report(self):
        """レポート生成"""
        report_file = Path("_docs") / f"{datetime.now().strftime('%Y-%m-%d')}_gguf_conversion_report.md"
        report_file.parent.mkdir(exist_ok=True)
        
        report = f"""# GGUF変換・Ollama配備レポート

## 変換概要
- **実行日時**: {datetime.now().isoformat()}
- **総モデル数**: {len(MODELS_CONFIG)}
- **成功**: {len([r for r in self.results if r.success])}
- **失敗**: {len([r for r in self.results if not r.success])}

## 変換結果

| モデル名 | 量子化 | ファイルサイズ | ステータス |
|---------|--------|---------------|-----------|
"""
        
        for result in self.results:
            status = "[OK]" if result.success else "[NG]"
            report += f"| {result.model_name} | {result.quantization} | {result.file_size_mb:.2f} MB | {status} |\n"
        
        report += """
## Ollamaモデル一覧

配備されたモデル：
"""
        
        for result in self.results:
            if result.success:
                model_name = f"so8t-{result.model_name.lower()}"
                report += f"- `{model_name}`\n"
        
        report += """
## 使用方法

```bash
# モデル一覧確認
ollama list

# 推論実行
ollama run so8t-phi4-so8t-ja-finetuned-q4_k_m "日本語で質問してください"

# 温度パラメータ変更
ollama run so8t-phi4-so8t-ja-finetuned-q4_k_m --temperature 0.5 "質問"
```

## 次のステップ
- [READY] Phase 5: Ollama推論比較評価
- [READY] 温度パラメータ分析
- [READY] 品質・速度・安全性メトリクス評価
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Report saved to {report_file}")


def main():
    """メイン実行"""
    pipeline = ModelConversionPipeline()
    
    try:
        pipeline.process_all()
    except KeyboardInterrupt:
        print("\n[WARNING] Pipeline interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
