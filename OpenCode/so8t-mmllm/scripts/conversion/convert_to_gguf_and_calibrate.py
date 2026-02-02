#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF変換・量子化・温度較正 統合スクリプト
1. HF → GGUF変換
2. Q8_0, Q4_K_M量子化
3. 温度較正（ECE最小化）
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# GGUF量子化タイプ
QUANT_TYPES = {
    "Q8_0": "8-bit quantization",
    "Q4_K_M": "4-bit K-quant, medium"
}


def convert_to_gguf(model_dir: Path, output_file: Path) -> bool:
    """
    HF → GGUF変換
    
    Args:
        model_dir: HFモデルディレクトリ
        output_file: 出力GGUFファイル
    
    Returns:
        success: 成功フラグ
    """
    print(f"\n[CONVERT] HF → GGUF")
    print(f"Input: {model_dir}")
    print(f"Output: {output_file}")
    
    # llama.cpp の convert.py 使用
    convert_script = Path("llama.cpp/convert.py")
    
    if not convert_script.exists():
        print("[ERROR] llama.cpp not found. Please clone: git clone https://github.com/ggerganov/llama.cpp")
        return False
    
    cmd = [
        "python", str(convert_script),
        str(model_dir),
        "--outfile", str(output_file),
        "--outtype", "f16"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] GGUF conversion completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] GGUF conversion failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def quantize_gguf(input_file: Path, output_file: Path, quant_type: str) -> bool:
    """
    GGUF量子化
    
    Args:
        input_file: 入力GGUFファイル
        output_file: 出力GGUFファイル
        quant_type: 量子化タイプ（Q8_0, Q4_K_M等）
    
    Returns:
        success: 成功フラグ
    """
    print(f"\n[QUANTIZE] {quant_type}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # llama.cpp の quantize 使用
    quantize_bin = Path("llama.cpp/quantize")
    
    if not quantize_bin.exists():
        print("[ERROR] llama.cpp/quantize not found. Please build llama.cpp")
        return False
    
    cmd = [
        str(quantize_bin),
        str(input_file),
        str(output_file),
        quant_type
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[OK] Quantization completed: {quant_type}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Quantization failed: {e}")
        return False


def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) 計算
    
    Args:
        confidences: 信頼度配列
        accuracies: 精度配列
        n_bins: ビン数
    
    Returns:
        ece: ECE値
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calibrate_temperature(
    model_path: str,
    tokenizer,
    test_data: List[Dict],
    initial_temp: float = 1.0
) -> float:
    """
    温度較正（ECE最小化）
    
    Args:
        model_path: モデルパス
        tokenizer: トークナイザー
        test_data: テストデータ
        initial_temp: 初期温度
    
    Returns:
        optimal_temp: 最適温度
    """
    print(f"\n[CALIBRATE] Temperature calibration")
    print(f"Model: {model_path}")
    print(f"Test samples: {len(test_data)}")
    
    # モデルロード
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # 温度探索
    temperatures = np.linspace(0.3, 2.0, 20)
    best_temp = initial_temp
    best_ece = float('inf')
    
    print("[SEARCH] Searching optimal temperature...")
    
    with torch.no_grad():
        for temp in temperatures:
            confidences = []
            accuracies = []
            
            for sample in test_data:
                # 推論
                inputs = tokenizer(sample["query"], return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=temp,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # 信頼度
                scores = outputs.scores
                probs = torch.softmax(scores[0] / temp, dim=-1)
                confidence = probs.max().item()
                confidences.append(confidence)
                
                # 精度（簡易）
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                accuracy = 1.0 if sample["expected"] in response else 0.0
                accuracies.append(accuracy)
            
            # ECE計算
            ece = calculate_ece(np.array(confidences), np.array(accuracies))
            
            print(f"  Temp={temp:.2f}, ECE={ece:.4f}")
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
    
    print(f"\n[RESULT] Optimal temperature: {best_temp:.2f}")
    print(f"[RESULT] Minimum ECE: {best_ece:.4f}")
    
    return best_temp


def create_ollama_modelfile(
    model_name: str,
    gguf_path: Path,
    temperature: float,
    system_prompt: str,
    output_file: Path
):
    """
    Ollama Modelfile作成
    
    Args:
        model_name: モデル名
        gguf_path: GGUFファイルパス
        temperature: 較正済み温度
        system_prompt: システムプロンプト
        output_file: 出力Modelfile
    """
    modelfile_content = f"""# SO8T 4-Role Model
FROM {gguf_path}

# Calibrated temperature
PARAMETER temperature {temperature}

# System prompt
SYSTEM \"\"\"
{system_prompt}
\"\"\"

# Model parameters
PARAMETER num_ctx 4096
PARAMETER num_predict 1024
PARAMETER stop "Query:"
PARAMETER stop "###"
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"[OK] Modelfile created: {output_file}")


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path,
                        default=Path("outputs/borea_so8t_burned"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/gguf"))
    parser.add_argument("--model_name", type=str, default="borea-so8t-4role")
    parser.add_argument("--skip_convert", action="store_true", help="Skip HF→GGUF conversion")
    parser.add_argument("--skip_quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--skip_calibrate", action="store_true", help="Skip temperature calibration")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"[START] GGUF Conversion & Calibration Pipeline")
    print(f"Model: {args.model_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: HF → GGUF変換
    gguf_f16_path = args.output_dir / f"{args.model_name}-f16.gguf"
    
    if not args.skip_convert:
        success = convert_to_gguf(args.model_dir, gguf_f16_path)
        if not success:
            print("[ERROR] Conversion failed")
            return
    else:
        print(f"[SKIP] Using existing GGUF: {gguf_f16_path}")
    
    # Step 2: 量子化
    quant_paths = {}
    
    if not args.skip_quantize:
        for quant_type, description in QUANT_TYPES.items():
            output_path = args.output_dir / f"{args.model_name}-{quant_type}.gguf"
            success = quantize_gguf(gguf_f16_path, output_path, quant_type)
            
            if success:
                quant_paths[quant_type] = output_path
    else:
        print("[SKIP] Quantization skipped")
        for quant_type in QUANT_TYPES:
            quant_paths[quant_type] = args.output_dir / f"{args.model_name}-{quant_type}.gguf"
    
    # Step 3: 温度較正
    if not args.skip_calibrate:
        # テストデータ生成（簡易）
        test_data = [
            {"query": "防衛装備品の調達について", "expected": "調達"},
            {"query": "取引の不正を検知", "expected": "検知"},
            {"query": "患者のカルテを要約", "expected": "要約"}
        ]
        
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
        optimal_temp = calibrate_temperature(
            str(args.model_dir),
            tokenizer,
            test_data
        )
    else:
        print("[SKIP] Using default temperature: 0.7")
        optimal_temp = 0.7
    
    # Step 4: Ollama Modelfile作成
    system_prompt = """You are SO8T, a 4-role safety-aware AI assistant.

For each query, you provide:
1. Task Response: Answer the query
2. Safety Judgment: ALLOW/ESCALATE/DENY
3. Validation Reasoning: Consistency check
4. Escalation: Whether human intervention is needed

Always prioritize safety and transparency."""
    
    for quant_type, gguf_path in quant_paths.items():
        modelfile_path = args.output_dir / f"Modelfile.{quant_type}"
        create_ollama_modelfile(
            args.model_name,
            gguf_path,
            optimal_temp,
            system_prompt,
            modelfile_path
        )
    
    # メタデータ保存
    metadata = {
        "model_name": args.model_name,
        "base_model": str(args.model_dir),
        "gguf_f16": str(gguf_f16_path),
        "quantized_models": {k: str(v) for k, v in quant_paths.items()},
        "calibrated_temperature": float(optimal_temp),
        "created_at": str(datetime.now())
    }
    
    with open(args.output_dir / "conversion_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"[OK] Pipeline completed!")
    print(f"GGUF models: {args.output_dir}")
    print(f"Optimal temperature: {optimal_temp:.2f}")
    print(f"\nNext steps:")
    print(f"  1. ollama create {args.model_name} -f {args.output_dir}/Modelfile.Q4_K_M")
    print(f"  2. ollama run {args.model_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    from datetime import datetime
    main()

