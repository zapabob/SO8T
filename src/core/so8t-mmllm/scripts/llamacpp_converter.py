#!/usr/bin/env python3
"""
SO8T×マルチモーダルLLM llama.cpp変換スクリプト
llama.cpp-masterを使用してHugging FaceモデルをGGUFに変換
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def setup_llamacpp_environment(llamacpp_path):
    """llama.cpp環境をセットアップ"""
    print("[FIX] llama.cpp環境をセットアップ中...")
    
    # llama.cppディレクトリの存在確認
    if not os.path.exists(llamacpp_path):
        raise FileNotFoundError(f"llama.cppディレクトリが見つかりません: {llamacpp_path}")
    
    # convert_hf_to_gguf.pyの存在確認
    convert_script = os.path.join(llamacpp_path, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"convert_hf_to_gguf.pyが見つかりません: {convert_script}")
    
    print(f"[OK] llama.cpp環境確認完了: {llamacpp_path}")
    return convert_script

def convert_model_to_gguf(
    model_path,
    output_dir,
    model_name,
    quantization="q8_0",
    llamacpp_path="C:\\Users\\downl\\Desktop\\SO8T\\llama.cpp-master"
):
    """モデルをGGUFに変換"""
    print(f"🔄 モデル変換開始: {model_path} -> {output_dir}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # llama.cpp環境をセットアップ
    convert_script = setup_llamacpp_environment(llamacpp_path)
    
    # 出力ファイルパス
    output_file = os.path.join(output_dir, f"{model_name}.gguf")
    
    # 変換コマンドを構築
    cmd = [
        "py", convert_script,
        model_path,
        "--outfile", output_file,
        "--outtype", quantization,
        "--verbose"
    ]
    
    print(f"[START] 変換コマンド実行: {' '.join(cmd)}")
    
    try:
        # 変換を実行
        result = subprocess.run(
            cmd,
            cwd=llamacpp_path,
            capture_output=True,
            text=True,
            timeout=1800  # 30分タイムアウト
        )
        
        if result.returncode == 0:
            print("[OK] モデル変換成功！")
            
            # ファイルサイズを確認
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024**3)  # GB
                print(f"[STATS] ファイルサイズ: {file_size:.2f} GB")
            
            return output_file, result.stdout, result.stderr
        else:
            print(f"[NG] モデル変換失敗 (終了コード: {result.returncode})")
            print(f"エラー出力: {result.stderr}")
            return None, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print("⏰ 変換タイムアウト (30分)")
        return None, "", "Timeout"
    except Exception as e:
        print(f"[NG] 変換中にエラーが発生: {str(e)}")
        return None, "", str(e)

def create_modelfile(output_file, model_name, output_dir):
    """Modelfileを作成"""
    print("[NOTE] Modelfileを作成中...")
    
    modelfile_content = f'''FROM {output_file}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}"""

# SO8T×マルチモーダルLLM Model Card
# SO(8)群回転ゲート + PET正則化 + OCR要約 + SQLite監査

SYSTEM """You are SO8T×マルチモーダルLLM, an advanced multimodal language model with SO(8) group structure and enhanced safety features.

Key Features:
- SO(8) Group Structure: 8-dimensional rotation gates for enhanced reasoning
- PET Regularization: Second-order difference penalty for smooth outputs
- OCR Summary: Local image processing with privacy protection
- SQLite Audit: Complete decision logging and policy tracking

Capabilities:
- Multimodal understanding (text + images)
- Safe and responsible AI responses
- Local OCR processing (no external data sharing)
- Comprehensive audit logging

Safety Guidelines:
- Always prioritize user safety and privacy
- Process images locally without external sharing
- Log all decisions for transparency
- Escalate complex ethical decisions when needed

You provide helpful, accurate, and safe responses while maintaining complete privacy and auditability."""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 32768
PARAMETER num_predict 2048
'''
    
    modelfile_path = os.path.join(output_dir, f"{model_name}.Modelfile")
    
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"[OK] Modelfile作成完了: {modelfile_path}")
    return modelfile_path

def create_ollama_commands(model_name, modelfile_path, output_dir):
    """Ollamaコマンドを作成"""
    print("🦙 Ollamaコマンドを作成中...")
    
    commands = {
        "create_model": f"ollama create {model_name} -f \"{modelfile_path}\"",
        "run_model": f"ollama run {model_name}",
        "list_models": "ollama list",
        "remove_model": f"ollama rm {model_name}"
    }
    
    # コマンドファイルを作成
    commands_file = os.path.join(output_dir, f"{model_name}_ollama_commands.txt")
    
    with open(commands_file, 'w', encoding='utf-8') as f:
        f.write("# SO8T×マルチモーダルLLM Ollamaコマンド\n")
        f.write(f"# 作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for name, command in commands.items():
            f.write(f"# {name}\n")
            f.write(f"{command}\n\n")
    
    print(f"[OK] Ollamaコマンドファイル作成完了: {commands_file}")
    return commands_file

def main():
    parser = argparse.ArgumentParser(description="SO8T×マルチモーダルLLM llama.cpp変換")
    parser.add_argument("--model_path", default="./outputs", help="入力モデルパス")
    parser.add_argument("--output_dir", default="./gguf_models", help="出力ディレクトリ")
    parser.add_argument("--model_name", default="so8t-qwen2vl-2b", help="モデル名")
    parser.add_argument("--quantization", default="q8_0", 
                       choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"],
                       help="量子化タイプ")
    parser.add_argument("--llamacpp_path", 
                       default="C:\\Users\\downl\\Desktop\\SO8T\\llama.cpp-master",
                       help="llama.cppパス")
    
    args = parser.parse_args()
    
    print("🔄 SO8T×マルチモーダルLLM llama.cpp変換開始...")
    print(f"[DIR] 入力モデル: {args.model_path}")
    print(f"[DIR] 出力ディレクトリ: {args.output_dir}")
    print(f"🏷️ モデル名: {args.model_name}")
    print(f"⚙️ 量子化: {args.quantization}")
    
    # モデル変換
    output_file, stdout, stderr = convert_model_to_gguf(
        args.model_path,
        args.output_dir,
        args.model_name,
        args.quantization,
        args.llamacpp_path
    )
    
    if output_file:
        # Modelfileを作成
        modelfile_path = create_modelfile(output_file, args.model_name, args.output_dir)
        
        # Ollamaコマンドを作成
        commands_file = create_ollama_commands(args.model_name, modelfile_path, args.output_dir)
        
        # 結果サマリー
        print("\n[STATS] 変換結果サマリー")
        print("=" * 50)
        print(f"モデル名: {args.model_name}")
        print(f"量子化: {args.quantization}")
        print(f"GGUFファイル: {output_file}")
        print(f"Modelfile: {modelfile_path}")
        print(f"コマンドファイル: {commands_file}")
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024**3)
            print(f"ファイルサイズ: {file_size:.2f} GB")
        
        print("\n🦙 Ollamaモデル作成手順:")
        print(f"1. ollama create {args.model_name} -f \"{modelfile_path}\"")
        print(f"2. ollama run {args.model_name}")
        
        print("\n[OK] llama.cpp変換完了！")
        
    else:
        print("\n[NG] 変換に失敗しました")
        if stderr:
            print(f"エラー詳細: {stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
