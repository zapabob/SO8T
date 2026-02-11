# SO8T×マルチモーダルLLM GGUF変換テスト
# GGUF変換とllama.cpp推論検証を実施

param(
    [string]$ModelPath = "./outputs",
    [string]$OutputDir = "./gguf_test_results",
    [string]$ModelName = "so8t-qwen2vl-2b"
)

Write-Host "🔄 SO8T×マルチモーダルLLM GGUF変換テスト開始..." -ForegroundColor Green

# 仮想環境のアクティベート
Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 出力ディレクトリの作成
Write-Host "📁 出力ディレクトリを作成中..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# GGUF変換テストスクリプトの実行
Write-Host "🎯 GGUF変換テストを実行中..." -ForegroundColor Yellow

$ggufTestScript = @"
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

# パスを追加
sys.path.append('src')

from training.trainer_with_pet import SO8TIntegratedTrainer
from modules.qwen2vl_wrapper import create_so8t_qwen2vl_model

def check_llama_cpp_availability():
    """llama.cppの利用可能性をチェック"""
    print("🔍 llama.cppの利用可能性をチェック中...")
    
    results = []
    
    # 1. llama-cpp-pythonの確認
    try:
        import llama_cpp
        print(f"  ✅ llama-cpp-python: {llama_cpp.__version__}")
        results.append({
            "component": "llama_cpp_python",
            "available": True,
            "version": llama_cpp.__version__
        })
    except ImportError:
        print("  ❌ llama-cpp-python: インストールされていません")
        results.append({
            "component": "llama_cpp_python",
            "available": False,
            "error": "Not installed"
        })
    
    # 2. llama.cppバイナリの確認
    llama_cpp_paths = [
        "llama.cpp/convert.py",
        "llama.cpp/main",
        "llama-cpp-python",
        "llama-cpp-python.exe"
    ]
    
    for path in llama_cpp_paths:
        if shutil.which(path):
            print(f"  ✅ llama.cppバイナリ: {path}")
            results.append({
                "component": "llama_cpp_binary",
                "available": True,
                "path": path
            })
            break
    else:
        print("  ⚠️ llama.cppバイナリ: 見つかりません")
        results.append({
            "component": "llama_cpp_binary",
            "available": False,
            "error": "Binary not found"
        })
    
    return results

def test_model_conversion(model_path, output_dir, model_name):
    """モデル変換をテスト"""
    print("🔄 モデル変換をテスト中...")
    
    results = []
    
    try:
        # 1. 学習済みモデルを読み込み
        print("  📦 学習済みモデルを読み込み中...")
        trainer = SO8TIntegratedTrainer(
            model_path=model_path,
            config_path=os.path.join(model_path, "config.json"),
            output_dir=output_dir
        )
        trainer.setup_components()
        
        print(f"  ✅ モデル読み込み完了")
        print(f"     回転ゲート: {'有効' if trainer.rotation_gate is not None else '無効'}")
        print(f"     PET損失: {'有効' if trainer.pet_loss is not None else '無効'}")
        
        results.append({
            "test": "model_loading",
            "success": True,
            "rotation_gate": trainer.rotation_gate is not None,
            "pet_loss": trainer.pet_loss is not None
        })
        
    except Exception as e:
        print(f"  ❌ モデル読み込みエラー: {str(e)}")
        results.append({
            "test": "model_loading",
            "success": False,
            "error": str(e)
        })
        return results
    
    try:
        # 2. Hugging Face形式で保存
        print("  💾 Hugging Face形式で保存中...")
        hf_path = os.path.join(output_dir, f"{model_name}_hf")
        trainer.model.save_pretrained(hf_path)
        
        # 設定ファイルもコピー
        config_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        for config_file in config_files:
            src_path = os.path.join(model_path, config_file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, hf_path)
        
        print(f"  ✅ Hugging Face形式で保存完了: {hf_path}")
        
        results.append({
            "test": "hf_saving",
            "success": True,
            "hf_path": hf_path
        })
        
    except Exception as e:
        print(f"  ❌ Hugging Face保存エラー: {str(e)}")
        results.append({
            "test": "hf_saving",
            "success": False,
            "error": str(e)
        })
        return results
    
    try:
        # 3. GGUF変換設定を作成
        print("  ⚙️ GGUF変換設定を作成中...")
        gguf_config = {
            "model_name": model_name,
            "model_path": hf_path,
            "output_path": os.path.join(output_dir, f"{model_name}.gguf"),
            "quantization": "Q8_0",
            "description": "SO8T×マルチモーダルLLM (焼き込み済み)",
            "conversion_script": "llama.cpp/convert.py",
            "conversion_command": f"python llama.cpp/convert.py {hf_path} --outfile {os.path.join(output_dir, f'{model_name}.gguf')} --outtype q8_0"
        }
        
        config_path = os.path.join(output_dir, "gguf_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(gguf_config, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 変換設定を作成: {config_path}")
        
        results.append({
            "test": "config_creation",
            "success": True,
            "config_path": config_path,
            "gguf_config": gguf_config
        })
        
    except Exception as e:
        print(f"  ❌ 設定作成エラー: {str(e)}")
        results.append({
            "test": "config_creation",
            "success": False,
            "error": str(e)
        })
    
    try:
        # 4. 実際のGGUF変換を試行
        print("  🔄 GGUF変換を試行中...")
        
        # llama.cppのconvert.pyを探す
        convert_script = None
        for path in ["llama.cpp/convert.py", "convert.py", "llama-cpp-python"]:
            if os.path.exists(path) or shutil.which(path):
                convert_script = path
                break
        
        if convert_script:
            print(f"  📝 変換スクリプトを使用: {convert_script}")
            
            # 変換コマンドを実行
            gguf_path = os.path.join(output_dir, f"{model_name}.gguf")
            cmd = [
                "python", convert_script,
                hf_path,
                "--outfile", gguf_path,
                "--outtype", "q8_0"
            ]
            
            print(f"  🚀 変換コマンド実行: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"  ✅ GGUF変換成功: {gguf_path}")
                    results.append({
                        "test": "gguf_conversion",
                        "success": True,
                        "gguf_path": gguf_path,
                        "command": cmd,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    })
                else:
                    print(f"  ❌ GGUF変換失敗: {result.stderr}")
                    results.append({
                        "test": "gguf_conversion",
                        "success": False,
                        "error": result.stderr,
                        "command": cmd
                    })
            
            except subprocess.TimeoutExpired:
                print("  ⏰ GGUF変換タイムアウト")
                results.append({
                    "test": "gguf_conversion",
                    "success": False,
                    "error": "Timeout"
                })
            
            except Exception as e:
                print(f"  ❌ 変換実行エラー: {str(e)}")
                results.append({
                    "test": "gguf_conversion",
                    "success": False,
                    "error": str(e)
                })
        
        else:
            print("  ⚠️ 変換スクリプトが見つかりません。手動変換が必要です。")
            results.append({
                "test": "gguf_conversion",
                "success": False,
                "error": "Conversion script not found",
                "manual_conversion_required": True
            })
    
    except Exception as e:
        print(f"  ❌ GGUF変換エラー: {str(e)}")
        results.append({
            "test": "gguf_conversion",
            "success": False,
            "error": str(e)
        })
    
    return results

def test_ollama_integration(output_dir, model_name):
    """Ollama統合をテスト"""
    print("🦙 Ollama統合をテスト中...")
    
    results = []
    
    try:
        # 1. Modelfileを作成
        print("  📝 Modelfileを作成中...")
        modelfile_content = f"""FROM {os.path.join(output_dir, f"{model_name}.gguf")}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}\"\"\"

# SO8T×マルチモーダルLLM Model Card
# SO(8)群回転ゲート + PET正則化 + OCR要約 + SQLite監査

SYSTEM \"\"\"You are SO8T×マルチモーダルLLM, an advanced multimodal language model with SO(8) group structure and enhanced safety features.

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

You provide helpful, accurate, and safe responses while maintaining complete privacy and auditability.\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 32768
PARAMETER num_predict 2048
"""
        
        modelfile_path = os.path.join(output_dir, f"{model_name}.Modelfile")
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"  ✅ Modelfileを作成: {modelfile_path}")
        
        results.append({
            "test": "modelfile_creation",
            "success": True,
            "modelfile_path": modelfile_path
        })
        
    except Exception as e:
        print(f"  ❌ Modelfile作成エラー: {str(e)}")
        results.append({
            "test": "modelfile_creation",
            "success": False,
            "error": str(e)
        })
    
    try:
        # 2. Ollamaモデル作成を試行
        print("  🦙 Ollamaモデル作成を試行中...")
        
        # ollamaコマンドの確認
        if shutil.which("ollama"):
            print("  ✅ ollamaコマンドが見つかりました")
            
            # モデル作成コマンド
            create_cmd = ["ollama", "create", model_name, "-f", modelfile_path]
            print(f"  🚀 モデル作成コマンド: {' '.join(create_cmd)}")
            
            try:
                result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print(f"  ✅ Ollamaモデル作成成功: {model_name}")
                    results.append({
                        "test": "ollama_model_creation",
                        "success": True,
                        "model_name": model_name,
                        "command": create_cmd,
                        "stdout": result.stdout
                    })
                    
                    # 3. モデル実行テスト
                    print("  🧪 モデル実行テスト中...")
                    run_cmd = ["ollama", "run", model_name, "画像を説明してください。"]
                    
                    try:
                        run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
                        
                        if run_result.returncode == 0:
                            print(f"  ✅ モデル実行成功")
                            print(f"     応答: {run_result.stdout[:100]}...")
                            results.append({
                                "test": "ollama_model_run",
                                "success": True,
                                "response": run_result.stdout,
                                "command": run_cmd
                            })
                        else:
                            print(f"  ❌ モデル実行失敗: {run_result.stderr}")
                            results.append({
                                "test": "ollama_model_run",
                                "success": False,
                                "error": run_result.stderr
                            })
                    
                    except subprocess.TimeoutExpired:
                        print("  ⏰ モデル実行タイムアウト")
                        results.append({
                            "test": "ollama_model_run",
                            "success": False,
                            "error": "Timeout"
                        })
                    
                    except Exception as e:
                        print(f"  ❌ モデル実行エラー: {str(e)}")
                        results.append({
                            "test": "ollama_model_run",
                            "success": False,
                            "error": str(e)
                        })
                
                else:
                    print(f"  ❌ Ollamaモデル作成失敗: {result.stderr}")
                    results.append({
                        "test": "ollama_model_creation",
                        "success": False,
                        "error": result.stderr,
                        "command": create_cmd
                    })
            
            except subprocess.TimeoutExpired:
                print("  ⏰ モデル作成タイムアウト")
                results.append({
                    "test": "ollama_model_creation",
                    "success": False,
                    "error": "Timeout"
                })
            
            except Exception as e:
                print(f"  ❌ モデル作成エラー: {str(e)}")
                results.append({
                    "test": "ollama_model_creation",
                    "success": False,
                    "error": str(e)
                })
        
        else:
            print("  ⚠️ ollamaコマンドが見つかりません")
            results.append({
                "test": "ollama_model_creation",
                "success": False,
                "error": "ollama command not found"
            })
    
    except Exception as e:
        print(f"  ❌ Ollama統合エラー: {str(e)}")
        results.append({
            "test": "ollama_integration",
            "success": False,
            "error": str(e)
        })
    
    return results

def analyze_results(all_results):
    """結果を分析"""
    print("\\n📊 GGUF変換テスト結果分析")
    print("=" * 50)
    
    # 各テストの成功率を計算
    test_success = {}
    for result in all_results:
        test_name = result.get('test', 'unknown')
        if test_name not in test_success:
            test_success[test_name] = {'success': 0, 'total': 0}
        
        test_success[test_name]['total'] += 1
        if result.get('success', False):
            test_success[test_name]['success'] += 1
    
    print("📈 テスト結果:")
    for test_name, stats in test_success.items():
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {test_name}: {stats['success']}/{stats['total']} ({success_rate:.3f})")
    
    # 総合成功率
    total_success = sum(stats['success'] for stats in test_success.values())
    total_tests = sum(stats['total'] for stats in test_success.values())
    overall_success_rate = total_success / total_tests if total_tests > 0 else 0.0
    
    print(f"\\n📊 総合成功率: {overall_success_rate:.3f}")
    
    # 推奨事項
    print("\\n💡 推奨事項:")
    if not test_success.get('gguf_conversion', {}).get('success', 0):
        print("  - llama.cppのconvert.pyをインストールしてGGUF変換を完了してください")
    if not test_success.get('ollama_model_creation', {}).get('success', 0):
        print("  - Ollamaをインストールしてモデル実行をテストしてください")
    
    return {
        "test_success": test_success,
        "overall_success_rate": overall_success_rate
    }

def main():
    print("🔄 SO8T×マルチモーダルLLM GGUF変換テスト開始...")
    
    # 各テストを実行
    print("\\n🎯 llama.cpp利用可能性チェック開始...")
    availability_results = check_llama_cpp_availability()
    
    print("\\n🎯 モデル変換テスト開始...")
    conversion_results = test_model_conversion('$ModelPath', '$OutputDir', '$ModelName')
    
    print("\\n🎯 Ollama統合テスト開始...")
    ollama_results = test_ollama_integration('$OutputDir', '$ModelName')
    
    # 全結果を統合
    all_results = availability_results + conversion_results + ollama_results
    
    # 結果を分析
    analysis = analyze_results(all_results)
    
    # 結果を保存
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": "$ModelPath",
        "output_dir": "$OutputDir",
        "model_name": "$ModelName",
        "availability_results": availability_results,
        "conversion_results": conversion_results,
        "ollama_results": ollama_results,
        "analysis": analysis
    }
    
    results_file = "$OutputDir/gguf_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📁 結果を保存しました: {results_file}")
    print(f"📊 総合成功率: {analysis['overall_success_rate']:.3f}")
    
    print("\\n✅ GGUF変換テスト完了！")

if __name__ == "__main__":
    main()
"@

# GGUF変換テストスクリプトを実行
$ggufTestScript | py -3

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ GGUF変換テスト完了！" -ForegroundColor Green
    Write-Host "📁 結果ディレクトリ: $OutputDir" -ForegroundColor Cyan
    Write-Host "📊 結果ファイル: $OutputDir/gguf_test_results.json" -ForegroundColor Cyan
    Write-Host "🔄 GGUFファイル: $OutputDir/$ModelName.gguf" -ForegroundColor Cyan
    Write-Host "🦙 Modelfile: $OutputDir/$ModelName.Modelfile" -ForegroundColor Cyan
} else {
    Write-Error "❌ GGUF変換テスト中にエラーが発生しました"
    exit 1
}
