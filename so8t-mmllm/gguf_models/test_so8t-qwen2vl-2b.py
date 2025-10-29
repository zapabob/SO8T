#!/usr/bin/env python3
"""
SO8TLLM 
so8t-qwen2vl-2b
"""

import subprocess
import json
import time
from datetime import datetime

def test_model_creation(model_name, modelfile_path):
    """"""
    print(f" : {model_name}")
    
    try:
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(" ")
            return True
        else:
            print(f" : {result.stderr}")
            return False
    except Exception as e:
        print(f" : {str(e)}")
        return False

def test_model_inference(model_name, test_prompts):
    """"""
    print(f" : {model_name}")
    
    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"   {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        
        try:
            cmd = ["ollama", "run", model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"   ")
                results.append({
                    "prompt": prompt,
                    "response": result.stdout,
                    "success": True
                })
            else:
                print(f"   : {result.stderr}")
                results.append({
                    "prompt": prompt,
                    "response": result.stderr,
                    "success": False
                })
        except Exception as e:
            print(f"   : {str(e)}")
            results.append({
                "prompt": prompt,
                "response": str(e),
                "success": False
            })
    
    return results

def main():
    model_name = "so8t-qwen2vl-2b"
    modelfile_path = r"C:\Users\downl\Desktop\SO8T\so8t-mmllm\gguf_models\so8t-qwen2vl-2b.Modelfile"
    
    # 
    test_prompts = [
        "",
        "",
        "",
        ": 2x + 5 = 13",
        ": AI"
    ]
    
    print(" SO8TLLM ...")
    
    # 
    if test_model_creation(model_name, modelfile_path):
        # 
        results = test_model_inference(model_name, test_prompts)
        
        # 
        results_file = r"C:\Users\downl\Desktop\SO8T\so8t-mmllm\gguf_models\so8t-qwen2vl-2b_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "test_results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f" : {results_file}")
        
        # 
        success_count = sum(1 for r in results if r["success"])
        success_rate = success_count / len(results)
        print(f" : {success_rate:.2%} ({success_count}/{len(results)})")
    
    print(" ")

if __name__ == "__main__":
    main()
