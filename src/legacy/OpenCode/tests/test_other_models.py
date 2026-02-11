#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
他のOllamaモデルでテストするスクリプト
"""

import requests
import json
import time

def test_model(model_name, prompt):
    """指定されたモデルでテスト"""
    print(f"\n=== {model_name} テスト ===")
    print(f"プロンプト: {prompt}")
    
    ollama_url = "http://localhost:11434"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            response_time = end_time - start_time
            
            print(f"応答時間: {response_time:.2f}秒")
            print(f"生成テキスト: '{generated_text}'")
            print(f"生成テキスト長: {len(generated_text)}")
            
            # 詳細情報
            print(f"done: {result.get('done', False)}")
            print(f"done_reason: {result.get('done_reason', 'N/A')}")
            print(f"eval_count: {result.get('eval_count', 0)}")
            
            return len(generated_text) > 0
        else:
            print(f"エラー: {response.status_code}")
            print(f"エラー内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"例外エラー: {e}")
        return False

def main():
    print("他のOllamaモデルでテスト開始！")
    
    # 利用可能なモデルリスト
    models_to_test = [
        "so8t-phi31-mini-128k-enhanced-lightweight:latest",
        "so8t-phi31-mini-128k-enhanced-32gb:latest",
        "so8t-phi3-vision-enhanced:latest",
        "so8t-simple:latest",
        "llama3.2:3b"
    ]
    
    test_prompt = "こんにちは！自己紹介をお願いします。"
    
    results = {}
    
    for model in models_to_test:
        success = test_model(model, test_prompt)
        results[model] = success
        time.sleep(2)  # サーバー負荷軽減
    
    print("\n=== テスト結果サマリー ===")
    for model, success in results.items():
        status = "成功" if success else "失敗"
        print(f"{model}: {status}")

if __name__ == "__main__":
    main()
