#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプルなOllamaテスト
"""

import requests
import json

def simple_test():
    """シンプルなテスト"""
    print("シンプルなOllamaテスト開始！")
    
    # まず利用可能なモデルを確認
    print("\n1. 利用可能なモデル確認...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"利用可能モデル数: {len(models)}")
            for model in models[:5]:  # 最初の5個だけ表示
                print(f"  - {model['name']}")
        else:
            print(f"エラー: {response.status_code}")
            return
    except Exception as e:
        print(f"接続エラー: {e}")
        return
    
    # シンプルなモデルでテスト
    print("\n2. llama3.2:3bでテスト...")
    test_prompt = "Hello! Please introduce yourself."
    
    payload = {
        "model": "llama3.2:3b",
        "prompt": test_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"プロンプト: {test_prompt}")
            print(f"生成テキスト: '{generated_text}'")
            print(f"生成テキスト長: {len(generated_text)}")
            
            if len(generated_text) > 0:
                print("✅ テキスト生成成功！")
            else:
                print("❌ テキスト生成失敗（空のレスポンス）")
        else:
            print(f"APIエラー: {response.status_code}")
            print(f"エラー内容: {response.text}")
            
    except Exception as e:
        print(f"リクエストエラー: {e}")
    
    # SO8Tモデルでテスト
    print("\n3. SO8Tモデルでテスト...")
    payload["model"] = "so8t-qwen2vl-2b:latest"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"プロンプト: {test_prompt}")
            print(f"生成テキスト: '{generated_text}'")
            print(f"生成テキスト長: {len(generated_text)}")
            
            if len(generated_text) > 0:
                print("✅ SO8Tテキスト生成成功！")
            else:
                print("❌ SO8Tテキスト生成失敗（空のレスポンス）")
                print("詳細情報:")
                print(f"  done: {result.get('done', False)}")
                print(f"  done_reason: {result.get('done_reason', 'N/A')}")
                print(f"  eval_count: {result.get('eval_count', 0)}")
        else:
            print(f"APIエラー: {response.status_code}")
            print(f"エラー内容: {response.text}")
            
    except Exception as e:
        print(f"リクエストエラー: {e}")

if __name__ == "__main__":
    simple_test()
