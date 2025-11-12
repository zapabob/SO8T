#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版Ollamaテスト - stopパラメータを無効化
"""

import requests
import json
import time

def test_so8t_without_stop():
    """stopパラメータを無効化してSO8Tモデルをテスト"""
    print("修正版SO8Tテスト開始！")
    
    ollama_url = "http://localhost:11434"
    model_name = "so8t-qwen2vl-2b:latest"
    
    # stopパラメータを無効化するためのペイロード
    payload = {
        "model": model_name,
        "prompt": "こんにちは！自己紹介をお願いします。",
        "stream": False,
        "options": {
            "stop": [],  # stopパラメータを空にする
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 512,  # 生成トークン数を制限
            "num_ctx": 2048      # コンテキストサイズを制限
        }
    }
    
    print(f"プロンプト: {payload['prompt']}")
    print("stopパラメータを無効化してテスト中...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=60
        )
        end_time = time.time()
        
        print(f"HTTPステータス: {response.status_code}")
        print(f"応答時間: {end_time - start_time:.2f}秒")
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            
            print(f"生成テキスト: '{generated_text}'")
            print(f"生成テキスト長: {len(generated_text)}")
            print(f"done: {result.get('done', False)}")
            print(f"done_reason: {result.get('done_reason', 'N/A')}")
            print(f"eval_count: {result.get('eval_count', 0)}")
            
            if len(generated_text) > 0:
                print("✅ SO8Tテキスト生成成功！")
                return True
            else:
                print("❌ SO8Tテキスト生成失敗（空のレスポンス）")
                return False
        else:
            print(f"APIエラー: {response.status_code}")
            print(f"エラー内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"リクエストエラー: {e}")
        return False

def test_simple_prompt():
    """シンプルなプロンプトでテスト"""
    print("\nシンプルなプロンプトでテスト...")
    
    ollama_url = "http://localhost:11434"
    model_name = "so8t-qwen2vl-2b:latest"
    
    # よりシンプルなプロンプト
    payload = {
        "model": model_name,
        "prompt": "Hello",
        "stream": False,
        "options": {
            "stop": [],
            "temperature": 0.8,
            "num_predict": 100
        }
    }
    
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"プロンプト: Hello")
            print(f"生成テキスト: '{generated_text}'")
            print(f"生成テキスト長: {len(generated_text)}")
            
            return len(generated_text) > 0
        else:
            print(f"APIエラー: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"エラー: {e}")
        return False

def main():
    print("修正版Ollamaテスト開始！")
    print("stopパラメータを無効化してSO8Tモデルをテストするで！")
    
    # 1. stopパラメータ無効化テスト
    success1 = test_so8t_without_stop()
    
    # 2. シンプルなプロンプトテスト
    success2 = test_simple_prompt()
    
    print("\n=== テスト結果 ===")
    print(f"stopパラメータ無効化テスト: {'成功' if success1 else '失敗'}")
    print(f"シンプルプロンプトテスト: {'成功' if success2 else '失敗'}")
    
    if success1 or success2:
        print("🎉 SO8Tモデルが正常に動作しています！")
    else:
        print("💥 SO8Tモデルに問題があります。")

if __name__ == "__main__":
    main()
