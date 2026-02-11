#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama APIレスポンスのデバッグスクリプト
"""

import requests
import json
import time

def debug_ollama_response():
    """Ollama APIのレスポンスを詳しく調べる"""
    print("Ollama APIレスポンスデバッグ開始！")
    
    ollama_url = "http://localhost:11434"
    model_name = "so8t-qwen2vl-2b:latest"
    
    # シンプルなプロンプトでテスト
    prompt = "こんにちは！"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"プロンプト: {prompt}")
    print(f"モデル: {model_name}")
    print("APIリクエスト送信中...")
    
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
        print(f"レスポンスヘッダー: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"JSONレスポンス: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 各フィールドを個別に確認
            print("\n=== レスポンス解析 ===")
            for key, value in result.items():
                print(f"{key}: {type(value)} = {repr(value)}")
            
            # 生成テキストの確認
            if 'response' in result:
                generated_text = result['response']
                print(f"\n生成テキスト (長さ: {len(generated_text)}):")
                print(f"'{generated_text}'")
            else:
                print("'response'フィールドが見つかりません")
                
        else:
            print(f"エラー: {response.status_code}")
            print(f"エラー内容: {response.text}")
            
    except Exception as e:
        print(f"例外エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ollama_response()
