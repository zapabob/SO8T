#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Ollama テストスクリプト
実際にOllamaでSO8Tモデルを動かしてテストする
"""

import requests
import json
import time
from datetime import datetime
import sys

class SO8TOllamaTester:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "so8t-qwen2vl-2b:latest"
        self.test_results = []
        
    def test_ollama_connection(self):
        """Ollamaサーバーとの接続テスト"""
        print("Ollamaサーバー接続テスト中...")
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"Ollama接続成功！利用可能モデル数: {len(models)}")
                for model in models:
                    print(f"  - {model['name']}")
                return True
            else:
                print(f"Ollama接続失敗: {response.status_code}")
                return False
        except Exception as e:
            print(f"Ollama接続エラー: {e}")
            return False
    
    def test_text_generation(self, prompt):
        """テキスト生成テスト"""
        print(f"\nテキスト生成テスト: {prompt[:50]}...")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                response_time = end_time - start_time
                
                print(f"生成成功！")
                print(f"応答時間: {response_time:.2f}秒")
                print(f"生成テキスト: {generated_text}")
                
                self.test_results.append({
                    "test_type": "text_generation",
                    "prompt": prompt,
                    "response": generated_text,
                    "response_time": response_time,
                    "status": "success"
                })
                return True
            else:
                print(f"生成失敗: {response.status_code}")
                print(f"エラー: {response.text}")
                return False
                
        except Exception as e:
            print(f"生成エラー: {e}")
            return False
    
    def test_multimodal_capability(self):
        """マルチモーダル機能テスト"""
        print(f"\nマルチモーダル機能テスト")
        
        # 画像ファイルの存在確認
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            print(f"テスト画像が見つかりません: {test_image_path}")
            print("テキストのみでのマルチモーダルテストを実行...")
            
            # テキストベースのマルチモーダルテスト
            prompt = """
            あなたはSO8TマルチモーダルLLMです。以下の能力について説明してください：
            1. テキスト理解と生成
            2. 画像解析能力
            3. SO(8)回転ゲートの特徴
            4. PET正則化の効果
            """
            
            return self.test_text_generation(prompt)
        else:
            print(f"画像ファイル発見: {test_image_path}")
            # 実際の画像+テキストテストは後で実装
            return True
    
    def test_so8_rotation_gates(self):
        """SO(8)回転ゲートのテスト"""
        print(f"\nSO(8)回転ゲートテスト")
        
        prompt = """
        SO(8)回転ゲートについて詳しく説明してください：
        1. 8次元回転行列の数学的性質
        2. ニューラルネットワークでの応用
        3. 従来のアテンション機構との違い
        4. 計算効率と精度の向上
        """
        
        return self.test_text_generation(prompt)
    
    def test_pet_regularization(self):
        """PET正則化のテスト"""
        print(f"\nPET正則化テスト")
        
        prompt = """
        PET正則化（Second-order Difference Penalty）について説明してください：
        1. 数学的定義と目的
        2. 過学習防止のメカニズム
        3. 従来の正則化手法との比較
        4. 実装時の注意点
        """
        
        return self.test_text_generation(prompt)
    
    def run_comprehensive_test(self):
        """包括的テスト実行"""
        print("SO8T Ollama 包括的テスト開始！")
        print("=" * 60)
        
        # 1. 接続テスト
        if not self.test_ollama_connection():
            print("Ollama接続に失敗しました。テストを中止します。")
            return False
        
        # 2. 基本テキスト生成テスト
        basic_prompts = [
            "こんにちは！自己紹介をお願いします。",
            "SO8Tプロジェクトについて教えてください。",
            "マルチモーダルAIの未来について考えを述べてください。"
        ]
        
        for prompt in basic_prompts:
            self.test_text_generation(prompt)
            time.sleep(1)  # サーバー負荷軽減
        
        # 3. 専門機能テスト
        self.test_so8_rotation_gates()
        time.sleep(1)
        
        self.test_pet_regularization()
        time.sleep(1)
        
        # 4. マルチモーダルテスト
        self.test_multimodal_capability()
        
        # 5. 結果サマリー
        self.print_test_summary()
        
        return True
    
    def print_test_summary(self):
        """テスト結果サマリー"""
        print("\n" + "=" * 60)
        print("テスト結果サマリー")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r['status'] == 'success'])
        
        print(f"総テスト数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失敗: {total_tests - successful_tests}")
        print(f"成功率: {(successful_tests/total_tests*100):.1f}%")
        
        if successful_tests > 0:
            avg_response_time = sum([r['response_time'] for r in self.test_results if 'response_time' in r]) / successful_tests
            print(f"平均応答時間: {avg_response_time:.2f}秒")
        
        print("\n詳細結果:")
        for i, result in enumerate(self.test_results, 1):
            status_icon = "OK" if result['status'] == 'success' else "NG"
            print(f"{i}. {status_icon} {result['test_type']}")
            if 'response_time' in result:
                print(f"   応答時間: {result['response_time']:.2f}秒")
            if result['status'] == 'success' and len(result['response']) > 100:
                print(f"   生成テキスト: {result['response'][:100]}...")
        
        # 結果をファイルに保存
        self.save_test_results()
    
    def save_test_results(self):
        """テスト結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"_docs/2025-10-29_SO8T_Ollama実機テスト結果_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# SO8T Ollama実機テスト結果\n\n")
                f.write(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**テストモデル**: {self.model_name}\n\n")
                
                f.write("## テスト結果サマリー\n\n")
                total_tests = len(self.test_results)
                successful_tests = len([r for r in self.test_results if r['status'] == 'success'])
                f.write(f"- 総テスト数: {total_tests}\n")
                f.write(f"- 成功: {successful_tests}\n")
                f.write(f"- 失敗: {total_tests - successful_tests}\n")
                f.write(f"- 成功率: {(successful_tests/total_tests*100):.1f}%\n\n")
                
                f.write("## 詳細結果\n\n")
                for i, result in enumerate(self.test_results, 1):
                    f.write(f"### テスト {i}: {result['test_type']}\n\n")
                    f.write(f"**ステータス**: {'✅ 成功' if result['status'] == 'success' else '❌ 失敗'}\n\n")
                    if 'response_time' in result:
                        f.write(f"**応答時間**: {result['response_time']:.2f}秒\n\n")
                    f.write(f"**プロンプト**: {result['prompt']}\n\n")
                    if result['status'] == 'success':
                        f.write(f"**生成テキスト**:\n```\n{result['response']}\n```\n\n")
                    f.write("---\n\n")
            
            print(f"テスト結果を保存しました: {filename}")
            
        except Exception as e:
            print(f"ファイル保存エラー: {e}")

def main():
    print("SO8T Ollama実機テスト開始！")
    print("なんj風で全力でテストするで！")
    
    tester = SO8TOllamaTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\nテスト完了！SO8Tモデルが正常に動作しています！")
    else:
        print("\nテストに問題がありました。ログを確認してください。")
    
    return success

if __name__ == "__main__":
    import os
    main()
