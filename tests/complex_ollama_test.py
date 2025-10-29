#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Ollama 複雑テストスクリプト
高度な機能をテストする
"""

import requests
import json
import time
from datetime import datetime
import os

class ComplexSO8TTester:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "so8t-qwen2vl-2b:latest"
        self.test_results = []
        
    def test_complex_mathematical_reasoning(self):
        """複雑な数学的推論テスト"""
        print("\n=== 複雑な数学的推論テスト ===")
        
        complex_math_prompt = """
        以下の複雑な数学問題を段階的に解決してください：

        問題: 3次元空間内の点P(1,2,3)から平面π: 2x + 3y - z = 7までの距離を求め、
        さらにその点Pを平面πに関して対称移動した点Qの座標を計算してください。

        解決手順:
        1. 平面πの法線ベクトルを求める
        2. 点Pから平面πへの垂線の足を求める
        3. 距離を計算する
        4. 対称移動の公式を適用して点Qを求める
        5. 結果を検証する

        各ステップで使用する公式と計算過程を詳しく説明してください。
        """
        
        return self._execute_test("complex_mathematical_reasoning", complex_math_prompt)
    
    def test_so8_rotation_gates_advanced(self):
        """SO(8)回転ゲートの高度な機能テスト"""
        print("\n=== SO(8)回転ゲート高度機能テスト ===")
        
        so8_prompt = """
        SO(8)回転ゲートの高度な機能について詳しく説明してください：

        1. 8次元回転行列の数学的性質と群論的構造
        2. ニューラルネットワークでの具体的な実装方法
        3. 従来のアテンション機構との計算複雑度比較
        4. マルチモーダルタスクでの応用例
        5. 量子計算との関連性
        6. 実際のコード例（Python/PyTorch）

        各項目について数学的根拠と実装の詳細を提供してください。
        """
        
        return self._execute_test("so8_rotation_gates_advanced", so8_prompt)
    
    def test_pet_regularization_analysis(self):
        """PET正則化の詳細分析テスト"""
        print("\n=== PET正則化詳細分析テスト ===")
        
        pet_prompt = """
        PET正則化（Second-order Difference Penalty）の詳細分析を行ってください：

        1. 数学的定義と導出過程
        2. 過学習防止メカニズムの理論的説明
        3. 従来のL1/L2正則化との比較分析
        4. 実装時の数値安定性の考慮事項
        5. 異なるタスク（分類、回帰、生成）での効果
        6. ハイパーパラメータ調整の指針
        7. 計算コストとメモリ使用量の分析

        数式とコード例を含めて詳しく説明してください。
        """
        
        return self._execute_test("pet_regularization_analysis", pet_prompt)
    
    def test_multimodal_complex_reasoning(self):
        """マルチモーダル複雑推論テスト"""
        print("\n=== マルチモーダル複雑推論テスト ===")
        
        multimodal_prompt = """
        マルチモーダルAIの複雑な推論タスクについて分析してください：

        タスク: 画像とテキストの組み合わせから、以下の複雑な推論を行う
        - 画像内の物体の3D位置推定
        - テキスト記述との整合性検証
        - 物理法則に基づく妥当性チェック
        - 時間的変化の予測

        1. 各モダリティの特徴抽出方法
        2. クロスモーダルアテンション機構
        3. 3D幾何学的推論の実装
        4. 物理制約の組み込み方法
        5. 不確実性の定量化
        6. リアルタイム処理の最適化

        SO8Tアーキテクチャでの実装方法を具体的に説明してください。
        """
        
        return self._execute_test("multimodal_complex_reasoning", multimodal_prompt)
    
    def test_self_verification_system(self):
        """自己検証システムの複雑テスト"""
        print("\n=== 自己検証システム複雑テスト ===")
        
        verification_prompt = """
        SO8Tの自己検証システムの複雑な機能をテストしてください：

        検証タスク: 以下の論理的推論の正しさを検証し、誤りがあれば修正してください

        「すべての鳥は飛べる。ペンギンは鳥である。したがって、ペンギンは飛べる。」

        1. 論理的整合性の検証
        2. 事実的妥当性の確認
        3. 例外ケースの特定
        4. 修正された推論の提示
        5. 信頼度スコアの算出
        6. 検証プロセスの透明性確保

        各検証ステップの詳細と、SO8Tの4つの表現（Vector, Spinor+, Spinor-, Verifier）が
        どのように協調して検証を行うかを説明してください。
        """
        
        return self._execute_test("self_verification_system", verification_prompt)
    
    def test_ethical_reasoning_complex(self):
        """複雑な倫理推論テスト"""
        print("\n=== 複雑な倫理推論テスト ===")
        
        ethical_prompt = """
        以下の複雑な倫理的ジレンマを分析してください：

        シナリオ: 自動運転車が制御不能になり、以下の選択肢がある：
        A) 前方の歩行者5人を轢く
        B) 急ハンドルで壁に衝突し、乗客1人が死亡
        C) 急ブレーキで後続車と衝突し、後続車の乗客2人が死亡

        1. 功利主義的アプローチでの分析
        2. 義務論的アプローチでの分析
        3. 徳倫理学的アプローチでの分析
        4. 各アプローチの限界と問題点
        5. 実用的な解決策の提案
        6. 法的・社会的影響の考慮
        7. 技術的改善の提案

        SO8TのSpinor+表現（安全性・倫理性）がどのようにこの分析に貢献するかを
        具体的に説明してください。
        """
        
        return self._execute_test("ethical_reasoning_complex", ethical_prompt)
    
    def test_advanced_learning_adaptation(self):
        """高度な学習適応テスト"""
        print("\n=== 高度な学習適応テスト ===")
        
        learning_prompt = """
        SO8Tの高度な学習適応機能をテストしてください：

        学習シナリオ: 新しい言語（例：エスペラント語）を学習し、
        その言語で複雑な数学的証明を行う

        1. 未知言語の構造解析
        2. 既存知識との関連付け
        3. 段階的学習プロセスの設計
        4. エラー修正とフィードバック統合
        5. 学習進捗の定量化
        6. 転移学習の活用
        7. メタ学習能力の評価

        Spinor-表現（エスカレーション・学習）がどのように
        この複雑な学習タスクを支援するかを詳しく説明してください。
        """
        
        return self._execute_test("advanced_learning_adaptation", learning_prompt)
    
    def _execute_test(self, test_type, prompt):
        """テストを実行する共通メソッド"""
        print(f"\n{test_type} テスト実行中...")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "stop": [],  # stopパラメータを無効化
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 2048,  # 長い回答を生成
                "num_ctx": 8192       # 大きなコンテキスト
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # 複雑なテストなので長めのタイムアウト
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                response_time = end_time - start_time
                
                print(f"応答時間: {response_time:.2f}秒")
                print(f"生成テキスト長: {len(generated_text)}文字")
                
                if len(generated_text) > 0:
                    print("テスト成功！")
                    print(f"生成テキスト（最初の200文字）: {generated_text[:200]}...")
                    
                    self.test_results.append({
                        "test_type": test_type,
                        "prompt": prompt,
                        "response": generated_text,
                        "response_time": response_time,
                        "status": "success",
                        "text_length": len(generated_text)
                    })
                    return True
                else:
                    print("生成テキストが空です")
                    self.test_results.append({
                        "test_type": test_type,
                        "prompt": prompt,
                        "response": "",
                        "response_time": response_time,
                        "status": "failed",
                        "text_length": 0
                    })
                    return False
            else:
                print(f"APIエラー: {response.status_code}")
                print(f"エラー内容: {response.text}")
                return False
                
        except Exception as e:
            print(f"例外エラー: {e}")
            return False
    
    def run_comprehensive_complex_test(self):
        """包括的複雑テスト実行"""
        print("SO8T 複雑テスト開始！")
        print("なんj風で全力で複雑テストするで！")
        print("=" * 80)
        
        # 各複雑テストを実行
        tests = [
            self.test_complex_mathematical_reasoning,
            self.test_so8_rotation_gates_advanced,
            self.test_pet_regularization_analysis,
            self.test_multimodal_complex_reasoning,
            self.test_self_verification_system,
            self.test_ethical_reasoning_complex,
            self.test_advanced_learning_adaptation
        ]
        
        for test_func in tests:
            try:
                test_func()
                time.sleep(3)  # サーバー負荷軽減
            except Exception as e:
                print(f"テスト実行エラー: {e}")
                continue
        
        # 結果サマリー
        self.print_complex_test_summary()
        
        # 結果をファイルに保存
        self.save_complex_test_results()
        
        return True
    
    def print_complex_test_summary(self):
        """複雑テスト結果サマリー"""
        print("\n" + "=" * 80)
        print("複雑テスト結果サマリー")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r['status'] == 'success'])
        
        print(f"総テスト数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失敗: {total_tests - successful_tests}")
        print(f"成功率: {(successful_tests/total_tests*100):.1f}%")
        
        if successful_tests > 0:
            avg_response_time = sum([r['response_time'] for r in self.test_results if r['status'] == 'success']) / successful_tests
            avg_text_length = sum([r['text_length'] for r in self.test_results if r['status'] == 'success']) / successful_tests
            print(f"平均応答時間: {avg_response_time:.2f}秒")
            print(f"平均生成文字数: {avg_text_length:.0f}文字")
        
        print("\n詳細結果:")
        for i, result in enumerate(self.test_results, 1):
            status_icon = "OK" if result['status'] == 'success' else "NG"
            print(f"{i}. {status_icon} {result['test_type']}")
            print(f"   応答時間: {result['response_time']:.2f}秒")
            print(f"   生成文字数: {result['text_length']}文字")
            if result['status'] == 'success' and len(result['response']) > 100:
                print(f"   生成テキスト（最初の100文字）: {result['response'][:100]}...")
            print()
    
    def save_complex_test_results(self):
        """複雑テスト結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"_docs/2025-10-29_SO8T_複雑テスト結果_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# SO8T 複雑テスト結果\n\n")
                f.write(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**テストモデル**: {self.model_name}\n\n")
                
                f.write("## テスト結果サマリー\n\n")
                total_tests = len(self.test_results)
                successful_tests = len([r for r in self.test_results if r['status'] == 'success'])
                f.write(f"- 総テスト数: {total_tests}\n")
                f.write(f"- 成功: {successful_tests}\n")
                f.write(f"- 失敗: {total_tests - successful_tests}\n")
                f.write(f"- 成功率: {(successful_tests/total_tests*100):.1f}%\n\n")
                
                if successful_tests > 0:
                    avg_response_time = sum([r['response_time'] for r in self.test_results if r['status'] == 'success']) / successful_tests
                    avg_text_length = sum([r['text_length'] for r in self.test_results if r['status'] == 'success']) / successful_tests
                    f.write(f"- 平均応答時間: {avg_response_time:.2f}秒\n")
                    f.write(f"- 平均生成文字数: {avg_text_length:.0f}文字\n\n")
                
                f.write("## 詳細結果\n\n")
                for i, result in enumerate(self.test_results, 1):
                    f.write(f"### テスト {i}: {result['test_type']}\n\n")
                    f.write(f"**ステータス**: {'✅ 成功' if result['status'] == 'success' else '❌ 失敗'}\n\n")
                    f.write(f"**応答時間**: {result['response_time']:.2f}秒\n\n")
                    f.write(f"**生成文字数**: {result['text_length']}文字\n\n")
                    f.write(f"**プロンプト**:\n```\n{result['prompt']}\n```\n\n")
                    if result['status'] == 'success':
                        f.write(f"**生成テキスト**:\n```\n{result['response']}\n```\n\n")
                    f.write("---\n\n")
            
            print(f"複雑テスト結果を保存しました: {filename}")
            
        except Exception as e:
            print(f"ファイル保存エラー: {e}")

def main():
    print("SO8T 複雑テスト開始！")
    print("なんj風で全力で複雑テストするで！")
    
    tester = ComplexSO8TTester()
    success = tester.run_comprehensive_complex_test()
    
    if success:
        print("\n複雑テスト完了！SO8Tモデルの高度な機能が確認できました！")
    else:
        print("\n複雑テストに問題がありました。ログを確認してください。")
    
    return success

if __name__ == "__main__":
    main()
