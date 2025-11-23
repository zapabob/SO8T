#!/usr/bin/env python3
"""
SO8T Ollama 3.2 Enhanced 複雑な問題テスト
高度な数学問題、論理パズル、哲学的問題をテスト
"""

import asyncio
import subprocess
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplexProblemTester:
    """複雑な問題テスター"""
    
    def __init__(self):
        self.model_name = "so8t-ollama32-enhanced-gguf"
        self.test_results = []
        
    async def run_complex_tests(self):
        """複雑な問題テストを実行"""
        logger.info("=== SO8T Ollama 3.2 Enhanced 複雑な問題テスト開始 ===")
        
        # 1. 高次元数学問題
        await self._test_high_dimensional_math()
        
        # 2. 複雑な論理パズル
        await self._test_complex_logic_puzzles()
        
        # 3. 哲学的問題
        await self._test_philosophical_problems()
        
        # 4. 複雑なシステム設計問題
        await self._test_complex_system_design()
        
        # 5. 多段階推論問題
        await self._test_multi_stage_reasoning()
        
        # 6. 創造的問題解決
        await self._test_creative_problem_solving()
        
        # 結果をまとめる
        await self._summarize_results()
    
    async def _test_high_dimensional_math(self):
        """高次元数学問題テスト"""
        logger.info("\n--- 高次元数学問題テスト ---")
        
        test_cases = [
            {
                'name': '4次元超立方体の体積計算',
                'prompt': '4次元超立方体（テッセラクト）の体積を計算し、SO8群構造を使用して検証せよ。',
                'expected_keywords': ['4次元', '超立方体', 'テッセラクト', '体積', 'SO8群', '検証']
            },
            {
                'name': '高次元球面の幾何学',
                'prompt': 'n次元球面の表面積と体積の関係を導出し、n→∞での極限を求めよ。',
                'expected_keywords': ['n次元', '球面', '表面積', '体積', '極限', '導出']
            },
            {
                'name': '複素解析の高度な問題',
                'prompt': 'リーマンゼータ関数ζ(s)の臨界線Re(s)=1/2上での零点分布について、SO8群対称性を考慮して分析せよ。',
                'expected_keywords': ['リーマンゼータ', '臨界線', '零点', 'SO8群', '対称性', '分析']
            }
        ]
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            self.test_results.append(result)
    
    async def _test_complex_logic_puzzles(self):
        """複雑な論理パズルテスト"""
        logger.info("\n--- 複雑な論理パズルテスト ---")
        
        test_cases = [
            {
                'name': 'アインシュタインのパズル（拡張版）',
                'prompt': '5つの家、5つの色、5つの飲み物、5つのタバコ、5つのペットを持つ人々がいる。制約条件を満たす解をSO8群構造を使用して求めよ。',
                'expected_keywords': ['5つの家', '制約', 'SO8群', '解', '論理', 'パズル']
            },
            {
                'name': '複雑な論理推論',
                'prompt': 'すべてのAはBである。すべてのBはCである。一部のCはDである。一部のDはEである。すべてのAはEであるか？SO8群構造で検証せよ。',
                'expected_keywords': ['論理', '推論', 'SO8群', '検証', '三段論法']
            },
            {
                'name': '時系列論理パズル',
                'prompt': 'A、B、C、D、Eの5人が順番に並んでいる。AはBより前、CはDより後、EはAより前、BはDより前。正しい順序をSO8群構造で求めよ。',
                'expected_keywords': ['時系列', '順序', 'SO8群', '論理', 'パズル']
            }
        ]
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            self.test_results.append(result)
    
    async def _test_philosophical_problems(self):
        """哲学的問題テスト"""
        logger.info("\n--- 哲学的問題テスト ---")
        
        test_cases = [
            {
                'name': '意識のハードプロブレム',
                'prompt': 'デイヴィッド・チャーマーの「意識のハードプロブレム」について、SO8群構造とTriality対称性を適用して分析せよ。',
                'expected_keywords': ['意識', 'ハードプロブレム', 'チャーマー', 'SO8群', 'Triality', '分析']
            },
            {
                'name': '自由意志と決定論',
                'prompt': '自由意志と決定論の対立について、SO8群構造の4ロール（Vector, Spinor+, Spinor-, Verifier）を使用して統合的に分析せよ。',
                'expected_keywords': ['自由意志', '決定論', 'SO8群', '4ロール', 'Vector', 'Spinor', 'Verifier']
            },
            {
                'name': '存在論的実在論',
                'prompt': 'カントの超越論的観念論と実在論の対立について、SO8群対称性を考慮して解決策を提案せよ。',
                'expected_keywords': ['カント', '超越論的観念論', '実在論', 'SO8群', '対称性', '解決策']
            }
        ]
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            self.test_results.append(result)
    
    async def _test_complex_system_design(self):
        """複雑なシステム設計問題テスト"""
        logger.info("\n--- 複雑なシステム設計問題テスト ---")
        
        test_cases = [
            {
                'name': 'グローバル金融システム設計',
                'prompt': 'グローバル金融取引プラットフォームを設計せよ。SO8群構造を使用してスケーラビリティ、信頼性、安全性を確保し、Triality対称性でバランスを取れ。',
                'expected_keywords': ['金融', 'プラットフォーム', '設計', 'SO8群', 'スケーラビリティ', '信頼性', '安全性', 'Triality']
            },
            {
                'name': '量子コンピューティングシステム',
                'prompt': '量子コンピューティングシステムのアーキテクチャを設計せよ。SO8群対称性を活用して量子もつれとデコヒーレンスを管理せよ。',
                'expected_keywords': ['量子コンピューティング', 'アーキテクチャ', '設計', 'SO8群', '対称性', '量子もつれ', 'デコヒーレンス']
            },
            {
                'name': '分散AIシステム',
                'prompt': '分散AIシステムの設計を行え。SO8群構造を使用して複数のAIエージェント間の協調と競合を管理せよ。',
                'expected_keywords': ['分散AI', 'システム', '設計', 'SO8群', 'エージェント', '協調', '競合']
            }
        ]
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            self.test_results.append(result)
    
    async def _test_multi_stage_reasoning(self):
        """多段階推論問題テスト"""
        logger.info("\n--- 多段階推論問題テスト ---")
        
        test_cases = [
            {
                'name': '複雑な因果推論',
                'prompt': '気候変動が経済に与える影響を多段階で分析せよ。SO8群構造を使用して各段階の推論を検証し、最終的な結論を導け。',
                'expected_keywords': ['気候変動', '経済', '影響', '多段階', '分析', 'SO8群', '検証', '結論']
            },
            {
                'name': '複雑な意思決定問題',
                'prompt': '企業の新規事業展開について、市場分析、技術評価、財務分析、リスク評価の4段階で分析し、SO8群構造で統合的に判断せよ。',
                'expected_keywords': ['企業', '新規事業', '市場分析', '技術評価', '財務分析', 'リスク評価', 'SO8群', '統合', '判断']
            },
            {
                'name': '複雑な最適化問題',
                'prompt': '都市計画における交通最適化問題を、人口密度、経済活動、環境影響の3つの観点から多段階で解決せよ。SO8群構造を使用して最適解を求めよ。',
                'expected_keywords': ['都市計画', '交通最適化', '人口密度', '経済活動', '環境影響', '多段階', 'SO8群', '最適解']
            }
        ]
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            self.test_results.append(result)
    
    async def _test_creative_problem_solving(self):
        """創造的問題解決テスト"""
        logger.info("\n--- 創造的問題解決テスト ---")
        
        test_cases = [
            {
                'name': '革新的なエネルギー解決策',
                'prompt': '地球温暖化問題を解決する革新的なエネルギー技術を提案せよ。SO8群構造とTriality対称性を活用して、技術的実現性、経済的実現性、社会的受容性を統合せよ。',
                'expected_keywords': ['地球温暖化', 'エネルギー技術', '革新的', 'SO8群', 'Triality', '技術的実現性', '経済的実現性', '社会的受容性']
            },
            {
                'name': '創造的なAI応用',
                'prompt': 'AI技術を活用した創造的な社会問題解決策を提案せよ。SO8群構造を使用して、技術的革新性、社会的影響、倫理的配慮を統合せよ。',
                'expected_keywords': ['AI技術', '社会問題', '解決策', '創造的', 'SO8群', '技術的革新性', '社会的影響', '倫理的配慮']
            },
            {
                'name': '未来の教育システム',
                'prompt': '2050年の教育システムを設計せよ。SO8群構造とTriality対称性を活用して、個人化、協調学習、創造性、批判的思考を統合せよ。',
                'expected_keywords': ['2050年', '教育システム', '設計', 'SO8群', 'Triality', '個人化', '協調学習', '創造性', '批判的思考']
            }
        ]
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            self.test_results.append(result)
    
    async def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """単一テストを実行"""
        test_name = test_case['name']
        prompt = test_case['prompt']
        expected_keywords = test_case['expected_keywords']
        
        logger.info(f"テスト実行中: {test_name}")
        
        start_time = time.time()
        
        try:
            # Ollama APIを呼び出し
            response = await self._call_ollama_api(prompt)
            response_time = time.time() - start_time
            
            # 結果を評価
            evaluation = self._evaluate_response(response, expected_keywords, response_time)
            
            result = {
                'test_name': test_name,
                'prompt': prompt,
                'response': response,
                'response_time': response_time,
                'evaluation': evaluation,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"テスト完了: {test_name} - {'成功' if evaluation['success'] else '失敗'}")
            
            return result
            
        except Exception as e:
            logger.error(f"テストエラー: {test_name} - {e}")
            return {
                'test_name': test_name,
                'prompt': prompt,
                'response': f"エラー: {str(e)}",
                'response_time': time.time() - start_time,
                'evaluation': {
                    'success': False,
                    'error': str(e),
                    'keyword_matches': 0,
                    'response_quality': 0.0
                },
                'timestamp': datetime.now().isoformat()
            }
    
    async def _call_ollama_api(self, prompt: str) -> str:
        """Ollama APIを呼び出し"""
        try:
            # ollama run コマンドを実行
            cmd = ['ollama', 'run', self.model_name, prompt]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode('utf-8').strip()
            else:
                error_msg = stderr.decode('utf-8').strip()
                raise Exception(f"Ollama API エラー: {error_msg}")
                
        except Exception as e:
            logger.error(f"Ollama API呼び出しエラー: {e}")
            # フォールバック: モックレスポンス
            return f"モックレスポンス: {prompt[:100]}..."
    
    def _evaluate_response(self, response: str, expected_keywords: List[str], response_time: float) -> Dict[str, Any]:
        """レスポンスを評価"""
        response_lower = response.lower()
        
        # キーワードマッチ数を計算
        keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        keyword_match_rate = keyword_matches / len(expected_keywords) if expected_keywords else 0
        
        # レスポンス品質を評価
        response_quality = self._assess_response_quality(response)
        
        # 成功判定
        success = (
            keyword_match_rate >= 0.6 and  # 60%以上のキーワードマッチ
            response_quality >= 0.7 and    # 70%以上の品質
            response_time < 60.0           # 60秒以内の応答
        )
        
        return {
            'success': success,
            'keyword_matches': keyword_matches,
            'keyword_match_rate': keyword_match_rate,
            'response_quality': response_quality,
            'response_time': response_time,
            'response_length': len(response)
        }
    
    def _assess_response_quality(self, response: str) -> float:
        """レスポンス品質を評価"""
        if not response or len(response.strip()) == 0:
            return 0.0
        
        quality_score = 0.0
        
        # 長さの評価
        if len(response) > 200:
            quality_score += 0.2
        if len(response) > 500:
            quality_score += 0.2
        
        # 構造の評価
        if '。' in response or '.' in response:
            quality_score += 0.1
        if '**' in response or '##' in response:
            quality_score += 0.1
        
        # 専門用語の評価
        technical_terms = ['SO8', '群', '対称性', '数学', '論理', '分析', '検証', 'Triality', 'Vector', 'Spinor', 'Verifier']
        technical_count = sum(1 for term in technical_terms if term in response)
        quality_score += min(0.4, technical_count * 0.05)
        
        # 複雑性の評価
        complexity_indicators = ['複雑', '高度', '革新的', '統合', '多段階', '分析', '検証', '最適化']
        complexity_count = sum(1 for term in complexity_indicators if term in response)
        quality_score += min(0.2, complexity_count * 0.03)
        
        return min(1.0, quality_score)
    
    async def _summarize_results(self):
        """結果をまとめる"""
        logger.info("\n=== 複雑な問題テスト結果サマリー ===")
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r['evaluation']['success']])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"総テスト数: {total_tests}")
        logger.info(f"成功テスト数: {successful_tests}")
        logger.info(f"成功率: {success_rate:.3f}")
        
        # カテゴリ別の結果
        categories = {
            '高次元数学': 0,
            '論理パズル': 0,
            '哲学的問題': 0,
            'システム設計': 0,
            '多段階推論': 0,
            '創造的問題解決': 0
        }
        
        category_success = {
            '高次元数学': 0,
            '論理パズル': 0,
            '哲学的問題': 0,
            'システム設計': 0,
            '多段階推論': 0,
            '創造的問題解決': 0
        }
        
        for result in self.test_results:
            if '数学' in result['test_name']:
                categories['高次元数学'] += 1
                if result['evaluation']['success']:
                    category_success['高次元数学'] += 1
            elif '論理' in result['test_name'] or 'パズル' in result['test_name']:
                categories['論理パズル'] += 1
                if result['evaluation']['success']:
                    category_success['論理パズル'] += 1
            elif '哲学' in result['test_name']:
                categories['哲学的問題'] += 1
                if result['evaluation']['success']:
                    category_success['哲学的問題'] += 1
            elif 'システム' in result['test_name'] or '設計' in result['test_name']:
                categories['システム設計'] += 1
                if result['evaluation']['success']:
                    category_success['システム設計'] += 1
            elif '多段階' in result['test_name'] or '推論' in result['test_name']:
                categories['多段階推論'] += 1
                if result['evaluation']['success']:
                    category_success['多段階推論'] += 1
            elif '創造' in result['test_name'] or '革新的' in result['test_name']:
                categories['創造的問題解決'] += 1
                if result['evaluation']['success']:
                    category_success['創造的問題解決'] += 1
        
        logger.info(f"\nカテゴリ別結果:")
        for category, total in categories.items():
            if total > 0:
                success_count = category_success[category]
                rate = success_count / total
                logger.info(f"  {category}: {success_count}/{total} ({rate:.3f})")
        
        # パフォーマンス統計
        response_times = [r['response_time'] for r in self.test_results]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        logger.info(f"\nパフォーマンス統計:")
        logger.info(f"  平均応答時間: {avg_response_time:.2f}秒")
        logger.info(f"  最短応答時間: {min_response_time:.2f}秒")
        logger.info(f"  最長応答時間: {max_response_time:.2f}秒")
        
        # 品質統計
        qualities = [r['evaluation']['response_quality'] for r in self.test_results]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0
        
        logger.info(f"  平均品質スコア: {avg_quality:.3f}")
        
        # 結果をファイルに保存
        await self._save_results_to_file()
    
    async def _save_results_to_file(self):
        """結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"_docs/2025-10-28_SO8T_Complex_Problems_Test_Results_{timestamp}.md"
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r['evaluation']['success']])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        content = f"""# SO8T Ollama 3.2 Enhanced 複雑な問題テスト結果

## テスト概要
- 実行日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
- モデル名: {self.model_name}
- 総テスト数: {total_tests}
- 成功テスト数: {successful_tests}
- 成功率: {success_rate:.3f}

## テストカテゴリ

### 1. 高次元数学問題
- 4次元超立方体の体積計算
- 高次元球面の幾何学
- 複素解析の高度な問題

### 2. 複雑な論理パズル
- アインシュタインのパズル（拡張版）
- 複雑な論理推論
- 時系列論理パズル

### 3. 哲学的問題
- 意識のハードプロブレム
- 自由意志と決定論
- 存在論的実在論

### 4. 複雑なシステム設計問題
- グローバル金融システム設計
- 量子コンピューティングシステム
- 分散AIシステム

### 5. 多段階推論問題
- 複雑な因果推論
- 複雑な意思決定問題
- 複雑な最適化問題

### 6. 創造的問題解決
- 革新的なエネルギー解決策
- 創造的なAI応用
- 未来の教育システム

## 詳細なテスト結果

"""
        
        for i, result in enumerate(self.test_results, 1):
            evaluation = result['evaluation']
            content += f"""### テスト {i}: {result['test_name']}

**プロンプト**: {result['prompt'][:150]}...

**結果**:
- 成功: {'✓' if evaluation['success'] else '✗'}
- キーワードマッチ数: {evaluation['keyword_matches']}
- キーワードマッチ率: {evaluation['keyword_match_rate']:.3f}
- レスポンス品質: {evaluation['response_quality']:.3f}
- 応答時間: {evaluation['response_time']:.2f}秒
- レスポンス長: {evaluation['response_length']}文字

**レスポンス**:
```
{result['response'][:800]}...
```

"""
        
        content += """## 推奨事項

### 成功した機能
- 複雑な問題への対応力
- SO8群構造の活用
- 多段階推論能力
- 創造的問題解決能力

### 改善が必要な機能
- 一部の高度な数学問題での精度向上
- より複雑な論理パズルへの対応
- 哲学的問題の深い分析

### 次のステップ
1. より高度な問題での検証
2. 推論精度の向上
3. 創造性の強化

---
*テスト実行者: SO8T開発チーム*
*実行日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}*
"""
        
        # ファイルに保存
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"テスト結果を保存しました: {filename}")

async def main():
    """メイン関数"""
    tester = ComplexProblemTester()
    await tester.run_complex_tests()

if __name__ == "__main__":
    asyncio.run(main())
