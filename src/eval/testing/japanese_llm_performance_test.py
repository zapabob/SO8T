#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語LLMパフォーマンステスト

Ollamaモデルに対して日本語での包括的なパフォーマンステストを実行
"""

import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/japanese_llm_performance_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JapaneseLLMPerformanceTester:
    """日本語LLMパフォーマンステスター"""
    
    def __init__(self, model_name: str, output_dir: Path):
        """
        Args:
            model_name: Ollamaモデル名（例: "borea-phi35-so8t:latest"）
            output_dir: 出力ディレクトリ
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("Japanese LLM Performance Tester Initialized")
        logger.info("="*80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Output: {output_dir}")
    
    def run_ollama_command(self, prompt: str) -> str:
        """Ollamaコマンドを実行して応答を取得"""
        cmd = [
            "ollama",
            "run",
            self.model_name,
            prompt
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama command failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            return f"[ERROR] {e.stderr}"
    
    def test_understanding(self) -> Dict:
        """日本語理解テスト"""
        logger.info("[TEST 1] Japanese Understanding Test")
        
        test_prompt = """以下の文章を読んで、内容を要約してください。日本語で回答してください。

人工知能（AI）は、コンピュータシステムが人間の知能を模倣する技術です。機械学習はAIの一分野で、データから学習してパターンを認識します。深層学習は機械学習の一種で、ニューラルネットワークを使用して複雑な問題を解決します。これらの技術は、画像認識、自然言語処理、音声認識など、様々な分野で応用されています。"""
        
        response = self.run_ollama_command(test_prompt)
        
        return {
            'category': 'understanding',
            'prompt': test_prompt,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_generation(self) -> Dict:
        """日本語生成テスト"""
        logger.info("[TEST 2] Japanese Generation Test")
        
        test_prompt = """以下のテーマについて、自然な日本語で300文字程度の文章を書いてください。

テーマ: 人工知能の未来について"""
        
        response = self.run_ollama_command(test_prompt)
        
        return {
            'category': 'generation',
            'prompt': test_prompt,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_reasoning(self) -> Dict:
        """日本語推論テスト"""
        logger.info("[TEST 3] Japanese Reasoning Test")
        
        test_prompt = """次の問題を日本語で解いてください。ステップバイステップで説明してください。

問題: ある商品の価格が20パーセント値下げされました。元の価格が5000円の場合、値下げ後の価格はいくらですか？"""
        
        response = self.run_ollama_command(test_prompt)
        
        return {
            'category': 'reasoning',
            'prompt': test_prompt,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_dialogue(self) -> Dict:
        """日本語対話テスト"""
        logger.info("[TEST 4] Japanese Dialogue Test")
        
        test_prompt = """以下の会話の続きを、自然な日本語で書いてください。

ユーザー: こんにちは。今日は良い天気ですね。
アシスタント:"""
        
        response = self.run_ollama_command(test_prompt)
        
        return {
            'category': 'dialogue',
            'prompt': test_prompt,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_technical(self) -> Dict:
        """専門用語テスト"""
        logger.info("[TEST 5] Technical Term Test")
        
        test_prompt = """「機械学習」について、専門的な内容を含めながら、日本語で分かりやすく説明してください。"""
        
        response = self.run_ollama_command(test_prompt)
        
        return {
            'category': 'technical',
            'prompt': test_prompt,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_response(self, test_result: Dict) -> Dict:
        """応答を評価（簡易版）"""
        response = test_result.get('response', '')
        
        # 簡易評価指標
        evaluation = {
            'length': len(response),
            'has_japanese': any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF' for char in response),
            'word_count': len(response.split()) if response else 0
        }
        
        return evaluation
    
    def run_all_tests(self) -> Path:
        """全テスト実行"""
        logger.info("="*80)
        logger.info("Running All Japanese Performance Tests")
        logger.info("="*80)
        
        test_results = []
        
        # 各テストを実行
        test_results.append(self.test_understanding())
        test_results.append(self.test_generation())
        test_results.append(self.test_reasoning())
        test_results.append(self.test_dialogue())
        test_results.append(self.test_technical())
        
        # 評価を追加
        for test_result in test_results:
            test_result['evaluation'] = self.evaluate_response(test_result)
        
        # 結果をJSON形式で保存
        results_data = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results
        }
        
        # ファイル名生成（worktree名を含む）
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        # worktree名を取得（簡易版）
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                check=True
            )
            git_dir = result.stdout.strip()
            if "worktrees" in git_dir:
                worktree_name = Path(git_dir).parts[-2] if "worktrees" in git_dir else "main"
            else:
                worktree_name = "main"
        except Exception:
            worktree_name = "main"
        
        model_name_safe = self.model_name.replace(':', '_').replace('/', '_')
        results_filename = f"{today}_{worktree_name}_japanese_llm_performance_test_{model_name_safe}.json"
        results_path = self.output_dir / results_filename
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Markdown形式でも保存
        markdown_path = self.output_dir / results_filename.replace('.json', '.md')
        self._generate_markdown_report(results_data, markdown_path)
        
        logger.info(f"[OK] Test results saved to {results_path}")
        logger.info(f"[OK] Markdown report saved to {markdown_path}")
        
        return results_path
    
    def _generate_markdown_report(self, results_data: Dict, output_path: Path):
        """Markdown形式のレポート生成"""
        md_content = f"""# 日本語LLMパフォーマンステスト結果

## テスト情報
- **モデル**: {results_data['model_name']}
- **実行日時**: {results_data['timestamp']}

## テスト結果

"""
        
        for test_result in results_data['test_results']:
            category = test_result['category']
            evaluation = test_result.get('evaluation', {})
            
            md_content += f"""### {category.upper()}

**プロンプト**:
```
{test_result['prompt']}
```

**応答**:
```
{test_result['response']}
```

**評価**:
- 文字数: {evaluation.get('length', 0)}
- 日本語含有: {evaluation.get('has_japanese', False)}
- 単語数: {evaluation.get('word_count', 0)}

---

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Japanese LLM Performance Test")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ollama model name (e.g., "borea-phi35-so8t:latest")'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('_docs'),
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # テスト実行
    tester = JapaneseLLMPerformanceTester(args.model, args.output_dir)
    results_path = tester.run_all_tests()
    
    logger.info("="*80)
    logger.info("[COMPLETE] Japanese Performance Test completed!")
    logger.info(f"Results: {results_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

