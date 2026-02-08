#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codex経由/thinking形式データセット作成スクリプト

OpenAI APIまたはClaude APIをターミナル経由で呼び出し、
/thinking形式（思考ステップ+最終回答）のデータセットを生成
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import random

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/create_codex_thinking_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CodexThinkingDatasetGenerator:
    """Codex（OpenAI/Claude API）を使用した/thinking形式データセット生成"""
    
    def __init__(self, api_type: str = "openai", api_key: str = None):
        """
        Args:
            api_type: APIタイプ（"openai" or "claude"）
            api_key: APIキー（Noneの場合は環境変数から取得）
        """
        self.api_type = api_type
        self.api_key = api_key or os.environ.get(f"{api_type.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"{api_type.upper()}_API_KEY environment variable is required")
        
        logger.info(f"[INIT] CodexThinkingDatasetGenerator initialized with {api_type} API")
    
    def _call_openai_api(self, prompt: str, model: str = "gpt-4") -> str:
        """OpenAI APIを呼び出し"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert AI assistant that generates thinking steps and final answers in Japanese. Always provide detailed thinking steps before the final answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[ERROR] OpenAI API call failed: {e}")
            return ""
    
    def _call_claude_api(self, prompt: str, model: str = "claude-3-opus-20240229") -> str:
        """Claude APIを呼び出し"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"[ERROR] Claude API call failed: {e}")
            return ""
    
    def _call_api(self, prompt: str, model: str = None) -> str:
        """APIを呼び出し（統一インターフェース）"""
        if self.api_type == "openai":
            return self._call_openai_api(prompt, model or "gpt-4")
        elif self.api_type == "claude":
            return self._call_claude_api(prompt, model or "claude-3-opus-20240229")
        else:
            raise ValueError(f"Unknown API type: {self.api_type}")
    
    def _build_thinking_prompt(self, user_query: str) -> str:
        """/thinking形式のプロンプトを構築"""
        prompt = (
            "以下の問題に対して、まず「# 思考ステップ」で詳細に考えを整理し、"
            "その後「# 最終回答」でユーザーへの短い答えだけを出してください。\n\n"
            f"問題: {user_query}\n\n"
            "答え:\n# 思考ステップ\n"
        )
        return prompt
    
    def _extract_thinking_and_final(self, text: str) -> Dict[str, str]:
        """思考ステップと最終回答を抽出"""
        result = {
            'thinking': '',
            'final': ''
        }
        
        # # 思考ステップと# 最終回答で分割
        import re
        
        # 思考ステップを抽出
        thinking_match = re.search(r'#\s*思考ステップ\s*\n(.*?)(?=\n#\s*最終回答|\Z)', text, re.DOTALL)
        if thinking_match:
            result['thinking'] = thinking_match.group(1).strip()
        
        # 最終回答を抽出
        final_match = re.search(r'#\s*最終回答\s*\n(.*?)(?=\n#|\Z)', text, re.DOTALL)
        if final_match:
            result['final'] = final_match.group(1).strip()
        
        # パターンが見つからない場合、簡易的な分割を試みる
        if not result['thinking'] and not result['final']:
            # 「# 思考ステップ」や「# 最終回答」がない場合、改行で分割
            lines = text.split('\n')
            thinking_lines = []
            final_lines = []
            in_thinking = True
            
            for line in lines:
                if '# 最終回答' in line or '# 最終' in line:
                    in_thinking = False
                    continue
                if '# 思考ステップ' in line or '# 思考' in line:
                    continue
                
                if in_thinking:
                    thinking_lines.append(line)
                else:
                    final_lines.append(line)
            
            result['thinking'] = '\n'.join(thinking_lines).strip()
            result['final'] = '\n'.join(final_lines).strip()
        
        return result
    
    def _evaluate_quality(self, thinking: str, final: str) -> float:
        """品質スコアを評価"""
        score = 0.0
        
        # 思考ステップの存在チェック
        if thinking and len(thinking.strip()) > 10:
            score += 0.4
        
        # 最終回答の存在チェック
        if final and len(final.strip()) > 5:
            score += 0.4
        
        # 思考ステップの詳細度
        if thinking:
            thinking_length = len(thinking)
            if 50 <= thinking_length <= 1000:
                score += 0.1
            elif thinking_length > 1000:
                score += 0.05
        
        # 最終回答の適切な長さ
        if final:
            final_length = len(final)
            if 10 <= final_length <= 500:
                score += 0.1
        
        return min(score, 1.0)
    
    def generate_thinking_samples(
        self,
        prompts: List[str],
        num_samples_per_prompt: int = 1
    ) -> List[Dict]:
        """
        /thinking形式のサンプルを生成
        
        Args:
            prompts: プロンプトリスト
            num_samples_per_prompt: プロンプトあたりのサンプル数
        
        Returns:
            /thinking形式サンプルのリスト
        """
        samples = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"[GENERATE] Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            for sample_idx in range(num_samples_per_prompt):
                # /thinking形式のプロンプトを構築
                thinking_prompt = self._build_thinking_prompt(prompt)
                
                # APIを呼び出し
                response_text = self._call_api(thinking_prompt)
                
                if not response_text:
                    logger.warning(f"[WARNING] Failed to generate response for prompt {i+1}")
                    continue
                
                # 思考ステップと最終回答を抽出
                thinking_parts = self._extract_thinking_and_final(response_text)
                
                # 品質評価
                quality_score = self._evaluate_quality(
                    thinking_parts.get('thinking', ''),
                    thinking_parts.get('final', '')
                )
                
                # サンプルを作成
                sample = {
                    "instruction": prompt,
                    "input": "",
                    "output": f"# 思考ステップ\n{thinking_parts.get('thinking', '')}\n\n# 最終回答\n{thinking_parts.get('final', '')}",
                    "thinking": thinking_parts.get('thinking', ''),
                    "final": thinking_parts.get('final', ''),
                    "quality_score": quality_score,
                    "created_at": datetime.now().isoformat(),
                    "source": f"codex_{self.api_type}"
                }
                
                samples.append(sample)
                logger.info(f"[OK] Generated sample {sample_idx+1} for prompt {i+1} (quality: {quality_score:.2f})")
                
                # APIレート制限対策
                time.sleep(1)
        
        return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate /thinking format dataset using Codex (OpenAI/Claude API)"
    )
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["openai", "claude"],
        default="openai",
        help="API type (openai or claude)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (if not provided, uses environment variable)"
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="Path to prompts file (JSONL format, one prompt per line, or JSON with 'instruction' field)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output file path (JSONL format)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per prompt (default: 1)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # プロンプトを読み込み
    prompts = []
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    # JSONL形式を試す
                    data = json.loads(line)
                    if isinstance(data, dict):
                        prompt = data.get('instruction', data.get('prompt', data.get('text', '')))
                    else:
                        prompt = str(data)
                except json.JSONDecodeError:
                    # プレーンテキストとして扱う
                    prompt = line.strip()
                
                if prompt:
                    prompts.append(prompt)
    
    logger.info(f"[MAIN] Loaded {len(prompts)} prompts from {args.prompts_file}")
    
    if not prompts:
        logger.error("[ERROR] No prompts found in input file")
        return
    
    # データセット生成器を初期化
    generator = CodexThinkingDatasetGenerator(
        api_type=args.api_type,
        api_key=args.api_key
    )
    
    # /thinking形式サンプルを生成
    logger.info(f"[MAIN] Generating {args.num_samples} samples per prompt...")
    samples = generator.generate_thinking_samples(
        prompts=prompts,
        num_samples_per_prompt=args.num_samples
    )
    
    # 品質フィルタリング
    if args.min_quality > 0:
        filtered_samples = [s for s in samples if s.get('quality_score', 0.0) >= args.min_quality]
        logger.info(f"[FILTER] Filtered {len(samples)} -> {len(filtered_samples)} samples (min_quality={args.min_quality})")
        samples = filtered_samples
    
    # 出力ファイルに保存
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"[SUCCESS] Generated {len(samples)} /thinking format samples")
    logger.info(f"[SAVE] Saved to {output_file}")
    
    # 統計情報を出力
    avg_quality = sum(s.get('quality_score', 0.0) for s in samples) / len(samples) if samples else 0.0
    logger.info(f"[STATS] Average quality score: {avg_quality:.2f}")
    logger.info(f"[STATS] Quality distribution:")
    quality_ranges = {
        "0.0-0.5": 0,
        "0.5-0.7": 0,
        "0.7-0.9": 0,
        "0.9-1.0": 0
    }
    for sample in samples:
        q = sample.get('quality_score', 0.0)
        if q < 0.5:
            quality_ranges["0.0-0.5"] += 1
        elif q < 0.7:
            quality_ranges["0.5-0.7"] += 1
        elif q < 0.9:
            quality_ranges["0.7-0.9"] += 1
        else:
            quality_ranges["0.9-1.0"] += 1
    
    for range_name, count in quality_ranges.items():
        logger.info(f"  {range_name}: {count} samples")


if __name__ == "__main__":
    main()







