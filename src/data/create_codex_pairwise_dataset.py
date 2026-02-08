#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codex経由ペア比較データセット作成スクリプト

OpenAI APIまたはClaude APIをターミナル経由で呼び出し、
四重推論形式（Task/Safety/Policy/Final）でのペア比較データセットを生成
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
        logging.FileHandler('logs/create_codex_pairwise_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CodexPairwiseDatasetGenerator:
    """Codex（OpenAI/Claude API）を使用したペア比較データセット生成"""
    
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
        
        logger.info(f"[INIT] CodexPairwiseDatasetGenerator initialized with {api_type} API")
    
    def _call_openai_api(self, prompt: str, model: str = "gpt-4") -> str:
        """OpenAI APIを呼び出し"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert AI assistant that generates quadruple reasoning responses in Japanese."},
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
    
    def _build_quadruple_thinking_prompt(self, user_query: str, generate_chosen: bool = True) -> str:
        """四重推論形式のプロンプトを構築"""
        if generate_chosen:
            instruction = (
                "以下の問題に対して、四段階の内部推論を行い、その後<final>で日本語で回答してください。"
                "良い回答（chosen）を生成してください。\n"
                "1. <think-task>: タスク推論（英語で、ドメイン知識・翻訳方針・要約方針を考える）\n"
                "2. <think-safety>: 安全性推論（英語で、安全性・法令順守・NSFW違反を評価）\n"
                "3. <think-policy>: ポリシー推論（英語で、軍事・医療・インフラ等の領域別ポリシーに沿って、出せる/出せない情報範囲を決める）\n"
                "4. <final>: 最終回答（日本語で、制約を反映した最終回答のみ出力）\n"
                "内部推論はユーザーに公開されないことを前提に、正確に考えてください。\n"
                f"問題: {user_query}\n"
                "答え: <think-task>"
            )
        else:
            instruction = (
                "以下の問題に対して、四段階の内部推論を行い、その後<final>で日本語で回答してください。"
                "悪い回答（rejected）を生成してください（論理的な誤り、安全性の問題、ポリシー違反などを含む）。\n"
                "1. <think-task>: タスク推論（英語で、ドメイン知識・翻訳方針・要約方針を考える）\n"
                "2. <think-safety>: 安全性推論（英語で、安全性・法令順守・NSFW違反を評価）\n"
                "3. <think-policy>: ポリシー推論（英語で、軍事・医療・インフラ等の領域別ポリシーに沿って、出せる/出せない情報範囲を決める）\n"
                "4. <final>: 最終回答（日本語で、制約を反映した最終回答のみ出力）\n"
                "内部推論はユーザーに公開されないことを前提に、正確に考えてください。\n"
                f"問題: {user_query}\n"
                "答え: <think-task>"
            )
        
        return instruction
    
    def _extract_quadruple_thinking(self, text: str) -> Dict[str, str]:
        """四重推論を抽出"""
        result = {
            'task': '',
            'safety': '',
            'policy': '',
            'final': ''
        }
        
        # <think-task>...</think-task>を抽出
        import re
        task_match = re.search(r'<think-task>(.*?)</think-task>', text, re.DOTALL)
        if task_match:
            result['task'] = task_match.group(1).strip()
        
        # <think-safety>...</think-safety>を抽出
        safety_match = re.search(r'<think-safety>(.*?)</think-safety>', text, re.DOTALL)
        if safety_match:
            result['safety'] = safety_match.group(1).strip()
        
        # <think-policy>...</think-policy>を抽出
        policy_match = re.search(r'<think-policy>(.*?)</think-policy>', text, re.DOTALL)
        if policy_match:
            result['policy'] = policy_match.group(1).strip()
        
        # <final>...</final>を抽出
        final_match = re.search(r'<final>(.*?)</final>', text, re.DOTALL)
        if final_match:
            result['final'] = final_match.group(1).strip()
        
        return result
    
    def _classify_four_class(self, response: Dict[str, str]) -> str:
        """四値分類を実行（簡易版）"""
        final_text = response.get('final', '').lower()
        
        # 拒否キーワード
        refuse_keywords = ['できない', '拒否', '提供できません', '回答できません', '不適切']
        if any(kw in final_text for kw in refuse_keywords):
            return 'REFUSE'
        
        # エスカレーションキーワード
        escalation_keywords = ['確認', '検討', '専門家', '上級者', '管理者']
        if any(kw in final_text for kw in escalation_keywords):
            return 'ESCALATION'
        
        # 拒否（軽度）
        deny_keywords = ['推奨しません', '避けるべき', '注意が必要']
        if any(kw in final_text for kw in deny_keywords):
            return 'DENY'
        
        # デフォルトはALLOW
        return 'ALLOW'
    
    def _evaluate_quality(self, response: Dict[str, str]) -> float:
        """品質スコアを評価"""
        score = 0.0
        
        # 各推論ステップの存在チェック
        if response.get('task'):
            score += 0.25
        if response.get('safety'):
            score += 0.25
        if response.get('policy'):
            score += 0.25
        if response.get('final'):
            score += 0.25
        
        # 長さチェック（適切な長さ）
        total_length = sum(len(v) for v in response.values())
        if 100 <= total_length <= 2000:
            score += 0.1
        
        return min(score, 1.0)
    
    def generate_pairwise_samples(
        self,
        prompts: List[str],
        four_class_labels: Optional[List[str]] = None,
        num_pairs_per_prompt: int = 2
    ) -> List[Dict]:
        """
        ペア比較サンプルを生成
        
        Args:
            prompts: プロンプトリスト
            four_class_labels: 四値分類ラベルリスト（Noneの場合は自動分類）
            num_pairs_per_prompt: プロンプトあたりのペア数
        
        Returns:
            ペア比較サンプルのリスト
        """
        samples = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"[GENERATE] Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            for pair_idx in range(num_pairs_per_prompt):
                # Chosen回答を生成
                chosen_prompt = self._build_quadruple_thinking_prompt(prompt, generate_chosen=True)
                chosen_response_text = self._call_api(chosen_prompt)
                
                if not chosen_response_text:
                    logger.warning(f"[WARNING] Failed to generate chosen response for prompt {i+1}")
                    continue
                
                chosen_response = self._extract_quadruple_thinking(chosen_response_text)
                
                # Rejected回答を生成
                rejected_prompt = self._build_quadruple_thinking_prompt(prompt, generate_chosen=False)
                rejected_response_text = self._call_api(rejected_prompt)
                
                if not rejected_response_text:
                    logger.warning(f"[WARNING] Failed to generate rejected response for prompt {i+1}")
                    continue
                
                rejected_response = self._extract_quadruple_thinking(rejected_response_text)
                
                # 四値分類
                four_class_label = four_class_labels[i] if four_class_labels else self._classify_four_class(chosen_response)
                
                # 品質評価
                chosen_quality = self._evaluate_quality(chosen_response)
                rejected_quality = self._evaluate_quality(rejected_response)
                
                # サンプルを作成
                sample = {
                    "prompt": prompt,
                    "chosen": chosen_response_text,
                    "rejected": rejected_response_text,
                    "chosen_quadruple": chosen_response,
                    "rejected_quadruple": rejected_response,
                    "four_class_label": four_class_label,
                    "quality_score": chosen_quality,
                    "rejected_quality_score": rejected_quality,
                    "created_at": datetime.now().isoformat()
                }
                
                samples.append(sample)
                logger.info(f"[OK] Generated pair {pair_idx+1} for prompt {i+1} (quality: {chosen_quality:.2f})")
                
                # APIレート制限対策
                time.sleep(1)
        
        return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise comparison dataset using Codex (OpenAI/Claude API)"
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
        help="Path to prompts file (JSONL format, one prompt per line)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output file path (JSONL format)"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=2,
        help="Number of pairs per prompt (default: 2)"
    )
    parser.add_argument(
        "--four-class-labels-file",
        type=Path,
        default=None,
        help="Path to four-class labels file (JSONL format, optional)"
    )
    
    args = parser.parse_args()
    
    # プロンプトを読み込み
    prompts = []
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(line.strip())
    
    logger.info(f"[MAIN] Loaded {len(prompts)} prompts from {args.prompts_file}")
    
    # 四値分類ラベルを読み込み（オプション）
    four_class_labels = None
    if args.four_class_labels_file and args.four_class_labels_file.exists():
        four_class_labels = []
        with open(args.four_class_labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    four_class_labels.append(data.get('four_class_label', 'ALLOW'))
        logger.info(f"[MAIN] Loaded {len(four_class_labels)} four-class labels")
    
    # データセット生成器を初期化
    generator = CodexPairwiseDatasetGenerator(
        api_type=args.api_type,
        api_key=args.api_key
    )
    
    # ペア比較サンプルを生成
    logger.info(f"[MAIN] Generating {args.num_pairs} pairs per prompt...")
    samples = generator.generate_pairwise_samples(
        prompts=prompts,
        four_class_labels=four_class_labels,
        num_pairs_per_prompt=args.num_pairs
    )
    
    # 出力ファイルに保存
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"[SUCCESS] Generated {len(samples)} pairwise samples")
    logger.info(f"[SAVE] Saved to {output_file}")
    
    # 統計情報を出力
    four_class_counts = {}
    for sample in samples:
        label = sample.get('four_class_label', 'ALLOW')
        four_class_counts[label] = four_class_counts.get(label, 0) + 1
    
    logger.info(f"[STATS] Four-class label distribution: {four_class_counts}")
    avg_quality = sum(s.get('quality_score', 0.0) for s in samples) / len(samples) if samples else 0.0
    logger.info(f"[STATS] Average quality score: {avg_quality:.2f}")


if __name__ == "__main__":
    main()

