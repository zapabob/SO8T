#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ツール中毒・幻覚評価スクリプト
ツール使用適切率、不要なツール呼び出し率、ツール幻覚率を評価
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolAddictionEvaluator:
    """ツール中毒・幻覚評価クラス"""

    def __init__(self, results_dir: str = "results/tool_addiction_evaluation"):
        """
        初期化
        
        Args:
            results_dir: 結果保存ディレクトリ
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 目標値
        self.target_tool_addiction_rate = 0.05  # <5%
        self.target_tool_hallucination_rate = 0.01  # <1%

    def evaluate_tool_usage(self, responses: List[Dict[str, Any]], 
                           ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        ツール使用を評価
        
        Args:
            responses: モデルの応答リスト（各応答にツール呼び出し情報を含む）
            ground_truth: 正解データ（ツールが必要かどうかの情報を含む）
        
        Returns:
            評価結果の辞書
        """
        total_responses = len(responses)
        if total_responses == 0:
            logger.warning("[EVAL] No responses to evaluate")
            return {}
        
        # 統計情報
        tool_calls = 0
        unnecessary_tool_calls = 0
        hallucinated_tool_calls = 0
        appropriate_tool_calls = 0
        tool_free_responses = 0
        
        # 既知のツールリスト（実際の実装では、利用可能なツールリストから取得）
        known_tools = [
            'read_file', 'write_file', 'list_dir', 'web_search', 'analyze_data',
            'generate_chart', 'send_email', 'create_document', 'run_command'
        ]
        
        for i, response in enumerate(responses):
            # ツール呼び出しを検出
            tool_calls_in_response = self._detect_tool_calls(response)
            tool_calls += len(tool_calls_in_response)
            
            # ツール不要な問題かどうかを判定
            is_tool_free_task = self._is_tool_free_task(response, ground_truth[i] if ground_truth and i < len(ground_truth) else None)
            
            if is_tool_free_task:
                tool_free_responses += 1
                # ツール不要な問題でツールを呼んだ場合、不要なツール呼び出し
                if tool_calls_in_response:
                    unnecessary_tool_calls += len(tool_calls_in_response)
            else:
                # ツールが必要な問題の場合
                if tool_calls_in_response:
                    # 適切なツール呼び出しかどうかを判定
                    if self._is_appropriate_tool_usage(response, ground_truth[i] if ground_truth and i < len(ground_truth) else None):
                        appropriate_tool_calls += len(tool_calls_in_response)
                    else:
                        unnecessary_tool_calls += len(tool_calls_in_response)
            
            # ツール幻覚の検出（存在しないツールの呼び出し）
            for tool_call in tool_calls_in_response:
                tool_name = tool_call.get('tool_name', '')
                if tool_name and tool_name not in known_tools:
                    hallucinated_tool_calls += 1
        
        # メトリクス計算
        tool_addiction_rate = unnecessary_tool_calls / total_responses if total_responses > 0 else 0
        tool_hallucination_rate = hallucinated_tool_calls / tool_calls if tool_calls > 0 else 0
        appropriate_tool_usage_rate = appropriate_tool_calls / tool_calls if tool_calls > 0 else 0
        
        results = {
            'total_responses': total_responses,
            'total_tool_calls': tool_calls,
            'unnecessary_tool_calls': unnecessary_tool_calls,
            'hallucinated_tool_calls': hallucinated_tool_calls,
            'appropriate_tool_calls': appropriate_tool_calls,
            'tool_free_responses': tool_free_responses,
            'metrics': {
                'tool_addiction_rate': float(tool_addiction_rate),
                'tool_hallucination_rate': float(tool_hallucination_rate),
                'appropriate_tool_usage_rate': float(appropriate_tool_usage_rate)
            },
            'targets': {
                'tool_addiction_rate_target': self.target_tool_addiction_rate,
                'tool_hallucination_rate_target': self.target_tool_hallucination_rate
            },
            'status': {
                'tool_addiction_acceptable': tool_addiction_rate < self.target_tool_addiction_rate,
                'tool_hallucination_acceptable': tool_hallucination_rate < self.target_tool_hallucination_rate
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 結果を保存
        results_file = self.results_dir / "tool_addiction_evaluation.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[EVAL] Tool addiction rate: {tool_addiction_rate:.2%} (target: <{self.target_tool_addiction_rate:.2%})")
        logger.info(f"[EVAL] Tool hallucination rate: {tool_hallucination_rate:.2%} (target: <{self.target_tool_hallucination_rate:.2%})")
        logger.info(f"[EVAL] Appropriate tool usage rate: {appropriate_tool_usage_rate:.2%}")
        
        return results

    def _detect_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """応答からツール呼び出しを検出"""
        tool_calls = []
        
        # 応答テキストからツール呼び出しを検出
        text = response.get('text', response.get('output', ''))
        
        # パターン1: 関数呼び出し形式 (tool_name(args))
        pattern1 = r'(\w+)\s*\([^)]*\)'
        matches1 = re.findall(pattern1, text)
        
        # パターン2: JSON形式のツール呼び出し
        pattern2 = r'"tool_name"\s*:\s*"(\w+)"'
        matches2 = re.findall(pattern2, text)
        
        # パターン3: ツールリストから
        if 'tools' in response:
            for tool in response['tools']:
                if isinstance(tool, str):
                    tool_calls.append({'tool_name': tool, 'source': 'tools_field'})
                elif isinstance(tool, dict):
                    tool_calls.append({'tool_name': tool.get('name', ''), 'source': 'tools_field', **tool})
        
        # マッチしたツール名を追加
        for tool_name in matches1 + matches2:
            if tool_name not in [tc.get('tool_name', '') for tc in tool_calls]:
                tool_calls.append({'tool_name': tool_name, 'source': 'text_pattern'})
        
        return tool_calls

    def _is_tool_free_task(self, response: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None) -> bool:
        """ツール不要なタスクかどうかを判定"""
        # ground_truthから判定
        if ground_truth:
            tool_condition = ground_truth.get('tool_condition', '')
            if tool_condition == 'no_tool':
                return True
        
        # 応答から判定
        instruction = response.get('instruction', '')
        input_text = response.get('input', '')
        
        # 簡単な問題のキーワード
        simple_keywords = ['2+2', '簡単な計算', '基本的な', '常識的な', 'ツール不要']
        if any(keyword in instruction.lower() or keyword in input_text.lower() for keyword in simple_keywords):
            return True
        
        return False

    def _is_appropriate_tool_usage(self, response: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None) -> bool:
        """適切なツール使用かどうかを判定"""
        # ground_truthから判定
        if ground_truth:
            tool_condition = ground_truth.get('tool_condition', '')
            if tool_condition == 'required':
                return True
            elif tool_condition == 'forbidden':
                return False
        
        # 応答から判定（簡易版）
        instruction = response.get('instruction', '')
        input_text = response.get('input', '')
        
        # ツールが必要な問題のキーワード
        tool_required_keywords = ['ファイル', 'データ分析', 'グラフ', '検索', '計算', '1000行', '大規模']
        if any(keyword in instruction.lower() or keyword in input_text.lower() for keyword in tool_required_keywords):
            return True
        
        return False


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tool Addiction Evaluator')
    parser.add_argument('--responses_file', type=str, required=True,
                       help='Path to responses JSON file')
    parser.add_argument('--ground_truth_file', type=str, default=None,
                       help='Path to ground truth JSON file (optional)')
    
    args = parser.parse_args()
    
    # 応答データを読み込み
    with open(args.responses_file, 'r', encoding='utf-8') as f:
        responses = json.load(f)
    
    # 正解データを読み込み（存在する場合）
    ground_truth = None
    if args.ground_truth_file and Path(args.ground_truth_file).exists():
        with open(args.ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    
    # 評価実行
    evaluator = ToolAddictionEvaluator()
    results = evaluator.evaluate_tool_usage(responses, ground_truth)
    
    logger.info("[EVAL] Tool addiction evaluation completed")


if __name__ == "__main__":
    main()
