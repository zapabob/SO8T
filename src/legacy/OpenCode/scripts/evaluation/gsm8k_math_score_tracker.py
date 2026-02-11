#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K/MATHスコア追跡と評価スクリプト
エージェント化前後のスコアを比較し、スコア維持・向上を確認
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSM8KMathScoreTracker:
    """GSM8K/MATHスコア追跡クラス"""

    def __init__(self, results_dir: str = "results/gsm8k_math_tracking"):
        """
        初期化
        
        Args:
            results_dir: 結果保存ディレクトリ
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracking_file = self.results_dir / "score_tracking.json"
        self.baseline_file = self.results_dir / "baseline_scores.json"
        
        # ベースラインスコアを読み込み（存在する場合）
        self.baseline_scores = self._load_baseline_scores()
        
        # 許容範囲（±2%以内）
        self.tolerance = 0.02

    def _load_baseline_scores(self) -> Dict[str, float]:
        """ベースラインスコアを読み込み"""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_baseline_scores(self, gsm8k_score: float, math_score: float):
        """ベースラインスコアを保存（エージェント化前）"""
        baseline = {
            'gsm8k': gsm8k_score,
            'math': math_score,
            'timestamp': datetime.now().isoformat(),
            'description': 'Baseline scores before agentification'
        }
        
        with open(self.baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        
        self.baseline_scores = baseline
        logger.info(f"[BASELINE] Saved baseline scores: GSM8K={gsm8k_score:.2f}%, MATH={math_score:.2f}%")

    def track_scores(self, model_name: str, gsm8k_score: float, math_score: float, 
                    metadata: Optional[Dict] = None):
        """
        スコアを追跡
        
        Args:
            model_name: モデル名
            gsm8k_score: GSM8Kスコア（パーセント）
            math_score: MATHスコア（パーセント）
            metadata: 追加メタデータ
        """
        # 既存の追跡データを読み込み
        tracking_data = self._load_tracking_data()
        
        # 新しいエントリを追加
        entry = {
            'model_name': model_name,
            'gsm8k_score': gsm8k_score,
            'math_score': math_score,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # ベースラインとの比較
        if self.baseline_scores:
            entry['gsm8k_change'] = gsm8k_score - self.baseline_scores.get('gsm8k', 0)
            entry['math_change'] = math_score - self.baseline_scores.get('math', 0)
            entry['gsm8k_maintained'] = abs(entry['gsm8k_change']) <= self.tolerance * 100
            entry['math_maintained'] = abs(entry['math_change']) <= self.tolerance * 100
        
        tracking_data['entries'].append(entry)
        
        # 保存
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[TRACK] Tracked scores for {model_name}: GSM8K={gsm8k_score:.2f}%, MATH={math_score:.2f}%")
        
        # ベースラインとの比較結果を表示
        if self.baseline_scores:
            gsm8k_change = entry.get('gsm8k_change', 0)
            math_change = entry.get('math_change', 0)
            logger.info(f"[COMPARE] GSM8K change: {gsm8k_change:+.2f}% (maintained: {entry.get('gsm8k_maintained', False)})")
            logger.info(f"[COMPARE] MATH change: {math_change:+.2f}% (maintained: {entry.get('math_maintained', False)})")

    def _load_tracking_data(self) -> Dict[str, Any]:
        """追跡データを読み込み"""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'entries': []}

    def generate_report(self) -> Dict[str, Any]:
        """評価レポートを生成"""
        tracking_data = self._load_tracking_data()
        
        if not tracking_data['entries']:
            logger.warning("[REPORT] No tracking data available")
            return {}
        
        # 統計情報を計算
        gsm8k_scores = [e['gsm8k_score'] for e in tracking_data['entries']]
        math_scores = [e['math_score'] for e in tracking_data['entries']]
        
        report = {
            'baseline': self.baseline_scores,
            'current_scores': {
                'gsm8k': {
                    'mean': float(np.mean(gsm8k_scores)),
                    'std': float(np.std(gsm8k_scores)),
                    'min': float(np.min(gsm8k_scores)),
                    'max': float(np.max(gsm8k_scores))
                },
                'math': {
                    'mean': float(np.mean(math_scores)),
                    'std': float(np.std(math_scores)),
                    'min': float(np.min(math_scores)),
                    'max': float(np.max(math_scores))
                }
            },
            'comparison': {},
            'maintenance_status': {}
        }
        
        # ベースラインとの比較
        if self.baseline_scores:
            baseline_gsm8k = self.baseline_scores.get('gsm8k', 0)
            baseline_math = self.baseline_scores.get('math', 0)
            
            report['comparison'] = {
                'gsm8k': {
                    'baseline': baseline_gsm8k,
                    'current_mean': report['current_scores']['gsm8k']['mean'],
                    'change': report['current_scores']['gsm8k']['mean'] - baseline_gsm8k,
                    'change_percent': ((report['current_scores']['gsm8k']['mean'] - baseline_gsm8k) / baseline_gsm8k * 100) if baseline_gsm8k > 0 else 0
                },
                'math': {
                    'baseline': baseline_math,
                    'current_mean': report['current_scores']['math']['mean'],
                    'change': report['current_scores']['math']['mean'] - baseline_math,
                    'change_percent': ((report['current_scores']['math']['mean'] - baseline_math) / baseline_math * 100) if baseline_math > 0 else 0
                }
            }
            
            # 維持状況
            gsm8k_maintained = abs(report['comparison']['gsm8k']['change']) <= self.tolerance * 100
            math_maintained = abs(report['comparison']['math']['change']) <= self.tolerance * 100
            
            report['maintenance_status'] = {
                'gsm8k_maintained': gsm8k_maintained,
                'math_maintained': math_maintained,
                'both_maintained': gsm8k_maintained and math_maintained,
                'tolerance': self.tolerance * 100
            }
        
        # レポートを保存
        report_file = self.results_dir / "evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[REPORT] Generated evaluation report: {report_file}")
        
        return report


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GSM8K/MATH Score Tracker')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name')
    parser.add_argument('--gsm8k_score', type=float, required=True,
                       help='GSM8K score (percentage)')
    parser.add_argument('--math_score', type=float, required=True,
                       help='MATH score (percentage)')
    parser.add_argument('--baseline', action='store_true',
                       help='Set as baseline scores (before agentification)')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate evaluation report')
    
    args = parser.parse_args()
    
    tracker = GSM8KMathScoreTracker()
    
    if args.baseline:
        tracker.save_baseline_scores(args.gsm8k_score, args.math_score)
    else:
        tracker.track_scores(args.model_name, args.gsm8k_score, args.math_score)
    
    if args.generate_report:
        report = tracker.generate_report()
        logger.info("[REPORT] Evaluation report generated successfully")


if __name__ == "__main__":
    main()
