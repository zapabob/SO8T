#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ品質検証スクリプト
- 重複除去、品質スコアフィルタリング
- ドメイン分布バランス確認
- 統計レポート生成（_docs/に保存）
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
from collections import Counter
import numpy as np
from tqdm import tqdm


class DataQualityValidator:
    """データ品質検証器"""
    
    def __init__(self, collected_dir: Path, synthetic_dir: Path, output_dir: Path):
        self.collected_dir = collected_dir
        self.synthetic_dir = synthetic_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.collected_samples = []
        self.synthetic_samples = []
        self.merged_samples = []
        self.duplicates_removed = 0
    
    def load_data(self):
        """データ読み込み"""
        print(f"\n[LOAD] Loading collected data from {self.collected_dir}...")
        for file in self.collected_dir.glob("*.jsonl"):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.collected_samples.append(json.loads(line))
        print(f"[OK] Loaded {len(self.collected_samples):,} collected samples")
        
        print(f"\n[LOAD] Loading synthetic data from {self.synthetic_dir}...")
        for file in self.synthetic_dir.glob("*.jsonl"):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.synthetic_samples.append(json.loads(line))
        print(f"[OK] Loaded {len(self.synthetic_samples):,} synthetic samples")
    
    def _compute_text_hash(self, text: str) -> str:
        """テキストハッシュ計算"""
        # 正規化：小文字化、空白削除
        normalized = text.lower().replace(" ", "").replace("\n", "")
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def remove_duplicates(self):
        """重複除去"""
        print(f"\n[DEDUP] Removing duplicates...")
        
        seen_hashes: Set[str] = set()
        unique_samples = []
        
        all_samples = self.collected_samples + self.synthetic_samples
        
        for sample in tqdm(all_samples, desc="Deduplication"):
            # テキスト取得
            text = sample.get("text") or sample.get("query", "")
            if not text:
                continue
            
            # ハッシュ計算
            text_hash = self._compute_text_hash(text)
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_samples.append(sample)
            else:
                self.duplicates_removed += 1
        
        self.merged_samples = unique_samples
        print(f"[OK] Removed {self.duplicates_removed:,} duplicates")
        print(f"[OK] Unique samples: {len(self.merged_samples):,}")
    
    def apply_quality_filter(self, min_quality: float = 0.7):
        """品質フィルタリング"""
        print(f"\n[FILTER] Applying quality filter (threshold: {min_quality})...")
        
        filtered_samples = []
        removed_count = 0
        
        for sample in tqdm(self.merged_samples, desc="Quality filtering"):
            # 収集データの品質スコア
            if "quality_score" in sample:
                if sample["quality_score"] >= min_quality:
                    filtered_samples.append(sample)
                else:
                    removed_count += 1
            else:
                # 合成データは全て通過
                filtered_samples.append(sample)
        
        self.merged_samples = filtered_samples
        print(f"[OK] Removed {removed_count:,} low-quality samples")
        print(f"[OK] Filtered samples: {len(self.merged_samples):,}")
    
    def check_domain_balance(self) -> Dict[str, int]:
        """ドメイン分布確認"""
        print(f"\n[BALANCE] Checking domain distribution...")
        
        domain_counts = Counter()
        for sample in self.merged_samples:
            domain = sample.get("domain", "unknown")
            domain_counts[domain] += 1
        
        total = sum(domain_counts.values())
        print(f"\nDomain distribution:")
        for domain, count in domain_counts.most_common():
            percentage = (count / total) * 100
            print(f"  {domain}: {count:,} ({percentage:.1f}%)")
        
        return dict(domain_counts)
    
    def compute_statistics(self) -> Dict:
        """統計情報計算"""
        print(f"\n[STATS] Computing statistics...")
        
        # 基本統計
        total_samples = len(self.merged_samples)
        collected_count = len(self.collected_samples)
        synthetic_count = len(self.synthetic_samples)
        
        # ドメイン分布
        domain_dist = self.check_domain_balance()
        
        # 決定分布（合成データのみ）
        decision_counts = Counter()
        for sample in self.merged_samples:
            if "decision" in sample:
                decision_counts[sample["decision"]] += 1
        
        # 品質スコア統計（収集データのみ）
        quality_scores = [s["quality_score"] for s in self.merged_samples if "quality_score" in s]
        
        stats = {
            "total_samples": total_samples,
            "collected_samples": collected_count,
            "synthetic_samples": synthetic_count,
            "duplicates_removed": self.duplicates_removed,
            "domain_distribution": domain_dist,
            "decision_distribution": dict(decision_counts),
            "quality_statistics": {
                "count": len(quality_scores),
                "mean": float(np.mean(quality_scores)) if quality_scores else 0.0,
                "std": float(np.std(quality_scores)) if quality_scores else 0.0,
                "min": float(np.min(quality_scores)) if quality_scores else 0.0,
                "max": float(np.max(quality_scores)) if quality_scores else 0.0
            },
            "validation_time": datetime.now().isoformat()
        }
        
        return stats
    
    def save_validated_data(self):
        """検証済みデータ保存"""
        print(f"\n[SAVE] Saving validated data...")
        
        # ドメイン別保存
        domain_groups = {}
        for sample in self.merged_samples:
            domain = sample.get("domain", "unknown")
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(sample)
        
        for domain, samples in domain_groups.items():
            output_file = self.output_dir / f"validated_{domain}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"[OK] Saved {len(samples):,} samples to {output_file}")
    
    def generate_report(self, stats: Dict):
        """統計レポート生成"""
        print(f"\n[REPORT] Generating quality report...")
        
        report_file = Path("_docs") / f"{datetime.now().strftime('%Y-%m-%d')}_data_quality_report.md"
        report_file.parent.mkdir(exist_ok=True)
        
        report = f"""# データ品質検証レポート

## 検証概要
- **検証日時**: {stats['validation_time']}
- **総サンプル数**: {stats['total_samples']:,}
- **収集サンプル**: {stats['collected_samples']:,}
- **合成サンプル**: {stats['synthetic_samples']:,}
- **重複除去**: {stats['duplicates_removed']:,}

## ドメイン分布
"""
        total = stats['total_samples']
        for domain, count in sorted(stats['domain_distribution'].items(), key=lambda x: -x[1]):
            percentage = (count / total) * 100
            report += f"- **{domain}**: {count:,} samples ({percentage:.1f}%)\n"
        
        if stats['decision_distribution']:
            report += "\n## 判定分布（合成データ）\n"
            decision_total = sum(stats['decision_distribution'].values())
            for decision, count in sorted(stats['decision_distribution'].items()):
                percentage = (count / decision_total) * 100
                report += f"- **{decision}**: {count:,} ({percentage:.1f}%)\n"
        
        if stats['quality_statistics']['count'] > 0:
            qs = stats['quality_statistics']
            report += f"""
## 品質スコア統計（収集データ）
- **平均**: {qs['mean']:.3f}
- **標準偏差**: {qs['std']:.3f}
- **最小値**: {qs['min']:.3f}
- **最大値**: {qs['max']:.3f}
- **サンプル数**: {qs['count']:,}
"""
        
        report += f"""
## 検証項目
- [OK] 重複除去完了
- [OK] 品質フィルタリング適用（閾値: 0.7）
- [OK] ドメイン分布確認
- [OK] 統計計算完了
- [OK] 検証済みデータ保存完了

## データ品質評価
"""
        # ドメインバランス評価
        domain_counts = list(stats['domain_distribution'].values())
        if domain_counts:
            max_count = max(domain_counts)
            min_count = min(domain_counts)
            balance_ratio = min_count / max_count if max_count > 0 else 0
            
            if balance_ratio > 0.8:
                balance_status = "良好"
            elif balance_ratio > 0.5:
                balance_status = "許容範囲"
            else:
                balance_status = "要改善"
            
            report += f"- **ドメインバランス**: {balance_status} (比率: {balance_ratio:.2f})\n"
        
        # 品質スコア評価
        if stats['quality_statistics']['count'] > 0:
            avg_quality = stats['quality_statistics']['mean']
            if avg_quality > 0.8:
                quality_status = "高品質"
            elif avg_quality > 0.7:
                quality_status = "良好"
            else:
                quality_status = "要改善"
            
            report += f"- **平均品質**: {quality_status} (スコア: {avg_quality:.3f})\n"
        
        # 重複率評価
        total_before_dedup = stats['collected_samples'] + stats['synthetic_samples']
        dup_rate = (stats['duplicates_removed'] / total_before_dedup) * 100 if total_before_dedup > 0 else 0
        report += f"- **重複率**: {dup_rate:.2f}%\n"
        
        report += """
## 次のステップ
- [READY] 学習データとして使用可能
- [READY] Phase 2: 学習レシピ詳細化に進む
- [READY] Phase 3: 学習実行準備完了
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Report saved to {report_file}")
        
        # JSON統計も保存
        stats_file = self.output_dir / "validation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[OK] Statistics saved to {stats_file}")
    
    def validate(self):
        """検証実行"""
        print(f"\n{'='*60}")
        print(f"[START] Data Quality Validation")
        print(f"{'='*60}\n")
        
        self.load_data()
        self.remove_duplicates()
        self.apply_quality_filter()
        stats = self.compute_statistics()
        self.save_validated_data()
        self.generate_report(stats)
        
        print(f"\n{'='*60}")
        print(f"[OK] Data quality validation completed!")
        print(f"Total validated samples: {stats['total_samples']:,}")
        print(f"{'='*60}\n")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Quality Validation")
    parser.add_argument("--collected", type=Path, default=Path("data/collected"),
                        help="Collected data directory")
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic"),
                        help="Synthetic data directory")
    parser.add_argument("--output", type=Path, default=Path("data/validated"),
                        help="Output directory")
    args = parser.parse_args()
    
    validator = DataQualityValidator(
        collected_dir=args.collected,
        synthetic_dir=args.synthetic,
        output_dir=args.output
    )
    
    try:
        validator.validate()
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()
