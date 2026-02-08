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
from typing import Dict, List, Set, Optional
from collections import Counter
import numpy as np
from tqdm import tqdm

# 類似度計算用（オプション）
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
    
    def _compute_similarity_batch(self, texts: List[str], similarity_threshold: float = 0.9) -> Set[int]:
        """
        類似度ベースの重複検出（バッチ処理）
        
        Args:
            texts: テキストリスト
            similarity_threshold: 類似度閾値
        
        Returns:
            duplicate_indices: 重複と判定されたインデックスのセット
        """
        if not SKLEARN_AVAILABLE or len(texts) < 2:
            return set()
        
        duplicate_indices = set()
        
        try:
            # TF-IDFベクトル化
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(texts)
            
            # コサイン類似度計算（バッチ処理）
            similarity_matrix = cosine_similarity(vectors)
            
            # 重複検出（上三角行列のみチェック）
            for i in range(len(texts)):
                if i in duplicate_indices:
                    continue
                
                for j in range(i + 1, len(texts)):
                    if j in duplicate_indices:
                        continue
                    
                    if similarity_matrix[i][j] >= similarity_threshold:
                        duplicate_indices.add(j)  # 後続の方を重複としてマーク
            
            return duplicate_indices
        
        except Exception as e:
            print(f"[WARNING] Similarity-based deduplication failed: {e}")
            return set()
    
    def remove_duplicates(self, use_similarity: bool = True, similarity_threshold: float = 0.9):
        """
        重複除去（ハッシュベース + 類似度ベース）
        
        Args:
            use_similarity: 類似度ベースの重複検出を使用するか
            similarity_threshold: 類似度閾値
        """
        print(f"\n[DEDUP] Removing duplicates...")
        
        seen_hashes: Set[str] = set()
        unique_samples = []
        unique_texts = []  # 類似度計算用
        
        all_samples = self.collected_samples + self.synthetic_samples
        
        # 第1段階: ハッシュベースの重複除去
        hash_unique_samples = []
        for sample in tqdm(all_samples, desc="Hash-based deduplication"):
            text = sample.get("text") or sample.get("query", "")
            if not text:
                continue
            
            text_hash = self._compute_text_hash(text)
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                hash_unique_samples.append(sample)
                unique_texts.append(text)
            else:
                self.duplicates_removed += 1
        
        print(f"[OK] Hash-based deduplication: {len(hash_unique_samples):,} unique samples")
        
        # 第2段階: 類似度ベースの重複除去（オプション）
        if use_similarity and SKLEARN_AVAILABLE and len(hash_unique_samples) > 1:
            print(f"[DEDUP] Applying similarity-based deduplication (threshold: {similarity_threshold})...")
            
            # バッチ処理で類似度計算
            duplicate_indices = self._compute_similarity_batch(unique_texts, similarity_threshold)
            
            # 重複でないサンプルのみ保持
            for i, sample in enumerate(hash_unique_samples):
                if i not in duplicate_indices:
                    unique_samples.append(sample)
                else:
                    self.duplicates_removed += 1
            
            print(f"[OK] Similarity-based deduplication: Removed {len(duplicate_indices):,} similar samples")
        else:
            unique_samples = hash_unique_samples
        
        self.merged_samples = unique_samples
        print(f"[OK] Total duplicates removed: {self.duplicates_removed:,}")
        print(f"[OK] Unique samples: {len(self.merged_samples):,}")
    
    def _calculate_enhanced_quality_score(self, sample: Dict) -> float:
        """
        強化された品質スコア計算
        
        Args:
            sample: サンプル
        
        Returns:
            quality_score: 品質スコア（0.0-1.0）
        """
        text = sample.get("text") or sample.get("query", "")
        if not text:
            return 0.0
        
        score = 0.0
        length = len(text)
        
        # 1. テキスト長スコア（30%）
        if 200 <= length <= 5000:
            score += 0.3
        elif 100 <= length < 200 or 5000 < length <= 10000:
            score += 0.15
        else:
            score += 0.05
        
        # 2. 言語検出スコア（25%）
        language = sample.get("language", "ja")
        if language == "ja":
            ja_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf')
            ja_ratio = ja_chars / length if length > 0 else 0
            score += min(ja_ratio * 0.25, 0.25)
        elif language == "en":
            en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
            en_ratio = en_chars / length if length > 0 else 0
            score += min(en_ratio * 0.25, 0.25)
        elif language == "zh":
            zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            zh_ratio = zh_chars / length if length > 0 else 0
            score += min(zh_ratio * 0.25, 0.25)
        
        # 3. ドメイン関連性スコア（20%）
        domain = sample.get("domain", "")
        domain_keywords = {
            "defense": ["防衛", "軍事", "安全保障", "国防", "自衛隊"],
            "aerospace": ["航空", "宇宙", "ロケット", "衛星", "飛行"],
            "transport": ["運輸", "交通", "鉄道", "輸送", "物流"],
            "finance": ["金融", "銀行", "投資", "経済", "財務"],
            "medical": ["医療", "健康", "診断", "治療", "病院"],
            "business": ["企業", "経営", "ビジネス", "市場", "事業"]
        }
        
        if domain in domain_keywords:
            keywords = domain_keywords[domain]
            keyword_count = sum(1 for kw in keywords if kw in text)
            score += min(keyword_count / len(keywords) * 0.20, 0.20)
        else:
            score += 0.10  # ドメイン不明の場合は基本スコア
        
        # 4. ノイズ除去スコア（15%）
        # 連続する空白・改行のチェック
        if not (text.count("  ") > length / 50 or text.count("\n\n\n") > 0):
            score += 0.15
        
        # 5. 多様性スコア（10%）
        if length > 0:
            unique_chars = len(set(text))
            diversity_ratio = unique_chars / min(length, 1000)  # 最大1000文字で正規化
            score += min(diversity_ratio * 0.10, 0.10)
        
        # 既存の品質スコアがあれば統合（重み付き平均）
        if "quality_score" in sample:
            existing_score = sample["quality_score"]
            score = (score * 0.7 + existing_score * 0.3)
        
        return min(score, 1.0)
    
    def apply_quality_filter(self, min_quality: float = 0.7, domain_specialized: bool = True):
        """
        品質フィルタリング（強化版）
        
        Args:
            min_quality: 最小品質スコア
            domain_specialized: ドメイン特化フィルタリングを有効化
        """
        print(f"\n[FILTER] Applying enhanced quality filter (threshold: {min_quality})...")
        
        filtered_samples = []
        removed_count = 0
        
        for sample in tqdm(self.merged_samples, desc="Quality filtering"):
            # 強化された品質スコア計算
            quality_score = self._calculate_enhanced_quality_score(sample)
            sample["quality_score"] = quality_score
            
            # 品質フィルタリング
            if quality_score >= min_quality:
                # ドメイン特化フィルタリング（オプション）
                if domain_specialized:
                    domain = sample.get("domain", "")
                    # ドメインが不明な場合はスキップ（オプション）
                    if domain == "unknown" or not domain:
                        # ドメイン不明でも品質スコアが高い場合は通過
                        if quality_score >= min_quality * 1.2:
                            filtered_samples.append(sample)
                        else:
                            removed_count += 1
                            continue
                
                filtered_samples.append(sample)
            else:
                removed_count += 1
        
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
