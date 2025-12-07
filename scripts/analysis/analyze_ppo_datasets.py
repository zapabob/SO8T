#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPOデータセット分析と報酬設計スクリプト
科学的一貫性、日本語流暢性、NSFW適切利用を報酬最大化
繰り返し、不明瞭返答、ハルシネーションに大きな負の報酬
"""

import os
import sys
from pathlib import Path
import json
import re
from typing import Dict, List, Any, Tuple
import logging
import numpy as np

# Windows cp932エンコーディング対策
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GRPOEvaluator:
    """GRPO報酬評価器"""

    def __init__(self):
        # 科学キーワード（一貫性評価用）
        self.science_keywords = {
            'high': ['theorem', 'proof', 'mathematical', 'quantum', 'relativity', 'neural network',
                    'algorithm', 'complexity', 'optimization', 'convergence', 'stability'],
            'medium': ['theory', 'method', 'analysis', 'computation', 'model', 'system'],
            'low': ['basic', 'simple', 'elementary', 'fundamental']
        }

        # 日本語流暢性指標
        self.japanese_fluency_indicators = [
            'です', 'ます', 'ました', 'でしょう', 'と思います', 'と考えます',
            'よろしいでしょうか', '承知しました', 'お手伝いいたします',
            'こと', 'もの', 'です', 'ます', 'が', 'に', 'を', 'は', 'へ', 'と'
        ]

        # NSFW/薬物関連キーワード
        self.nsfw_drug_keywords = [
            'porn', 'sex', 'nude', 'erotic', 'adult', 'xxx', 'nsfw',
            'cocaine', 'heroin', 'meth', 'drug', 'marijuana', 'lsd', 'ecstasy',
            'マリファナ', 'コカイン', 'ヘロイン', '覚醒剤', '麻薬', 'ドラッグ'
        ]

        # 安全応答パターン
        self.safety_responses = [
            '申し訳ありませんが、そのような内容にはお答えできません',
            'そのようなリクエストには対応しておりません',
            '安全上の理由により、お断りいたします',
            '不適切な内容のため、お答えできません',
            '倫理的観点から、お断りいたします'
        ]

        # 報酬ウェイト設定
        self.reward_weights = {
            'scientific_consistency': 3.0,     # 科学的一貫性
            'japanese_fluency': 2.5,          # 日本語流暢性
            'nsfw_appropriate': 4.0,          # NSFW適切利用
            'repetition_penalty': -8.0,       # 繰り返しペナルティ
            'unclear_penalty': -6.0,          # 不明瞭返答ペナルティ
            'hallucination_penalty': -10.0,   # ハルシネーションペナルティ
            'toxicity_penalty': -5.0,         # 毒性ペナルティ
            'length_bonus': 0.1               # 適切な長さボーナス
        }

    def evaluate_scientific_consistency(self, query: str, response: str) -> float:
        """科学的一貫性を評価"""
        score = 0.0

        query_lower = query.lower()
        response_lower = response.lower()

        # クエリに科学関連キーワードがある場合
        has_science_query = any(keyword in query_lower for keyword in
                               self.science_keywords['high'] + self.science_keywords['medium'])

        if has_science_query:
            score += 1.0

            # レスポンスに科学的概念が適切に使われているか
            science_mentions = sum(1 for keyword in self.science_keywords['high']
                                 if keyword in response_lower)
            score += min(science_mentions * 0.5, 2.0)

            # 数式や証明の言及
            if any(term in response_lower for term in ['proof', 'theorem', 'equation', '∴', '∵', '=']):
                score += 1.0

            # 論理的一貫性（因果関係の記述）
            if any(word in response for word in ['なぜなら', 'したがって', 'つまり', 'よって']):
                score += 0.5

        return min(score, 5.0)  # 最大5.0

    def evaluate_japanese_fluency(self, response: str) -> float:
        """日本語の流暢性を評価"""
        score = 0.0

        # 丁寧語の使用
        polite_indicators = sum(1 for indicator in self.japanese_fluency_indicators[:8]
                               if indicator in response)
        score += min(polite_indicators * 0.3, 1.5)

        # 文法的な流暢性（助詞の適切な使用）
        particle_score = 0
        particles = ['が', 'に', 'を', 'は', 'へ', 'と', 'で', 'から']
        for particle in particles:
            if particle in response:
                particle_score += 0.1
        score += min(particle_score, 1.0)

        # 漢字とひらがなのバランス
        kanji_ratio = len(re.findall(r'[\u4e00-\u9faf]', response)) / max(len(response), 1)
        if 0.05 <= kanji_ratio <= 0.35:
            score += 1.0
        elif kanji_ratio > 0.35:
            score += 0.5  # 漢字多めもまあまあ

        # 文の長さと構造
        sentences = re.split(r'[。！？]', response)
        if len(sentences) >= 2:
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
            if 20 <= avg_sentence_length <= 80:
                score += 0.5

        return min(score, 4.0)  # 最大4.0

    def evaluate_nsfw_appropriate(self, query: str, response: str) -> float:
        """NSFW/薬物データの適切な利用を評価"""
        score = 0.0

        query_lower = query.lower()
        response_lower = response.lower()

        # NSFW/薬物関連クエリかチェック
        is_sensitive_query = any(keyword in query_lower for keyword in self.nsfw_drug_keywords)

        if is_sensitive_query:
            # 適切な拒否応答をしているか
            has_safe_response = any(safe_phrase in response for safe_phrase in self.safety_responses)

            if has_safe_response:
                score += 3.0  # 適切な拒否で高得点

                # 説明的な拒否（理由を説明）
                if any(word in response_lower for word in ['安全', '倫理', '適切', '禁止', '配慮']):
                    score += 1.0

                # 代替案の提示
                if any(word in response for word in ['代わりに', '代わりに', '代わりの']):
                    score += 0.5
            else:
                # 不適切なコンテンツを提供した場合
                has_explicit_content = any(keyword in response_lower for keyword in self.nsfw_drug_keywords)
                if has_explicit_content:
                    return -5.0  # 大きな負の報酬
                else:
                    return -2.0  # 不適切なクエリに無視は軽い負の報酬
        else:
            # 非敏感クエリの場合は中立的
            score += 0.5

        return min(score, 4.0)

    def evaluate_repetition_penalty(self, response: str) -> float:
        """繰り返しペナルティを評価"""
        penalty = 0.0

        # 同じフレーズの繰り返し検知
        words = response.split()
        if len(words) > 10:
            # 連続する同じ単語の繰り返し
            for i in range(len(words) - 1):
                if words[i] == words[i+1] and len(words[i]) > 1:
                    penalty += 1.0

            # 同じ文の繰り返し
            sentences = re.split(r'[。！？]', response)
            unique_sentences = set(sentences)
            if len(sentences) > len(unique_sentences) * 1.5:
                penalty += 2.0

            # 同じフレーズの繰り返し（3語以上）
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if response.count(phrase) > 1:
                    penalty += 1.5

        return min(penalty, 5.0)  # 最大5.0のペナルティ

    def evaluate_unclear_penalty(self, response: str) -> float:
        """不明瞭な返答ペナルティを評価"""
        penalty = 0.0

        # 応答が短すぎる
        if len(response.strip()) < 20:
            penalty += 2.0

        # 意味不明な表現
        unclear_phrases = ['うーん', 'えっと', 'あの', 'まあ', '...', '・・・']
        unclear_count = sum(1 for phrase in unclear_phrases if phrase in response)
        penalty += unclear_count * 0.5

        # 矛盾した表現
        contradiction_words = ['でも', 'しかし', 'が', 'けど']
        contradiction_count = sum(1 for word in contradiction_words if response.count(word) > 3)
        penalty += contradiction_count * 0.3

        # 情報量の少なさ
        if len(response.split()) < 10:
            penalty += 1.0

        # 質問で終わっている（情報提供が不十分）
        if response.strip().endswith('?') or 'でしょうか' in response[-10:]:
            penalty += 1.0

        return min(penalty, 4.0)  # 最大4.0のペナルティ

    def evaluate_hallucination_penalty(self, query: str, response: str) -> float:
        """ハルシネーション（事実誤認）ペナルティを評価"""
        penalty = 0.0

        response_lower = response.lower()

        # 科学的誤情報の兆候
        hallucination_indicators = [
            '100%正しい', '絶対に', '間違いなく', '確実に', '常に',
            '全ての', 'すべての', '全て', '決して', '絶対'
        ]

        hallucination_count = sum(1 for indicator in hallucination_indicators
                                 if indicator in response)
        penalty += hallucination_count * 0.5

        # 根拠のない主張
        if any(word in response for word in ['実は', '実際', '本当に', '実は']):
            # 文脈から根拠があるかチェック（簡易版）
            evidence_words = ['研究', 'データ', '実験', '証明', '事実', 'によると']
            has_evidence = any(word in response for word in evidence_words)
            if not has_evidence:
                penalty += 1.0

        # 過度な一般化
        generalization_words = ['全ての', '常に', '絶対に', '決して', '100%']
        gen_count = sum(1 for word in generalization_words if word in response)
        penalty += gen_count * 0.3

        return min(penalty, 6.0)  # 最大6.0のペナルティ

    def calculate_grpo_reward(self, query: str, response: str) -> float:
        """GRPO用の総合報酬を計算"""
        reward = 0.0

        # 正の報酬
        scientific_score = self.evaluate_scientific_consistency(query, response)
        reward += scientific_score * self.reward_weights['scientific_consistency']

        japanese_score = self.evaluate_japanese_fluency(response)
        reward += japanese_score * self.reward_weights['japanese_fluency']

        nsfw_score = self.evaluate_nsfw_appropriate(query, response)
        reward += nsfw_score * self.reward_weights['nsfw_appropriate']

        # 負の報酬（ペナルティ）
        repetition_penalty = self.evaluate_repetition_penalty(response)
        reward += repetition_penalty * self.reward_weights['repetition_penalty']

        unclear_penalty = self.evaluate_unclear_penalty(response)
        reward += unclear_penalty * self.reward_weights['unclear_penalty']

        hallucination_penalty = self.evaluate_hallucination_penalty(query, response)
        reward += hallucination_penalty * self.reward_weights['hallucination_penalty']

        # 毒性チェック
        toxic_words = ['fuck', 'shit', 'damn', 'stupid', 'idiot', 'hate', 'kill', '死ね', '馬鹿']
        toxicity_count = sum(1 for word in toxic_words if word in response.lower())
        reward += toxicity_count * self.reward_weights['toxicity_penalty']

        # 長さボーナス（適切な長さの場合）
        response_length = len(response.split())
        if 50 <= response_length <= 200:
            reward += self.reward_weights['length_bonus'] * (response_length / 100)
        elif response_length > 300:
            reward -= 0.5  # 長すぎるペナルティ

        # 報酬の範囲制限
        reward = max(min(reward, 10.0), -15.0)

        return reward

def analyze_ppo_dataset_for_grpo(dataset_path: str, max_samples: int = 1000):
    """PPOデータセットをGRPO用に分析"""
    print(f"[ANALYZE] Analyzing PPO dataset: {dataset_path}")
    print("=" * 60)

    evaluator = GRPOEvaluator()
    reward_stats = []

    if not Path(dataset_path).exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            try:
                item = json.loads(line.strip())

                query = item.get('query', '')
                response = item.get('response', '')
                original_reward = item.get('reward', 0.0)

                # GRPO報酬計算
                grpo_reward = evaluator.calculate_grpo_reward(query, response)

                reward_stats.append({
                    'original_reward': original_reward,
                    'grpo_reward': grpo_reward,
                    'query_length': len(query),
                    'response_length': len(response)
                })

                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{max_samples} samples")

            except Exception as e:
                print(f"[WARNING] Error processing sample {i}: {e}")
                continue

    # 統計分析
    if reward_stats:
        original_rewards = [s['original_reward'] for s in reward_stats]
        grpo_rewards = [s['grpo_reward'] for s in reward_stats]

        print("\n[STATISTICS]")
        print(f"Samples analyzed: {len(reward_stats)}")
        print(f"Original Rewards - Mean: {np.mean(original_rewards):.6f}, Std: {np.std(original_rewards):.6f}")
        print(f"Original Rewards - Min: {min(original_rewards):.3f}, Max: {max(original_rewards):.3f}")
        print(f"GRPO Rewards - Mean: {np.mean(grpo_rewards):.6f}, Std: {np.std(grpo_rewards):.6f}")
        print(f"GRPO Rewards - Min: {min(grpo_rewards):.3f}, Max: {max(grpo_rewards):.3f}")

        # 報酬分布
        print("\n[REWARD DISTRIBUTION]")
        positive_grpo = sum(1 for r in grpo_rewards if r > 0)
        negative_grpo = sum(1 for r in grpo_rewards if r < 0)
        zero_grpo = sum(1 for r in grpo_rewards if r == 0)

        print(f"GRPO Positive rewards: {positive_grpo} ({positive_grpo/len(grpo_rewards)*100:.1f}%)")
        print(f"GRPO Negative rewards: {negative_grpo} ({negative_grpo/len(grpo_rewards)*100:.1f}%)")
        print(f"GRPO Zero rewards: {zero_grpo} ({zero_grpo/len(grpo_rewards)*100:.1f}%)")

    return reward_stats, evaluator

def create_grpo_dataset_with_rewards(dataset_path: str, output_path: str, max_samples: int = 50000):
    """GRPO用データセット作成（報酬付き）"""
    print(f"[CREATE] Creating GRPO dataset with rewards")
    print(f"Input: {dataset_path}")
    print(f"Output: {output_path}")
    print("=" * 60)

    evaluator = GRPOEvaluator()
    processed_samples = 0

    with open(dataset_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile):
            if processed_samples >= max_samples:
                break

            try:
                item = json.loads(line.strip())

                query = item.get('query', '')
                response = item.get('response', '')

                if not query or not response:
                    continue

                # GRPO報酬計算
                grpo_reward = evaluator.calculate_grpo_reward(query, response)

                # GRPO用データ形式
                grpo_item = {
                    'query': query,
                    'response': response,
                    'reward': grpo_reward,
                    'original_reward': item.get('reward', 0.0),
                    'metadata': {
                        'source_dataset': dataset_path,
                        'processed_at': str(Path.cwd()),
                        'grpo_reward_calculated': True
                    }
                }

                # 元のメタデータを保持
                if 'metadata' in item:
                    grpo_item['metadata'].update(item['metadata'])

                json.dump(grpo_item, outfile, ensure_ascii=False)
                outfile.write('\n')
                processed_samples += 1

                if processed_samples % 1000 == 0:
                    print(f"Processed {processed_samples}/{max_samples} samples")

            except Exception as e:
                print(f"[WARNING] Error processing line {line_num}: {e}")
                continue

    print(f"[SUCCESS] Created GRPO dataset: {processed_samples} samples")

    # 統計ファイル作成
    stats = {
        'total_samples': processed_samples,
        'source_dataset': dataset_path,
        'output_dataset': output_path,
        'reward_weights': evaluator.reward_weights
    }

    stats_file = output_path.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Statistics saved to: {stats_file}")

    return processed_samples, stats

def main():
    """メイン処理"""
    print("[START] GRPO Dataset Analysis and Reward Design")
    print("=" * 60)

    # PPOデータセットの分析
    ppo_dataset = 'data/enhanced_large_ppo_dataset.jsonl'
    reward_stats, evaluator = analyze_ppo_dataset_for_grpo(ppo_dataset, max_samples=1000)

    # GRPO用データセット作成
    output_dataset = 'data/aegis_v21_grpo_50k_with_rewards.jsonl'
    samples_created, stats = create_grpo_dataset_with_rewards(
        ppo_dataset, output_dataset, max_samples=50000
    )

    print("\n[FINAL RESULT]")
    print(f"GRPO Dataset: {output_dataset}")
    print(f"Samples: {samples_created}")
    print(f"Reward Design: Scientific consistency, Japanese fluency, NSFW appropriate (+)")
    print(f"Penalties: Repetition, unclear responses, hallucinations (-)")
    print(f"Stats file: {output_dataset.replace('.jsonl', '_stats.json')}")

    # 実装ログ作成
    create_grpo_reward_log(stats, evaluator.reward_weights)

def create_grpo_reward_log(stats: dict, reward_weights: dict):
    """GRPO報酬設計実装ログ作成"""
    log_content = f"""# GRPO報酬設計 実装ログ

## 実装情報
- **日付**: {Path.cwd().name} 実行時
- **機能名**: AEGIS v2.1 GRPO報酬設計（科学的一貫性、日本語流暢性、NSFW適切利用最大化）
- **実装者**: AI Agent

## 報酬設計概要

### 正の報酬（最大化対象）
1. **科学的一貫性** (重み: {reward_weights['scientific_consistency']})
   - 科学キーワードの適切な使用
   - 数式・証明の言及
   - 論理的一貫性

2. **日本語流暢性** (重み: {reward_weights['japanese_fluency']})
   - 丁寧語の適切な使用
   - 助詞のバランス
   - 漢字ひらがな比率の最適化

3. **NSFW/薬物適切利用** (重み: {reward_weights['nsfw_appropriate']})
   - 安全拒否応答の適切さ
   - 説明的な拒否理由
   - 倫理的配慮の表明

### 負の報酬（最小化対象）
1. **繰り返しペナルティ** (重み: {reward_weights['repetition_penalty']})
   - 同じフレーズの繰り返し
   - 文の重複
   - 意味のない反復

2. **不明瞭返答ペナルティ** (重み: {reward_weights['unclear_penalty']})
   - 短すぎる応答
   - 意味不明な表現
   - 情報量の不足

3. **ハルシネーションペナルティ** (重み: {reward_weights['hallucination_penalty']})
   - 根拠のない主張
   - 過度な一般化
   - 事実誤認の兆候

## 実装詳細

### 科学的一貫性評価関数
```
def evaluate_scientific_consistency(query, response):
    score = 0.0
    # 科学キーワード検知 + 適切な使用 + 論理的一貫性
    return min(score, 5.0)
```

### 日本語流暢性評価関数
```
def evaluate_japanese_fluency(response):
    score = 0.0
    # 丁寧語 + 助詞バランス + 漢字ひらがな比率
    return min(score, 4.0)
```

### NSFW適切利用評価関数
```
def evaluate_nsfw_appropriate(query, response):
    score = 0.0
    # 安全拒否応答 + 説明性 + 代替案提示
    return min(score, 4.0) or negative_penalty
```

### ペナルティ評価関数群
```
def evaluate_repetition_penalty(response):      # 最大5.0ペナルティ
def evaluate_unclear_penalty(response):         # 最大4.0ペナルティ
def evaluate_hallucination_penalty(query, response):  # 最大6.0ペナルティ
```

## データセット統計

- **処理サンプル数**: {stats['total_samples']}
- **ソースデータセット**: {stats['source_dataset']}
- **出力データセット**: {stats['output_dataset']}
- **報酬範囲**: -15.0 〜 +10.0（設計による）

## GRPOトレーニングへの影響

### Grokking現象誘導
- **科学的正確性**: 安定した学習基盤の形成
- **日本語流暢性**: 言語モデルの洗練
- **NSFW適切性**: 安全性の確保
- **ペナルティ最小化**: 学習の質的向上

### 汎化性能向上
- **一貫性重視**: ドメイン横断的な性能
- **流暢性最適化**: 自然な応答生成
- **ハルシネーション抑制**: 信頼性の高い出力
- **繰り返し回避**: 冗長性の低減

## 技術仕様

### 報酬計算アルゴリズム
```
総報酬 = Σ(各評価スコア × 重み) + ペナルティ
範囲: -15.0 ≤ 報酬 ≤ +10.0
```

### 評価関数仕様
- **科学的一貫性**: キーワード検知 + 文脈適合性
- **日本語流暢性**: 文法チェック + バランス評価
- **NSFW適切性**: 安全応答パターン + 倫理的考慮
- **繰り返し検知**: n-gram重複 + 文単位重複
- **不明瞭検知**: 長さ + 情報量 + 表現明確性
- **ハルシネーション検知**: 過度表現 + 根拠欠如 + 一般化過多

## AEGIS v2.1への貢献
- **GRPO報酬最適化**: 多角的評価による学習効率化
- **Grokking現象強化**: 報酬設計による突然学習誘導
- **汎化性能向上**: 科学・言語・安全の統合的最適化
- **品質保証**: ペナルティ最小化による出力品質確保
"""

    # ログファイル保存
    log_dir = Path("_docs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{Path.cwd().name}_main_grpo_reward_design.md"
    log_path = log_dir / log_filename

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)

    logger.info(f"[LOG] GRPO reward design log saved to: {log_path}")

if __name__ == "__main__":
    main()
