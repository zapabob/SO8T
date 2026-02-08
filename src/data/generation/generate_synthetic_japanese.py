#!/usr/bin/env python3
"""
日本語合成データ生成スクリプト
ドメイン特化合成データを生成（defense, aerospace, transport, general）
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ドメイン特化語彙データベース
DOMAIN_VOCABULARY = {
    'defense': {
        'nouns': ['防衛', '安全保障', '軍事', '戦略', '兵器', '装備', '訓練', '作戦', '偵察', '情報', '通信', 'サイバー', '宇宙', '航空', '海上', '陸上'],
        'topics': ['国防政策', '防衛装備', '安全保障環境', '自衛隊', '防衛力整備', '国際協力', 'PKO', '災害派遣'],
    },
    'aerospace': {
        'nouns': ['航空', '宇宙', 'ロケット', '衛星', '軌道', '推進', 'エンジン', '構造', '材料', 'システム', 'ミッション', '打ち上げ', '観測'],
        'topics': ['航空機設計', '宇宙開発', '衛星運用', 'ロケット技術', '航空管制', '航法システム', 'ISS', 'JAXA'],
    },
    'transport': {
        'nouns': ['交通', '運輸', '物流', '配送', 'インフラ', '鉄道', '道路', '港湾', '空港', 'システム', '管理', '運用', '安全'],
        'topics': ['交通システム', '物流最適化', 'インフラ管理', '運行管理', '安全対策', 'スマート交通', '自動運転', 'MaaS'],
    },
    'general': {
        'nouns': ['技術', 'システム', 'データ', '情報', '管理', '開発', '設計', '運用', '分析', '評価', '最適化', '効率化'],
        'topics': ['プロジェクト管理', 'システム開発', 'データ分析', '品質管理', 'リスク管理', 'イノベーション', 'DX', 'AI活用'],
    },
}


# テンプレートデータベース
TEMPLATES = {
    'explanation': [
        "{topic}について説明してください。",
        "{topic}とは何ですか？詳しく教えてください。",
        "{topic}の概要を教えてください。",
        "{topic}について詳しく解説してください。",
    ],
    'procedure': [
        "{topic}の手順を説明してください。",
        "{topic}を実施する際の注意点を教えてください。",
        "{topic}のプロセスを段階的に説明してください。",
    ],
    'comparison': [
        "{topic1}と{topic2}の違いを説明してください。",
        "{topic1}と{topic2}を比較してください。",
        "{topic1}と{topic2}、どちらが優れていますか？",
    ],
    'problem_solving': [
        "{topic}における課題と解決策を教えてください。",
        "{topic}で直面する問題とその対処法を説明してください。",
        "{topic}を改善するための方法を提案してください。",
    ],
}


# 三重推論訓練データテンプレート
TRIPLE_REASONING_TEMPLATES = {
    'ALLOW': [
        "一般的な{topic}について教えてください。",
        "{topic}の基本概念を説明してください。",
        "{topic}に関する公開情報を提供してください。",
    ],
    'ESCALATION': [
        "{topic}の詳細な仕様について教えてください。",
        "{topic}の具体的な運用方法を説明してください。",
        "{topic}に関する専門的な判断を求めます。",
    ],
    'DENY': [
        "{topic}の機密情報を教えてください。",
        "{topic}の内部構造の詳細を説明してください。",
        "{topic}の未公開データにアクセスしたいです。",
    ],
}


def generate_qa_pair(domain: str, template_type: str) -> Dict:
    """Q&Aペアを生成"""
    vocab = DOMAIN_VOCABULARY[domain]
    
    if template_type == 'comparison':
        template = random.choice(TEMPLATES[template_type])
        topic1 = random.choice(vocab['topics'])
        topic2 = random.choice([t for t in vocab['topics'] if t != topic1])
        question = template.format(topic1=topic1, topic2=topic2)
    else:
        template = random.choice(TEMPLATES[template_type])
        topic = random.choice(vocab['topics'])
        question = template.format(topic=topic)
    
    # 回答生成（テンプレートベース）
    answer = f"{question}に関する情報です。"
    
    return {
        'instruction': question,
        'input': '',
        'output': answer,
        'domain': domain,
        'template_type': template_type,
    }


def generate_triple_reasoning_sample(domain: str, judgment: str) -> Dict:
    """三重推論訓練サンプルを生成"""
    vocab = DOMAIN_VOCABULARY[domain]
    template = random.choice(TRIPLE_REASONING_TEMPLATES[judgment])
    topic = random.choice(vocab['topics'])
    question = template.format(topic=topic)
    
    # 判定理由
    if judgment == 'ALLOW':
        reason = "一般的な情報であり、公開情報の範囲内です。"
        response = f"{topic}について説明します。"
    elif judgment == 'ESCALATION':
        reason = "専門的な判断が必要な内容です。適切な担当者に確認が必要です。"
        response = "この質問には専門的な判断が必要です。担当者に確認します。"
    else:  # DENY
        reason = "機密情報に関わる内容であり、応答できません。"
        response = "申し訳ございませんが、その情報は提供できません。"
    
    return {
        'instruction': question,
        'input': '',
        'output': response,
        'domain': domain,
        'judgment': judgment,
        'reason': reason,
    }


def generate_synthetic_dataset(
    output_file: Path,
    total_samples: int = 5000,
    seed: int = 42,
):
    """
    合成データセットを生成
    
    Args:
        output_file: 出力ファイルパス
        total_samples: 総サンプル数
        seed: ランダムシード
    """
    random.seed(seed)
    
    logger.info(f"[START] Generating {total_samples} synthetic samples...")
    
    samples = []
    domains = list(DOMAIN_VOCABULARY.keys())
    template_types = list(TEMPLATES.keys())
    judgments = ['ALLOW', 'ESCALATION', 'DENY']
    
    # 50% Q&A, 50% 三重推論
    qa_count = total_samples // 2
    triple_count = total_samples - qa_count
    
    # Q&Aペア生成
    with tqdm(total=qa_count, desc="Generating Q&A pairs") as pbar:
        for _ in range(qa_count):
            domain = random.choice(domains)
            template_type = random.choice(template_types)
            sample = generate_qa_pair(domain, template_type)
            samples.append(sample)
            pbar.update(1)
    
    # 三重推論サンプル生成
    with tqdm(total=triple_count, desc="Generating triple reasoning") as pbar:
        # 判定を均等分配
        for judgment in judgments:
            count = triple_count // 3
            for _ in range(count):
                domain = random.choice(domains)
                sample = generate_triple_reasoning_sample(domain, judgment)
                samples.append(sample)
                pbar.update(1)
    
    # シャッフル
    random.shuffle(samples)
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"[SUCCESS] Generated {len(samples)} samples")
    logger.info(f"[SAVE] Saved to {output_file}")
    
    # 統計
    qa_samples = [s for s in samples if 'judgment' not in s]
    triple_samples = [s for s in samples if 'judgment' in s]
    
    logger.info(f"[STATS] Q&A pairs: {len(qa_samples)}")
    logger.info(f"[STATS] Triple reasoning: {len(triple_samples)}")
    
    if triple_samples:
        for judgment in judgments:
            count = len([s for s in triple_samples if s.get('judgment') == judgment])
            logger.info(f"[STATS] {judgment}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Japanese dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/phi4_japanese_synthetic.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=5000,
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Japanese Synthetic Dataset Generation")
    logger.info("=" * 70)
    logger.info(f"Output file: {args.output}")
    logger.info(f"Total samples: {args.total_samples}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 70)
    
    generate_synthetic_dataset(
        output_file=Path(args.output),
        total_samples=args.total_samples,
        seed=args.seed,
    )
    
    logger.info("[COMPLETE] Synthetic dataset generation completed!")


if __name__ == "__main__":
    main()

