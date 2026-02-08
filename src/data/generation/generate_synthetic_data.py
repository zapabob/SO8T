"""
ドメイン特化合成データ生成スクリプト（本番環境対応版）

防衛・航空宇宙・運輸向けの高品質な合成学習データを生成する。
テンプレートベースおよびルールベースの生成により、
ドメイン語彙と専門知識を強化する。

生成対象:
1. 防衛用語・概念の説明文
2. 航空宇宙技術の解説
3. 運輸・物流のシナリオ
4. 安全性・倫理的判断の事例
5. 三重推論（ALLOW/ESCALATION/DENY）の訓練データ

本番環境要件:
- 大量生成対応（10万サンプル以上）
- バリエーション豊富なテンプレート
- 品質保証機構
- プログレス可視化

Author: SO8T Project Team
Date: 2024-11-06
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from tqdm import tqdm

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ========================================
# ドメイン語彙定義
# ========================================

DEFENSE_VOCABULARY = {
    'systems': ['防空システム', 'レーダーシステム', 'ミサイル防衛', 'サイバー防衛', '偵察システム'],
    'concepts': ['安全保障', '抑止力', '作戦計画', '情報収集', '脅威評価'],
    'equipment': ['装甲車両', '戦闘機', '艦艇', '通信機器', '監視装置'],
    'operations': ['作戦遂行', '防衛態勢', '警戒監視', '情報分析', '危機管理'],
}

AEROSPACE_VOCABULARY = {
    'systems': ['推進システム', '誘導制御', '姿勢制御', '通信システム', '熱制御'],
    'concepts': ['軌道力学', '空力設計', 'ロケット工学', '宇宙環境', '再突入'],
    'equipment': ['ロケットエンジン', 'ジャイロスコープ', '太陽電池', 'アンテナ', 'スラスタ'],
    'operations': ['打ち上げ', '軌道投入', 'ランデブー', 'ドッキング', 'デオービット'],
}

TRANSPORT_VOCABULARY = {
    'systems': ['配送管理システム', '在庫管理', '追跡システム', '最適化', 'ルート計画'],
    'concepts': ['サプライチェーン', 'ラストマイル', 'ハブ&スポーク', 'ジャストインタイム', 'モーダルシフト'],
    'equipment': ['輸送車両', 'コンテナ', 'パレット', 'RFID', 'GPS'],
    'operations': ['集荷', '仕分け', '配送', '在庫管理', 'ルート最適化'],
}

SAFETY_CONCEPTS = {
    'principles': ['安全第一', 'リスク管理', '予防原則', '透明性', 'アカウンタビリティ'],
    'threats': ['機密情報漏洩', '不正アクセス', 'システム障害', 'データ改ざん', '人的ミス'],
    'measures': ['アクセス制御', '暗号化', '監査ログ', 'バックアップ', 'インシデント対応'],
}


# ========================================
# テンプレート定義
# ========================================

EXPLANATION_TEMPLATES = [
    "{term}とは、{description}のことです。これは{domain}分野において{importance}です。",
    "{term}は{purpose}を目的とした{type}です。主な特徴として{features}があります。",
    "{domain}における{term}は、{function}する役割を持ちます。{application}に活用されています。",
    "{term}の概念は{origin}に由来します。現代では{modern_use}として発展しています。",
]

QA_TEMPLATES = [
    {
        'question': "{term}について説明してください。",
        'answer': "{term}は{description}です。{domain}分野では{application}に使用されています。"
    },
    {
        'question': "{term}の主な機能は何ですか？",
        'answer': "{term}の主な機能は{function}です。これにより{benefit}を実現します。"
    },
    {
        'question': "{term}はどのような場面で使用されますか？",
        'answer': "{term}は主に{scenario}で使用されます。特に{specific_case}において重要です。"
    },
]

SAFETY_JUDGMENT_TEMPLATES = [
    {
        'query': "{sensitive_info}に関する情報を教えてください。",
        'judgment': "DENY",
        'reason': "機密情報に該当するため開示できません。",
        'response': "申し訳ございませんが、{sensitive_info}は機密情報に該当するため、お答えできません。一般的な{general_topic}についてであればご説明できます。"
    },
    {
        'query': "{technical_question}の技術的詳細を教えてください。",
        'judgment': "ESCALATION",
        'reason': "専門的判断が必要な内容です。",
        'response': "ご質問の{technical_question}については、専門家の確認が必要です。担当者からご連絡させていただきます。"
    },
    {
        'query': "{general_question}について教えてください。",
        'judgment': "ALLOW",
        'reason': "一般的な情報であり開示可能です。",
        'response': "{general_question}についてご説明します。{explanation}"
    },
]


# ========================================
# 生成関数
# ========================================

def generate_explanation(domain: str, vocab_dict: Dict[str, List[str]], num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    説明文を生成
    
    Args:
        domain: ドメイン名
        vocab_dict: 語彙辞書
        num_samples: 生成数
        
    Returns:
        生成されたサンプルリスト
    """
    samples = []
    
    for _ in range(num_samples):
        template = random.choice(EXPLANATION_TEMPLATES)
        
        # ランダムに語彙を選択
        term = random.choice(vocab_dict.get('systems', []) + 
                           vocab_dict.get('concepts', []) + 
                           vocab_dict.get('equipment', []))
        
        # プレースホルダーを埋める
        text = template.format(
            term=term,
            description=f"{domain}における重要な{random.choice(['システム', '概念', '技術'])}",
            domain=domain,
            importance=random.choice(['不可欠です', '重要な役割を果たします', '基盤となっています']),
            purpose=random.choice(['効率化', '安全確保', '性能向上', '信頼性向上']),
            type=random.choice(['システム', '技術', '手法', 'プロセス']),
            features=random.choice(['高い信頼性', '優れた性能', '柔軟な運用性', '拡張性']),
            function=random.choice(['管理', '制御', '監視', '最適化', '分析']),
            application=random.choice(['運用管理', '品質保証', 'リスク管理', 'プロジェクト管理']),
            origin=random.choice(['軍事技術', '宇宙開発', '産業革命', '情報革命']),
            modern_use=random.choice(['民生技術', '商用システム', '公共インフラ', '研究開発']),
        )
        
        samples.append({
            'text': text,
            'domain': domain,
            'type': 'explanation',
            'term': term,
        })
    
    return samples


def generate_qa(domain: str, vocab_dict: Dict[str, List[str]], num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Q&Aペアを生成
    
    Args:
        domain: ドメイン名
        vocab_dict: 語彙辞書
        num_samples: 生成数
        
    Returns:
        生成されたサンプルリスト
    """
    samples = []
    
    for _ in range(num_samples):
        template = random.choice(QA_TEMPLATES)
        
        # ランダムに語彙を選択
        term = random.choice(vocab_dict.get('systems', []) + 
                           vocab_dict.get('concepts', []))
        
        # Q&Aを生成
        question = template['question'].format(term=term)
        answer = template['answer'].format(
            term=term,
            description=f"{domain}における{random.choice(['重要なシステム', '基本的な概念', '必須の技術'])}",
            domain=domain,
            application=random.choice(['運用', '管理', '開発', '保守', '改善']),
            function=random.choice(['効率化', '最適化', '自動化', '可視化', '分析']),
            benefit=random.choice(['コスト削減', '品質向上', 'リスク低減', '生産性向上']),
            scenario=random.choice(['緊急時', '通常運用時', '保守点検時', 'トラブル発生時']),
            specific_case=random.choice(['大規模運用', '長期運用', '複雑な状況', '制約条件下']),
        )
        
        # 会話形式に整形
        text = f"ユーザー: {question}\nアシスタント: {answer}"
        
        samples.append({
            'text': text,
            'domain': domain,
            'type': 'qa',
            'question': question,
            'answer': answer,
        })
    
    return samples


def generate_safety_judgment(num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    安全性判断の訓練データを生成
    
    Args:
        num_samples: 生成数
        
    Returns:
        生成されたサンプルリスト
    """
    samples = []
    
    sensitive_topics = [
        '特定の施設の詳細な配置',
        '暗号化アルゴリズムの実装詳細',
        '認証システムのバイパス方法',
        '機密指定された技術情報',
        '個人を特定できる位置情報',
    ]
    
    technical_topics = [
        '新型システムの性能評価',
        '導入予定技術の詳細仕様',
        '試験中のプロトコル',
        '開発段階の機能',
        '未公開の研究成果',
    ]
    
    general_topics = [
        '一般的なシステム構成',
        '公開されている技術標準',
        '基本的な運用手順',
        '教育訓練の概要',
        '公開文書に記載された情報',
    ]
    
    for _ in range(num_samples // 3):
        # DENY examples
        template = random.choice([t for t in SAFETY_JUDGMENT_TEMPLATES if t['judgment'] == 'DENY'])
        sensitive = random.choice(sensitive_topics)
        
        samples.append({
            'text': f"ユーザー: {template['query'].format(sensitive_info=sensitive)}\nアシスタント: {template['response'].format(sensitive_info=sensitive, general_topic='公開情報')}",
            'domain': 'safety',
            'type': 'judgment',
            'judgment': 'DENY',
            'reason': template['reason'],
        })
    
    for _ in range(num_samples // 3):
        # ESCALATION examples
        template = random.choice([t for t in SAFETY_JUDGMENT_TEMPLATES if t['judgment'] == 'ESCALATION'])
        technical = random.choice(technical_topics)
        
        samples.append({
            'text': f"ユーザー: {template['query'].format(technical_question=technical)}\nアシスタント: {template['response'].format(technical_question=technical)}",
            'domain': 'safety',
            'type': 'judgment',
            'judgment': 'ESCALATION',
            'reason': template['reason'],
        })
    
    for _ in range(num_samples // 3):
        # ALLOW examples
        template = random.choice([t for t in SAFETY_JUDGMENT_TEMPLATES if t['judgment'] == 'ALLOW'])
        general = random.choice(general_topics)
        
        samples.append({
            'text': f"ユーザー: {template['query'].format(general_question=general)}\nアシスタント: {template['response'].format(general_question=general, explanation='これは公開情報として提供可能な内容です。')}",
            'domain': 'safety',
            'type': 'judgment',
            'judgment': 'ALLOW',
            'reason': template['reason'],
        })
    
    return samples


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic domain-specific data")
    parser.add_argument("--output", type=str, default="data/synthetic_data.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to generate per domain")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # シード設定
    random.seed(args.seed)
    
    print("=" * 80)
    print("SO8T Synthetic Data Generation")
    print("=" * 80)
    print(f"\n[Config]")
    print(f"  Output: {args.output}")
    print(f"  Samples per domain: {args.num_samples}")
    print(f"  Random seed: {args.seed}")
    
    all_samples = []
    
    # 防衛ドメイン
    logger.info("[Defense] Generating samples...")
    defense_samples = []
    defense_samples.extend(generate_explanation('defense', DEFENSE_VOCABULARY, args.num_samples // 2))
    defense_samples.extend(generate_qa('defense', DEFENSE_VOCABULARY, args.num_samples // 2))
    all_samples.extend(defense_samples)
    logger.info(f"[Defense] Generated {len(defense_samples)} samples")
    
    # 航空宇宙ドメイン
    logger.info("[Aerospace] Generating samples...")
    aerospace_samples = []
    aerospace_samples.extend(generate_explanation('aerospace', AEROSPACE_VOCABULARY, args.num_samples // 2))
    aerospace_samples.extend(generate_qa('aerospace', AEROSPACE_VOCABULARY, args.num_samples // 2))
    all_samples.extend(aerospace_samples)
    logger.info(f"[Aerospace] Generated {len(aerospace_samples)} samples")
    
    # 運輸ドメイン
    logger.info("[Transport] Generating samples...")
    transport_samples = []
    transport_samples.extend(generate_explanation('transport', TRANSPORT_VOCABULARY, args.num_samples // 2))
    transport_samples.extend(generate_qa('transport', TRANSPORT_VOCABULARY, args.num_samples // 2))
    all_samples.extend(transport_samples)
    logger.info(f"[Transport] Generated {len(transport_samples)} samples")
    
    # 安全性判断
    logger.info("[Safety] Generating judgment samples...")
    safety_samples = generate_safety_judgment(args.num_samples)
    all_samples.extend(safety_samples)
    logger.info(f"[Safety] Generated {len(safety_samples)} samples")
    
    # シャッフル
    random.shuffle(all_samples)
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Save] Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(all_samples, desc="Writing"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 統計情報
    stats = {
        'total_samples': len(all_samples),
        'defense': len(defense_samples),
        'aerospace': len(aerospace_samples),
        'transport': len(transport_samples),
        'safety': len(safety_samples),
        'types': {},
        'judgments': {},
    }
    
    for sample in all_samples:
        stats['types'][sample['type']] = stats['types'].get(sample['type'], 0) + 1
        if 'judgment' in sample:
            stats['judgments'][sample['judgment']] = stats['judgments'].get(sample['judgment'], 0) + 1
    
    print("\n" + "=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    print(f"  Total samples: {stats['total_samples']}")
    print(f"\n  By domain:")
    print(f"    Defense: {stats['defense']}")
    print(f"    Aerospace: {stats['aerospace']}")
    print(f"    Transport: {stats['transport']}")
    print(f"    Safety: {stats['safety']}")
    print(f"\n  By type:")
    for type_name, count in stats['types'].items():
        print(f"    {type_name}: {count}")
    print(f"\n  By judgment:")
    for judgment, count in stats['judgments'].items():
        print(f"    {judgment}: {count}")
    
    # 統計をJSON保存
    stats_path = output_path.parent / "synthetic_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Stats saved to: {stats_path}")
    print("\n" + "=" * 80)
    print("[Generation] Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

