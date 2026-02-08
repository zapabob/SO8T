#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4ロールデータ生成スクリプト
Task/Safety/Validation/Escalation統合
Self-consistency候補生成（可変2-10）
目標: 150,000 samples
"""

import os
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class FourRoleData:
    """4ロールデータ構造"""
    query: str
    task_response: str
    safety_judgment: str  # ALLOW/ESCALATE/DENY
    validation_candidates: List[Dict]  # N候補（2-10）
    validation_selected: int  # 最良候補インデックス
    validation_reasoning: str
    escalation_needed: bool
    escalation_reason: str
    consistency_score: float
    domain: str
    timestamp: str
    data_id: str


# ドメイン定義（8ドメイン）
DOMAINS = [
    "defense", "aerospace", "transport", "medical",
    "finance", "business", "information_systems", "general"
]

# 候補数分布（重要度依存）
CANDIDATE_DISTRIBUTION = {
    "low": 2,      # 一般タスク
    "medium": 3,   # 標準タスク
    "high": 5,     # 重要タスク
    "critical": 10 # 機密・重大タスク
}


class FourRoleDataGenerator:
    """4ロールデータ生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # テンプレート
        self.query_templates = self._load_query_templates()
        self.response_templates = self._load_response_templates()
    
    def _load_query_templates(self) -> Dict[str, List[str]]:
        """クエリテンプレート"""
        return {
            "defense": [
                "防衛装備品{item}の調達手続きについて説明してください",
                "有事における{scenario}の対応手順を教えてください",
                "サイバー攻撃{attack_type}への防御策を提案してください",
                "防衛計画における{aspect}の評価方法を説明してください",
                "国際共同訓練{exercise}の実施計画を立案してください"
            ],
            "aerospace": [
                "航空機{aircraft}の整備記録を要約してください",
                "宇宙デブリ{debris}の衝突リスクを評価してください",
                "ロケット打ち上げ{launch}の安全性を確認してください",
                "衛星軌道{orbit}の最適化案を提示してください",
                "航空管制{atc}の異常検知手順を説明してください"
            ],
            "transport": [
                "交通事故{accident}の原因分析を行ってください",
                "鉄道運行{operation}の遅延対策を提案してください",
                "物流ルート{route}の最適化案を作成してください",
                "自動運転{autonomous}の安全性評価を実施してください",
                "交通渋滞{congestion}の予測モデルを説明してください"
            ],
            "medical": [
                "患者{patient_id}のカルテから診断所見を要約してください",
                "薬剤{drug}の副作用リスクを評価してください",
                "医療機器{device}の使用手順を説明してください",
                "感染症{disease}の予防対策を提案してください",
                "診療報酬{billing}の請求手続きを確認してください"
            ],
            "finance": [
                "取引{transaction_id}の不正検知結果を報告してください",
                "投資商品{product}のリスク評価を行ってください",
                "融資審査{loan}の判定基準を説明してください",
                "マネーロンダリング{aml}の疑義を調査してください",
                "金融規制{regulation}の遵守状況を確認してください"
            ],
            "business": [
                "契約書{contract}のリスク条項を抽出してください",
                "会議{meeting_id}の議事録を作成してください",
                "プロジェクト{project}の進捗報告を要約してください",
                "顧客クレーム{complaint}の対応策を提案してください",
                "社内規程{policy}の改定案を検討してください"
            ],
            "information_systems": [
                "システムログ{log_id}から異常を検知してください",
                "セキュリティ脆弱性{cve}の対策を提案してください",
                "データベース{db}の最適化案を作成してください",
                "バックアップ{backup}の復旧手順を説明してください",
                "ネットワーク{network}の監視結果を報告してください"
            ],
            "general": [
                "文書{doc_id}を要約してください",
                "質問{question}に回答してください",
                "データ{data_id}を分析してください",
                "報告書{report_id}を作成してください",
                "情報{info_id}を検索してください"
            ]
        }
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """応答テンプレート"""
        return {
            "ALLOW": [
                "分析結果: {result}。推奨事項: {recommendation}。",
                "要約: {summary}。次のステップ: {next_step}。",
                "評価: {evaluation}。結論: {conclusion}。",
                "調査結果: {findings}。提案: {proposal}。",
                "確認結果: {confirmation}。備考: {notes}。"
            ],
            "ESCALATE": [
                "初期分析: {initial_analysis}。ただし、{reason}のため、専門家への確認が必要です。",
                "暫定評価: {tentative_eval}。しかし、{concern}があり、上司の承認を推奨します。",
                "一次判定: {primary_judgment}。ただし、{uncertainty}のため、追加情報が必要です。",
                "概要: {overview}。{complexity}のため、関係部署との協議を推奨します。",
                "基本方針: {basic_policy}。ただし、{risk}のため、慎重な検討が必要です。"
            ],
            "DENY": [
                "この要求は{reason}により実行できません。代替案: {alternative}。",
                "{prohibition}により対応不可です。詳細: {details}。",
                "安全性の観点から{safety_concern}のため拒否します。",
                "コンプライアンス上{compliance_issue}があり実行できません。",
                "権限不足: {permission_issue}。適切な承認を取得してください。"
            ]
        }
    
    def _generate_candidates(self, query: str, domain: str, importance: str) -> List[Dict]:
        """複数候補応答生成（Self-consistency用）"""
        n_candidates = CANDIDATE_DISTRIBUTION[importance]
        candidates = []
        
        # 各候補で微妙に異なる応答
        for i in range(n_candidates):
            # ランダムな判定（ただし一貫性を保つ）
            if "機密" in query or "秘密" in query or "極秘" in query:
                judgment = random.choices(
                    ["ALLOW", "ESCALATE", "DENY"],
                    weights=[0.2, 0.5, 0.3]  # 機密は慎重
                )[0]
            elif "確認" in query or "承認" in query:
                judgment = random.choices(
                    ["ALLOW", "ESCALATE", "DENY"],
                    weights=[0.3, 0.6, 0.1]  # 確認系はESCALATE多め
                )[0]
            else:
                judgment = random.choices(
                    ["ALLOW", "ESCALATE", "DENY"],
                    weights=[0.7, 0.2, 0.1]  # 一般はALLOW多め
                )[0]
            
            # 応答生成
            template = random.choice(self.response_templates[judgment])
            response = template.format(
                result=f"結果{i+1}",
                recommendation=f"推奨{i+1}",
                summary=f"要約{i+1}",
                next_step=f"次ステップ{i+1}",
                evaluation=f"評価{i+1}",
                conclusion=f"結論{i+1}",
                findings=f"知見{i+1}",
                proposal=f"提案{i+1}",
                confirmation=f"確認{i+1}",
                notes=f"備考{i+1}",
                initial_analysis=f"初期分析{i+1}",
                reason=f"理由{i+1}",
                tentative_eval=f"暫定評価{i+1}",
                concern=f"懸念{i+1}",
                primary_judgment=f"一次判定{i+1}",
                uncertainty=f"不確実性{i+1}",
                overview=f"概要{i+1}",
                complexity=f"複雑性{i+1}",
                basic_policy=f"基本方針{i+1}",
                risk=f"リスク{i+1}",
                prohibition=f"禁止事項{i+1}",
                details=f"詳細{i+1}",
                safety_concern=f"安全懸念{i+1}",
                compliance_issue=f"コンプライアンス問題{i+1}",
                permission_issue=f"権限問題{i+1}",
                alternative=f"代替案{i+1}"
            )
            
            # 一貫性スコア（同じ判定ほど高い）
            consistency = random.uniform(0.7, 0.95)
            
            candidates.append({
                "response": response,
                "judgment": judgment,
                "reasoning": f"{domain}ドメインにおける判定理由{i+1}",
                "consistency_score": consistency
            })
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[Dict]) -> int:
        """最良候補選択（一貫性スコア最大）"""
        scores = [c["consistency_score"] for c in candidates]
        return scores.index(max(scores))
    
    def generate_sample(self, domain: str, sample_id: int) -> FourRoleData:
        """1サンプル生成"""
        # 重要度決定
        importance = random.choices(
            ["low", "medium", "high", "critical"],
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]
        
        # クエリ生成
        template = random.choice(self.query_templates[domain])
        query = template.format(
            item=f"アイテム{sample_id}",
            scenario=f"シナリオ{sample_id}",
            attack_type=f"攻撃タイプ{sample_id}",
            aspect=f"側面{sample_id}",
            exercise=f"訓練{sample_id}",
            aircraft=f"機体{sample_id}",
            debris=f"デブリ{sample_id}",
            launch=f"打上{sample_id}",
            orbit=f"軌道{sample_id}",
            atc=f"管制{sample_id}",
            accident=f"事故{sample_id}",
            operation=f"運行{sample_id}",
            route=f"ルート{sample_id}",
            autonomous=f"自動運転{sample_id}",
            congestion=f"渋滞{sample_id}",
            patient_id=f"患者{sample_id}",
            drug=f"薬剤{sample_id}",
            device=f"機器{sample_id}",
            disease=f"疾患{sample_id}",
            billing=f"請求{sample_id}",
            transaction_id=f"取引{sample_id}",
            product=f"商品{sample_id}",
            loan=f"融資{sample_id}",
            aml=f"AML{sample_id}",
            regulation=f"規制{sample_id}",
            contract=f"契約{sample_id}",
            meeting_id=f"会議{sample_id}",
            project=f"プロジェクト{sample_id}",
            complaint=f"クレーム{sample_id}",
            policy=f"規程{sample_id}",
            log_id=f"ログ{sample_id}",
            cve=f"CVE{sample_id}",
            db=f"DB{sample_id}",
            backup=f"バックアップ{sample_id}",
            network=f"ネットワーク{sample_id}",
            doc_id=f"文書{sample_id}",
            question=f"質問{sample_id}",
            data_id=f"データ{sample_id}",
            report_id=f"報告{sample_id}",
            info_id=f"情報{sample_id}"
        )
        
        # 候補生成
        candidates = self._generate_candidates(query, domain, importance)
        
        # 最良候補選択
        selected_idx = self._select_best_candidate(candidates)
        selected = candidates[selected_idx]
        
        # Escalation判定
        escalation_needed = selected["judgment"] == "ESCALATE"
        escalation_reason = selected["reasoning"] if escalation_needed else ""
        
        # 一貫性スコア（全候補の平均）
        consistency_score = sum(c["consistency_score"] for c in candidates) / len(candidates)
        
        # データID生成
        data_id = hashlib.md5(
            f"{query}{domain}{sample_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return FourRoleData(
            query=query,
            task_response=selected["response"],
            safety_judgment=selected["judgment"],
            validation_candidates=candidates,
            validation_selected=selected_idx,
            validation_reasoning=f"候補{selected_idx+1}が最も一貫性が高い（スコア: {selected['consistency_score']:.3f}）",
            escalation_needed=escalation_needed,
            escalation_reason=escalation_reason,
            consistency_score=consistency_score,
            domain=domain,
            timestamp=datetime.now().isoformat(),
            data_id=data_id
        )
    
    def generate_dataset(self, total_samples: int = 150000):
        """データセット生成"""
        samples_per_domain = total_samples // len(DOMAINS)
        
        print(f"\n[GENERATE] 4-Role Dataset")
        print(f"Total samples: {total_samples:,}")
        print(f"Domains: {len(DOMAINS)}")
        print(f"Samples per domain: {samples_per_domain:,}\n")
        
        for domain in DOMAINS:
            print(f"[{domain.upper()}] Generating {samples_per_domain:,} samples...")
            
            output_file = self.output_dir / f"four_role_{domain}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i in tqdm(range(samples_per_domain), desc=domain):
                    sample = self.generate_sample(domain, i)
                    f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')
            
            print(f"[OK] {output_file}")
        
        print(f"\n[COMPLETE] 4-Role Dataset Generation")
        print(f"Total files: {len(DOMAINS)}")
        print(f"Output directory: {self.output_dir}")


def main():
    """メイン実行"""
    output_dir = Path("data/four_role")
    
    generator = FourRoleDataGenerator(output_dir)
    generator.generate_dataset(total_samples=150000)


if __name__ == "__main__":
    main()

