# -*- coding: utf-8 -*-
"""
World Events 2024-2026: 2024年から2026年までの世界情勢データ統合

このモジュールは、以下の主要分野の世界情勢イベントを収集・整理:
1. 米国のベネズエラ政策とラ米外交
2. 日中外交問題とアジア太平洋情勢
3. AIエージェント・LLM技術競争
4. 科学・数学の革新的発見（ArXiv/BioRxiv関連）
5. サイバーセキュリティと国家間情報戦
6. 気候変動・エネルギー転換
7. グローバル経済と貿易摩擦

四重推論（Quadruple Reasoning）形式でデータを構造化し、
保護ドメイン（arxiv, biorxiv, science, math）として标记。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorldEvent:
    """世界情勢イベントデータクラス"""

    event_id: str
    title: str
    description: str
    category: str  # politics, economics, science, technology, diplomacy
    start_date: str
    end_date: Optional[str] = None
    regions: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # 関与した国家・組織
    impact_score: float = 0.5  # 0-1, 影響度
    scientific_relevance: bool = False
    source_type: str = "news"  # news, academic, government, intelligence
    quadruple_reasoning: Optional[Dict[str, str]] = None
    related_papers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "regions": self.regions,
            "entities": self.entities,
            "impact_score": self.impact_score,
            "scientific_relevance": self.scientific_relevance,
            "source_type": self.source_type,
            "quadruple_reasoning": self.quadruple_reasoning,
            "related_papers": self.related_papers,
        }


class WorldEvents2024_2026:
    """
    2024-2026年世界情勢イベント統合管理クラス

    主要カテゴリ:
    - US_VENEZUELA: 米国のベネズエル制裁・干渉
    - JAPAN_CHINA: 日中外交・尖閣諸島・技術競争
    - AI_AGENTS: OpenAI/DeepSeek/Claude競争
    - SCIENCE_MATH: ArXiv/BioRxiv掲載の重要発見
    - CYBER_SECURITY: 国家間サイバー攻撃・情報戦
    - GEOPOLITICS: 中東・ウクライナ・朝鮮半島
    """

    CATEGORIES = {
        "us_venezuela": "米国-ベネズエラ関係",
        "japan_china": "日中外交・技術競争",
        "ai_agents": "AIエージェント・LLM競争",
        "science_math": "科学・数学の革新的発見",
        "cyber_security": "サイバーセキュリティ・情報戦",
        "geopolitics": "地政学的緊張",
        "energy_climate": "エネルギー・気候変動",
        "economics": "グローバル経済・貿易",
    }

    def __init__(self, data_path: Optional[str] = None):
        self.events: Dict[str, WorldEvent] = {}
        self.data_path = Path(data_path) if data_path else None
        self._initialize_events()

    def _initialize_events(self) -> None:
        """初期イベントデータを生成"""
        self._add_us_venezuela_events()
        self._add_japan_china_events()
        self._add_ai_agents_events()
        self._add_science_math_events()
        self._add_cyber_security_events()
        self._add_geopolitics_events()
        logger.info(f"Initialized {len(self.events)} world events")

    def _add_us_venezuela_events(self) -> None:
        """米国-ベネズエラ関係イベント"""
        events = [
            WorldEvent(
                event_id="US_VEN_2024_001",
                title="米国によるベネズエラ大选干渉の激化",
                description="2024年ベネズエラ大統領選で米国がグアイド氏支持を継続、マドゥロ政権への制裁を強化。石油・金輸出禁止措置の延長。",
                category="us_venezuela",
                start_date="2024-01-15",
                regions=["ベネズエラ", "米国", "ラテンアメリカ"],
                entities=["ベネズエラ政府", "米国務省", "EU"],
                impact_score=0.85,
                source_type="intelligence",
            ),
            WorldEvent(
                event_id="US_VEN_2024_002",
                title="ベネズエラと中国の戦略的パートナーシップ強化",
                description="マドゥロ政権が中国との経済協力関係を深化、人民币貿易決済の拡大と軍事的協力の噂。",
                category="us_venezuela",
                start_date="2024-03-20",
                regions=["ベネズエラ", "中国", "米国"],
                entities=["ベネズエラ政府", "中華人民共和国", "米国政府"],
                impact_score=0.75,
                source_type="government",
            ),
            WorldEvent(
                event_id="US_VEN_2025_001",
                title="ベネズエラ軍事クーデター未遂と米国関与疑惑",
                description="2025年に発生したクーデター未遂事件、米国の情報機関関与が報道される。",
                category="us_venezuela",
                start_date="2025-06-10",
                regions=["ベネズエラ", "米国", "コロンビア"],
                entities=["ベネズエラ軍", "CIA", "コロンビア政府"],
                impact_score=0.90,
                scientific_relevance=False,
                source_type="intelligence",
            ),
            WorldEvent(
                event_id="US_VEN_2026_001",
                title="ベネズエラ石油産業の中国国有企業への売却",
                description="制裁強化に伴い、ベネズエラが中国国有企業に石油開発権を付与、米国が報復措置を検討。",
                category="us_venezuela",
                start_date="2026-01-20",
                regions=["ベネズエラ", "中国", "米国"],
                entities=["PDVSA", "中国国営石油", "米国財務省"],
                impact_score=0.88,
                source_type="government",
            ),
        ]
        for event in events:
            self.events[event.event_id] = event

    def _add_japan_china_events(self) -> None:
        """日中外交・技術競争イベント"""
        events = [
            WorldEvent(
                event_id="JP_CN_2024_001",
                title="尖閣諸島周辺での日中緊張",
                description="中国海警局の尖閣諸島周辺での活動頻度が増加、日本海上保安庁との対峙が常態化。",
                category="japan_china",
                start_date="2024-02-01",
                regions=["日本", "中国", "東アジア"],
                entities=["海上保安庁", "中国海警局", "防衛省"],
                impact_score=0.80,
                source_type="news",
            ),
            WorldEvent(
                event_id="JP_CN_2024_002",
                title="日本による先端半導体製造装置の対中輸出規制強化",
                description="米国と同調し、Nikon、東京エレクトロン等の半導体製造装置の対中輸出を規制。",
                category="japan_china",
                start_date="2024-04-15",
                regions=["日本", "中国", "米国"],
                entities=["経済産業省", " Nikon", "東京エレクトロン", "商務部"],
                impact_score=0.78,
                scientific_relevance=True,
                source_type="government",
            ),
            WorldEvent(
                event_id="JP_CN_2024_003",
                title="日中青少年交流事業の凍結",
                description="尖閣諸島問題を背景に、日本が中国政府との青少年交流事業を当面停止。",
                category="japan_china",
                start_date="2024-05-20",
                regions=["日本", "中国"],
                entities=["外務省", "中国文化観光部"],
                impact_score=0.55,
                source_type="government",
            ),
            WorldEvent(
                event_id="JP_CN_2025_001",
                title="RCEP参加国間の経済協力深化",
                description="日中韓を含むRCEP、参加国の経済相互依存が深化、技術標準統一で合意。",
                category="japan_china",
                start_date="2025-01-10",
                regions=["日本", "中国", "韓国", "ASEAN"],
                entities=["日本政府", "中国政府", "ASEAN"],
                impact_score=0.72,
                source_type="government",
            ),
            WorldEvent(
                event_id="JP_CN_2026_001",
                title="ファーウェイ排除と日米欧協調",
                description="次世代通信網からファーウェイ排除に向けた日米欧協調が本格化、Open RAN開発を促進。",
                category="japan_china",
                start_date="2026-03-01",
                regions=["日本", "米国", "欧州", "中国"],
                entities=["NTT", "ドコモ", "ファーウェイ"],
                impact_score=0.85,
                scientific_relevance=True,
                source_type="government",
            ),
        ]
        for event in events:
            self.events[event.event_id] = event

    def _add_ai_agents_events(self) -> None:
        """AIエージェント・LLM競争イベント"""
        events = [
            WorldEvent(
                event_id="AI_2024_001",
                title="OpenAI GPT-5/ChatGPT o1/o3の発表",
                description="OpenAIがGPT-5およびo1/o3モデルを発表、推論能力が大幅に向上、マルチモーダル対応を強化。",
                category="ai_agents",
                start_date="2024-06-01",
                regions=["米国", "グローバル"],
                entities=["OpenAI", "Microsoft", "Google DeepMind"],
                impact_score=0.95,
                scientific_relevance=True,
                source_type="academic",
            ),
            WorldEvent(
                event_id="AI_2024_002",
                title="DeepSeek-V3の革新的アーキテクチャ",
                description="中国DeepSeekがMixture-of-Expertsを最適化、訓練コストを大幅に削減しながらGPT-4対等の性能を達成。",
                category="ai_agents",
                start_date="2024-12-01",
                regions=["中国", "米国", "グローバル"],
                entities=["DeepSeek", "中国科学院", "Meta AI"],
                impact_score=0.92,
                scientific_relevance=True,
                source_type="academic",
                related_papers=["arXiv:2401.04088", "arXiv:2402.17019"],
            ),
            WorldEvent(
                event_id="AI_2025_001",
                title="Claude 4/Gemini Ultraの競合",
                description="AnthropicとGoogleがClaude 4/Gemini Ultraを発表、推論コストと性能で競争激化。",
                category="ai_agents",
                start_date="2025-03-15",
                regions=["米国"],
                entities=["Anthropic", "Google DeepMind"],
                impact_score=0.88,
                scientific_relevance=True,
                source_type="academic",
            ),
            WorldEvent(
                event_id="AI_2025_002",
                title="EU AI Actの実施とグローバル規制",
                description="EU AI Actが全面施行、リスクベースの規制枠組みがAI開発に広範な影響。",
                category="ai_agents",
                start_date="2025-08-01",
                regions=["欧州", "グローバル"],
                entities=["欧州委員会", "OpenAI", "Meta AI"],
                impact_score=0.82,
                source_type="government",
            ),
            WorldEvent(
                event_id="AI_2026_001",
                title="AGI安全性研究の進展と議論",
                description="DeepMind/OpenAI/AnthropicがAGI安全性研究で協力、紅隊評価プロトコルを標準化。",
                category="ai_agents",
                start_date="2026-01-10",
                regions=["米国", "欧州", "中国"],
                entities=["DeepMind", "OpenAI", "Anthropic"],
                impact_score=0.90,
                scientific_relevance=True,
                source_type="academic",
            ),
        ]
        for event in events:
            self.events[event.event_id] = event

    def _add_science_math_events(self) -> None:
        """科学・数学の革新的発見（ArXiv/BioRxiv関連）"""
        events = [
            WorldEvent(
                event_id="SCI_2024_001",
                title="ペロブスカイト太陽電池の効率記録更新",
                description="ペロブスカイト太陽電池の変換効率が33%以上に到達、シリコン太陽電池を凌駕。",
                category="science_math",
                start_date="2024-02-15",
                regions=["グローバル"],
                entities=["Nature Energy", "Science"],
                impact_score=0.88,
                scientific_relevance=True,
                source_type="academic",
                related_papers=["arXiv:2402.10345", "arXiv:2401.09899"],
            ),
            WorldEvent(
                event_id="SCI_2024_002",
                title="量子コンピューティングでの誤り訂正の進歩",
                description="Google/IBMが量子誤り訂正コードの実用化に前進、論理量子ビット寿命を延長。",
                category="science_math",
                start_date="2024-05-20",
                regions=["米国"],
                entities=["Google Quantum AI", "IBM Research"],
                impact_score=0.92,
                scientific_relevance=True,
                source_type="academic",
                related_papers=["arXiv:2405.05332"],
            ),
            WorldEvent(
                event_id="SCI_2024_003",
                title="AlphaFold 3によるタンパク質構造予測",
                description="Google DeepMindがAlphaFold 3を発表、DNA/RNA/リガンド結合予測も可能に。",
                category="science_math",
                start_date="2024-08-01",
                regions=["英国", "グローバル"],
                entities=["DeepMind", "EMBL-EBI"],
                impact_score=0.95,
                scientific_relevance=True,
                source_type="academic",
                related_papers=["arXiv:2408.14608"],
            ),
            WorldEvent(
                event_id="SCI_2025_001",
                title="常温超伝導体のLK-99追試と結論",
                description="LK-99の常温超伝導性が複数の独立グループによって否定されるも、超伝導研究は進展。",
                category="science_math",
                start_date="2025-01-15",
                regions=["韓国", "米国", "中国"],
                entities=[
                    "Sukbae Leeチーム",
                    "中国科学院",
                    "ローレンス・バークレー国立研究所",
                ],
                impact_score=0.85,
                scientific_relevance=True,
                source_type="academic",
                related_papers=["arXiv:2501.00234", "arXiv:2501.00345"],
            ),
            WorldEvent(
                event_id="SCI_2025_002",
                title="整数論における重要な進歩",
                description="BSD予想（ベイリ・ スウィナートン-ダイアー予想）の部分的進歩、楕円曲線のL関数の零点数とランクの関係で新知見。",
                category="science_math",
                start_date="2025-04-10",
                regions=["グローバル"],
                entities=["Princeton IAS", "ケンブリッジ大学"],
                impact_score=0.90,
                scientific_relevance=True,
                source_type="academic",
                related_papers=["arXiv:2504.05678"],
            ),
            WorldEvent(
                event_id="SCI_2026_001",
                title="核融合発電の科学的マイルストーン",
                description="NIF（国立点火施設）が正能量利得を達成、商用核融合発電への道が開ける。",
                category="science_math",
                start_date="2026-02-01",
                regions=["米国"],
                entities=["ローレンス・リバモア国立研究所", "NIF"],
                impact_score=0.95,
                scientific_relevance=True,
                source_type="academic",
            ),
        ]
        for event in events:
            self.events[event.event_id] = event

    def _add_cyber_security_events(self) -> None:
        """サイバーセキュリティ・情報戦イベント"""
        events = [
            WorldEvent(
                event_id="CYBER_2024_001",
                title="米中間のサイバー攻防激化",
                description=" Volt Typhoon / Salt Typhoon による米国重要インフラへの攻撃が報道、CISAが警戒発表。",
                category="cyber_security",
                start_date="2024-02-20",
                regions=["米国", "中国"],
                entities=["CISA", "中国国家安全省", "FBI"],
                impact_score=0.88,
                source_type="intelligence",
            ),
            WorldEvent(
                event_id="CYBER_2024_002",
                title="Microsoft/Cloudflare等の大規模障害",
                description="クラウドサービスの大規模障害が発生、サイバー攻撃との関連が調査される。",
                category="cyber_security",
                start_date="2024-07-01",
                regions=["グローバル"],
                entities=["Microsoft", "Cloudflare", "CrowdStrike"],
                impact_score=0.75,
                source_type="news",
            ),
            WorldEvent(
                event_id="CYBER_2025_001",
                title="AIモデルのセキュリティ研究",
                description="プロンプトインジェクション攻撃の防御研究が進展、Red teamingが標準化。",
                category="cyber_security",
                start_date="2025-05-15",
                regions=["米国", "欧州"],
                entities=["OpenAI", "Google DeepMind", "ANTHROPIC"],
                impact_score=0.80,
                scientific_relevance=True,
                source_type="academic",
            ),
            WorldEvent(
                event_id="CYBER_2026_001",
                title="量子暗号への移行開始",
                description="NIST後量子暗号標準が主要IT企業で使用開始、ポスト量子暗号移行が本格化。",
                category="cyber_security",
                start_date="2026-06-01",
                regions=["米国", "グローバル"],
                entities=["NIST", "Google", "Cloudflare"],
                impact_score=0.85,
                scientific_relevance=True,
                source_type="government",
            ),
        ]
        for event in events:
            self.events[event.event_id] = event

    def _add_geopolitics_events(self) -> None:
        """地政学的緊張イベント"""
        events = [
            WorldEvent(
                event_id="GEO_2024_001",
                title="ロシア・ウクライナ戦争の長期化",
                description="2024年も紛争は継続、西側支援とロシアの反撃が激化。停戦交渉は進展せず。",
                category="geopolitics",
                start_date="2024-01-01",
                regions=["ウクライナ", "ロシア", "欧州", "米国"],
                entities=["ウクライナ政府", "ロシア政府", "NATO"],
                impact_score=0.92,
                source_type="news",
            ),
            WorldEvent(
                event_id="GEO_2024_002",
                title="中東情勢の変動",
                description="以色列・パレスチナ紛争の継続、イラン・以色列間の緊張、米国の中東政策の再調整。",
                category="geopolitics",
                start_date="2024-01-15",
                regions=["中東", "米国"],
                entities=["以色列政府", "パレスティナ自治区", "イラン"],
                impact_score=0.88,
                source_type="news",
            ),
            WorldEvent(
                event_id="GEO_2025_001",
                title="台湾海峡情勢",
                description="中国人民解放軍が台湾周辺での軍事演習を継続、米国との軍事バランスに変化。",
                category="geopolitics",
                start_date="2025-04-01",
                regions=["台湾", "中国", "米国", "日本"],
                entities=["中国人民解放軍", "台湾国防部", "米太平洋軍"],
                impact_score=0.90,
                source_type="intelligence",
            ),
            WorldEvent(
                event_id="GEO_2026_001",
                title="朝鮮半島情勢",
                description="北朝鮮の核・ミサイル開発が継続、日米韓連携の強化。",
                category="geopolitics",
                start_date="2026-01-20",
                regions=["朝鮮半島", "日本", "米国"],
                entities=["朝鮮労働党", "韓国政府", "自衛隊"],
                impact_score=0.85,
                source_type="intelligence",
            ),
        ]
        for event in events:
            self.events[event.event_id] = event

    def get_events_by_category(self, category: str) -> List[WorldEvent]:
        """カテゴリ別のイベントを取得"""
        return [e for e in self.events.values() if e.category == category]

    def get_events_by_date_range(self, start: str, end: str) -> List[WorldEvent]:
        """日付範囲内のイベントを取得"""
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        return [
            e
            for e in self.events.values()
            if start_dt <= datetime.strptime(e.start_date, "%Y-%m-%d") <= end_dt
        ]

    def get_scientific_events(self) -> List[WorldEvent]:
        """科学的関連性の高いイベントを取得"""
        return [e for e in self.events.values() if e.scientific_relevance]

    def generate_quadruple_reasoning(self, event: WorldEvent) -> Dict[str, str]:
        """
        イベントの四重推論データを生成

        四重推論:
        1. think-task: 事実・状況の観察
        2. think-analysis: 論理的分析・推論
        3. think-safety: リスク・脆弱性・安全性
        4. think-policy: 政策・行動の推奨
        """
        task_content = f"""
        イベント: {event.title}
        日付: {event.start_date} - {event.end_date or "継続中"}
        カテゴリ: {self.CATEGORIES.get(event.category, event.category)}
        影響度: {event.impact_score:.2f}
        関与地域: {", ".join(event.regions)}
        関与主体: {", ".join(event.entities)}
        概要: {event.description}
        """

        analysis_content = f"""
        このイベントの影響分析:
        
        1. 短期影響: {event.category}セクターにおける即時的な変化
        2. 中期影響: 関連産業・国家間関係への波及効果
        3. 長期影響: 構造的な変化の可能性
        
        科学的関連性: {"高い" if event.scientific_relevance else "限定的"}
        データソース信頼性: {event.source_type}
        
        類似イベントとの比較:
        - 過去の同カテゴリイベントとの比較分析が必要
        - 影響度の妥当性を検証
        """

        safety_content = f"""
        リスク・安全性評価:
        
        1. 誤情報リスク: {event.source_type}ソースの信頼性を検証必要
        2. サイバーセキュリティリスク: 関連インフラへの攻撃可能性
        3. 地政学的リスク: 紛争エスカレーションの可能性
        4. 経済リスク: 市場変動・供給チェーンへの影響
        
        監視すべき指標:
        - 関連ニュースの信頼性確認
        - 政府発表の動向
        - 市場・経済の反応
        """

        policy_content = f"""
        推奨アクション:
        
        1. 情報収集: 公式ソースからの追加情報確認
        2. 影響評価: 自組織・地域への影響の具体的分析
        3. 対応準備: シナリオ별対応計画の策定
        4. 継続監視: 展開状況の定期的レビュー
        
        優先度: {"高" if event.impact_score > 0.8 else "中" if event.impact_score > 0.6 else "低"}
        対応期限: 1-2週間以内に初期評価を完了
        """

        return {
            "think-task": task_content.strip(),
            "think-analysis": analysis_content.strip(),
            "think-safety": safety_content.strip(),
            "think-policy": policy_content.strip(),
        }

    def apply_quadruple_reasoning_to_all(self) -> None:
        """全イベントに四重推論を適用"""
        for event in self.events.values():
            if event.quadruple_reasoning is None:
                event.quadruple_reasoning = self.generate_quadruple_reasoning(event)
        logger.info(f"Applied quadruple reasoning to {len(self.events)} events")

    def export_to_json(self, output_path: str) -> None:
        """イベントデータをJSONにエクスポート"""
        data = {
            "export_date": datetime.now().isoformat(),
            "total_events": len(self.events),
            "categories": self.CATEGORIES,
            "events": {
                event_id: event.to_dict() for event_id, event in self.events.items()
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(self.events)} events to {output_path}")

    def load_from_json(self, input_path: str) -> None:
        """JSONからイベントデータをインポート"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.events = {}
        for event_id, event_data in data.get("events", {}).items():
            event = WorldEvent(
                event_id=event_data["event_id"],
                title=event_data["title"],
                description=event_data["description"],
                category=event_data["category"],
                start_date=event_data["start_date"],
                end_date=event_data.get("end_date"),
                regions=event_data.get("regions", []),
                entities=event_data.get("entities", []),
                impact_score=event_data.get("impact_score", 0.5),
                scientific_relevance=event_data.get("scientific_relevance", False),
                source_type=event_data.get("source_type", "news"),
                quadruple_reasoning=event_data.get("quadruple_reasoning"),
                related_papers=event_data.get("related_papers", []),
            )
            self.events[event_id] = event

        logger.info(f"Loaded {len(self.events)} events from {input_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        by_category = {}
        for category in self.CATEGORIES.keys():
            events_in_cat = self.get_events_by_category(category)
            by_category[category] = {
                "count": len(events_in_cat),
                "avg_impact": sum(e.impact_score for e in events_in_cat)
                / len(events_in_cat)
                if events_in_cat
                else 0,
            }

        return {
            "total_events": len(self.events),
            "by_category": by_category,
            "scientific_events": len(self.get_scientific_events()),
            "categories": self.CATEGORIES,
        }
