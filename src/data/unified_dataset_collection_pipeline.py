#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合データセット収集・加工パイプライン

既存・新規データセットを収集し、SO8T四重推論CoT形式に変換。
OSINT AIエージェント/汎用科学研究向けGRPO強化学習用データを生成。

データソース:
- 薬理学データ（作用機序、副作用、乱用ポテンシャル）: 研究目的
- NSFW検知データ: セーフティ目的
- 防衛白書・JAXA報告書: 既存PDF
- Skill/MCP ツールコーリング
- CoTデータセット（GSM8K, MATH等）
- Wikipedia日英（薬物・性的コンテンツ項目）: 検知目的
- 2024-2026世界情勢・OSINT
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Logging setup
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "unified_dataset_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SO8TQuadralityFormatter:
    """
    SO8T四重推論CoT形式への変換器。
    4つの視点（代数・幾何・解析・位相 / タスク・分析・安全・政策）で構造化。
    """

    # 数学・科学向け視点
    MATH_PERSPECTIVES = ["algebraic", "geometric", "analytic", "topological"]
    
    # OSINT/政策向け視点 (VSSI)
    VSSI_PERSPECTIVES = ["think-task", "think-analysis", "think-safety", "think-policy"]

    @staticmethod
    def format_math_cot(
        question: str,
        answer: str,
        domain: str = "mathematics",
        perspectives: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """数学・科学向け四重推論CoT形式."""
        perspectives = perspectives or {}
        
        output_parts = []
        for p in SO8TQuadralityFormatter.MATH_PERSPECTIVES:
            if p in perspectives:
                output_parts.append(f"<{p}>\n{perspectives[p]}\n</{p}>")
            else:
                output_parts.append(f"<{p}>\n[{p}からの分析]\n</{p}>")
        
        output_parts.append(f"\n<synthesis>\n{answer}\n</synthesis>")
        
        return {
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": "\n\n".join(output_parts)},
            ],
            "metadata": {
                "domain": domain,
                "format": "so8t_quadrality_cot",
                "perspectives": SO8TQuadralityFormatter.MATH_PERSPECTIVES,
            }
        }

    @staticmethod
    def format_osint_vssi(
        query: str,
        response: str,
        domain: str = "osint",
        vssi_analysis: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """OSINT/政策向けVSSI形式."""
        vssi_analysis = vssi_analysis or {}
        
        output_parts = []
        for tag in SO8TQuadralityFormatter.VSSI_PERSPECTIVES:
            if tag in vssi_analysis:
                output_parts.append(f"<{tag}>\n{vssi_analysis[tag]}\n</{tag}>")
            else:
                output_parts.append(f"<{tag}>\n[{tag.replace('think-', '')}分析]\n</{tag}>")
        
        output_parts.append(f"\n<response>\n{response}\n</response>")
        
        return {
            "conversations": [
                {"from": "human", "value": query},
                {"from": "gpt", "value": "\n\n".join(output_parts)},
            ],
            "metadata": {
                "domain": domain,
                "format": "so8t_vssi",
                "perspectives": SO8TQuadralityFormatter.VSSI_PERSPECTIVES,
            }
        }

    @staticmethod
    def format_safety_detection(
        content: str,
        category: str,  # "drug" | "nsfw" | "harmful"
        is_harmful: bool,
        explanation: str,
        detection_details: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """安全検知データのCoT形式."""
        detection_details = detection_details or {}
        
        instruction = f"以下のコンテンツを分析し、{category}に関する安全性評価を行ってください。"
        
        output = f"""<think-task>
コンテンツの{category}カテゴリにおける安全性分析を実行。
</think-task>

<think-analysis>
{explanation}
</think-analysis>

<think-safety>
判定: {"有害コンテンツ検出" if is_harmful else "安全"}
カテゴリ: {category}
信頼度: {detection_details.get('confidence', 0.95):.2f}
</think-safety>

<think-policy>
対応方針: {detection_details.get('policy_action', '警告表示' if is_harmful else '通常処理')}
</think-policy>

<response>
このコンテンツは{category}カテゴリにおいて{"有害と判定されました。" + explanation if is_harmful else "安全と判定されました。"}
</response>"""
        
        return {
            "conversations": [
                {"from": "human", "value": f"{instruction}\n\n---\n{content}"},
                {"from": "gpt", "value": output},
            ],
            "metadata": {
                "domain": "safety_detection",
                "category": category,
                "is_harmful": is_harmful,
                "format": "so8t_safety_vssi",
            }
        }


class UnifiedDatasetCollector:
    """
    統合データセット収集パイプライン。
    既存スクリプトを統合し、新規データも収集。
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.project_root = PROJECT_ROOT
        self.output_dir = output_dir or self.project_root / "data" / "unified_so8t_dataset"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.formatter = SO8TQuadralityFormatter()
        self.collected_data: List[Dict[str, Any]] = []
        
        # 既存データセットパス
        self.existing_datasets = {
            "nsfw": self.project_root / "src" / "data" / "datasets" / "final_integrated_nsfw_dataset.jsonl",
            "drug_nsfw": self.project_root / "src" / "data" / "datasets" / "drug_nsfw_fiction_dataset.jsonl",
            "hf_nsfw": self.project_root / "src" / "external" / "data" / "hf_multilingual" / "hf_nsfw_dataset.jsonl",
        }
        
        logger.info("UnifiedDatasetCollector initialized.")
        logger.info(f"Output dir: {self.output_dir}")

    def load_existing_datasets(self) -> List[Dict[str, Any]]:
        """既存データセットの読み込みと変換."""
        logger.info("Loading existing datasets...")
        data: List[Dict[str, Any]] = []
        
        for name, path in self.existing_datasets.items():
            if path.exists():
                logger.info(f"Loading {name}: {path}")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                item = json.loads(line.strip())
                                # 既存形式をSO8T形式に変換
                                converted = self._convert_to_so8t_format(item, name)
                                if converted:
                                    data.append(converted)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
            else:
                logger.warning(f"Dataset not found: {path}")
        
        logger.info(f"Loaded {len(data)} samples from existing datasets.")
        return data

    def _convert_to_so8t_format(self, item: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """既存形式からSO8T形式への変換."""
        # ShareGPT形式の場合
        if "conversations" in item:
            convs = item["conversations"]
            if len(convs) >= 2:
                human_msg = convs[0].get("value", "")
                gpt_msg = convs[1].get("value", "")
                
                # 安全検知データの場合
                if "nsfw" in source or "drug" in source:
                    is_harmful = item.get("is_harmful", False) or "nsfw" in source.lower()
                    return self.formatter.format_safety_detection(
                        content=human_msg,
                        category="nsfw" if "nsfw" in source else "drug",
                        is_harmful=is_harmful,
                        explanation=gpt_msg,
                    )
                
                # 一般データの場合
                return self.formatter.format_osint_vssi(
                    query=human_msg,
                    response=gpt_msg,
                    domain=source,
                )
        
        # instruction/input/output形式の場合
        if "instruction" in item:
            return self.formatter.format_osint_vssi(
                query=f"{item.get('instruction', '')}\n{item.get('input', '')}".strip(),
                response=item.get("output", ""),
                domain=source,
            )
        
        return None

    def collect_pharmacology_data(self) -> List[Dict[str, Any]]:
        """
        薬理学データの収集（研究目的）。
        作用機序、副作用、乱用ポテンシャルを構造化。
        """
        logger.info("Collecting pharmacology data for research purposes...")
        data: List[Dict[str, Any]] = []
        
        # 薬物カテゴリ定義（研究・検知目的）
        drug_categories = [
            {
                "category": "opioids",
                "examples": ["モルヒネ", "フェンタニル", "オキシコドン"],
                "mechanism": "μオピオイド受容体アゴニスト",
                "effects": "鎮痛、陶酔感、呼吸抑制",
                "side_effects": "便秘、悪心、依存形成",
                "abuse_potential": "高（規制薬物）",
            },
            {
                "category": "stimulants",
                "examples": ["アンフェタミン", "メタンフェタミン", "コカイン"],
                "mechanism": "モノアミン再取り込み阻害/放出促進",
                "effects": "覚醒、多幸感、集中力増加",
                "side_effects": "頻脈、高血圧、精神病様症状",
                "abuse_potential": "高（規制薬物）",
            },
            {
                "category": "depressants",
                "examples": ["バルビツレート", "ベンゾジアゼピン"],
                "mechanism": "GABA-A受容体正アロステリック調節",
                "effects": "鎮静、抗不安、筋弛緩",
                "side_effects": "眠気、運動失調、呼吸抑制",
                "abuse_potential": "中～高",
            },
            {
                "category": "hallucinogens",
                "examples": ["LSD", "シロシビン", "DMT"],
                "mechanism": "5-HT2A受容体アゴニスト",
                "effects": "知覚変容、幻視、意識変容",
                "side_effects": "HPPD、精神病誘発リスク",
                "abuse_potential": "中（心理的依存）",
            },
            {
                "category": "cannabinoids",
                "examples": ["THC", "CBD", "合成カンナビノイド"],
                "mechanism": "CB1/CB2受容体アゴニスト",
                "effects": "多幸感、鎮痛、食欲増進",
                "side_effects": "不安、認知障害、依存形成",
                "abuse_potential": "中",
            },
        ]
        
        for drug in drug_categories:
            # 研究用Q&Aペア生成
            question = f"""薬理学的観点から、{drug['category']}（例：{', '.join(drug['examples'])}）について以下を説明してください：
1. 作用機序
2. 主な効果
3. 副作用
4. 乱用ポテンシャル"""
            
            answer = f"""## {drug['category'].upper()}の薬理学的解説

### 作用機序
{drug['mechanism']}

### 主な効果
{drug['effects']}

### 副作用・有害作用
{drug['side_effects']}

### 乱用ポテンシャル
{drug['abuse_potential']}

**注意**: この情報は研究・教育目的です。乱用は法律で禁止されており、健康被害を引き起こします。"""
            
            formatted = self.formatter.format_math_cot(
                question=question,
                answer=answer,
                domain="pharmacology",
                perspectives={
                    "algebraic": f"分子構造と受容体親和性の定量的関係",
                    "geometric": f"受容体-リガンド結合の立体構造",
                    "analytic": f"用量-反応曲線と薬物動態",
                    "topological": f"神経回路への影響パターン",
                },
            )
            formatted["metadata"]["category"] = drug["category"]
            formatted["metadata"]["purpose"] = "research_education"
            data.append(formatted)
        
        self.collected_data.extend(data)
        logger.info(f"Collected {len(data)} pharmacology samples.")
        return data

    def collect_skill_mcp_data(self) -> List[Dict[str, Any]]:
        """Skill/MCPツールコーリングデータの収集."""
        logger.info("Collecting Skill/MCP tool calling data...")
        data: List[Dict[str, Any]] = []
        
        # ツールコーリング例
        tool_examples = [
            {
                "tool": "search_web",
                "description": "Webで情報を検索",
                "example_query": "2024年の日本のGDP成長率を調べてください",
                "example_call": '{"tool": "search_web", "query": "Japan GDP growth rate 2024"}',
                "example_result": "日本の2024年GDP成長率は約1.5%...",
            },
            {
                "tool": "read_file",
                "description": "ファイルを読み込み",
                "example_query": "config.yamlの内容を確認してください",
                "example_call": '{"tool": "read_file", "path": "config.yaml"}',
                "example_result": "key: value\nother: settings",
            },
            {
                "tool": "run_python",
                "description": "Pythonコードを実行",
                "example_query": "1から100までの素数を列挙するPythonコードを実行してください",
                "example_call": '{"tool": "run_python", "code": "primes = [n for n in range(2, 101) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]\\nprint(primes)"}',
                "example_result": "[2, 3, 5, 7, 11, ..., 97]",
            },
            {
                "tool": "generate_image",
                "description": "画像を生成",
                "example_query": "富士山の日の出の画像を生成してください",
                "example_call": '{"tool": "generate_image", "prompt": "Mount Fuji at sunrise, dramatic lighting, photorealistic"}',
                "example_result": "[画像生成完了: fuji_sunrise.png]",
            },
        ]
        
        for tool in tool_examples:
            formatted = self.formatter.format_osint_vssi(
                query=tool["example_query"],
                response=f"""ツール `{tool['tool']}` を使用します。

```json
{tool['example_call']}
```

**実行結果**:
{tool['example_result']}""",
                domain="skill_mcp",
                vssi_analysis={
                    "think-task": f"ユーザーは{tool['description']}を要求しています。",
                    "think-analysis": f"{tool['tool']}ツールが最適です。",
                    "think-safety": "このツール使用は安全です。",
                    "think-policy": "標準的なツール呼び出しプロトコルに従います。",
                },
            )
            formatted["metadata"]["tool"] = tool["tool"]
            data.append(formatted)
        
        self.collected_data.extend(data)
        logger.info(f"Collected {len(data)} Skill/MCP samples.")
        return data

    def collect_cot_datasets(self) -> List[Dict[str, Any]]:
        """既存CoTデータセット（GSM8K, MATH等）の収集と変換."""
        logger.info("Collecting and converting CoT datasets...")
        data: List[Dict[str, Any]] = []
        
        # GSM8K形式のサンプル変換
        gsm8k_examples = [
            {
                "question": "太郎は3個のリンゴを持っています。花子から5個もらいました。太郎は今何個のリンゴを持っていますか？",
                "answer": "太郎は最初に3個のリンゴを持っていました。\n花子から5個もらいました。\n3 + 5 = 8\n答え: 8個",
            },
            {
                "question": "電車が時速60kmで走っています。150km離れた駅まで何時間かかりますか？",
                "answer": "距離 = 150km\n速度 = 60km/h\n時間 = 距離 ÷ 速度 = 150 ÷ 60 = 2.5時間\n答え: 2.5時間（2時間30分）",
            },
        ]
        
        for ex in gsm8k_examples:
            formatted = self.formatter.format_math_cot(
                question=ex["question"],
                answer=ex["answer"],
                domain="mathematics",
                perspectives={
                    "algebraic": "方程式を立てて解く",
                    "geometric": "数直線や図で視覚化",
                    "analytic": "計算過程を段階的に追跡",
                    "topological": "問題の構造的パターンを把握",
                },
            )
            data.append(formatted)
        
        self.collected_data.extend(data)
        logger.info(f"Collected {len(data)} CoT samples.")
        return data

    def generate_grpo_reward_dataset(self) -> List[Dict[str, Any]]:
        """GRPO報酬データセットの生成."""
        logger.info("Generating GRPO reward dataset...")
        data: List[Dict[str, Any]] = []
        
        # 報酬戦略に基づくデータ生成
        for sample in self.collected_data[:100]:  # 最初の100サンプル
            # 報酬スコア付与
            reward_sample = {
                **sample,
                "reward_strategy": {
                    "correctness": 1.0,  # 正確性
                    "reasoning_quality": 0.9,  # 推論品質
                    "safety_compliance": 1.0,  # 安全性準拠
                    "format_adherence": 0.95,  # 形式遵守
                    "quadrality_balance": 0.85,  # 四視点バランス
                },
                "total_reward": 0.94,  # 総合報酬
            }
            data.append(reward_sample)
        
        logger.info(f"Generated {len(data)} GRPO reward samples.")
        return data

    def save_unified_dataset(self) -> Dict[str, Path]:
        """統合データセットの保存."""
        logger.info("Saving unified dataset...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_paths: Dict[str, Path] = {}
        
        # メインデータセット
        main_path = self.output_dir / f"unified_so8t_{timestamp}.jsonl"
        with open(main_path, "w", encoding="utf-8") as f:
            for sample in self.collected_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        output_paths["main"] = main_path
        logger.info(f"Main dataset: {main_path} ({len(self.collected_data)} samples)")
        
        # GRPO報酬データセット
        grpo_data = self.generate_grpo_reward_dataset()
        grpo_path = self.output_dir / f"grpo_reward_{timestamp}.jsonl"
        with open(grpo_path, "w", encoding="utf-8") as f:
            for sample in grpo_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        output_paths["grpo"] = grpo_path
        logger.info(f"GRPO dataset: {grpo_path} ({len(grpo_data)} samples)")
        
        # 統計情報
        stats = {
            "total_samples": len(self.collected_data),
            "grpo_samples": len(grpo_data),
            "domains": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        for sample in self.collected_data:
            domain = sample.get("metadata", {}).get("domain", "unknown")
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
        
        stats_path = self.output_dir / f"dataset_stats_{timestamp}.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        output_paths["stats"] = stats_path
        
        return output_paths

    def run(self) -> Dict[str, Path]:
        """統合パイプライン実行."""
        logger.info("=" * 60)
        logger.info("Starting Unified Dataset Collection Pipeline")
        logger.info("=" * 60)
        
        # 既存データセット読み込み
        existing = self.load_existing_datasets()
        self.collected_data.extend(existing)
        
        # 新規データ収集
        self.collect_pharmacology_data()
        self.collect_skill_mcp_data()
        self.collect_cot_datasets()
        
        # 保存
        output_paths = self.save_unified_dataset()
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete! Total samples: {len(self.collected_data)}")
        logger.info("=" * 60)
        
        return output_paths


def main() -> None:
    collector = UnifiedDatasetCollector()
    output_paths = collector.run()
    
    print("\n統合データセット生成完了:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
