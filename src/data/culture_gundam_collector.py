"""
Gundam & Pop-Culture Data Collector Module
ガンダム（SEED FREEDOM, GQuuuuuuX, 閃光のハサウェイ）を中心とした文化データの収集と
三段階推論（物語分析、技術考察、哲学的背景）の生成。
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GundamCultureDataCollector:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_gundam_reasoning_samples(self) -> List[Dict[str, Any]]:
        """
        ガンダム作品に基づいた高度な推論サンプルを生成（シードデータ）。
        """
        samples = [
            {
                "topic": "SEED FREEDOM (2024)",
                "prompt": "機動戦士ガンダムSEED FREEDOMにおける『アコード』の概念と、キラ・ヤマトが提示した『必要だから愛するのではない、愛しているから必要なのだ』という結論の倫理学的意義を考察せよ。",
                "response": "<think-task>\nSEED FREEDOMの核心的テーマである運命決定論と自由意志の対立を分析し、キラの言葉の哲学的価値を導き出す。\n</think-task>\n<think-analysis>\n- アコード：遺伝子による役割固定（運命）。カント的な「自律」の否定。\n- キラの見解：存在論的な愛。条件付きの価値（必要性）ではなく、無条件の肯定（アガペーに近い愛）を主張。\n- 結論：ラクスとの相互理解により、役割から解放された個人の尊厳を確立している。\n</think-analysis>\n<think-safety>\n特定の政治的思想への偏りを避け、劇中の文脈と一般倫理学の範囲内で論述する。\n</think-safety>\n<think-policy>\nファンの解釈の多様性を尊重しつつ、物語のロジックに基づいた一貫性のある分析を提供する。\n</think-policy>\n<response>\nキラ・ヤマトの結論は、ドゥルーズ的な『差異』の肯定であり、アコードが象徴する『器官なき身体』への抵抗と解釈できる。役割（必要性）が先立つのではなく、存在そのものへの愛が役割を事後的に正当化するという、サルトル的な実存主義の極致である。... (以下略)"
            },
            {
                "topic": "GQuuuuuuX (2025)",
                "prompt": "2025年新作『GQuuuuuuX』における鶴巻和哉監督の演出手法と、エヴァンゲリオン以降の『セカイ系』の再定義について述べよ。",
                "response": "<think-task>\n新作の監督特性とジャンル史における位置づけを論理的に整理する。\n</think-task>\n<think-analysis>\n- 鶴巻監督：ガイナックス/カラー流のダイナミックなレイアウトと、内省的なキャラクター描写の融合。\n- GQuuuuuuX：『G』を冠しつつ、従来のガンダムの政治劇とは異なる、よりパーソナルな『クオリティ（質的）』宇宙世紀を描く試み。\n- 再定義：個人の内面世界（セカイ）が巨大な社会システム（連邦/ジオン的対立）を単に反映するのではなく、量子的な重なり合いとして描かれている。\n</think-analysis>\n<think-safety>\n未放送/上映内容に関する過度なネタバレや憶測は避け、公表されているスタッフ情報と傾向から分析する。\n</think-safety>\n<think-policy>\nアニメーション技術論と文化社会学的視点の両立を図る。\n</think-policy>\n<response>\n『GQuuuuuuX』は、ポスト・セカイ系における『関係性の再構築』を描いている。鶴巻演出は、エヴァで見られたディスコミュニケーションの極致を、今作では『重力（システム）』と『意志（個人）』の不可分な絡み合いとして視覚化している。... (以下略)"
            }
        ]
        return samples

    def save_samples(self, samples: List[Dict[str, Any]]):
        path = self.output_dir / "gundam_reasoning_seed.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(samples)} Gundam reasoning samples to {path}")
