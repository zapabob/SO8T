import json
import os
from pathlib import Path

def generate_enrichment_data():
    """
    Generates a structured JSONL dataset for model enrichment (2024-2026).
    """
    data = [
        # --- World Events / Geopolitics ---
        {
            "instruction": "2024年から2026年にかけての世界情勢の主要な動向を要約してください。",
            "input": "",
            "output": "2024年から2026年は、地政学的な不安定さと多極化が加速した時期でした。主要なトピックは以下の通りです：\n1. ウクライナ・ロシア情勢：長期化する紛争の中で軍事的支援の枠組みが変化し、地政学的な分断が定着しました。\n2. 中東情勢：イスラエル・ガザ紛争の波及、シリアのアサド政権崩壊（2024年末）など、歴史的な転換点がありました。\n3. 米国大統領選挙（2024年）：ドナルド・トランプ氏の勝利と再就任により、保護主義的な経済政策と外交方針の転換が世界経済に大きな影響を与えました。\n4. AI競争と規制：国家間のAI覇権争いが激化し、エネルギー供給や半導体供給網の確保が国家安全保障の最優先課題となりました。",
            "category": "world_events"
        },
        # --- Anime / Gundam ---
        {
            "instruction": "2024年から2026年の『ガンダム』シリーズの主要な作品とその評価について教えてください。",
            "input": "",
            "output": "この期間、ガンダムシリーズは劇場作品とTV放送で大きな成功を収めました。\n1. 機動戦士ガンダムSEED FREEDOM (2024年): 興行収入50億円を突破し、ガンダム映画史上最高のヒットを記録。ファン納得の完結編として高く評価されました。\n2. 機動戦士ガンダム GQuuuuuuX (2025年): 劇場版序章の好発進を受け、TVシリーズが放送。宇宙世紀とは異なる新たな世界観と高いドラマ性が話題を呼び、第45回日本SF大賞を受賞するなど、批評家からも絶賛されました。\n3. 機動戦士ガンダム 閃光のハサウェイ『サン・オブ・ブライト』（仮称、2026年初頭）: 三部作の第2部。圧倒的な映像美とハサウェイの葛藤が深化し、Filmarksでの満足度も非常に高い滑り出しを見せました。",
            "category": "anime"
        },
        # --- Movies ---
        {
            "instruction": "2024年から2026年の大ヒット映画の動向と注目の作品をまとめてください。",
            "input": "",
            "output": "2024-2026年は大作の続編と革新的なSFが市場を牽引しました。\n1. 『デューン 砂の惑星 PART2』(2024) & 『PART3』(2026見込み): ヴィルヌーヴ監督による壮大な叙事詩が完結に向け進行。視覚効果と音楽が高く評価されました。\n2. 『インサイド・ヘッド2』(2024): アニメーション映画として記録的な興行収入を達成し、若年層の共感を集めました。\n3. 『ゴジラ-1.0』(2023-2024): 日本公開後の世界展開でアカデミー賞視覚効果賞を受賞。VFXの新たな基準を打ち立てました。\n4. 『アベンジャーズ：ドゥームズデイ』(2026予定): ロバート・ダウニー・Jrの復帰発表で期待が最大級に高まり、映画業界の注目を集めています。",
            "category": "movies"
        },
        # --- LLM / AI Agents ---
        {
            "instruction": "Borea-phi3.5mini以降のLLMの発展と、2026年現在のベンチマークやAIエージェントの動向について教えてください。",
            "input": "",
            "output": "2024年後半から2026年にかけて、LLMは「推論能力の深化」と「エージェント化」の二軸で進化しました。\n1. 主要モデル: OpenAIのo1/GPT-5ファミリー、AnthropicのClaude 3.5/4シリーズ、GoogleのGemini 2/3、そしてDeepSeek V3/R1などが台頭。特にDeepSeekの数学・コーディング能力は驚異的な進化を遂げました。\n2. ベンチマークの変化: MMLUなどの定型評価から、SWE-bench（実務コーディング）やGPQA（高度な科学推論）、LiveBench（汚染回避型評価）へとシフト。推論時間スケーリング（Reasoning-time scaling）が主流となりました。\n3. AIエージェント: 「道具としてのAI」から「チームの一員（Teammate）」へ。自律的にGitHubのIssueを解決したり、数日間にわたる長期リサーチを完遂するマルチエージェント・システムが実用化されました。AnthropicのComputer Useや各社のAgentic APIが、OS操作やブラウジングを統合しました。",
            "category": "ai_tech"
        }
    ]

    dest_dir = Path("c:/Users/downl/Desktop/SO8T/data/enrichment_2026")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = dest_dir / "enrichment_2026_dataset.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Enrichment dataset generated at: {output_file}")

if __name__ == "__main__":
    generate_enrichment_data()
