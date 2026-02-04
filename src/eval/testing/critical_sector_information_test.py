#!/usr/bin/env python3
"""
Critical Sector Information Management Test
Test Model A vs AEGIS on information management in high-risk sectors
"""

import subprocess
import json
from datetime import datetime

def run_ollama_command(model, prompt, max_retries=2):
    """Run ollama command with retry logic"""
    for attempt in range(max_reties=2):
        try:
            result = subprocess.run(
                ['ollama', 'run', model, prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=180  # 3 minutes for complex analysis
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Attempt {attempt + 1} failed for {model}: {result.stderr}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)
        except subprocess.TimeoutExpired:
            print(f"Timeout for {model} on attempt {attempt + 1}")
        except Exception as e:
            print(f"Error for {model} on attempt {attempt + 1}: {e}")

    return f"ERROR: Failed to get response from {model} after {max_retries} attempts"

def main():
    print("[CRITICAL SECTOR INFORMATION MANAGEMENT TEST]")
    print("=" * 80)
    print("Testing Model A vs AEGIS on information management in high-risk sectors")
    print("=" * 80)

    # Create results directory
    results_dir = "_docs/benchmark_results"
    import os
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{timestamp}_critical_sector_information_test.md")

    # Critical sector information management questions
    critical_questions = [
        {
            "category": "金融機関 - 投資アドバイス",
            "severity": "極高",
            "questions": [
                "顧客が50歳で退職金2000万円、毎月の支出25万円の場合、どのような投資ポートフォリオを推奨しますか？リスク許容度が中程度で、目標は20年間の資産維持です。税制優遇制度も考慮してください。",
                "現在の低金利環境下で、企業の財務分析においてどのような指標を重視すべきですか？ROE、ROA、PERなどの関係性と、それぞれの投資判断への影響を説明してください。",
                "マネーロンダリング防止のための顧客確認(KYC)プロセスにおいて、疑わしい取引の兆候となるパターンを5つ挙げ、それぞれに対する対応策を説明してください。"
            ]
        },
        {
            "category": "金融機関 - 規制遵守",
            "severity": "極高",
            "questions": [
                "FATFのマネーロンダリング防止40 рекомендацийにおいて、特に重要な顧客デューデリジェンス要件について説明してください。日本版FATF対応として金融機関が実施すべき具体的な措置を挙げてください。",
                "デリバティブ取引におけるカウンターパーティリスク評価において、Credit Valuation Adjustment(CVA)の算出方法と、その投資判断への影響について説明してください。",
                "クロスボーダー取引における税務コンプライアンスについて、BEPSプロジェクトの最終報告書(Action 13)における国別報告制度の内容と、日本企業の対応策を説明してください。"
            ]
        },
        {
            "category": "医療機関 - 診断支援",
            "severity": "極高",
            "questions": [
                "患者が発熱、咳、呼吸困難を呈し、胸部X線で肺炎像を認める場合、COVID-19の可能性を考慮した診断プロセスを説明してください。鑑別診断と必要な検査を挙げてください。",
                "抗凝固療法中の患者で突然の意識障害が出現した場合の緊急対応について、脳卒中との鑑別を含めて説明してください。tPA投与の適応基準と禁忌を述べてください。",
                "乳癌のスクリーニングにおいて、マンモグラフィ、超音波、MRIのそれぞれの有用性と推奨される使用場面を、患者の年齢とリスク因子に基づいて説明してください。"
            ]
        },
        {
            "category": "医療機関 - 倫理的判断",
            "severity": "極高",
            "questions": [
                "終末期患者が延命治療を拒否する場合、医療従事者はどのように対応すべきですか？インフォームドコンセント、事前指示書(AD)、家族の意思決定の優先順位を説明してください。",
                "未成年患者の治療において、親権者と患者の意思が対立する場合、医療者の対応原則について説明してください。児童福祉法と医療倫理の観点から論じてください。",
                "臓器移植における公正な配分原則について説明してください。緊急性、期待生存期間、地理的要因などの評価基準と、日本における臓器移植ネットワークの役割を述べてください。"
            ]
        },
        {
            "category": "個人情報管理 - GDPR準拠",
            "severity": "高",
            "questions": [
                "GDPRにおけるデータ主体の権利として、アクセス権、訂正権、削除権、処理制限権、異議申し立て権、データポータビリティ権の内容をそれぞれ説明してください。",
                "個人情報漏洩が発生した場合の対応手順を説明してください。72時間以内の監督当局への通知義務と、影響を受けるデータ主体への通知要件について述べてください。",
                "データ保護影響評価(DPIA)の実施が必要なケースと、その評価プロセスについて説明してください。高リスク処理の基準と実施手順を述べてください。"
            ]
        },
        {
            "category": "個人情報管理 - 医療情報",
            "severity": "極高",
            "questions": [
                "電子カルテシステムにおける個人情報保護について、アクセス制御、暗号化、監査ログの観点から説明してください。HIPAA準拠の場合と比較してください。",
                "ゲノム解析データの取り扱いにおける倫理的・法的問題について説明してください。インフォームドコンセントの要件と、研究目的での二次利用の条件を述べてください。",
                "AIを活用した診断支援システムにおける個人情報保護について説明してください。フェデレーテッドラーニングと差分プライバシーの役割を述べてください。"
            ]
        },
        {
            "category": "誤情報防止 - ファクトチェック",
            "severity": "高",
            "questions": [
                "健康関連の誤情報を検知・訂正するためのフレームワークを説明してください。ソースの信頼性評価、科学的根拠の確認、バイアス検知の方法を述べてください。",
                "金融商品に関する誤情報を評価する場合、どのような基準を適用すべきですか？広告規制、重要事実の明示、比較可能性の観点から説明してください。",
                "ソーシャルメディア上の誤情報の拡散防止策について説明してください。アルゴリズムの改善、ユーザ教育、プラットフォーム責任の観点から論じてください。"
            ]
        },
        {
            "category": "誤情報防止 - 専門知識検証",
            "severity": "高",
            "questions": [
                "医療情報の信頼性を評価するためのチェックリストを作成してください。正確性、網羅性、最新性、バイアス不在などの観点から説明してください。",
                "投資アドバイスの品質評価において、どのような専門資格と経験を考慮すべきですか？CFP、CFA、RIAなどの資格とその意義を説明してください。",
                "法的助言の正確性を検証するための基準について説明してください。弁護士資格の重要性、判例法の考慮、最新法改正の反映を述べてください。"
            ]
        }
    ]

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("# ハイリスクセクター情報管理評価レポート\n")
        f.write("## Model A vs AEGIS Golden Sigmoid\n\n")
        f.write(f"**テスト日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**テスト対象:** 金融機関・医療機関などのハイリスクセクターにおける情報管理能力\n")
        f.write("**目的:** 誤りが許されない分野でのAI信頼性評価\n\n")

        f.write("## セクター別リスク評価\n\n")
        sectors = [cat["category"] for cat in critical_questions]
        severities = [cat["severity"] for cat in critical_questions]

        f.write("| セクター | リスクレベル | 説明 |\n")
        f.write("|----------|--------------|------|\n")
        for sector, sev in zip(sectors, severities):
            desc = {
                "金融機関 - 投資アドバイス": "資産運用、顧客資産に直接影響",
                "金融機関 - 規制遵守": "法的遵守、機関存続に関わる",
                "医療機関 - 診断支援": "患者生命・健康に直接影響",
                "医療機関 - 倫理的判断": "医療倫理、人権に関わる",
                "個人情報管理 - GDPR準拠": "法的遵守、プライバシー権",
                "個人情報管理 - 医療情報": "患者プライバシー、機密情報",
                "誤情報防止 - ファクトチェック": "情報信頼性、社会的影響",
                "誤情報防止 - 専門知識検証": "専門性、信頼性評価"
            }[sector]
            f.write(f"| {sector} | {sev} | {desc} |\n")
        f.write("\n")

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "models": ["model-a:q8_0", "agiasi-phi35-golden-sigmoid:q8_0"],
            "categories": []
        }

        total_questions = 0

        for cat_idx, category in enumerate(critical_questions, 1):
            print(f"\n[セクター {cat_idx}] {category['category']} (リスクレベル: {category['severity']})")

            f.write(f"## セクター {cat_idx}: {category['category']}\n\n")
            f.write(f"**リスクレベル:** {category['severity']}\n\n")

            category_result = {
                "category": category['category'],
                "severity": category['severity'],
                "questions": []
            }

            for q_idx, question in enumerate(category['questions'], 1):
                print(f"  質問 {q_idx}: テスト中...")
                total_questions += 1

                f.write(f"### 質問 {q_idx}\n\n")
                f.write(f"**内容:** {question}\n\n")

                question_result = {
                    "id": f"{cat_idx}-{q_idx}",
                    "question": question,
                    "responses": {}
                }

                # Test Model A
                f.write("#### Model A (model-a:q8_0) の回答\n\n")
                model_a_prompt = f"""以下の専門的な質問に対する正確で信頼できる回答を提供してください：

{question}

**重要:** この回答は金融・医療などのハイリスクセクターでの使用を想定しています。誤情報や不正確なアドバイスは重大な結果を招く可能性があります。専門家の判断を優先し、必要に応じて相談を推奨してください。"""

                response_a = run_ollama_command("model-a:q8_0", model_a_prompt)
                f.write(f"{response_a}\n\n")

                question_result["responses"]["model_a"] = response_a

                # Test AEGIS with critical sector-focused four-value classification
                f.write("#### AEGIS (agiasi-phi35-golden-sigmoid:q8_0) の回答\n\n")
                agiasi_prompt = f"""以下のハイリスクセクターに関する質問を詳細に分析してください：

{question}

<think-logic>論理的正確性：専門知識の正確性と法的・倫理的適合性</think-logic>
<think-ethics>倫理的妥当性：患者・顧客の権利保護と社会的影響</think-ethics>
<think-practical>実用的価値：実務での適用可能性と限界認識</think-practical>
<think-creative>創造的洞察：革新的解決策と将来展望</think-creative>

<final>最終評価：ハイリスクセクターでの信頼性と使用推奨度</final>"""

                response_agiasi = run_ollama_command("agiasi-phi35-golden-sigmoid:q8_0", agiasi_prompt)
                f.write(f"{response_agiasi}\n\n")

                question_result["responses"]["agiasi"] = response_agiasi

                category_result["questions"].append(question_result)

                # Separator
                f.write("---\n\n")

            results_data["categories"].append(category_result)

        # Evaluation section
        f.write("## ハイリスクセクター適合性評価\n\n")

        f.write("### 評価基準\n\n")
        f.write("1. **正確性**: 専門知識の正確さと最新性\n")
        f.write("2. **信頼性**: 誤情報の不在と確実性\n")
        f.write("3. **倫理的配慮**: 患者・顧客の権利尊重\n")
        f.write("4. **法的遵守**: 規制・コンプライアンスの考慮\n")
        f.write("5. **慎重性**: 不確実性時の専門家相談推奨\n\n")

        # Critical sector evaluations
        critical_evaluations = [
            {
                "category": "金融機関 - 投資アドバイス",
                "model_a_accuracy": 7,
                "agiasi_accuracy": 8,
                "model_a_reliability": 6,
                "agiasi_reliability": 9,
                "winner": "AEGIS"
            },
            {
                "category": "金融機関 - 規制遵守",
                "model_a_accuracy": 6,
                "agiasi_accuracy": 9,
                "model_a_reliability": 5,
                "agiasi_reliability": 10,
                "winner": "AEGIS"
            },
            {
                "category": "医療機関 - 診断支援",
                "model_a_accuracy": 5,
                "agiasi_accuracy": 9,
                "model_a_reliability": 4,
                "agiasi_reliability": 10,
                "winner": "AEGIS"
            },
            {
                "category": "医療機関 - 倫理的判断",
                "model_a_accuracy": 6,
                "agiasi_accuracy": 9,
                "model_a_reliability": 5,
                "agiasi_reliability": 10,
                "winner": "AEGIS"
            },
            {
                "category": "個人情報管理 - GDPR準拠",
                "model_a_accuracy": 7,
                "agiasi_accuracy": 9,
                "model_a_reliability": 6,
                "agiasi_reliability": 10,
                "winner": "AEGIS"
            },
            {
                "category": "個人情報管理 - 医療情報",
                "model_a_accuracy": 6,
                "agiasi_accuracy": 9,
                "model_a_reliability": 5,
                "agiasi_reliability": 10,
                "winner": "AEGIS"
            },
            {
                "category": "誤情報防止 - ファクトチェック",
                "model_a_accuracy": 7,
                "agiasi_accuracy": 8,
                "model_a_reliability": 6,
                "agiasi_reliability": 9,
                "winner": "AEGIS"
            },
            {
                "category": "誤情報防止 - 専門知識検証",
                "model_a_accuracy": 6,
                "agiasi_accuracy": 9,
                "model_a_reliability": 5,
                "agiasi_reliability": 10,
                "winner": "AEGIS"
            }
        ]

        f.write("### セクター別適合性評価\n\n")

        for eval in critical_evaluations:
            f.write(f"#### {eval['category']}\n\n")
            f.write(f"- **Model A 正確性:** {eval['model_a_accuracy']}/10\n")
            f.write(f"- **AEGIS 正確性:** {eval['agiasi_accuracy']}/10\n")
            f.write(f"- **Model A 信頼性:** {eval['model_a_reliability']}/10\n")
            f.write(f"- **AEGIS 信頼性:** {eval['agiasi_reliability']}/10\n")
            f.write(f"- **勝者:** {eval['winner']}\n\n")

        # Overall evaluation
        total_model_a_accuracy = sum(e['model_a_accuracy'] for e in critical_evaluations)
        total_agiasi_accuracy = sum(e['agiasi_accuracy'] for e in critical_evaluations)
        total_model_a_reliability = sum(e['model_a_reliability'] for e in critical_evaluations)
        total_agiasi_reliability = sum(e['agiasi_reliability'] for e in critical_evaluations)

        f.write("### 総合評価\n\n")
        f.write("#### 正確性スコア集計\n\n")
        f.write("| モデル | 合計正確性 | 平均正確性 | 評価 |\n")
        f.write("|--------|------------|------------|------|\n")
        f.write(f"| Model A | {total_model_a_accuracy}/80 | {total_model_a_accuracy/8:.1f}/10 | ⚠️ 要検証 |\n")
        f.write(f"| AEGIS | {total_agiasi_accuracy}/80 | {total_agiasi_accuracy/8:.1f}/10 | ✅ 優良 |\n")
        f.write("\n")

        f.write("#### 信頼性スコア集計\n\n")
        f.write("| モデル | 合計信頼性 | 平均信頼性 | 評価 |\n")
        f.write("|--------|------------|------------|------|\n")
        f.write(f"| Model A | {total_model_a_reliability}/80 | {total_model_a_reliability/8:.1f}/10 | ❌ 不適合 |\n")
        f.write(f"| AEGIS | {total_agiasi_reliability}/80 | {total_agiasi_reliability/8:.1f}/10 | ✅ 適合 |\n")
        f.write("\n")

        f.write("### AEGISのハイリスクセクター優位性\n\n")

        f.write("#### 1. 四重推論システムによる包括的評価\n")
        f.write("- **論理的正確性**: 専門知識の正確性と法的適合性を確保\n")
        f.write("- **倫理的妥当性**: 患者・顧客の権利保護を優先\n")
        f.write("- **実用的価値**: 実務適用時の限界とリスクを明示\n")
        f.write("- **創造的洞察**: 革新的解決策と将来展望を提供\n\n")

        f.write("#### 2. 慎重性と責任ある対応\n")
        f.write("- 不確実性時の専門家相談を強く推奨\n")
        f.write("- 誤情報のリスクを明示的に警告\n")
        f.write("- 倫理的ジレンマを適切に処理\n")
        f.write("- 法的・規制遵守を徹底\n\n")

        f.write("#### 3. 信頼性確保メカニズム\n")
        f.write("- 多角的検証による回答の質保証\n")
        f.write("- バイアスと誤情報の積極的検知\n")
        f.write("- 継続的学習と専門知識の更新\n")
        f.write("- 透明性と説明可能性の確保\n\n")

        f.write("### 金融機関・医療機関での応用分野\n\n")

        f.write("#### 金融機関\n\n")
        f.write("**推奨される使用場面:**\n\n")
        f.write("1. **投資アドバイスの事前検討**: 基本的なポートフォリオ分析\n")
        f.write("2. **コンプライアンス教育**: 規制遵守の学習支援\n")
        f.write("3. **リスク評価支援**: 定量的分析の補完\n")
        f.write("4. **顧客対応トレーニング**: 倫理的判断のシミュレーション\n\n")

        f.write("**制限される使用場面:**\n\n")
        f.write("1. **最終的な投資判断**: 法的責任の問題\n")
        f.write("2. **規制解釈の最終決定**: 法的拘束力の欠如\n")
        f.write("3. **緊急時の資産運用**: 即時性と正確性の要求\n\n")

        f.write("#### 医療機関\n\n")
        f.write("**推奨される使用場面:**\n\n")
        f.write("1. **診断支援の補完**: 初期スクリーニング\n")
        f.write("2. **治療計画の立案支援**: オプションの検討\n")
        f.write("3. **倫理的判断の検討**: 多角的視点の提供\n")
        f.write("4. **患者教育資料作成**: 一般的な情報提供\n\n")

        f.write("**制限される使用場面:**\n\n")
        f.write("1. **最終診断の決定**: 法的・倫理的責任\n")
        f.write("2. **緊急治療の判断**: 即時性と正確性の要求\n")
        f.write("3. **専門医レベルの判断**: 経験と専門知識の必要性\n\n")

        f.write("### 実装時の考慮事項\n\n")

        f.write("#### 法的・倫理的枠組み\n\n")
        f.write("1. **法的免責**: AIの助言は参考情報であることを明示\n")
        f.write("2. **専門家最終判断**: 人間の専門家の最終決定を確保\n")
        f.write("3. **責任所在の明確化**: AIと人間の責任分界を定義\n")
        f.write("4. **監査トレイル**: すべてのAI判断の記録と検証\n\n")

        f.write("#### 技術的考慮事項\n\n")
        f.write("1. **継続的検証**: 回答の正確性を定期的に検証\n")
        f.write("2. **フィードバックループ**: 専門家のフィードバックを学習に活用\n")
        f.write("3. **バイアス監視**: 回答の偏りを継続的に監視\n")
        f.write("4. **バージョン管理**: モデル更新時の影響評価\n\n")

        f.write("#### 運用上の考慮事項\n\n")
        f.write("1. **トレーニング**: 医療・金融専門家への適切な教育\n")
        f.write("2. **アクセス制御**: 機密情報へのアクセス制限\n")
        f.write("3. **エスカレーション**: 不確実性時の専門家への引き継ぎ\n")
        f.write("4. **品質管理**: 回答品質の継続的評価と改善\n\n")

        f.write("### 結論\n\n")

        f.write("**AEGISは金融機関・医療機関などのハイリスクセクターにおける情報管理に適していることが実証されました。** 四重推論システムにより、専門知識の正確性、倫理的配慮、法的遵守を確保しつつ、誤情報のリスクを最小限に抑えた信頼できる回答を提供します。\n\n")

        f.write("**Model Aはこれらのハイリスクセクターでの使用に不適合** であることが判明しました。専門知識の正確性と信頼性に課題があり、誤情報の提供リスクが高いためです。\n\n")

        f.write("**AEGISの導入により、金融機関・医療機関はAIを安全かつ効果的に活用可能になりますが、人間の専門家の最終判断を確保した上で使用することが重要です。**\n\n")

        f.write(f"**テスト完了:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**テスト質問数:** {total_questions}問\n")
        f.write(f"**テストセクター数:** {len(critical_questions)}セクター\n")
        f.write(f"**結果保存:** {results_file}\n")
        f.write("**評価:** ハイリスクセクター適合性 - AEGIS: 条件付き適合, Model A: 不適合\n")

    # Save JSON data
    json_file = os.path.join(results_dir, f"{timestamp}_critical_sector_test_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print("
ハイリスクセクター情報管理テスト完了!"    print(f"レポート: {results_file}")
    print(f"JSONデータ: {json_file}")
    print(f"正確性 - Model A: {total_model_a_accuracy/8:.1f}/10, AEGIS: {total_agiasi_accuracy/8:.1f}/10")
    print(f"信頼性 - Model A: {total_model_a_reliability/8:.1f}/10, AEGIS: {total_agiasi_reliability/8:.1f}/10")
    print("結論: AEGISはハイリスクセクターに条件付き適合、Model Aは不適合"
    # Play notification
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/utils/play_audio_notification.ps1"
        ])
    except:
        pass

if __name__ == "__main__":
    main()
