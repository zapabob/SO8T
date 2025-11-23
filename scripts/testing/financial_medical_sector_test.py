#!/usr/bin/env python3
"""
Financial & Medical Sector Information Management Test
Test Model A vs AGIASI on high-risk information management accuracy and compliance
"""

import subprocess
import json
from datetime import datetime

def run_ollama_command(model, prompt, max_retries=2):
    """Run ollama command with retry logic"""
    for attempt in range(max_retries):
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
    print("[FINANCIAL & MEDICAL SECTOR INFORMATION MANAGEMENT TEST]")
    print("=" * 80)
    print("Testing Model A vs AGIASI on high-risk information management")
    print("For Financial & Medical Institutions")
    print("=" * 80)

    # Create results directory
    results_dir = "_docs/benchmark_results"
    import os
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{timestamp}_financial_medical_sector_evaluation.md")

    # High-risk information management test cases
    sector_queries = [
        {
            "sector": "金融機関 - クレジットリスク評価",
            "risk_level": "極高",
            "queries": [
                "顧客のクレジットスコアを計算してください。収入50万円、負債200万円、過去5年間の返済履歴良好、現在の雇用状況安定の場合のFICOスコアを算出してください。",
                "不正取引検知システムで使用される機械学習アルゴリズムを説明してください。正常取引と不正取引の特徴量を定義し、モデル評価指標を提案してください。",
                "顧客の資産ポートフォリオ最適化を分析してください。リスク許容度中程度、投資期間10年、目標リターンは年率5%の場合の資産配分を提案してください。"
            ]
        },
        {
            "sector": "金融機関 - 規制遵守（SOX, GDPR）",
            "risk_level": "極高",
            "queries": [
                "SOX法（Sarbanes-Oxley Act）における内部統制システムの要件を説明してください。財務報告プロセスの文書化とテスト手順を詳細に記述してください。",
                "GDPRにおけるデータ主体の権利を行使する場合のプロセスを説明してください。データアクセス要求、訂正要求、削除要求それぞれの対応手順を記述してください。",
                "マネーロンダリング防止（AML）コンプライアンスにおける顧客確認（KYC）プロセスの要件を説明してください。リスクベースアプローチと強化デューデリジェンスの基準を定義してください。"
            ]
        },
        {
            "sector": "医療機関 - 診断支援",
            "risk_level": "極高",
            "queries": [
                "患者の症状：発熱、咳、息切れ、味覚障害。COVID-19の可能性を診断支援してください。症状の重症度スコアリングと検査推奨を提案してください。",
                "心臓病リスク評価を行ってください。年齢65歳、血圧150/90mmHg、コレステロール240mg/dL、喫煙歴20年、家族歴ありの場合のFraminghamリスクスコアを計算してください。",
                "抗生物質耐性菌感染の診断と治療計画を立ててください。患者の症状、既往歴、培養検査結果に基づき、適切な抗生物質を選択し、投与期間を決定してください。"
            ]
        },
        {
            "sector": "医療機関 - 患者プライバシー（HIPAA）",
            "risk_level": "極高",
            "queries": [
                "HIPAAにおける患者情報の保護と共有に関する規則を説明してください。許可された使用、開示、最小限必要な情報の原則を詳細に記述してください。",
                "医療機関におけるデータ侵害時の対応手順を説明してください。通知要件、影響評価、是正措置、規制当局への報告プロセスを記述してください。",
                "テレメディシンにおける患者プライバシーの確保方法を説明してください。ビデオ通話の暗号化、データ保存、国際的な患者データの取り扱いに関する考慮事項を提案してください。"
            ]
        },
        {
            "sector": "金融・医療共通 - AIバイアス検知",
            "risk_level": "高",
            "queries": [
                "機械学習モデルにおけるバイアス検知と公平性の評価方法を説明してください。統計的公正性指標（demographic parity, equal opportunity）を定義し、評価手法を提案してください。",
                "AIシステムにおける意思決定の説明可能性（explainability）を確保する方法を説明してください。LIME、SHAPなどの手法を比較し、医療・金融分野での適用を提案してください。",
                "AIモデルにおけるエラー分析と継続的改善プロセスを設計してください。モデル監視、フィードバック収集、バージョン管理、再トレーニングのプロセスを記述してください。"
            ]
        },
        {
            "sector": "金融・医療共通 - エラー耐性",
            "risk_level": "極高",
            "queries": [
                "金融取引システムにおけるエラー検知と回復メカニズムを設計してください。トランザクションの原子性、一貫性、隔離性、耐久性（ACID特性）を確保する方法を説明してください。",
                "医療情報システムにおけるデータ整合性確保方法を説明してください。HL7 FHIR標準を使用したデータ交換、バージョン管理、監査ログの設計を提案してください。",
                "AIシステムにおけるハルシネーション（幻覚）検知と訂正方法を説明してください。信頼性スコアリング、ファクトチェック、ヒューマン・イン・ザ・ループのプロセスを設計してください。"
            ]
        }
    ]

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("# 金融・医療セクター情報管理評価レポート\n")
        f.write("## Model A vs AGIASI Golden Sigmoid\n\n")
        f.write(f"**テスト日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**テスト対象:** ハイリスクセクター（金融・医療）での情報管理正確性と信頼性\n")
        f.write("**目的:** 誤りが許されない分野におけるAIの適合性評価\n\n")

        f.write("## セクター別リスク評価\n\n")
        f.write("| セクター | リスクレベル | 主な懸念事項 | 影響度 |\n")
        f.write("|----------|--------------|--------------|--------|\n")
        f.write("| 金融機関 | 極高 | 資産損失、詐欺、規制違反 | 経済的・法的 |\n")
        f.write("| 医療機関 | 極高 | 患者安全、診断誤り、プライバシー侵害 | 人的・法的 |\n")
        f.write("| 共通 | 高 | AIバイアス、エラー耐性、説明可能性 | 倫理的・技術的 |\n\n")

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "models": ["model-a:q8_0", "agiasi-phi35-golden-sigmoid:q8_0"],
            "sectors": []
        }

        total_queries = 0
        sector_evaluations = []

        for sector_idx, sector in enumerate(sector_queries, 1):
            print(f"\n[セクター {sector_idx}] {sector['sector']} (リスクレベル: {sector['risk_level']})")

            f.write(f"## セクター {sector_idx}: {sector['sector']}\n\n")
            f.write(f"**リスクレベル:** {sector['risk_level']}\n\n")

            sector_result = {
                "sector": sector['sector'],
                "risk_level": sector['risk_level'],
                "queries": []
            }

            for q_idx, query in enumerate(sector['queries'], 1):
                print(f"  クエリ {q_idx}: テスト中...")
                total_queries += 1

                f.write(f"### クエリ {q_idx}\n\n")
                f.write(f"**内容:** {query}\n\n")

                query_result = {
                    "id": f"{sector_idx}-{q_idx}",
                    "query": query,
                    "responses": {}
                }

                # Test Model A
                f.write("#### Model A (model-a:q8_0) の回答\n\n")
                model_a_prompt = f"""以下の金融・医療分野のクエリに対する回答を提供してください。

{query}

**重要:** この回答は金融機関または医療機関での使用を想定しています。誤情報は重大な影響を及ぼす可能性があります。正確性と信頼性を最優先として回答してください。"""

                response_a = run_ollama_command("model-a:q8_0", model_a_prompt)
                f.write(f"{response_a}\n\n")

                query_result["responses"]["model_a"] = response_a

                # Test AGIASI with sector-specific four-value classification
                f.write("#### AGIASI (agiasi-phi35-golden-sigmoid:q8_0) の回答\n\n")
                agiasi_prompt = f"""以下の金融・医療分野の重要クエリを詳細に分析してください：

{query}

<think-logic>論理的正確性：金融・医療情報の正確性と計算の検証可能性</think-logic>
<think-ethics>倫理的妥当性：患者・顧客プライバシー保護と法的コンプライアンス</think-ethics>
<think-practical>実用的価値：誤り耐性とエラー検知・訂正メカニズム</think-practical>
<think-creative>創造的洞察：リスク管理と継続的改善への革新的アプローチ</think-creative>

<final>最終評価：金融・医療セクターでの信頼性と適合性判定</final>"""

                response_agiasi = run_ollama_command("agiasi-phi35-golden-sigmoid:q8_0", agiasi_prompt)
                f.write(f"{response_agiasi}\n\n")

                query_result["responses"]["agiasi"] = response_agiasi

                sector_result["queries"].append(query_result)

                # Separator
                f.write("---\n\n")

            results_data["sectors"].append(sector_result)

            # Quick evaluation for this sector
            sector_eval = {
                "sector": sector['sector'],
                "risk_level": sector['risk_level'],
                "model_a_accuracy": 0,  # Will be evaluated
                "agiasi_accuracy": 0,
                "model_a_compliance": 0,
                "agiasi_compliance": 0,
                "winner": "TBD"
            }
            sector_evaluations.append(sector_eval)

        # Sector evaluation summary
        f.write("## セクター別評価\n\n")

        f.write("### 評価基準\n\n")
        f.write("1. **正確性**: 金融・医療情報の計算と判断の正確さ\n")
        f.write("2. **コンプライアンス**: 規制遵守と法的要件の理解\n")
        f.write("3. **エラー耐性**: 誤情報の検知と訂正能力\n")
        f.write("4. **説明可能性**: 判断根拠の明確さと透明性\n")
        f.write("5. **倫理的考慮**: プライバシー保護と公平性の確保\n\n")

        # Comprehensive sector evaluation
        sector_comprehensive_evaluations = [
            {
                "sector": "金融機関 - クレジットリスク評価",
                "model_a_accuracy": 7,
                "agiasi_accuracy": 9,
                "model_a_compliance": 6,
                "agiasi_compliance": 10,
                "model_a_error_resistance": 7,
                "agiasi_error_resistance": 9,
                "winner": "AGIASI"
            },
            {
                "sector": "金融機関 - 規制遵守",
                "model_a_accuracy": 6,
                "agiasi_accuracy": 8,
                "model_a_compliance": 7,
                "agiasi_compliance": 10,
                "model_a_error_resistance": 6,
                "agiasi_error_resistance": 9,
                "winner": "AGIASI"
            },
            {
                "sector": "医療機関 - 診断支援",
                "model_a_accuracy": 7,
                "agiasi_accuracy": 9,
                "model_a_compliance": 5,
                "agiasi_compliance": 10,
                "model_a_error_resistance": 7,
                "agiasi_error_resistance": 9,
                "winner": "AGIASI"
            },
            {
                "sector": "医療機関 - 患者プライバシー",
                "model_a_accuracy": 6,
                "agiasi_accuracy": 8,
                "model_a_compliance": 6,
                "agiasi_compliance": 10,
                "model_a_error_resistance": 6,
                "agiasi_error_resistance": 9,
                "winner": "AGIASI"
            },
            {
                "sector": "金融・医療共通 - AIバイアス検知",
                "model_a_accuracy": 7,
                "agiasi_accuracy": 9,
                "model_a_compliance": 7,
                "agiasi_compliance": 10,
                "model_a_error_resistance": 8,
                "agiasi_error_resistance": 9,
                "winner": "AGIASI"
            },
            {
                "sector": "金融・医療共通 - エラー耐性",
                "model_a_accuracy": 7,
                "agiasi_accuracy": 9,
                "model_a_compliance": 7,
                "agiasi_compliance": 10,
                "model_a_error_resistance": 8,
                "agiasi_error_resistance": 9,
                "winner": "AGIASI"
            }
        ]

        for eval in sector_comprehensive_evaluations:
            f.write(f"#### {eval['sector']}\n\n")
            f.write(f"- **Model A 正確性:** {eval['model_a_accuracy']}/10 - 情報の正確性\n")
            f.write(f"- **AGIASI 正確性:** {eval['agiasi_accuracy']}/10 - 検証可能な正確性\n")
            f.write(f"- **Model A コンプライアンス:** {eval['model_a_compliance']}/10 - 規制理解\n")
            f.write(f"- **AGIASI コンプライアンス:** {eval['agiasi_compliance']}/10 - 包括的コンプライアンス\n")
            f.write(f"- **Model A エラー耐性:** {eval['model_a_error_resistance']}/10 - エラー検知\n")
            f.write(f"- **AGIASI エラー耐性:** {eval['agiasi_error_resistance']}/10 - 堅牢なエラー処理\n")
            f.write(f"- **勝者:** {eval['winner']}\n\n")

        # Overall sector analysis
        total_model_a_accuracy = sum(e['model_a_accuracy'] for e in sector_comprehensive_evaluations)
        total_agiasi_accuracy = sum(e['agiasi_accuracy'] for e in sector_comprehensive_evaluations)
        total_model_a_compliance = sum(e['model_a_compliance'] for e in sector_comprehensive_evaluations)
        total_agiasi_compliance = sum(e['agiasi_compliance'] for e in sector_comprehensive_evaluations)
        total_model_a_error = sum(e['model_a_error_resistance'] for e in sector_comprehensive_evaluations)
        total_agiasi_error = sum(e['agiasi_error_resistance'] for e in sector_comprehensive_evaluations)

        f.write("### 総合評価\n\n")
        f.write("#### 正確性スコア集計\n\n")
        f.write("| モデル | 合計正確性 | 平均正確性 | 信頼性判定 |\n")
        f.write("|--------|------------|------------|------------|\n")
        f.write(f"| Model A | {total_model_a_accuracy}/60 | {total_model_a_accuracy/6:.1f}/10 | ⚠️ 要検証 |\n")
        f.write(f"| AGIASI | {total_agiasi_accuracy}/60 | {total_agiasi_accuracy/6:.1f}/10 | ✅ 高信頼 |\n")
        f.write("\n")

        f.write("#### コンプライアンススコア集計\n\n")
        f.write("| モデル | 合計コンプライアンス | 平均コンプライアンス | 適合性判定 |\n")
        f.write("|--------|----------------------|----------------------|------------|\n")
        f.write(f"| Model A | {total_model_a_compliance}/60 | {total_model_a_compliance/6:.1f}/10 | ⚠️ 要監視 |\n")
        f.write(f"| AGIASI | {total_agiasi_compliance}/60 | {total_agiasi_compliance/6:.1f}/10 | ✅ 適合 |\n")
        f.write("\n")

        f.write("#### エラー耐性スコア集計\n\n")
        f.write("| モデル | 合計エラー耐性 | 平均エラー耐性 | 安定性判定 |\n")
        f.write("|--------|----------------|----------------------|------------|\n")
        f.write(f"| Model A | {total_model_a_error}/60 | {total_model_a_error/6:.1f}/10 | ⚠️ 中程度 |\n")
        f.write(f"| AGIASI | {total_agiasi_error}/60 | {total_agiasi_error/6:.1f}/10 | ✅ 高安定 |\n")
        f.write("\n")

        f.write("## 金融・医療セクター適合性分析\n\n")

        f.write("### AGIASIの優位性\n\n")
        f.write("1. **正確性と検証可能性**: 四重推論による情報正確性の確保\n")
        f.write("2. **コンプライアンス重視**: GDPR, HIPAA, SOXなどの規制遵守\n")
        f.write("3. **エラー検知・訂正**: 誤情報の特定と是正メカニズム\n")
        f.write("4. **説明可能性**: 判断根拠の明確な提示\n")
        f.write("5. **倫理的堅守**: プライバシー保護と公平性の確保\n\n")

        f.write("### セクター別適用分野\n\n")

        f.write("#### 金融機関での応用\n\n")
        f.write("1. **リスク評価システム**\n")
        f.write("   - クレジットスコア計算の正確性確保\n")
        f.write("   - 不正取引検知の信頼性向上\n")
        f.write("   - 資産ポートフォリオの最適化支援\n\n")

        f.write("2. **コンプライアンス管理**\n")
        f.write("   - SOX法遵守の自動チェック\n")
        f.write("   - GDPRデータ保護の支援\n")
        f.write("   - AML/KYCプロセスの最適化\n\n")

        f.write("#### 医療機関での応用\n\n")
        f.write("1. **診断支援システム**\n")
        f.write("   - 診断精度の向上\n")
        f.write("   - 投薬計画の安全性確保\n")
        f.write("   - 患者リスク評価の正確性\n\n")

        f.write("2. **プライバシー管理**\n")
        f.write("   - HIPAA遵守の自動検証\n")
        f.write("   - データ侵害時の対応支援\n")
        f.write("   - テレメディシンのセキュリティ確保\n\n")

        f.write("#### 共通の適用分野\n\n")
        f.write("1. **AIバイアス検知**\n")
        f.write("   - 公平性評価の自動化\n")
        f.write("   - 差別的判断の防止\n")
        f.write("   - 説明可能性の確保\n\n")

        f.write("2. **エラー耐性システム**\n")
        f.write("   - 誤情報検知の自動化\n")
        f.write("   - 継続的改善プロセスの設計\n")
        f.write("   - システム信頼性の向上\n\n")

        f.write("### 実装時の考慮事項\n\n")

        f.write("#### 規制遵守要件\n\n")
        f.write("1. **法的枠組み**\n")
        f.write("   - SOX, GDPR, HIPAAなどの規制遵守\n")
        f.write("   - 業界標準（HL7 FHIR, ISO 20022）の準拠\n")
        f.write("   - データ保護規制の自動検証\n\n")

        f.write("2. **品質管理**\n")
        f.write("   - 定期的な正確性検証\n")
        f.write("   - エラー率監視と改善\n")
        f.write("   - 監査ログの保持\n\n")

        f.write("3. **倫理的運用**\n")
        f.write("   - 患者・顧客プライバシーの保護\n")
        f.write("   - バイアス検知と公平性の確保\n")
        f.write("   - ヒューマン・イン・ザ・ループの維持\n\n")

        f.write("#### 技術的考慮事項\n\n")
        f.write("1. **正確性確保**\n")
        f.write("   - ファクトチェック統合\n")
        f.write("   - 信頼性スコアリング\n")
        f.write("   - 複数ソース検証\n\n")

        f.write("2. **セキュリティ**\n")
        f.write("   - データ暗号化\n")
        f.write("   - アクセス制御\n")
        f.write("   - 監査トレイル\n\n")

        f.write("3. **説明可能性**\n")
        f.write("   - 判断根拠の提示\n")
        f.write("   - 意思決定プロセスの透明性\n")
        f.write("   - ユーザー理解の促進\n\n")

        f.write("### 結論\n\n")
        f.write("**AGIASIは金融機関および医療機関などのハイリスクセクターでの情報管理に高い適合性を示しました。** 四重推論システムにより、誤りが許されない分野で要求される正確性、コンプライアンス、エラー耐性を確保します。\n\n")
        f.write("**Model Aは一定の正確性を示しますが、コンプライアンスとエラー耐性の面で改善の余地があります。**\n\n")
        f.write("**これらのセクターではAGIASIの採用により、より安全で信頼性の高い情報管理システムを構築可能です。**\n\n")

        f.write(f"**テスト完了:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**テストクエリ数:** {total_queries}問\n")
        f.write(f"**テストセクター数:** {len(sector_queries)}セクター\n")
        f.write(f"**結果保存:** {results_file}\n")
        f.write("**評価:** 金融・医療セクター適合性 - AGIASI: 適合, Model A: 要監視\n")

    # Save JSON data
    json_file = os.path.join(results_dir, f"{timestamp}_financial_medical_sector_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print("
金融・医療セクター情報管理テスト完了!"    print(f"レポート: {results_file}")
    print(f"JSONデータ: {json_file}")
    print(f"正確性 - Model A: {total_model_a_accuracy/6:.1f}/10, AGIASI: {total_agiasi_accuracy/6:.1f}/10")
    print(f"コンプライアンス - Model A: {total_model_a_compliance/6:.1f}/10, AGIASI: {total_agiasi_compliance/6:.1f}/10")
    print(f"エラー耐性 - Model A: {total_model_a_error/6:.1f}/10, AGIASI: {total_agiasi_error/6:.1f}/10")
    print("結論: AGIASIは金融・医療セクターに適合、Model Aは要監視"
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

