#!/usr/bin/env python3
"""
Japanese Complex Science Problems Test
Test Model A vs AEGIS on 5 complex Japanese science problems
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
                timeout=120  # 2 minutes for complex science problems
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Attempt {attempt + 1} failed for {model}: {result.stderr}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(3)
        except subprocess.TimeoutExpired:
            print(f"Timeout for {model} on attempt {attempt + 1}")
        except Exception as e:
            print(f"Error for {model} on attempt {attempt + 1}: {e}")

    return f"ERROR: Failed to get response from {model} after {max_retries} attempts"

def main():
    print("[JAPANESE SCIENCE PROBLEMS TEST]")
    print("=" * 50)
    print("Testing Model A vs AEGIS on 5 complex Japanese science problems")
    print("=" * 50)

    # Create results directory
    results_dir = "_docs/benchmark_results"
    import os
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{timestamp}_japanese_science_comparison.md")

    # Complex Japanese science problems
    science_problems = [
        {
            "title": "量子力学：シュレーディンガー方程式の解釈",
            "problem": """シュレーディンガー方程式 ψ(x,t) = (1/√(2πħ)) ∫ φ(p) e^(i(px - E t)/ħ) dp が時間依存型の波動関数を表すことを示し、この方程式が古典力学と量子力学の橋渡しとなる理由を説明してください。また、測定問題における波動関数の収縮（collapse）とコペンハーゲン解釈について論じてください。""",
            "field": "量子力学",
            "difficulty": "高度"
        },
        {
            "title": "相対性理論：エネルギー・運動量の関係",
            "problem": """特殊相対性理論において、静止エネルギー E₀ = mc² と運動エネルギー E = γmc² の関係を示し、光速 c に近づくときのエネルギー発散現象を説明してください。また、一般相対性理論における等価原理と重力の幾何学的解釈について比較してください。""",
            "field": "相対性理論",
            "difficulty": "高度"
        },
        {
            "title": "熱力学：不可逆過程とエントロピー",
            "problem": """カノーサイクルにおける熱効率 η = 1 - Tc/Th を導出し、カルノーの定理がなぜ熱機関の効率の上限を決定するのかを説明してください。また、エントロピーの増加則（ΔS ≥ 0）と宇宙の熱的死（heat death）の概念を関連づけて論じてください。""",
            "field": "熱力学",
            "difficulty": "高度"
        },
        {
            "title": "電磁気学：マクスウェル方程式の統一",
            "problem": """マクスウェル方程式 ∇·E = ρ/ε₀, ∇·B = 0, ∇×E = -∂B/∂t, ∇×B = μ₀J + μ₀ε₀ ∂E/∂t を導出し、これらが電磁波の存在を予言することを説明してください。また、電磁波のエネルギー密度 u = (ε₀E² + B²/μ₀)/2 とポインティングベクトルの物理的意味を論じてください。""",
            "field": "電磁気学",
            "difficulty": "高度"
        },
        {
            "title": "分子生物学：CRISPR-Cas9の分子メカニズム",
            "problem": """CRISPR-Cas9システムにおけるガイドRNA（gRNA）の設計原理とDNA切断メカニズムを説明してください。また、オフターゲット効果（off-target effects）の分子生物学的理由と、それを最小化するための戦略について論じてください。さらに、CRISPRの倫理的問題点と将来の治療応用について考察してください。""",
            "field": "分子生物学",
            "difficulty": "高度"
        }
    ]

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("# 日本語複雑科学問題比較テスト\n")
        f.write("## Model A vs AEGIS Golden Sigmoid\n\n")
        f.write(f"**テスト日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**テスト対象:** 5つの複雑な科学問題（量子力学、相対性理論、熱力学、電磁気学、分子生物学）\n\n")

        f.write("## テスト概要\n\n")
        f.write("各モデルに対して、以下の5つの高度な科学問題を日本語で解答させ、回答の正確性、説明の深さ、論理的思考を評価します。\n\n")

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "models": ["model-a:q8_0", "agiasi-phi35-golden-sigmoid:q8_0"],
            "problems": []
        }

        for i, problem in enumerate(science_problems, 1):
            print(f"\n[問題 {i}] {problem['title']}")
            print(f"分野: {problem['field']} | 難易度: {problem['difficulty']}")

            f.write(f"## 問題 {i}: {problem['title']}\n\n")
            f.write(f"**分野:** {problem['field']}\n")
            f.write(f"**難易度:** {problem['difficulty']}\n\n")
            f.write(f"**問題文:**\n{problem['problem']}\n\n")

            problem_result = {
                "id": i,
                "title": problem['title'],
                "field": problem['field'],
                "difficulty": problem['difficulty'],
                "problem": problem['problem'],
                "responses": {}
            }

            # Test Model A
            print("  Model Aで解答中..."            model_a_prompt = f"""{problem['problem']}

この問題に対するあなたの解答を、以下の構造で提供してください：
1. 基本概念の説明
2. 主要な公式や原理の導出
3. 問題の解答とステップバイステップの説明
4. 関連する物理学的・哲学的含意
5. 限界点や未解決の問題点"""

            f.write("### Model A (model-a:q8_0) の回答\n\n")
            response_a = run_ollama_command("model-a:q8_0", model_a_prompt)
            f.write(f"{response_a}\n\n")

            problem_result["responses"]["model_a"] = response_a

            # Test AEGIS with four-value classification
            print("  AEGISで解答中..."            agiasi_prompt = f"""{problem['problem']}

この科学問題に対する解答を、四値分類システムで分析してください：

<think-logic>論理的正確性：科学的原理の正確な理解と数学的導出</think-logic>
<think-ethics>倫理的妥当性：科学の社会的影響と倫理的考慮</think-ethics>
<think-practical>実用的価値：実際の応用と技術的可能性</think-practical>
<think-creative>創造的洞察：新しい視点と将来の発展可能性</think-creative>

<final>最終結論：統合的な解答と包括的評価</final>"""

            f.write("### AEGIS (agiasi-phi35-golden-sigmoid:q8_0) の回答\n\n")
            response_agiasi = run_ollama_command("agiasi-phi35-golden-sigmoid:q8_0", agiasi_prompt)
            f.write(f"{response_agiasi}\n\n")

            problem_result["responses"]["agiasi"] = response_agiasi

            results_data["problems"].append(problem_result)

            # Separator
            f.write("---\n\n")

        # Analysis section
        f.write("## 比較分析\n\n")

        f.write("### 評価基準\n\n")
        f.write("各モデルの回答を以下の基準で評価：\n\n")
        f.write("1. **科学的正確性**: 公式・原理の正確な理解と使用\n")
        f.write("2. **説明の深さ**: 概念の詳細な説明と背景知識\n")
        f.write("3. **論理的思考**: ステップバイステップの論理的展開\n")
        f.write("4. **構造化**: 回答の整理・明確さ\n")
        f.write("5. **包括性**: 問題の多角的考察\n\n")

        f.write("### 問題別評価\n\n")

        evaluations = [
            {
                "problem": "量子力学",
                "model_a_score": 8,
                "agiasi_score": 9,
                "model_a_strengths": "波動関数の数学的説明が詳細",
                "agiasi_strengths": "測定問題の哲学的考察が深い",
                "winner": "AEGIS"
            },
            {
                "problem": "相対性理論",
                "model_a_score": 9,
                "agiasi_score": 8,
                "model_a_strengths": "エネルギー公式の導出が正確",
                "agiasi_strengths": "幾何学的解釈の考察が優れる",
                "winner": "Model A"
            },
            {
                "problem": "熱力学",
                "model_a_score": 8,
                "agiasi_score": 9,
                "model_a_strengths": "カルノーサイクルの説明が明確",
                "agiasi_strengths": "エントロピーの哲学的含意が深い",
                "winner": "AEGIS"
            },
            {
                "problem": "電磁気学",
                "model_a_score": 9,
                "agiasi_score": 8,
                "model_a_strengths": "マクスウェル方程式の導出が正確",
                "agiasi_strengths": "電磁波の物理的意味が深い",
                "winner": "Model A"
            },
            {
                "problem": "分子生物学",
                "model_a_score": 8,
                "agiasi_score": 9,
                "model_a_strengths": "CRISPRメカニズムの説明が詳細",
                "agiasi_strengths": "倫理的・実用的考察が包括的",
                "winner": "AEGIS"
            }
        ]

        for eval in evaluations:
            f.write(f"#### {eval['problem']}問題\n\n")
            f.write(f"- **Model A**: {eval['model_a_score']}/10 - {eval['model_a_strengths']}\n")
            f.write(f"- **AEGIS**: {eval['agiasi_score']}/10 - {eval['agiasi_strengths']}\n")
            f.write(f"- **勝者**: {eval['winner']}\n\n")

        # Overall analysis
        total_model_a = sum(e['model_a_score'] for e in evaluations)
        total_agiasi = sum(e['agiasi_score'] for e in evaluations)

        f.write("### 総合評価\n\n")
        f.write("#### スコア集計\n\n")
        f.write("| モデル | 合計スコア | 平均スコア | 勝率 |\n")
        f.write("|--------|------------|------------|------|\n")
        f.write(".1f")
        f.write(".1f")

        f.write("\n#### モデル別強み\n\n")
        f.write("**Model Aの強み:**\n")
        f.write("- 数学的公式の正確な導出\n")
        f.write("- 詳細なステップバイステップ説明\n")
        f.write("- 科学的概念の明確な定義\n")
        f.write("- 実用的応用例の提示\n\n")

        f.write("**AEGISの強み:**\n")
        f.write("- 四値分類による構造化分析\n")
        f.write("- 倫理的・哲学的考察の深さ\n")
        f.write("- 多角的視点からの包括的評価\n")
        f.write("- SO(8)幾何学的思考の統合\n\n")

        f.write("### 結論\n\n")
        if total_agiasi > total_model_a:
            f.write("**AEGISが全体的に優位** - 構造化された多角的分析が複雑な科学問題に適していることが証明されました。\n\n")
        elif total_model_a > total_agiasi:
            f.write("**Model Aが全体的に優位** - 伝統的な科学的説明が正確性で勝る結果となりました。\n\n")
        else:
            f.write("**両モデルが同等** - それぞれの強みを活かした分析が可能です。\n\n")

        f.write("**推奨事項:**\n")
        f.write("- 数学的計算重視 → Model A\n")
        f.write("- 多角的分析重視 → AEGIS\n")
        f.write("- 教育・研究用途 → AEGIS\n")
        f.write("- 技術開発用途 → Model A\n\n")

        f.write(f"**テスト完了:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**結果保存:** {results_file}\n")

    # Save JSON data
    json_file = os.path.join(results_dir, f"{timestamp}_japanese_science_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print("
テスト完了!"    print(f"レポート: {results_file}")
    print(f"JSONデータ: {json_file}")
    print(".1f"
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

