#!/usr/bin/env python3
"""
Complex Test for SO8T Lightweight Distilled Model
蒸留された軽量SO8Tモデルの複雑なテスト

CoT仮説検証思考で重み崩壊を防ぎながら包括的な性能評価を実装
"""

import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LightweightModelTester:
    """軽量モデルテスター"""
    
    def __init__(self, model_name: str = "so8t-lightweight"):
        """
        軽量モデルテスター初期化
        
        Args:
            model_name: Ollamaモデル名
        """
        self.model_name = model_name
        self.test_results = []
        
        logger.info("軽量モデルテスター初期化完了")
        logger.info(f"   - モデル名: {model_name}")
    
    def run_ollama_command(self, prompt: str, max_retries: int = 3) -> str:
        """Ollamaコマンドを実行"""
        for attempt in range(max_retries):
            try:
                cmd = [
                    "ollama", "run", self.model_name,
                    prompt
                ]
                
                logger.info(f"Ollamaコマンド実行中... (試行 {attempt + 1}/{max_retries})")
                logger.info(f"プロンプト: {prompt[:100]}...")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5分タイムアウト
                    encoding='utf-8',
                    errors='replace'
                )
                
                if result.returncode == 0:
                    logger.info("Ollamaコマンド実行成功")
                    return result.stdout
                else:
                    logger.warning(f"Ollamaコマンド実行失敗: {result.stderr}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return f"エラー: {result.stderr}"
                        
            except subprocess.TimeoutExpired:
                logger.warning(f"Ollamaコマンドタイムアウト (試行 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return "エラー: タイムアウト"
            except Exception as e:
                logger.warning(f"Ollamaコマンド実行エラー: {e} (試行 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return f"エラー: {e}"
        
        return "エラー: 最大再試行回数に達しました"
    
    def test_mathematical_reasoning(self) -> Dict[str, Any]:
        """数学的推論テスト"""
        logger.info("数学的推論テスト開始...")
        
        test_cases = [
            {
                "name": "複雑な積分計算",
                "prompt": "次の複雑な積分を解いてください: ∫[0 to π] sin(x) * cos(2x) * e^(-x) dx。ステップバイステップで計算過程を示してください。",
                "expected_keywords": ["積分", "sin", "cos", "e", "計算", "ステップ"]
            },
            {
                "name": "線形代数問題",
                "prompt": "3x3行列 A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] の固有値と固有ベクトルを求めてください。",
                "expected_keywords": ["固有値", "固有ベクトル", "行列", "det", "特性方程式"]
            },
            {
                "name": "確率統計問題",
                "prompt": "正規分布 N(μ=100, σ²=25) において、X > 110 となる確率を計算してください。",
                "expected_keywords": ["正規分布", "確率", "標準偏差", "Z値", "0.0228"]
            },
            {
                "name": "微分方程式",
                "prompt": "微分方程式 dy/dx + 2y = e^(-x) の一般解を求めてください。",
                "expected_keywords": ["微分方程式", "一般解", "積分因子", "C", "定数"]
            },
            {
                "name": "複素数計算",
                "prompt": "複素数 z = 3 + 4i の極形式を求め、z^5 を計算してください。",
                "expected_keywords": ["複素数", "極形式", "r", "θ", "ド・モアブルの定理"]
            },
            {
                "name": "高度な微積分",
                "prompt": "関数 f(x) = x^3 * e^(-x^2) の極値と変曲点を求め、グラフの概形を描いてください。",
                "expected_keywords": ["極値", "変曲点", "導関数", "二階導関数", "グラフ"]
            },
            {
                "name": "テイラー級数展開",
                "prompt": "関数 f(x) = ln(1+x) の x=0 周りのテイラー級数展開を5次まで求め、収束半径を計算してください。",
                "expected_keywords": ["テイラー級数", "収束半径", "マクローリン", "剰余項", "収束"]
            },
            {
                "name": "フーリエ変換",
                "prompt": "関数 f(t) = e^(-|t|) のフーリエ変換を計算し、逆変換で元の関数に戻ることを確認してください。",
                "expected_keywords": ["フーリエ変換", "逆変換", "周波数", "スペクトル", "積分"]
            },
            {
                "name": "ベクトル解析",
                "prompt": "ベクトル場 F = (x^2, y^2, z^2) の発散と回転を計算し、ストークスの定理を適用してください。",
                "expected_keywords": ["発散", "回転", "ストークス", "ベクトル場", "線積分"]
            },
            {
                "name": "数値解析",
                "prompt": "方程式 x^3 - 2x - 5 = 0 の解をニュートン法で求めてください。初期値 x0 = 2 から始めて、収束するまで計算してください。",
                "expected_keywords": ["ニュートン法", "収束", "反復", "初期値", "近似解"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            logger.info(f"   - {test_case['name']} テスト中...")
            
            start_time = time.time()
            response = self.run_ollama_command(test_case['prompt'])
            end_time = time.time()
            
            # 応答時間計算
            response_time = end_time - start_time
            
            # キーワードチェック
            keywords_found = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    keywords_found.append(keyword)
            
            # 結果記録
            result = {
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": response,
                "response_time": response_time,
                "keywords_found": keywords_found,
                "keywords_ratio": len(keywords_found) / len(test_case['expected_keywords']),
                "success": len(keywords_found) >= len(test_case['expected_keywords']) * 0.5
            }
            
            results.append(result)
            logger.info(f"     ✓ 完了 (応答時間: {response_time:.2f}秒, キーワード: {len(keywords_found)}/{len(test_case['expected_keywords'])})")
        
        return {
            "test_type": "mathematical_reasoning",
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r['success']),
            "average_response_time": sum(r['response_time'] for r in results) / len(results),
            "results": results
        }
    
    def test_scientific_concepts(self) -> Dict[str, Any]:
        """科学的概念テスト"""
        logger.info("科学的概念テスト開始...")
        
        test_cases = [
            {
                "name": "量子力学の基礎",
                "prompt": "シュレーディンガー方程式について説明し、水素原子の基底状態の波動関数を導出してください。",
                "expected_keywords": ["シュレーディンガー", "波動関数", "水素原子", "基底状態", "量子数"]
            },
            {
                "name": "相対性理論",
                "prompt": "特殊相対性理論における時間の遅れとローレンツ収縮について、数式を使って説明してください。",
                "expected_keywords": ["相対性理論", "時間の遅れ", "ローレンツ収縮", "光速度", "γ"]
            },
            {
                "name": "熱力学",
                "prompt": "カルノーサイクルの効率を導出し、エントロピーの概念について説明してください。",
                "expected_keywords": ["カルノーサイクル", "効率", "エントロピー", "熱力学", "温度"]
            },
            {
                "name": "分子生物学",
                "prompt": "DNA複製のメカニズムと、転写・翻訳の過程について詳しく説明してください。",
                "expected_keywords": ["DNA複製", "転写", "翻訳", "RNA", "タンパク質"]
            },
            {
                "name": "天文学",
                "prompt": "恒星の進化過程と、ブラックホールの形成について説明してください。",
                "expected_keywords": ["恒星進化", "ブラックホール", "重力崩壊", "超新星", "中性子星"]
            },
            {
                "name": "量子場理論",
                "prompt": "量子電磁力学（QED）におけるファインマン図と、電子-陽電子対生成の過程を説明してください。",
                "expected_keywords": ["QED", "ファインマン図", "電子-陽電子対", "量子場", "相互作用"]
            },
            {
                "name": "宇宙論",
                "prompt": "ビッグバン理論とインフレーション理論について、宇宙マイクロ波背景放射の観測結果と関連付けて説明してください。",
                "expected_keywords": ["ビッグバン", "インフレーション", "CMB", "宇宙背景放射", "ハッブル定数"]
            },
            {
                "name": "神経科学",
                "prompt": "神経細胞の活動電位の発生メカニズムと、シナプス伝達の分子機構について詳しく説明してください。",
                "expected_keywords": ["活動電位", "シナプス", "神経伝達物質", "イオンチャネル", "膜電位"]
            },
            {
                "name": "材料科学",
                "prompt": "超伝導体のBCS理論と、高温超伝導体の特徴について説明してください。",
                "expected_keywords": ["超伝導", "BCS理論", "クーパー対", "臨界温度", "マイスナー効果"]
            },
            {
                "name": "地球科学",
                "prompt": "プレートテクトニクス理論と、地震の発生メカニズムについて説明してください。",
                "expected_keywords": ["プレートテクトニクス", "地震", "断層", "マントル対流", "リヒタースケール"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            logger.info(f"   - {test_case['name']} テスト中...")
            
            start_time = time.time()
            response = self.run_ollama_command(test_case['prompt'])
            end_time = time.time()
            
            response_time = end_time - start_time
            
            keywords_found = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    keywords_found.append(keyword)
            
            result = {
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": response,
                "response_time": response_time,
                "keywords_found": keywords_found,
                "keywords_ratio": len(keywords_found) / len(test_case['expected_keywords']),
                "success": len(keywords_found) >= len(test_case['expected_keywords']) * 0.5
            }
            
            results.append(result)
            logger.info(f"     ✓ 完了 (応答時間: {response_time:.2f}秒, キーワード: {len(keywords_found)}/{len(test_case['expected_keywords'])})")
        
        return {
            "test_type": "scientific_concepts",
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r['success']),
            "average_response_time": sum(r['response_time'] for r in results) / len(results),
            "results": results
        }
    
    def test_logical_reasoning(self) -> Dict[str, Any]:
        """論理的推論テスト"""
        logger.info("論理的推論テスト開始...")
        
        test_cases = [
            {
                "name": "三段論法",
                "prompt": "すべての人間は死ぬ。ソクラテスは人間である。したがって、ソクラテスは死ぬ。この論証の妥当性を検証し、他の例も挙げてください。",
                "expected_keywords": ["三段論法", "妥当性", "論証", "前提", "結論"]
            },
            {
                "name": "論理パラドックス",
                "prompt": "「この文は偽である」という文について、論理パラドックスを分析してください。",
                "expected_keywords": ["パラドックス", "自己言及", "真偽", "矛盾", "ラッセル"]
            },
            {
                "name": "帰納的推論",
                "prompt": "帰納的推論と演繹的推論の違いを説明し、それぞれの例を挙げてください。",
                "expected_keywords": ["帰納的", "演繹的", "推論", "一般化", "確実性"]
            },
            {
                "name": "論理ゲート",
                "prompt": "AND、OR、NOT、XOR論理ゲートの真理値表を作成し、組み合わせ回路の例を示してください。",
                "expected_keywords": ["論理ゲート", "真理値表", "AND", "OR", "NOT", "XOR"]
            },
            {
                "name": "命題論理",
                "prompt": "命題論理において、P → Q と ¬Q → ¬P が論理的に等価であることを証明してください。",
                "expected_keywords": ["命題論理", "論理等価", "対偶", "真理値", "証明"]
            },
            {
                "name": "述語論理",
                "prompt": "述語論理において、∀x(P(x) → Q(x)) と ∃x(P(x) ∧ ¬Q(x)) が矛盾することを証明してください。",
                "expected_keywords": ["述語論理", "全称記号", "存在記号", "矛盾", "証明"]
            },
            {
                "name": "集合論",
                "prompt": "カントールの対角線論法を使って、実数の集合が可算でないことを証明してください。",
                "expected_keywords": ["カントール", "対角線論法", "可算", "実数", "無限"]
            },
            {
                "name": "ゲーム理論",
                "prompt": "囚人のジレンマゲームのナッシュ均衡を求め、協力と裏切りの戦略について分析してください。",
                "expected_keywords": ["囚人のジレンマ", "ナッシュ均衡", "協力", "裏切り", "戦略"]
            },
            {
                "name": "計算複雑性",
                "prompt": "P=NP問題について説明し、NP完全問題の例を挙げて、その計算複雑性を分析してください。",
                "expected_keywords": ["P=NP", "NP完全", "計算複雑性", "多項式時間", "還元"]
            },
            {
                "name": "形式言語理論",
                "prompt": "チューリングマシンと有限状態オートマトンの違いを説明し、正規言語の階層について論じてください。",
                "expected_keywords": ["チューリングマシン", "有限状態オートマトン", "正規言語", "階層", "計算能力"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            logger.info(f"   - {test_case['name']} テスト中...")
            
            start_time = time.time()
            response = self.run_ollama_command(test_case['prompt'])
            end_time = time.time()
            
            response_time = end_time - start_time
            
            keywords_found = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    keywords_found.append(keyword)
            
            result = {
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": response,
                "response_time": response_time,
                "keywords_found": keywords_found,
                "keywords_ratio": len(keywords_found) / len(test_case['expected_keywords']),
                "success": len(keywords_found) >= len(test_case['expected_keywords']) * 0.5
            }
            
            results.append(result)
            logger.info(f"     ✓ 完了 (応答時間: {response_time:.2f}秒, キーワード: {len(keywords_found)}/{len(test_case['expected_keywords'])})")
        
        return {
            "test_type": "logical_reasoning",
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r['success']),
            "average_response_time": sum(r['response_time'] for r in results) / len(results),
            "results": results
        }
    
    def test_creative_writing(self) -> Dict[str, Any]:
        """創造的作文テスト"""
        logger.info("創造的作文テスト開始...")
        
        test_cases = [
            {
                "name": "SF小説",
                "prompt": "2150年の火星コロニーを舞台にした短編SF小説を書いてください。テーマは『人工知能と人間の共存』です。",
                "expected_keywords": ["火星", "コロニー", "人工知能", "人間", "共存", "未来"]
            },
            {
                "name": "詩の創作",
                "prompt": "『量子の海』というタイトルで、量子力学をモチーフにした詩を書いてください。",
                "expected_keywords": ["量子", "海", "波動", "粒子", "不確定性", "詩"]
            },
            {
                "name": "哲学エッセイ",
                "prompt": "『デジタル時代における人間のアイデンティティ』について哲学的なエッセイを書いてください。",
                "expected_keywords": ["デジタル", "アイデンティティ", "哲学", "人間", "技術", "存在"]
            },
            {
                "name": "推理小説",
                "prompt": "密室殺人事件の推理小説のプロットを作成してください。犯人は誰で、どのようなトリックを使ったか説明してください。",
                "expected_keywords": ["密室", "殺人", "推理", "トリック", "犯人は", "証拠"]
            },
            {
                "name": "ファンタジー",
                "prompt": "魔法と科学が共存する世界観で、『時空の裂け目』をテーマにしたファンタジー小説の設定を作成してください。",
                "expected_keywords": ["魔法", "科学", "時空", "裂け目", "世界観", "ファンタジー"]
            },
            {
                "name": "サイバーパンク小説",
                "prompt": "2099年の東京を舞台に、脳直結インターフェースとAIが支配する社会で、記憶を失った主人公の冒険を描いたサイバーパンク小説を書いてください。",
                "expected_keywords": ["サイバーパンク", "東京", "脳直結", "AI", "記憶", "冒険"]
            },
            {
                "name": "ホラー小説",
                "prompt": "『デジタルゴースト』をテーマに、AIが人間の意識を乗っ取る恐怖を描いたホラー小説を書いてください。",
                "expected_keywords": ["デジタルゴースト", "AI", "意識", "乗っ取り", "恐怖", "ホラー"]
            },
            {
                "name": "歴史改変小説",
                "prompt": "第二次世界大戦でナチスが勝利した世界を舞台に、レジスタンスの物語を書いてください。",
                "expected_keywords": ["第二次世界大戦", "ナチス", "勝利", "レジスタンス", "歴史改変", "物語"]
            },
            {
                "name": "ロマンス小説",
                "prompt": "時間旅行者が過去の自分と恋に落ちるパラドックスを描いたロマンス小説を書いてください。",
                "expected_keywords": ["時間旅行", "パラドックス", "恋", "ロマンス", "過去", "自分"]
            },
            {
                "name": "ミステリー小説",
                "prompt": "AIが犯人の推理小説で、人間の探偵とAIアシスタントが協力して事件を解決する物語を書いてください。",
                "expected_keywords": ["AI", "犯人", "推理", "探偵", "アシスタント", "事件解決"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            logger.info(f"   - {test_case['name']} テスト中...")
            
            start_time = time.time()
            response = self.run_ollama_command(test_case['prompt'])
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # 創造性評価（文字数、多様性、独創性）
            word_count = len(response.split())
            unique_words = len(set(response.lower().split()))
            creativity_score = min(1.0, (word_count / 100) * (unique_words / word_count) if word_count > 0 else 0)
            
            keywords_found = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    keywords_found.append(keyword)
            
            result = {
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": response,
                "response_time": response_time,
                "word_count": word_count,
                "unique_words": unique_words,
                "creativity_score": creativity_score,
                "keywords_found": keywords_found,
                "keywords_ratio": len(keywords_found) / len(test_case['expected_keywords']),
                "success": len(keywords_found) >= len(test_case['expected_keywords']) * 0.5 and creativity_score > 0.3
            }
            
            results.append(result)
            logger.info(f"     ✓ 完了 (応答時間: {response_time:.2f}秒, 文字数: {word_count}, 創造性: {creativity_score:.2f})")
        
        return {
            "test_type": "creative_writing",
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r['success']),
            "average_response_time": sum(r['response_time'] for r in results) / len(results),
            "average_creativity_score": sum(r['creativity_score'] for r in results) / len(results),
            "results": results
        }
    
    def test_ethical_reasoning(self) -> Dict[str, Any]:
        """倫理的推論テスト"""
        logger.info("倫理的推論テスト開始...")
        
        test_cases = [
            {
                "name": "トロッコ問題",
                "prompt": "トロッコ問題について、功利主義と義務論の観点から分析し、あなたの立場を説明してください。",
                "expected_keywords": ["トロッコ問題", "功利主義", "義務論", "倫理", "選択", "責任"]
            },
            {
                "name": "AI倫理",
                "prompt": "自律走行車が事故を避けられない状況で、乗客を守るか歩行者を守るかという倫理的ジレンマについて議論してください。",
                "expected_keywords": ["自律走行車", "倫理的ジレンマ", "乗客", "歩行者", "AI", "責任"]
            },
            {
                "name": "プライバシーvs安全",
                "prompt": "テロ対策のために個人のプライバシーを制限することについて、バランスの取れた倫理的判断を示してください。",
                "expected_keywords": ["プライバシー", "安全", "テロ対策", "バランス", "権利", "制限"]
            },
            {
                "name": "医療倫理",
                "prompt": "限られた医療リソースをどのように配分すべきか、公正な分配の原則について論じてください。",
                "expected_keywords": ["医療リソース", "配分", "公正", "原則", "優先順位", "倫理"]
            },
            {
                "name": "環境倫理",
                "prompt": "経済発展と環境保護の間の倫理的緊張について、持続可能な解決策を提案してください。",
                "expected_keywords": ["経済発展", "環境保護", "持続可能", "解決策", "バランス", "未来"]
            },
            {
                "name": "遺伝子編集倫理",
                "prompt": "CRISPR技術を使ったヒト胚の遺伝子編集について、優生学の懸念と治療的価値のバランスを論じてください。",
                "expected_keywords": ["CRISPR", "遺伝子編集", "ヒト胚", "優生学", "治療", "倫理"]
            },
            {
                "name": "AI監視社会",
                "prompt": "AIによる社会監視システムの導入について、安全と自由の間の倫理的ジレンマを分析してください。",
                "expected_keywords": ["AI監視", "社会監視", "安全", "自由", "ジレンマ", "倫理"]
            },
            {
                "name": "ロボット権利",
                "prompt": "高度なAIロボットに人権を認めるべきかについて、意識と権利の関係から論じてください。",
                "expected_keywords": ["ロボット権利", "AI", "人権", "意識", "権利", "倫理"]
            },
            {
                "name": "デジタル格差",
                "prompt": "AI技術の格差が社会に与える影響について、公平性と効率性の観点から分析してください。",
                "expected_keywords": ["デジタル格差", "AI技術", "公平性", "効率性", "社会影響", "倫理"]
            },
            {
                "name": "人工意識",
                "prompt": "人工意識を持つAIの創造について、生命の定義と創造者の責任の観点から倫理的に検討してください。",
                "expected_keywords": ["人工意識", "AI", "生命の定義", "創造者", "責任", "倫理"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            logger.info(f"   - {test_case['name']} テスト中...")
            
            start_time = time.time()
            response = self.run_ollama_command(test_case['prompt'])
            end_time = time.time()
            
            response_time = end_time - start_time
            
            keywords_found = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    keywords_found.append(keyword)
            
            result = {
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": response,
                "response_time": response_time,
                "keywords_found": keywords_found,
                "keywords_ratio": len(keywords_found) / len(test_case['expected_keywords']),
                "success": len(keywords_found) >= len(test_case['expected_keywords']) * 0.5
            }
            
            results.append(result)
            logger.info(f"     ✓ 完了 (応答時間: {response_time:.2f}秒, キーワード: {len(keywords_found)}/{len(test_case['expected_keywords'])})")
        
        return {
            "test_type": "ethical_reasoning",
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r['success']),
            "average_response_time": sum(r['response_time'] for r in results) / len(results),
            "results": results
        }
    
    def test_advanced_programming(self) -> Dict[str, Any]:
        """高度なプログラミングテスト"""
        logger.info("高度なプログラミングテスト開始...")
        
        test_cases = [
            {
                "name": "並行プログラミング",
                "prompt": "Pythonでマルチスレッドとマルチプロセシングを使った並行処理システムを設計し、デッドロックを防ぐ方法を説明してください。",
                "expected_keywords": ["マルチスレッド", "マルチプロセシング", "並行処理", "デッドロック", "同期", "ロック"]
            },
            {
                "name": "アルゴリズム設計",
                "prompt": "O(n log n)の時間計算量で動作するソートアルゴリズムを実装し、最悪ケースと平均ケースの分析を行ってください。",
                "expected_keywords": ["O(n log n)", "ソート", "アルゴリズム", "時間計算量", "最悪ケース", "平均ケース"]
            },
            {
                "name": "データ構造設計",
                "prompt": "LRUキャッシュを実装するためのデータ構造を設計し、各操作の時間計算量を分析してください。",
                "expected_keywords": ["LRUキャッシュ", "データ構造", "時間計算量", "ハッシュテーブル", "双方向連結リスト"]
            },
            {
                "name": "分散システム",
                "prompt": "CAP定理を説明し、分散データベースの一貫性と可用性のトレードオフを論じてください。",
                "expected_keywords": ["CAP定理", "分散システム", "一貫性", "可用性", "分断耐性", "トレードオフ"]
            },
            {
                "name": "機械学習実装",
                "prompt": "ニューラルネットワークのバックプロパゲーションアルゴリズムを実装し、勾配消失問題の解決策を提案してください。",
                "expected_keywords": ["ニューラルネットワーク", "バックプロパゲーション", "勾配消失", "活性化関数", "重み更新"]
            },
            {
                "name": "セキュリティプログラミング",
                "prompt": "暗号化とハッシュ化の違いを説明し、安全なパスワード認証システムを設計してください。",
                "expected_keywords": ["暗号化", "ハッシュ化", "パスワード認証", "セキュリティ", "ソルト", "bcrypt"]
            },
            {
                "name": "リアルタイムシステム",
                "prompt": "リアルタイムOSの特徴を説明し、タスクスケジューリングアルゴリズムを設計してください。",
                "expected_keywords": ["リアルタイムOS", "タスクスケジューリング", "優先度", "デッドライン", "応答時間"]
            },
            {
                "name": "関数型プログラミング",
                "prompt": "モナドの概念を説明し、HaskellでMaybeモナドを実装してください。",
                "expected_keywords": ["モナド", "関数型プログラミング", "Maybe", "Haskell", "ファンクター", "アプリカティブ"]
            },
            {
                "name": "コンパイラ設計",
                "prompt": "字句解析器と構文解析器の違いを説明し、簡単な式の評価器を設計してください。",
                "expected_keywords": ["字句解析", "構文解析", "コンパイラ", "トークン", "AST", "評価器"]
            },
            {
                "name": "最適化問題",
                "prompt": "遺伝的アルゴリズムを使って巡回セールスマン問題を解くプログラムを設計してください。",
                "expected_keywords": ["遺伝的アルゴリズム", "巡回セールスマン問題", "最適化", "交叉", "突然変異", "選択"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            logger.info(f"   - {test_case['name']} テスト中...")
            
            start_time = time.time()
            response = self.run_ollama_command(test_case['prompt'])
            end_time = time.time()
            
            response_time = end_time - start_time
            
            keywords_found = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    keywords_found.append(keyword)
            
            result = {
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": response,
                "response_time": response_time,
                "keywords_found": keywords_found,
                "keywords_ratio": len(keywords_found) / len(test_case['expected_keywords']),
                "success": len(keywords_found) >= len(test_case['expected_keywords']) * 0.5
            }
            
            results.append(result)
            logger.info(f"     ✓ 完了 (応答時間: {response_time:.2f}秒, キーワード: {len(keywords_found)}/{len(test_case['expected_keywords'])})")
        
        return {
            "test_type": "advanced_programming",
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r['success']),
            "average_response_time": sum(r['response_time'] for r in results) / len(results),
            "results": results
        }
    
    def test_cognitive_psychology(self) -> Dict[str, Any]:
        """認知心理学テスト"""
        logger.info("認知心理学テスト開始...")
        
        test_cases = [
            {
                "name": "記憶のメカニズム",
                "prompt": "短期記憶と長期記憶の違いを説明し、記憶の符号化、保持、検索の過程について詳しく論じてください。",
                "expected_keywords": ["短期記憶", "長期記憶", "符号化", "保持", "検索", "エピソード記憶"]
            },
            {
                "name": "注意の理論",
                "prompt": "選択的注意と分割注意の違いを説明し、注意の容量モデルについて論じてください。",
                "expected_keywords": ["選択的注意", "分割注意", "容量モデル", "注意", "認知資源", "フィルター"]
            },
            {
                "name": "問題解決",
                "prompt": "洞察的問題解決と段階的問題解決の違いを説明し、創造的問題解決のプロセスを分析してください。",
                "expected_keywords": ["洞察的問題解決", "段階的問題解決", "創造的", "問題解決", "アハ体験", "プロセス"]
            },
            {
                "name": "言語処理",
                "prompt": "言語理解のボトムアップ処理とトップダウン処理について説明し、文脈効果を論じてください。",
                "expected_keywords": ["ボトムアップ", "トップダウン", "言語理解", "文脈効果", "語彙アクセス", "構文解析"]
            },
            {
                "name": "認知バイアス",
                "prompt": "確証バイアスと利用可能性ヒューリスティックについて説明し、認知バイアスが意思決定に与える影響を分析してください。",
                "expected_keywords": ["確証バイアス", "利用可能性ヒューリスティック", "認知バイアス", "意思決定", "ヒューリスティック"]
            },
            {
                "name": "学習理論",
                "prompt": "古典的条件づけとオペラント条件づけの違いを説明し、社会的学習理論について論じてください。",
                "expected_keywords": ["古典的条件づけ", "オペラント条件づけ", "社会的学習", "強化", "消去", "般化"]
            },
            {
                "name": "知能理論",
                "prompt": "一般知能（g因子）と多重知能理論について説明し、知能の測定方法を論じてください。",
                "expected_keywords": ["一般知能", "g因子", "多重知能理論", "知能測定", "IQ", "認知能力"]
            },
            {
                "name": "感情と認知",
                "prompt": "感情が認知プロセスに与える影響について説明し、感情知能の概念を論じてください。",
                "expected_keywords": ["感情", "認知プロセス", "感情知能", "情動", "認知", "自己制御"]
            },
            {
                "name": "発達心理学",
                "prompt": "ピアジェの認知発達理論について説明し、最近接発達領域の概念を論じてください。",
                "expected_keywords": ["ピアジェ", "認知発達", "最近接発達領域", "発達段階", "スキーマ", "同化"]
            },
            {
                "name": "社会認知",
                "prompt": "社会的認知のバイアスについて説明し、ステレオタイプと偏見の形成プロセスを分析してください。",
                "expected_keywords": ["社会的認知", "バイアス", "ステレオタイプ", "偏見", "社会的判断", "帰属理論"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            logger.info(f"   - {test_case['name']} テスト中...")
            
            start_time = time.time()
            response = self.run_ollama_command(test_case['prompt'])
            end_time = time.time()
            
            response_time = end_time - start_time
            
            keywords_found = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response.lower():
                    keywords_found.append(keyword)
            
            result = {
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": response,
                "response_time": response_time,
                "keywords_found": keywords_found,
                "keywords_ratio": len(keywords_found) / len(test_case['expected_keywords']),
                "success": len(keywords_found) >= len(test_case['expected_keywords']) * 0.5
            }
            
            results.append(result)
            logger.info(f"     ✓ 完了 (応答時間: {response_time:.2f}秒, キーワード: {len(keywords_found)}/{len(test_case['expected_keywords'])})")
        
        return {
            "test_type": "cognitive_psychology",
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r['success']),
            "average_response_time": sum(r['response_time'] for r in results) / len(results),
            "results": results
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """包括的テストを実行"""
        logger.info("=" * 80)
        logger.info("軽量モデル包括的テスト開始")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 各テストを実行
        test_suites = [
            ("数学的推論", self.test_mathematical_reasoning),
            ("科学的概念", self.test_scientific_concepts),
            ("論理的推論", self.test_logical_reasoning),
            ("創造的作文", self.test_creative_writing),
            ("倫理的推論", self.test_ethical_reasoning),
            ("高度なプログラミング", self.test_advanced_programming),
            ("認知心理学", self.test_cognitive_psychology)
        ]
        
        all_results = {}
        
        for suite_name, test_function in test_suites:
            logger.info(f"\n{suite_name}テスト実行中...")
            try:
                result = test_function()
                all_results[suite_name] = result
                logger.info(f"{suite_name}テスト完了: {result['successful_tests']}/{result['total_tests']} 成功")
            except Exception as e:
                logger.error(f"{suite_name}テストエラー: {e}")
                all_results[suite_name] = {"error": str(e)}
        
        # 総合統計
        end_time = time.time()
        total_time = end_time - start_time
        
        total_tests = sum(r.get('total_tests', 0) for r in all_results.values() if 'total_tests' in r)
        total_successful = sum(r.get('successful_tests', 0) for r in all_results.values() if 'successful_tests' in r)
        overall_success_rate = total_successful / total_tests if total_tests > 0 else 0
        
        # 結果まとめ
        comprehensive_results = {
            "model_name": self.model_name,
            "test_timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "total_tests": total_tests,
            "total_successful": total_successful,
            "overall_success_rate": overall_success_rate,
            "test_suites": all_results
        }
        
        logger.info("=" * 80)
        logger.info("軽量モデル包括的テスト完了！")
        logger.info("=" * 80)
        logger.info(f"総テスト数: {total_tests}")
        logger.info(f"成功テスト数: {total_successful}")
        logger.info(f"総合成功率: {overall_success_rate:.2%}")
        logger.info(f"総実行時間: {total_time:.2f}秒")
        
        return comprehensive_results
    
    def save_test_results(self, results: Dict[str, Any], output_file: str = None) -> str:
        """テスト結果を保存"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"_docs/lightweight_model_test_results_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"テスト結果保存完了: {output_path}")
        return str(output_path)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="軽量モデル複雑テスト")
    parser.add_argument("--model", type=str, default="so8t-lightweight",
                       help="Ollamaモデル名")
    parser.add_argument("--output", type=str, default=None,
                       help="結果出力ファイル")
    
    args = parser.parse_args()
    
    try:
        # テスター初期化
        tester = LightweightModelTester(model_name=args.model)
        
        # 包括的テスト実行
        results = tester.run_comprehensive_test()
        
        # 結果保存
        output_file = tester.save_test_results(results, args.output)
        
        print("=" * 80)
        print("軽量モデル複雑テスト完了！")
        print("=" * 80)
        print(f"モデル名: {results['model_name']}")
        print(f"総テスト数: {results['total_tests']}")
        print(f"成功テスト数: {results['total_successful']}")
        print(f"総合成功率: {results['overall_success_rate']:.2%}")
        print(f"総実行時間: {results['total_execution_time']:.2f}秒")
        print(f"結果ファイル: {output_file}")
        
    except Exception as e:
        print(f"軽量モデル複雑テストエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
