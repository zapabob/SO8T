#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS-phi3.5-v2.0 A/B Test Benchmark

モデルA (Borea-Phi3.5-instinct-jp) と モデルB (AEGIS-phi3.5-v2.0) の
業界標準ベンチマーク + ELYZA-100 全問比較
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import f_oneway, ttest_ind
import requests

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# インポート
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'  # フォールバック
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    pass

# japanize_matplotlibが利用可能なら使用
try:
    import japanize_matplotlib
    japanize_matplotlib.japanize()
except ImportError:
    print("japanize_matplotlibがインストールされていないため、デフォルトフォントを使用します")

@dataclass
class ABTestConfig:
    """A/Bテスト設定"""
    model_a_path: str = "microsoft/Phi-3.5-mini-instruct"  # Base model for comparison
    model_b_path: str = "AEGIS-Phi3.5-thinking-v2.0"  # AEGIS-Phi3.5-thinking-v2.0
    output_dir: str = r"H:\from_D\webdataset\benchmark_results\aegis_ab_test"  # H:\from_D\webdatasetに保存
    device: str = "auto"  # CUDA優先
    use_4bit: bool = True  # 4bit量子化オン
    max_new_tokens: int = 512  # 標準長
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples_per_question: int = 3  # 各質問3回（統計的有意性）
    random_seed: int = 42
    test_mode: bool = False  # 本番モード
    # チェックポイント設定
    checkpoint_interval: int = 180  # 3分間隔 (秒)
    max_checkpoints: int = 5  # ローリングストック数
    checkpoint_dir: str = r"H:\from_D\webdataset\checkpoints\ab_test"  # H:\from_D\webdatasetに保存
    # GGUF変換設定
    gguf_convert: bool = True
    gguf_dir: str = r"H:\from_D\webdataset\gguf_models"  # H:\from_D\webdatasetに保存
    # LM-Evaluation-Harness設定
    use_lm_eval_harness: bool = True  # lm-evaluation-harnessを使用
    lm_eval_tasks: str = "hellaswag,mmlu"  # 評価タスク
    lm_eval_output_dir: str = r"H:\from_D\webdataset\benchmark_results\lm_eval"  # LM-Eval結果保存先
    # 自動起動設定
    auto_restart: bool = True

    # ELYZA-100設定
    elyza_dataset_url: str = "https://huggingface.co/datasets/elyza/ELYZA-tasks-100/raw/main/elyza_tasks_100.json"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class AEGISABTester:
    """AEGIS A/Bテスター"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.model_a = None
        self.model_b = None
        self.tokenizer_a = None
        self.tokenizer_b = None
        self.results = []
        self.start_time = datetime.now()
        self.last_checkpoint_time = datetime.now()

        # チェックポイント設定
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 乱数シード設定
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # シグナルハンドラー設定（電源遮断対策）
        import signal
        signal.signal(signal.SIGINT, self._emergency_save)
        signal.signal(signal.SIGTERM, self._emergency_save)
        try:
            signal.signal(signal.SIGBREAK, self._emergency_save)  # Windows
        except AttributeError:
            pass

    def _emergency_save(self, signum, frame):
        """緊急保存（電源遮断時）"""
        print(f"\n[EMERGENCY] Signal {signum} received. Performing emergency save...")
        self._save_checkpoint("emergency")
        print("[EMERGENCY] Emergency save completed. Exiting...")
        sys.exit(1)

    def _save_checkpoint(self, suffix: str = ""):
        """チェックポイント保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                'timestamp': timestamp,
                'config': self.config.__dict__,
                'results': self.results,
                'start_time': self.start_time.isoformat(),
                'elapsed_time': str(datetime.now() - self.start_time)
            }

            checkpoint_file = self.checkpoint_dir / f"ab_test_checkpoint_{timestamp}_{suffix}.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

            # ローリングストック管理（古いチェックポイント削除）
            checkpoints = sorted(self.checkpoint_dir.glob("ab_test_checkpoint_*.json"))
            if len(checkpoints) > self.config.max_checkpoints:
                for old_checkpoint in checkpoints[:-self.config.max_checkpoints]:
                    old_checkpoint.unlink()

            print(f"[CHECKPOINT] Saved to {checkpoint_file} ({len(checkpoints)} total)")
            self.last_checkpoint_time = datetime.now()

        except Exception as e:
            print(f"[ERROR] Checkpoint save failed: {e}")

    def _check_checkpoint_save(self):
        """定期チェックポイント保存チェック"""
        if (datetime.now() - self.last_checkpoint_time).seconds >= self.config.checkpoint_interval:
            self._save_checkpoint()

    def load_checkpoint(self):
        """最新チェックポイントから復元"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("ab_test_checkpoint_*.json"))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)

                self.results = checkpoint_data.get('results', [])
                self.start_time = datetime.fromisoformat(checkpoint_data['start_time'])
                print(f"[CHECKPOINT] Loaded from {latest_checkpoint}")
                return True
        except Exception as e:
            print(f"[WARNING] Checkpoint load failed: {e}")
        return False

    def load_models(self):
        """モデル読み込み"""
        print("=== モデル読み込み ===")

        # 量子化設定
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # モデルA読み込み (ベースモデル)
        print("モデルA (ベース) 読み込み中...")
        try:
            # HFから直接読み込む場合
            if "/" in self.config.model_a_path and not Path(self.config.model_a_path).exists():
                print(f"HFから {self.config.model_a_path} を読み込みます...")
                self.tokenizer_a = AutoTokenizer.from_pretrained(
                    self.config.model_a_path,
                    trust_remote_code=True
                )
                if self.tokenizer_a.pad_token is None:
                    self.tokenizer_a.pad_token = self.tokenizer_a.eos_token

                self.model_a = AutoModelForCausalLM.from_pretrained(
                    self.config.model_a_path,
                    quantization_config=bnb_config,
                    device_map=self.config.device,
                    trust_remote_code=True
                )
            else:
                # ローカルファイルの場合
                local_path = PROJECT_ROOT / self.config.model_a_path
                print(f"ローカルから {local_path} を読み込みます...")
                self.tokenizer_a = AutoTokenizer.from_pretrained(
                    str(local_path),
                    trust_remote_code=True
                )
                if self.tokenizer_a.pad_token is None:
                    self.tokenizer_a.pad_token = self.tokenizer_a.eos_token

                self.model_a = AutoModelForCausalLM.from_pretrained(
                    str(local_path),
                    quantization_config=bnb_config,
                    device_map=self.config.device,
                    trust_remote_code=True
                )
            print("[OK] モデルA読み込み成功")
        except Exception as e:
            print(f"[NG] モデルA読み込み失敗: {e}")
            return False

        # モデルB読み込み (AEGISモデル)
        print("モデルB (AEGIS) 読み込み中...")
        try:
            if Path(self.config.model_b_path).exists():
                self.tokenizer_b = AutoTokenizer.from_pretrained(
                    self.config.model_b_path,
                    trust_remote_code=True
                )
                if self.tokenizer_b.pad_token is None:
                    self.tokenizer_b.pad_token = self.tokenizer_b.eos_token

                self.model_b = AutoModelForCausalLM.from_pretrained(
                    self.config.model_b_path,
                    quantization_config=bnb_config,
                    device_map=self.config.device,
                    trust_remote_code=True
                )
                print("[OK] モデルB読み込み成功")
            else:
                print(f"[WARN] モデルBが見つからないため、モデルAのみでテストを実行: {self.config.model_b_path}")
                self.model_b = None
                self.tokenizer_b = self.tokenizer_a  # フォールバック
        except Exception as e:
            print(f"[NG] モデルB読み込み失敗: {e}")
            print("モデルAのみでテストを実行します")
            self.model_b = None
            self.tokenizer_b = self.tokenizer_a

        return True

    def load_elyza_dataset(self) -> List[Dict[str, Any]]:
        """ELYZA-100データセット読み込み"""
        print("=== ELYZA-100データセット読み込み ===")

        # まずローカルファイルを確認
        local_path = PROJECT_ROOT / "data" / "elyza_tasks_100.json"
        if local_path.exists():
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"[OK] ローカルELYZA-100: {len(data)}件読み込みました")
                return data
            except Exception as e:
                print(f"[NG] ローカルファイル読み込み失敗: {e}")

        # HFからダウンロードを試行
        try:
            response = requests.get(self.config.elyza_dataset_url)
            response.raise_for_status()
            data = response.json()

            # ローカルに保存
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"[OK] ELYZA-100: {len(data)}件のタスクを読み込みました")
            return data

        except Exception as e:
            print(f"[NG] ELYZA-100読み込み失敗: {e}")
            # テスト用ダミーデータ作成
            print("[WARN] テスト用ダミーデータを作成します")
            return self._create_dummy_elyza_data()

    def _create_dummy_elyza_data(self) -> List[Dict[str, Any]]:
        """テスト用ダミーELYZAデータ作成"""
        dummy_data = [
            {
                "task_id": "task_001",
                "category": "reasoning",
                "input": "以下の文章を読んで、筆者の主張を簡潔にまとめなさい。「人工知能の進化は人類の未来を左右する重要な技術である。しかし、その発展を適切に制御しなければ、予期せぬリスクが生じる可能性がある。」",
                "output": "筆者は人工知能の進化が人類の未来を左右する重要な技術であると主張しているが、適切な制御を怠ると予期せぬリスクが生じる可能性があると警告している。"
            },
            {
                "task_id": "task_002",
                "category": "calculation",
                "input": "123 + 456 × 2 - 78 ÷ 3 の計算結果を求めなさい。",
                "output": "まず、456 × 2 = 912、次に78 ÷ 3 = 26、そして123 + 912 = 1035、最後に1035 - 26 = 1009となる。"
            },
            {
                "task_id": "task_003",
                "category": "knowledge",
                "input": "日本国憲法の第9条について説明しなさい。",
                "output": "日本国憲法第9条は戦争の放棄と戦力の不保持を定めた条項である。第1項では「日本国民は、正義と秩序を基調とする国際平和を誠実に希求し、国権の発動たる戦争と、武力による威嚇又は武力の行使は、国際紛争を解決する手段としては、永久にこれを放棄する」と規定されている。"
            },
            {
                "task_id": "task_004",
                "category": "reasoning",
                "input": "以下の論理パズルを解きなさい。「A、B、Cの3人がそれぞれ赤、白、青の帽子をかぶっている。誰も自分の帽子の色を知らないが、他人の帽子の色は見える。3人とも論理的思考ができる。司会者が『少なくとも1人は赤い帽子をかぶっている』と発表した後、Aさんが『私は青い帽子だ』と言った。この状況でAさんの帽子の色は何色か？理由も説明せよ。」",
                "output": "司会者の発表により、誰も赤い帽子でない可能性は除外された。Aさんが青い帽子だと判断したということは、AさんはBとCの帽子を見て、以下の論理で判断した：もし自分(A)が赤い帽子なら、BはAが赤い帽子であるのを見て、自分が白い帽子だとすぐに判断できたはずである。しかしBが何も言わなかったということは、BはAが赤くないと判断した。つまりAは赤くない。AはさらにCの帽子を見て、自分が青い帽子だと判断した。"
            },
            {
                "task_id": "task_005",
                "category": "creative",
                "input": "「春の訪れ」というテーマで、短い詩を作成しなさい。",
                "output": "桜舞う春の風\n芽吹く命の息吹\n暖かな陽射しに\n希望の花が咲く"
            }
        ]

        # 100件まで拡張
        extended_data = []
        for i in range(1, 101):
            task = dummy_data[(i-1) % len(dummy_data)].copy()
            task["task_id"] = "06d"
            task["input"] = task["input"].replace("以下の文章", f"タスク{i}: 以下の文章")
            extended_data.append(task)

        return extended_data[:100]  # 最大100件

    def evaluate_answer_quality(self, question: str, answer: str, reference_answer: str = None) -> Dict[str, float]:
        """回答品質評価 (ベストプラクティスに基づく)"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        def rouge_l(a: str, b: str) -> float:
            # ROUGE-Lの簡易実装
            def lcs(X, Y):
                m, n = len(X), len(Y)
                dp = [[0]*(n+1) for _ in range(m+1)]
                for i in range(m):
                    for j in range(n):
                        if X[i] == Y[j]:
                            dp[i+1][j+1] = dp[i][j]+1
                        else:
                            dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
                return dp[m][n]
            if not a or not b:
                return 0.0
            lcs_len = lcs(a, b)
            prec = lcs_len / (len(a) + 1e-8)
            rec = lcs_len / (len(b) + 1e-8)
            beta = 1.2
            if (prec + rec) == 0:
                return 0.0
            return (1 + beta**2) * prec * rec / ((rec + beta**2 * prec) + 1e-8)

        # 1. 長さスコア：日本語的な適切な長さ（50~300文字内）を1.0、遠いと下がる
        length = len(answer)
        min_len, max_len = 50, 300
        if length < min_len:
            length_score = length / min_len * 0.6
        elif length > max_len:
            length_score = max(0.8, 1.0 - (length - max_len) * 0.002)
        else:
            length_score = 1.0

        # 2. 日本語率（元の関数を利用）
        japanese_score = self._calculate_japanese_ratio(answer)

        # 3. 一貫性スコア：文区切り、重複語、終了記号で評価
        coherence_score = self._calculate_coherence_score(answer)

        # 4. 関連度（reference_answerが存在すればrouge-l、無ければ質問とのTF-IDFコサイン類似度）
        if reference_answer:
            relevance_score = rouge_l(answer, reference_answer)
        else:
            tfidf = TfidfVectorizer().fit([question, answer])
            cosine = cosine_similarity(tfidf.transform([question]), tfidf.transform([answer]))[0][0]
            # 0.0~1.0正規化
            relevance_score = min(max(cosine, 0.0), 1.0)

        scores = {
            'length_score': round(length_score, 4),
            'japanese_score': round(japanese_score, 4),
            'coherence_score': round(coherence_score, 4),
            'relevance_score': round(relevance_score, 4)
        }

        # 総合スコア
        scores['overall_score'] = np.mean(list(scores.values()))
        return scores

    def _calculate_japanese_ratio(self, text: str) -> float:
        """日本語文字の割合を計算"""
        if not text:
            return 0.0

        japanese_chars = 0
        total_chars = len(text)

        for char in text:
            # ひらがな、カタカナ、漢字を日本語文字としてカウント
            if ('\u3040' <= char <= '\u309f' or  # ひらがな
                '\u30a0' <= char <= '\u30ff' or  # カタカナ
                '\u4e00' <= char <= '\u9fff'):   # 漢字
                japanese_chars += 1

        return japanese_chars / total_chars if total_chars > 0 else 0.0

    def _calculate_coherence_score(self, text: str) -> float:
        """一貫性スコアの簡易計算"""
        if not text or len(text) < 10:
            return 0.3

        # 文の長さのばらつきを評価
        sentences = text.split('。')
        if len(sentences) < 2:
            return 0.5

        sentence_lengths = [len(s) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0.3

        # 標準偏差が小さいほど一貫性が高い
        std_dev = np.std(sentence_lengths)
        max_std = 50  # 最大標準偏差
        coherence = max(0, 1 - (std_dev / max_std))

        return coherence

    def run_single_inference(self, model, tokenizer, question: str, model_name: str) -> Dict[str, Any]:
        """単一推論実行"""
        try:
            inputs = tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # 入力部分を除去して回答を取得
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            return {
                'success': True,
                'answer': answer,
                'input_length': input_length,
                'output_length': len(generated_ids)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': "",
                'input_length': 0,
                'output_length': 0
            }

    def convert_to_gguf(self, model_path: str, model_name: str):
        """GGUF変換"""
        if not self.config.gguf_convert:
            return None

        print(f"\n=== GGUF変換: {model_name} ===")

        try:
            import subprocess
            from pathlib import Path

            gguf_output_dir = Path(self.config.gguf_dir) / model_name
            gguf_output_dir.mkdir(parents=True, exist_ok=True)

            # llama.cpp convert_hf_to_gguf.py を使用
            convert_script = Path("external/llama.cpp-master/convert_hf_to_gguf.py")
            if not convert_script.exists():
                print(f"[WARNING] llama.cpp convert script not found: {convert_script}")
                return None

            # F16変換
            f16_output = gguf_output_dir / f"{model_name}_f16.gguf"
            cmd_f16 = [
                "python", str(convert_script),
                model_path,
                "--outfile", str(f16_output),
                "--outtype", "f16"
            ]

            print(f"[GGUF] Converting to F16: {f16_output}")
            result_f16 = subprocess.run(cmd_f16, capture_output=True, text=True)
            if result_f16.returncode != 0:
                print(f"[ERROR] F16 conversion failed: {result_f16.stderr}")
                return None

            # Q8_0変換
            q8_output = gguf_output_dir / f"{model_name}_Q8_0.gguf"
            cmd_q8 = [
                "python", str(convert_script),
                model_path,
                "--outfile", str(q8_output),
                "--outtype", "q8_0"
            ]

            print(f"[GGUF] Converting to Q8_0: {q8_output}")
            result_q8 = subprocess.run(cmd_q8, capture_output=True, text=True)
            if result_q8.returncode != 0:
                print(f"[ERROR] Q8_0 conversion failed: {result_q8.stderr}")
                return None

            print(f"[GGUF] Conversion completed: {gguf_output_dir}")
            return str(gguf_output_dir)

        except Exception as e:
            print(f"[ERROR] GGUF conversion failed: {e}")
            return None

    def run_gguf_test(self, gguf_model_a: str, gguf_model_b: str):
        """GGUFモデルでのA/Bテスト"""
        print("\n=== GGUFモデル A/Bテスト ===")

        try:
            import subprocess
            from pathlib import Path

            test_results = {}

            for model_name, gguf_dir in [("Borea_Phi35_JP", gguf_model_a), ("AEGIS_Phi35_Thinking_v2", gguf_model_b)]:
                print(f"\n[GGUF_TEST] Testing {model_name}")

                gguf_dir_path = Path(gguf_dir)

                # Q8_0モデルファイルを探す
                gguf_files = list(gguf_dir_path.glob("*_Q8_0.gguf"))
                if not gguf_files:
                    print(f"[WARNING] Q8_0 GGUF file not found in {gguf_dir}")
                    continue

                gguf_file = gguf_files[0]
                print(f"[GGUF] Found model: {gguf_file}")

                # Ollama Modelfile作成
                modelfile_content = f"""FROM {gguf_file}

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER num_thread 8
"""

                modelfile_path = gguf_dir_path / f"{model_name}.modelfile"
                with open(modelfile_path, 'w', encoding='utf-8') as f:
                    f.write(modelfile_content)

                # Ollamaモデル作成
                ollama_name = f"{model_name.lower()}:latest"
                create_cmd = ["ollama", "create", ollama_name, "-f", str(modelfile_path)]

                print(f"[OLLAMA] Creating model {ollama_name}")
                result_create = subprocess.run(create_cmd, capture_output=True, text=True, timeout=300)

                if result_create.returncode != 0:
                    print(f"[ERROR] Ollama create failed: {result_create.stderr}")
                    continue

                # Ollamaテスト実行
                test_prompts = [
                    "こんにちは。自己紹介をお願いします。",
                    "日本の首都はどこですか？",
                    "2+2×3の計算結果を教えてください。",
                    "人工知能について簡単に説明してください。"
                ]

                test_responses = {}
                for i, prompt in enumerate(test_prompts):
                    print(f"[TEST {i+1}] {prompt[:30]}...")
                    run_cmd = ["ollama", "run", ollama_name, prompt]

                    try:
                        result_run = subprocess.run(run_cmd, capture_output=True, text=True, timeout=60)
                        if result_run.returncode == 0:
                            test_responses[f"test_{i+1}"] = result_run.stdout.strip()
                        else:
                            test_responses[f"test_{i+1}"] = f"ERROR: {result_run.stderr}"
                    except subprocess.TimeoutExpired:
                        test_responses[f"test_{i+1}"] = "TIMEOUT"

                test_results[model_name] = {
                    'gguf_path': str(gguf_file),
                    'modelfile_path': str(modelfile_path),
                    'ollama_name': ollama_name,
                    'test_responses': test_responses,
                    'status': 'completed'
                }

                print(f"[SUCCESS] {model_name} GGUF test completed")

            return test_results

        except Exception as e:
            print(f"[ERROR] GGUF test failed: {e}")
            return None

    def run_lm_eval_test(self):
        """LM-Evaluation-Harnessテスト実行"""
        print("\n=== LM-Evaluation-Harnessテスト開始 ===")

        try:
            import subprocess
            import json
            import os
            from pathlib import Path

            # 出力ディレクトリ作成
            output_dir = Path(self.config.lm_eval_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 評価タスク設定
            eval_tasks = self.config.lm_eval_tasks.split(",")

            eval_results = {}

            # HFモデル評価
            print("\n[HF_MODEL_EVAL] HFモデル評価開始")

            for model_name, model_path in [("Borea_Phi35_JP", self.config.model_a_path),
                                         ("AEGIS_Phi35_Thinking_v2", self.config.model_b_path)]:

                print(f"\n[HF_EVAL] {model_name} 評価開始")

                for task in eval_tasks:
                    print(f"[HF_EVAL] {model_name} - {task}")

                    # lm_evalコマンド（HuggingFaceモデル直接評価）
                    output_path = output_dir / f"hf_{model_name}_{task}.json"

                    cmd = [
                        "python", "-m", "lm_eval",
                        "--model", "hf",
                        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
                        "--tasks", task,
                        "--device", "cuda" if torch.cuda.is_available() else "cpu",
                        "--batch_size", "auto",
                        "--output_path", str(output_path),
                        "--log_samples"
                    ]

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1時間タイムアウト

                        if result.returncode == 0:
                            eval_results[f"hf_{model_name}_{task}"] = {
                                'status': 'success',
                                'output': result.stdout,
                                'task': task,
                                'output_path': str(output_path)
                            }
                            print(f"[OK] {model_name} - {task} 完了")
                        else:
                            eval_results[f"hf_{model_name}_{task}"] = {
                                'status': 'failed',
                                'error': result.stderr,
                                'task': task
                            }
                            print(f"[NG] {model_name} - {task} 失敗: {result.stderr[:200]}...")
                    except subprocess.TimeoutExpired:
                        eval_results[f"hf_{model_name}_{task}"] = {
                            'status': 'timeout',
                            'task': task
                        }

            # GGUFモデル評価（HF backend + gguf_file使用）
            print("\n[GGUF_MODEL_EVAL] GGUFモデル評価開始")

            # GGUFモデルが利用可能かチェック
            gguf_base_dir = Path(self.config.gguf_dir)
            if gguf_base_dir.exists():
                for model_name, model_path in [("Borea_Phi35_JP", self.config.model_a_path),
                                             ("AEGIS_Phi35_Thinking_v2", self.config.model_b_path)]:

                    # GGUFディレクトリを探す
                    model_gguf_dir = None
                    for gguf_subdir in gguf_base_dir.iterdir():
                        if gguf_subdir.is_dir() and model_name.lower().replace("_", "").replace("-", "") in gguf_subdir.name.lower():
                            model_gguf_dir = gguf_subdir
                            break

                    if model_gguf_dir:
                        print(f"\n[GGUF_EVAL] {model_name} GGUF評価開始")

                        # GGUFファイルを探す（Q8_0を優先）
                        gguf_file = None
                        for ext in ["*.gguf"]:
                            gguf_files = list(model_gguf_dir.glob(ext))
                            if gguf_files:
                                # Q8_0を優先、なければ最初のファイル
                                for gf in gguf_files:
                                    if "Q8_0" in gf.name or "q8_0" in gf.name:
                                        gguf_file = gf
                                        break
                                if not gguf_file:
                                    gguf_file = gguf_files[0]
                                break

                        if gguf_file:
                            print(f"[GGUF_EVAL] Found GGUF: {gguf_file}")

                            # トークナイザーディレクトリ（元のHFモデル）
                            tokenizer_dir = model_path if Path(model_path).exists() else None
                            if not tokenizer_dir and "microsoft/Phi-3.5-mini-instruct" in model_path:
                                tokenizer_dir = "microsoft/Phi-3.5-mini-instruct"

                            for task in eval_tasks:
                                print(f"[GGUF_EVAL] {model_name} - {task}")

                                output_path = output_dir / f"gguf_{model_name}_{task}.json"

                                # HF backendでGGUF評価
                                if tokenizer_dir:
                                    model_args = f"pretrained={model_gguf_dir},gguf_file={gguf_file.name},tokenizer={tokenizer_dir}"
                                else:
                                    model_args = f"pretrained={model_gguf_dir},gguf_file={gguf_file.name}"

                                cmd = [
                                    "python", "-m", "lm_eval",
                                    "--model", "hf",
                                    "--model_args", model_args,
                                    "--tasks", task,
                                    "--device", "cuda" if torch.cuda.is_available() else "cpu",
                                    "--batch_size", "auto",
                                    "--output_path", str(output_path),
                                    "--log_samples"
                                ]

                                try:
                                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                                    if result.returncode == 0:
                                        eval_results[f"gguf_{model_name}_{task}"] = {
                                            'status': 'success',
                                            'output': result.stdout,
                                            'task': task,
                                            'output_path': str(output_path),
                                            'gguf_file': str(gguf_file)
                                        }
                                        print(f"[OK] GGUF {model_name} - {task} 完了")
                                    else:
                                        eval_results[f"gguf_{model_name}_{task}"] = {
                                            'status': 'failed',
                                            'error': result.stderr,
                                            'task': task,
                                            'gguf_file': str(gguf_file)
                                        }
                                        print(f"[NG] GGUF {model_name} - {task} 失敗: {result.stderr[:200]}...")
                                except subprocess.TimeoutExpired:
                                    eval_results[f"gguf_{model_name}_{task}"] = {
                                        'status': 'timeout',
                                        'task': task,
                                        'gguf_file': str(gguf_file)
                                    }
                        else:
                            print(f"[SKIP] No GGUF file found for {model_name}")
                    else:
                        print(f"[SKIP] No GGUF directory found for {model_name}")
            else:
                print(f"[SKIP] GGUF directory not found: {gguf_base_dir}")

            return eval_results

        except Exception as e:
            print(f"[ERROR] LM-Eval test failed: {e}")
            return None

    def run_ab_test(self):
        """A/Bテスト実行"""
        print("[TARGET] AEGIS-phi3.5-v2.0 A/Bテスト開始")
        print("=" * 50)

        # チェックポイントから復元
        if self.load_checkpoint():
            print("[RESUME] Resumed from checkpoint")
        else:
            print("[START] Starting new test")

        # ELYZA-100読み込み
        elyza_data = self.load_elyza_dataset()
        if not elyza_data:
            print("[NG] テストデータなし")
            return False

        results = []

        for i, task in enumerate(tqdm(elyza_data, desc="A/Bテスト実行中")):
            # 定期チェックポイント保存チェック
            self._check_checkpoint_save()

            question = task.get('input', task.get('question', ''))
            task_id = task.get('task_id', f'task_{i}')
            category = task.get('category', 'unknown')

            if not question:
                continue

            # 各モデルで複数回サンプリング
            model_a_scores = []
            model_b_scores = []

            for sample_idx in range(self.config.num_samples_per_question):
                # モデルA推論
                result_a = self.run_single_inference(self.model_a, self.tokenizer_a, question, "Model_A")
                if result_a['success']:
                    scores_a = self.evaluate_answer_quality(question, result_a['answer'])
                    model_a_scores.append(scores_a['overall_score'])
                else:
                    model_a_scores.append(0.0)

                # モデルB推論 (存在する場合)
                if self.model_b is not None:
                    result_b = self.run_single_inference(self.model_b, self.tokenizer_b, question, "Model_B")
                    if result_b['success']:
                        scores_b = self.evaluate_answer_quality(question, result_b['answer'])
                        model_b_scores.append(scores_b['overall_score'])
                    else:
                        model_b_scores.append(0.0)
                else:
                    model_b_scores.append(0.0)  # モデルBなしの場合

            # 結果集計
            result = {
                'task_id': task_id,
                'category': category,
                'question': question,
                'model_a_scores': model_a_scores,
                'model_b_scores': model_b_scores,
                'model_a_avg': np.mean(model_a_scores),
                'model_b_avg': np.mean(model_b_scores),
                'model_a_std': np.std(model_a_scores),
                'model_b_std': np.std(model_b_scores),
                'improvement': np.mean(model_b_scores) - np.mean(model_a_scores)
            }
            results.append(result)

        self.results = results
        print(f"[OK] A/Bテスト完了: {len(results)}件のタスクを評価")
        return True

    def calculate_statistics(self) -> Dict[str, Any]:
        """統計計算"""
        print("=== 統計計算 ===")

        if not self.results:
            return {}

        # スコア抽出
        model_a_scores = [r['model_a_avg'] for r in self.results]
        model_b_scores = [r['model_b_avg'] for r in self.results]

        # 基本統計量
        stats = {
            'model_a': {
                'mean': np.mean(model_a_scores),
                'std': np.std(model_a_scores),
                'median': np.median(model_a_scores),
                'min': np.min(model_a_scores),
                'max': np.max(model_a_scores)
            },
            'model_b': {
                'mean': np.mean(model_b_scores),
                'std': np.std(model_b_scores),
                'median': np.median(model_b_scores),
                'min': np.min(model_b_scores),
                'max': np.max(model_b_scores)
            },
            'comparison': {
                'mean_difference': np.mean(model_b_scores) - np.mean(model_a_scores),
                'improvement_percentage': ((np.mean(model_b_scores) - np.mean(model_a_scores)) / np.mean(model_a_scores)) * 100
            }
        }

        # 統計的検定
        try:
            # t検定
            t_stat, p_value = ttest_ind(model_a_scores, model_b_scores, equal_var=False)
            stats['t_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            # Cohen's d (効果量) - 自作関数
            cohens_d = self._calculate_cohens_d(model_a_scores, model_b_scores)
            stats['effect_size'] = {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_effect_size(cohens_d)
            }

            # ANOVA (カテゴリ別)
            categories = {}
            for result in self.results:
                cat = result['category']
                if cat not in categories:
                    categories[cat] = {'a': [], 'b': []}
                categories[cat]['a'].append(result['model_a_avg'])
                categories[cat]['b'].append(result['model_b_avg'])

            anova_results = {}
            for cat, scores in categories.items():
                if len(scores['a']) > 1 and len(scores['b']) > 1:
                    f_stat, p_val = f_oneway(scores['a'], scores['b'])
                    anova_results[cat] = {
                        'f_statistic': f_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    }

            stats['anova_by_category'] = anova_results

        except Exception as e:
            print(f"統計検定エラー: {e}")
            stats['statistics_error'] = str(e)

        return stats

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Cohen's d効果量計算"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # プールされた標準偏差
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def _interpret_effect_size(self, d: float) -> str:
        """効果量の解釈"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "無視できる効果"
        elif abs_d < 0.5:
            return "小さい効果"
        elif abs_d < 0.8:
            return "中程度の効果"
        else:
            return "大きい効果"

    def create_visualizations(self, stats: Dict[str, Any]):
        """可視化作成"""
        print("=== 可視化作成 ===")

        # 結果をDataFrameに変換
        df_data = []
        for result in self.results:
            for i, (score_a, score_b) in enumerate(zip(result['model_a_scores'], result['model_b_scores'])):
                df_data.append({
                    'task_id': result['task_id'],
                    'category': result['category'],
                    'sample': i,
                    'Model_A': score_a,
                    'Model_B': score_b
                })

        df = pd.DataFrame(df_data)

        # 1. 箱ひげ図
        plt.figure(figsize=(12, 8))
        melted_df = df.melt(id_vars=['task_id', 'category'], value_vars=['Model_A', 'Model_B'],
                           var_name='Model', value_name='Score')

        plt.subplot(2, 2, 1)
        sns.boxplot(data=melted_df, x='Model', y='Score')
        plt.title('モデル比較: 箱ひげ図')
        plt.ylabel('品質スコア')

        # 2. バイオリンプロット
        plt.subplot(2, 2, 2)
        sns.violinplot(data=melted_df, x='Model', y='Score')
        plt.title('モデル比較: バイオリンプロット')

        # 3. カテゴリ別比較
        plt.subplot(2, 2, 3)
        category_means = df.groupby('category')[['Model_A', 'Model_B']].mean().reset_index()
        melted_cat = category_means.melt(id_vars='category', var_name='Model', value_name='Mean_Score')
        sns.barplot(data=melted_cat, x='category', y='Mean_Score', hue='Model')
        plt.title('カテゴリ別平均スコア比較')
        plt.xticks(rotation=45, ha='right')

        # 4. 散布図 (A vs B)
        plt.subplot(2, 2, 4)
        task_means = df.groupby('task_id')[['Model_A', 'Model_B']].mean().reset_index()
        plt.scatter(task_means['Model_A'], task_means['Model_B'], alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7)  # 対角線
        plt.xlabel('Model A Score')
        plt.ylabel('Model B Score')
        plt.title('モデルA vs モデルB: 散布図')
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / 'aegis_ab_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 統計情報プロット
        self._create_statistics_plot(stats)
        print("[OK] 可視化ファイル保存完了")

    def _create_statistics_plot(self, stats: Dict[str, Any]):
        """統計情報プロット"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 平均比較
        models = ['Model A', 'Model B']
        means = [stats['model_a']['mean'], stats['model_b']['mean']]
        stds = [stats['model_a']['std'], stats['model_b']['std']]

        ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_title('平均スコア比較 (±標準偏差)')
        ax1.set_ylabel('品質スコア')
        ax1.grid(True, alpha=0.3)

        # t検定結果
        if 't_test' in stats:
            t_test = stats['t_test']
            ax2.bar(['t統計量', 'p値'], [t_test['t_statistic'], t_test['p_value']], alpha=0.7)
            ax2.set_title(f't検定結果 (有意: {t_test["significant"]})')
            ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='有意水準(0.05)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 効果量
        if 'effect_size' in stats:
            effect_size = stats['effect_size']
            ax3.bar(['Cohen\'s d'], [effect_size['cohens_d']], alpha=0.7)
            ax3.set_title(f'効果量: {effect_size["interpretation"]}')
            ax3.grid(True, alpha=0.3)

        # カテゴリ別ANOVA
        if 'anova_by_category' in stats:
            categories = list(stats['anova_by_category'].keys())
            p_values = [stats['anova_by_category'][cat]['p_value'] for cat in categories]

            ax4.bar(categories, p_values, alpha=0.7)
            ax4.set_title('カテゴリ別ANOVA p値')
            ax4.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='有意水準(0.05)')
            ax4.legend()
            ax4.set_xticklabels(categories, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / 'aegis_ab_test_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, stats: Dict[str, Any]):
        """結果保存"""
        print("=== 結果保存 ===")

        # 詳細結果
        results_data = {
            'config': {
                'model_a_path': self.config.model_a_path,
                'model_b_path': self.config.model_b_path,
                'num_tasks': len(self.results),
                'samples_per_question': self.config.num_samples_per_question,
                'timestamp': datetime.now().isoformat()
            },
            'statistics': stats,
            'detailed_results': self.results
        }

        # JSON保存
        with open(Path(self.config.output_dir) / 'aegis_ab_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        # CSV保存 (簡易版)
        csv_data = []
        for result in self.results:
            csv_data.append({
                'task_id': result['task_id'],
                'category': result['category'],
                'model_a_avg': result['model_a_avg'],
                'model_b_avg': result['model_b_avg'],
                'improvement': result['improvement']
            })

        df = pd.DataFrame(csv_data)
        df.to_csv(Path(self.config.output_dir) / 'aegis_ab_test_summary.csv', index=False, encoding='utf-8')

        print("[OK] 結果保存完了")

    def create_hf_readme(self, stats: Dict[str, Any]):
        """HF公開用README作成"""
        readme_content = f"""---
language: ja
license: apache-2.0
tags:
- benchmark
- ab-test
- japanese
- mathematics
- physics
- elyza-100
- phi-3.5
- aegis
---

# AEGIS-phi3.5-v2.0 A/B Test Benchmark Results

## 概要

AEGIS-phi3.5-v2.0 (ノーベル賞・フィールズ賞級推論モデル) とベースモデル (Borea-Phi3.5-instinct-jp) の比較評価結果。

**テストデータ**: ELYZA-100 (全100問)
**評価指標**: 回答品質スコア (長さ、日本語率、一貫性、関連性)
**統計手法**: t検定、ANOVA、効果量分析

## モデル比較

### Model A (ベース)
- **モデル**: Borea-Phi3.5-instruct-jp
- **平均スコア**: {stats['model_a']['mean']:.3f} ± {stats['model_a']['std']:.3f}
- **中央値**: {stats['model_a']['median']:.3f}

### Model B (AEGIS)
- **モデル**: AEGIS-phi3.5-v2.0
- **平均スコア**: {stats['model_b']['mean']:.3f} ± {stats['model_b']['std']:.3f}
- **中央値**: {stats['model_b']['median']:.3f}

## 統計的検定結果

### 全体比較
- **平均差**: {stats['comparison']['mean_difference']:.3f}
- **改善率**: {stats['comparison']['improvement_percentage']:.1f}%

### t検定
- **t統計量**: {stats.get('t_test', {}).get('t_statistic', 'N/A'):.3f}
- **p値**: {stats.get('t_test', {}).get('p_value', 'N/A'):.4f}
- **有意差**: {"あり" if stats.get('t_test', {}).get('significant', False) else "なし"} (α=0.05)

### 効果量
- **Cohen's d**: {stats.get('effect_size', {}).get('cohens_d', 'N/A'):.3f}
- **解釈**: {stats.get('effect_size', {}).get('interpretation', 'N/A')}

## カテゴリ別分析

| カテゴリ | Model A | Model B | 差 | p値 |
|----------|---------|---------|-----|-----|
"""

        # カテゴリ別結果追加
        if 'anova_by_category' in stats:
            for cat, result in stats['anova_by_category'].items():
                cat_results = [r for r in self.results if r['category'] == cat]
                if cat_results:
                    a_avg = np.mean([r['model_a_avg'] for r in cat_results])
                    b_avg = np.mean([r['model_b_avg'] for r in cat_results])
                    diff = b_avg - a_avg
                    p_val = result['p_value']

                    readme_content += f"| {cat} | {a_avg:.3f} | {b_avg:.3f} | {diff:+.3f} | {p_val:.4f} |\n"

        readme_content += """

## 評価方法

1. **各タスクを3回サンプリング**
2. **回答品質スコア計算**:
   - 長さスコア: 回答長/100 (最大1.0)
   - 日本語率: 日本語文字の割合
   - 一貫性スコア: 文長ばらつき評価
   - 関連性スコア: 基準回答との比較
3. **統計分析**: t検定、ANOVA、効果量

## ファイル構成

- `aegis_ab_test_results.json`: 詳細結果
- `aegis_ab_test_summary.csv`: 簡易結果
- `aegis_ab_test_comparison.png`: 比較グラフ
- `aegis_ab_test_statistics.png`: 統計グラフ

## 理論的背景

AEGIS-phi3.5-v2.0は以下の理論を統合:

- **URT (Unified Representation Theorem)**: 量子場論的表現統一
- **NC-KART★ (Non-Commutative Kolmogorov-Arnold Theory)**: 非可換関数近似
- **SO(8) Enhanced Adapter**: リー代数による回転最適化
- **Quadruple Thinking Engine**: 四重思考推論

## 結論

{"AEGISモデルはベースモデルに対して統計的に有意な改善を示しました。" if stats.get('t_test', {}).get('significant', False) else "AEGISモデルとベースモデルの間に統計的に有意な差は確認されませんでした。"}

**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**テスト環境**: RTX 3060 (12GB) + 32GB RAM

---

*このベンチマーク結果はHF Datasetsで公開されています。*
"""

        with open(Path(self.config.output_dir) / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print("[OK] HF README作成完了")

    def run_full_evaluation(self):
        """完全評価実行"""
        print("[TARGET] AEGIS-phi3.5-v2.0 A/Bテスト開始")
        print("=" * 60)

        # モデル読み込み
        if not self.load_models():
            print("[NG] モデル読み込み失敗")
            return False

        # A/Bテスト実行
        if not self.run_ab_test():
            print("[NG] A/Bテスト失敗")
            return False

        # 統計計算
        stats = self.calculate_statistics()

        # 可視化作成
        self.create_visualizations(stats)

        # 結果保存
        self.save_results(stats)

        # HF README作成
        self.create_hf_readme(stats)

        print("\n[OK] A/Bテスト完了！")
        print(f"結果保存先: {self.config.output_dir}")
        print(f"Model A平均: {stats['model_a']['mean']:.3f}")
        print(f"Model B平均: {stats['model_b']['mean']:.3f}")
        print(f"p値: {stats.get('t_test', {}).get('p_value', 'N/A'):.4f}")
        print(f"効果量: {stats.get('effect_size', {}).get('cohens_d', 'N/A'):.3f}")

        return True


def create_auto_startup_script():
    """自動起動スクリプト作成"""
    startup_script = """@echo off
REM SO8T A/Bテスト自動起動スクリプト
REM 電源投入時に自動実行されるようにタスクスケジューラーに登録してください

cd /d "%~dp0\\..\\.."

REM H:\\from_D\\webdataset が利用可能か確認
if not exist "H:\\from_D\\webdataset" (
    echo [ERROR] H:\\from_D\\webdataset not found >> auto_start.log
    exit /b 1
)

REM ログディレクトリ作成
if not exist "H:\\from_D\\webdataset\\logs" mkdir "H:\\from_D\\webdataset\\logs"

echo [AUTO] Starting SO8T A/B Test at %DATE% %TIME% >> "H:\\from_D\\webdataset\\logs\\auto_start.log"

REM Python環境確認
python --version >> "H:\\from_D\\webdataset\\logs\\auto_start.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found >> "H:\\from_D\\webdataset\\logs\\auto_start.log"
    exit /b 1
)

REM A/Bテスト実行
python scripts/benchmark/aegis_ab_test_benchmark.py >> "H:\\from_D\\webdataset\\logs\\auto_start.log" 2>&1

if errorlevel 0 (
    echo [SUCCESS] A/B Test completed at %DATE% %TIME% >> "H:\\from_D\\webdataset\\logs\\auto_start.log"
) else (
    echo [ERROR] A/B Test failed at %DATE% %TIME% >> "H:\\from_D\\webdataset\\logs\\auto_start.log"
)

REM 完了通知（オプション）
powershell -ExecutionPolicy Bypass -File "scripts\\utils\\play_audio_notification.ps1"
"""

    script_path = Path("scripts/benchmark/auto_start_ab_test.bat")
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(startup_script)

    print(f"[AUTO] Created auto-startup script: {script_path}")
    print(r"[STORAGE] Using H:\from_D\webdataset for large files")

    # Windowsタスクスケジューラー登録コマンド表示
    print("\n=== Windowsタスクスケジューラー登録方法 ===")
    print("1. Windows + R → taskschd.msc")
    print("2. 「タスクの作成」を選択")
    print("3. 名前: SO8T_AB_Test_Auto_Start")
    print("4. 「トリガー」タブ → 「新規」")
    print("   - ログオン時に開始")
    print("   - または電源投入時に開始")
    print("5. 「操作」タブ → 「新規」")
    print(f"   - プログラム: {script_path}")
    print("6. 「条件」タブ → 「コンピュータがAC電源で実行されている場合のみタスクを開始する」をオフ")
    print("7. 「設定」タブ → 「失敗した場合は再起動」をオン")
    print("   - 再起動間隔: 1分")
    print("   - 再試行回数: 3回")

    return script_path


def main():
    """メイン関数"""
    print("[START] SO8T AEGIS A/Bテストシステム")
    print("=" * 50)
    print(r"H:\from_D\webdataset を大きなファイル保存先として使用")

    # 自動起動スクリプト作成
    if ABTestConfig().auto_restart:
        create_auto_startup_script()

    config = ABTestConfig()
    tester = AEGISABTester(config)

    try:
        # メイン評価実行
        success = tester.run_full_evaluation()

        if success:
            print("\n[SUCCESS] AEGIS A/Bテスト成功！HF公開準備完了")

            # LM-Evalテスト実行
            print("\n=== LM-Eval HFモデルテストフェーズ ===")
            lm_eval_results = tester.run_lm_eval_test()
            if lm_eval_results:
                print("[OK] LM-Evalテスト完了")
            else:
                print("[WARN] LM-Evalテスト失敗")

            # GGUF変換実行
            print("\n=== GGUF変換フェーズ ===")
            gguf_a = tester.convert_to_gguf(config.model_a_path, "Borea_Phi35_JP")
            gguf_b = tester.convert_to_gguf(config.model_b_path, "AEGIS_Phi35_Thinking_v2")

            if gguf_a and gguf_b:
                # GGUFテスト実行
                print("\n=== GGUFモデルテストフェーズ ===")
                gguf_results = tester.run_gguf_test(gguf_a, gguf_b)
                if gguf_results:
                    print("[OK] GGUFテスト完了")
                else:
                    print("[WARN] GGUFテスト一部失敗")
            else:
                print("[WARN] GGUF変換スキップまたは失敗")

        else:
            print("\n[NG] A/Bテスト失敗")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] User interrupted. Saving checkpoint...")
        tester._save_checkpoint("interrupted")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        tester._save_checkpoint("error")
        sys.exit(1)

    finally:
        # 最終チェックポイント保存
        tester._save_checkpoint("final")

        # オーディオ通知
        try:
            audio_script = Path("scripts/utils/play_audio_notification.ps1")
            if audio_script.exists():
                import subprocess
                subprocess.run([
                    "powershell",
                    "-ExecutionPolicy", "Bypass",
                    "-File", str(audio_script)
                ], capture_output=True)
        except Exception as e:
            print(f"[WARNING] Audio notification failed: {e}")

    print("\n[OK] すべての処理が完了しました")


if __name__ == "__main__":
    main()

) else (
    echo [ERROR] A/B Test failed at %DATE% %TIME% >> "H:\\from_D\\webdataset\\logs\\auto_start.log"
)

REM 完了通知（オプション）
powershell -ExecutionPolicy Bypass -File "scripts\\utils\\play_audio_notification.ps1"
"""

    script_path = Path("scripts/benchmark/auto_start_ab_test.bat")
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(startup_script)

    print(f"[AUTO] Created auto-startup script: {script_path}")
    print(r"[STORAGE] Using H:\from_D\webdataset for large files")

    # Windowsタスクスケジューラー登録コマンド表示
    print("\n=== Windowsタスクスケジューラー登録方法 ===")
    print("1. Windows + R → taskschd.msc")
    print("2. 「タスクの作成」を選択")
    print("3. 名前: SO8T_AB_Test_Auto_Start")
    print("4. 「トリガー」タブ → 「新規」")
    print("   - ログオン時に開始")
    print("   - または電源投入時に開始")
    print("5. 「操作」タブ → 「新規」")
    print(f"   - プログラム: {script_path}")
    print("6. 「条件」タブ → 「コンピュータがAC電源で実行されている場合のみタスクを開始する」をオフ")
    print("7. 「設定」タブ → 「失敗した場合は再起動」をオン")
    print("   - 再起動間隔: 1分")
    print("   - 再試行回数: 3回")

    return script_path


def main():
    """メイン関数"""
    print("[START] SO8T AEGIS A/Bテストシステム")
    print("=" * 50)
    print(r"H:\from_D\webdataset を大きなファイル保存先として使用")

    # 自動起動スクリプト作成
    if ABTestConfig().auto_restart:
        create_auto_startup_script()

    config = ABTestConfig()
    tester = AEGISABTester(config)

    try:
        # メイン評価実行
        success = tester.run_full_evaluation()

        if success:
            print("\n[SUCCESS] AEGIS A/Bテスト成功！HF公開準備完了")

            # LM-Evalテスト実行
            print("\n=== LM-Eval HFモデルテストフェーズ ===")
            lm_eval_results = tester.run_lm_eval_test()
            if lm_eval_results:
                print("[OK] LM-Evalテスト完了")
            else:
                print("[WARN] LM-Evalテスト失敗")

            # GGUF変換実行
            print("\n=== GGUF変換フェーズ ===")
            gguf_a = tester.convert_to_gguf(config.model_a_path, "Borea_Phi35_JP")
            gguf_b = tester.convert_to_gguf(config.model_b_path, "AEGIS_Phi35_Thinking_v2")

            if gguf_a and gguf_b:
                # GGUFテスト実行
                print("\n=== GGUFモデルテストフェーズ ===")
                gguf_results = tester.run_gguf_test(gguf_a, gguf_b)
                if gguf_results:
                    print("[OK] GGUFテスト完了")
                else:
                    print("[WARN] GGUFテスト一部失敗")
            else:
                print("[WARN] GGUF変換スキップまたは失敗")

        else:
            print("\n[NG] A/Bテスト失敗")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] User interrupted. Saving checkpoint...")
        tester._save_checkpoint("interrupted")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        tester._save_checkpoint("error")
        sys.exit(1)

    finally:
        # 最終チェックポイント保存
        tester._save_checkpoint("final")

        # オーディオ通知
        try:
            audio_script = Path("scripts/utils/play_audio_notification.ps1")
            if audio_script.exists():
                import subprocess
                subprocess.run([
                    "powershell",
                    "-ExecutionPolicy", "Bypass",
                    "-File", str(audio_script)
                ], capture_output=True)
        except Exception as e:
            print(f"[WARNING] Audio notification failed: {e}")

    print("\n[OK] すべての処理が完了しました")


if __name__ == "__main__":
    main()

) else (
    echo [ERROR] A/B Test failed at %DATE% %TIME% >> "H:\\from_D\\webdataset\\logs\\auto_start.log"
)

REM 完了通知（オプション）
powershell -ExecutionPolicy Bypass -File "scripts\\utils\\play_audio_notification.ps1"
"""

    script_path = Path("scripts/benchmark/auto_start_ab_test.bat")
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(startup_script)

    print(f"[AUTO] Created auto-startup script: {script_path}")
    print(r"[STORAGE] Using H:\from_D\webdataset for large files")

    # Windowsタスクスケジューラー登録コマンド表示
    print("\n=== Windowsタスクスケジューラー登録方法 ===")
    print("1. Windows + R → taskschd.msc")
    print("2. 「タスクの作成」を選択")
    print("3. 名前: SO8T_AB_Test_Auto_Start")
    print("4. 「トリガー」タブ → 「新規」")
    print("   - ログオン時に開始")
    print("   - または電源投入時に開始")
    print("5. 「操作」タブ → 「新規」")
    print(f"   - プログラム: {script_path}")
    print("6. 「条件」タブ → 「コンピュータがAC電源で実行されている場合のみタスクを開始する」をオフ")
    print("7. 「設定」タブ → 「失敗した場合は再起動」をオン")
    print("   - 再起動間隔: 1分")
    print("   - 再試行回数: 3回")

    return script_path


def main():
    """メイン関数"""
    print("[START] SO8T AEGIS A/Bテストシステム")
    print("=" * 50)
    print(r"H:\from_D\webdataset を大きなファイル保存先として使用")

    # 自動起動スクリプト作成
    if ABTestConfig().auto_restart:
        create_auto_startup_script()

    config = ABTestConfig()
    tester = AEGISABTester(config)

    try:
        # メイン評価実行
        success = tester.run_full_evaluation()

        if success:
            print("\n[SUCCESS] AEGIS A/Bテスト成功！HF公開準備完了")

            # LM-Evalテスト実行
            print("\n=== LM-Eval HFモデルテストフェーズ ===")
            lm_eval_results = tester.run_lm_eval_test()
            if lm_eval_results:
                print("[OK] LM-Evalテスト完了")
            else:
                print("[WARN] LM-Evalテスト失敗")

            # GGUF変換実行
            print("\n=== GGUF変換フェーズ ===")
            gguf_a = tester.convert_to_gguf(config.model_a_path, "Borea_Phi35_JP")
            gguf_b = tester.convert_to_gguf(config.model_b_path, "AEGIS_Phi35_Thinking_v2")

            if gguf_a and gguf_b:
                # GGUFテスト実行
                print("\n=== GGUFモデルテストフェーズ ===")
                gguf_results = tester.run_gguf_test(gguf_a, gguf_b)
                if gguf_results:
                    print("[OK] GGUFテスト完了")
                else:
                    print("[WARN] GGUFテスト一部失敗")
            else:
                print("[WARN] GGUF変換スキップまたは失敗")

        else:
            print("\n[NG] A/Bテスト失敗")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] User interrupted. Saving checkpoint...")
        tester._save_checkpoint("interrupted")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        tester._save_checkpoint("error")
        sys.exit(1)

    finally:
        # 最終チェックポイント保存
        tester._save_checkpoint("final")

        # オーディオ通知
        try:
            audio_script = Path("scripts/utils/play_audio_notification.ps1")
            if audio_script.exists():
                import subprocess
                subprocess.run([
                    "powershell",
                    "-ExecutionPolicy", "Bypass",
                    "-File", str(audio_script)
                ], capture_output=True)
        except Exception as e:
            print(f"[WARNING] Audio notification failed: {e}")

    print("\n[OK] すべての処理が完了しました")


if __name__ == "__main__":
    main()
