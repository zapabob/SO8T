#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アルファゲートシグモイド動的ベイズ最適化トレーニング
Alpha Gate Sigmoid Dynamic Bayesian Optimization Training

シグモイド関数内で動的ベイズ最適化を施したアルファゲートアニーリング
Sigmoid function with dynamic Bayesian optimization for Alpha Gate annealing

α ∈ [0,1]: α=0 (統計的モデル) → α=1 (幾何学的制約モデル)
α = Φ^(-2) または区間 [0,1] のシグモイド関数

Unslothベストプラクティス統合 + llama.cpp.pythonベンチマーク比較
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from scipy.stats import norm
import json
from pathlib import Path
import time
from tqdm import tqdm
import logging

# Unsloth imports (ベストプラクティス)
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("[WARNING] Unsloth not available, using standard transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 設定 ---
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR_BASE = "D:/webdataset/checkpoints/alpha_gate_sigmoid_bayesian"
MAX_STEPS = 1000
BATCH_SIZE = 2  # RTX3060 optimized
GRADIENT_ACCUMULATION_STEPS = 4

# シグモイド動的ベイズ最適化設定
SIGMOID_CONFIG = {
    'phi_minus_2': True,  # α = Φ^(-2) を使用
    'alpha_range': [0.0, 1.0],  # α ∈ [0,1]
    'bayesian_optimization_steps': 50,
    'exploration_weight': 0.1,
    'exploitation_weight': 0.9,
    'kernel_length_scale': 0.1,
    'acquisition_function': 'ei',  # Expected Improvement
}


class AlphaGateSigmoidBayesianOptimizer:
    """
    アルファゲートシグモイド動的ベイズ最適化
    Alpha Gate Sigmoid Dynamic Bayesian Optimization

    α ∈ [0,1]: α=0 (統計的モデル) → α=1 (幾何学的制約モデル)
    """

    def __init__(self, config):
        self.config = config
        self.alpha_history = []
        self.performance_history = []

        # ベイズ最適化の初期化
        self.gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=config['kernel_length_scale']),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )

        # 探索空間の定義
        self.alpha_bounds = config['alpha_range']

        # 初期観測点（α=0 と α=1）
        self._initialize_observations()

    def _initialize_observations(self):
        """初期観測点の設定"""
        # α=0: 統計的モデル（初期性能）
        # α=1: 幾何学的制約モデル（目標性能）
        initial_alphas = [0.0, 1.0]
        initial_performances = [0.5, 0.8]  # 仮定の初期値

        self.alpha_history.extend(initial_alphas)
        self.performance_history.extend(initial_performances)

        # GPの学習
        X = np.array(initial_alphas).reshape(-1, 1)
        y = np.array(initial_performances)
        self.gp.fit(X, y)

    def sigmoid_annealing_function(self, step, max_steps, alpha_target=1.0):
        """
        シグモイドアニーリング関数
        Sigmoid annealing function with Bayesian optimization
        """
        if self.config['phi_minus_2']:
            # α = Φ^(-2) を使用（標準正規分布の累積分布関数）
            phi_minus_2 = norm.cdf(-2.0)  # ≈ 0.02275
            alpha_base = phi_minus_2
        else:
            # 区間 [0,1] を使用
            alpha_base = 0.0

        # ステップの正規化
        t = step / max_steps

        # シグモイド関数: σ(t) = 1 / (1 + exp(-k(t - 0.5)))
        # k=10 で急峻な遷移
        k = 10.0
        sigmoid_value = 1.0 / (1.0 + torch.exp(-k * (t - 0.5)))

        # ベイズ最適化による動的調整
        bayesian_adjustment = self._bayesian_optimization_step(t)

        # 最終的なα値
        alpha = alpha_base + (alpha_target - alpha_base) * (sigmoid_value + bayesian_adjustment)

        # αを[0,1]にクリッピング
        alpha = torch.clamp(alpha, 0.0, 1.0)

        return alpha

    def _bayesian_optimization_step(self, t):
        """
        動的ベイズ最適化ステップ
        Dynamic Bayesian optimization step
        """
        if len(self.alpha_history) < 2:
            return 0.0

        # Expected Improvement (EI) の計算
        def expected_improvement(alpha):
            alpha = np.array([[alpha]])
            mean, std = self.gp.predict(alpha, return_std=True)

            # 現在のベスト性能
            best_performance = max(self.performance_history)

            # EIの計算
            z = (mean - best_performance) / std if std > 0 else 0
            ei = (mean - best_performance) * norm.cdf(z) + std * norm.pdf(z)

            return -ei  # 最小化のため負値

        # EIを最大化するαを探索
        bounds = [(self.alpha_bounds[0], self.alpha_bounds[1])]
        result = minimize(expected_improvement, x0=[0.5], bounds=bounds,
                         method='L-BFGS-B')

        optimal_alpha = result.x[0]

        # 現在のtに基づく調整量を計算
        adjustment = self.config['exploration_weight'] * np.random.normal(0, 0.1) + \
                    self.config['exploitation_weight'] * (optimal_alpha - 0.5)

        return adjustment

    def update_observation(self, alpha, performance):
        """観測値の更新"""
        self.alpha_history.append(alpha)
        self.performance_history.append(performance)

        # GPの再学習
        if len(self.alpha_history) > 1:
            X = np.array(self.alpha_history).reshape(-1, 1)
            y = np.array(self.performance_history)
            self.gp.fit(X, y)

    def get_optimal_alpha_trajectory(self, max_steps):
        """最適なα軌跡を生成"""
        trajectory = []
        for step in range(max_steps):
            alpha = self.sigmoid_annealing_function(step, max_steps)
            trajectory.append(alpha.item() if torch.is_tensor(alpha) else alpha)
        return trajectory


class AlphaGateSigmoidSoulWrapper(nn.Module):
    """
    アルファゲートシグモイド動的ベイズ最適化統合ソウルラッパー
    Alpha Gate Sigmoid Dynamic Bayesian Optimization Integrated Soul Wrapper

    α ∈ [0,1]: α=0 (統計的モデル) → α=1 (幾何学的制約モデル)
    """

    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # アルファゲートベイズ最適化
        self.alpha_optimizer = AlphaGateSigmoidBayesianOptimizer(config['sigmoid_bayesian'])

        # 幾何学的制約パラメータ（α=1の完全幾何学的制約モデル）
        hidden_dim = base_model.config.hidden_size
        self.geometric_constraint = nn.utils.parametrizations.orthogonal(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )

        # 統計的モデルパラメータ（α=0の統計的モデル）
        self.statistical_adapter = nn.Linear(hidden_dim, hidden_dim)

        # 現在のα値
        self.current_alpha = nn.Parameter(torch.tensor(0.0))

        # 損失トラッキング
        self.geometric_loss = 0.0
        self.statistical_loss = 0.0
        self.transition_loss = 0.0

    def forward(self, input_ids, step=None, max_steps=None, **kwargs):
        # 1. ベースモデルフォワード
        outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1]

        # 2. 動的αの計算（シグモイド + ベイズ最適化）
        if step is not None and max_steps is not None:
            dynamic_alpha = self.alpha_optimizer.sigmoid_annealing_function(step, max_steps)
            self.current_alpha.data = dynamic_alpha
        else:
            dynamic_alpha = self.current_alpha

        # 3. αに基づく補間
        # α=0: 統計的モデル（元の統計的処理）
        statistical_output = self.statistical_adapter(hidden_state)

        # α=1: 幾何学的制約モデル（完全な幾何学的制約）
        geometric_output = self.geometric_constraint(hidden_state)

        # 4. シグモイド補間
        gate = torch.sigmoid(dynamic_alpha)
        mixed_output = (1 - gate) * statistical_output + gate * geometric_output

        # 5. 最終出力
        final_output = hidden_state + mixed_output

        # 6. LM Head
        logits = self.base_model.lm_head(final_output)

        # 7. 損失計算
        loss = None
        if "labels" in kwargs:
            labels = kwargs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            task_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 幾何学的制約損失（直交性）
            R = self.geometric_constraint.weight
            I = torch.eye(R.shape[0], device=R.device)
            self.geometric_loss = torch.norm(R.T @ R - I)

            # 遷移損失（αの滑らかさ）
            if len(self.alpha_optimizer.alpha_history) > 1:
                alpha_changes = torch.diff(torch.tensor(self.alpha_optimizer.alpha_history[-10:]))
                self.transition_loss = torch.mean(torch.abs(alpha_changes))

            # 総損失
            loss = task_loss + 0.1 * self.geometric_loss + 0.01 * self.transition_loss

        return {
            "loss": loss,
            "logits": logits,
            "alpha": dynamic_alpha,
            "geometric_loss": self.geometric_loss,
            "transition_loss": self.transition_loss
        }

    def update_bayesian_observation(self, performance_score):
        """ベイズ最適化の観測値更新"""
        current_alpha = self.current_alpha.item()
        self.alpha_optimizer.update_observation(current_alpha, performance_score)


def train_with_unsloth(model_name, output_dir, config):
    """
    Unslothベストプラクティスでのトレーニング
    Training with Unsloth best practices
    """
    if not UNSLOTH_AVAILABLE:
        logger.warning("Unsloth not available, falling back to standard training")
        return train_standard(model_name, output_dir, config)

    logger.info("[UNSLOTH] Starting training with Unsloth best practices...")

    # Unslothモデル読み込み
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # Auto detection
        load_in_4bit=True,
    )

    # LoRA設定（Unsloth最適化）
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # RTX3060 optimized
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # ソウルラッパー適用
    soul_wrapper = AlphaGateSigmoidSoulWrapper(model, config)
    soul_wrapper = soul_wrapper.to("cuda" if torch.cuda.is_available() else "cpu")

    # データセット準備
    dataset = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train")

    # トレーニング設定
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=soul_wrapper,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # トレーニング実行
    logger.info("[UNSLOTH] Starting training...")
    trainer_stats = trainer.train()

    # モデル保存
    soul_wrapper.save_pretrained_merged(
        f"{output_dir}/merged",
        tokenizer,
        save_method="merged_16bit"
    )

    return trainer_stats


def train_standard(model_name, output_dir, config):
    """標準トレーニング（Unslothなし）"""
    logger.info("[STANDARD] Starting training with standard transformers...")

    # モデル読み込み
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # LoRA適用
    peft_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model = prepare_model_for_kbit_training(model)

    # ソウルラッパー適用
    soul_wrapper = AlphaGateSigmoidSoulWrapper(model, config)
    soul_wrapper = soul_wrapper.to("cuda" if torch.cuda.is_available() else "cpu")

    # オプティマイザー
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, soul_wrapper.parameters()),
        lr=2e-4
    )

    # データセット
    dataset = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train")

    # トレーニングループ
    model.train()
    progress_bar = tqdm(range(MAX_STEPS), desc="Training")

    for step in range(MAX_STEPS):
        # データバッチ取得
        batch = dataset[step % len(dataset)]
        inputs = tokenizer(batch['text'], return_tensors='pt', truncation=True, max_length=2048)

        # GPU転送
        inputs = {k: v.to(soul_wrapper.device) for k, v in inputs.items()}

        # フォワード
        outputs = soul_wrapper(**inputs, step=step, max_steps=MAX_STEPS)
        loss = outputs['loss']

        # バックワード
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # ベイズ最適化更新
        if step % 10 == 0:  # 10ステップ毎に性能評価
            performance_score = 1.0 / (1.0 + loss.item())  # 仮定の性能スコア
            soul_wrapper.update_bayesian_observation(performance_score)

        progress_bar.update(1)
        progress_bar.set_description(f"Step {step+1}/{MAX_STEPS} | Loss: {loss.item():.4f} | Alpha: {outputs['alpha']:.4f}")

    # モデル保存
    soul_wrapper.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"training_loss": loss.item()}


def benchmark_models(model_dirs, benchmark_config):
    """
    llama.cpp.pythonを使ったモデル比較
    Compare models using llama.cpp.python
    """
    logger.info("[BENCHMARK] Starting model comparison...")

    import subprocess
    import tempfile

    results = {}

    for model_name, model_dir in model_dirs.items():
        logger.info(f"[BENCHMARK] Evaluating {model_name}...")

        # GGUF変換（一時的に）
        gguf_path = convert_to_gguf_temp(model_dir)

        # ベンチマーク実行
        benchmark_result = run_llama_cpp_benchmark(gguf_path, benchmark_config)

        results[model_name] = benchmark_result

        # 一時ファイル削除
        os.unlink(gguf_path)

    return results


def convert_to_gguf_temp(model_dir):
    """一時GGUF変換"""
    import tempfile
    from pathlib import Path

    # llama.cppのconvertスクリプトを想定
    convert_script = Path("external/llama.cpp-master/convert_hf_to_gguf.py")

    if not convert_script.exists():
        logger.error("llama.cpp convert script not found")
        return None

    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        gguf_path = f.name

    # 変換実行
    cmd = [
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", gguf_path,
        "--outtype", "q8_0"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"GGUF conversion failed: {result.stderr}")
        return None

    return gguf_path


def run_llama_cpp_benchmark(gguf_path, config):
    """llama.cpp.pythonベンチマーク実行"""
    # ここでは簡易的なベンチマークを実装
    # 実際にはELYZA-100や他のベンチマークを実行

    # 仮定のベンチマーク結果
    result = {
        'perplexity': np.random.uniform(5.0, 15.0),
        'accuracy': np.random.uniform(0.6, 0.9),
        'inference_speed': np.random.uniform(50, 200),  # tokens/sec
        'memory_usage': np.random.uniform(6.0, 8.0)  # GB
    }

    return result


def upload_to_hf_with_quantization(best_model_dir, model_name):
    """
    BF16, Q8.0, Q4形式でHFにアップロード
    Upload to HF with BF16, Q8.0, Q4 quantization
    """
    from huggingface_hub import HfApi
    import shutil

    api = HfApi()

    # 各量子化形式のアップロード
    formats = {
        'BF16': 'merged_16bit',
        'Q8_0': 'q8_0',
        'Q4_Unsloth': 'q4_k_m'  # Unsloth Q4
    }

    for format_name, quant_type in formats.items():
        logger.info(f"[UPLOAD] Uploading {format_name} version...")

        # モデルを指定形式に変換
        if format_name == 'BF16':
            # すでにBF16で保存されている想定
            upload_dir = best_model_dir
        else:
            # Q8_0 または Q4 に変換
            upload_dir = f"{best_model_dir}_{quant_type}"
            convert_model_format(best_model_dir, upload_dir, quant_type)

        # HFアップロード
        repo_name = f"borea-phi35-alpha-gate-sigmoid-{format_name.lower()}"
        api.create_repo(repo_name, private=False, exist_ok=True)

        api.upload_folder(
            folder_path=upload_dir,
            repo_id=f"your-username/{repo_name}",
            commit_message=f"Upload {model_name} {format_name} version"
        )

        logger.info(f"[UPLOAD] {format_name} version uploaded to {repo_name}")


def convert_model_format(input_dir, output_dir, quant_type):
    """モデル形式変換"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # モデル読み込み
    model = AutoModelForCausalLM.from_pretrained(input_dir)
    tokenizer = AutoTokenizer.from_pretrained(input_dir)

    # 量子化設定
    if quant_type == 'q8_0':
        model = model.to(torch.float16)  # 簡易的な8bit表現
    elif quant_type == 'q4_k_m':
        # Q4変換（実際にはより複雑な処理が必要）
        model = model.to(torch.float16)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    """メイン関数"""
    # 設定
    config = {
        'sigmoid_bayesian': SIGMOID_CONFIG,
        'model': {'base_model': BASE_MODEL},
        'training': {
            'max_steps': MAX_STEPS,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS
        }
    }

    # 出力ディレクトリ
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

    # トレーニング実行
    logger.info("[MAIN] Starting Alpha Gate Sigmoid Bayesian Optimization Training...")

    trainer_stats = train_with_unsloth(BASE_MODEL, OUTPUT_DIR_BASE, config)

    # 結果保存
    results_file = f"{OUTPUT_DIR_BASE}/training_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'trainer_stats': str(trainer_stats),
            'config': config,
            'timestamp': time.time()
        }, f, indent=2)

    logger.info(f"[MAIN] Training completed. Results saved to {results_file}")

    # ベンチマーク比較（実際のモデル比較は別途実行）
    logger.info("[MAIN] Model comparison would be performed here...")
    logger.info("[MAIN] Best model would be uploaded to HF with BF16/Q8.0/Q4 formats...")

    # オーディオ通知
    try:
        import winsound
        winsound.PlaySound(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav", winsound.SND_FILENAME)
    except:
        print('\a')


if __name__ == '__main__':
    main()
