#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imatrixデータ収集スクリプト
GGUF量子化のための重要度行列計算
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImatrixCollector:
    """
    imatrixデータ収集クラス
    """

    def __init__(self, model_path: str, output_path: str, samples: int = 100000):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.samples = samples

        # モデル読み込み
        logger.info(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # imatrix計算用のデータ構造
        self.parameter_importance = {}

    def collect_imatrix_data(self):
        """imatrixデータ収集実行"""
        logger.info(f"Starting imatrix data collection with {self.samples} samples")

        # フック設定
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._activation_hook(mod, inp, out, name)
                )
                hooks.append(hook)

        try:
            # サンプル生成と処理
            self._process_samples()

            # imatrix計算
            self._calculate_imatrix()

            # 保存
            self._save_imatrix()

        finally:
            # フック解除
            for hook in hooks:
                hook.remove()

        logger.info(f"imatrix data collection completed: {self.output_path}")

    def _activation_hook(self, module, input_tensor, output_tensor, layer_name: str):
        """活性化フック"""
        # 活性化データの収集
        activations = output_tensor.detach().cpu().numpy()

        # 重要度計算のための統計情報収集
        if layer_name not in self.parameter_importance:
            self.parameter_importance[layer_name] = {
                "activations": [],
                "weights": module.weight.detach().cpu().numpy()
            }

        # メモリ効率のため、サンプル数を制限
        if len(self.parameter_importance[layer_name]["activations"]) < 1000:
            self.parameter_importance[layer_name]["activations"].append(
                np.mean(np.abs(activations), axis=(0, 1))  # 平均絶対活性化
            )

    def _process_samples(self):
        """サンプル処理"""
        logger.info("Processing samples for imatrix calculation")

        # サンプルテキスト生成（実際の使用時はデータセットから読み込み）
        sample_texts = self._generate_sample_texts()

        with torch.no_grad():
            for i, text in enumerate(tqdm(sample_texts, desc="Processing samples")):
                if i >= self.samples:
                    break

                # トークナイズ
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # 順伝播
                outputs = self.model(**inputs)

                # 定期的にメモリ解放
                if i % 100 == 0:
                    torch.cuda.empty_cache()

    def _generate_sample_texts(self) -> list:
        """サンプルテキスト生成"""
        # 実際の使用時はデータセットから読み込む
        # ここではデモ用のサンプルを生成

        base_texts = [
            # 数学的推論
            "数学の問題を解く：方程式 2x + 3 = 7 の解を求めよ。",
            "証明：すべての自然数 n に対して、n^2 + n は偶数であることを示せ。",
            "微積分：関数 f(x) = x^2 の導関数を求めよ。",

            # 科学的推論
            "物理：ニュートンの運動法則を説明せよ。",
            "化学：水の分子式は何ですか？",
            "生物：DNAの構造について説明せよ。",

            # 一般言語理解
            "日本国憲法の第一条の内容を説明せよ。",
            "経済学：需要と供給の関係について述べよ。",
            "歴史：第二次世界大戦の原因を分析せよ。",

            # プログラミング
            "Pythonでフィボナッチ数列を生成する関数を書け。",
            "アルゴリズム：クイックソートの仕組みを説明せよ。",
            "データ構造：スタックとキューの違いは何ですか？",

            # 論理的推論
            "論理パズル：3つの箱があり、それぞれに異なる果物が入っている場合...",
            "確率：コインを3回投げたときの表が出る回数の期待値を求めよ。",
            "統計：正規分布の性質について説明せよ。"
        ]

        # サンプル拡張
        sample_texts = []
        for text in base_texts:
            sample_texts.append(text)
            # バリエーション生成
            sample_texts.append(f"詳細に説明：{text}")
            sample_texts.append(f"ステップバイステップで：{text}")
            sample_texts.append(f"例を挙げて：{text}")

        # 目標サンプル数に達するまで繰り返し
        extended_texts = []
        while len(extended_texts) < self.samples:
            extended_texts.extend(sample_texts)

        return extended_texts[:self.samples]

    def _calculate_imatrix(self):
        """imatrix計算"""
        logger.info("Calculating imatrix values")

        imatrix_data = {}

        for layer_name, data in self.parameter_importance.items():
            activations = np.array(data["activations"])
            weights = data["weights"]

            if len(activations) == 0:
                logger.warning(f"No activation data for layer {layer_name}")
                continue

            # 活性化の統計量
            activation_mean = np.mean(activations, axis=0)
            activation_std = np.std(activations, axis=0) + 1e-8  # ゼロ除算防止

            # 重要度スコア計算
            importance_scores = activation_mean / activation_std

            # レイヤーの重要度行列として保存
            imatrix_data[layer_name] = {
                "importance_scores": importance_scores.tolist(),
                "activation_stats": {
                    "mean": activation_mean.tolist(),
                    "std": activation_std.tolist(),
                    "samples": len(activations)
                },
                "weight_shape": weights.shape
            }

        self.imatrix_data = imatrix_data
        logger.info(f"Calculated imatrix for {len(imatrix_data)} layers")

    def _save_imatrix(self):
        """imatrix保存"""
        # 出力ディレクトリ作成
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # imatrixデータ保存
        imatrix_output = {
            "metadata": {
                "model_path": str(self.model_path),
                "samples_processed": self.samples,
                "collection_date": time.time(),
                "format_version": "1.0"
            },
            "layers": self.imatrix_data
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(imatrix_output, f, indent=2, ensure_ascii=False)

        logger.info(f"imatrix data saved to {self.output_path}")

        # 統計情報表示
        total_parameters = sum(
            np.prod(layer_data["weight_shape"])
            for layer_data in self.imatrix_data.values()
        )

        protected_parameters = sum(
            np.sum(np.array(layer_data["importance_scores"]) > np.percentile(
                np.array(layer_data["importance_scores"]), 90
            ))
            for layer_data in self.imatrix_data.values()
        )

        logger.info(f"Total parameters analyzed: {total_parameters}")
        logger.info(f"High-importance parameters (top 10%): {protected_parameters}")
        logger.info(f"Protection ratio: {protected_parameters / total_parameters:.1%}")


def main():
    parser = argparse.ArgumentParser(description='imatrix Data Collection for GGUF Quantization')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--output', required=True, help='Output path for imatrix data')
    parser.add_argument('--samples', type=int, default=100000, help='Number of samples to process')

    args = parser.parse_args()

    collector = ImatrixCollector(args.model, args.output, args.samples)
    collector.collect_imatrix_data()


if __name__ == "__main__":
    main()