#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学中心キャリブレーションデータ作成スクリプト
I-Matrix量子化のための数学・論理推論データセット作成

SO(8)アダプターの重要度行列を正確に計算するため、
数学・論理・幾何学関連の問題を大量に含むデータを生成
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import random

class MathCalibrationDataGenerator:
    """数学中心キャリブレーションデータ生成器"""

    def __init__(self):
        # 数学・論理・幾何学関連の問題テンプレート
        self.math_templates = [
            # 基本算数
            "Calculate: {} + {} = ?",
            "What is {} multiplied by {}?",
            "Solve for x: {}x + {} = {}",
            "If {} apples cost ${}, how much do {} apples cost?",

            # 代数・方程式
            "Solve the equation: {}x² + {}x + {} = 0",
            "Find the derivative of: d/dx({})",
            "Simplify: ({} + {}) * ({} - {})",
            "Factor: {}x² + {}x + {}",

            # 幾何学（SO(8)アダプターに重要）
            "Calculate the area of a circle with radius {}",
            "What is the volume of a sphere with radius {}?",
            "In triangle ABC, angle A is {}, angle B is {}, find angle C",
            "Calculate the distance between points ({},{}) and ({},{})",

            # 論理・推論
            "If all A are B, and some B are C, then:",
            "Complete the sequence: {}, {}, {}, {}",
            "Which of these is different: {}, {}, {}, {}",
            "If P implies Q, and P is true, then Q is:",

            # 統計・確率
            "Calculate the mean of: {}",
            "What is the probability of rolling a {} on a die?",
            "Find the standard deviation of: {}",
            "In a normal distribution with mean {} and std {}, find P(X > {})",

            # 離散数学・組み合わせ
            "How many ways to arrange {} distinct objects?",
            "Calculate C({}, {})",
            "Find the greatest common divisor of {} and {}",
            "Solve the recurrence: a_n = {}a_{n-1} + {}a_{n-2}",

            # 線形代数（SO(8)アダプターの基礎）
            "Find the determinant of matrix: [[{},{}],[{},{}]]",
            "Calculate the eigenvalues of: λ² - {}λ + {} = 0",
            "Find the inverse of: [[{},{}],[{},{}]]",
            "Compute the dot product: ({},{},{}) · ({},{},{})",

            # 微積分
            "Evaluate the integral: ∫{} dx from {} to {}",
            "Find the limit: lim(x→{}) ({})/(x-{})",
            "Solve the differential equation: dy/dx = {}",
            "Find the Taylor series of {} around x={}",

            # 数論
            "Is {} a prime number?",
            "Find all factors of {}",
            "Solve the congruence: {}x ≡ {} mod {}",
            "Calculate φ({})",  # Euler's totient function
        ]

        # SO(8)特化の問題（幾何学的推論）
        self.so8_specific_templates = [
            "In 8-dimensional space, calculate the angle between vectors ({}) and ({})",
            "Find the SO(8) rotation matrix for {} degrees around axis {}",
            "Compute the Lie bracket [X{}, X{}] in so(8)",
            "Calculate the Killing form on so(8) generators X{} and X{}",
            "Find the adjoint representation of SO(8) element corresponding to rotation in plane {}-{}",
            "Compute the exponential map exp(tX{}) where X{} is a generator of so(8)",
            "Find the fundamental representation matrix for SO(8) rotation in {}-dimensional subspace",
            "Calculate the volume of the 8-ball with radius {} using hypersphere formula",
            "Find the intersection of two 4-spheres in 8-dimensional space",
            "Compute the Clifford algebra representation relevant to SO(8) spinors",
        ]

        # 数値範囲
        self.number_ranges = {
            'small': (1, 10),
            'medium': (10, 100),
            'large': (100, 1000),
            'fraction': (1, 20),  # 分数用
        }

    def generate_math_problem(self, template: str) -> str:
        """テンプレートから数学問題を生成"""
        # プレースホルダーを数値で置き換え
        problem = template

        # {} を数値で置き換え
        placeholders = problem.count('{}')
        for _ in range(placeholders):
            if 'radius' in problem or 'distance' in problem:
                # 幾何学関連は小数点以下1桁
                num = round(random.uniform(1, 10), 1)
            elif 'probability' in problem:
                # 確率は0-1
                num = round(random.uniform(0, 1), 2)
            elif 'degree' in problem or 'angle' in problem:
                # 角度は0-360
                num = random.randint(0, 360)
            else:
                # 通常は整数
                num = random.randint(1, 100)

            problem = problem.replace('{}' , str(num), 1)

        return problem

    def generate_calibration_data(self, num_samples: int = 10000) -> List[str]:
        """キャリブレーションデータを生成"""
        print(f"[INFO] Generating {num_samples} math calibration samples...")

        calibration_data = []

        # 通常の数学問題
        math_count = int(num_samples * 0.7)  # 70%
        for _ in range(math_count):
            template = random.choice(self.math_templates)
            problem = self.generate_math_problem(template)
            calibration_data.append(problem)

        # SO(8)特化問題
        so8_count = int(num_samples * 0.3)  # 30%
        for _ in range(so8_count):
            template = random.choice(self.so8_specific_templates)
            problem = self.generate_math_problem(template)
            calibration_data.append(problem)

        # 重複除去とシャッフル
        calibration_data = list(set(calibration_data))
        random.shuffle(calibration_data)

        print(f"[INFO] Generated {len(calibration_data)} unique calibration samples")
        return calibration_data

    def save_calibration_file(self, data: List[str], output_file: str):
        """キャリブレーションデータをファイルに保存"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Saving calibration data to {output_file}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(line + '\n')

        print(f"[SUCCESS] Saved {len(data)} calibration samples")

    def create_imatrix_data(self, output_file: str, num_samples: int = 10000):
        """I-Matrix用のキャリブレーションデータを作成"""
        print("=" * 60)
        print("Creating Math-Focused Calibration Data for I-Matrix")
        print("=" * 60)

        # キャリブレーションデータ生成
        calibration_data = self.generate_calibration_data(num_samples)

        # 保存
        self.save_calibration_file(calibration_data, output_file)

        # 統計情報表示
        print("\n[STATISTICS]")
        print(f"Total samples: {len(calibration_data)}")
        print(".1f")
        print(".1f")

        # サンプル表示
        print("\n[SAMPLE PROBLEMS]")
        for i, sample in enumerate(calibration_data[:10]):
            print("2d")

        print(f"\n[SUCCESS] Calibration data saved to: {output_file}")
        print("Use this file with llama-imatrix for math-preserving quantization!")

def main():
    """メイン関数"""
    generator = MathCalibrationDataGenerator()

    # I-Matrix用キャリブレーションデータ作成
    output_file = "data/calibration/math_calibration_data.txt"
    generator.create_imatrix_data(output_file, num_samples=50000)

if __name__ == "__main__":
    main()
