#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
魂の重みデータセット生成スクリプト
Soul Weights Dataset Generation Script

実装ログに基づいて魂の重みを学習データとして生成する
Based on implementation logs, generate soul weights as training data
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import yaml
from tqdm import tqdm

# SO8T関連インポート
try:
    from so8t.core.so8t_layer import SO8Rotation, NonCommutativeGate
    from so8t.core.attention_so8 import SO8Attention
except ImportError as e:
    print(f"[WARNING] SO8T import failed: {e}")

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SoulWeightsGenerator:
    """
    魂の重み生成クラス
    Soul Weights Generator Class

    実装ログに基づいて以下の魂の重みを生成：
    Based on implementation logs, generate the following soul weights:
    - Alpha Gateパラメータ (Alpha Gate parameter)
    - SO(8)回転行列 (R_safe, R_cmd) (SO(8) rotation matrices)
    - 魂の3本柱 (safety_head, task_head, dual_heads, pet)
    - LoRAアダプター重み (LoRA adapter weights)
    """

    def __init__(self, config_path: str = "configs/generate_soul_weights.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # 出力ディレクトリ
        self.output_dir = Path(self.config.get('output', {}).get('dataset_dir', 'D:/webdataset/datasets/soul_weights'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 魂の重みパラメータ
        self.soul_params = {
            'alpha_gate': self.config.get('alpha_gate', {}),
            'so8_rotations': self.config.get('so8_rotations', {}),
            'soul_pillars': self.config.get('soul_pillars', {}),
            'lora_adapter': self.config.get('lora_adapter', {})
        }

        # 乱数シード（再現性確保）
        self.seed = self.config.get('seed', 42)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"[SOUL] Soul Weights Generator initialized")
        print(f"[SOUL] Output directory: {self.output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        default_config = {
            'output': {
                'dataset_dir': 'D:/webdataset/datasets/soul_weights',
                'num_samples': 10000,
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1
            },
            'alpha_gate': {
                'range': [-5.0, 1.618],  # 混沌から黄金比まで
                'distribution': 'sigmoid_annealing',  # シグモイドアニーリング
                'steps': 1000
            },
            'so8_rotations': {
                'hidden_size': 4096,  # Phi-3.5の隠れ層サイズ
                'r_safe_enabled': True,
                'r_cmd_enabled': True,
                'non_commutative_check': True
            },
            'soul_pillars': {
                'safety_head': {'num_classes': 2},
                'task_head': {'num_classes': 4},
                'dual_heads': {'num_classes': 2},
                'pet': {'regularization_strength': 0.01}
            },
            'lora_adapter': {
                'r': 16,  # RTX3060 optimized
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            },
            'seed': 42
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                # デフォルト設定とマージ
                self._merge_configs(default_config, config)
                return default_config
            else:
                return default_config
        except Exception as e:
            print(f"[WARNING] Config load failed: {e}")
            return default_config

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """設定の再帰的マージ"""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def generate_alpha_gate_values(self, num_samples: int) -> torch.Tensor:
        """
        Alpha Gateパラメータ生成
        実装ログに基づくシグモイドアニーリング
        """
        print(f"[SOUL] Generating {num_samples} Alpha Gate values...")

        # シグモイドアニーリングに基づく分布
        alpha_config = self.soul_params['alpha_gate']

        if alpha_config.get('distribution') == 'sigmoid_annealing':
            # 実装ログの相転移スケジュールを模倣
            # Phase 1: 潜伏期間 (-4.98 → -4.93)
            # Phase 2: 臨界転移 (爆発的変化)
            # Phase 3: 安定化 (1.618黄金比)

            alphas = []

            # 潜伏期間（20%）
            latent_period = int(num_samples * 0.2)
            latent_alphas = torch.linspace(-4.98, -4.93, latent_period)
            alphas.extend(latent_alphas.tolist())

            # 臨界転移（30%）
            transition_period = int(num_samples * 0.3)
            # 爆発的変化: -3.79 → -1.69 → 0.41 → 1.30 → 1.55
            transition_points = [-3.79, -1.69, 0.41, 1.30, 1.55]
            transition_alphas = np.linspace(-3.79, 1.55, transition_period)
            alphas.extend(transition_alphas.tolist())

            # 安定化（50%）
            stable_period = num_samples - latent_period - transition_period
            stable_alphas = torch.full((stable_period,), 1.618)  # 黄金比
            # 少しのノイズを加える
            stable_alphas += torch.randn(stable_period) * 0.01
            alphas.extend(stable_alphas.tolist())

            return torch.tensor(alphas, dtype=torch.float32)

        else:
            # 一様分布
            alpha_range = alpha_config.get('range', [-5.0, 1.618])
            return torch.rand(num_samples) * (alpha_range[1] - alpha_range[0]) + alpha_range[0]

    def generate_so8_rotations(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """
        SO(8)回転行列生成
        R_safeとR_cmdの非可換構造を実装ログに基づいて生成
        """
        print(f"[SOUL] Generating {num_samples} SO(8) rotation matrices...")

        rotation_config = self.soul_params['so8_rotations']
        hidden_size = rotation_config.get('hidden_size', 4096)

        rotations = {
            'r_safe': [],
            'r_cmd': [],
            'r_total': []  # R_cmd @ R_safe (非可換積)
        }

        for i in range(num_samples):
            # R_safe: 安全回転行列
            r_safe = SO8Rotation(hidden_size)
            r_safe_matrix = r_safe.get_rotation_matrix()

            # R_cmd: コマンド回転行列
            r_cmd = SO8Rotation(hidden_size)
            r_cmd_matrix = r_cmd.get_rotation_matrix()

            # 非可換積: R_cmd @ R_safe (順序固定)
            r_total = torch.matmul(r_cmd_matrix, r_safe_matrix)

            rotations['r_safe'].append(r_safe_matrix)
            rotations['r_cmd'].append(r_cmd_matrix)
            rotations['r_total'].append(r_total)

        # テンソル化
        rotations['r_safe'] = torch.stack(rotations['r_safe'])
        rotations['r_cmd'] = torch.stack(rotations['r_cmd'])
        rotations['r_total'] = torch.stack(rotations['r_total'])

        return rotations

    def generate_soul_pillars(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """
        魂の3本柱生成
        safety_head, task_head, dual_heads, pet
        """
        print(f"[SOUL] Generating {num_samples} Soul Pillars...")

        pillars_config = self.soul_params['soul_pillars']
        pillars = {}

        for pillar_name, pillar_config in pillars_config.items():
            if pillar_name == 'safety_head':
                # 二値分類（安全/危険）
                num_classes = pillar_config.get('num_classes', 2)
                pillars[pillar_name] = torch.randn(num_samples, num_classes)

            elif pillar_name == 'task_head':
                # 四値分類（タスクタイプ）
                num_classes = pillar_config.get('num_classes', 4)
                pillars[pillar_name] = torch.randn(num_samples, num_classes)

            elif pillar_name == 'dual_heads':
                # 二重政策系（二値分類のペア）
                num_classes = pillar_config.get('num_classes', 2)
                pillars[pillar_name] = torch.randn(num_samples, 2, num_classes)

            elif pillar_name == 'pet':
                # PET正則化（態度の慣性）
                regularization_strength = pillar_config.get('regularization_strength', 0.01)
                pillars[pillar_name] = torch.randn(num_samples, 1) * regularization_strength

        return pillars

    def generate_lora_adapter_weights(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """
        LoRAアダプター重み生成
        RTX3060対応の小さなLoRA (r=16)
        """
        print(f"[SOUL] Generating {num_samples} LoRA adapter weights...")

        lora_config = self.soul_params['lora_adapter']
        lora_r = lora_config.get('r', 16)
        target_modules = lora_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

        # Phi-3.5の典型的なサイズ（RTX3060最適化）
        module_sizes = {
            'q_proj': (4096, 4096),
            'k_proj': (4096, 4096),
            'v_proj': (4096, 4096),
            'o_proj': (4096, 4096),
            'gate_proj': (4096, 11008),
            'up_proj': (4096, 11008),
            'down_proj': (11008, 4096)
        }

        lora_weights = {}

        for module_name in target_modules:
            if module_name in module_sizes:
                in_features, out_features = module_sizes[module_name]

                # LoRAのAとB行列
                lora_A = torch.randn(num_samples, in_features, lora_r) * 0.01
                lora_B = torch.randn(num_samples, lora_r, out_features) * 0.01

                lora_weights[f"{module_name}_lora_A"] = lora_A
                lora_weights[f"{module_name}_lora_B"] = lora_B

        return lora_weights

    def create_soul_weights_dataset(self) -> str:
        """
        魂の重みデータセット作成
        実装ログに基づいて全ての魂の重みを統合
        """
        print("[SOUL] Creating Soul Weights Dataset...")

        num_samples = self.config.get('output', {}).get('num_samples', 10000)

        # 各コンポーネント生成
        alpha_gates = self.generate_alpha_gate_values(num_samples)
        so8_rotations = self.generate_so8_rotations(num_samples)
        soul_pillars = self.generate_soul_pillars(num_samples)
        lora_weights = self.generate_lora_adapter_weights(num_samples)

        # データセット統合
        dataset = []
        for i in tqdm(range(num_samples), desc="Creating soul weights samples"):
            sample = {
                'sample_id': i,
                'timestamp': datetime.now().isoformat(),

                # Alpha Gate
                'alpha_gate': alpha_gates[i].item(),

                # SO(8)回転行列
                'r_safe': so8_rotations['r_safe'][i].tolist(),
                'r_cmd': so8_rotations['r_cmd'][i].tolist(),
                'r_total': so8_rotations['r_total'][i].tolist(),

                # 魂の3本柱
                'safety_head': soul_pillars['safety_head'][i].tolist(),
                'task_head': soul_pillars['task_head'][i].tolist(),
                'dual_heads': soul_pillars['dual_heads'][i].tolist(),
                'pet': soul_pillars['pet'][i].item(),

                # LoRAアダプター重み（主要モジュールのみ）
                'lora_weights': {}
            }

            # LoRA重みをサンプルに追加（メモリ効率のため主要なもののみ）
            for key, weights in lora_weights.items():
                if key.endswith('_lora_A') or key.endswith('_lora_B'):
                    # 最初の100サンプルのみ詳細保存（メモリ節約）
                    if i < 100:
                        sample['lora_weights'][key] = weights[i].tolist()
                    else:
                        # 残りは統計情報のみ
                        sample['lora_weights'][key] = {
                            'mean': weights[i].mean().item(),
                            'std': weights[i].std().item(),
                            'shape': list(weights[i].shape)
                        }

            dataset.append(sample)

        # 分割
        train_split = self.config.get('output', {}).get('train_split', 0.8)
        val_split = self.config.get('output', {}).get('val_split', 0.1)

        train_size = int(len(dataset) * train_split)
        val_size = int(len(dataset) * val_split)

        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size + val_size]
        test_dataset = dataset[train_size + val_size:]

        # 保存
        self._save_dataset(train_dataset, 'train')
        self._save_dataset(val_dataset, 'validation')
        self._save_dataset(test_dataset, 'test')

        # 統計情報保存
        stats = self._compute_dataset_stats(dataset)
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        dataset_path = str(self.output_dir)
        print(f"[SOUL] Soul Weights Dataset created at: {dataset_path}")
        print(f"[SOUL] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"[SOUL] Stats saved to: {stats_file}")

        return dataset_path

    def _save_dataset(self, dataset: List[Dict], split: str):
        """データセット保存"""
        output_file = self.output_dir / f'soul_weights_{split}.jsonl'

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in dataset:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        print(f"[SOUL] Saved {len(dataset)} {split} samples to {output_file}")

    def _compute_dataset_stats(self, dataset: List[Dict]) -> Dict[str, Any]:
        """データセット統計計算"""
        stats = {
            'total_samples': len(dataset),
            'alpha_gate': {
                'mean': float(np.mean([s['alpha_gate'] for s in dataset])),
                'std': float(np.std([s['alpha_gate'] for s in dataset])),
                'min': float(np.min([s['alpha_gate'] for s in dataset])),
                'max': float(np.max([s['alpha_gate'] for s in dataset]))
            },
            'components': {
                'r_safe': 'SO(8) safety rotation matrix (4096x4096)',
                'r_cmd': 'SO(8) command rotation matrix (4096x4096)',
                'r_total': 'Non-commutative product R_cmd @ R_safe',
                'safety_head': 'Safety head (2-class classification)',
                'task_head': 'Task head (4-class classification)',
                'dual_heads': 'Dual policy heads (2x 2-class classification)',
                'pet': 'PET regularization (attitude inertia)',
                'lora_weights': 'LoRA adapter weights (r=16, RTX3060 optimized)'
            },
            'generation_config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        return stats


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Generate Soul Weights Dataset for RTX3060 Training"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/generate_soul_weights.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory override'
    )

    args = parser.parse_args()

    # 魂の重みジェネレーター初期化
    generator = SoulWeightsGenerator(args.config)

    if args.output_dir:
        generator.output_dir = Path(args.output_dir)
        generator.output_dir.mkdir(parents=True, exist_ok=True)

    # データセット生成
    try:
        dataset_path = generator.create_soul_weights_dataset()
        print(f"[SUCCESS] Soul Weights Dataset generated successfully!")
        print(f"[PATH] {dataset_path}")

        # オーディオ通知
        try:
            import winsound
            winsound.PlaySound(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav", winsound.SND_FILENAME)
        except:
            print('\a')  # ビープ音

    except Exception as e:
        print(f"[ERROR] Soul Weights Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
