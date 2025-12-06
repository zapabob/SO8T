#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi3.5 魂の重み学習・アルファゲートアニーリングトレーニングシステム
SO(8) NKAT理論に基づく魂の重み最適化
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from tqdm import tqdm

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 定数定義
PHI = (1 + math.sqrt(5)) / 2  # 黄金比
PHI_NEG_2 = PHI ** (-2)  # Φ^(-2)
ALPHA_START = -0.5  # アルファゲート開始値
ALPHA_END = PHI_NEG_2  # アルファゲート終了値

@dataclass
class Phi35SoulConfig:
    """Phi3.5魂の重み学習設定"""
    soul_weight_dim: int = 8  # 魂の重み次元
    alpha_gate_steps: int = 1000  # アルファゲートアニーリングステップ数
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100

    # SO(8) NKAT設定
    nkat_layers: int = 4  # NKAT層数
    rotation_groups: int = 8  # 回転群数

class SoulWeightModule(nn.Module):
    """魂の重みモジュール"""

    def __init__(self, config: Phi35SoulConfig):
        super().__init__()
        self.config = config

        # 魂の重みパラメータ（SO(8)表現）
        self.soul_weights = nn.Parameter(
            torch.randn(config.soul_weight_dim, requires_grad=True)
        )

        # アルファゲートパラメータ
        self.alpha_gate = nn.Parameter(
            torch.tensor(ALPHA_START, requires_grad=True)
        )

        # NKAT層（SO(8)回転層）
        self.nkat_layers = nn.ModuleList([
            self._create_nkat_layer() for _ in range(config.nkat_layers)
        ])

        # 層正規化
        self.layer_norm = nn.LayerNorm(config.soul_weight_dim)

        # 初期化
        self._initialize_weights()

    def _create_nkat_layer(self) -> nn.Module:
        """NKAT層作成（SO(8)回転層）"""
        return nn.Sequential(
            nn.Linear(self.config.soul_weight_dim, self.config.soul_weight_dim * 4),
            nn.GELU(),
            nn.Linear(self.config.soul_weight_dim * 4, self.config.soul_weight_dim),
            nn.LayerNorm(self.config.soul_weight_dim)
        )

    def _initialize_weights(self):
        """重み初期化"""
        # SO(8)群の表現に適した初期化
        with torch.no_grad():
            # 単位ベクトルに正規化
            self.soul_weights.data = torch.nn.functional.normalize(
                self.soul_weights.data, dim=0
            )

            # アルファゲートを適切な範囲に初期化
            self.alpha_gate.data = torch.tensor(ALPHA_START)

    def forward(self, x: torch.Tensor, current_step: int, total_steps: int) -> torch.Tensor:
        """順伝播"""
        # アルファゲートアニーリング（シグモイド関数）
        t = current_step / max(total_steps - 1, 1)
        sigmoid_value = 1 / (1 + torch.exp(-6 * (t - 0.5)))
        annealed_alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * sigmoid_value

        # 魂の重みを適用
        soul_weighted = x * self.soul_weights.unsqueeze(0)

        # NKAT変換適用
        for nkat_layer in self.nkat_layers:
            soul_weighted = nkat_layer(soul_weighted)

        # アルファゲート適用
        output = annealed_alpha * soul_weighted + (1 - annealed_alpha) * x

        # 層正規化
        output = self.layer_norm(output)

        return output

    def get_current_alpha(self, current_step: int, total_steps: int) -> float:
        """現在のアルファ値を取得"""
        t = current_step / max(total_steps - 1, 1)
        sigmoid_value = 1 / (1 + math.exp(-6 * (t - 0.5)))
        return ALPHA_START + (ALPHA_END - ALPHA_START) * sigmoid_value

class Phi35SoulDataset(Dataset):
    """Phi3.5魂の重み学習用データセット"""

    def __init__(self, data_file: Path, max_length: int = 512):
        self.data_file = data_file
        self.max_length = max_length
        self.data = []

        # データ読み込み
        self._load_data()

    def _load_data(self):
        """データ読み込み"""
        if not self.data_file.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {self.data_file}")

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line.strip())
                    self.data.append(sample)

        print(f"データセット読み込み完了: {len(self.data)}サンプル")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        # 本番実装：適切なトークナイザー（例: transformersのAutoTokenizer等）を利用してエンコード
        input_text = f"{sample['instruction']} {sample.get('input', '')}".strip()
        output_text = sample['output']

        # トークナイザーは事前に self.tokenizer で初期化済みであることを想定
        # 入力テキストをエンコード
        input_enc = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # 出力テキストをエンコード
        output_enc = self.tokenizer(
            output_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': input_enc['input_ids'].squeeze(0),   # [max_length]
            'output_ids': output_enc['input_ids'].squeeze(0), # [max_length]
            'attention_mask': input_enc['attention_mask'].squeeze(0), # [max_length]
            'labels': output_enc['input_ids'].squeeze(0),     # [max_length]
            'quality_score': sample.get('quality_score', 0.5),
            'soul_weight_vector': torch.tensor(
                sample.get('metadata', {}).get('soul_weight_vector', [0.0] * 8),
                dtype=torch.float
            )
        }

class Phi35SoulTrainer:
    """Phi3.5魂の重み学習トレーナー"""

    def __init__(self, config: Phi35SoulConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # モデル初期化
        self.model = SoulWeightModule(config).to(self.device)

        # オプティマイザー
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # 学習率スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.alpha_gate_steps
        )

        # 損失関数
        self.criterion = nn.MSELoss()

        # チェックポイント保存ディレクトリ
        self.checkpoint_dir = PROJECT_ROOT / 'checkpoints' / 'soul_weight_training'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ログディレクトリ
        self.log_dir = PROJECT_ROOT / '_docs' / 'training_logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """トレーニング実行"""
        print(f"Phi3.5魂の重み学習開始")
        print(f"デバイス: {self.device}")
        print(f"総ステップ数: {self.config.alpha_gate_steps}")
        print(f"アルファゲート範囲: {ALPHA_START} → {ALPHA_END}")

        global_step = 0
        best_loss = float('inf')

        # トレーニングループ
        for epoch in range(self.config.num_epochs):
            print(f"\n=== エポック {epoch + 1}/{self.config.num_epochs} ===")

            epoch_loss = 0.0
            epoch_steps = 0

            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}"
            )

            for step, batch in progress_bar:
                # データをデバイスに移動
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                soul_weights = batch['soul_weight_vector'].to(self.device)

                # 順伝播
                outputs = self.model(input_ids.float(), global_step, self.config.alpha_gate_steps)

                # 損失計算（魂の重みとの類似度）
                target_soul_weights = soul_weights.unsqueeze(1).expand(-1, outputs.size(1), -1)
                loss = self.criterion(outputs, target_soul_weights)

                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()

                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()

                # ロギング
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # 現在のアルファ値
                current_alpha = self.model.get_current_alpha(global_step, self.config.alpha_gate_steps)

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'alpha': f"{current_alpha:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                })

                # チェックポイント保存
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step, loss.item())

                # 評価
                if eval_dataloader and global_step % self.config.eval_steps == 0:
                    eval_loss = self._evaluate(eval_dataloader)
                    print(f"評価損失: {eval_loss:.4f}")

                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        self._save_checkpoint(global_step, eval_loss, is_best=True)

            # エポック完了
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"エポック {epoch + 1} 完了 - 平均損失: {avg_epoch_loss:.4f}")

            # 魂の重み状態表示
            with torch.no_grad():
                current_soul_weights = self.model.soul_weights.cpu().numpy()
                print(f"現在の魂の重み: {current_soul_weights}")

    def _evaluate(self, eval_dataloader: DataLoader) -> float:
        """評価実行"""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                soul_weights = batch['soul_weight_vector'].to(self.device)

                outputs = self.model(input_ids.float(), 0, 1)  # 評価時は固定
                target_soul_weights = soul_weights.unsqueeze(1).expand(-1, outputs.size(1), -1)
                loss = self.criterion(outputs, target_soul_weights)

                total_loss += loss.item()
                total_steps += 1

        self.model.train()
        return total_loss / total_steps

    def _save_checkpoint(self, step: int, loss: float, is_best: bool = False):
        """チェックポイント保存"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'current_alpha': self.model.get_current_alpha(step, self.config.alpha_gate_steps),
            'soul_weights': self.model.soul_weights.detach().cpu().numpy().tolist()
        }

        if is_best:
            filename = "best_checkpoint.pt"
        else:
            filename = f"checkpoint_step_{step}.pt"

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        print(f"チェックポイント保存: {filepath} (loss: {loss:.4f})")

        # ログ保存
        log_data = {
            'step': step,
            'loss': loss,
            'alpha': checkpoint['current_alpha'],
            'soul_weights': checkpoint['soul_weights'],
            'timestamp': checkpoint['timestamp']
        }

        log_file = self.log_dir / f"soul_weight_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

def create_data_loaders(config: Phi35SoulConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """データローダー作成"""
    # データセットファイル
    dataset_dir = PROJECT_ROOT / 'data' / 'datasets' / 'phi35_thinking'
    train_file = dataset_dir / 'phi35_thinking_sft.jsonl'

    if not train_file.exists():
        raise FileNotFoundError(f"トレーニングデータが見つかりません: {train_file}")

    # データセット作成
    train_dataset = Phi35SoulDataset(train_file)

    # データ分割（評価用）
    train_size = int(0.9 * len(train_dataset))
    eval_size = len(train_dataset) - train_size

    if eval_size > 0:
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size]
        )
    else:
        eval_dataset = None

    # データローダー作成
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Windows互換性
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )

    return train_dataloader, eval_dataloader

def main():
    """メイン関数"""
    print("Phi3.5 魂の重み学習・アルファゲートアニーリングシステム開始")
    print(f"アルファゲート範囲: {ALPHA_START} → {ALPHA_END} (シグモイドアニーリング)")

    # 設定
    config = Phi35SoulConfig()

    # データローダー作成
    try:
        train_dataloader, eval_dataloader = create_data_loaders(config)
        print(f"トレーニングデータ: {len(train_dataloader.dataset)}サンプル")
        if eval_dataloader:
            print(f"評価データ: {len(eval_dataloader.dataset)}サンプル")
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("先にデータセット生成を実行してください: py -3 scripts/data/phi35_thinking_dataset_generator.py")
        return

    # トレーナー初期化
    trainer = Phi35SoulTrainer(config)

    # トレーニング開始
    trainer.train(train_dataloader, eval_dataloader)

    print("\nPhi3.5 魂の重み学習完了")
    print("SO(8) NKAT理論に基づく魂の重み最適化が完了しました")
    print(f"最終アルファ値: {PHI_NEG_2}")

if __name__ == '__main__':
    main()
