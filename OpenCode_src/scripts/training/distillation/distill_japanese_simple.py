#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡単な日本語蒸留ファインチューニング
Simple Japanese Distillation Fine-tuning
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Tuple
import gc

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleJapaneseDataset(Dataset):
    """簡単な日本語データセット"""
    
    def __init__(self, data_path: str, max_length: int = 512):
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """データを読み込み"""
        logger.info(f"データを読み込み中: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # データ形式を統一
        processed_data = []
        for item in data:
            if 'instruction' in item and 'input' in item and 'output' in item:
                # 既存の形式を新しい形式に変換
                processed_data.append({
                    'prompt': f"{item['instruction']}\n{item['input']}",
                    'response': item['output']
                })
            elif 'prompt' in item and 'response' in item:
                # 既に新しい形式
                processed_data.append(item)
        
        logger.info(f"読み込み完了: {len(processed_data)}件のデータ")
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # プロンプトとレスポンスを結合
        text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
        
        # 簡単なトークン化（文字レベル）
        tokens = [ord(c) for c in text[:self.max_length]]
        
        # パディング
        while len(tokens) < self.max_length:
            tokens.append(0)  # パディングトークン
        
        return {
            'input_ids': torch.tensor(tokens[:self.max_length], dtype=torch.long),
            'labels': torch.tensor(tokens[:self.max_length], dtype=torch.long)
        }

class SimpleDistillationTrainer:
    """簡単な蒸留ファインチューニングトレーナー"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"デバイス: {self.device}")
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
    def create_simple_model(self, vocab_size: int = 65536, hidden_size: int = 512, num_layers: int = 4):
        """簡単なモデルを作成"""
        class SimpleModel(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        dim_feedforward=hidden_size * 4,
                        batch_first=True
                    ) for _ in range(num_layers)
                ])
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, input_ids, labels=None):
                x = self.embedding(input_ids)
                
                for layer in self.layers:
                    x = layer(x)
                
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                return {
                    'logits': logits,
                    'loss': loss
                }
        
        return SimpleModel(vocab_size, hidden_size, num_layers)
    
    def train_epoch(self, dataloader, model, optimizer, epoch):
        """1エポックの訓練"""
        model.train()
        
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # データをデバイスに移動
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 勾配をクリア
            optimizer.zero_grad()
            
            # モデルの出力
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            # バックプロパゲーション
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # オプティマイザー更新
            optimizer.step()
            
            # 損失を記録
            total_loss += loss.item()
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}"
            })
            
            # メモリクリーンアップ
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, model, optimizer):
        """チェックポイントを保存"""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        logger.info(f"チェックポイント保存: {checkpoint_path}")
    
    def train(self, data_path: str, num_epochs: int = 3, batch_size: int = 4, learning_rate: float = 1e-3):
        """蒸留ファインチューニングを実行"""
        logger.info("簡単な蒸留ファインチューニング開始")
        
        # モデル作成
        model = self.create_simple_model()
        model = model.to(self.device)
        
        # データセット作成
        dataset = SimpleJapaneseDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # オプティマイザー設定
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学習率スケジューラー
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # 訓練ループ
        best_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs} 開始")
            
            # 1エポック訓練
            avg_loss = self.train_epoch(dataloader, model, optimizer, epoch + 1)
            
            # 学習率更新
            scheduler.step()
            
            # 履歴記録
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # チェックポイント保存
            self.save_checkpoint(epoch + 1, avg_loss, model, optimizer)
            
            # ベストモデル保存
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"ベストモデル保存: {best_model_path}")
            
            # メモリクリーンアップ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 最終モデル保存
        final_model_path = os.path.join(self.output_dir, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        
        # 設定保存
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_loss': best_loss,
            'training_history': training_history
        }
        
        config_path = os.path.join(self.output_dir, "distillation_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("簡単な蒸留ファインチューニング完了")
        logger.info(f"ベスト損失: {best_loss:.4f}")
        logger.info(f"最終モデル: {final_model_path}")
        
        return training_history

def main():
    """メイン関数"""
    logger.info("簡単な日本語蒸留ファインチューニング開始")
    
    # パス設定（Dドライブに出力）
    data_path = "models/japanese_finetuned/japanese_dataset.json"
    output_dir = "D:/japanese_finetuned_distilled_simple"
    
    # トレーナー作成
    trainer = SimpleDistillationTrainer(output_dir)
    
    # 蒸留ファインチューニング実行
    training_history = trainer.train(
        data_path=data_path,
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-3
    )
    
    logger.info("簡単な蒸留ファインチューニング完了！")
    
    # 結果表示
    print("\n=== 簡単な蒸留ファインチューニング結果 ===")
    for epoch_data in training_history:
        print(f"Epoch {epoch_data['epoch']}: Loss={epoch_data['loss']:.4f}")

if __name__ == "__main__":
    main()
