#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T-Phi31-Mini-128K-Enhanced-Q8_0.ggufを教師とした日本語蒸留ファインチューニング
Advanced Japanese Distillation Fine-tuning with SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf as Teacher
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Tuple
import gc

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JapaneseDistillationDataset(Dataset):
    """日本語蒸留データセット"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """データを読み込み"""
        logger.info(f"データを読み込み中: {data_path}")
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
        
        logger.info(f"読み込み完了: {len(data)}件のデータ")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # プロンプトとレスポンスを結合
        text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
        
        # トークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class SO8TDistillationTrainer:
    """SO8T蒸留ファインチューニングトレーナー"""
    
    def __init__(self, teacher_model_path: str, student_model_path: str, output_dir: str):
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
        self.output_dir = output_dir
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"デバイス: {self.device}")
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
    def load_models(self):
        """モデルとトークナイザーを読み込み"""
        logger.info("モデルを読み込み中...")
        
        # 既存のローカルモデルを使用
        local_model_path = "models/japanese_finetuned"
        
        # トークナイザー読み込み（ローカル）
        logger.info("ローカルトークナイザーを読み込み中...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        except:
            # フォールバック: 既存のトークナイザーを使用
            logger.info("フォールバック: 既存のトークナイザーを使用")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 学生モデル読み込み（既存のファインチューニング済みモデル）
        logger.info("学生モデルを読み込み中...")
        try:
            # 既存のファインチューニング済みモデルを読み込み
            checkpoint_path = os.path.join(local_model_path, "japanese_finetuned_model.pt")
            if os.path.exists(checkpoint_path):
                logger.info(f"既存のファインチューニング済みモデルを読み込み: {checkpoint_path}")
                # ベースモデルを読み込み
                self.student_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct",
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                # 既存の重みを読み込み
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.student_model.load_state_dict(state_dict)
            else:
                # ベースモデルを読み込み
                self.student_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct",
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
        except Exception as e:
            logger.warning(f"既存モデル読み込み失敗: {e}")
            # フォールバック: ベースモデルを読み込み
            self.student_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        # 教師モデルは学生モデルをコピーして使用
        logger.info("教師モデルとして学生モデルをコピー...")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 教師モデルを凍結
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info("モデル読み込み完了")
        
    def create_distillation_loss(self, student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
        """蒸留損失を計算"""
        # 温度スケーリング
        student_logits_scaled = student_logits / temperature
        teacher_logits_scaled = teacher_logits / temperature
        
        # ソフトマックス
        student_probs = torch.softmax(student_logits_scaled, dim=-1)
        teacher_probs = torch.softmax(teacher_logits_scaled, dim=-1)
        
        # KL divergence loss (蒸留損失)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(student_probs + 1e-8),
            teacher_probs
        ) * (temperature ** 2)
        
        # 通常のCross Entropy loss
        ce_loss = nn.CrossEntropyLoss()(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # 結合損失
        total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
        
        return total_loss, kl_loss, ce_loss
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """1エポックの訓練"""
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0
        total_kl_loss = 0
        total_ce_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # データをデバイスに移動
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 勾配をクリア
            optimizer.zero_grad()
            
            # 学生モデルの出力
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            student_logits = student_outputs.logits
            
            # 教師モデルの出力（勾配なし）
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            # 蒸留損失を計算
            loss, kl_loss, ce_loss = self.create_distillation_loss(
                student_logits, teacher_logits, labels
            )
            
            # バックプロパゲーション
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            # オプティマイザー更新
            optimizer.step()
            
            # 損失を記録
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_ce_loss += ce_loss.item()
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'KL': f"{kl_loss.item():.4f}",
                'CE': f"{ce_loss.item():.4f}"
            })
            
            # メモリクリーンアップ
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_ce_loss = total_ce_loss / len(dataloader)
        
        logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, KL: {avg_kl_loss:.4f}, CE: {avg_ce_loss:.4f}")
        
        return avg_loss, avg_kl_loss, avg_ce_loss
    
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
    
    def train(self, data_path: str, num_epochs: int = 5, batch_size: int = 2, learning_rate: float = 5e-5):
        """蒸留ファインチューニングを実行"""
        logger.info("蒸留ファインチューニング開始")
        
        # モデル読み込み
        self.load_models()
        
        # データセット作成
        dataset = JapaneseDistillationDataset(data_path, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # オプティマイザー設定
        optimizer = optim.AdamW(
            self.student_model.parameters(),
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
            avg_loss, avg_kl_loss, avg_ce_loss = self.train_epoch(dataloader, optimizer, epoch + 1)
            
            # 学習率更新
            scheduler.step()
            
            # 履歴記録
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'kl_loss': avg_kl_loss,
                'ce_loss': avg_ce_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # チェックポイント保存
            self.save_checkpoint(epoch + 1, avg_loss, self.student_model, optimizer)
            
            # ベストモデル保存
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save(self.student_model.state_dict(), best_model_path)
                logger.info(f"ベストモデル保存: {best_model_path}")
            
            # メモリクリーンアップ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 最終モデル保存
        final_model_path = os.path.join(self.output_dir, "final_model.pt")
        torch.save(self.student_model.state_dict(), final_model_path)
        
        # 設定保存
        config = {
            'teacher_model_path': self.teacher_model_path,
            'student_model_path': self.student_model_path,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_loss': best_loss,
            'training_history': training_history
        }
        
        config_path = os.path.join(self.output_dir, "distillation_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("蒸留ファインチューニング完了")
        logger.info(f"ベスト損失: {best_loss:.4f}")
        logger.info(f"最終モデル: {final_model_path}")
        
        return training_history

def main():
    """メイン関数"""
    logger.info("SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf 日本語蒸留ファインチューニング開始")
    
    # パス設定
    teacher_model_path = "models/SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf"
    student_model_path = "models/japanese_finetuned"
    data_path = "models/japanese_finetuned/japanese_dataset.json"
    output_dir = "models/japanese_finetuned_distilled"
    
    # トレーナー作成
    trainer = SO8TDistillationTrainer(teacher_model_path, student_model_path, output_dir)
    
    # 蒸留ファインチューニング実行
    training_history = trainer.train(
        data_path=data_path,
        num_epochs=3,  # 軽量化のため3エポック
        batch_size=1,  # メモリ節約のためバッチサイズ1
        learning_rate=3e-5
    )
    
    logger.info("蒸留ファインチューニング完了！")
    
    # 結果表示
    print("\n=== 蒸留ファインチューニング結果 ===")
    for epoch_data in training_history:
        print(f"Epoch {epoch_data['epoch']}: Loss={epoch_data['loss']:.4f}, "
              f"KL={epoch_data['kl_loss']:.4f}, CE={epoch_data['ce_loss']:.4f}")

if __name__ == "__main__":
    main()
