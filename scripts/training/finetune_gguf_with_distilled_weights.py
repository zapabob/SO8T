#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUFファイルを蒸留済みPTファイルの重みでファインチューニング
Fine-tune GGUF file with distilled PT weights
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
import gguf

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GGUFToPyTorchConverter:
    """GGUFファイルをPyTorchモデルに変換"""
    
    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self.gguf_reader = None
        
    def load_gguf(self):
        """GGUFファイルを読み込み"""
        logger.info(f"GGUFファイルを読み込み中: {self.gguf_path}")
        self.gguf_reader = gguf.GGUFReader(self.gguf_path, 'r')
        logger.info("GGUFファイル読み込み完了")
        
    def get_model_config(self):
        """モデル設定を取得"""
        if not self.gguf_reader:
            self.load_gguf()
            
        config = {}
        
        # メタデータから設定を取得
        for key, value in self.gguf_reader.fields.items():
            if key.startswith('llama.'):
                config_key = key.replace('llama.', '')
                config[config_key] = value
            elif key.startswith('general.'):
                config_key = key.replace('general.', '')
                config[config_key] = value
                
        return config
        
    def get_tensor_data(self, tensor_name: str):
        """テンソルデータを取得"""
        if not self.gguf_reader:
            self.load_gguf()
            
        for tensor in self.gguf_reader.tensors:
            if tensor.name == tensor_name:
                return tensor.data
        return None

class DistilledWeightApplier:
    """蒸留済み重みを適用"""
    
    def __init__(self, distilled_model_path: str, gguf_path: str):
        self.distilled_model_path = distilled_model_path
        self.gguf_path = gguf_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_enhanced_model(self, config: Dict[str, Any]):
        """強化されたモデルを作成"""
        class EnhancedSO8TModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                
                # 基本パラメータ
                self.vocab_size = config.get('embedding_length', 32000)
                self.hidden_size = config.get('embedding_length', 512)
                self.num_layers = config.get('block_count', 4)
                self.num_heads = config.get('head_count', 8)
                self.intermediate_size = config.get('feed_forward_length', 2048)
                
                # 埋め込み層
                self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
                
                # Transformer層
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=self.hidden_size,
                        nhead=self.num_heads,
                        dim_feedforward=self.intermediate_size,
                        batch_first=True,
                        dropout=0.1
                    ) for _ in range(self.num_layers)
                ])
                
                # 出力層
                self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)
                
                # レイヤー正規化
                self.layer_norm = nn.LayerNorm(self.hidden_size)
                
            def forward(self, input_ids, attention_mask=None, labels=None):
                # 埋め込み
                x = self.embedding(input_ids)
                
                # Transformer層
                for layer in self.layers:
                    x = layer(x, src_key_padding_mask=attention_mask)
                
                # レイヤー正規化
                x = self.layer_norm(x)
                
                # 出力
                logits = self.lm_head(x)
                
                # 損失計算
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                return {
                    'logits': logits,
                    'loss': loss
                }
        
        return EnhancedSO8TModel(config)
    
    def load_distilled_weights(self, model):
        """蒸留済み重みを読み込み"""
        logger.info(f"蒸留済み重みを読み込み中: {self.distilled_model_path}")
        
        try:
            distilled_state = torch.load(self.distilled_model_path, map_location=self.device)
            
            # 重みを適用（可能な部分のみ）
            model_state = model.state_dict()
            applied_count = 0
            
            for name, param in model_state.items():
                if name in distilled_state:
                    if param.shape == distilled_state[name].shape:
                        model_state[name] = distilled_state[name]
                        applied_count += 1
                        logger.info(f"重み適用: {name}")
                    else:
                        logger.warning(f"形状不一致: {name} - モデル: {param.shape}, 蒸留: {distilled_state[name].shape}")
            
            model.load_state_dict(model_state)
            logger.info(f"蒸留済み重み適用完了: {applied_count}個の重みを適用")
            
        except Exception as e:
            logger.warning(f"蒸留済み重み読み込み失敗: {e}")
            logger.info("ランダム初期化で続行")
    
    def apply_gguf_weights(self, model, gguf_converter):
        """GGUF重みを適用"""
        logger.info("GGUF重みを適用中...")
        
        try:
            # 埋め込み重み
            embedding_weight = gguf_converter.get_tensor_data('token_embd.weight')
            if embedding_weight is not None:
                model.embedding.weight.data = torch.tensor(embedding_weight, dtype=torch.float32)
                logger.info("埋め込み重み適用完了")
            
            # 出力重み
            output_weight = gguf_converter.get_tensor_data('output.weight')
            if output_weight is not None:
                model.lm_head.weight.data = torch.tensor(output_weight, dtype=torch.float32)
                logger.info("出力重み適用完了")
                
        except Exception as e:
            logger.warning(f"GGUF重み適用失敗: {e}")

class JapaneseFinetuningDataset(Dataset):
    """日本語ファインチューニングデータセット"""
    
    def __init__(self, data_path: str, max_length: int = 1024):
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
                processed_data.append({
                    'prompt': f"{item['instruction']}\n{item['input']}",
                    'response': item['output']
                })
            elif 'prompt' in item and 'response' in item:
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
            tokens.append(0)
        
        return {
            'input_ids': torch.tensor(tokens[:self.max_length], dtype=torch.long),
            'labels': torch.tensor(tokens[:self.max_length], dtype=torch.long)
        }

class GGUFFinetuningTrainer:
    """GGUFファインチューニングトレーナー"""
    
    def __init__(self, gguf_path: str, distilled_model_path: str, output_dir: str):
        self.gguf_path = gguf_path
        self.distilled_model_path = distilled_model_path
        self.output_dir = output_dir
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"デバイス: {self.device}")
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
    def load_and_enhance_model(self):
        """モデルを読み込み・強化"""
        logger.info("モデルを読み込み・強化中...")
        
        # GGUFコンバーター
        gguf_converter = GGUFToPyTorchConverter(self.gguf_path)
        gguf_converter.load_gguf()
        config = gguf_converter.get_model_config()
        
        # 重み適用器
        weight_applier = DistilledWeightApplier(self.distilled_model_path, self.gguf_path)
        
        # モデル作成
        model = weight_applier.create_enhanced_model(config)
        model = model.to(self.device)
        
        # 蒸留済み重みを適用
        weight_applier.load_distilled_weights(model)
        
        # GGUF重みを適用
        weight_applier.apply_gguf_weights(model, gguf_converter)
        
        logger.info("モデル読み込み・強化完了")
        return model
    
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
        checkpoint_path = os.path.join(self.output_dir, f"finetuned_checkpoint_epoch_{epoch}.pt")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        logger.info(f"チェックポイント保存: {checkpoint_path}")
    
    def train(self, data_path: str, num_epochs: int = 2, batch_size: int = 2, learning_rate: float = 5e-5):
        """ファインチューニングを実行"""
        logger.info("GGUFファインチューニング開始")
        
        # モデル読み込み・強化
        model = self.load_and_enhance_model()
        
        # データセット作成
        dataset = JapaneseFinetuningDataset(data_path)
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
                best_model_path = os.path.join(self.output_dir, "finetuned_best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"ベストモデル保存: {best_model_path}")
            
            # メモリクリーンアップ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 最終モデル保存
        final_model_path = os.path.join(self.output_dir, "finetuned_final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        
        # 設定保存
        config = {
            'gguf_path': self.gguf_path,
            'distilled_model_path': self.distilled_model_path,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_loss': best_loss,
            'training_history': training_history
        }
        
        config_path = os.path.join(self.output_dir, "finetuning_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("GGUFファインチューニング完了")
        logger.info(f"ベスト損失: {best_loss:.4f}")
        logger.info(f"最終モデル: {final_model_path}")
        
        return training_history

def main():
    """メイン関数"""
    logger.info("GGUFファインチューニング開始")
    
    # パス設定
    gguf_path = "archive/so8t-vl-2b-instruct-complete.gguf"
    distilled_model_path = "D:/japanese_finetuned_distilled_simple/final_model.pt"
    data_path = "models/japanese_finetuned/japanese_dataset.json"
    output_dir = "D:/so8t_gguf_finetuned"
    
    # トレーナー作成
    trainer = GGUFFinetuningTrainer(gguf_path, distilled_model_path, output_dir)
    
    # ファインチューニング実行
    training_history = trainer.train(
        data_path=data_path,
        num_epochs=2,
        batch_size=2,
        learning_rate=5e-5
    )
    
    logger.info("GGUFファインチューニング完了！")
    
    # 結果表示
    print("\n=== GGUFファインチューニング結果 ===")
    for epoch_data in training_history:
        print(f"Epoch {epoch_data['epoch']}: Loss={epoch_data['loss']:.4f}")

if __name__ == "__main__":
    main()

