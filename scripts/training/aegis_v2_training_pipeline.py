#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS-v2.0-Phi3.5-thinking 統合トレーニングパイプライン
AEGIS-v2.0-Phi3.5-thinking Integrated Training Pipeline

SO8VIT + Alpha Gate Annealing + PPO + Internal Inference Enhancement
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import time
import atexit
import signal
import psutil

# カスタムモジュールインポート
from .ppo_internal_inference import PPOInternalInferenceTrainer, MetaInferenceController
from .alpha_gate_annealing import SigmoidAlphaGateAnnealing
from ..models.so8vit import SO8VIT
from ..data.dataset_collection_cleansing import create_ppo_training_dataset

logger = logging.getLogger(__name__)


class AutoCheckpointManager:
    """
    自動チェックポイント管理システム
    3分ごとの自動保存、5個上限ローリングストック、電源投入時自動再開
    """

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5, auto_save_interval: int = 180):
        """
        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
            max_checkpoints: 最大チェックポイント数（ローリングストック）
            auto_save_interval: 自動保存間隔（秒）
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.auto_save_interval = auto_save_interval

        # 自動保存タイマー
        self.auto_save_timer = None
        self.is_running = False
        self.last_save_time = time.time()

        # シグナルハンドラー登録（異常終了時保存）
        self._register_signal_handlers()

        # atexit登録（プログラム終了時保存）
        atexit.register(self.force_save_checkpoint)

        logger.info(f"[AUTO-CP] Auto checkpoint manager initialized: {auto_save_interval}s interval, {max_checkpoints} max checkpoints")

    def start_auto_save(self, save_callback):
        """自動保存を開始"""
        if self.is_running:
            return

        self.save_callback = save_callback
        self.is_running = True

        def auto_save_loop():
            while self.is_running:
                time.sleep(self.auto_save_interval)
                if self.is_running:  # まだ実行中か確認
                    try:
                        self.save_callback(auto=True)
                        self.last_save_time = time.time()
                        logger.info("[AUTO-CP] Auto checkpoint saved")
                    except Exception as e:
                        logger.error(f"[AUTO-CP] Auto save failed: {e}")

        self.auto_save_timer = threading.Thread(target=auto_save_loop, daemon=True)
        self.auto_save_timer.start()

        logger.info("[AUTO-CP] Auto checkpoint saving started")

    def stop_auto_save(self):
        """自動保存を停止"""
        self.is_running = False
        if self.auto_save_timer:
            self.auto_save_timer.join(timeout=5)
        logger.info("[AUTO-CP] Auto checkpoint saving stopped")

    def save_checkpoint(self, checkpoint_data: Dict[str, Any], filename_prefix: str = "checkpoint"):
        """チェックポイント保存（ローリングストック管理）"""
        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename

        # 保存
        torch.save(checkpoint_data, filepath)

        # ローリングストック管理
        self._manage_rolling_stock(filename_prefix)

        logger.info(f"[AUTO-CP] Checkpoint saved: {filepath}")
        return filepath

    def _manage_rolling_stock(self, filename_prefix: str):
        """ローリングストック管理（古いチェックポイントを削除）"""
        # 同プレフィックスのチェックポイントを列挙
        pattern = f"{filename_prefix}_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        if len(checkpoints) > self.max_checkpoints:
            # 作成時刻でソート（古い順）
            checkpoints.sort(key=lambda x: x.stat().st_mtime)

            # 古いものを削除
            to_delete = checkpoints[:len(checkpoints) - self.max_checkpoints]
            for old_cp in to_delete:
                try:
                    old_cp.unlink()
                    logger.info(f"[AUTO-CP] Removed old checkpoint: {old_cp.name}")
                except Exception as e:
                    logger.error(f"[AUTO-CP] Failed to remove {old_cp.name}: {e}")

    def find_latest_checkpoint(self, filename_prefix: str = "checkpoint") -> Optional[Path]:
        """最新のチェックポイントを検索"""
        pattern = f"{filename_prefix}_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        if not checkpoints:
            return None

        # 最新のものを返す
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0]

    def _register_signal_handlers(self):
        """シグナルハンドラー登録"""
        def signal_handler(signum, frame):
            logger.warning(f"[AUTO-CP] Received signal {signum}, forcing checkpoint save...")
            self.force_save_checkpoint()

        # SIGINT (Ctrl+C), SIGTERM (終了要求) を処理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Windows用にSIGBREAKも
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def force_save_checkpoint(self):
        """強制チェックポイント保存"""
        if hasattr(self, 'save_callback') and self.save_callback:
            try:
                self.save_callback(auto=True, force=True)
                logger.info("[AUTO-CP] Force checkpoint saved on exit")
            except Exception as e:
                logger.error(f"[AUTO-CP] Force save failed: {e}")

    @staticmethod
    def detect_power_resume() -> bool:
        """電源投入時の再開を検出"""
        # システム起動時間をチェック（簡易版）
        try:
            boot_time = psutil.boot_time()
            current_time = time.time()
            uptime_hours = (current_time - boot_time) / 3600

            # 起動時間が短い場合は電源投入後と判断
            if uptime_hours < 1.0:  # 1時間以内
                logger.info("[AUTO-CP] Power resume detected (system uptime: %.1f hours)", uptime_hours)
                return True

        except Exception as e:
            logger.warning(f"[AUTO-CP] Could not detect power resume: {e}")

        return False

class MultimodalThinkingDataset(Dataset):
    """
    マルチモーダルThinkingデータセット
    """

    def __init__(self, data_path: str, tokenizer, image_processor=None, max_length: int = 2048):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        # データ読み込み
        self.samples = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """データセット読み込み"""
        samples = []

        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    samples.append(sample)
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get('samples', [])

        logger.info(f"Loaded {len(samples)} samples from {self.data_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # テキスト処理
        text = sample.get('text', '')
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'text': text,
            'domain': sample.get('domain', 'general'),
            'quality_score': sample.get('quality_score', 0.5),
            'language': sample.get('language', 'en'),
            'complexity_score': sample.get('complexity_score', 0.5)
        }

        # 魂の重みデータがある場合は追加
        if 'soul_weights' in sample:
            result['soul_weights'] = sample['soul_weights']
            result['has_soul_weights'] = True
        else:
            result['has_soul_weights'] = False

        # 画像処理（ある場合）
        if sample.get('has_image', False) and self.image_processor:
            # 実際の画像処理はここに実装
            result['image'] = torch.randn(3, 224, 224)  # ダミー
            result['has_image'] = True
        else:
            result['has_image'] = False

        return result


class AEGISv2IntegratedTrainer:
    """
    AEGIS-v2.0-Phi3.5-thinking 統合トレーナー
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # コンポーネント初期化
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # モデルコンポーネント
        self.base_model = None  # Borea-phi3.5-instinct-jp (frozen)
        self.so8vit = None      # SO8VIT for multimodal processing
        self.ppo_trainer = PPOInternalInferenceTrainer(config.get('ppo_config', {}))

        # アルファゲートアニーリング
        self.alpha_gate = SigmoidAlphaGateAnnealing(config.get('alpha_gate_config', {}))

        # トレーニング設定
        self.max_epochs = config.get('max_epochs', 10)
        self.batch_size = config.get('batch_size', 1)  # RTX3060対応
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        self.learning_rate = config.get('learning_rate', 1e-6)
        self.save_steps = config.get('save_steps', 100)

        # 出力ディレクトリ
        self.output_dir = Path(config.get('output_dir', 'D:/webdataset/models/aegis_v2_phi35_thinking'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # トレーニング状態
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')

        # 統計情報
        self.training_stats = {
            'loss_history': [],
            'alpha_history': [],
            'inference_quality_history': [],
            'phase_transitions': [],
            'orthogonal_errors': [],
            'meta_control_actions': []
        }

        # 自動チェックポイント管理
        self.auto_cp_manager = AutoCheckpointManager(
            checkpoint_dir=self.output_dir / "checkpoints",
            max_checkpoints=config.get('max_checkpoints', 5),
            auto_save_interval=config.get('auto_save_interval', 180)  # 3分
        )

        # 電源投入時自動再開フラグ
        self.power_resume_detected = AutoCheckpointManager.detect_power_resume()

    def setup_models(self, base_model_path: str):
        """モデルセットアップ"""
        logger.info("[AEGIS-v2] Setting up models...")

        # Base model (Borea-phi3.5-instinct-jp) - frozen
        self.base_model = self._load_base_model(base_model_path)
        self._freeze_base_model()

        # SO8VIT for multimodal processing
        so8vit_config = self.config.get('so8vit_config', {})
        self.so8vit = SO8VIT(
            img_size=so8vit_config.get('img_size', 224),
            patch_size=so8vit_config.get('patch_size', 16),
            embed_dim=so8vit_config.get('embed_dim', 768),
            depth=so8vit_config.get('depth', 12),
            num_heads=so8vit_config.get('num_heads', 12),
            multimodal=True
        ).to(self.device)

        # PPO trainer setup
        self.ppo_trainer.setup_models(
            policy_model=self.base_model,
            value_model=self._create_value_head()
        )

        logger.info("[AEGIS-v2] Models setup completed")

    def _load_base_model(self, model_path: str):
        """Base model読み込み"""
        # Unslothまたは通常のtransformersで読み込み
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_path,
                max_seq_length=self.config.get('max_seq_length', 4096),
                dtype=None,
                load_in_4bit=False,  # BF16で読み込み
            )
            logger.info("[AEGIS-v2] Loaded base model with Unsloth")
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("[AEGIS-v2] Loaded base model with transformers")

        return model

    def _freeze_base_model(self):
        """Base modelの重みを凍結"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 選択的に一部の層をunfreeze（adapter layersなど）
        if hasattr(self.base_model, 'layers'):
            # 最後の数層のみトレーニング可能に
            num_unfrozen_layers = self.config.get('num_unfrozen_layers', 2)
            for layer in self.base_model.layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.info("[AEGIS-v2] Base model frozen (selective unfreezing applied)")

    def _create_value_head(self):
        """Value head作成"""
        # Base modelの最後の層を利用
        value_head = nn.Linear(self.base_model.config.hidden_size, 1)
        return value_head.to(self.device)

    def prepare_dataset(self, dataset_config: Dict[str, Any]) -> DataLoader:
        """データセット準備"""
        logger.info("[AEGIS-v2] Preparing dataset...")

        # データセット収集・クレンジング
        dataset_result = create_ppo_training_dataset(dataset_config)
        dataset_path = dataset_result['ppo_dataset_path']

        # Tokenizer取得
        tokenizer = self.base_model.tokenizer if hasattr(self.base_model, 'tokenizer') else None
        if not tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.get('tokenizer_path', 'microsoft/phi-3.5-mini-instruct'))

        # データセット作成
        dataset = MultimodalThinkingDataset(
            data_path=dataset_path,
            tokenizer=tokenizer,
            max_length=self.config.get('max_seq_length', 2048)
        )

        # DataLoader作成
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows対応
            pin_memory=True
        )

        logger.info(f"[AEGIS-v2] Dataset prepared: {len(dataset)} samples")
        return dataloader

    def train(self, dataloader: DataLoader):
        """統合トレーニング実行（自動チェックポイント対応）"""
        logger.info("[AEGIS-v2] Starting integrated training...")

        # 電源投入時自動再開チェック
        if self.power_resume_detected:
            latest_cp = self.auto_cp_manager.find_latest_checkpoint()
            if latest_cp and latest_cp.exists():
                logger.info(f"[AEGIS-v2] Power resume detected, loading checkpoint: {latest_cp}")
                self.load_checkpoint(str(latest_cp))
                logger.info("[AEGIS-v2] Checkpoint loaded, resuming training...")
            else:
                logger.info("[AEGIS-v2] Power resume detected but no checkpoint found, starting fresh")

        # オプティマイザー設定
        optimizer = self._setup_optimizer()
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        # 自動チェックポイント開始
        self.auto_cp_manager.start_auto_save(lambda auto=False, force=False: self.save_checkpoint(auto=auto, force=force))

        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                epoch_loss = 0
                epoch_steps = 0

                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.max_epochs}", initial=self.current_step % len(dataloader))

                # データローダーを途中から再開
                start_batch = self.current_step % len(dataloader)
                dataloader_iter = iter(dataloader)

                # 途中から開始するためにスキップ
                for _ in range(start_batch):
                    try:
                        next(dataloader_iter)
                    except StopIteration:
                        break

                for batch_idx, batch in enumerate(dataloader_iter, start=start_batch):
                    # バッチデータをデバイスに移動
                    batch = self._prepare_batch(batch)

                    # アルファゲートアニーリング
                    alpha, aux_info = self.alpha_gate.forward(epoch_loss / max(1, epoch_steps))

                    # 推論状態情報
                    inference_state = self._extract_inference_state(batch, aux_info)

                    # PPOトレーニングステップ
                    training_info = self.ppo_trainer.train_step({
                        'states': batch['input_ids'],
                        'actions': batch['input_ids'],  # 自己回帰の場合は同じ
                        'rewards': batch.get('rewards', torch.zeros_like(batch['input_ids'], dtype=torch.float)),
                        'old_log_probs': batch.get('old_log_probs', torch.zeros_like(batch['input_ids'], dtype=torch.float)),
                        'advantages': batch.get('advantages', torch.zeros_like(batch['input_ids'], dtype=torch.float)),
                        'returns': batch.get('returns', torch.zeros_like(batch['input_ids'], dtype=torch.float)),
                        'inference_state': inference_state
                    })

                    # SO8VITによるマルチモーダル処理（画像がある場合）
                    if batch.get('has_image', False).any():
                        self._process_multimodal_batch(batch, alpha)

                    # 損失計算と逆伝播
                    loss = training_info['total_loss']
                    epoch_loss += loss
                    epoch_steps += 1

                    # 勾配蓄積
                    loss = loss / self.gradient_accumulation_steps
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # 勾配更新
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.ppo_trainer.policy_model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.ppo_trainer.policy_model.parameters(), 1.0)
                            optimizer.step()

                        optimizer.zero_grad()

                    # 統計記録
                    self._record_training_stats(training_info, aux_info, alpha)

                    # プログレスバー更新
                    progress_bar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'alpha': f"{alpha:.4f}",
                        'temp': f"{training_info['current_temperature']:.2f}",
                        'quality': f"{training_info['inference_quality']:.3f}"
                    })

                    self.current_step += 1

                    # 手動チェックポイント保存（エポック完了時）
                    if self.current_step % self.save_steps == 0:
                        self.save_checkpoint()

                # エポック完了
                if epoch_steps > 0:
                    avg_epoch_loss = epoch_loss / epoch_steps
                    logger.info(f"[AEGIS-v2] Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

                    # 最良モデル保存
                    if avg_epoch_loss < self.best_loss:
                        self.best_loss = avg_epoch_loss
                        self.save_checkpoint(best=True)

        except KeyboardInterrupt:
            logger.info("[AEGIS-v2] Training interrupted by user")
            self.save_checkpoint(force=True)
        except Exception as e:
            logger.error(f"[AEGIS-v2] Training failed: {e}")
            self.save_checkpoint(force=True)
            raise
        finally:
            # 自動チェックポイント停止
            self.auto_cp_manager.stop_auto_save()

        # 最終チェックポイント保存
        self.save_checkpoint(final=True)

        # トレーニング完了レポート
        self._generate_training_report()

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """バッチデータ準備（魂の重み対応）"""
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                prepared_batch[key] = torch.tensor(value).to(self.device)
            elif key == 'soul_weights' and isinstance(value, list):
                # 魂の重みデータをテンソル化
                soul_weights_tensor = {}
                for sw_key, sw_value in value[0].items() if value else {}:
                    if isinstance(sw_value, (int, float)):
                        soul_weights_tensor[sw_key] = torch.tensor(sw_value).to(self.device)
                    elif isinstance(sw_value, list):
                        soul_weights_tensor[sw_key] = torch.tensor(sw_value).to(self.device)
                prepared_batch[key] = soul_weights_tensor
            else:
                prepared_batch[key] = value

        return prepared_batch

    def _extract_inference_state(self, batch: Dict[str, Any], aux_info: Dict[str, Any]) -> Dict[str, Any]:
        """推論状態情報抽出（魂の重み対応）"""
        inference_state = {
            'complexity_score': batch.get('complexity_score', torch.tensor(0.5)).mean().item(),
            'confidence_score': aux_info.get('inference_quality', 0.5),
            'abstraction_level': aux_info.get('alpha', 0.5),
            'goal_clarity': batch.get('quality_score', torch.tensor(0.5)).mean().item(),
            'adaptation_score': aux_info.get('phase_transition', {}).get('loss_gradient', 0.5)
        }

        # 魂の重みデータがある場合は統合
        if batch.get('has_soul_weights', False) and 'soul_weights' in batch:
            soul_weights = batch['soul_weights']
            inference_state.update({
                'soul_alpha': soul_weights.get('alpha_gate', 0.5),
                'soul_safety_score': torch.tensor(soul_weights.get('safety_head', [0.5, 0.5])).softmax(dim=-1)[0].item(),
                'soul_task_complexity': len(soul_weights.get('task_head', [0.0]*4)) / 4.0,
                'soul_pet_inertia': soul_weights.get('pet', 0.0),
                'has_soul_weights': True
            })
        else:
            inference_state['has_soul_weights'] = False

        return inference_state

    def _process_multimodal_batch(self, batch: Dict[str, Any], alpha: float):
        """マルチモーダルバッチ処理"""
        # 画像があるサンプルを処理
        image_mask = batch['has_image']
        if image_mask.any():
            images = batch.get('image', torch.randn(len(image_mask), 3, 224, 224).to(self.device))

            # SO8VITで画像処理
            image_embeds = self.so8vit.forward_features(images)[0]  # [batch_size, embed_dim]

            # テキスト埋め込み取得
            text_embeds = self.base_model.get_input_embeddings()(batch['input_ids'])  # [batch_size, seq_len, embed_dim]
            text_embeds = text_embeds.mean(dim=1)  # [batch_size, embed_dim]

            # モダリティ統合（アルファゲート適用）
            combined_embeds = alpha * image_embeds + (1 - alpha) * text_embeds

            # 統合表現をモデルにフィードバック
            # （実際の実装では、cross-attentionなどを通じて統合）

    def _setup_optimizer(self):
        """オプティマイザー設定"""
        # トレーニング可能なパラメータのみ
        trainable_params = []
        for name, param in self.ppo_trainer.policy_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        if self.so8vit:
            trainable_params.extend(list(self.so8vit.parameters()))

        optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )

        return optimizer

    def _record_training_stats(self, training_info: Dict[str, Any], aux_info: Dict[str, Any], alpha: float):
        """トレーニング統計記録"""
        self.training_stats['loss_history'].append(training_info['total_loss'])
        self.training_stats['alpha_history'].append(alpha)
        self.training_stats['inference_quality_history'].append(training_info['inference_quality'])

        if aux_info.get('phase_transition', {}).get('transition_detected'):
            self.training_stats['phase_transitions'].append({
                'step': self.current_step,
                'type': aux_info['phase_transition']['transition_type'],
                'info': aux_info['phase_transition']
            })

        self.training_stats['meta_control_actions'].append(training_info['meta_control_action'])

    def save_checkpoint(self, best: bool = False, final: bool = False, auto: bool = False, force: bool = False):
        """チェックポイント保存（自動チェックポイント対応）"""
        checkpoint_data = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'policy_model_state_dict': self.ppo_trainer.policy_model.state_dict(),
            'so8vit_state_dict': self.so8vit.state_dict() if self.so8vit else None,
            'ppo_trainer_state': self.ppo_trainer.__dict__,
            'alpha_gate_state': self.alpha_gate.__dict__,
            'training_stats': self.training_stats,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'auto_saved': auto,
            'force_saved': force
        }

        # 自動チェックポイントマネージャーを使用
        if auto or force:
            # 自動保存の場合はマネージャーに任せる
            saved_path = self.auto_cp_manager.save_checkpoint(checkpoint_data, "auto_checkpoint")
            return saved_path
        else:
            # 手動保存
            if final:
                filename = "checkpoint_final.pt"
            elif best:
                filename = "checkpoint_best.pt"
            else:
                filename = f"checkpoint_step_{self.current_step}.pt"

            checkpoint_path = self.output_dir / "checkpoints" / filename
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"[AEGIS-v2] Manual checkpoint saved: {checkpoint_path}")

            # 最新チェックポイントのシンボリックリンク
            if not best and not final:
                latest_path = self.output_dir / "checkpoints" / "checkpoint_latest.pt"
                if latest_path.exists():
                    latest_path.unlink()
                try:
                    latest_path.symlink_to(filename)
                except OSError:
                    # Windowsではシンボリックリンクが使えない場合がある
                    pass

            return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイント読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_step = checkpoint.get('step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        self.ppo_trainer.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        if self.so8vit and checkpoint.get('so8vit_state_dict'):
            self.so8vit.load_state_dict(checkpoint['so8vit_state_dict'])

        # 状態復元
        self.ppo_trainer.__dict__.update(checkpoint.get('ppo_trainer_state', {}))
        self.alpha_gate.__dict__.update(checkpoint.get('alpha_gate_state', {}))
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

        logger.info(f"[AEGIS-v2] Checkpoint loaded: {checkpoint_path}")

    def _generate_training_report(self):
        """トレーニングレポート生成"""
        report_path = self.output_dir / "training_report.json"

        # トレーニングサマリー
        summary = {
            'total_steps': self.current_step,
            'total_epochs': self.current_epoch + 1,
            'best_loss': self.best_loss,
            'final_loss': self.training_stats['loss_history'][-1] if self.training_stats['loss_history'] else None,
            'average_inference_quality': np.mean(self.training_stats['inference_quality_history']),
            'total_phase_transitions': len(self.training_stats['phase_transitions']),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        # グラフ生成
        self._generate_training_plots()

        logger.info(f"[AEGIS-v2] Training report generated: {report_path}")

    def _generate_training_plots(self):
        """トレーニンググラフ生成"""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # 損失推移グラフ
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_stats['loss_history'], label='Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('AEGIS-v2.0 Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

        # アルファ値推移グラフ
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_stats['alpha_history'], label='Alpha Value', color='orange')
        plt.axhline(y=(1 + math.sqrt(5)) / 2 * -2, color='red', linestyle='--',
                   label=f'φ^(-2) = {(1 + math.sqrt(5)) / 2 * -2:.3f}')
        plt.xlabel('Step')
        plt.ylabel('Alpha')
        plt.title('Alpha Gate Annealing Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'alpha_annealing.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 推論品質推移グラフ
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_stats['inference_quality_history'], label='Inference Quality', color='green')
        plt.xlabel('Step')
        plt.ylabel('Quality Score')
        plt.title('Internal Inference Quality')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'inference_quality.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_aegis_v2_training_config() -> Dict[str, Any]:
    """AEGIS-v2.0トレーニング設定作成"""
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_epochs': 10,
        'batch_size': 1,  # RTX3060対応
        'gradient_accumulation_steps': 4,
        'learning_rate': 1e-6,
        'weight_decay': 0.01,
        'max_seq_length': 2048,
        'save_steps': 100,
        'num_unfrozen_layers': 2,

        # 自動チェックポイント設定
        'auto_save_interval': 180,  # 3分
        'max_checkpoints': 5,       # ローリングストック上限

        'output_dir': 'D:/webdataset/models/aegis_v2_phi35_thinking',

        'so8vit_config': {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12
        },

        'ppo_config': {
            'embed_dim': 4096,
            'num_heads': 32,
            'max_thinking_tokens': 100,
            'ppo': {
                'clip_ratio': 0.2,
                'value_coeff': 0.5,
                'entropy_coeff': 0.01,
                'max_grad_norm': 0.5,
                'policy_lr': 1e-6,
                'value_lr': 1e-5
            },
            'meta_control': {
                'high_entropy_threshold': 2.0,
                'low_entropy_threshold': 0.5,
                'cooling_rate': 0.9,
                'heating_rate': 1.1,
                'adaptation_rate': 0.01
            }
        },

        'alpha_gate_config': {
            'initial_alpha': -0.5,  # 初期値 -0.5
            'target_alpha': 1.0,
            'sigmoid_k': 0.1,
            'max_steps': 10000
        },

        'dataset_config': {
            'total_samples': 50000,  # Phi3.5パラメータ数に基づく
            'domain_ratios': {
                'mathematics': 0.3,
                'science': 0.25,
                'programming': 0.2,
                'multimodal': 0.15,
                'reasoning': 0.1
            },
            'include_nsfw': True,  # 検出目的のみ
            'quality_thresholds': {'acceptable': 0.6},
            'license_filter': ['mit', 'apache-2.0'],
            'output_dir': 'D:/webdataset/datasets/ppo_training'
        }
    }

    return config


def run_complete_aegis_v2_pipeline():
    """
    完全自動AEGIS-v2.0パイプライン実行
    1. データセット収集・前処理
    2. モデルセットアップ
    3. トレーニング実行（自動チェックポイント対応）
    """
    logger.info("[AEGIS-v2] Starting complete AEGIS-v2.0 automated pipeline...")

    # AEGIS-v2.0トレーニング設定
    config = create_aegis_v2_training_config()

    try:
        # Phase 1: データセット収集・前処理
        logger.info("[AEGIS-v2] Phase 1: Dataset collection and preprocessing...")
        dataset_config = config['dataset_config']
        dataset_result = create_ppo_training_dataset(dataset_config)
        dataset_path = dataset_result['ppo_dataset_path']
        logger.info(f"[AEGIS-v2] Dataset prepared: {dataset_path}")

        # Phase 2: モデルセットアップ
        logger.info("[AEGIS-v2] Phase 2: Model setup...")
        trainer = AEGISv2IntegratedTrainer(config)
        base_model_path = "D:/webdataset/models/borea_phi35_instruct_jp/final"
        trainer.setup_models(base_model_path)
        logger.info("[AEGIS-v2] Models setup completed")

        # Phase 3: データセット準備
        logger.info("[AEGIS-v2] Phase 3: Dataset preparation...")
        dataloader = trainer.prepare_dataset(config['dataset_config'])
        logger.info("[AEGIS-v2] Dataset loaded and ready for training")

        # Phase 4: トレーニング実行（自動チェックポイント対応）
        logger.info("[AEGIS-v2] Phase 4: Training execution with auto-checkpointing...")
        trainer.train(dataloader)

        # Phase 5: 最終レポート生成
        logger.info("[AEGIS-v2] Phase 5: Final report generation...")
        trainer._generate_training_report()

        logger.info("[AEGIS-v2] Complete AEGIS-v2.0 pipeline finished successfully!")
        return True

    except Exception as e:
        logger.error(f"[AEGIS-v2] Pipeline failed: {e}")
        # エラー時もチェックポイント保存
        if 'trainer' in locals():
            trainer.save_checkpoint(force=True)
        raise


def create_automatic_resume_script():
    """電源投入時自動再開スクリプト生成"""
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS-v2.0 自動再開スクリプト
電源投入時に自動的にトレーニングを再開
"""

import os
import sys
import time
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("AEGIS-v2.0 Auto Resume Script")
    print("=" * 40)

    # 電源投入検出
    try:
        import psutil
        boot_time = psutil.boot_time()
        current_time = time.time()
        uptime_hours = (current_time - boot_time) / 3600

        if uptime_hours < 1.0:
            print(".1f"            print("Starting AEGIS-v2.0 training resume...")
        else:
            print(".1f"            print("Normal startup, starting fresh training...")

    except ImportError:
        print("psutil not available, assuming normal startup")
    except Exception as e:
        print(f"Could not detect power resume: {e}")

    # AEGIS-v2.0パイプライン実行
    try:
        from scripts.training.aegis_v2_training_pipeline import run_complete_aegis_v2_pipeline
        success = run_complete_aegis_v2_pipeline()
        if success:
            print("AEGIS-v2.0 training completed successfully!")
        else:
            print("AEGIS-v2.0 training failed!")
    except Exception as e:
        print(f"AEGIS-v2.0 training failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''

    script_path = Path(__file__).parent / "auto_resume_aegis_v2.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # 実行権限付与（Unix系の場合）
    try:
        os.chmod(script_path, 0o755)
    except:
        pass  # Windowsでは無視

    print(f"Auto resume script created: {script_path}")
    return script_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AEGIS-v2.0 Complete Automated Pipeline')
    parser.add_argument('--resume', action='store_true', help='Force resume from checkpoint')
    parser.add_argument('--create_resume_script', action='store_true', help='Create auto-resume script')

    args = parser.parse_args()

    if args.create_resume_script:
        create_automatic_resume_script()
    else:
        # 通常の実行
        success = run_complete_aegis_v2_pipeline()
        if success:
            print("AEGIS-v2.0 Phi3.5-thinking training completed successfully!")
        else:
            print("AEGIS-v2.0 training failed!")
            sys.exit(1)
