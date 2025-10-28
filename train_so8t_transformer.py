"""
SO8T Transformer Training Script

This script trains the complete SO8T Transformer model from scratch,
replacing the base Qwen2.5 model with SO8T-native components.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.so8t_transformer import SO8TTransformerForCausalLM, SO8TTransformerConfig
from training.so8t_dataset_loader import SO8TDatasetLoader
from utils.memory_monitor import MemoryMonitor
from utils.checkpoint_manager import CheckpointManager


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('so8t_transformer_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_so8t_config(config: dict) -> SO8TTransformerConfig:
    """Create SO8T Transformer configuration."""
    return SO8TTransformerConfig(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        intermediate_size=config['model']['intermediate_size'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_key_value_heads=config['model']['num_key_value_heads'],
        hidden_act=config['model']['hidden_act'],
        max_position_embeddings=config['model']['max_position_embeddings'],
        rms_norm_eps=config['model']['rms_norm_eps'],
        rope_theta=config['model']['rope_theta'],
        attention_bias=config['model']['attention_bias'],
        attention_dropout=config['model']['attention_dropout'],
        use_cache=config['model']['use_cache'],
        # SO8T specific parameters
        rotation_dim=config['so8t']['rotation_dim'],
        safety_weight=config['so8t']['safety_weight'],
        cmd_weight=config['so8t']['cmd_weight'],
        pet_lambda=config['so8t']['pet_lambda'],
        group_monitoring=config['so8t']['group_monitoring'],
        # Memory optimization
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        use_flash_attention=config['training']['use_flash_attention'],
    )


def setup_model(config: dict, device: torch.device) -> SO8TTransformerForCausalLM:
    """Setup SO8T Transformer model."""
    logger = logging.getLogger(__name__)
    
    # Create SO8T configuration
    so8t_config = create_so8t_config(config)
    
    # Create model
    model = SO8TTransformerForCausalLM(so8t_config)
    
    # Move to device
    model = model.to(device)
    
    # Enable gradient checkpointing if specified
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        
    logger.info(f"SO8T Transformer model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def setup_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Setup optimizer for SO8T Transformer."""
    logger = logging.getLogger(__name__)
    
    # Get optimizer parameters
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    logger.info(f"Optimizer created with learning_rate={learning_rate}, weight_decay={weight_decay}")
    
    return optimizer


def setup_dataloader(config: dict) -> DataLoader:
    """Setup data loader for SO8T training."""
    logger = logging.getLogger(__name__)
    
    # Create dataset loader
    dataset_loader = SO8TDatasetLoader(
        data_path=config['data']['train_path'],
        tokenizer_name=config['model']['tokenizer_name'],
        max_length=config['data']['max_length'],
        safety_labels=config['data']['safety_labels'],
        authority_labels=config['data']['authority_labels']
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset_loader,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=config['training']['dataloader_pin_memory'],
        collate_fn=dataset_loader.collate_fn
    )
    
    logger.info(f"Data loader created with {len(dataloader)} batches")
    
    return dataloader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    epoch: int
) -> dict:
    """Train one epoch of SO8T Transformer."""
    logger = logging.getLogger(__name__)
    
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_safety_loss = 0.0
    total_authority_loss = 0.0
    num_batches = 0
    
    # Memory monitor
    memory_monitor = MemoryMonitor(device)
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else None
        safety_labels = batch['safety_labels'].to(device) if 'safety_labels' in batch else None
        authority_labels = batch['authority_labels'].to(device) if 'authority_labels' in batch else None
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            safety_labels=safety_labels,
            authority_labels=authority_labels,
            return_dict=True
        )
        
        # Get losses
        loss = outputs['loss']
        task_loss = outputs['task_loss']
        safety_loss = outputs['safety_loss']
        authority_loss = outputs['authority_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training']['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        if task_loss is not None:
            total_task_loss += task_loss.item()
        if safety_loss is not None:
            total_safety_loss += safety_loss.item()
        if authority_loss is not None:
            total_authority_loss += authority_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'task_loss': f"{task_loss.item() if task_loss is not None else 0:.4f}",
            'safety_loss': f"{safety_loss.item() if safety_loss is not None else 0:.4f}",
            'authority_loss': f"{authority_loss.item() if authority_loss is not None else 0:.4f}",
            'gpu_mem': f"{memory_monitor.get_memory_usage():.1f}GB"
        })
        
        # Log every N steps
        if batch_idx % config['training']['logging_steps'] == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}: "
                f"loss={loss.item():.4f}, "
                f"task_loss={task_loss.item() if task_loss is not None else 0:.4f}, "
                f"safety_loss={safety_loss.item() if safety_loss is not None else 0:.4f}, "
                f"authority_loss={authority_loss.item() if authority_loss is not None else 0:.4f}, "
                f"gpu_mem={memory_monitor.get_memory_usage():.1f}GB"
            )
    
    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_task_loss = total_task_loss / num_batches
    avg_safety_loss = total_safety_loss / num_batches
    avg_authority_loss = total_authority_loss / num_batches
    
    return {
        'loss': avg_loss,
        'task_loss': avg_task_loss,
        'safety_loss': avg_safety_loss,
        'authority_loss': avg_authority_loss
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SO8T Transformer')
    parser.add_argument('--config', type=str, default='configs/so8t_transformer_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto, cuda, cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting SO8T Transformer training")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Setup model
    model = setup_model(config, device)
    
    # Setup optimizer
    optimizer = setup_optimizer(model, config)
    
    # Setup data loader
    dataloader = setup_dataloader(config)
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=config['training']['output_dir'],
        max_checkpoints=config['training']['save_total_limit']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint(args.resume)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Resumed from checkpoint: {args.resume}, epoch: {start_epoch}")
    
    # Training loop
    logger.info("Starting training loop")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"Starting epoch {epoch}")
        
        # Train epoch
        metrics = train_epoch(model, dataloader, optimizer, device, config, epoch)
        
        # Log epoch metrics
        logger.info(
            f"Epoch {epoch} completed: "
            f"loss={metrics['loss']:.4f}, "
            f"task_loss={metrics['task_loss']:.4f}, "
            f"safety_loss={metrics['safety_loss']:.4f}, "
            f"authority_loss={metrics['authority_loss']:.4f}"
        )
        
        # Save checkpoint
        if epoch % config['training']['save_steps'] == 0:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config['training']['num_epochs'],
        metrics=metrics,
        is_final=True
    )
    logger.info(f"Final model saved: {final_checkpoint_path}")
    
    logger.info("SO8T Transformer training completed")


if __name__ == "__main__":
    main()
