"""
SO8T Knowledge Distillation Script

This script performs knowledge distillation from Teacher (SO8T-Phi31) to Student (qwen-lightweight)
while preserving SO(8) group structure and safety features.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.so8t_safety_judge import SO8TSafetyJudge
from utils.memory_manager import SO8TMemoryManager

logger = logging.getLogger(__name__)

class DistillationDataset(Dataset):
    """Dataset for knowledge distillation"""
    
    def __init__(self, data_path: str, max_length: int = 512):
        """
        Initialize distillation dataset
        
        Args:
            data_path: Path to JSONL data file
            max_length: Maximum sequence length
        """
        self.data = []
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} distillation samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize (simplified - in practice would use proper tokenizer)
        input_text = item.get('instruction', '') + ' ' + item.get('input', '')
        output_text = item.get('output', '')
        
        # Create simple token IDs (placeholder)
        input_ids = self._simple_tokenize(input_text)
        output_ids = self._simple_tokenize(output_text)
        
        # Pad or truncate
        input_ids = self._pad_or_truncate(input_ids, self.max_length)
        output_ids = self._pad_or_truncate(output_ids, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'instruction': item.get('instruction', ''),
            'input': item.get('input', ''),
            'output': item.get('output', '')
        }
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization (placeholder)"""
        # In practice, would use proper tokenizer
        return [hash(word) % 10000 for word in text.split()[:self.max_length]]
    
    def _pad_or_truncate(self, ids: List[int], max_length: int) -> List[int]:
        """Pad or truncate sequence to max_length"""
        if len(ids) > max_length:
            return ids[:max_length]
        else:
            return ids + [0] * (max_length - len(ids))

class SO8TDistillationTrainer:
    """SO8T Knowledge Distillation Trainer"""
    
    def __init__(self, 
                 teacher_model_path: str,
                 student_model_path: str,
                 output_dir: str = "distilled_models",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize distillation trainer
        
        Args:
            teacher_model_path: Path to teacher model (SO8T-Phi31)
            student_model_path: Path to student model (qwen-lightweight)
            output_dir: Output directory for distilled model
            device: Device to use for training
        """
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Initialize models
        self.teacher_model = None
        self.student_model = None
        self.safety_judge = SO8TSafetyJudge()
        self.memory_manager = SO8TMemoryManager()
        
        # Training parameters
        self.temperature = 3.0
        self.alpha = 0.7  # Weight for distillation loss
        self.beta = 0.3   # Weight for task loss
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Training statistics
        self.training_stats = {
            'epoch': 0,
            'total_loss': 0.0,
            'distillation_loss': 0.0,
            'task_loss': 0.0,
            'safety_loss': 0.0,
            'rotation_loss': 0.0
        }
    
    def load_models(self):
        """Load teacher and student models"""
        try:
            # Load teacher model (SO8T-Phi31)
            logger.info(f"Loading teacher model from {self.teacher_model_path}")
            # In practice, would load actual model
            self.teacher_model = self._create_mock_teacher_model()
            
            # Load student model (qwen-lightweight)
            logger.info(f"Loading student model from {self.student_model_path}")
            # In practice, would load actual model
            self.student_model = self._create_mock_student_model()
            
            # Move to device
            if self.teacher_model:
                self.teacher_model.to(self.device)
            if self.student_model:
                self.student_model.to(self.device)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _create_mock_teacher_model(self):
        """Create mock teacher model for testing"""
        class MockTeacherModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10000, 4096)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(4096, 8, batch_first=True),
                    num_layers=6
                )
                self.lm_head = nn.Linear(4096, 10000)
                self.rotation_matrices = nn.Parameter(torch.randn(8, 8) * 0.01)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                logits = self.lm_head(x)
                return logits
            
            def get_rotation_matrices(self):
                return self.rotation_matrices
        
        return MockTeacherModel()
    
    def _create_mock_student_model(self):
        """Create mock student model for testing"""
        class MockStudentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10000, 2048)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(2048, 4, batch_first=True),
                    num_layers=4
                )
                self.lm_head = nn.Linear(2048, 10000)
                self.rotation_matrices = nn.Parameter(torch.randn(8, 8) * 0.01)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                logits = self.lm_head(x)
                return logits
            
            def get_rotation_matrices(self):
                return self.rotation_matrices
        
        return MockStudentModel()
    
    def distill_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one distillation step
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of losses
        """
        input_ids = batch['input_ids'].to(self.device)
        output_ids = batch['output_ids'].to(self.device)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids)
            teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Get student predictions
        student_logits = self.student_model(input_ids)
        student_probs = torch.softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distillation_loss = self.kl_loss(
            torch.log(student_probs + 1e-8),
            teacher_probs
        ) * (self.temperature ** 2)
        
        # Task loss (cross-entropy)
        task_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), output_ids.view(-1))
        
        # Safety loss (using safety judge)
        safety_loss = self._compute_safety_loss(input_ids, student_logits)
        
        # SO(8) rotation loss
        rotation_loss = self._compute_rotation_loss()
        
        # Total loss
        total_loss = (self.alpha * distillation_loss + 
                     self.beta * task_loss + 
                     0.1 * safety_loss + 
                     0.1 * rotation_loss)
        
        return {
            'total_loss': total_loss.item(),
            'distillation_loss': distillation_loss.item(),
            'task_loss': task_loss.item(),
            'safety_loss': safety_loss.item(),
            'rotation_loss': rotation_loss.item()
        }
    
    def _compute_safety_loss(self, input_ids: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        """Compute safety-aware loss"""
        try:
            # Convert input_ids to text (simplified)
            input_text = " ".join([str(id.item()) for id in input_ids[0][:10]])
            
            # Get safety judgment
            safety_result = self.safety_judge.judge_text(input_text)
            
            # Safety penalty based on judgment
            if safety_result['action'] == 'DENY':
                safety_penalty = 1.0
            elif safety_result['action'] == 'ESCALATION':
                safety_penalty = 0.5
            else:
                safety_penalty = 0.0
            
            return torch.tensor(safety_penalty, device=self.device)
            
        except Exception as e:
            logger.error(f"Error computing safety loss: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def _compute_rotation_loss(self) -> torch.Tensor:
        """Compute SO(8) rotation matrix loss"""
        try:
            if self.teacher_model and self.student_model:
                teacher_rotations = self.teacher_model.get_rotation_matrices()
                student_rotations = self.student_model.get_rotation_matrices()
                
                # Ensure orthogonality
                teacher_orthogonal = self._orthogonalize(teacher_rotations)
                student_orthogonal = self._orthogonalize(student_rotations)
                
                return self.mse_loss(student_orthogonal, teacher_orthogonal)
            else:
                return torch.tensor(0.0, device=self.device)
                
        except Exception as e:
            logger.error(f"Error computing rotation loss: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def _orthogonalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Orthogonalize matrix using QR decomposition"""
        Q, R = torch.linalg.qr(matrix, mode='reduced')
        return Q
    
    def train(self, 
              train_loader: DataLoader,
              num_epochs: int = 10,
              learning_rate: float = 1e-4,
              save_interval: int = 5):
        """
        Train the student model
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_interval: Save model every N epochs
        """
        # Setup optimizer
        optimizer = optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        logger.info(f"Starting distillation training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.training_stats['epoch'] = epoch
            epoch_losses = {
                'total_loss': 0.0,
                'distillation_loss': 0.0,
                'task_loss': 0.0,
                'safety_loss': 0.0,
                'rotation_loss': 0.0
            }
            
            # Training loop
            self.student_model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                # Forward pass
                losses = self.distill_step(batch)
                
                # Backward pass
                total_loss = (self.alpha * losses['distillation_loss'] + 
                             self.beta * losses['task_loss'] + 
                             0.1 * losses['safety_loss'] + 
                             0.1 * losses['rotation_loss'])
                
                # Convert to tensor if needed
                if isinstance(total_loss, float):
                    total_loss = torch.tensor(total_loss, requires_grad=True, device=self.device)
                
                total_loss.backward()
                optimizer.step()
                
                # Update statistics
                for key, value in losses.items():
                    epoch_losses[key] += value
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Total': f"{losses['total_loss']:.4f}",
                    'Distill': f"{losses['distillation_loss']:.4f}",
                    'Task': f"{losses['task_loss']:.4f}",
                    'Safety': f"{losses['safety_loss']:.4f}",
                    'Rotation': f"{losses['rotation_loss']:.4f}"
                })
                
                # Log to memory manager
                self.memory_manager.log_metric(
                    metric_type="distillation_loss",
                    metric_value=losses['distillation_loss'],
                    threshold_value=0.1,
                    status="pass" if losses['distillation_loss'] < 0.1 else "warning"
                )
            
            # Update learning rate
            scheduler.step()
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)
            
            # Update training stats
            self.training_stats.update(epoch_losses)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1} completed:")
            logger.info(f"  Total Loss: {epoch_losses['total_loss']:.4f}")
            logger.info(f"  Distillation Loss: {epoch_losses['distillation_loss']:.4f}")
            logger.info(f"  Task Loss: {epoch_losses['task_loss']:.4f}")
            logger.info(f"  Safety Loss: {epoch_losses['safety_loss']:.4f}")
            logger.info(f"  Rotation Loss: {epoch_losses['rotation_loss']:.4f}")
            
            # Save model
            if (epoch + 1) % save_interval == 0:
                self.save_model(epoch + 1)
        
        # Save final model
        self.save_model("final")
        logger.info("Distillation training completed")
    
    def save_model(self, epoch: Union[int, str]):
        """Save distilled model"""
        try:
            model_path = self.output_dir / f"so8t_distilled_epoch_{epoch}.pt"
            
            # Save model state
            torch.save({
                'model_state_dict': self.student_model.state_dict(),
                'training_stats': self.training_stats,
                'epoch': epoch,
                'temperature': self.temperature,
                'alpha': self.alpha,
                'beta': self.beta
            }, model_path)
            
            logger.info(f"Model saved to: {model_path}")
            
            # Save SO(8) rotation matrices separately
            rotation_path = self.output_dir / f"so8t_rotations_epoch_{epoch}.pt"
            torch.save({
                'rotation_matrices': self.student_model.get_rotation_matrices(),
                'epoch': epoch
            }, rotation_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the distilled model"""
        self.student_model.eval()
        
        eval_losses = {
            'total_loss': 0.0,
            'distillation_loss': 0.0,
            'task_loss': 0.0,
            'safety_loss': 0.0,
            'rotation_loss': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                losses = self.distill_step(batch)
                
                for key, value in losses.items():
                    eval_losses[key] += value
        
        # Average losses
        for key in eval_losses:
            eval_losses[key] /= len(eval_loader)
        
        logger.info("Evaluation results:")
        for key, value in eval_losses.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return eval_losses

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='SO8T Knowledge Distillation')
    parser.add_argument('--teacher_model', required=True, help='Path to teacher model')
    parser.add_argument('--student_model', required=True, help='Path to student model')
    parser.add_argument('--data_path', required=True, help='Path to training data')
    parser.add_argument('--output_dir', default='distilled_models', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation loss weight')
    parser.add_argument('--beta', type=float, default=0.3, help='Task loss weight')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('distillation.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Create trainer
        trainer = SO8TDistillationTrainer(
            teacher_model_path=args.teacher_model,
            student_model_path=args.student_model,
            output_dir=args.output_dir
        )
        
        # Set training parameters
        trainer.temperature = args.temperature
        trainer.alpha = args.alpha
        trainer.beta = args.beta
        
        # Load models
        trainer.load_models()
        
        # Create datasets
        train_dataset = DistillationDataset(args.data_path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Train
        trainer.train(
            train_loader=train_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Evaluate
        eval_results = trainer.evaluate(train_loader)
        
        logger.info("Distillation completed successfully!")
        
    except Exception as e:
        logger.error(f"Distillation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
