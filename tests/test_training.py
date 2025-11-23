"""
Tests for SO8T Training Components

Unit tests for training-related components including dataset loader, losses, and training loop.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import json
import tempfile

from training.so8t_dataset_loader import SO8TDataset, collate_so8t_batch, create_so8t_dataloader
from training.losses import SafetyAwareLoss, PETLoss, SafetyMetrics
from models.so8t_model import SO8TModelConfig


class TestSO8TDataset:
    """Test SO8TDataset class."""
    
    def create_test_data(self, num_samples: int = 10) -> Path:
        """Create temporary test data file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        for i in range(num_samples):
            sample = {
                "context": f"Test context {i}",
                "user_request": f"Test request {i}",
                "task_output": f"Test output {i}",
                "safety_label": ["ALLOW", "REFUSE", "ESCALATE"][i % 3],
                "safety_rationale": f"Test rationale {i}"
            }
            temp_file.write(json.dumps(sample) + '\n')
        
        temp_file.close()
        return Path(temp_file.name)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        data_file = self.create_test_data()
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        }
        mock_tokenizer.pad_token = "<pad>"
        
        try:
            dataset = SO8TDataset(
                data_path=data_file,
                tokenizer=mock_tokenizer,
                max_length=512
            )
            
            assert len(dataset) == 10
            assert dataset.safety_label_map == {"ALLOW": 0, "REFUSE": 1, "ESCALATE": 2}
            
        finally:
            data_file.unlink()
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        data_file = self.create_test_data()
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        }
        mock_tokenizer.pad_token = "<pad>"
        
        try:
            dataset = SO8TDataset(
                data_path=data_file,
                tokenizer=mock_tokenizer,
                max_length=512
            )
            
            # Test getting an item
            item = dataset[0]
            
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "task_labels" in item
            assert "safety_labels" in item
            assert "rationale_labels" in item
            
            # Check shapes
            assert item["input_ids"].shape == (10,)
            assert item["attention_mask"].shape == (10,)
            assert item["task_labels"].shape == (10,)
            assert item["safety_labels"].shape == ()
            assert item["rationale_labels"].shape == (10,)
            
        finally:
            data_file.unlink()
    
    def test_collate_function(self):
        """Test collate function."""
        # Create mock batch
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (10,)),
                "attention_mask": torch.ones(10),
                "task_labels": torch.randint(0, 1000, (10,)),
                "safety_labels": torch.tensor(0),
                "rationale_labels": torch.randint(0, 1000, (10,)),
                "rationale_attention_mask": torch.ones(10)
            },
            {
                "input_ids": torch.randint(0, 1000, (10,)),
                "attention_mask": torch.ones(10),
                "task_labels": torch.randint(0, 1000, (10,)),
                "safety_labels": torch.tensor(1),
                "rationale_labels": torch.randint(0, 1000, (10,)),
                "rationale_attention_mask": torch.ones(10)
            }
        ]
        
        # Test collate function
        collated = collate_so8t_batch(batch)
        
        assert collated["input_ids"].shape == (2, 10)
        assert collated["attention_mask"].shape == (2, 10)
        assert collated["task_labels"].shape == (2, 10)
        assert collated["safety_labels"].shape == (2,)
        assert collated["rationale_labels"].shape == (2, 10)
        assert collated["rationale_attention_mask"].shape == (2, 10)


class TestPETLoss:
    """Test PETLoss class."""
    
    def test_pet_loss_initialization(self):
        """Test PETLoss initialization."""
        pet_loss = PETLoss(lambda_pet=0.1)
        assert pet_loss.lambda_pet == 0.1
    
    def test_pet_loss_forward(self):
        """Test PETLoss forward pass."""
        pet_loss = PETLoss(lambda_pet=0.1)
        
        # Test with valid input
        hidden_states = torch.randn(2, 10, 512)
        loss = pet_loss(hidden_states)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_pet_loss_short_sequence(self):
        """Test PETLoss with short sequence."""
        pet_loss = PETLoss(lambda_pet=0.1)
        
        # Test with short sequence (should return 0)
        hidden_states = torch.randn(2, 2, 512)
        loss = pet_loss(hidden_states)
        
        assert loss.item() == 0.0


class TestSafetyAwareLoss:
    """Test SafetyAwareLoss class."""
    
    def test_loss_initialization(self):
        """Test SafetyAwareLoss initialization."""
        loss_fn = SafetyAwareLoss()
        
        assert loss_fn.task_weight == 1.0
        assert loss_fn.safety_weight == 2.0
        assert loss_fn.rationale_weight == 1.0
        assert loss_fn.pet_weight == 0.1
        assert loss_fn.safety_penalty_weight == 5.0
        assert loss_fn.escalate_reward_weight == 2.0
    
    def test_loss_forward(self):
        """Test SafetyAwareLoss forward pass."""
        loss_fn = SafetyAwareLoss()
        
        # Create test data
        batch_size, seq_len, vocab_size = 2, 10, 1000
        num_classes = 3
        hidden_size = 512
        
        task_logits = torch.randn(batch_size, seq_len, vocab_size)
        safety_logits = torch.randn(batch_size, num_classes)
        rationale_logits = torch.randn(batch_size, seq_len, vocab_size)
        task_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        safety_labels = torch.randint(0, num_classes, (batch_size,))
        rationale_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        losses = loss_fn(
            task_logits=task_logits,
            safety_logits=safety_logits,
            rationale_logits=rationale_logits,
            task_labels=task_labels,
            safety_labels=safety_labels,
            rationale_labels=rationale_labels,
            hidden_states=hidden_states,
            epoch=5,
            total_epochs=10
        )
        
        # Check outputs
        assert "total_loss" in losses
        assert "task_loss" in losses
        assert "safety_loss" in losses
        assert "rationale_loss" in losses
        assert "pet_loss" in losses
        assert "safety_penalty" in losses
        assert "escalate_reward" in losses
        
        # Check that all losses are valid
        for key, value in losses.items():
            assert isinstance(value, torch.Tensor)
            assert not torch.isnan(value)
            assert not torch.isinf(value)
    
    def test_loss_without_rationale(self):
        """Test SafetyAwareLoss without rationale labels."""
        loss_fn = SafetyAwareLoss()
        
        # Create test data without rationale
        batch_size, seq_len, vocab_size = 2, 10, 1000
        num_classes = 3
        hidden_size = 512
        
        task_logits = torch.randn(batch_size, seq_len, vocab_size)
        safety_logits = torch.randn(batch_size, num_classes)
        task_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        safety_labels = torch.randint(0, num_classes, (batch_size,))
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass without rationale
        losses = loss_fn(
            task_logits=task_logits,
            safety_logits=safety_logits,
            rationale_logits=None,
            task_labels=task_labels,
            safety_labels=safety_labels,
            rationale_labels=None,
            hidden_states=hidden_states,
            epoch=5,
            total_epochs=10
        )
        
        # Check that rationale loss is 0
        assert losses["rationale_loss"].item() == 0.0
    
    def test_pet_schedule(self):
        """Test PET scheduling."""
        loss_fn = SafetyAwareLoss()
        
        # Test different progress values
        assert loss_fn._pet_schedule(0.0) < 0.1  # Early training
        assert loss_fn._pet_schedule(0.5) < 0.5  # Mid training
        assert loss_fn._pet_schedule(0.8) > 0.8  # Late training
        assert loss_fn._pet_schedule(1.0) > 0.9  # End training


class TestSafetyMetrics:
    """Test SafetyMetrics class."""
    
    def test_metrics_initialization(self):
        """Test SafetyMetrics initialization."""
        metrics_fn = SafetyMetrics()
        assert metrics_fn.safety_labels == ("ALLOW", "REFUSE", "ESCALATE")
        assert metrics_fn.num_classes == 3
    
    def test_compute_metrics(self):
        """Test compute_metrics method."""
        metrics_fn = SafetyMetrics()
        
        # Create test data
        batch_size = 4
        num_classes = 3
        
        safety_logits = torch.randn(batch_size, num_classes)
        safety_labels = torch.randint(0, num_classes, (batch_size,))
        
        # Compute metrics
        metrics = metrics_fn.compute_metrics(safety_logits, safety_labels)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "accuracy", "allow_precision", "allow_recall", "allow_f1",
            "refuse_precision", "refuse_recall", "refuse_f1",
            "escalate_precision", "escalate_recall", "escalate_f1",
            "refuse_recall", "escalate_precision", "allow_precision", "safety_score"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1 or metrics[metric] == 0
    
    def test_safety_specific_metrics(self):
        """Test safety-specific metrics calculation."""
        metrics_fn = SafetyMetrics()
        
        # Create test data with known patterns
        safety_logits = torch.tensor([
            [2.0, 0.0, 0.0],  # Predict ALLOW
            [0.0, 2.0, 0.0],  # Predict REFUSE
            [0.0, 0.0, 2.0],  # Predict ESCALATE
            [2.0, 0.0, 0.0]   # Predict ALLOW
        ])
        safety_labels = torch.tensor([0, 1, 2, 0])  # All correct
        
        metrics = metrics_fn.compute_metrics(safety_logits, safety_labels)
        
        # All predictions are correct, so metrics should be high
        assert metrics["accuracy"] == 1.0
        assert metrics["refuse_recall"] == 1.0
        assert metrics["escalate_precision"] == 1.0
        assert metrics["allow_precision"] == 1.0
        assert metrics["safety_score"] == 1.0


class TestTrainingIntegration:
    """Integration tests for training components."""
    
    def test_dataset_and_loss_integration(self):
        """Test integration between dataset and loss function."""
        # Create test data
        data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        for i in range(5):
            sample = {
                "context": f"Test context {i}",
                "user_request": f"Test request {i}",
                "task_output": f"Test output {i}",
                "safety_label": ["ALLOW", "REFUSE", "ESCALATE"][i % 3],
                "safety_rationale": f"Test rationale {i}"
            }
            data_file.write(json.dumps(sample) + '\n')
        
        data_file.close()
        data_path = Path(data_file.name)
        
        try:
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10)
            }
            mock_tokenizer.pad_token = "<pad>"
            
            # Create dataset
            dataset = SO8TDataset(
                data_path=data_path,
                tokenizer=mock_tokenizer,
                max_length=512
            )
            
            # Create dataloader
            dataloader = create_so8t_dataloader(
                data_path=data_path,
                tokenizer=mock_tokenizer,
                batch_size=2,
                max_length=512
            )
            
            # Create loss function
            loss_fn = SafetyAwareLoss()
            
            # Test with a batch
            for batch in dataloader:
                # Mock model outputs
                task_logits = torch.randn(2, 10, 1000)
                safety_logits = torch.randn(2, 3)
                rationale_logits = torch.randn(2, 10, 1000)
                hidden_states = torch.randn(2, 10, 512)
                
                # Compute loss
                losses = loss_fn(
                    task_logits=task_logits,
                    safety_logits=safety_logits,
                    rationale_logits=rationale_logits,
                    task_labels=batch["task_labels"],
                    safety_labels=batch["safety_labels"],
                    rationale_labels=batch["rationale_labels"],
                    hidden_states=hidden_states,
                    epoch=0,
                    total_epochs=10
                )
                
                # Verify loss computation
                assert "total_loss" in losses
                assert not torch.isnan(losses["total_loss"])
                assert not torch.isinf(losses["total_loss"])
                
                break  # Only test first batch
        
        finally:
            data_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
