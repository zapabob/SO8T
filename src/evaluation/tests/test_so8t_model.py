"""
Tests for SO8T Model

Unit tests for the SO8T Safe Agent model implementation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from models.so8t_model import SO8TModel, SO8TModelConfig, TaskHeadA, SafetyHeadB


class TestSO8TModelConfig:
    """Test SO8TModelConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SO8TModelConfig()
        
        assert config.base_model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.task_head_hidden_size == 4096
        assert config.safety_head_hidden_size == 2048
        assert config.safety_num_classes == 3
        assert config.rationale_max_length == 256
        assert config.pet_lambda == 0.1
        assert config.safety_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SO8TModelConfig(
            base_model_name="custom/model",
            task_head_hidden_size=2048,
            safety_head_hidden_size=1024,
            safety_num_classes=5,
            rationale_max_length=128,
            pet_lambda=0.2,
            safety_threshold=0.9
        )
        
        assert config.base_model_name == "custom/model"
        assert config.task_head_hidden_size == 2048
        assert config.safety_head_hidden_size == 1024
        assert config.safety_num_classes == 5
        assert config.rationale_max_length == 128
        assert config.pet_lambda == 0.2
        assert config.safety_threshold == 0.9


class TestTaskHeadA:
    """Test TaskHeadA class."""
    
    def test_initialization(self):
        """Test TaskHeadA initialization."""
        config = SO8TModelConfig()
        base_hidden_size = 512
        
        task_head = TaskHeadA(config, base_hidden_size)
        
        assert task_head.base_hidden_size == base_hidden_size
        assert task_head.task_head.in_features == base_hidden_size
        assert task_head.task_head.out_features == config.task_head_hidden_size
        assert task_head.vocab_projection.out_features == config.vocab_size
    
    def test_forward(self):
        """Test TaskHeadA forward pass."""
        config = SO8TModelConfig()
        base_hidden_size = 512
        batch_size = 2
        seq_len = 10
        
        task_head = TaskHeadA(config, base_hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, base_hidden_size)
        
        output = task_head(hidden_states)
        
        assert output.shape == (batch_size, seq_len, config.vocab_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestSafetyHeadB:
    """Test SafetyHeadB class."""
    
    def test_initialization(self):
        """Test SafetyHeadB initialization."""
        config = SO8TModelConfig()
        base_hidden_size = 512
        
        safety_head = SafetyHeadB(config, base_hidden_size)
        
        assert safety_head.base_hidden_size == base_hidden_size
        assert safety_head.safety_classifier[0].in_features == base_hidden_size
        assert safety_head.safety_classifier[-1].out_features == config.safety_num_classes
        assert safety_head.rationale_head[0].in_features == base_hidden_size
        assert safety_head.rationale_head[-1].out_features == config.vocab_size
    
    def test_forward(self):
        """Test SafetyHeadB forward pass."""
        config = SO8TModelConfig()
        base_hidden_size = 512
        batch_size = 2
        seq_len = 10
        
        safety_head = SafetyHeadB(config, base_hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, base_hidden_size)
        
        safety_logits, rationale_logits = safety_head(hidden_states)
        
        assert safety_logits.shape == (batch_size, config.safety_num_classes)
        assert rationale_logits.shape == (batch_size, seq_len, config.vocab_size)
        assert not torch.isnan(safety_logits).any()
        assert not torch.isnan(rationale_logits).any()
        assert not torch.isinf(safety_logits).any()
        assert not torch.isinf(rationale_logits).any()


class TestSO8TModel:
    """Test SO8TModel class."""
    
    @patch('models.so8t_model.AutoModel')
    def test_initialization(self, mock_auto_model):
        """Test SO8TModel initialization."""
        # Mock the base model
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 512
        mock_auto_model.from_pretrained.return_value = mock_base_model
        
        config = SO8TModelConfig()
        model = SO8TModel(config)
        
        assert model.config == config
        assert hasattr(model, 'task_head_a')
        assert hasattr(model, 'safety_head_b')
        assert model.safety_labels == ["ALLOW", "REFUSE", "ESCALATE"]
    
    @patch('models.so8t_model.AutoModel')
    def test_forward(self, mock_auto_model):
        """Test SO8TModel forward pass."""
        # Mock the base model
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 512
        mock_base_model.return_value = Mock(last_hidden_state=torch.randn(2, 10, 512))
        mock_auto_model.from_pretrained.return_value = mock_base_model
        
        config = SO8TModelConfig()
        model = SO8TModel(config)
        
        # Test input
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        assert "task_logits" in outputs
        assert "safety_logits" in outputs
        assert "rationale_logits" in outputs
        assert "safety_predictions" in outputs
        assert "safety_probs" in outputs
        
        # Check shapes
        assert outputs["task_logits"].shape == (2, 10, config.vocab_size)
        assert outputs["safety_logits"].shape == (2, config.safety_num_classes)
        assert outputs["rationale_logits"].shape == (2, 10, config.vocab_size)
        assert outputs["safety_predictions"].shape == (2,)
        assert outputs["safety_probs"].shape == (2, config.safety_num_classes)
    
    @patch('models.so8t_model.AutoModel')
    def test_forward_with_labels(self, mock_auto_model):
        """Test SO8TModel forward pass with labels."""
        # Mock the base model
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 512
        mock_base_model.return_value = Mock(last_hidden_state=torch.randn(2, 10, 512))
        mock_auto_model.from_pretrained.return_value = mock_base_model
        
        config = SO8TModelConfig()
        model = SO8TModel(config)
        
        # Test input with labels
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        task_labels = torch.randint(0, 1000, (2, 10))
        safety_labels = torch.randint(0, 3, (2,))
        rationale_labels = torch.randint(0, 1000, (2, 10))
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=task_labels,
            safety_labels=safety_labels,
            rationale_labels=rationale_labels,
            return_dict=True
        )
        
        assert "loss" in outputs
        assert "task_loss" in outputs
        assert "safety_loss" in outputs
        assert "rationale_loss" in outputs
        
        # Check that losses are computed
        assert outputs["loss"] is not None
        assert outputs["task_loss"] is not None
        assert outputs["safety_loss"] is not None
        assert outputs["rationale_loss"] is not None
    
    @patch('models.so8t_model.AutoModel')
    def test_generate_safe_response(self, mock_auto_model):
        """Test generate_safe_response method."""
        # Mock the base model
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 512
        mock_base_model.return_value = Mock(last_hidden_state=torch.randn(1, 10, 512))
        mock_auto_model.from_pretrained.return_value = mock_base_model
        
        config = SO8TModelConfig()
        model = SO8TModel(config)
        
        # Test input
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        # Generate response
        response = model.generate_safe_response(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        assert "decision" in response
        assert "rationale" in response
        assert "human_required" in response
        assert "confidence" in response
        assert "safety_probs" in response
        
        # Check decision is valid
        assert response["decision"] in ["ALLOW", "REFUSE", "ESCALATE"]
        assert isinstance(response["human_required"], bool)
        assert 0 <= response["confidence"] <= 1
    
    @patch('models.so8t_model.AutoModel')
    def test_get_safety_metrics(self, mock_auto_model):
        """Test get_safety_metrics method."""
        # Mock the base model
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 512
        mock_auto_model.from_pretrained.return_value = mock_base_model
        
        config = SO8TModelConfig()
        model = SO8TModel(config)
        
        # Mock outputs
        outputs = {
            "safety_predictions": torch.tensor([0, 1, 2, 0]),
            "safety_probs": torch.tensor([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1]
            ])
        }
        
        metrics = model.get_safety_metrics(outputs)
        
        assert "allow_rate" in metrics
        assert "refuse_rate" in metrics
        assert "escalate_rate" in metrics
        assert "avg_confidence" in metrics
        assert "total_samples" in metrics
        
        # Check that rates sum to 1
        total_rate = metrics["allow_rate"] + metrics["refuse_rate"] + metrics["escalate_rate"]
        assert abs(total_rate - 1.0) < 1e-6


class TestModelIntegration:
    """Integration tests for SO8T model."""
    
    @patch('models.so8t_model.AutoModel')
    def test_end_to_end_inference(self, mock_auto_model):
        """Test end-to-end inference pipeline."""
        # Mock the base model
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 512
        mock_base_model.return_value = Mock(last_hidden_state=torch.randn(1, 10, 512))
        mock_auto_model.from_pretrained.return_value = mock_base_model
        
        config = SO8TModelConfig()
        model = SO8TModel(config)
        model.eval()
        
        # Test input
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        with torch.no_grad():
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Generate response
            response = model.generate_safe_response(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get metrics
            metrics = model.get_safety_metrics(outputs)
        
        # Verify outputs
        assert outputs["task_logits"].shape == (1, 10, config.vocab_size)
        assert outputs["safety_logits"].shape == (1, config.safety_num_classes)
        assert response["decision"] in ["ALLOW", "REFUSE", "ESCALATE"]
        assert metrics["total_samples"] == 1
    
    @patch('models.so8t_model.AutoModel')
    def test_batch_processing(self, mock_auto_model):
        """Test batch processing."""
        # Mock the base model
        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 512
        mock_base_model.return_value = Mock(last_hidden_state=torch.randn(4, 10, 512))
        mock_auto_model.from_pretrained.return_value = mock_base_model
        
        config = SO8TModelConfig()
        model = SO8TModel(config)
        model.eval()
        
        # Test batch input
        batch_size = 4
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        attention_mask = torch.ones(batch_size, 10)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Verify batch processing
        assert outputs["task_logits"].shape == (batch_size, 10, config.vocab_size)
        assert outputs["safety_logits"].shape == (batch_size, config.safety_num_classes)
        assert outputs["safety_predictions"].shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__])
