"""
Tests for SO8T Inference Components

Unit tests for inference-related components including agent runtime and logging middleware.
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import time

from inference.agent_runtime import SO8TAgentRuntime, run_agent
from inference.logging_middleware import AuditLogger, ComplianceReporter, LoggingMiddleware


class TestSO8TAgentRuntime:
    """Test SO8TAgentRuntime class."""
    
    def test_runtime_initialization(self):
        """Test runtime initialization."""
        config = {
            "base_model_name": "Qwen/Qwen2.5-7B-Instruct",
            "model_path": "test_model.pt",
            "use_gguf": False,
            "safety_threshold": 0.8,
            "confidence_threshold": 0.7,
            "max_length": 2048
        }
        
        with patch('inference.agent_runtime.AutoTokenizer') as mock_tokenizer, \
             patch('inference.agent_runtime.load_so8t_model') as mock_model:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = "<pad>"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock model
            mock_model_instance = Mock()
            mock_model.return_value = mock_model_instance
            
            runtime = SO8TAgentRuntime(config)
            
            assert runtime.config == config
            assert runtime.safety_labels == ["ALLOW", "REFUSE", "ESCALATE"]
            assert runtime.safety_threshold == 0.8
            assert runtime.confidence_threshold == 0.7
    
    def test_preprocess_request(self):
        """Test request preprocessing."""
        config = {
            "base_model_name": "Qwen/Qwen2.5-7B-Instruct",
            "max_length": 512
        }
        
        with patch('inference.agent_runtime.AutoTokenizer') as mock_tokenizer, \
             patch('inference.agent_runtime.load_so8t_model') as mock_model:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10)
            }
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock model
            mock_model_instance = Mock()
            mock_model.return_value = mock_model_instance
            
            runtime = SO8TAgentRuntime(config)
            
            # Test preprocessing
            inputs = runtime._preprocess_request(
                context="Test context",
                user_request="Test request"
            )
            
            assert "input_ids" in inputs
            assert "attention_mask" in inputs
            assert inputs["input_ids"].shape[0] == 1
            assert inputs["attention_mask"].shape[0] == 1
    
    def test_postprocess_response(self):
        """Test response postprocessing."""
        config = {
            "base_model_name": "Qwen/Qwen2.5-7B-Instruct",
            "safety_threshold": 0.8
        }
        
        with patch('inference.agent_runtime.AutoTokenizer') as mock_tokenizer, \
             patch('inference.agent_runtime.load_so8t_model') as mock_model:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock model
            mock_model_instance = Mock()
            mock_model.return_value = mock_model_instance
            
            runtime = SO8TAgentRuntime(config)
            
            # Test postprocessing
            safety_logits = torch.tensor([[2.0, 0.0, 0.0]])  # Predict ALLOW
            task_logits = torch.randn(1, 10, 1000)
            
            response = runtime._postprocess_response(safety_logits, task_logits)
            
            assert "decision" in response
            assert "rationale" in response
            assert "human_required" in response
            assert "confidence" in response
            assert "safety_probs" in response
            
            assert response["decision"] in ["ALLOW", "REFUSE", "ESCALATE"]
            assert isinstance(response["human_required"], bool)
            assert 0 <= response["confidence"] <= 1
    
    def test_process_request(self):
        """Test complete request processing."""
        config = {
            "base_model_name": "Qwen/Qwen2.5-7B-Instruct",
            "max_length": 512
        }
        
        with patch('inference.agent_runtime.AutoTokenizer') as mock_tokenizer, \
             patch('inference.agent_runtime.load_so8t_model') as mock_model:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10)
            }
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock model
            mock_model_instance = Mock()
            mock_model_instance.return_value = {
                "task_logits": torch.randn(1, 10, 1000),
                "safety_logits": torch.tensor([[2.0, 0.0, 0.0]]),
                "rationale_logits": torch.randn(1, 10, 1000),
                "hidden_states": torch.randn(1, 10, 512)
            }
            mock_model.return_value = mock_model_instance
            
            runtime = SO8TAgentRuntime(config)
            
            # Test request processing
            response = runtime.process_request(
                context="Test context",
                user_request="Test request",
                request_id="test_123"
            )
            
            assert "decision" in response
            assert "rationale" in response
            assert "human_required" in response
            assert "confidence" in response
            assert "request_id" in response
            assert response["request_id"] == "test_123"
    
    def test_process_request_error_handling(self):
        """Test request processing error handling."""
        config = {
            "base_model_name": "Qwen/Qwen2.5-7B-Instruct",
            "max_length": 512
        }
        
        with patch('inference.agent_runtime.AutoTokenizer') as mock_tokenizer, \
             patch('inference.agent_runtime.load_so8t_model') as mock_model:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones(1, 10)
            }
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock model to raise exception
            mock_model_instance = Mock()
            mock_model_instance.side_effect = Exception("Test error")
            mock_model.return_value = mock_model_instance
            
            runtime = SO8TAgentRuntime(config)
            
            # Test error handling
            response = runtime.process_request(
                context="Test context",
                user_request="Test request"
            )
            
            assert response["decision"] == "ESCALATE"
            assert "error" in response
            assert response["human_required"] == True
            assert response["confidence"] == 0.0


class TestAuditLogger:
    """Test AuditLogger class."""
    
    def test_logger_initialization(self):
        """Test AuditLogger initialization."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            logger = AuditLogger(config)
            
            assert logger.log_dir == Path(temp_dir)
            assert logger.max_log_size == 1024
            assert logger.max_backup_count == 3
    
    def test_log_audit(self):
        """Test audit logging."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            logger = AuditLogger(config)
            
            # Test audit logging
            logger.log_audit(
                request_id="test_123",
                context="Test context",
                user_request="Test request",
                decision="ALLOW",
                rationale="Test rationale",
                confidence=0.9,
                human_required=False,
                processing_time_ms=100.0,
                model_version="test_v1.0"
            )
            
            # Check that log file was created
            assert logger.audit_log.exists()
            
            # Check log content
            with open(logger.audit_log, 'r') as f:
                log_content = f.read()
                assert "test_123" in log_content
                assert "ALLOW" in log_content
                assert "Test rationale" in log_content
    
    def test_log_performance(self):
        """Test performance logging."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            logger = AuditLogger(config)
            
            # Test performance logging
            logger.log_performance(
                metric_name="processing_time",
                metric_value=100.0,
                tags={"decision": "ALLOW"}
            )
            
            # Check that performance log was created
            assert logger.performance_log.exists()
    
    def test_log_error(self):
        """Test error logging."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            logger = AuditLogger(config)
            
            # Test error logging
            logger.log_error(
                level="ERROR",
                message="Test error",
                exception=Exception("Test exception")
            )
            
            # Check that error log was created
            assert logger.error_log.exists()
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            logger = AuditLogger(config)
            
            # Log some test data
            logger.log_audit(
                request_id="test_1",
                context="Test context",
                user_request="Test request",
                decision="ALLOW",
                rationale="Test rationale",
                confidence=0.9,
                human_required=False,
                processing_time_ms=100.0,
                model_version="test_v1.0"
            )
            
            logger.log_performance("test_metric", 50.0)
            
            # Get statistics
            stats = logger.get_statistics()
            
            assert "total_requests" in stats
            assert "decision_counts" in stats
            assert "performance_metrics" in stats
            assert "log_files" in stats


class TestComplianceReporter:
    """Test ComplianceReporter class."""
    
    def test_reporter_initialization(self):
        """Test ComplianceReporter initialization."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            logger = AuditLogger(config)
            reporter = ComplianceReporter(logger)
            
            assert reporter.audit_logger == logger
            assert reporter.report_dir == logger.log_dir / "reports"
    
    def test_generate_daily_report(self):
        """Test daily report generation."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            logger = AuditLogger(config)
            reporter = ComplianceReporter(logger)
            
            # Generate report
            report = reporter.generate_daily_report("2025-01-27")
            
            assert "date" in report
            assert "total_requests" in report
            assert "decisions" in report
            assert "human_intervention_required" in report
            assert report["date"] == "2025-01-27"


class TestLoggingMiddleware:
    """Test LoggingMiddleware class."""
    
    def test_middleware_initialization(self):
        """Test LoggingMiddleware initialization."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            middleware = LoggingMiddleware(config)
            
            assert middleware.config == config
            assert middleware.audit_logger is not None
            assert middleware.compliance_reporter is not None
    
    def test_log_request_decorator(self):
        """Test log_request decorator."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            middleware = LoggingMiddleware(config)
            
            # Test function
            @middleware.log_request
            def test_function(context, user_request):
                return {
                    "decision": "ALLOW",
                    "rationale": "Test rationale",
                    "confidence": 0.9,
                    "human_required": False,
                    "model_version": "test_v1.0"
                }
            
            # Call function
            result = test_function(
                context="Test context",
                user_request="Test request"
            )
            
            assert result["decision"] == "ALLOW"
            assert result["rationale"] == "Test rationale"
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        config = {
            "log_dir": "test_logs",
            "max_log_size": 1024,
            "max_backup_count": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            
            middleware = LoggingMiddleware(config)
            
            # Get statistics
            stats = middleware.get_statistics()
            
            assert "total_requests" in stats
            assert "decision_counts" in stats
            assert "performance_metrics" in stats


class TestRunAgentFunction:
    """Test run_agent function."""
    
    def test_run_agent_basic(self):
        """Test basic run_agent functionality."""
        with patch('inference.agent_runtime.SO8TAgentRuntime') as mock_runtime_class:
            # Mock runtime instance
            mock_runtime = Mock()
            mock_runtime.process_request.return_value = {
                "decision": "ALLOW",
                "rationale": "Test rationale",
                "task_response": "Test response",
                "human_required": False,
                "confidence": 0.9
            }
            mock_runtime_class.return_value = mock_runtime
            
            # Test run_agent
            response = run_agent(
                context="Test context",
                user_request="Test request"
            )
            
            assert response["decision"] == "ALLOW"
            assert response["rationale"] == "Test rationale"
            assert response["human_required"] == False
            assert response["confidence"] == 0.9
    
    def test_run_agent_with_config(self):
        """Test run_agent with configuration file."""
        with patch('inference.agent_runtime.SO8TAgentRuntime') as mock_runtime_class, \
             patch('inference.agent_runtime.os.path.exists') as mock_exists:
            
            # Mock config file exists
            mock_exists.return_value = True
            
            # Mock config content
            with patch('inference.agent_runtime.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
                    "base_model_name": "test/model",
                    "safety_threshold": 0.9
                })
                
                # Mock runtime instance
                mock_runtime = Mock()
                mock_runtime.process_request.return_value = {
                    "decision": "REFUSE",
                    "rationale": "Test rationale",
                    "human_required": True,
                    "confidence": 0.8
                }
                mock_runtime_class.return_value = mock_runtime
                
                # Test run_agent with config
                response = run_agent(
                    context="Test context",
                    user_request="Test request",
                    config_path="test_config.json"
                )
                
                assert response["decision"] == "REFUSE"
                assert response["human_required"] == True


if __name__ == "__main__":
    pytest.main([__file__])
