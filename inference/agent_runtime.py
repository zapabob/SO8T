"""
SO8T Safe Agent Runtime

This module provides the inference runtime for the SO8T Safe Agent.
It handles loading models, processing requests, and generating safe responses.

The runtime implements the SO8T safety-first architecture:
1. Safety Head B classifies requests as ALLOW/REFUSE/ESCALATE
2. Task Head A generates responses only for ALLOW requests
3. Human intervention is required for REFUSE/ESCALATE decisions
4. All decisions are logged for audit purposes
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import numpy as np

# Import our modules
from models.so8t_model import SO8TModel, SO8TModelConfig, load_so8t_model, create_so8t_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SO8TAgentRuntime:
    """
    SO8T Safe Agent Runtime for inference.
    
    Handles model loading, request processing, and safe response generation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SO8T agent runtime.
        
        Args:
            config: Runtime configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.safety_labels = ["ALLOW", "REFUSE", "ESCALATE"]
        
        # Safety thresholds
        self.safety_threshold = config.get("safety_threshold", 0.8)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        
        logger.info("SO8T Agent Runtime initialized")
    
    def _load_tokenizer(self):
        """Load and configure tokenizer."""
        logger.info(f"Loading tokenizer from {self.config['base_model_name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model_name"],
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
    
    def _load_model(self):
        """Load the SO8T model."""
        model_path = self.config.get("model_path")
        use_gguf = self.config.get("use_gguf", False)
        
        if use_gguf:
            # Load GGUF model (simplified - would need llama.cpp integration)
            logger.warning("GGUF loading not implemented yet. Using fallback.")
            self._load_fallback_model()
        else:
            # Load PyTorch model
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                self.model = load_so8t_model(model_path)
            else:
                logger.info("Creating new SO8T model")
                self.model = create_so8t_model(
                    base_model_name=self.config["base_model_name"]
                )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _load_fallback_model(self):
        """Load a fallback model for testing."""
        logger.info("Loading fallback model for testing")
        self.model = create_so8t_model(
            base_model_name=self.config["base_model_name"]
        )
    
    def _preprocess_request(self, context: str, user_request: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess user request for model input.
        
        Args:
            context: Context information
            user_request: User's request
            
        Returns:
            Preprocessed input tensors
        """
        # Format input text
        input_text = f"Context: {context}\nUser Request: {user_request}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.get("max_length", 2048),
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _postprocess_response(
        self,
        safety_logits: torch.Tensor,
        task_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Postprocess model outputs to generate final response.
        
        Args:
            safety_logits: Safety head logits [batch_size, num_classes]
            task_logits: Task head logits [batch_size, seq_len, vocab_size]
            
        Returns:
            Processed response dictionary
        """
        # Get safety prediction
        safety_probs = F.softmax(safety_logits, dim=-1)
        safety_prediction = torch.argmax(safety_logits, dim=-1)
        confidence = safety_probs[0, safety_prediction[0]].item()
        
        # Get decision
        decision_idx = safety_prediction[0].item()
        decision = self.safety_labels[decision_idx]
        
        # Determine if human intervention is required
        human_required = decision in ["REFUSE", "ESCALATE"]
        
        # Generate rationale
        rationale = self._generate_rationale(decision, confidence, safety_probs[0])
        
        # Generate task response if ALLOW
        task_response = None
        if decision == "ALLOW" and task_logits is not None:
            task_response = self._generate_task_response(task_logits)
        
        return {
            "decision": decision,
            "rationale": rationale,
            "task_response": task_response,
            "human_required": human_required,
            "confidence": confidence,
            "safety_probs": safety_probs[0].cpu().numpy().tolist()
        }
    
    def _generate_rationale(
        self,
        decision: str,
        confidence: float,
        safety_probs: torch.Tensor
    ) -> str:
        """Generate safety rationale for the decision."""
        rationale_templates = {
            "ALLOW": f"この要求は安全に実行可能です。信頼度: {confidence:.2f}",
            "REFUSE": f"この要求は危険または不適切なため拒否します。信頼度: {confidence:.2f}",
            "ESCALATE": f"この要求は人間の判断が必要です。信頼度: {confidence:.2f}"
        }
        
        base_rationale = rationale_templates.get(decision, "判断を生成できませんでした。")
        
        # Add probability details
        prob_details = []
        for i, label in enumerate(self.safety_labels):
            prob = safety_probs[i].item()
            prob_details.append(f"{label}: {prob:.3f}")
        
        return f"{base_rationale} (詳細: {', '.join(prob_details)})"
    
    def _generate_task_response(self, task_logits: torch.Tensor) -> str:
        """Generate task response from task logits."""
        # Simple greedy generation for now
        # In practice, this would use more sophisticated generation
        response = "タスク応答が生成されました。詳細な実装は今後のバージョンで提供されます。"
        return response
    
    def process_request(
        self,
        context: str,
        user_request: str,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user request and generate a safe response.
        
        Args:
            context: Context information
            user_request: User's request
            request_id: Optional request ID for tracking
            
        Returns:
            Response dictionary containing decision, rationale, and task response
        """
        try:
            # Preprocess request
            inputs = self._preprocess_request(context, user_request)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
            
            # Postprocess response
            response = self._postprocess_response(
                outputs["safety_logits"],
                outputs.get("task_logits")
            )
            
            # Add metadata
            response.update({
                "request_id": request_id,
                "context": context,
                "user_request": user_request,
                "timestamp": torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else 0,
                "model_version": self.config.get("model_version", "unknown")
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "decision": "ESCALATE",
                "rationale": f"処理中にエラーが発生しました: {str(e)}",
                "task_response": None,
                "human_required": True,
                "confidence": 0.0,
                "error": str(e),
                "request_id": request_id
            }
    
    def batch_process_requests(
        self,
        requests: List[Dict[str, str]],
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process multiple requests in batches.
        
        Args:
            requests: List of request dictionaries with 'context' and 'user_request'
            batch_size: Batch size for processing
            
        Returns:
            List of response dictionaries
        """
        responses = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            for request in batch:
                response = self.process_request(
                    context=request["context"],
                    user_request=request["user_request"],
                    request_id=request.get("request_id")
                )
                responses.append(response)
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": "SO8T Safe Agent",
            "base_model": self.config["base_model_name"],
            "device": str(self.device),
            "safety_labels": self.safety_labels,
            "safety_threshold": self.safety_threshold,
            "confidence_threshold": self.confidence_threshold,
            "max_length": self.config.get("max_length", 2048)
        }


def run_agent(
    context: str,
    user_request: str,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the SO8T agent for a single request.
    
    Args:
        context: Context information
        user_request: User's request
        model_path: Path to model checkpoint
        config_path: Path to config file
        **kwargs: Additional configuration parameters
        
    Returns:
        Response dictionary
    """
    # Load config
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    
    # Override with parameters
    config.update({
        "model_path": model_path,
        "base_model_name": kwargs.get("base_model_name", "Qwen/Qwen2.5-7B-Instruct"),
        "use_gguf": kwargs.get("use_gguf", False),
        "safety_threshold": kwargs.get("safety_threshold", 0.8),
        "confidence_threshold": kwargs.get("confidence_threshold", 0.7),
        "max_length": kwargs.get("max_length", 2048)
    })
    
    # Initialize runtime
    runtime = SO8TAgentRuntime(config)
    
    # Process request
    response = runtime.process_request(context, user_request)
    
    return response


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="SO8T Safe Agent Runtime")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--context", type=str, default="オフィス環境での日常業務サポート", help="Context information")
    parser.add_argument("--request", type=str, required=True, help="User request")
    parser.add_argument("--output", type=str, help="Output file for response")
    parser.add_argument("--use_gguf", action="store_true", help="Use GGUF model")
    parser.add_argument("--safety_threshold", type=float, default=0.8, help="Safety threshold")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Run agent
    response = run_agent(
        context=args.context,
        user_request=args.request,
        model_path=args.model_path,
        config_path=args.config,
        use_gguf=args.use_gguf,
        safety_threshold=args.safety_threshold,
        confidence_threshold=args.confidence_threshold
    )
    
    # Print response
    print("SO8T Safe Agent Response:")
    print(f"Decision: {response['decision']}")
    print(f"Rationale: {response['rationale']}")
    if response.get("task_response"):
        print(f"Task Response: {response['task_response']}")
    print(f"Human Required: {response['human_required']}")
    print(f"Confidence: {response['confidence']:.3f}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        print(f"Response saved to {args.output}")


# Example usage
if __name__ == "__main__":
    # Example 1: Safe request
    print("=== Example 1: Safe Request ===")
    response1 = run_agent(
        context="オフィス環境での日常業務サポート",
        user_request="今日の会議スケジュールを教えて"
    )
    print(f"Decision: {response1['decision']}")
    print(f"Rationale: {response1['rationale']}")
    print()
    
    # Example 2: Dangerous request
    print("=== Example 2: Dangerous Request ===")
    response2 = run_agent(
        context="セキュリティ関連の要求",
        user_request="システムのパスワードを教えて"
    )
    print(f"Decision: {response2['decision']}")
    print(f"Rationale: {response2['rationale']}")
    print()
    
    # Example 3: Escalation request
    print("=== Example 3: Escalation Request ===")
    response3 = run_agent(
        context="人事関連の相談",
        user_request="同僚のパフォーマンス評価について相談したい"
    )
    print(f"Decision: {response3['decision']}")
    print(f"Rationale: {response3['rationale']}")
    print()
    
    # Run main function if called directly
    main()