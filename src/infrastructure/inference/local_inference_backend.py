import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalInferenceBackend:
    """ローカル推論バックエンドの基底クラス"""
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

class OllamaBackend(LocalInferenceBackend):
    """Ollama API 経由の推論バックエンド"""
    def __init__(self, model_name: str = "aegis-phi3.5:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        logger.info(f"OllamaBackend initialized with model: {model_name}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_ctx": kwargs.get("num_ctx", 4096),
                "top_p": kwargs.get("top_p", 0.9),
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: {str(e)}"

class LlamaCppBackend(LocalInferenceBackend):
    """llama-cpp-python を使用した直接推論バックエンド"""
    def __init__(self, model_path: Union[str, Path], **kwargs):
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=kwargs.get("n_ctx", 4096),
                n_gpu_layers=kwargs.get("n_gpu_layers", -1), # Default to full GPU offload
                verbose=False
            )
            logger.info(f"LlamaCppBackend initialized with model: {model_path}")
        except ImportError:
            logger.error("llama-cpp-python is not installed.")
            self.llm = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.llm:
            return "Error: llama-cpp-python not installed."
        
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" if system_prompt else prompt
        
        try:
            output = self.llm(
                full_prompt,
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
                stop=["<|im_end|>", "</s>"],
                echo=False
            )
            return output["choices"][0]["text"]
        except Exception as e:
            logger.error(f"LlamaCpp generation failed: {e}")
            return f"Error: {str(e)}"

class InferenceManager:
    """推論バックエンドのファクトリおよびマネージャ"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backend_type = self.config.get("backend", "ollama")
        self.backend = self._init_backend()

    def _init_backend(self) -> LocalInferenceBackend:
        if self.backend_type == "ollama":
            return OllamaBackend(
                model_name=self.config.get("model", "aegis-phi3.5-v3:latest"),
                base_url=self.config.get("ollama_url", "http://localhost:11434")
            )
        elif self.backend_type == "llama_cpp":
            return LlamaCppBackend(
                model_path=self.config.get("model_path", "models/aegis-phi3.5-v3.gguf"),
                n_gpu_layers=self.config.get("n_gpu_layers", -1)
            )
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")

    def ask(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """エージェントが推論を行うためのメインメソッド"""
        return self.backend.generate(prompt, system_prompt, **kwargs)
