#!/usr/bin/env python3
"""
SO8T/thinking QLoRA Inference Script

Load and run inference with SO8T/thinking QLoRA model.
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Local imports
sys.path.append('src')

class SO8TThinkingInference:
    """SO8T/thinking model inference class."""

    def __init__(self, base_model_path: str, adapter_path: str):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path

        print("Loading SO8T/thinking QLoRA model...")
        self.load_model()

    def load_model(self):
        """Load the SO8T-enhanced QLoRA model."""
        # Quantization config for inference
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.adapter_path,
            device_map="auto"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("SO8T/thinking model loaded successfully!")
        print(f"Base model: {self.base_model_path}")
        print(f"Adapter: {self.adapter_path}")

    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate response using SO8T/thinking model."""
        # Add reasoning markers for SO8T
        if any(keyword in prompt.lower() for keyword in ['solve', 'prove', 'calculate', 'reason']):
            enhanced_prompt = f"<so8t_reasoning_mathematical>\n{prompt}\n</so8t_reasoning_mathematical>"
        else:
            enhanced_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the enhanced prompt part if it was added
        if enhanced_prompt != prompt:
            response = full_response.replace(enhanced_prompt, "").strip()
        else:
            response = full_response

        # 🧠 SO8T/thinking 推論モード: /thinkingの中身は表示せず、thinkのみ表示
        # 物理学者ボブの視点で、思考プロセスを隠して最終回答のみ表示
        if "<think>" in response or "<thinking>" in response:
            # thinkingタグが見つかった場合、thinking内容を隠して最終回答のみ表示
            import re

            # <think>...</think> または <thinking>...</thinking> を削除
            response = re.sub(r'<think>.*?</think>', '[thinking...]', response, flags=re.DOTALL)
            response = re.sub(r'<thinking>.*?</thinking>', '[thinking...]', response, flags=re.DOTALL)

            # 最終回答部分を抽出（もしある場合）
            if "[thinking...]" in response:
                parts = response.split("[thinking...]")
                if len(parts) > 1:
                    # thinking後の部分を最終回答として表示
                    final_answer = parts[-1].strip()
                    if final_answer:
                        response = f"🤔 Thinking complete.\n\n💡 {final_answer}"
                    else:
                        response = "🤔 Thinking complete. (No final answer provided)"

        return response

    def chat(self):
        """Interactive chat mode."""
        print("=" * 60)
        print("SO8T/thinking Interactive Chat")
        print("Type 'quit' to exit")
        print("=" * 60)

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            print("SO8T thinking...", end=" ", flush=True)
            response = self.generate_response(user_input)
            print("\nSO8T:", response)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SO8T/thinking Inference")
    parser.add_argument("--base-model", type=str, default="models/Borea-Phi-3.5-mini-Instruct-Jp", help="Base model path")
    parser.add_argument("--adapter-path", type=str, required=True, help="LoRA adapter path")
    parser.add_argument("--prompt", type=str, help="Single prompt to process")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")

    args = parser.parse_args()

    # Initialize inference
    inference = SO8TThinkingInference(args.base_model, args.adapter_path)

    if args.chat:
        # Interactive chat mode
        inference.chat()
    elif args.prompt:
        # Single prompt mode
        print("Prompt:", args.prompt)
        print("Generating response...")
        response = inference.generate_response(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print("\nSO8T Response:")
        print(response)
    else:
        print("Use --prompt for single inference or --chat for interactive mode")

if __name__ == "__main__":
    main()


