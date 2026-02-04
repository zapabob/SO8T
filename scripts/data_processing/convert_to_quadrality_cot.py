import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Mocking the formatting tool if not available
try:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))
    from models.thinking_tokens import format_thinking_output
except ImportError:
    def format_thinking_output(thinking, final, **kwargs):
        return f"<think>\n{thinking}\n</think>\n{final}"

class QuadralityConverter:
    """
    Converts various CoT/Tool formats to SO8T Quadrality Reasoning.
    Vector -> Spinor+ -> Spinor- -> Integration
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1. Extract instruction/input/answer
        messages = sample.get("messages", [])
        instruction = ""
        user_input = ""
        answer = ""
        
        if messages:
            for msg in messages:
                if msg["role"] == "user":
                    user_input = msg["content"]
                if msg["role"] == "assistant":
                    answer = msg.get("content", "")
                    if "tool_calls" in msg:
                        answer += f"\n[TOOL_CALLS] {json.dumps(msg['tool_calls'])}"
        else:
            # Check for R1-CoT specific fields
            user_input = sample.get("query", sample.get("input", ""))
            instruction = sample.get("instruction", "")
            
            # Extract existing reasoning
            existing_cot = sample.get("think", "")
            
            # Extract answer/output
            if "r1-gen" in sample:
                r1_gen = sample["r1-gen"]
                if "</think>" in r1_gen:
                    answer = r1_gen.split("</think>")[-1].strip()
                else:
                    answer = r1_gen.strip()
            else:
                answer = sample.get("output", sample.get("text", sample.get("answers", "")))

        if not answer and not user_input:
            return None

        # Clean up existing_cot if it was extracted from answer in previous step
        if not existing_cot and "<think>" in answer:
            parts = answer.split("</think>")
            existing_cot = parts[0].replace("<think>", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""

        # 3. Build Quadrality Phases
        title = sample.get("metadata", {}).get("dataset", "mcp_cot")
        
        vector = f"[Vector_State]\n- Context: {title}\n- Goal: Process tool-calling or complex reasoning request.\n- Initial State: {user_input[:200]}..."
        
        spinor_plus = f"[Spinor_Plus_Logic]\n- Path: Deductive analysis of the request constraints.\n- Logic: Mapping inputs to required tool schemas or logical steps.\n- Existing Context: {existing_cot[:200]}..." if existing_cot else \
                      f"[Spinor_Plus_Logic]\n- Path: Standard deductive solution path."

        spinor_minus = f"[Spinor_Minus_Synthesis]\n- Critique: What are the edge cases? Is the tool call safe or optimal?\n- Abduction: Exploring alternative interpretations of the user intent."
        
        integration = f"[Quadrality_Integration]\n- Synthesis: Reconciling standard logic with critical alternatives.\n- Execution: Finalizing the solution/tool-call sequence."

        thinking = "\n\n".join([vector, spinor_plus, spinor_minus, integration])
        
        formatted_output = format_thinking_output(thinking=thinking, final=answer, use_redacted=True)

        return {
            "instruction": instruction,
            "input": user_input,
            "output": formatted_output,
            "metadata": {
                "source": sample.get("metadata", {}).get("source", "hf_fetch"),
                "converted_to_quadrality": True
            }
        }

    def process_file(self, input_path: Path, output_path: Path):
        count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "r", encoding="utf-8") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                try:
                    sample = json.loads(line)
                    converted = self.convert_sample(sample)
                    if converted:
                        f_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        count += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing line: {e}")
                    continue
        print(f"Converted {count} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    converter = QuadralityConverter(verbose=args.verbose)
    converter.process_file(Path(args.input), Path(args.output))
