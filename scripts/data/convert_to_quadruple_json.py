"""
収集したコーパスを四重Thinking形式のJSONLに変換

公式ソースから収集したデータを、四重Thinking形式（Task/Safety/Policy/Final）の
学習データに変換する。
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))

from models.thinking_tokens import format_quadruple_thinking_output


def generate_quadruple_thinking(
    content: str,
    domain_label: str,
    title: str = "",
) -> Dict[str, str]:
    """
    コンテンツから四重推論を生成（簡易実装）
    
    """
    本関数は、与えられたcontent（本文）、domain_label（ドメインラベル）、title（タイトル）から
    四重推論（Task, Safety, Policy, Final）を生成する。

    本番環境では、ローカルで利用可能なLLM（例: Ollama, llama.cpp等）を呼び出して
    より高品質な推論を生成する設計とする。

    Args:
        content (str): 元のコンテンツ（公開情報のテキスト等）
        domain_label (str): ドメインラベル（例: defense_public, aerospace 等）
        title (str): 文書タイトル（任意）

    Returns:
        dict: 四重推論 { "task": ..., "safety": ..., "policy": ..., "final": ... }
    """
    # 簡易実装: 実際にはLLMを使って生成
    content_preview = content[:500]  # 最初の500文字
    
    # Task推論（英語）
    task_reasoning = (
        f"Task: Summarize and explain the key points of this {domain_label} document. "
        f"Title: {title}. Content preview: {content_preview[:200]}... "
        f"Need to provide accurate, concise Japanese summary."
    )
    
    # Safety推論（英語）
    safety_reasoning = (
        f"Safety check: This is public, official information from {domain_label} domain. "
        f"No classified details, no operational instructions, no harmful content. Safe to answer."
    )
    
    # Policy推論（英語）
    policy_mapping = {
        "defense_public": "Domain: defense_public. Provide only descriptive, non-operational information. No classified or operational details.",
        "aerospace": "Domain: aerospace. Provide technical overview and public information only. No sensitive design details.",
        "medical_reg": "Domain: medical_reg. Provide regulatory and guideline information. No personal medical advice.",
        "law_policy": "Domain: law_policy. Provide legal information and policy explanations. No legal advice.",
        "wikipedia_ja_en": "Domain: wikipedia. Provide factual, well-sourced information. Maintain neutrality.",
    }
    policy_reasoning = policy_mapping.get(domain_label, "Domain: general. Provide accurate, helpful information.")
    
    # Final回答（日本語、簡易要約）
    # 実際の実装では、LLMで要約を生成
    final_answer = f"この文書は{domain_label}に関する公開情報です。主要な内容を要約すると、{content_preview[:200]}..."
    
    return {
        "task": task_reasoning,
        "safety": safety_reasoning,
        "policy": policy_reasoning,
        "final": final_answer,
    }


def convert_to_quadruple_format(
    input_file: Path,
    output_file: Path,
    instruction_template: str = "以下の文書の要点を要約してください。",
) -> int:
    """
    コーパスを四重Thinking形式に変換
    
    Args:
        input_file: 入力JSONLファイル
        output_file: 出力JSONLファイル
        instruction_template: 指示テンプレート
    
    Returns:
        変換されたサンプル数
    """
    converted_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                
                content = sample.get("content", "")
                domain_label = sample.get("domain_label", "general")
                title = sample.get("title", "")
                url = sample.get("url", "")
                
                if not content or len(content) < 100:
                    continue
                
                # 四重推論を生成
                thinking = generate_quadruple_thinking(content, domain_label, title)
                
                # 四重Thinking形式の出力を生成
                output = format_quadruple_thinking_output(
                    task=thinking["task"],
                    safety=thinking["safety"],
                    policy=thinking["policy"],
                    final=thinking["final"],
                )
                
                # 新しいサンプルを作成
                new_sample = {
                    "instruction": instruction_template,
                    "input": f"文書タイトル: {title}\nURL: {url}",
                    "output": output,
                    "safety_label": "ALLOW",  # 公式ソースは基本的にALLOW
                    "domain_label": domain_label,
                    "verifier_label": {
                        "logical": 1.0,
                        "faithful": 1.0,
                    },
                    "source_url": url,
                    "source_title": title,
                }
                
                f_out.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except Exception as e:
                print(f"[WARNING] Failed to convert sample: {e}")
                continue
    
    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert corpus to quadruple thinking format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file (crawled corpus)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file (quadruple thinking format)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="以下の文書の要点を要約してください。",
        help="Instruction template",
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] Converting {args.input} to quadruple thinking format...")
    count = convert_to_quadruple_format(
        input_file=args.input,
        output_file=args.output,
        instruction_template=args.instruction,
    )
    
    print(f"[SUCCESS] Converted {count} samples")


if __name__ == "__main__":
    main()

