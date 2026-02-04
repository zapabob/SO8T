import json
import os
from pathlib import Path
from typing import List, Dict, Any

def generate_tool_calling_data():
    """
    Generates synthetic tool-calling samples for SO8T AEGIS models.
    Covers Academic Search and OSINT tools.
    """
    dest_dir = Path("c:/Users/downl/Desktop/SO8T/data/tool_calling")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    samples = [
        {
            "messages": [
                {"role": "user", "content": "2025年のLLM推論最適化に関する最新の論文をArxivで探してください。"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "arxiv_search",
                                "arguments": '{"query": "LLM inference optimization", "year": 2025, "sort_by": "cited"}'
                            }
                        }
                    ]
                }
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Sakana AIのAI Scientist 2について、BioRxivでも関連論文があるか確認してください。"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_def456",
                            "type": "function",
                            "function": {
                                "name": "biorxiv_search",
                                "arguments": '{"query": "AI Scientist 2 Sakana AI"}'
                            }
                        }
                    ]
                }
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "特定の組織に関連するOSINT情報を収集し、最近の動向をまとめて。"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_ghi789",
                            "type": "function",
                            "function": {
                                "name": "osint_aggregate",
                                "arguments": '{"query": "recent activities and data leaks", "depth": "advanced"}'
                            }
                        }
                    ]
                }
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "DeepSeek-V3の論文で引用されている重要な技術文献をリストアップして。"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_jkl012",
                            "type": "function",
                            "function": {
                                "name": "semantic_scholar_citations",
                                "arguments": '{"paper_id": "DeepSeek-V3", "limit": 10}'
                            }
                        }
                    ]
                }
            ]
        }
    ]
    
    output_file = dest_dir / "tool_calling_v1.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print(f"[TOOL_CALLING] Dataset generated at: {output_file}")
    return output_file

if __name__ == "__main__":
    generate_tool_calling_data()
