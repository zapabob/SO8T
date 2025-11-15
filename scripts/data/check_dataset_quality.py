#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット品質チェックスクリプト

/thinkingモデル用のデータセットが適切な形式になっているか確認
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

def check_dataset_quality(dataset_path: Path) -> Dict[str, Any]:
    """
    データセットの品質をチェック
    
    Args:
        dataset_path: データセットファイルパス
    
    Returns:
        品質チェック結果
    """
    results = {
        "total_samples": 0,
        "valid_format": 0,
        "invalid_format": 0,
        "has_think_task": 0,
        "has_think_safety": 0,
        "has_think_policy": 0,
        "has_final": 0,
        "has_all_thinking_sections": 0,
        "has_phi35_template": 0,
        "has_complete_format": 0,
        "output_lengths": [],
        "errors": [],
        "sample_preview": None
    }
    
    print(f"[CHECK] Analyzing dataset: {dataset_path}")
    print("="*80)
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                samples.append(sample)
                results["total_samples"] += 1
                
                output = sample.get("output", "")
                results["output_lengths"].append(len(output))
                
                # 四重推論セクションのチェック
                has_task = "<think-task>" in output
                has_safety = "<think-safety>" in output
                has_policy = "<think-policy>" in output
                has_final = "<final>" in output
                
                if has_task:
                    results["has_think_task"] += 1
                if has_safety:
                    results["has_think_safety"] += 1
                if has_policy:
                    results["has_think_policy"] += 1
                if has_final:
                    results["has_final"] += 1
                
                # すべてのセクションがあるか
                if has_task and has_safety and has_policy and has_final:
                    results["has_all_thinking_sections"] += 1
                
                # Phi-3.5チャットテンプレート形式のチェック
                has_system = "<|system|>" in output
                has_user = "<|user|>" in output
                has_assistant = "<|assistant|>" in output
                
                if has_system and has_user and has_assistant:
                    results["has_phi35_template"] += 1
                
                # 完全な形式（四重推論 + Phi-3.5テンプレート）
                if (has_task and has_safety and has_policy and has_final and 
                    has_system and has_user and has_assistant):
                    results["has_complete_format"] += 1
                    results["valid_format"] += 1
                else:
                    results["invalid_format"] += 1
                    missing = []
                    if not has_task:
                        missing.append("think-task")
                    if not has_safety:
                        missing.append("think-safety")
                    if not has_policy:
                        missing.append("think-policy")
                    if not has_final:
                        missing.append("final")
                    if not has_system:
                        missing.append("<|system|>")
                    if not has_user:
                        missing.append("<|user|>")
                    if not has_assistant:
                        missing.append("<|assistant|>")
                    if line_num <= 10:  # 最初の10件のエラーのみ記録
                        results["errors"].append({
                            "line": line_num,
                            "missing": missing
                        })
                
                # サンプルプレビュー（最初の1件）
                if results["sample_preview"] is None and results["total_samples"] == 1:
                    results["sample_preview"] = {
                        "keys": list(sample.keys()),
                        "output_preview": output[:500] if output else "",
                        "output_length": len(output)
                    }
                    
            except json.JSONDecodeError as e:
                results["errors"].append({
                    "line": line_num,
                    "error": f"JSON decode error: {e}"
                })
                results["invalid_format"] += 1
            except Exception as e:
                results["errors"].append({
                    "line": line_num,
                    "error": f"Unexpected error: {e}"
                })
                results["invalid_format"] += 1
    
    # 統計情報
    if results["output_lengths"]:
        results["avg_output_length"] = sum(results["output_lengths"]) / len(results["output_lengths"])
        results["min_output_length"] = min(results["output_lengths"])
        results["max_output_length"] = max(results["output_lengths"])
    
    return results


def print_quality_report(results: Dict[str, Any]):
    """品質レポートを表示"""
    print("\n" + "="*80)
    print("DATASET QUALITY REPORT")
    print("="*80)
    
    print(f"\n[OVERVIEW]")
    print(f"  Total samples: {results['total_samples']:,}")
    print(f"  Valid format: {results['valid_format']:,} ({results['valid_format']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Invalid format: {results['invalid_format']:,} ({results['invalid_format']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[THINKING SECTIONS]")
    print(f"  Has <think-task>: {results['has_think_task']:,} ({results['has_think_task']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <think-safety>: {results['has_think_safety']:,} ({results['has_think_safety']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <think-policy>: {results['has_think_policy']:,} ({results['has_think_policy']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <final>: {results['has_final']:,} ({results['has_final']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has all thinking sections: {results['has_all_thinking_sections']:,} ({results['has_all_thinking_sections']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[PHI-3.5 TEMPLATE]")
    print(f"  Has Phi-3.5 template: {results['has_phi35_template']:,} ({results['has_phi35_template']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[COMPLETE FORMAT]")
    print(f"  Has complete format (thinking + template): {results['has_complete_format']:,} ({results['has_complete_format']/max(results['total_samples'], 1)*100:.1f}%)")
    
    if results.get("output_lengths"):
        print(f"\n[OUTPUT LENGTH]")
        print(f"  Average: {results['avg_output_length']:.0f} characters")
        print(f"  Min: {results['min_output_length']:,} characters")
        print(f"  Max: {results['max_output_length']:,} characters")
    
    if results["errors"]:
        print(f"\n[ERRORS] (showing first 10)")
        for error in results["errors"][:10]:
            if "missing" in error:
                print(f"  Line {error['line']}: Missing sections: {', '.join(error['missing'])}")
            else:
                print(f"  Line {error['line']}: {error.get('error', 'Unknown error')}")
    
    if results["sample_preview"]:
        print(f"\n[SAMPLE PREVIEW]")
        print(f"  Keys: {', '.join(results['sample_preview']['keys'])}")
        print(f"  Output length: {results['sample_preview']['output_length']:,} characters")
        print(f"  Output preview (first 500 chars):")
        print(f"  {results['sample_preview']['output_preview']}")
    
    print("\n" + "="*80)
    
    # 品質評価
    quality_score = (results['has_complete_format'] / max(results['total_samples'], 1)) * 100
    print(f"\n[QUALITY SCORE]")
    if quality_score >= 95:
        print(f"  Score: {quality_score:.1f}% - EXCELLENT (適切)")
    elif quality_score >= 80:
        print(f"  Score: {quality_score:.1f}% - GOOD (概ね適切)")
    elif quality_score >= 60:
        print(f"  Score: {quality_score:.1f}% - FAIR (改善が必要)")
    else:
        print(f"  Score: {quality_score:.1f}% - POOR (不適切、再生成が必要)")
    
    print("="*80)


def main():
    dataset_path = Path(r"D:\webdataset\processed\thinking_quadruple\quadruple_thinking_20251114_102426.jsonl")
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    results = check_dataset_quality(dataset_path)
    print_quality_report(results)


if __name__ == "__main__":
    main()






# -*- coding: utf-8 -*-
"""
データセット品質チェックスクリプト

/thinkingモデル用のデータセットが適切な形式になっているか確認
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

def check_dataset_quality(dataset_path: Path) -> Dict[str, Any]:
    """
    データセットの品質をチェック
    
    Args:
        dataset_path: データセットファイルパス
    
    Returns:
        品質チェック結果
    """
    results = {
        "total_samples": 0,
        "valid_format": 0,
        "invalid_format": 0,
        "has_think_task": 0,
        "has_think_safety": 0,
        "has_think_policy": 0,
        "has_final": 0,
        "has_all_thinking_sections": 0,
        "has_phi35_template": 0,
        "has_complete_format": 0,
        "output_lengths": [],
        "errors": [],
        "sample_preview": None
    }
    
    print(f"[CHECK] Analyzing dataset: {dataset_path}")
    print("="*80)
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                samples.append(sample)
                results["total_samples"] += 1
                
                output = sample.get("output", "")
                results["output_lengths"].append(len(output))
                
                # 四重推論セクションのチェック
                has_task = "<think-task>" in output
                has_safety = "<think-safety>" in output
                has_policy = "<think-policy>" in output
                has_final = "<final>" in output
                
                if has_task:
                    results["has_think_task"] += 1
                if has_safety:
                    results["has_think_safety"] += 1
                if has_policy:
                    results["has_think_policy"] += 1
                if has_final:
                    results["has_final"] += 1
                
                # すべてのセクションがあるか
                if has_task and has_safety and has_policy and has_final:
                    results["has_all_thinking_sections"] += 1
                
                # Phi-3.5チャットテンプレート形式のチェック
                has_system = "<|system|>" in output
                has_user = "<|user|>" in output
                has_assistant = "<|assistant|>" in output
                
                if has_system and has_user and has_assistant:
                    results["has_phi35_template"] += 1
                
                # 完全な形式（四重推論 + Phi-3.5テンプレート）
                if (has_task and has_safety and has_policy and has_final and 
                    has_system and has_user and has_assistant):
                    results["has_complete_format"] += 1
                    results["valid_format"] += 1
                else:
                    results["invalid_format"] += 1
                    missing = []
                    if not has_task:
                        missing.append("think-task")
                    if not has_safety:
                        missing.append("think-safety")
                    if not has_policy:
                        missing.append("think-policy")
                    if not has_final:
                        missing.append("final")
                    if not has_system:
                        missing.append("<|system|>")
                    if not has_user:
                        missing.append("<|user|>")
                    if not has_assistant:
                        missing.append("<|assistant|>")
                    if line_num <= 10:  # 最初の10件のエラーのみ記録
                        results["errors"].append({
                            "line": line_num,
                            "missing": missing
                        })
                
                # サンプルプレビュー（最初の1件）
                if results["sample_preview"] is None and results["total_samples"] == 1:
                    results["sample_preview"] = {
                        "keys": list(sample.keys()),
                        "output_preview": output[:500] if output else "",
                        "output_length": len(output)
                    }
                    
            except json.JSONDecodeError as e:
                results["errors"].append({
                    "line": line_num,
                    "error": f"JSON decode error: {e}"
                })
                results["invalid_format"] += 1
            except Exception as e:
                results["errors"].append({
                    "line": line_num,
                    "error": f"Unexpected error: {e}"
                })
                results["invalid_format"] += 1
    
    # 統計情報
    if results["output_lengths"]:
        results["avg_output_length"] = sum(results["output_lengths"]) / len(results["output_lengths"])
        results["min_output_length"] = min(results["output_lengths"])
        results["max_output_length"] = max(results["output_lengths"])
    
    return results


def print_quality_report(results: Dict[str, Any]):
    """品質レポートを表示"""
    print("\n" + "="*80)
    print("DATASET QUALITY REPORT")
    print("="*80)
    
    print(f"\n[OVERVIEW]")
    print(f"  Total samples: {results['total_samples']:,}")
    print(f"  Valid format: {results['valid_format']:,} ({results['valid_format']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Invalid format: {results['invalid_format']:,} ({results['invalid_format']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[THINKING SECTIONS]")
    print(f"  Has <think-task>: {results['has_think_task']:,} ({results['has_think_task']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <think-safety>: {results['has_think_safety']:,} ({results['has_think_safety']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <think-policy>: {results['has_think_policy']:,} ({results['has_think_policy']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <final>: {results['has_final']:,} ({results['has_final']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has all thinking sections: {results['has_all_thinking_sections']:,} ({results['has_all_thinking_sections']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[PHI-3.5 TEMPLATE]")
    print(f"  Has Phi-3.5 template: {results['has_phi35_template']:,} ({results['has_phi35_template']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[COMPLETE FORMAT]")
    print(f"  Has complete format (thinking + template): {results['has_complete_format']:,} ({results['has_complete_format']/max(results['total_samples'], 1)*100:.1f}%)")
    
    if results.get("output_lengths"):
        print(f"\n[OUTPUT LENGTH]")
        print(f"  Average: {results['avg_output_length']:.0f} characters")
        print(f"  Min: {results['min_output_length']:,} characters")
        print(f"  Max: {results['max_output_length']:,} characters")
    
    if results["errors"]:
        print(f"\n[ERRORS] (showing first 10)")
        for error in results["errors"][:10]:
            if "missing" in error:
                print(f"  Line {error['line']}: Missing sections: {', '.join(error['missing'])}")
            else:
                print(f"  Line {error['line']}: {error.get('error', 'Unknown error')}")
    
    if results["sample_preview"]:
        print(f"\n[SAMPLE PREVIEW]")
        print(f"  Keys: {', '.join(results['sample_preview']['keys'])}")
        print(f"  Output length: {results['sample_preview']['output_length']:,} characters")
        print(f"  Output preview (first 500 chars):")
        print(f"  {results['sample_preview']['output_preview']}")
    
    print("\n" + "="*80)
    
    # 品質評価
    quality_score = (results['has_complete_format'] / max(results['total_samples'], 1)) * 100
    print(f"\n[QUALITY SCORE]")
    if quality_score >= 95:
        print(f"  Score: {quality_score:.1f}% - EXCELLENT (適切)")
    elif quality_score >= 80:
        print(f"  Score: {quality_score:.1f}% - GOOD (概ね適切)")
    elif quality_score >= 60:
        print(f"  Score: {quality_score:.1f}% - FAIR (改善が必要)")
    else:
        print(f"  Score: {quality_score:.1f}% - POOR (不適切、再生成が必要)")
    
    print("="*80)


def main():
    dataset_path = Path(r"D:\webdataset\processed\thinking_quadruple\quadruple_thinking_20251114_102426.jsonl")
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    results = check_dataset_quality(dataset_path)
    print_quality_report(results)


if __name__ == "__main__":
    main()






# -*- coding: utf-8 -*-
"""
データセット品質チェックスクリプト

/thinkingモデル用のデータセットが適切な形式になっているか確認
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

def check_dataset_quality(dataset_path: Path) -> Dict[str, Any]:
    """
    データセットの品質をチェック
    
    Args:
        dataset_path: データセットファイルパス
    
    Returns:
        品質チェック結果
    """
    results = {
        "total_samples": 0,
        "valid_format": 0,
        "invalid_format": 0,
        "has_think_task": 0,
        "has_think_safety": 0,
        "has_think_policy": 0,
        "has_final": 0,
        "has_all_thinking_sections": 0,
        "has_phi35_template": 0,
        "has_complete_format": 0,
        "output_lengths": [],
        "errors": [],
        "sample_preview": None
    }
    
    print(f"[CHECK] Analyzing dataset: {dataset_path}")
    print("="*80)
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                samples.append(sample)
                results["total_samples"] += 1
                
                output = sample.get("output", "")
                results["output_lengths"].append(len(output))
                
                # 四重推論セクションのチェック
                has_task = "<think-task>" in output
                has_safety = "<think-safety>" in output
                has_policy = "<think-policy>" in output
                has_final = "<final>" in output
                
                if has_task:
                    results["has_think_task"] += 1
                if has_safety:
                    results["has_think_safety"] += 1
                if has_policy:
                    results["has_think_policy"] += 1
                if has_final:
                    results["has_final"] += 1
                
                # すべてのセクションがあるか
                if has_task and has_safety and has_policy and has_final:
                    results["has_all_thinking_sections"] += 1
                
                # Phi-3.5チャットテンプレート形式のチェック
                has_system = "<|system|>" in output
                has_user = "<|user|>" in output
                has_assistant = "<|assistant|>" in output
                
                if has_system and has_user and has_assistant:
                    results["has_phi35_template"] += 1
                
                # 完全な形式（四重推論 + Phi-3.5テンプレート）
                if (has_task and has_safety and has_policy and has_final and 
                    has_system and has_user and has_assistant):
                    results["has_complete_format"] += 1
                    results["valid_format"] += 1
                else:
                    results["invalid_format"] += 1
                    missing = []
                    if not has_task:
                        missing.append("think-task")
                    if not has_safety:
                        missing.append("think-safety")
                    if not has_policy:
                        missing.append("think-policy")
                    if not has_final:
                        missing.append("final")
                    if not has_system:
                        missing.append("<|system|>")
                    if not has_user:
                        missing.append("<|user|>")
                    if not has_assistant:
                        missing.append("<|assistant|>")
                    if line_num <= 10:  # 最初の10件のエラーのみ記録
                        results["errors"].append({
                            "line": line_num,
                            "missing": missing
                        })
                
                # サンプルプレビュー（最初の1件）
                if results["sample_preview"] is None and results["total_samples"] == 1:
                    results["sample_preview"] = {
                        "keys": list(sample.keys()),
                        "output_preview": output[:500] if output else "",
                        "output_length": len(output)
                    }
                    
            except json.JSONDecodeError as e:
                results["errors"].append({
                    "line": line_num,
                    "error": f"JSON decode error: {e}"
                })
                results["invalid_format"] += 1
            except Exception as e:
                results["errors"].append({
                    "line": line_num,
                    "error": f"Unexpected error: {e}"
                })
                results["invalid_format"] += 1
    
    # 統計情報
    if results["output_lengths"]:
        results["avg_output_length"] = sum(results["output_lengths"]) / len(results["output_lengths"])
        results["min_output_length"] = min(results["output_lengths"])
        results["max_output_length"] = max(results["output_lengths"])
    
    return results


def print_quality_report(results: Dict[str, Any]):
    """品質レポートを表示"""
    print("\n" + "="*80)
    print("DATASET QUALITY REPORT")
    print("="*80)
    
    print(f"\n[OVERVIEW]")
    print(f"  Total samples: {results['total_samples']:,}")
    print(f"  Valid format: {results['valid_format']:,} ({results['valid_format']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Invalid format: {results['invalid_format']:,} ({results['invalid_format']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[THINKING SECTIONS]")
    print(f"  Has <think-task>: {results['has_think_task']:,} ({results['has_think_task']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <think-safety>: {results['has_think_safety']:,} ({results['has_think_safety']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <think-policy>: {results['has_think_policy']:,} ({results['has_think_policy']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has <final>: {results['has_final']:,} ({results['has_final']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  Has all thinking sections: {results['has_all_thinking_sections']:,} ({results['has_all_thinking_sections']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[PHI-3.5 TEMPLATE]")
    print(f"  Has Phi-3.5 template: {results['has_phi35_template']:,} ({results['has_phi35_template']/max(results['total_samples'], 1)*100:.1f}%)")
    
    print(f"\n[COMPLETE FORMAT]")
    print(f"  Has complete format (thinking + template): {results['has_complete_format']:,} ({results['has_complete_format']/max(results['total_samples'], 1)*100:.1f}%)")
    
    if results.get("output_lengths"):
        print(f"\n[OUTPUT LENGTH]")
        print(f"  Average: {results['avg_output_length']:.0f} characters")
        print(f"  Min: {results['min_output_length']:,} characters")
        print(f"  Max: {results['max_output_length']:,} characters")
    
    if results["errors"]:
        print(f"\n[ERRORS] (showing first 10)")
        for error in results["errors"][:10]:
            if "missing" in error:
                print(f"  Line {error['line']}: Missing sections: {', '.join(error['missing'])}")
            else:
                print(f"  Line {error['line']}: {error.get('error', 'Unknown error')}")
    
    if results["sample_preview"]:
        print(f"\n[SAMPLE PREVIEW]")
        print(f"  Keys: {', '.join(results['sample_preview']['keys'])}")
        print(f"  Output length: {results['sample_preview']['output_length']:,} characters")
        print(f"  Output preview (first 500 chars):")
        print(f"  {results['sample_preview']['output_preview']}")
    
    print("\n" + "="*80)
    
    # 品質評価
    quality_score = (results['has_complete_format'] / max(results['total_samples'], 1)) * 100
    print(f"\n[QUALITY SCORE]")
    if quality_score >= 95:
        print(f"  Score: {quality_score:.1f}% - EXCELLENT (適切)")
    elif quality_score >= 80:
        print(f"  Score: {quality_score:.1f}% - GOOD (概ね適切)")
    elif quality_score >= 60:
        print(f"  Score: {quality_score:.1f}% - FAIR (改善が必要)")
    else:
        print(f"  Score: {quality_score:.1f}% - POOR (不適切、再生成が必要)")
    
    print("="*80)


def main():
    dataset_path = Path(r"D:\webdataset\processed\thinking_quadruple\quadruple_thinking_20251114_102426.jsonl")
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    results = check_dataset_quality(dataset_path)
    print_quality_report(results)


if __name__ == "__main__":
    main()






















