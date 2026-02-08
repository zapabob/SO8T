#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途別データセット生成モジュール

ファインチューニング用、RAG用、評価用のデータセットを生成します。

Usage:
    from so8t_mmllm.scripts.data.purpose_specific_dataset_generator import PurposeSpecificDatasetGenerator
    generator = PurposeSpecificDatasetGenerator()
    generator.generate_finetuning_dataset(input_file, output_file)
    generator.generate_rag_dataset(input_file, output_dir)
    generator.generate_evaluation_dataset(input_file, output_file)
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib

from tqdm import tqdm


class PurposeSpecificDatasetGenerator:
    """用途別データセット生成クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 設定辞書
        """
        self.config = config or {
            # ファインチューニング用設定
            "finetuning": {
                "instruction_template": "以下の文章を読んで、内容を要約してください。",
                "response_template": "{text}",
                "max_length": 2048,
                "min_length": 100
            },
            # RAG用設定
            "rag": {
                "chunk_size": 512,
                "chunk_overlap": 128,
                "min_chunk_size": 100
            },
            # 評価用設定
            "evaluation": {
                "task_types": ["understanding", "generation", "reasoning"],
                "samples_per_task": 100
            }
        }
    
    def generate_finetuning_dataset(
        self,
        input_file: Path,
        output_file: Path,
        format_type: str = "instruction"
    ) -> int:
        """
        ファインチューニング用データセット生成
        
        Args:
            input_file: 入力データファイル（JSONL）
            output_file: 出力データファイル（JSONL）
            format_type: フォーマットタイプ（"instruction", "conversation"）
        
        Returns:
            generated_count: 生成されたサンプル数
        """
        print(f"[FINETUNING] Generating finetuning dataset from {input_file}...")
        
        finetuning_samples = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing samples"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    text = sample.get("text", sample.get("content", ""))
                    
                    if not text or len(text) < self.config["finetuning"]["min_length"]:
                        continue
                    
                    # 長さ制限
                    if len(text) > self.config["finetuning"]["max_length"]:
                        text = text[:self.config["finetuning"]["max_length"]]
                    
                    if format_type == "instruction":
                        # インストラクション形式
                        finetuning_sample = {
                            "instruction": self.config["finetuning"]["instruction_template"],
                            "input": text[:len(text)//2],  # 前半を入力
                            "output": text[len(text)//2:],  # 後半を出力
                            "domain": sample.get("domain", "unknown"),
                            "language": sample.get("language", "ja")
                        }
                    elif format_type == "conversation":
                        # 対話形式
                        finetuning_sample = {
                            "conversations": [
                                {
                                    "from": "user",
                                    "value": self.config["finetuning"]["instruction_template"]
                                },
                                {
                                    "from": "assistant",
                                    "value": text
                                }
                            ],
                            "domain": sample.get("domain", "unknown"),
                            "language": sample.get("language", "ja")
                        }
                    else:
                        continue
                    
                    finetuning_samples.append(finetuning_sample)
                
                except json.JSONDecodeError:
                    continue
        
        # 保存
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in finetuning_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"[OK] Generated {len(finetuning_samples):,} finetuning samples")
        return len(finetuning_samples)
    
    def generate_rag_dataset(
        self,
        input_file: Path,
        output_dir: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> int:
        """
        RAG用データセット生成（チャンク分割 + メタデータ付与）
        
        Args:
            input_file: 入力データファイル（JSONL）
            output_dir: 出力ディレクトリ
            chunk_size: チャンクサイズ（文字数）
            chunk_overlap: チャンクオーバーラップ（文字数）
        
        Returns:
            generated_count: 生成されたチャンク数
        """
        print(f"[RAG] Generating RAG dataset from {input_file}...")
        
        chunk_size = chunk_size or self.config["rag"]["chunk_size"]
        chunk_overlap = chunk_overlap or self.config["rag"]["chunk_overlap"]
        min_chunk_size = self.config["rag"]["min_chunk_size"]
        
        rag_chunks = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Chunking documents"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    text = sample.get("text", sample.get("content", ""))
                    
                    if not text:
                        continue
                    
                    # チャンク分割
                    for i in range(0, len(text), chunk_size - chunk_overlap):
                        chunk_text = text[i:i + chunk_size]
                        
                        if len(chunk_text) < min_chunk_size:
                            continue
                        
                        # チャンクID生成
                        chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                        
                        # RAG用チャンク作成
                        rag_chunk = {
                            "chunk_id": chunk_id,
                            "chunk_text": chunk_text,
                            "source_url": sample.get("url", ""),
                            "source_domain": sample.get("domain", "unknown"),
                            "source_language": sample.get("language", "ja"),
                            "chunk_index": i // (chunk_size - chunk_overlap),
                            "chunk_start": i,
                            "chunk_end": i + len(chunk_text),
                            "source_length": len(text),
                            "metadata": {
                                "crawled_at": sample.get("crawled_at", ""),
                                "quality_score": sample.get("quality_score", 0.0),
                                "relevance_score": sample.get("relevance_score", 0.0)
                            }
                        }
                        
                        rag_chunks.append(rag_chunk)
                
                except json.JSONDecodeError:
                    continue
        
        # 保存（言語・ドメイン別）
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 言語×ドメインで分類
        categorized = defaultdict(list)
        for chunk in rag_chunks:
            key = f"{chunk['source_language']}_{chunk['source_domain']}"
            categorized[key].append(chunk)
        
        total_chunks = 0
        for key, chunks in categorized.items():
            output_file = output_dir / f"rag_chunks_{key}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            total_chunks += len(chunks)
            print(f"[OK] Saved {len(chunks):,} chunks to {output_file.name}")
        
        # メタデータインデックス作成
        metadata = {
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "by_language": {},
            "by_domain": {},
            "created_at": datetime.now().isoformat()
        }
        
        for chunk in rag_chunks:
            lang = chunk['source_language']
            domain = chunk['source_domain']
            metadata['by_language'][lang] = metadata['by_language'].get(lang, 0) + 1
            metadata['by_domain'][domain] = metadata['by_domain'].get(domain, 0) + 1
        
        metadata_file = output_dir / "rag_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Generated {total_chunks:,} RAG chunks")
        print(f"[OK] Metadata saved to {metadata_file}")
        
        return total_chunks
    
    def generate_evaluation_dataset(
        self,
        input_file: Path,
        output_file: Path,
        task_types: Optional[List[str]] = None
    ) -> int:
        """
        評価用データセット生成
        
        Args:
            input_file: 入力データファイル（JSONL）
            output_file: 出力データファイル（JSONL）
            task_types: タスクタイプリスト（["understanding", "generation", "reasoning"]）
        
        Returns:
            generated_count: 生成されたサンプル数
        """
        print(f"[EVALUATION] Generating evaluation dataset from {input_file}...")
        
        task_types = task_types or self.config["evaluation"]["task_types"]
        samples_per_task = self.config["evaluation"]["samples_per_task"]
        
        # 入力データ読み込み
        input_samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    input_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        evaluation_samples = []
        
        # タスクタイプ別にサンプル生成
        for task_type in task_types:
            print(f"[EVALUATION] Generating {task_type} tasks...")
            
            task_samples = []
            for sample in input_samples[:samples_per_task * 3]:  # 余裕を持って読み込み
                text = sample.get("text", sample.get("content", ""))
                if not text or len(text) < 100:
                    continue
                
                if task_type == "understanding":
                    # 理解タスク: 要約、質問応答など
                    eval_sample = {
                        "task_type": "understanding",
                        "task": "要約",
                        "input": text,
                        "expected_output": text[:len(text)//3],  # 簡易的な期待出力
                        "domain": sample.get("domain", "unknown"),
                        "language": sample.get("language", "ja")
                    }
                
                elif task_type == "generation":
                    # 生成タスク: 続き生成、文章生成など
                    eval_sample = {
                        "task_type": "generation",
                        "task": "続き生成",
                        "input": text[:len(text)//2],
                        "expected_output": text[len(text)//2:],
                        "domain": sample.get("domain", "unknown"),
                        "language": sample.get("language", "ja")
                    }
                
                elif task_type == "reasoning":
                    # 推論タスク: 論理的推論、因果関係など
                    # 簡易的な推論タスク（実際にはより複雑な推論が必要）
                    eval_sample = {
                        "task_type": "reasoning",
                        "task": "論理的推論",
                        "input": text,
                        "expected_output": "推論結果",  # 実際には適切な推論結果を生成
                        "domain": sample.get("domain", "unknown"),
                        "language": sample.get("language", "ja")
                    }
                
                else:
                    continue
                
                task_samples.append(eval_sample)
                
                if len(task_samples) >= samples_per_task:
                    break
            
            evaluation_samples.extend(task_samples)
            print(f"[OK] Generated {len(task_samples):,} {task_type} samples")
        
        # 保存
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in evaluation_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"[OK] Generated {len(evaluation_samples):,} evaluation samples")
        return len(evaluation_samples)


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Purpose-specific dataset generation")
    parser.add_argument("--purpose", type=str, required=True,
                        choices=["finetuning", "rag", "evaluation"],
                        help="Dataset purpose")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input data file (JSONL)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output file or directory")
    parser.add_argument("--format", type=str, default="instruction",
                        choices=["instruction", "conversation"],
                        help="Format type (for finetuning)")
    args = parser.parse_args()
    
    generator = PurposeSpecificDatasetGenerator()
    
    try:
        if args.purpose == "finetuning":
            count = generator.generate_finetuning_dataset(
                args.input,
                args.output,
                format_type=args.format
            )
        elif args.purpose == "rag":
            count = generator.generate_rag_dataset(
                args.input,
                args.output
            )
        elif args.purpose == "evaluation":
            count = generator.generate_evaluation_dataset(
                args.input,
                args.output
            )
        
        print(f"\n[SUCCESS] Generated {count:,} samples")
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()







