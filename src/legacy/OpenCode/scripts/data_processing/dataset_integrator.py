#!/usr/bin/env python3
"""
Dataset Integration Script for Sunset Pipeline
Combines PDF extractions and citation data into unified training dataset
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Generator
from datetime import datetime
import sys
from scripts.utils.progress import progress

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DatasetIntegrator:
    """データセット統合クラス"""
    
    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.sunset_data_dir = project_root / "data" / "sunset_pipeline"
    
    def log(self, message: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [INTEGRATOR] {message}")
    
    def integrate_all(self, output_file: Path) -> Dict[str, int]:
        """全データソースを統合"""
        stats = {
            "pdf_documents": 0,
            "arxiv_papers": 0,
            "biorxiv_papers": 0,
            "total": 0
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # PDF documents
            self.log("Processing PDF documents...")
            for item in self._process_pdfs():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                stats["pdf_documents"] += 1
                stats["total"] += 1
            
            # Arxiv papers
            self.log("Processing Arxiv papers...")
            for item in self._process_citations("arxiv"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                stats["arxiv_papers"] += 1
                stats["total"] += 1
            
            # BioRxiv papers
            self.log("Processing BioRxiv papers...")
            for item in self._process_citations("biorxiv"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                stats["biorxiv_papers"] += 1
                stats["total"] += 1
        
        self.log(f"Integration complete: {stats['total']} items")
        return stats
    
    def _process_pdfs(self) -> Generator[Dict, None, None]:
        """PDF抽出データを処理"""
        pdf_dir = self.sunset_data_dir / "raw" / "pdfs"
        if not pdf_dir.exists():
            self.log(f"PDF directory not found: {pdf_dir}")
            return
        
        for json_file in pdf_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 学習用フォーマットに変換
                training_item = self._pdf_to_training_format(data)
                if training_item:
                    yield training_item
                    
            except Exception as e:
                self.log(f"Error processing {json_file}: {e}")
    
    def _process_citations(self, source: str) -> Generator[Dict, None, None]:
        """引用データを処理"""
        if source == "arxiv":
            data_dir = self.sunset_data_dir / "raw" / "arxiv_citations"
        else:
            data_dir = self.sunset_data_dir / "raw" / "biorxiv_citations"
        
        if not data_dir.exists():
            self.log(f"Citation directory not found: {data_dir}")
            return
        
        for jsonl_file in data_dir.glob("*.jsonl"):
            if "_test" in jsonl_file.name or "_checkpoint" in jsonl_file.name:
                continue  # テストファイルとチェックポイントをスキップ
            
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            paper = json.loads(line.strip())
                            training_item = self._paper_to_training_format(paper)
                            if training_item:
                                yield training_item
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                self.log(f"Error processing {jsonl_file}: {e}")
    
    def _pdf_to_training_format(self, pdf_data: Dict) -> Dict[str, Any]:
        """PDFデータを学習用フォーマットに変換"""
        title = pdf_data.get("structured_data", {}).get("title", pdf_data.get("filename", ""))
        full_text = pdf_data.get("full_text", "")
        
        if not full_text:
            return None
        
        # テキストを適切な長さに切り詰め
        content = full_text[:4000]
        
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"以下の文書「{title}」の内容について説明してください。"
                },
                {
                    "role": "assistant",
                    "content": f"文書「{title}」の内容を説明します。\n\n{content}"
                }
            ],
            "metadata": {
                "source": "pdf",
                "filename": pdf_data.get("filename"),
                "page_count": pdf_data.get("page_count"),
                "data_type": "document_understanding"
            }
        }
    
    def _paper_to_training_format(self, paper: Dict) -> Dict[str, Any]:
        """論文データを学習用フォーマットに変換"""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        authors = paper.get("authors", [])
        
        if not title or not abstract:
            return None
        
        author_str = ", ".join(authors[:5]) if authors else "著者不明"
        
        return {
            "messages": [
                {
                    "role": "user", 
                    "content": f"論文「{title}」（著者: {author_str}）の研究内容を説明してください。"
                },
                {
                    "role": "assistant",
                    "content": f"論文「{title}」の研究内容を説明します。\n\n{abstract}"
                }
            ],
            "metadata": {
                "source": paper.get("source"),
                "paper_id": paper.get("paper_id"),
                "citation_count": paper.get("citation_count"),
                "year": paper.get("year"),
                "data_type": "scientific_reasoning"
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Dataset Integrator for Sunset Pipeline')
    parser.add_argument('--output', '-o', default='data/sunset_pipeline/processed/combined_training_dataset.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / args.output
    
    integrator = DatasetIntegrator(project_root, verbose=args.verbose)
    
    try:
        stats = integrator.integrate_all(output_file)
        
        print(f"[SUCCESS] Dataset integration complete")
        print(f"[STATS] PDF documents: {stats['pdf_documents']}")
        print(f"[STATS] Arxiv papers: {stats['arxiv_papers']}")
        print(f"[STATS] BioRxiv papers: {stats['biorxiv_papers']}")
        print(f"[STATS] Total items: {stats['total']}")
        print(f"[INFO] Output saved to: {output_file}")
        
    except Exception as e:
        print(f"[ERROR] Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
