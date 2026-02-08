#!/usr/bin/env python3
"""
PDF Extractor for Sunset Pipeline
Extracts structured text and metadata from PDF files using PyMuPDF (fitz)
"""

import fitz  # PyMuPDF
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class PDFExtractor:
    """PDF構造化抽出クラス"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def log(self, message: str):
        if self.verbose:
            print(f"[PDF_EXTRACTOR] {message}")
    
    def extract_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        PDFファイルからテキストと構造を抽出
        
        Args:
            pdf_path: PDFファイルのパス
            
        Returns:
            構造化されたPDFデータ
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.log(f"Opening PDF: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        
        result = {
            "source": "pdf",
            "filename": pdf_path.name,
            "filepath": str(pdf_path.absolute()),
            "extraction_timestamp": datetime.now().isoformat(),
            "metadata": self._extract_metadata(doc),
            "page_count": len(doc),
            "pages": [],
            "full_text": "",
            "sections": [],
            "structured_data": {}
        }
        
        all_text = []
        all_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_data = self._extract_page(page, page_num + 1)
            result["pages"].append(page_data)
            all_text.append(page_data["text"])
            all_blocks.extend(page_data.get("blocks", []))
        
        result["full_text"] = "\n\n".join(all_text)
        result["sections"] = self._identify_sections(all_blocks)
        result["structured_data"] = self._create_structured_data(result)
        
        doc.close()
        
        self.log(f"Extracted {result['page_count']} pages, {len(result['full_text'])} characters")
        
        return result
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """PDFメタデータを抽出"""
        metadata = doc.metadata
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "format": metadata.get("format", ""),
        }
    
    def _extract_page(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """ページからテキストとブロック構造を抽出"""
        # テキスト抽出（構造保持）
        text = page.get_text("text")
        
        # ブロック構造の取得
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        processed_blocks = []
        for block in blocks:
            if block.get("type") == 0:  # テキストブロック
                block_text = ""
                font_sizes = []
                is_bold = False
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                        font_sizes.append(span.get("size", 0))
                        if "bold" in span.get("font", "").lower():
                            is_bold = True
                    block_text += "\n"
                
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                
                processed_blocks.append({
                    "type": "text",
                    "text": block_text.strip(),
                    "bbox": block.get("bbox", []),
                    "avg_font_size": avg_font_size,
                    "is_bold": is_bold,
                    "is_heading": avg_font_size > 12 or is_bold  # 簡易見出し判定
                })
            elif block.get("type") == 1:  # 画像ブロック
                processed_blocks.append({
                    "type": "image",
                    "bbox": block.get("bbox", []),
                    "width": block.get("width", 0),
                    "height": block.get("height", 0)
                })
        
        return {
            "page_number": page_num,
            "text": text,
            "blocks": processed_blocks,
            "width": page.rect.width,
            "height": page.rect.height
        }
    
    def _identify_sections(self, blocks: List[Dict]) -> List[Dict[str, Any]]:
        """ブロックからセクション構造を識別"""
        sections = []
        current_section = None
        
        for block in blocks:
            if block.get("type") != "text":
                continue
            
            text = block.get("text", "").strip()
            if not text:
                continue
            
            # 見出しの判定
            if block.get("is_heading") and len(text) < 200:
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "title": text,
                    "content": [],
                    "level": self._determine_heading_level(block)
                }
            else:
                if current_section:
                    current_section["content"].append(text)
                else:
                    # 最初のセクションがない場合
                    current_section = {
                        "title": "",
                        "content": [text],
                        "level": 0
                    }
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _determine_heading_level(self, block: Dict) -> int:
        """見出しレベルを推定"""
        font_size = block.get("avg_font_size", 12)
        if font_size >= 18:
            return 1
        elif font_size >= 14:
            return 2
        elif font_size >= 12 and block.get("is_bold"):
            return 3
        return 4
    
    def _create_structured_data(self, result: Dict) -> Dict[str, Any]:
        """学習用の構造化データを生成"""
        # タイトルの抽出
        title = result["metadata"].get("title", "")
        if not title and result["sections"]:
            title = result["sections"][0].get("title", result["filename"])
        
        # コンテンツの要約生成用テキスト
        content_parts = []
        for section in result["sections"]:
            if section["title"]:
                content_parts.append(f"## {section['title']}")
            content_parts.extend(section["content"])
        
        return {
            "title": title or result["filename"],
            "summary_text": "\n\n".join(content_parts[:5]),  # 最初の5セクション
            "section_count": len(result["sections"]),
            "word_count": len(result["full_text"].split()),
            "char_count": len(result["full_text"])
        }
    
    def to_training_format(self, extracted_data: Dict) -> Dict[str, Any]:
        """
        抽出データを学習用JSONL形式に変換
        DeepSeek-GLPO互換フォーマット
        """
        title = extracted_data["structured_data"]["title"]
        content = extracted_data["full_text"][:8000]  # トークン制限対応
        
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"以下の文書「{title}」の内容を要約してください。"
                },
                {
                    "role": "assistant", 
                    "content": f"文書「{title}」の内容を要約します。\n\n{content[:2000]}..."
                }
            ],
            "metadata": {
                "source": "pdf",
                "filename": extracted_data["filename"],
                "page_count": extracted_data["page_count"],
                "extraction_timestamp": extracted_data["extraction_timestamp"]
            }
        }


def main():
    parser = argparse.ArgumentParser(description='PDF Extractor for Sunset Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input PDF file path')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--training-format', action='store_true', help='Output in training JSONL format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    extractor = PDFExtractor(verbose=args.verbose)
    
    try:
        result = extractor.extract_pdf(args.input)
        
        if args.training_format:
            output_data = extractor.to_training_format(result)
        else:
            output_data = result
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"[SUCCESS] Output saved to: {output_path}")
        else:
            print(json.dumps(output_data, ensure_ascii=False, indent=2))
        
        print(f"[INFO] Extracted {result['page_count']} pages, {result['structured_data']['word_count']} words")
        
    except Exception as e:
        print(f"[ERROR] Failed to extract PDF: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
