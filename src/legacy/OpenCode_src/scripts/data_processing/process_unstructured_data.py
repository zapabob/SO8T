#!/usr/bin/env python3
"""
非構造データ処理パイプライン
PDF、白書などの非構造データをダウンロード、構造化、クレンジング
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    import requests
    from bs4 import BeautifulSoup
    import PyPDF2
    import pdfplumber
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("[ERROR] Required packages not installed")
    print("[INFO] Install with: pip install requests beautifulsoup4 PyPDF2 pdfplumber")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnstructuredDataProcessor:
    """非構造データ処理クラス"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root
        
        self.raw_dir = self.project_root / "data" / "unstructured" / "raw"
        self.processed_dir = self.project_root / "data" / "unstructured" / "processed"
        self.cleaned_dir = self.project_root / "data" / "unstructured" / "cleaned"
        
        # ディレクトリ作成
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    def download_government_whitepapers(self, sources: List[Dict[str, Any]]) -> List[Path]:
        """官公庁の白書をダウンロード"""
        logger.info("[DOWNLOAD] Downloading government whitepapers...")
        
        downloaded_files = []
        
        for source in sources:
            try:
                source_url = source.get('source_url')
                domain = source.get('domain', 'unknown')
                source_id = source.get('id', 'unknown')
                
                logger.info(f"[DOWNLOAD] Downloading {domain} whitepaper from {source_url}")
                
                # PDFリンクを取得（実装は簡略化）
                # 実際には各サイトの構造に応じて実装が必要
                pdf_urls = self._extract_pdf_urls(source_url, domain)
                
                for pdf_url in pdf_urls[:5]:  # 最大5ファイル
                    try:
                        pdf_path = self._download_pdf(pdf_url, domain, source_id)
                        if pdf_path:
                            downloaded_files.append(pdf_path)
                    except Exception as e:
                        logger.warning(f"[WARN] Failed to download {pdf_url}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"[ERROR] Failed to process {source.get('id')}: {e}")
                continue
        
        logger.info(f"[DOWNLOAD] Downloaded {len(downloaded_files)} PDF files")
        return downloaded_files
    
    def _extract_pdf_urls(self, base_url: str, domain: str) -> List[str]:
        """PDF URLを抽出（簡略実装）"""
        try:
            response = requests.get(base_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            pdf_urls = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf'):
                    if href.startswith('http'):
                        pdf_urls.append(href)
                    else:
                        pdf_urls.append(f"{base_url.rstrip('/')}/{href.lstrip('/')}")
            
            return pdf_urls[:10]  # 最大10個
        except Exception as e:
            logger.warning(f"[WARN] Failed to extract PDF URLs from {base_url}: {e}")
            return []
    
    def _download_pdf(self, url: str, domain: str, source_id: str) -> Optional[Path]:
        """PDFをダウンロード"""
        try:
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # ファイル名を生成
            filename = url.split('/')[-1] or f"{source_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
            safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
            
            pdf_path = self.raw_dir / domain / safe_filename
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"[OK] Downloaded: {pdf_path.name}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to download {url}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """PDFからテキストを抽出"""
        try:
            # pdfplumberを使用（より正確）
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            logger.warning(f"[WARN] pdfplumber failed, trying PyPDF2: {e}")
            try:
                # PyPDF2フォールバック
                text_content = []
                with open(pdf_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                
                return '\n\n'.join(text_content)
            except Exception as e2:
                logger.error(f"[ERROR] Failed to extract text from {pdf_path}: {e2}")
                return ""
    
    def structure_text_data(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """テキストを構造化データに変換"""
        structured_data = []
        
        # 段落に分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # 短すぎる段落はスキップ
                continue
            
            structured_item = {
                'id': f"{metadata.get('source_id', 'unknown')}_{i}",
                'text': paragraph,
                'domain': metadata.get('domain', 'unknown'),
                'source': metadata.get('source', 'unknown'),
                'source_url': metadata.get('source_url', ''),
                'extracted_at': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            structured_data.append(structured_item)
        
        return structured_data
    
    def sanitize_and_clean(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """データをサニタイズ・クレンジング"""
        logger.info(f"[CLEAN] Sanitizing and cleaning {len(data)} items...")
        
        cleaned_data = []
        
        for item in data:
            # テキストクレンジング
            text = item.get('text', '')
            
            # 不要な文字を削除
            text = re.sub(r'\s+', ' ', text)  # 連続する空白を1つに
            text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF.,!?;:()\[\]{}"\'-]', '', text)  # 特殊文字を削除
            
            # 長さチェック
            if len(text) < 50 or len(text) > 10000:
                continue
            
            # 機密情報のマスキング（簡略実装）
            text = self._mask_sensitive_info(text)
            
            # クレンジング済みデータ
            cleaned_item = {
                **item,
                'text': text,
                'cleaned_at': datetime.now().isoformat(),
                'text_length': len(text),
                'word_count': len(text.split())
            }
            
            cleaned_data.append(cleaned_item)
        
        logger.info(f"[CLEAN] Cleaned {len(cleaned_data)} items (removed {len(data) - len(cleaned_data)})")
        return cleaned_data
    
    def _mask_sensitive_info(self, text: str) -> str:
        """機密情報をマスキング"""
        # メールアドレス
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # 電話番号（簡略）
        text = re.sub(r'\b\d{2,4}-\d{2,4}-\d{2,4}\b', '[PHONE]', text)
        
        # 個人情報（簡略）
        # より高度な実装が必要
        
        return text
    
    def save_structured_data(self, data: List[Dict[str, Any]], output_path: Path):
        """構造化データを保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(data)} items to {output_path}")
    
    def process_government_sources(self, sources: List[Dict[str, Any]]) -> Path:
        """官公庁データソースを処理"""
        logger.info("[PROCESS] Processing government sources...")
        
        # 1. ダウンロード
        pdf_files = self.download_government_whitepapers(sources)
        
        # 2. テキスト抽出と構造化
        all_structured_data = []
        
        for pdf_path in pdf_files:
            try:
                # テキスト抽出
                text = self.extract_text_from_pdf(pdf_path)
                
                if not text:
                    continue
                
                # メタデータ
                metadata = {
                    'source_id': pdf_path.stem,
                    'domain': pdf_path.parent.name,
                    'source': 'government_whitepaper',
                    'source_url': '',  # URLを記録する必要あり
                    'file_path': str(pdf_path)
                }
                
                # 構造化
                structured = self.structure_text_data(text, metadata)
                all_structured_data.extend(structured)
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to process {pdf_path}: {e}")
                continue
        
        # 3. クレンジング
        cleaned_data = self.sanitize_and_clean(all_structured_data)
        
        # 4. 保存
        output_path = self.cleaned_dir / f"government_whitepapers_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.save_structured_data(cleaned_data, output_path)
        
        return output_path


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process unstructured data (PDFs, whitepapers)')
    parser.add_argument('--sources', type=str,
                       help='JSON file with data sources')
    parser.add_argument('--domain', type=str,
                       help='Specific domain to process (defense, aerospace, semiconductor, infrastructure)')
    
    args = parser.parse_args()
    
    processor = UnstructuredDataProcessor()
    
    # データソースを読み込み
    if args.sources:
        with open(args.sources, 'r', encoding='utf-8') as f:
            sources = json.load(f)
    else:
        # デフォルトの官公庁データソース
        sources = [
            {
                'id': 'japan_defense_white_paper',
                'type': 'government_white_paper',
                'domain': '防衛',
                'source_url': 'https://www.mod.go.jp/j/publication/wp/',
                'format': 'pdf'
            },
            {
                'id': 'japan_aerospace_white_paper',
                'type': 'government_white_paper',
                'domain': '航空宇宙',
                'source_url': 'https://www.mext.go.jp/a_menu/kagaku/space/',
                'format': 'pdf'
            },
            {
                'id': 'japan_semiconductor_policy',
                'type': 'government_policy',
                'domain': '半導体',
                'source_url': 'https://www.meti.go.jp/policy/mono_info_service/mono/electronics/',
                'format': 'pdf'
            },
            {
                'id': 'japan_infrastructure_white_paper',
                'type': 'government_white_paper',
                'domain': 'インフラ',
                'source_url': 'https://www.mlit.go.jp/',
                'format': 'pdf'
            }
        ]
    
    # ドメインフィルタリング
    if args.domain:
        sources = [s for s in sources if s.get('domain') == args.domain]
    
    # 処理実行
    output_path = processor.process_government_sources(sources)
    
    print(f"\n[SUCCESS] Processed unstructured data")
    print(f"[OUTPUT] {output_path}")


if __name__ == "__main__":
    main()
