#!/usr/bin/env python3
"""
Citation Fetcher for Sunset Pipeline
Fetches top-cited papers from Arxiv/BioRxiv using Semantic Scholar API
"""

import json
import argparse
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import os
from tqdm import tqdm

def load_env_file(filepath: str = ".env"):
    """Simple .env loader"""
    path = Path(filepath)
    if not path.exists():
        return
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


class CitationFetcher:
    """Semantic Scholar APIを使用した引用上位論文取得クラス"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,authors,citationCount,publicationDate,externalIds,venue,year"
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        self.api_key = api_key
        self.verbose = verbose
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self.session.headers["User-Agent"] = "SunsetPipeline/1.0"
    
    def log(self, message: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [CITATION_FETCHER] {message}")
    
    def fetch_papers_by_source(
        self,
        source: str,
        query: Optional[str] = None,
        start_year: int = 2024,
        end_year: int = 2026,
        max_papers: int = 100000,
        output_file: Optional[str] = None,
        checkpoint_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        特定ソースから引用上位論文を取得
        
        Args:
            source: "arxiv" or "biorxiv"
            start_year: 開始年
            end_year: 終了年
            max_papers: 最大取得件数
            output_file: 出力JSOLファイル（逐次保存）
            checkpoint_file: チェックポイントファイル
        
        Returns:
            論文リスト
        """
        self.log(f"Fetching {source} papers from {start_year} to {end_year}")
        self.log(f"Target: {max_papers} papers")
        
        papers = []
        offset = 0
        batch_size = 100  # API制限に準拠
        last_checkpoint_time = time.time()
        checkpoint_interval = 300  # 5分間隔でチェックポイント
        
        # チェックポイントからの復旧（ローリングバックアップ対応）
        checkpoint = None
        if checkpoint_file:
            checkpoint = self._load_best_checkpoint(checkpoint_file)
            if checkpoint:
                offset = checkpoint.get("offset", 0)
                papers = checkpoint.get("papers", [])
                self.log(f"Resuming from checkpoint: offset={offset}, papers={len(papers)}")
        
        # 出力ファイルの準備（追記モード）
        output_path = Path(output_file) if output_file else None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pbar = None
        try:
            while len(papers) < max_papers:
                batch = self._fetch_batch(
                    source=source,
                    query=query,
                    start_year=start_year,
                    end_year=end_year,
                    offset=offset,
                    limit=batch_size
                )
                
                if not batch:
                    self.log("No more papers available")
                    break
                
                papers.extend(batch)
                offset += len(batch)
                
                # tqdmの更新または初期化
                if pbar is None:
                    pbar = tqdm(total=max_papers, desc=f"Fetching {source}", unit="paper")
                pbar.update(len(batch))
                
                # 逐次保存
                if output_path:
                    self._append_to_jsonl(output_path, batch)
                
                # チェックポイント保存（5分間隔、ローリング3バックアップ）
                current_time = time.time()
                if checkpoint_file and (current_time - last_checkpoint_time) >= checkpoint_interval:
                    self._save_checkpoint_rolling(checkpoint_file, {
                        "source": source,
                        "offset": offset,
                        "papers_count": len(papers),
                        "output_file": str(output_file) if output_file else "",
                        "start_year": start_year,
                        "end_year": end_year,
                        "max_papers": max_papers,
                        "timestamp": datetime.now().isoformat()
                    })
                    last_checkpoint_time = current_time
                
                # Rate limit対応
                # API Keyあり: 1 request / sec (limit 1/sec) -> 1.1s wait
                # API Keyなし: 100 requests / 5 min (limit 1/3sec) -> 3.1s wait
                wait_time = 1.1 if self.api_key else 3.1
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            self.log("Interrupted by user, saving checkpoint...")
            if checkpoint_file:
                self._save_checkpoint_rolling(checkpoint_file, {
                    "source": source,
                    "offset": offset,
                    "papers_count": len(papers),
                    "output_file": str(output_file) if output_file else "",
                    "start_year": start_year,
                    "end_year": end_year,
                    "max_papers": max_papers,
                    "timestamp": datetime.now().isoformat()
                })
        
        if pbar:
            pbar.close()
            
        self.log(f"Completed: {len(papers)} papers fetched")
        return papers[:max_papers]
    
    def _fetch_batch(
        self,
        source: str,
        query: Optional[str],
        start_year: int,
        end_year: int,
        offset: int,
        limit: int,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """バッチで論文を取得（リトライ対応）"""
        # Semantic Scholarのbulk search APIを使用
        endpoint = f"{self.BASE_URL}/paper/search/bulk"
        
        # ソースに応じたフィルタリング戦略
        if not query:
            if source.lower() == "arxiv":
                query = "machine learning OR artificial intelligence OR deep learning OR neural network OR computer vision OR natural language processing OR reinforcement learning OR transformer"
            elif source.lower() == "biorxiv":
                query = "biology OR genomics OR molecular biology OR neuroscience OR bioinformatics"
            else:
                query = "science"
        
        params = {
            "query": query,
            "fields": self.FIELDS,
            "sort": "citationCount:desc",
            "year": f"{start_year}-{end_year}",
            "offset": offset,
            "limit": limit
        }
        
        for retry in range(max_retries):
            try:
                # タイムアウトを長めに設定
                response = self.session.get(endpoint, params=params, timeout=60)
                
                if response.status_code == 429:
                    # Rate limit - wait and retry
                    wait_time = 60 * (retry + 1)
                    self.log(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 400:
                    self.log(f"Bad request: {response.text[:200]}")
                    return []
                
                response.raise_for_status()
                data = response.json()
                
                papers = []
                for paper in data.get("data", []):
                    processed = self._process_paper(paper, source)
                    if processed:
                        papers.append(processed)
                
                return papers
                
            except requests.exceptions.Timeout:
                wait_time = 10 * (retry + 1)
                self.log(f"Timeout on attempt {retry + 1}/{max_retries}, waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except requests.exceptions.RequestException as e:
                self.log(f"API request failed (attempt {retry + 1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    time.sleep(5 * (retry + 1))
                    continue
                return []
        
        self.log(f"All {max_retries} retries failed at offset {offset}")
        return []
    
    def _process_paper(self, paper: Dict, source: str) -> Optional[Dict[str, Any]]:
        """論文データを構造化"""
        external_ids = paper.get("externalIds", {}) or {}
        
        # ソースIDの取得（フィルタリングせず、あれば使用）
        if source.lower() == "arxiv":
            source_id = external_ids.get("ArXiv", "") or external_ids.get("DOI", "") or paper.get("paperId", "")
        elif source.lower() == "biorxiv":
            source_id = external_ids.get("DOI", "") or paper.get("paperId", "")
        else:
            source_id = paper.get("paperId", "")
        
        authors = paper.get("authors", []) or []
        author_names = [a.get("name", "") for a in authors if a.get("name")]
        
        return {
            "source": source.lower(),
            "paper_id": paper.get("paperId", ""),
            "source_id": source_id or paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", "") or "",
            "authors": author_names,
            "citation_count": paper.get("citationCount", 0) or 0,
            "publication_date": paper.get("publicationDate", ""),
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
            "external_ids": external_ids
        }
    
    def _append_to_jsonl(self, path: Path, papers: List[Dict]):
        """JSOLファイルに追記"""
        with open(path, 'a', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
    
    def _save_checkpoint_rolling(self, path: str, data: Dict, max_backups: int = 3):
        """ローリングバックアップ付きチェックポイント保存"""
        path_obj = Path(path)
        base_name = path_obj.stem
        parent_dir = path_obj.parent
        
        # ローリングバックアップ（古いものを削除、新しいものを保持）
        # 逆順に処理: 3->削除, 2->3, 1->2
        for i in range(max_backups, 0, -1):
            old_backup = parent_dir / f"{base_name}.{i}.json"
            new_backup = parent_dir / f"{base_name}.{i+1}.json"
            
            if old_backup.exists():
                if i >= max_backups:
                    old_backup.unlink()  # 最も古いバックアップを削除
                else:
                    # Windows対応: renameの代わりにreplaceを使用（上書き許可）
                    old_backup.replace(new_backup)
        
        # 現在のチェックポイントをバックアップ (current -> 1)
        if path_obj.exists():
            backup_1 = parent_dir / f"{base_name}.1.json"
            path_obj.replace(backup_1)
        
        # 新しいチェックポイントを保存
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log(f"Checkpoint saved: {path} (rolling {max_backups} backups)")
    
    def _load_best_checkpoint(self, path: str) -> Optional[Dict]:
        """最新の有効なチェックポイントを読み込み（バックアップも試行）"""
        path_obj = Path(path)
        
        # メインのチェックポイントを試行
        checkpoint = self._load_checkpoint(str(path_obj))
        if checkpoint:
            return checkpoint
        
        # バックアップを順番に試行
        for i in range(1, 4):
            backup_path = path_obj.parent / f"{path_obj.stem}.{i}.json"
            checkpoint = self._load_checkpoint(str(backup_path))
            if checkpoint:
                self.log(f"Recovered from backup: {backup_path}")
                return checkpoint
        
        return None
    
    def _load_checkpoint(self, path: str) -> Optional[Dict]:
        """チェックポイントを読み込み"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def to_training_format(self, paper: Dict) -> Dict[str, Any]:
        """論文データを学習用JSONL形式に変換"""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        authors = ", ".join(paper.get("authors", [])[:5])  # 最初の5著者
        
        if not abstract:
            return None
        
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"論文「{title}」（著者: {authors}）の概要を説明してください。"
                },
                {
                    "role": "assistant",
                    "content": f"論文「{title}」の概要を説明します。\n\n{abstract}"
                }
            ],
            "metadata": {
                "source": paper.get("source"),
                "paper_id": paper.get("paper_id"),
                "citation_count": paper.get("citation_count"),
                "year": paper.get("year")
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Citation Fetcher for Sunset Pipeline')
    parser.add_argument('--source', '-s', choices=['arxiv', 'biorxiv'],
                       required=True, help='Paper source')
    parser.add_argument('--max-papers', '-n', type=int, default=100000,
                       help='Maximum number of papers to fetch')
    parser.add_argument('--start-year', type=int, default=2024,
                       help='Start year (default: 2024)')
    parser.add_argument('--end-year', type=int, default=2026,
                       help='End year (default: 2026)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSONL file path')
    parser.add_argument('--checkpoint', '-c', help='Checkpoint file path')
    parser.add_argument('--api-key', help='Semantic Scholar API key')
    parser.add_argument('--query', '-q', help='Search query')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    # .env ファイルの読み込み
    load_env_file()
    
    args = parser.parse_args()
    
    # API Keyの優先順位: CLI引数 > 環境変数
    api_key = args.api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    
    fetcher = CitationFetcher(api_key=api_key, verbose=args.verbose)
    
    # チェックポイントファイルのデフォルト設定
    checkpoint_file = args.checkpoint
    if not checkpoint_file:
        output_path = Path(args.output)
        checkpoint_file = str(output_path.parent / f"{output_path.stem}_checkpoint.json")
    
    try:
        papers = fetcher.fetch_papers_by_source(
            source=args.source,
            query=args.query,
            start_year=args.start_year,
            end_year=args.end_year,
            max_papers=args.max_papers,
            output_file=args.output,
            checkpoint_file=checkpoint_file
        )
        
        print(f"[SUCCESS] Fetched {len(papers)} papers")
        print(f"[INFO] Output saved to: {args.output}")
        
        # サンプル表示
        if papers:
            print("\n[SAMPLE] Top 3 cited papers:")
            for i, paper in enumerate(papers[:3], 1):
                print(f"  {i}. {paper['title'][:60]}... (citations: {paper['citation_count']})")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch papers: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
