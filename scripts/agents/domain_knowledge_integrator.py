#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ドメイン別知識統合モジュール

コーディング知識、科学知識、日本語・英語の知識を統合し、
ドメイン別知識の優先順位付けと統合を実装。

Usage:
    python scripts/agents/domain_knowledge_integrator.py --query "Pythonのリスト操作" --domain coding
"""

import sys
import json
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/domain_knowledge_integrator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 知識ベースインポート
try:
    sys.path.insert(0, str(PROJECT_ROOT / "utils"))
    from memory_manager import SO8TMemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False
    logger.warning("SO8TMemoryManager not available")

# RAG/CoGインポート
try:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipelines"))
    from vector_store_creation import RAGVectorStoreCreator
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    logger.warning("RAGVectorStoreCreator not available")


class DomainKnowledgeIntegrator:
    """ドメイン別知識統合クラス"""
    
    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        rag_store_path: Optional[str] = None,
        coding_data_path: Optional[str] = None,
        science_data_path: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            knowledge_base_path: 知識ベースのパス
            rag_store_path: RAGストアのパス
            coding_data_path: コーディングデータのパス
            science_data_path: 科学データのパス
        """
        self.knowledge_base_path = knowledge_base_path
        self.rag_store_path = rag_store_path
        self.coding_data_path = coding_data_path
        self.science_data_path = science_data_path
        
        # メモリマネージャーの初期化
        self.memory_manager = None
        if MEMORY_MANAGER_AVAILABLE:
            self._initialize_memory_manager()
        
        # RAGストアの初期化
        self.rag_store = None
        if VECTOR_STORE_AVAILABLE and rag_store_path:
            self._initialize_rag_store()
        
        # ドメイン別の重み付け
        self.domain_weights = {
            'coding': 1.5,  # コーディング知識は優先度高め
            'science': 1.3,  # 科学知識も優先度高め
            'japanese': 1.2,  # 日本語知識
            'english': 1.2,  # 英語知識
            'general': 1.0,  # 一般知識
            'documentation': 1.4,  # ドキュメンテーション
            'github': 1.3,  # GitHubリポジトリ
            'engineer_sites': 1.2  # エンジニア向けサイト
        }
        
        logger.info("="*80)
        logger.info("Domain Knowledge Integrator Initialized")
        logger.info("="*80)
        logger.info(f"Memory Manager: {self.memory_manager is not None}")
        logger.info(f"RAG Store: {self.rag_store is not None}")
    
    def _initialize_memory_manager(self):
        """メモリマネージャーを初期化"""
        try:
            db_path = self.knowledge_base_path or "database/so8t_memory.db"
            self.memory_manager = SO8TMemoryManager(db_path=db_path)
            logger.info(f"[INTEGRATOR] Memory manager initialized: {db_path}")
        except Exception as e:
            logger.warning(f"[INTEGRATOR] Failed to initialize memory manager: {e}")
    
    def _initialize_rag_store(self):
        """RAGストアを初期化"""
        try:
            rag_path = Path(self.rag_store_path)
            if rag_path.exists():
                # RAGチャンクファイルを検索
                rag_ready_dir = rag_path / "rag_ready"
                if rag_ready_dir.exists():
                    rag_chunks = list(rag_ready_dir.glob("rag_chunks_*.jsonl"))
                    if rag_chunks:
                        logger.info(f"[INTEGRATOR] Found {len(rag_chunks)} RAG chunk files")
                        self.rag_store = {
                            'path': rag_ready_dir,
                            'chunks': sorted(rag_chunks, key=lambda p: p.stat().st_mtime, reverse=True)
                        }
                    else:
                        logger.warning("[INTEGRATOR] No RAG chunk files found")
                else:
                    logger.warning(f"[INTEGRATOR] RAG ready directory not found: {rag_ready_dir}")
            else:
                logger.warning(f"[INTEGRATOR] RAG store path not found: {rag_path}")
        except Exception as e:
            logger.warning(f"[INTEGRATOR] Failed to initialize RAG store: {e}")
    
    def detect_domain(self, query: str) -> str:
        """
        クエリからドメインを検出
        
        Args:
            query: クエリテキスト
        
        Returns:
            ドメイン名
        """
        query_lower = query.lower()
        
        # コーディング関連キーワード
        coding_keywords = [
            'code', 'programming', 'function', 'class', 'method', 'algorithm',
            'python', 'javascript', 'java', 'rust', 'c++', 'sql', 'html', 'css',
            'コード', 'プログラミング', '関数', 'クラス', 'メソッド', 'アルゴリズム'
        ]
        
        # 科学関連キーワード
        science_keywords = [
            'science', 'research', 'experiment', 'theory', 'hypothesis',
            'physics', 'chemistry', 'biology', 'mathematics',
            '科学', '研究', '実験', '理論', '仮説', '物理', '化学', '生物', '数学'
        ]
        
        # ドメインスコアを計算
        coding_score = sum(1 for keyword in coding_keywords if keyword in query_lower)
        science_score = sum(1 for keyword in science_keywords if keyword in query_lower)
        
        if coding_score > science_score and coding_score > 0:
            return 'coding'
        elif science_score > 0:
            return 'science'
        else:
            return 'general'
    
    def search_coding_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """
        コーディング知識を検索
        
        Args:
            query: 検索クエリ
            limit: 最大結果数
        
        Returns:
            知識エントリのリスト
        """
        results = []
        
        # コーディングデータから検索
        if self.coding_data_path:
            try:
                coding_path = Path(self.coding_data_path)
                if coding_path.exists():
                    coding_files = list(coding_path.glob("coding_*.jsonl"))
                    query_lower = query.lower()
                    
                    for coding_file in coding_files[:5]:  # 最新5ファイルを検索
                        with open(coding_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    sample = json.loads(line.strip())
                                    text = sample.get('text', '').lower()
                                    code_blocks = sample.get('code_blocks', [])
                                    
                                    # クエリマッチング
                                    if query_lower in text or any(query_lower in str(cb).lower() for cb in code_blocks):
                                        results.append({
                                            'id': sample.get('original_id', ''),
                                            'topic': 'coding',
                                            'content': sample.get('text', ''),
                                            'code_blocks': code_blocks,
                                            'task_type': sample.get('task_type', 'unknown'),
                                            'confidence': 0.8,
                                            'source': 'coding_dataset'
                                        })
                                        if len(results) >= limit:
                                            break
                                except json.JSONDecodeError:
                                    continue
                        if len(results) >= limit:
                            break
            except Exception as e:
                logger.debug(f"[INTEGRATOR] Coding knowledge search failed: {e}")
        
        # RAGストアから検索
        if self.rag_store and self.rag_store.get('chunks'):
            try:
                query_lower = query.lower()
                for chunk_file in self.rag_store['chunks'][:10]:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                chunk = json.loads(line.strip())
                                domain = chunk.get('domain', '').lower()
                                
                                # コーディング関連ドメインを優先
                                if domain in ['programming', 'technology', 'coding', 'github']:
                                    chunk_text = chunk.get('chunk_text', '').lower()
                                    if query_lower in chunk_text:
                                        results.append({
                                            'id': chunk.get('chunk_id', ''),
                                            'topic': 'coding',
                                            'content': chunk.get('chunk_text', ''),
                                            'confidence': 0.7,
                                            'source': 'rag_store'
                                        })
                                        if len(results) >= limit:
                                            break
                            except json.JSONDecodeError:
                                continue
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.debug(f"[INTEGRATOR] RAG store search failed: {e}")
        
        return results[:limit]
    
    def search_science_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """
        科学知識を検索
        
        Args:
            query: 検索クエリ
            limit: 最大結果数
        
        Returns:
            知識エントリのリスト
        """
        results = []
        
        # 科学データから検索
        if self.science_data_path:
            try:
                science_path = Path(self.science_data_path)
                if science_path.exists():
                    science_files = list(science_path.glob("*.jsonl"))
                    query_lower = query.lower()
                    
                    for science_file in science_files[:5]:
                        with open(science_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    sample = json.loads(line.strip())
                                    category = sample.get('category', '').lower()
                                    
                                    # 科学関連カテゴリを優先
                                    if category in ['science', 'research', 'education']:
                                        text = sample.get('text', '').lower()
                                        if query_lower in text:
                                            results.append({
                                                'id': sample.get('id', ''),
                                                'topic': 'science',
                                                'content': sample.get('text', ''),
                                                'category': category,
                                                'confidence': 0.8,
                                                'source': 'science_dataset'
                                            })
                                            if len(results) >= limit:
                                                break
                                except json.JSONDecodeError:
                                    continue
                        if len(results) >= limit:
                            break
            except Exception as e:
                logger.debug(f"[INTEGRATOR] Science knowledge search failed: {e}")
        
        # RAGストアから検索
        if self.rag_store and self.rag_store.get('chunks'):
            try:
                query_lower = query.lower()
                for chunk_file in self.rag_store['chunks'][:10]:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                chunk = json.loads(line.strip())
                                domain = chunk.get('domain', '').lower()
                                category = chunk.get('category', '').lower()
                                
                                # 科学関連ドメイン/カテゴリを優先
                                if domain in ['science', 'research', 'education'] or category in ['science', 'research']:
                                    chunk_text = chunk.get('chunk_text', '').lower()
                                    if query_lower in chunk_text:
                                        results.append({
                                            'id': chunk.get('chunk_id', ''),
                                            'topic': 'science',
                                            'content': chunk.get('chunk_text', ''),
                                            'confidence': 0.7,
                                            'source': 'rag_store'
                                        })
                                        if len(results) >= limit:
                                            break
                            except json.JSONDecodeError:
                                continue
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.debug(f"[INTEGRATOR] RAG store search failed: {e}")
        
        return results[:limit]
    
    def search_language_knowledge(self, query: str, language: str = 'ja', limit: int = 5) -> List[Dict]:
        """
        言語別知識を検索
        
        Args:
            query: 検索クエリ
            language: 言語（ja/en）
            limit: 最大結果数
        
        Returns:
            知識エントリのリスト
        """
        results = []
        
        # RAGストアから検索
        if self.rag_store and self.rag_store.get('chunks'):
            try:
                query_lower = query.lower()
                for chunk_file in self.rag_store['chunks'][:10]:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                chunk = json.loads(line.strip())
                                chunk_language = chunk.get('language', 'unknown')
                                
                                # 言語が一致する場合
                                if chunk_language == language:
                                    chunk_text = chunk.get('chunk_text', '').lower()
                                    if query_lower in chunk_text or any(word in chunk_text for word in query_lower.split()):
                                        results.append({
                                            'id': chunk.get('chunk_id', ''),
                                            'topic': f'{language}_knowledge',
                                            'content': chunk.get('chunk_text', ''),
                                            'language': language,
                                            'confidence': 0.7,
                                            'source': 'rag_store'
                                        })
                                        if len(results) >= limit:
                                            break
                            except json.JSONDecodeError:
                                continue
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.debug(f"[INTEGRATOR] Language knowledge search failed: {e}")
        
        # メモリマネージャーから検索
        if self.memory_manager:
            try:
                memory_results = self.memory_manager.search_knowledge(
                    query=query,
                    limit=limit
                )
                # 言語でフィルタリング
                for result in memory_results:
                    content = result.get('content', '')
                    if language == 'ja' and any(ord(c) > 0x3040 for c in content):
                        results.append({
                            **result,
                            'topic': f'{language}_knowledge',
                            'language': language,
                            'source': 'memory_manager'
                        })
                    elif language == 'en' and not any(ord(c) > 0x3040 for c in content):
                        results.append({
                            **result,
                            'topic': f'{language}_knowledge',
                            'language': language,
                            'source': 'memory_manager'
                        })
            except Exception as e:
                logger.debug(f"[INTEGRATOR] Memory manager search failed: {e}")
        
        return results[:limit]
    
    def integrate_knowledge(
        self,
        query: str,
        domains: Optional[List[str]] = None,
        limit_per_domain: int = 3
    ) -> Dict[str, Any]:
        """
        ドメイン別知識を統合
        
        Args:
            query: 検索クエリ
            domains: 検索するドメインのリスト（Noneの場合は自動検出）
            limit_per_domain: ドメインあたりの最大結果数
        
        Returns:
            統合された知識の辞書
        """
        # ドメインを自動検出
        if domains is None:
            detected_domain = self.detect_domain(query)
            domains = [detected_domain]
            logger.info(f"[INTEGRATOR] Detected domain: {detected_domain}")
        
        all_results = []
        
        # 各ドメインから知識を検索
        for domain in domains:
            domain_results = []
            
            if domain == 'coding':
                domain_results = self.search_coding_knowledge(query, limit=limit_per_domain)
            elif domain == 'science':
                domain_results = self.search_science_knowledge(query, limit=limit_per_domain)
            elif domain == 'japanese':
                domain_results = self.search_language_knowledge(query, language='ja', limit=limit_per_domain)
            elif domain == 'english':
                domain_results = self.search_language_knowledge(query, language='en', limit=limit_per_domain)
            else:
                # 一般検索
                if self.memory_manager:
                    try:
                        general_results = self.memory_manager.search_knowledge(query, limit=limit_per_domain)
                        domain_results = general_results
                    except Exception:
                        pass
            
            # ドメイン重みを適用
            weight = self.domain_weights.get(domain, 1.0)
            for result in domain_results:
                result['domain'] = domain
                result['weighted_confidence'] = result.get('confidence', 0.5) * weight
                all_results.append(result)
        
        # 信頼度でソート
        all_results.sort(key=lambda x: x.get('weighted_confidence', 0.0), reverse=True)
        
        # 統合コンテキストを生成
        integrated_context = self._generate_integrated_context(all_results)
        
        return {
            'query': query,
            'domains': domains,
            'results': all_results,
            'total_results': len(all_results),
            'integrated_context': integrated_context,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_integrated_context(self, results: List[Dict], max_length: int = 2000) -> str:
        """
        統合コンテキストを生成
        
        Args:
            results: 知識エントリのリスト
            max_length: 最大長
        
        Returns:
            統合コンテキストテキスト
        """
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result.get('content', '')
            domain = result.get('domain', 'unknown')
            source = result.get('source', 'unknown')
            
            # コンテキスト部分を生成
            context_part = f"[{domain.upper()}] {content[:300]}"
            
            if current_length + len(context_part) > max_length:
                break
            
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n\n".join(context_parts)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Domain Knowledge Integrator')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    parser.add_argument('--domain', type=str, nargs='+', help='Domains to search')
    parser.add_argument('--knowledge-base', type=str, help='Knowledge base path')
    parser.add_argument('--rag-store', type=str, help='RAG store path')
    parser.add_argument('--coding-data', type=str, help='Coding data path')
    parser.add_argument('--science-data', type=str, help='Science data path')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    integrator = DomainKnowledgeIntegrator(
        knowledge_base_path=args.knowledge_base,
        rag_store_path=args.rag_store,
        coding_data_path=args.coding_data,
        science_data_path=args.science_data
    )
    
    result = integrator.integrate_knowledge(
        query=args.query,
        domains=args.domain
    )
    
    # 結果を出力
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"[INTEGRATOR] Result saved to {output_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()































































