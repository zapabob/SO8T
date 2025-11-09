#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合AIエージェントコア

四重推論と四値分類と/thinkを統合したAIエージェントコア。
ドメイン別知識の統合検索、RAG/CoGによる知識拡張を実装。

Usage:
    python scripts/agents/unified_ai_agent.py --query "Pythonでリストをソートする方法"
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_ai_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SO8Tモデルインポート
try:
    import torch
    from transformers import AutoTokenizer
    from models.so8t_thinking_model import SO8TThinkingModel
    from models.safety_aware_so8t import SafetyAwareSO8TConfig
    from models.thinking_tokens import (
        build_quadruple_thinking_prompt,
        extract_quadruple_thinking
    )
    SO8T_AVAILABLE = True
except ImportError as e:
    SO8T_AVAILABLE = False
    logger.warning(f"SO8T model not available: {e}")

# 四値分類インポート
try:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipelines"))
    from web_scraping_data_pipeline import QuadrupleClassifier
    QUADRUPLE_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    QUADRUPLE_CLASSIFIER_AVAILABLE = False
    logger.warning(f"QuadrupleClassifier not available: {e}")

# 知識ベースインポート
try:
    sys.path.insert(0, str(PROJECT_ROOT / "utils"))
    from memory_manager import SO8TMemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    logger.warning(f"SO8TMemoryManager not available: {e}")

# RAG/CoGインポート
try:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipelines"))
    from vector_store_creation import RAGVectorStoreCreator, CoGKnowledgeGraphCreator
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    VECTOR_STORE_AVAILABLE = False
    logger.warning(f"Vector store creators not available: {e}")


class UnifiedAIAgent:
    """統合AIエージェントコア"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        knowledge_base_path: Optional[str] = None,
        rag_store_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        初期化
        
        Args:
            model_path: SO8Tモデルのパス
            knowledge_base_path: 知識ベースのパス
            rag_store_path: RAGストアのパス
            device: デバイス（cuda/cpu）
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_path = model_path
        self.knowledge_base_path = knowledge_base_path
        self.rag_store_path = rag_store_path
        
        # SO8Tモデルの初期化
        self.so8t_model = None
        self.tokenizer = None
        if SO8T_AVAILABLE:
            self._initialize_so8t_model()
        
        # 四値分類器の初期化
        self.quadruple_classifier = None
        if QUADRUPLE_CLASSIFIER_AVAILABLE:
            self._initialize_quadruple_classifier()
        
        # 知識ベースマネージャーの初期化
        self.memory_manager = None
        if MEMORY_MANAGER_AVAILABLE:
            self._initialize_memory_manager()
        
        # RAGストアの初期化
        self.rag_store = None
        if VECTOR_STORE_AVAILABLE and rag_store_path:
            self._initialize_rag_store()
        
        logger.info("="*80)
        logger.info("Unified AI Agent Initialized")
        logger.info("="*80)
        logger.info(f"SO8T Model: {self.so8t_model is not None}")
        logger.info(f"Quadruple Classifier: {self.quadruple_classifier is not None}")
        logger.info(f"Memory Manager: {self.memory_manager is not None}")
        logger.info(f"RAG Store: {self.rag_store is not None}")
    
    def _initialize_so8t_model(self):
        """SO8Tモデルを初期化"""
        try:
            if self.model_path is None:
                # デフォルトモデルパスを探す
                default_paths = [
                    "D:/webdataset/models/so8t-phi4-so8t-ja-finetuned",
                    "models/so8t-phi4-so8t-ja-finetuned",
                    "so8t-mmllm/models/so8t-phi4-so8t-ja-finetuned"
                ]
                for path in default_paths:
                    if Path(path).exists():
                        self.model_path = path
                        break
                
                if self.model_path is None:
                    logger.warning("[AGENT] SO8T model not found, using default")
                    self.model_path = "microsoft/Phi-3-mini-4k-instruct"
            
            logger.info(f"[AGENT] Loading SO8T model from: {self.model_path}")
            
            # トークナイザーを読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # SO8T設定
            so8t_config = SafetyAwareSO8TConfig(
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
                intermediate_size=11008,
                max_position_embeddings=4096,
                use_so8_rotation=True,
                use_safety_head=True,
                use_verifier_head=True
            )
            
            # SO8TThinkingModelを読み込み
            self.so8t_model = SO8TThinkingModel(
                base_model_name_or_path=self.model_path,
                so8t_config=so8t_config,
                use_redacted_tokens=False,
                use_quadruple_thinking=True
            )
            
            # トークナイザーを設定
            self.so8t_model.set_tokenizer(self.tokenizer)
            
            # 評価モードに設定
            self.so8t_model.eval()
            self.so8t_model.to(self.device)
            
            logger.info("[AGENT] SO8T model loaded successfully")
            
        except Exception as e:
            logger.error(f"[AGENT] Failed to initialize SO8T model: {e}")
            self.so8t_model = None
    
    def _initialize_quadruple_classifier(self):
        """四値分類器を初期化"""
        try:
            self.quadruple_classifier = QuadrupleClassifier(so8t_model_path=self.model_path)
            logger.info("[AGENT] Quadruple classifier initialized")
        except Exception as e:
            logger.warning(f"[AGENT] Failed to initialize quadruple classifier: {e}")
            self.quadruple_classifier = None
    
    def _initialize_memory_manager(self):
        """メモリマネージャーを初期化"""
        try:
            db_path = self.knowledge_base_path or "database/so8t_memory.db"
            self.memory_manager = SO8TMemoryManager(db_path=db_path)
            logger.info(f"[AGENT] Memory manager initialized: {db_path}")
        except Exception as e:
            logger.warning(f"[AGENT] Failed to initialize memory manager: {e}")
            self.memory_manager = None
    
    def _initialize_rag_store(self):
        """RAGストアを初期化"""
        try:
            rag_path = Path(self.rag_store_path)
            if rag_path.exists():
                # RAGチャンクファイルを検索
                rag_chunks = list(rag_path.glob("rag_ready/rag_chunks_*.jsonl"))
                if rag_chunks:
                    logger.info(f"[AGENT] Found {len(rag_chunks)} RAG chunk files")
                    self.rag_store = {
                        'path': rag_path,
                        'chunks': rag_chunks
                    }
                else:
                    logger.warning("[AGENT] No RAG chunk files found")
            else:
                logger.warning(f"[AGENT] RAG store path not found: {rag_path}")
        except Exception as e:
            logger.warning(f"[AGENT] Failed to initialize RAG store: {e}")
            self.rag_store = None
    
    def search_domain_knowledge(self, query: str, domain: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """
        ドメイン別知識を検索
        
        Args:
            query: 検索クエリ
            domain: ドメイン（オプション）
            limit: 最大結果数
        
        Returns:
            知識エントリのリスト
        """
        results = []
        
        # メモリマネージャーから検索
        if self.memory_manager:
            try:
                memory_results = self.memory_manager.search_knowledge(
                    query=query,
                    topic=domain,
                    limit=limit
                )
                results.extend(memory_results)
            except Exception as e:
                logger.debug(f"[AGENT] Memory manager search failed: {e}")
        
        # RAGストアから検索（簡易版：キーワードマッチング）
        if self.rag_store and self.rag_store.get('chunks'):
            try:
                query_lower = query.lower()
                for chunk_file in self.rag_store['chunks'][:10]:  # 最新10ファイルを検索
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                chunk = json.loads(line.strip())
                                chunk_text = chunk.get('chunk_text', '').lower()
                                if query_lower in chunk_text or any(word in chunk_text for word in query_lower.split()):
                                    results.append({
                                        'id': chunk.get('chunk_id', ''),
                                        'topic': chunk.get('domain', 'unknown'),
                                        'content': chunk.get('chunk_text', ''),
                                        'confidence': 0.7,  # 簡易マッチングのため固定値
                                        'source': 'rag_store'
                                    })
                                    if len(results) >= limit:
                                        break
                            except json.JSONDecodeError:
                                continue
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.debug(f"[AGENT] RAG store search failed: {e}")
        
        return results[:limit]
    
    def generate_quadruple_thinking(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        四重推論を生成
        
        Args:
            query: ユーザークエリ
            context: 追加コンテキスト（ドメイン別知識など）
        
        Returns:
            四重推論結果の辞書
        """
        if not self.so8t_model or not self.tokenizer:
            logger.error("[AGENT] SO8T model not available")
            return {
                'error': 'SO8T model not available',
                'thinking': None,
                'final': None
            }
        
        try:
            # コンテキストを追加したプロンプトを構築
            if context:
                enhanced_query = f"Context:\n{context}\n\nQuery: {query}"
            else:
                enhanced_query = query
            
            # 四重推論プロンプトを構築
            prompt = build_quadruple_thinking_prompt(enhanced_query)
            
            # 四重推論を生成
            result = self.so8t_model.generate_thinking(
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                device=self.device
            )
            
            # 四重推論を抽出
            full_text = result.get('full_text', '')
            task_text, safety_text, policy_text, final_text = extract_quadruple_thinking(full_text)
            
            return {
                'task': task_text,
                'safety': safety_text,
                'policy': policy_text,
                'final': final_text,
                'full_text': full_text
            }
            
        except Exception as e:
            logger.error(f"[AGENT] Failed to generate quadruple thinking: {e}")
            return {
                'error': str(e),
                'thinking': None,
                'final': None
            }
    
    def classify_four_class(self, sample: Dict) -> Dict:
        """
        四値分類を実行
        
        Args:
            sample: 分類するサンプル
        
        Returns:
            四値分類結果を含むサンプル
        """
        if not self.quadruple_classifier:
            logger.warning("[AGENT] Quadruple classifier not available, using default")
            return {
                **sample,
                'four_class_label': 'ALLOW',
                'four_class_label_id': 0
            }
        
        try:
            return self.quadruple_classifier.classify_quadruple(sample)
        except Exception as e:
            logger.error(f"[AGENT] Failed to classify: {e}")
            return {
                **sample,
                'four_class_label': 'ALLOW',
                'four_class_label_id': 0
            }
    
    def process_query(
        self,
        query: str,
        user_id: str = "default",
        use_knowledge: bool = True,
        use_classification: bool = True
    ) -> Dict[str, Any]:
        """
        クエリを処理（統合フロー）
        
        Args:
            query: ユーザークエリ
            user_id: ユーザーID
            use_knowledge: ドメイン別知識を使用するか
            use_classification: 四値分類を使用するか
        
        Returns:
            処理結果の辞書
        """
        logger.info(f"[AGENT] Processing query: {query[:100]}...")
        
        result = {
            'query': query,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'knowledge_context': None,
            'quadruple_thinking': None,
            'four_class_classification': None,
            'final_answer': None,
            'safety_label': None
        }
        
        # 1. ドメイン別知識を検索
        if use_knowledge:
            try:
                knowledge_results = self.search_domain_knowledge(query, limit=5)
                if knowledge_results:
                    context = "\n\n".join([
                        f"[{r.get('topic', 'unknown')}] {r.get('content', '')[:200]}"
                        for r in knowledge_results
                    ])
                    result['knowledge_context'] = context
                    logger.info(f"[AGENT] Found {len(knowledge_results)} knowledge entries")
            except Exception as e:
                logger.warning(f"[AGENT] Knowledge search failed: {e}")
        
        # 2. 四重推論を生成
        try:
            thinking_result = self.generate_quadruple_thinking(
                query,
                context=result.get('knowledge_context')
            )
            result['quadruple_thinking'] = thinking_result
        except Exception as e:
            logger.error(f"[AGENT] Quadruple thinking generation failed: {e}")
            result['quadruple_thinking'] = {'error': str(e)}
        
        # 3. 四値分類を実行
        if use_classification and result.get('quadruple_thinking'):
            try:
                sample = {
                    'text': query,
                    'category': 'general',
                    'language': 'ja' if any(ord(c) > 0x3040 for c in query) else 'en'
                }
                classification_result = self.classify_four_class(sample)
                result['four_class_classification'] = classification_result.get('quadruple_classification', {})
                result['four_class_label'] = classification_result.get('four_class_label', 'ALLOW')
            except Exception as e:
                logger.warning(f"[AGENT] Four-class classification failed: {e}")
                result['four_class_label'] = 'ALLOW'
        else:
            result['four_class_label'] = 'ALLOW'
        
        # 4. 最終回答を抽出
        if result.get('quadruple_thinking') and result['quadruple_thinking'].get('final'):
            result['final_answer'] = result['quadruple_thinking']['final']
            result['safety_label'] = result.get('four_class_label', 'ALLOW')
        else:
            result['final_answer'] = "回答を生成できませんでした。"
            result['safety_label'] = 'REFUSE'
        
        logger.info(f"[AGENT] Processing completed: {result['safety_label']}")
        return result


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Unified AI Agent')
    parser.add_argument('--query', type=str, required=True, help='User query')
    parser.add_argument('--model-path', type=str, help='SO8T model path')
    parser.add_argument('--knowledge-base', type=str, help='Knowledge base path')
    parser.add_argument('--rag-store', type=str, help='RAG store path')
    parser.add_argument('--user-id', type=str, default='default', help='User ID')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    agent = UnifiedAIAgent(
        model_path=args.model_path,
        knowledge_base_path=args.knowledge_base,
        rag_store_path=args.rag_store
    )
    
    result = agent.process_query(
        query=args.query,
        user_id=args.user_id
    )
    
    # 結果を出力
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"[AGENT] Result saved to {output_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

