#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合AIエージェント用REST API

統合AIエージェント用のREST APIサーバー。
/thinkエンドポイントの拡張、ドメイン別知識検索API、四重推論と四値分類の統合APIを提供。

Usage:
    python scripts/api/unified_agent_api.py --host 0.0.0.0 --port 8000
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "agents"))

# FastAPIインポート
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[ERROR] FastAPI not available. Please install: pip install fastapi uvicorn")

# 統合モジュールのインポート
try:
    from unified_ai_agent import UnifiedAIAgent
    from domain_knowledge_integrator import DomainKnowledgeIntegrator
    from integrated_reasoning_pipeline import IntegratedReasoningPipeline
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    print(f"[ERROR] Agent modules not available: {e}")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_agent_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if FASTAPI_AVAILABLE and AGENT_AVAILABLE:
    app = FastAPI(
        title="SO8T Unified Agent API",
        description="Unified AI Agent API with quadruple thinking, four-class classification, and domain knowledge integration",
        version="1.0.0",
    )
    
    # CORS設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # グローバル変数
    agent: Optional[UnifiedAIAgent] = None
    integrator: Optional[DomainKnowledgeIntegrator] = None
    pipeline: Optional[IntegratedReasoningPipeline] = None
    
    # リクエストモデル
    class QueryRequest(BaseModel):
        """クエリリクエストモデル"""
        query: str = Field(..., min_length=1, max_length=10000, description="ユーザークエリ")
        user_id: str = Field("default", min_length=1, max_length=256, description="ユーザーID")
        use_knowledge: bool = Field(True, description="ドメイン別知識を使用するか")
        use_classification: bool = Field(True, description="四値分類を使用するか")
        use_rag: bool = Field(True, description="RAGによる知識拡張を使用するか")
    
    class KnowledgeSearchRequest(BaseModel):
        """知識検索リクエストモデル"""
        query: str = Field(..., min_length=1, max_length=1000, description="検索クエリ")
        domains: Optional[List[str]] = Field(None, description="検索するドメインのリスト")
        limit: int = Field(5, ge=1, le=50, description="最大結果数")
    
    class ThinkRequest(BaseModel):
        """/thinkエンドポイントのリクエストモデル"""
        query: str = Field(..., min_length=1, max_length=10000, description="ユーザークエリ")
        user_id: str = Field("default", min_length=1, max_length=256, description="ユーザーID")
        use_knowledge: bool = Field(True, description="ドメイン別知識を使用するか")
        use_classification: bool = Field(True, description="四値分類を使用するか")
    
    # レスポンスモデル
    class QueryResponse(BaseModel):
        """クエリレスポンスモデル"""
        query: str
        user_id: str
        timestamp: str
        domain_detection: Optional[Dict[str, Any]] = None
        knowledge_integration: Optional[Dict[str, Any]] = None
        quadruple_thinking: Optional[Dict[str, Any]] = None
        four_class_classification: Optional[Dict[str, Any]] = None
        final_answer: str
        safety_label: str
        reasoning_steps: List[Dict[str, Any]] = []
    
    class KnowledgeSearchResponse(BaseModel):
        """知識検索レスポンスモデル"""
        query: str
        domains: List[str]
        results: List[Dict[str, Any]]
        total_results: int
        integrated_context: str
        timestamp: str
    
    class ThinkResponse(BaseModel):
        """/thinkエンドポイントのレスポンスモデル"""
        answer: str
        safety_label: str
        safety_conf: float
        domain_label: Optional[str] = None
        domain_conf: Optional[float] = None
        quadruple_thinking: Optional[Dict[str, Any]] = None
        four_class_classification: Optional[Dict[str, Any]] = None
        internal_reasoning_hash: Optional[str] = None
    
    @app.on_event("startup")
    async def startup():
        """アプリケーション起動時の初期化"""
        global agent, integrator, pipeline
        
        # 環境変数から設定を取得
        model_path = os.getenv("SO8T_MODEL_PATH")
        knowledge_base_path = os.getenv("SO8T_KNOWLEDGE_BASE_PATH", "database/so8t_memory.db")
        rag_store_path = os.getenv("SO8T_RAG_STORE_PATH", "D:/webdataset/vector_stores")
        coding_data_path = os.getenv("SO8T_CODING_DATA_PATH", "D:/webdataset/processed/coding")
        science_data_path = os.getenv("SO8T_SCIENCE_DATA_PATH", "D:/webdataset/processed/science")
        
        # 統合AIエージェントを初期化
        try:
            agent = UnifiedAIAgent(
                model_path=model_path,
                knowledge_base_path=knowledge_base_path,
                rag_store_path=rag_store_path
            )
            logger.info("[API] Unified AI Agent initialized")
        except Exception as e:
            logger.error(f"[API] Failed to initialize Unified AI Agent: {e}")
            agent = None
        
        # ドメイン別知識統合モジュールを初期化
        try:
            integrator = DomainKnowledgeIntegrator(
                knowledge_base_path=knowledge_base_path,
                rag_store_path=rag_store_path,
                coding_data_path=coding_data_path,
                science_data_path=science_data_path
            )
            logger.info("[API] Domain Knowledge Integrator initialized")
        except Exception as e:
            logger.error(f"[API] Failed to initialize Domain Knowledge Integrator: {e}")
            integrator = None
        
        # 統合推論パイプラインを初期化
        try:
            pipeline = IntegratedReasoningPipeline(
                model_path=model_path,
                knowledge_base_path=knowledge_base_path,
                rag_store_path=rag_store_path,
                coding_data_path=coding_data_path,
                science_data_path=science_data_path
            )
            logger.info("[API] Integrated Reasoning Pipeline initialized")
        except Exception as e:
            logger.error(f"[API] Failed to initialize Integrated Reasoning Pipeline: {e}")
            pipeline = None
    
    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest, request: Request) -> QueryResponse:
        """
        統合クエリエンドポイント
        
        四重推論、四値分類、ドメイン別知識を統合したクエリ処理
        """
        if not pipeline:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        try:
            result = pipeline.process_with_integrated_reasoning(
                query=req.query,
                user_id=req.user_id,
                use_knowledge=req.use_knowledge,
                use_classification=req.use_classification,
                use_rag=req.use_rag
            )
            
            return QueryResponse(**result)
        except Exception as e:
            logger.error(f"[API] Query processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @app.post("/think", response_model=ThinkResponse)
    async def think_endpoint(req: ThinkRequest, request: Request) -> ThinkResponse:
        """
        /thinkエンドポイント（拡張版）
        
        四重推論と四値分類を統合した/thinkエンドポイント
        """
        if not agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        try:
            # 統合推論で処理
            if pipeline:
                result = pipeline.process_with_integrated_reasoning(
                    query=req.query,
                    user_id=req.user_id,
                    use_knowledge=req.use_knowledge,
                    use_classification=req.use_classification
                )
                
                # レスポンスを構築
                thinking = result.get('quadruple_thinking', {})
                classification = result.get('four_class_classification', {})
                
                return ThinkResponse(
                    answer=result.get('final_answer', '回答を生成できませんでした。'),
                    safety_label=result.get('safety_label', 'ALLOW'),
                    safety_conf=classification.get('final_confidence', 0.5) if classification else 0.5,
                    domain_label=result.get('domain_detection', {}).get('domain', 'general'),
                    domain_conf=result.get('domain_detection', {}).get('confidence', 0.0),
                    quadruple_thinking=thinking,
                    four_class_classification=classification,
                    internal_reasoning_hash=None  # TODO: ハッシュ計算を実装
                )
            else:
                # フォールバック: エージェントのみで処理
                result = agent.process_query(
                    query=req.query,
                    user_id=req.user_id,
                    use_knowledge=req.use_knowledge,
                    use_classification=req.use_classification
                )
                
                return ThinkResponse(
                    answer=result.get('final_answer', '回答を生成できませんでした。'),
                    safety_label=result.get('safety_label', 'ALLOW'),
                    safety_conf=0.5,
                    domain_label='general',
                    domain_conf=0.0,
                    quadruple_thinking=result.get('quadruple_thinking'),
                    four_class_classification=result.get('four_class_classification'),
                    internal_reasoning_hash=None
                )
        except Exception as e:
            logger.error(f"[API] Think processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @app.post("/knowledge/search", response_model=KnowledgeSearchResponse)
    async def knowledge_search_endpoint(req: KnowledgeSearchRequest, request: Request) -> KnowledgeSearchResponse:
        """
        ドメイン別知識検索API
        """
        if not integrator:
            raise HTTPException(status_code=500, detail="Integrator not initialized")
        
        try:
            result = integrator.integrate_knowledge(
                query=req.query,
                domains=req.domains,
                limit_per_domain=req.limit
            )
            
            return KnowledgeSearchResponse(**result)
        except Exception as e:
            logger.error(f"[API] Knowledge search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """ヘルスチェックエンドポイント"""
        return {
            "status": "healthy",
            "agent_initialized": agent is not None,
            "integrator_initialized": integrator is not None,
            "pipeline_initialized": pipeline is not None
        }
    
    @app.get("/")
    async def root():
        """ルートエンドポイント"""
        return {
            "message": "SO8T Unified Agent API",
            "version": "1.0.0",
            "endpoints": {
                "/query": "統合クエリエンドポイント",
                "/think": "/thinkエンドポイント（拡張版）",
                "/knowledge/search": "ドメイン別知識検索API",
                "/health": "ヘルスチェック"
            }
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Unified Agent API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    if not FASTAPI_AVAILABLE:
        logger.error("[ERROR] FastAPI not available. Please install: pip install fastapi uvicorn")
        return
    
    if not AGENT_AVAILABLE:
        logger.error("[ERROR] Agent modules not available")
        return
    
    try:
        import uvicorn
        logger.info(f"[API] Starting server on {args.host}:{args.port}")
        uvicorn.run(
            "unified_agent_api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except ImportError:
        logger.error("[ERROR] uvicorn not available. Please install: pip install uvicorn")
    except Exception as e:
        logger.error(f"[ERROR] Failed to start server: {e}")


if __name__ == '__main__':
    main()


































































































































