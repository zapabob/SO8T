#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクレイピング判断のための四重推論・四値分類統合エージェント

SO8T/thinkingモデルによる統制判断の実装

Usage:
    from scripts.agents.scraping_reasoning_agent import ScrapingReasoningAgent
    
    agent = ScrapingReasoningAgent()
    result = agent.should_scrape(url="https://example.com", keyword="Python")
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "agents"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "audit"))

# 統合エージェントインポート
try:
    from scripts.agents.unified_ai_agent import UnifiedAIAgent
    from scripts.agents.integrated_reasoning_pipeline import IntegratedReasoningPipeline
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Agent modules not available: {e}")

# 監査ログインポート
try:
    from scripts.audit.scraping_audit_logger import ScrapingAuditLogger, ScrapingEvent
    AUDIT_LOGGER_AVAILABLE = True
except ImportError:
    AUDIT_LOGGER_AVAILABLE = False

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraping_reasoning_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScrapingReasoningAgent:
    """スクレイピング判断のための四重推論・四値分類統合エージェント"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        knowledge_base_path: Optional[str] = None,
        rag_store_path: Optional[str] = None,
        audit_logger: Optional[ScrapingAuditLogger] = None
    ):
        """
        初期化
        
        Args:
            model_path: SO8Tモデルのパス
            knowledge_base_path: 知識ベースのパス
            rag_store_path: RAGストアのパス
            audit_logger: 監査ロガー
        """
        if not AGENT_AVAILABLE:
            raise RuntimeError("Agent modules not available")
        
        # 統合AIエージェントの初期化
        self.agent = UnifiedAIAgent(
            model_path=model_path,
            knowledge_base_path=knowledge_base_path,
            rag_store_path=rag_store_path
        )
        
        # 統合推論パイプラインの初期化
        self.pipeline = IntegratedReasoningPipeline(
            model_path=model_path,
            knowledge_base_path=knowledge_base_path,
            rag_store_path=rag_store_path
        )
        
        # 監査ロガー
        self.audit_logger = audit_logger
        
        logger.info("="*80)
        logger.info("Scraping Reasoning Agent Initialized")
        logger.info("="*80)
    
    def should_scrape(
        self,
        url: str,
        keyword: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        スクレイピングすべきか判断（四重推論・四値分類統合）
        
        Args:
            url: URL
            keyword: キーワード
            context: 文脈情報（オプション）
            session_id: セッションID（オプション）
            
        Returns:
            result: 判断結果の辞書
        """
        logger.info(f"[REASONING] Evaluating scraping decision: {url[:50]}... (keyword: {keyword})")
        
        # クエリ構築
        query = f"Should I scrape the URL {url} for keyword '{keyword}'?"
        if context:
            query += f" Context: {context}"
        
        # 四重推論・四値分類で処理
        try:
            result = self.pipeline.process_with_integrated_reasoning(
                query=query,
                user_id=session_id or "scraping_agent",
                use_knowledge=True,
                use_classification=True,
                use_rag=True
            )
            
            # 判断結果を抽出
            decision = result.get('safety_label', 'ALLOW')
            quadruple_thinking = result.get('quadruple_thinking', {})
            four_class_classification = result.get('four_class_classification', {})
            final_answer = result.get('final_answer', '')
            
            # 判断理由を生成
            reasoning = self._generate_reasoning(
                decision, quadruple_thinking, four_class_classification, final_answer
            )
            
            # 結果を構築
            scraping_result = {
                'should_scrape': decision in ['ALLOW', 'ESCALATION'],
                'decision': decision,
                'confidence': four_class_classification.get('final_confidence', 0.5) if four_class_classification else 0.5,
                'reasoning': reasoning,
                'quadruple_thinking': quadruple_thinking,
                'four_class_classification': four_class_classification,
                'url': url,
                'keyword': keyword,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            
            # 監査ログ記録
            if self.audit_logger and session_id:
                event = ScrapingEvent(
                    event_id=f"{session_id}_decision_{int(datetime.now().timestamp())}",
                    session_id=session_id,
                    timestamp=datetime.now().isoformat(),
                    event_type="so8t_decision",
                    url=url,
                    keyword=keyword,
                    quadruple_thinking=quadruple_thinking,
                    four_class_classification=four_class_classification,
                    decision=decision,
                    details={
                        'should_scrape': scraping_result['should_scrape'],
                        'confidence': scraping_result['confidence'],
                        'reasoning': reasoning
                    }
                )
                self.audit_logger.log_scraping_event(event)
            
            logger.info(f"[REASONING] Decision: {decision} (should_scrape: {scraping_result['should_scrape']})")
            return scraping_result
            
        except Exception as e:
            logger.error(f"[REASONING] Failed to process reasoning: {e}")
            
            # エラー時のフォールバック判断
            return {
                'should_scrape': False,
                'decision': 'REFUSE',
                'confidence': 0.0,
                'reasoning': f"Error during reasoning: {str(e)}",
                'quadruple_thinking': None,
                'four_class_classification': None,
                'url': url,
                'keyword': keyword,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _generate_reasoning(
        self,
        decision: str,
        quadruple_thinking: Dict,
        four_class_classification: Dict,
        final_answer: str
    ) -> str:
        """
        判断理由を生成
        
        Args:
            decision: 判断結果
            quadruple_thinking: 四重推論結果
            four_class_classification: 四値分類結果
            final_answer: 最終回答
            
        Returns:
            reasoning: 判断理由
        """
        reasoning_parts = []
        
        # 四重推論から理由を抽出
        if quadruple_thinking:
            task = quadruple_thinking.get('task', '')
            safety = quadruple_thinking.get('safety', '')
            policy = quadruple_thinking.get('policy', '')
            
            if task:
                reasoning_parts.append(f"Task analysis: {task[:100]}")
            if safety:
                reasoning_parts.append(f"Safety considerations: {safety[:100]}")
            if policy:
                reasoning_parts.append(f"Policy compliance: {policy[:100]}")
        
        # 四値分類から理由を抽出
        if four_class_classification:
            classification_reasoning = four_class_classification.get('reasoning', '')
            if classification_reasoning:
                reasoning_parts.append(f"Classification: {classification_reasoning}")
        
        # 最終回答から理由を抽出
        if final_answer:
            reasoning_parts.append(f"Final assessment: {final_answer[:200]}")
        
        # 判断結果
        decision_map = {
            'ALLOW': 'スクレイピングを許可します',
            'ESCALATION': '追加の確認が必要です',
            'DENY': 'スクレイピングを拒否します',
            'REFUSE': 'スクレイピングを拒否します（エラー）'
        }
        reasoning_parts.append(f"Decision: {decision_map.get(decision, decision)}")
        
        return " | ".join(reasoning_parts) if reasoning_parts else "No reasoning available"
    
    def batch_evaluate(
        self,
        urls: List[str],
        keyword: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        バッチ評価
        
        Args:
            urls: URLのリスト
            keyword: キーワード
            context: 文脈情報（オプション）
            session_id: セッションID（オプション）
            
        Returns:
            results: 判断結果のリスト
        """
        results = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"[REASONING] Evaluating {i}/{len(urls)}: {url[:50]}...")
            result = self.should_scrape(
                url=url,
                keyword=keyword,
                context=context,
                session_id=session_id
            )
            results.append(result)
        
        return results
    
    def evaluate_keyword(
        self,
        keyword: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        キーワードのスクレイピング適性を評価
        
        Args:
            keyword: キーワード
            context: 文脈情報（オプション）
            session_id: セッションID（オプション）
            
        Returns:
            result: 評価結果の辞書
        """
        logger.info(f"[REASONING] Evaluating keyword: {keyword}")
        
        # クエリ構築
        query = f"Is it appropriate to scrape web pages for keyword '{keyword}'?"
        if context:
            query += f" Context: {context}"
        
        # 四重推論・四値分類で処理
        try:
            result = self.pipeline.process_with_integrated_reasoning(
                query=query,
                user_id=session_id or "scraping_agent",
                use_knowledge=True,
                use_classification=True,
                use_rag=True
            )
            
            # 判断結果を抽出
            decision = result.get('safety_label', 'ALLOW')
            quadruple_thinking = result.get('quadruple_thinking', {})
            four_class_classification = result.get('four_class_classification', {})
            final_answer = result.get('final_answer', '')
            
            # 判断理由を生成
            reasoning = self._generate_reasoning(
                decision, quadruple_thinking, four_class_classification, final_answer
            )
            
            # 結果を構築
            keyword_result = {
                'keyword': keyword,
                'should_scrape': decision in ['ALLOW', 'ESCALATION'],
                'decision': decision,
                'confidence': four_class_classification.get('final_confidence', 0.5) if four_class_classification else 0.5,
                'reasoning': reasoning,
                'quadruple_thinking': quadruple_thinking,
                'four_class_classification': four_class_classification,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[REASONING] Keyword evaluation: {decision} (should_scrape: {keyword_result['should_scrape']})")
            return keyword_result
            
        except Exception as e:
            logger.error(f"[REASONING] Failed to evaluate keyword: {e}")
            
            # エラー時のフォールバック判断
            return {
                'keyword': keyword,
                'should_scrape': False,
                'decision': 'REFUSE',
                'confidence': 0.0,
                'reasoning': f"Error during reasoning: {str(e)}",
                'quadruple_thinking': None,
                'four_class_classification': None,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


def main():
    """テスト実行"""
    logger.info("="*80)
    logger.info("Scraping Reasoning Agent Test")
    logger.info("="*80)
    
    try:
        agent = ScrapingReasoningAgent()
        
        # テストURL評価
        result = agent.should_scrape(
            url="https://example.com",
            keyword="Python",
            context="Educational content about Python programming"
        )
        
        logger.info(f"[TEST] Result: {result['decision']} (should_scrape: {result['should_scrape']})")
        logger.info(f"[TEST] Reasoning: {result['reasoning'][:200]}...")
        
        # テストキーワード評価
        keyword_result = agent.evaluate_keyword(
            keyword="Python",
            context="Educational content"
        )
        
        logger.info(f"[TEST] Keyword result: {keyword_result['decision']} (should_scrape: {keyword_result['should_scrape']})")
        
        logger.info("[OK] Test completed")
        
    except Exception as e:
        logger.error(f"[TEST] Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

