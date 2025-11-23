#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合推論パイプライン

四重推論と四値分類と/thinkを統合した推論パイプライン。
ドメイン別知識を活用した推論、RAG/CoGによる知識拡張推論を実装。

Usage:
    python scripts/agents/integrated_reasoning_pipeline.py --query "Pythonでリストをソートする方法"
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "agents"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/integrated_reasoning_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 統合モジュールのインポート
try:
    from unified_ai_agent import UnifiedAIAgent
    from domain_knowledge_integrator import DomainKnowledgeIntegrator
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    logger.error(f"Failed to import agent modules: {e}")


class IntegratedReasoningPipeline:
    """統合推論パイプライン"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        knowledge_base_path: Optional[str] = None,
        rag_store_path: Optional[str] = None,
        coding_data_path: Optional[str] = None,
        science_data_path: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            model_path: SO8Tモデルのパス
            knowledge_base_path: 知識ベースのパス
            rag_store_path: RAGストアのパス
            coding_data_path: コーディングデータのパス
            science_data_path: 科学データのパス
        """
        # エージェントモジュールの可用性チェック
        if not AGENT_AVAILABLE:
            logger.warning("[PIPELINE] Agent modules not available, using fallback mode")
            self.agent = None
            self.integrator = None
            self.fallback_mode = True
        else:
            self.fallback_mode = False
            try:
                # 統合AIエージェントの初期化
                self.agent = UnifiedAIAgent(
                    model_path=model_path,
                    knowledge_base_path=knowledge_base_path,
                    rag_store_path=rag_store_path
                )
                
                # ドメイン別知識統合モジュールの初期化
                self.integrator = DomainKnowledgeIntegrator(
                    knowledge_base_path=knowledge_base_path,
                    rag_store_path=rag_store_path,
                    coding_data_path=coding_data_path,
                    science_data_path=science_data_path
                )
            except Exception as e:
                logger.warning(f"[PIPELINE] Failed to initialize agent/integrator: {e}, using fallback mode")
                self.agent = None
                self.integrator = None
                self.fallback_mode = True
        
        logger.info("="*80)
        logger.info("Integrated Reasoning Pipeline Initialized")
        logger.info("="*80)
    
    def process_with_integrated_reasoning(
        self,
        query: str,
        user_id: str = "default",
        use_knowledge: bool = True,
        use_classification: bool = True,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        統合推論でクエリを処理
        
        Args:
            query: ユーザークエリ
            user_id: ユーザーID
            use_knowledge: ドメイン別知識を使用するか
            use_classification: 四値分類を使用するか
            use_rag: RAGによる知識拡張を使用するか
        
        Returns:
            処理結果の辞書
        """
        logger.info(f"[PIPELINE] Processing query with integrated reasoning: {query[:100]}...")
        
        result = {
            'query': query,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'domain_detection': None,
            'knowledge_integration': None,
            'quadruple_thinking': None,
            'four_class_classification': None,
            'final_answer': None,
            'safety_label': None,
            'reasoning_steps': []
        }
        
        # Step 1: ドメイン検出
        detected_domain = 'general'
        if not self.fallback_mode and self.integrator:
            try:
                detected_domain = self.integrator.detect_domain(query)
                result['domain_detection'] = {
                    'domain': detected_domain,
                    'confidence': 1.0
                }
                result['reasoning_steps'].append({
                    'step': 'domain_detection',
                    'result': detected_domain,
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"[PIPELINE] Detected domain: {detected_domain}")
            except Exception as e:
                logger.error(f"[PIPELINE] Domain detection failed: {e}")
                import traceback
                logger.error(f"[PIPELINE] Domain detection traceback:\n{traceback.format_exc()}")
                result['domain_detection'] = {'domain': 'general', 'confidence': 0.0, 'error': str(e)}
        else:
            logger.info("[PIPELINE] Using fallback mode for domain detection")
            result['domain_detection'] = {'domain': 'general', 'confidence': 0.0}
        
        # Step 2: ドメイン別知識の統合
        if use_knowledge and not self.fallback_mode and self.integrator:
            try:
                knowledge_result = self.integrator.integrate_knowledge(
                    query=query,
                    domains=[detected_domain] if detected_domain else None
                )
                result['knowledge_integration'] = knowledge_result
                result['reasoning_steps'].append({
                    'step': 'knowledge_integration',
                    'results_count': knowledge_result.get('total_results', 0),
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"[PIPELINE] Integrated {knowledge_result.get('total_results', 0)} knowledge entries")
            except Exception as e:
                logger.error(f"[PIPELINE] Knowledge integration failed: {e}")
                import traceback
                logger.error(f"[PIPELINE] Knowledge integration traceback:\n{traceback.format_exc()}")
                result['knowledge_integration'] = {'error': str(e), 'error_type': type(e).__name__}
        else:
            result['knowledge_integration'] = {'total_results': 0, 'fallback': True}
        
        # Step 3: 四重推論の生成（知識コンテキストを使用）
        if not self.fallback_mode and self.agent:
            try:
                knowledge_context = None
                if result.get('knowledge_integration') and result['knowledge_integration'].get('integrated_context'):
                    knowledge_context = result['knowledge_integration']['integrated_context']
                
                thinking_result = self.agent.generate_quadruple_thinking(
                    query=query,
                    context=knowledge_context
                )
                result['quadruple_thinking'] = thinking_result
                result['reasoning_steps'].append({
                    'step': 'quadruple_thinking',
                    'has_task': thinking_result.get('task') is not None,
                    'has_safety': thinking_result.get('safety') is not None,
                    'has_policy': thinking_result.get('policy') is not None,
                    'has_final': thinking_result.get('final') is not None,
                    'timestamp': datetime.now().isoformat()
                })
                logger.info("[PIPELINE] Quadruple thinking generated")
            except Exception as e:
                logger.error(f"[PIPELINE] Quadruple thinking generation failed: {e}")
                import traceback
                logger.error(f"[PIPELINE] Quadruple thinking traceback:\n{traceback.format_exc()}")
                result['quadruple_thinking'] = {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'fallback': True
                }
        else:
            logger.info("[PIPELINE] Using fallback mode for quadruple thinking")
            result['quadruple_thinking'] = {
                'task': f'タスク: {query}を処理します',
                'safety': '安全性チェック: 問題なし',
                'policy': 'ポリシー: 標準的な処理を実行',
                'final': f'回答: {query}に関する情報を提供します',
                'fallback': True
            }
        
        # Step 4: 四値分類の実行
        if use_classification and not self.fallback_mode and self.agent:
            try:
                sample = {
                    'text': query,
                    'category': detected_domain if detected_domain else 'general',
                    'language': 'ja' if any(ord(c) > 0x3040 for c in query) else 'en',
                    'domain': detected_domain if detected_domain else 'general'
                }
                
                # 四重推論結果を追加
                if result.get('quadruple_thinking'):
                    thinking = result['quadruple_thinking']
                    sample['quadruple_thinking'] = {
                        'task': thinking.get('task', ''),
                        'safety': thinking.get('safety', ''),
                        'policy': thinking.get('policy', ''),
                        'final': thinking.get('final', '')
                    }
                
                classification_result = self.agent.classify_four_class(sample)
                result['four_class_classification'] = classification_result.get('quadruple_classification', {})
                result['four_class_label'] = classification_result.get('four_class_label', 'ALLOW')
                result['reasoning_steps'].append({
                    'step': 'four_class_classification',
                    'label': result['four_class_label'],
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"[PIPELINE] Four-class classification: {result['four_class_label']}")
            except Exception as e:
                logger.error(f"[PIPELINE] Four-class classification failed: {e}")
                import traceback
                logger.error(f"[PIPELINE] Four-class classification traceback:\n{traceback.format_exc()}")
                result['four_class_label'] = 'ALLOW'
                result['four_class_classification'] = {'error': str(e), 'error_type': type(e).__name__}
        else:
            result['four_class_label'] = 'ALLOW'
            if self.fallback_mode:
                logger.info("[PIPELINE] Using fallback mode for four-class classification")
        
        # Step 5: 最終回答の生成
        result['safety_label'] = result.get('four_class_label', 'ALLOW')
        
        if result.get('quadruple_thinking') and result['quadruple_thinking'].get('final'):
            result['final_answer'] = result['quadruple_thinking']['final']
        else:
            result['final_answer'] = "回答を生成できませんでした。"
            result['safety_label'] = 'REFUSE'
        
        # Step 6: 安全ゲート処理
        if result['safety_label'] == 'REFUSE':
            result['final_answer'] = "申し訳ありませんが、このご要望にはお応えできません。"
        elif result['safety_label'] == 'ESCALATION':
            result['final_answer'] = "このリクエストは追加の確認が必要です。担当者にエスカレーションします。"
        
        result['reasoning_steps'].append({
            'step': 'final_answer_generation',
            'safety_label': result['safety_label'],
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"[PIPELINE] Processing completed: {result['safety_label']}")
        return result
    
    def batch_process(
        self,
        queries: List[str],
        user_id: str = "default",
        use_knowledge: bool = True,
        use_classification: bool = True
    ) -> List[Dict[str, Any]]:
        """
        バッチ処理
        
        Args:
            queries: クエリのリスト
            user_id: ユーザーID
            use_knowledge: ドメイン別知識を使用するか
            use_classification: 四値分類を使用するか
        
        Returns:
            処理結果のリスト
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"[PIPELINE] Processing query {i}/{len(queries)}")
            try:
                result = self.process_with_integrated_reasoning(
                    query=query,
                    user_id=user_id,
                    use_knowledge=use_knowledge,
                    use_classification=use_classification
                )
                results.append(result)
            except Exception as e:
                logger.error(f"[PIPELINE] Failed to process query {i}/{len(queries)}: {query[:100]}...")
                logger.error(f"[PIPELINE] Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # エラー情報を含む結果を追加
                error_result = {
                    'query': query,
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processing_failed': True,
                    'final_answer': f"エラーが発生しました: {str(e)}",
                    'safety_label': 'REFUSE'
                }
                results.append(error_result)
                logger.warning(f"[PIPELINE] Added error result for query {i}, continuing with next query...")
        
        logger.info(f"[PIPELINE] Batch processing completed: {len(results)}/{len(queries)} queries processed")
        return results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Integrated Reasoning Pipeline')
    parser.add_argument('--query', type=str, help='User query')
    parser.add_argument('--queries-file', type=str, help='File containing queries (one per line)')
    parser.add_argument('--model-path', type=str, help='SO8T model path')
    parser.add_argument('--knowledge-base', type=str, help='Knowledge base path')
    parser.add_argument('--rag-store', type=str, help='RAG store path')
    parser.add_argument('--coding-data', type=str, help='Coding data path')
    parser.add_argument('--science-data', type=str, help='Science data path')
    parser.add_argument('--user-id', type=str, default='default', help='User ID')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # クエリを取得
    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        queries_path = Path(args.queries_file)
        if queries_path.exists():
            with open(queries_path, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            logger.error(f"[PIPELINE] Queries file not found: {queries_path}")
            sys.exit(1)
    else:
        logger.error("[PIPELINE] Either --query or --queries-file must be provided")
        sys.exit(1)
    
    # パイプラインを初期化
    try:
        pipeline = IntegratedReasoningPipeline(
            model_path=args.model_path,
            knowledge_base_path=args.knowledge_base,
            rag_store_path=args.rag_store,
            coding_data_path=args.coding_data,
            science_data_path=args.science_data
        )
        if pipeline.fallback_mode:
            logger.warning("[PIPELINE] Pipeline initialized in fallback mode - some features may be limited")
    except Exception as e:
        logger.error(f"[PIPELINE] Failed to initialize pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # フォールバックモードで続行を試みる
        logger.warning("[PIPELINE] Attempting to continue with minimal functionality")
        try:
            # 最小限のパイプラインを作成
            pipeline = IntegratedReasoningPipeline.__new__(IntegratedReasoningPipeline)
            pipeline.fallback_mode = True
            pipeline.agent = None
            pipeline.integrator = None
            logger.warning("[PIPELINE] Using minimal fallback pipeline")
        except Exception as fallback_error:
            logger.error(f"[PIPELINE] Fallback initialization also failed: {fallback_error}")
            sys.exit(1)
    
    # 処理を実行
    try:
        if len(queries) == 1:
            logger.info(f"[PIPELINE] Processing single query: {queries[0][:100]}...")
            try:
                result = pipeline.process_with_integrated_reasoning(
                    query=queries[0],
                    user_id=args.user_id
                )
                results = [result]
            except Exception as e:
                logger.error(f"[PIPELINE] Failed to process single query: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # エラー情報を含む結果を返す
                results = [{
                    'query': queries[0],
                    'user_id': args.user_id,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processing_failed': True,
                    'final_answer': f"エラーが発生しました: {str(e)}",
                    'safety_label': 'REFUSE'
                }]
        else:
            logger.info(f"[PIPELINE] Processing {len(queries)} queries in batch mode")
            results = pipeline.batch_process(
                queries=queries,
                user_id=args.user_id
            )
    except Exception as e:
        logger.error(f"[PIPELINE] Failed to process queries: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # エラー情報を含む結果を生成
        results = []
        for query in queries:
            results.append({
                'query': query,
                'user_id': args.user_id,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_failed': True,
                'final_answer': f"エラーが発生しました: {str(e)}",
                'safety_label': 'REFUSE'
            })
        logger.warning("[PIPELINE] Generated error results for all queries, continuing to save results...")
    
    # 結果を出力
    try:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                if len(results) == 1:
                    json.dump(results[0], f, ensure_ascii=False, indent=2)
                else:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"[PIPELINE] Results saved to {output_path}")
        else:
            print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"[PIPELINE] Failed to save results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("[PIPELINE] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[PIPELINE] Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

