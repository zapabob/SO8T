#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
CUDA分散処理統合

SO8Tモデル推論をCUDAにオフロードし、データ処理（画像解析、テキスト処理など）を
CUDAに分散処理する機能を提供します。

Usage:
    from scripts.utils.cuda_distributed_processor import CUDADistributedProcessor
    
    processor = CUDADistributedProcessor(device_id=0, batch_size=32)
    results = await processor.process_batch_so8t_inference(queries)
    processed_data = await processor.process_data_cuda(data_list)
"""

import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import queue
import threading

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# PyTorchインポート
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("[ERROR] PyTorch not installed. Install with: pip install torch")

# CUDA利用可能性チェック
CUDA_AVAILABLE = False
if TORCH_AVAILABLE:
    CUDA_AVAILABLE = torch.cuda.is_available()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cuda_distributed_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CUDADistributedProcessor:
    """CUDA分散処理クラス"""
    
    def __init__(
        self,
        device_id: int = 0,
        batch_size: int = 32,
        max_memory_fraction: float = 0.8,
        num_workers: int = 4
    ):
        """
        初期化
        
        Args:
            device_id: CUDAデバイスID
            batch_size: バッチサイズ
            max_memory_fraction: 最大メモリ使用率（0.0-1.0）
            num_workers: ワーカー数
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        if not CUDA_AVAILABLE:
            logger.warning("[CUDA] CUDA not available, using CPU")
            self.device = torch.device("cpu")
        else:
            if device_id >= torch.cuda.device_count():
                logger.warning(f"[CUDA] Device {device_id} not available, using device 0")
                device_id = 0
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id)
            
            # GPUメモリ制限を設定
            if max_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(max_memory_fraction, device_id)
        
        self.device_id = device_id
        self.batch_size = batch_size
        self.max_memory_fraction = max_memory_fraction
        self.num_workers = num_workers
        
        # バッチキュー
        self.batch_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()
        
        # SO8Tモデル（必要に応じてロード）
        self.so8t_model = None
        self.so8t_tokenizer = None
        
        logger.info("="*80)
        logger.info("CUDA Distributed Processor Initialized")
        logger.info("="*80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Device ID: {self.device_id}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Max memory fraction: {self.max_memory_fraction}")
        logger.info(f"CUDA available: {CUDA_AVAILABLE}")
        
        if CUDA_AVAILABLE:
            device_props = torch.cuda.get_device_properties(device_id)
            total_memory_gb = device_props.total_memory / (1024**3)
            logger.info(f"GPU: {device_props.name}")
            logger.info(f"Total memory: {total_memory_gb:.2f} GB")
            logger.info(f"Compute capability: {device_props.major}.{device_props.minor}")
    
    def load_so8t_model(self, model_path: Optional[str] = None) -> bool:
        """
        SO8Tモデルをロード
        
        Args:
            model_path: モデルパス（Noneの場合は自動検出）
        
        Returns:
            success: 成功フラグ
        """
        try:
            logger.info("[SO8T] Loading SO8T model for CUDA inference...")
            
            # SO8Tモデルローダーを使用
            try:
                from scripts.utils.so8t_model_loader import load_so8t_model
                model, tokenizer, success = load_so8t_model(
                    model_path=model_path,
                    device="cuda" if CUDA_AVAILABLE else "cpu",
                    use_quadruple_thinking=True,
                    use_redacted_tokens=False,
                    fallback_to_default=True
                )
                
                if success and model is not None and tokenizer is not None:
                    self.so8t_model = model
                    self.so8t_tokenizer = tokenizer
                    
                    # モデルをCUDAに移動
                    if CUDA_AVAILABLE:
                        self.so8t_model = self.so8t_model.to(self.device)
                        logger.info(f"[SO8T] Model moved to {self.device}")
                    
                    logger.info("[OK] SO8T model loaded successfully")
                    return True
                else:
                    logger.warning("[SO8T] Failed to load model using loader")
                    return False
                    
            except ImportError as e:
                logger.warning(f"[SO8T] SO8T model loader not available: {e}")
                return False
                
        except Exception as e:
            logger.error(f"[SO8T] Failed to load SO8T model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def process_batch_so8t_inference(
        self,
        queries: List[str],
        max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """
        バッチでSO8T推論を実行
        
        Args:
            queries: クエリのリスト
            max_length: 最大シーケンス長
        
        Returns:
            results: 推論結果のリスト
        """
        if self.so8t_model is None:
            logger.warning("[SO8T] Model not loaded, loading now...")
            success = self.load_so8t_model()
            if not success:
                logger.error("[SO8T] Failed to load model")
                return []
        
        if self.so8t_model is None or self.so8t_tokenizer is None:
            logger.error("[SO8T] Model or tokenizer not available")
            return []
        
        try:
            logger.info(f"[CUDA] Processing {len(queries)} queries in batches...")
            
            results = []
            
            # バッチ処理
            for i in range(0, len(queries), self.batch_size):
                batch_queries = queries[i:i + self.batch_size]
                batch_results = await self._process_batch_inference(batch_queries, max_length)
                results.extend(batch_results)
                
                logger.info(f"[CUDA] Processed batch {i // self.batch_size + 1}/{(len(queries) + self.batch_size - 1) // self.batch_size}")
            
            logger.info(f"[OK] Processed {len(results)} queries")
            return results
            
        except Exception as e:
            logger.error(f"[CUDA] Batch inference failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _process_batch_inference(
        self,
        queries: List[str],
        max_length: int
    ) -> List[Dict[str, Any]]:
        """
        バッチ推論を実行
        
        Args:
            queries: クエリのリスト
            max_length: 最大シーケンス長
        
        Returns:
            results: 推論結果のリスト
        """
        try:
            # トークナイズ
            inputs = self.so8t_tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # CUDAに移動
            if CUDA_AVAILABLE:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推論実行
            self.so8t_model.eval()
            with torch.no_grad():
                outputs = self.so8t_model(**inputs)
            
            # 結果をCPUに移動して処理
            if CUDA_AVAILABLE:
                logits = outputs.logits.cpu() if hasattr(outputs, 'logits') else None
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else None
            
            # 結果を構築
            results = []
            for i, query in enumerate(queries):
                result = {
                    'query': query,
                    'logits': logits[i].tolist() if logits is not None else None,
                    'processed_at': datetime.now().isoformat(),
                    'device': str(self.device)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"[CUDA] Inference failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def process_data_cuda(
        self,
        data_list: List[Dict[str, Any]],
        processing_type: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        データをCUDAで処理
        
        Args:
            data_list: 処理するデータのリスト
            processing_type: 処理タイプ（"text", "image", "embedding"など）
        
        Returns:
            processed_data: 処理済みデータのリスト
        """
        if not CUDA_AVAILABLE:
            logger.warning("[CUDA] CUDA not available, using CPU")
            return await self._process_data_cpu(data_list, processing_type)
        
        try:
            logger.info(f"[CUDA] Processing {len(data_list)} data items (type: {processing_type})...")
            
            processed_data = []
            
            # バッチ処理
            for i in range(0, len(data_list), self.batch_size):
                batch_data = data_list[i:i + self.batch_size]
                
                if processing_type == "text":
                    batch_processed = await self._process_text_batch_cuda(batch_data)
                elif processing_type == "image":
                    batch_processed = await self._process_image_batch_cuda(batch_data)
                elif processing_type == "embedding":
                    batch_processed = await self._process_embedding_batch_cuda(batch_data)
                else:
                    logger.warning(f"[CUDA] Unknown processing type: {processing_type}")
                    batch_processed = batch_data
                
                processed_data.extend(batch_processed)
                
                logger.info(f"[CUDA] Processed batch {i // self.batch_size + 1}/{(len(data_list) + self.batch_size - 1) // self.batch_size}")
            
            logger.info(f"[OK] Processed {len(processed_data)} data items")
            return processed_data
            
        except Exception as e:
            logger.error(f"[CUDA] Data processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return data_list
    
    async def _process_text_batch_cuda(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        テキストデータをCUDAでバッチ処理
        
        Args:
            data_list: テキストデータのリスト
        
        Returns:
            processed_data: 処理済みデータのリスト
        """
        try:
            # テキストを抽出
            texts = [item.get('text', '') for item in data_list]
            
            # トークナイズ（SO8Tトークナイザーが利用可能な場合）
            if self.so8t_tokenizer is not None:
                inputs = self.so8t_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # CUDAに移動
                if CUDA_AVAILABLE:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 処理結果を構築
                processed_data = []
                for i, item in enumerate(data_list):
                    processed_item = item.copy()
                    processed_item['tokenized'] = inputs['input_ids'][i].cpu().tolist() if CUDA_AVAILABLE else inputs['input_ids'][i].tolist()
                    processed_item['processed_at'] = datetime.now().isoformat()
                    processed_item['device'] = str(self.device)
                    processed_data.append(processed_item)
                
                return processed_data
            else:
                # トークナイザーが無い場合はそのまま返す
                return data_list
                
        except Exception as e:
            logger.error(f"[CUDA] Text processing failed: {e}")
            return data_list
    
    async def _process_image_batch_cuda(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        画像データをCUDAでバッチ処理
        
        Args:
            data_list: 画像データのリスト
        
        Returns:
            processed_data: 処理済みデータのリスト
        """
        # 画像処理は将来的に実装（現在はそのまま返す）
        logger.warning("[CUDA] Image processing not yet implemented")
        return data_list
    
    async def _process_embedding_batch_cuda(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        埋め込みデータをCUDAでバッチ処理
        
        Args:
            data_list: 埋め込みデータのリスト
        
        Returns:
            processed_data: 処理済みデータのリスト
        """
        try:
            # 埋め込みベクトルを抽出
            embeddings = []
            for item in data_list:
                if 'embedding' in item:
                    embeddings.append(item['embedding'])
                else:
                    embeddings.append(None)
            
            # CUDAで処理（例: 正規化、次元削減など）
            if embeddings and all(e is not None for e in embeddings):
                # テンソルに変換
                embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
                
                if CUDA_AVAILABLE:
                    embedding_tensor = embedding_tensor.to(self.device)
                    
                    # 正規化
                    embedding_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
                    
                    # CPUに戻す
                    embedding_tensor = embedding_tensor.cpu()
                else:
                    embedding_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
                
                # 処理結果を構築
                processed_data = []
                for i, item in enumerate(data_list):
                    processed_item = item.copy()
                    processed_item['embedding'] = embedding_tensor[i].tolist()
                    processed_item['processed_at'] = datetime.now().isoformat()
                    processed_item['device'] = str(self.device)
                    processed_data.append(processed_item)
                
                return processed_data
            else:
                return data_list
                
        except Exception as e:
            logger.error(f"[CUDA] Embedding processing failed: {e}")
            return data_list
    
    async def _process_data_cpu(
        self,
        data_list: List[Dict[str, Any]],
        processing_type: str
    ) -> List[Dict[str, Any]]:
        """
        CPUでデータを処理（フォールバック）
        
        Args:
            data_list: 処理するデータのリスト
            processing_type: 処理タイプ
        
        Returns:
            processed_data: 処理済みデータのリスト
        """
        logger.info(f"[CPU] Processing {len(data_list)} data items on CPU...")
        # CPU処理は簡易実装
        processed_data = []
        for item in data_list:
            processed_item = item.copy()
            processed_item['processed_at'] = datetime.now().isoformat()
            processed_item['device'] = 'cpu'
            processed_data.append(processed_item)
        return processed_data
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        GPUメモリ情報を取得
        
        Returns:
            memory_info: GPUメモリ情報
        """
        if not CUDA_AVAILABLE:
            return {'available': False}
        
        try:
            memory_allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)
            device_props = torch.cuda.get_device_properties(self.device_id)
            memory_total = device_props.total_memory / (1024**3)
            
            return {
                'available': True,
                'device_id': self.device_id,
                'device_name': device_props.name,
                'memory_total_gb': memory_total,
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'memory_free_gb': memory_total - memory_reserved,
                'memory_usage_percent': (memory_reserved / memory_total) * 100
            }
        except Exception as e:
            logger.error(f"[CUDA] Failed to get memory info: {e}")
            return {'available': False, 'error': str(e)}


async def main():
    """メイン関数（テスト用）"""
    processor = CUDADistributedProcessor(device_id=0, batch_size=8)
    
    try:
        # GPUメモリ情報を表示
        memory_info = processor.get_gpu_memory_info()
        logger.info(f"[CUDA] Memory info: {memory_info}")
        
        # SO8Tモデルをロード
        success = processor.load_so8t_model()
        if success:
            # テストクエリ
            test_queries = [
                "What is Python?",
                "Explain machine learning",
                "How does SO8T work?"
            ]
            
            # バッチ推論を実行
            results = await processor.process_batch_so8t_inference(test_queries)
            logger.info(f"[OK] Processed {len(results)} queries")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())



































































































































