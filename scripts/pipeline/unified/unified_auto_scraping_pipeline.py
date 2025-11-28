#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T統制Webスクレイピング全自動パイプライン

これまで作成したすべてのスクレイピングスクリプトを統合して
全自動パイプライン処理として実行します。

Usage:
    python scripts/pipelines/unified_auto_scraping_pipeline.py --daemon
"""

import sys
import logging
import asyncio
import argparse
import signal
import random
import json
from pathlib import Path
from typing import Dict, Optional, List
import subprocess

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_auto_scraping_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# チェックポイント管理、データセット作成、ベクトルストア作成モジュールをインポート
try:
    from scripts.pipelines.checkpoint_manager import CheckpointManager
    from scripts.pipelines.dataset_creator import DatasetCreator
    from scripts.pipelines.vector_store_creation import RAGVectorStoreCreator, CoGKnowledgeGraphCreator
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    MODULES_AVAILABLE = False


class UnifiedAutoScrapingPipeline:
    """SO8T統制Webスクレイピング全自動パイプライン"""
    
    def __init__(
        self,
        output_dir: Path,
        config_file: Optional[Path] = None,
        daemon_mode: bool = False,
        auto_restart: bool = True,
        max_restarts: int = 10,
        restart_delay: float = 3600.0
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            config_file: 設定ファイルパス
            daemon_mode: デーモンモード（バックグラウンド実行）
            auto_restart: 自動再起動
            max_restarts: 最大再起動回数
            restart_delay: 再起動待機時間（秒）
        """
        self.output_dir = Path(output_dir)
        self.config_file = config_file
        self.daemon_mode = daemon_mode
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        
        self.restart_count = 0
        self.running = True
        
        # スクレイピングスクリプト定義
        self.scraping_scripts = [
            {
                'name': 'parallel_deep_research',
                'script': 'scripts/data/parallel_deep_research_scraping.py',
                'batch': 'scripts/data/run_parallel_deep_research_scraping.bat',
                'enabled': True,
                'priority': 1,
                'output_dir': self.output_dir / 'parallel_deep_research'
            },
            {
                'name': 'arxiv_open_access',
                'script': 'scripts/data/arxiv_open_access_scraping.py',
                'batch': 'scripts/data/run_arxiv_background_scraping.bat',
                'enabled': True,
                'priority': 2,
                'output_dir': self.output_dir / 'arxiv_open_access'
            },
            {
                'name': 'auto_background',
                'script': 'scripts/data/so8t_auto_background_scraping.py',
                'batch': 'scripts/data/run_so8t_auto_background_scraping.bat',
                'enabled': True,
                'priority': 3,
                'output_dir': self.output_dir / 'auto_background'
            }
        ]
        
        # データパイプライン処理設定
        self.data_pipeline_script = 'scripts/pipelines/web_scraping_data_pipeline.py'
        self.data_pipeline_enabled = True
        self.data_pipeline_output_dir = self.output_dir / 'cleaned'
        
        # チェックポイント管理設定
        checkpoint_dir = self.output_dir.parent / 'checkpoints' / 'unified_pipeline'
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=180.0,  # 3分間隔
            max_checkpoints=5,  # 5つローリングストック
            resume_on_startup=False  # 電源投入時に既収集データを読み込まない
        )
        
        # データセット作成設定
        self.dataset_output_dir = self.output_dir / 'datasets'
        
        # ベクトルストア作成設定
        self.vector_store_output_dir = self.output_dir / 'vector_stores'
        
        # 収集済みサンプル（チェックポイント用）
        self.collected_samples: List[Dict] = []
        self.visited_urls: Dict[str, bool] = {}
        self.collected_count = 0
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler)
        
        logger.info("="*80)
        logger.info("Unified Auto Scraping Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Daemon mode: {self.daemon_mode}")
        logger.info(f"Auto restart: {self.auto_restart}")
        logger.info(f"Scraping scripts: {len(self.scraping_scripts)}")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        logger.info(f"[SIGNAL] Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def load_config(self) -> Dict:
        """設定ファイルを読み込み"""
        if self.config_file and self.config_file.exists():
            import yaml
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"[CONFIG] Loaded config from: {self.config_file}")
            return config
        else:
            # デフォルト設定
            return {
                'output_dir': str(self.output_dir),
                'scraping_scripts': [
                    {
                        'name': script['name'],
                        'enabled': script['enabled'],
                        'priority': script['priority']
                    }
                    for script in self.scraping_scripts
                ]
            }
    
    async def run_scraping_script(self, script_info: Dict) -> bool:
        """スクレイピングスクリプトを実行"""
        script_name = script_info['name']
        batch_file = PROJECT_ROOT / script_info['batch']
        
        if not batch_file.exists():
            logger.warning(f"[PIPELINE] Batch file not found: {batch_file}")
            return False
        
        try:
            logger.info(f"[PIPELINE] Starting scraping script: {script_name}")
            
            # バックグラウンドで実行
            process = subprocess.Popen(
                [str(batch_file)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            # プロセスが完了するまで待機（タイムアウト: 24時間）
            try:
                process.wait(timeout=86400)
                return_code = process.returncode
                
                if return_code == 0:
                    logger.info(f"[PIPELINE] Scraping script completed: {script_name}")
                    return True
                else:
                    logger.warning(f"[PIPELINE] Scraping script failed: {script_name} (return code: {return_code})")
                    return False
            
            except subprocess.TimeoutExpired:
                logger.warning(f"[PIPELINE] Scraping script timeout: {script_name}")
                process.kill()
                return False
        
        except Exception as e:
            logger.error(f"[PIPELINE] Failed to run script {script_name}: {e}", exc_info=True)
            return False
    
    async def run_data_pipeline(self) -> bool:
        """データパイプライン処理を実行（クレンジング・ラベル付け・四値分類）"""
        if not self.data_pipeline_enabled:
            logger.info("[DATA PIPELINE] Data pipeline disabled, skipping...")
            return True
        
        try:
            logger.info("[DATA PIPELINE] Starting data pipeline processing...")
            
            # データパイプラインスクリプトを実行
            pipeline_script = PROJECT_ROOT / self.data_pipeline_script
            
            if not pipeline_script.exists():
                logger.warning(f"[DATA PIPELINE] Pipeline script not found: {pipeline_script}")
                return False
            
            # 入力ディレクトリ（スクレイピング出力）を取得
            input_dir = self.output_dir / 'processed'
            input_dir.mkdir(parents=True, exist_ok=True)
            
            # 出力ディレクトリを作成
            self.data_pipeline_output_dir.mkdir(parents=True, exist_ok=True)
            
            # データパイプラインを実行
            cmd = [
                sys.executable,
                str(pipeline_script),
                '--input', str(input_dir),
                '--output', str(self.data_pipeline_output_dir),
                '--use-so8t',
                '--num-workers', '4',
                '--batch-size', '100'
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            # プロセスが完了するまで待機（タイムアウト: 12時間）
            try:
                process.wait(timeout=43200)
                return_code = process.returncode
                
                if return_code == 0:
                    logger.info("[DATA PIPELINE] Data pipeline processing completed successfully")
                    # チェックポイント保存
                    self._save_checkpoint('cleaning')
                    return True
                else:
                    logger.warning(f"[DATA PIPELINE] Data pipeline processing failed (return code: {return_code})")
                    # エラー出力をログに記録
                    stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                    if stderr_output:
                        logger.error(f"[DATA PIPELINE] Error output: {stderr_output[:1000]}")
                    return False
            
            except subprocess.TimeoutExpired:
                logger.warning("[DATA PIPELINE] Data pipeline processing timeout")
                process.kill()
                return False
        
        except Exception as e:
            logger.error(f"[DATA PIPELINE] Failed to run data pipeline: {e}", exc_info=True)
            return False
    
    async def run_dataset_creation(self) -> bool:
        """データセット作成を実行"""
        try:
            logger.info("[DATASET] Starting dataset creation...")
            
            if not MODULES_AVAILABLE:
                logger.warning("[DATASET] Dataset creator module not available, skipping...")
                return False
            
            # クレンジング済みデータを読み込み
            cleaned_samples = []
            for jsonl_file in self.data_pipeline_output_dir.glob("*.jsonl"):
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            sample = json.loads(line)
                            cleaned_samples.append(sample)
                        except json.JSONDecodeError:
                            continue
            
            if not cleaned_samples:
                logger.warning("[DATASET] No cleaned samples found, skipping dataset creation...")
                return False
            
            # データセット作成
            creator = DatasetCreator(output_dir=self.dataset_output_dir)
            dataset_file = creator.create_dataset(
                samples=cleaned_samples,
                dataset_name="so8t_dataset"
            )
            
            logger.info(f"[OK] Dataset created: {dataset_file.name}")
            
            # チェックポイント保存
            self._save_checkpoint('dataset')
            
            return True
        
        except Exception as e:
            logger.error(f"[DATASET] Failed to create dataset: {e}", exc_info=True)
            return False
    
    async def run_vector_store_creation(self) -> bool:
        """RAG/CoG用ベクトルストア作成を実行"""
        try:
            logger.info("[VECTOR STORE] Starting vector store creation...")
            
            if not MODULES_AVAILABLE:
                logger.warning("[VECTOR STORE] Vector store creator module not available, skipping...")
                return False
            
            # クレンジング済みデータを読み込み
            cleaned_samples = []
            for jsonl_file in self.data_pipeline_output_dir.glob("*.jsonl"):
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            sample = json.loads(line)
                            cleaned_samples.append(sample)
                        except json.JSONDecodeError:
                            continue
            
            if not cleaned_samples:
                logger.warning("[VECTOR STORE] No cleaned samples found, skipping vector store creation...")
                return False
            
            # RAG用ベクトルストア作成
            logger.info("[VECTOR STORE] Creating RAG vector store...")
            rag_creator = RAGVectorStoreCreator(
                output_dir=self.vector_store_output_dir,
                chunk_size=512,
                overlap=128
            )
            rag_chunks = rag_creator.chunk_documents(cleaned_samples)
            rag_creator.save_rag_data(rag_chunks)
            
            # CoG用ナレッジグラフ作成
            logger.info("[VECTOR STORE] Creating CoG knowledge graph...")
            cog_creator = CoGKnowledgeGraphCreator(output_dir=self.vector_store_output_dir)
            knowledge_graph = cog_creator.create_knowledge_graph(cleaned_samples)
            cog_creator.save_knowledge_graph(knowledge_graph)
            
            logger.info("[OK] Vector store creation completed successfully")
            
            # チェックポイント保存
            self._save_checkpoint('vector_store')
            
            return True
        
        except Exception as e:
            logger.error(f"[VECTOR STORE] Failed to create vector stores: {e}", exc_info=True)
            return False
    
    def _save_checkpoint(self, phase: str):
        """チェックポイントを保存"""
        try:
            self.checkpoint_manager.save_checkpoint(
                samples=self.collected_samples,
                visited_urls=self.visited_urls,
                collected_count=self.collected_count,
                phase=phase,
                additional_data={
                    'output_dir': str(self.output_dir),
                    'data_pipeline_output_dir': str(self.data_pipeline_output_dir),
                    'dataset_output_dir': str(self.dataset_output_dir),
                    'vector_store_output_dir': str(self.vector_store_output_dir)
                }
            )
        except Exception as e:
            logger.warning(f"[CHECKPOINT] Failed to save checkpoint: {e}")
    
    async def run_pipeline_session(self) -> bool:
        """パイプラインセッションを実行"""
        try:
            # 設定を読み込み
            config = self.load_config()
            
            # 有効なスクリプトを優先度順にソート
            enabled_scripts = [
                script for script in self.scraping_scripts
                if script['enabled'] and script['name'] in [s['name'] for s in config.get('scraping_scripts', []) if s.get('enabled', True)]
            ]
            enabled_scripts.sort(key=lambda x: x['priority'])
            
            logger.info(f"[SESSION] Starting pipeline session with {len(enabled_scripts)} scripts...")
            
            # Phase 1: 各スクレイピングスクリプトを順次実行
            scraping_success_count = 0
            for script_info in enabled_scripts:
                if not self.running:
                    break
                
                success = await self.run_scraping_script(script_info)
                
                if success:
                    scraping_success_count += 1
                else:
                    logger.warning(f"[SESSION] Script {script_info['name']} failed, continuing with next script...")
                
                # スクリプト間の待機
                if self.running:
                    await asyncio.sleep(random.uniform(60.0, 120.0))
            
            logger.info(f"[SESSION] Scraping phase completed: {scraping_success_count}/{len(enabled_scripts)} scripts succeeded")
            
            # チェックポイント保存（スクレイピングフェーズ）
            if scraping_success_count > 0:
                self._save_checkpoint('scraping')
            
            # Phase 2: データパイプライン処理（クレンジング・ラベル付け・四値分類）
            if scraping_success_count > 0:
                logger.info("[SESSION] Starting data pipeline processing phase...")
                data_pipeline_success = await self.run_data_pipeline()
                
                if data_pipeline_success:
                    logger.info("[SESSION] Data pipeline processing completed successfully")
                    
                    # Phase 3: データセット作成
                    logger.info("[SESSION] Starting dataset creation phase...")
                    dataset_success = await self.run_dataset_creation()
                    
                    if dataset_success:
                        logger.info("[SESSION] Dataset creation completed successfully")
                    else:
                        logger.warning("[SESSION] Dataset creation failed, but continuing...")
                    
                    # Phase 4: RAG/CoG用ベクトルストア作成
                    logger.info("[SESSION] Starting vector store creation phase...")
                    vector_store_success = await self.run_vector_store_creation()
                    
                    if vector_store_success:
                        logger.info("[SESSION] Vector store creation completed successfully")
                    else:
                        logger.warning("[SESSION] Vector store creation failed, but continuing...")
                else:
                    logger.warning("[SESSION] Data pipeline processing failed, skipping subsequent phases...")
            else:
                logger.warning("[SESSION] No scraping scripts succeeded, skipping data pipeline processing")
            
            logger.info("[SESSION] Pipeline session completed")
            return scraping_success_count > 0
            
        except Exception as e:
            logger.error(f"[SESSION] Pipeline session failed: {e}", exc_info=True)
            return False
    
    async def run_continuous(self):
        """連続実行ループ（チェックポイント管理付き）"""
        logger.info("[CONTINUOUS] Starting continuous pipeline loop...")
        
        # チェックポイントから復旧（resume_on_startupがFalseの場合はスキップ）
        checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint_data and self.checkpoint_manager.resume_on_startup:
            logger.info("[CONTINUOUS] Resuming from checkpoint...")
            self.collected_samples = checkpoint_data.get('samples', [])
            self.visited_urls = checkpoint_data.get('visited_urls', {})
            self.collected_count = checkpoint_data.get('collected_count', 0)
        else:
            logger.info("[CONTINUOUS] Starting fresh session (no checkpoint resume)")
        
        while self.running:
            try:
                # パイプラインセッションを実行（定期的にチェックポイントを保存）
                success = await self.run_pipeline_session()
                
                # セッション完了後にチェックポイント保存
                if success:
                    self._save_checkpoint('session_complete')
                
                if not success:
                    self.restart_count += 1
                    if self.restart_count >= self.max_restarts:
                        logger.error(f"[CONTINUOUS] Max restarts ({self.max_restarts}) reached, stopping...")
                        break
                    
                    if self.auto_restart:
                        logger.info(f"[CONTINUOUS] Restarting in {self.restart_delay} seconds... (attempt {self.restart_count}/{self.max_restarts})")
                        await asyncio.sleep(self.restart_delay)
                    else:
                        break
                else:
                    # 成功時は再起動カウントをリセット
                    self.restart_count = 0
                    
                    # 次のセッションまでの待機（24時間）
                    if self.running:
                        wait_time = 86400.0  # 24時間待機
                        logger.info(f"[CONTINUOUS] Waiting {wait_time/3600:.1f} hours before next session...")
                        await asyncio.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("[CONTINUOUS] Keyboard interrupt received, stopping...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"[CONTINUOUS] Unexpected error: {e}", exc_info=True)
                self.restart_count += 1
                if self.restart_count >= self.max_restarts:
                    logger.error(f"[CONTINUOUS] Max restarts reached, stopping...")
                    break
                
                if self.auto_restart:
                    logger.info(f"[CONTINUOUS] Restarting in {self.restart_delay} seconds...")
                    await asyncio.sleep(self.restart_delay)
                else:
                    break
        
        logger.info("[CONTINUOUS] Continuous pipeline loop stopped")
    
    def run_as_daemon(self):
        """デーモンとして実行（Windows対応）"""
        if self.daemon_mode:
            logger.info("[DAEMON] Running as daemon (background process)...")
        
        # 連続実行ループを開始
        asyncio.run(self.run_continuous())


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Unified Auto Scraping Pipeline")
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Configuration file path'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        default=False,
        help='Run as daemon (background process)'
    )
    parser.add_argument(
        '--auto-restart',
        action='store_true',
        default=True,
        help='Auto restart on failure'
    )
    parser.add_argument(
        '--max-restarts',
        type=int,
        default=10,
        help='Maximum restart attempts'
    )
    parser.add_argument(
        '--restart-delay',
        type=float,
        default=3600.0,
        help='Delay between restarts (seconds)'
    )
    
    args = parser.parse_args()
    
    # パイプライン作成
    pipeline = UnifiedAutoScrapingPipeline(
        output_dir=args.output,
        config_file=args.config,
        daemon_mode=args.daemon,
        auto_restart=args.auto_restart,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay
    )
    
    # 実行
    if args.daemon:
        pipeline.run_as_daemon()
    else:
        await pipeline.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())

