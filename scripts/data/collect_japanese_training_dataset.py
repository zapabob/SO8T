#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語学習用データセット収集スクリプト（Cursorブラウザ使用）

Wikipedia日本語、CC-100日本語、mc4日本語から学習用データセットを収集します。
Cursorのブラウザ機能（MCP Chrome DevTools）を使用してスクレイピングします。

Usage:
    python scripts/data/collect_japanese_training_dataset.py --output D:/webdataset/japanese_training_dataset
"""

import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import Counter
import time

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_japanese_training_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JapaneseTrainingDatasetCollector:
    """日本語学習用データセット収集クラス（Cursorブラウザ使用）"""
    
    def __init__(
        self,
        output_dir: Path,
        use_mcp_chrome_devtools: bool = True,
        num_tabs: int = 10,
        delay_per_action: float = 2.0
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            use_mcp_chrome_devtools: MCP Chrome DevToolsを使用するか
            num_tabs: 並列タブ数
            delay_per_action: アクション間の遅延（秒）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_mcp_chrome_devtools = use_mcp_chrome_devtools
        self.num_tabs = num_tabs
        self.delay_per_action = delay_per_action
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_samples: List[Dict] = []
        self.visited_urls: Set[str] = set()
        
        # MCP Chrome DevToolsラッパーを初期化
        self.mcp_wrapper = None
        if self.use_mcp_chrome_devtools:
            try:
                from scripts.utils.mcp_chrome_devtools_wrapper import MCPChromeDevTools
                import yaml
                
                config_path = PROJECT_ROOT / "configs" / "unified_master_pipeline_config.yaml"
                mcp_config = {}
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        phase1_config = config.get('phase1_parallel_scraping', {})
                        mcp_config = phase1_config.get('mcp_server', {})
                
                if mcp_config.get('enabled', True):
                    self.mcp_wrapper = MCPChromeDevTools(
                        transport=mcp_config.get('transport', 'stdio'),
                        command=mcp_config.get('command', 'npx'),
                        args=mcp_config.get('args', ['-y', '@modelcontextprotocol/server-chrome-devtools']),
                        url=mcp_config.get('url'),
                        timeout=mcp_config.get('timeout', 30000)
                    )
                    logger.info("[MCP] Chrome DevTools wrapper initialized")
                else:
                    logger.warning("[MCP] MCP server disabled, falling back to requests")
                    self.use_mcp_chrome_devtools = False
            except Exception as e:
                logger.warning(f"[MCP] Failed to initialize MCP Chrome DevTools: {e}")
                logger.warning("[MCP] Falling back to requests")
                self.use_mcp_chrome_devtools = False
        
        # データソース設定
        self.data_sources = {
            'wikipedia_ja': {
                'name': 'Wikipedia日本語',
                'target_samples': 40000,
                'base_url': 'https://ja.wikipedia.org',
                'api_url': 'https://ja.wikipedia.org/api/rest_v1/page/random/summary',
                'search_url': 'https://ja.wikipedia.org/w/api.php',
                'enabled': True
            },
            'cc100_ja': {
                'name': 'CC-100日本語',
                'target_samples': 30000,
                'base_url': 'https://data.statmt.org/cc-100',
                'enabled': True
            },
            'mc4_ja': {
                'name': 'mc4日本語',
                'target_samples': 30000,
                'base_url': 'https://huggingface.co/datasets/mc4',
                'enabled': True
            }
        }
        
        logger.info("="*80)
        logger.info("Japanese Training Dataset Collector Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"MCP Chrome DevTools: {self.use_mcp_chrome_devtools}")
        logger.info(f"Parallel tabs: {self.num_tabs}")
    
    async def collect_from_wikipedia(self) -> List[Dict]:
        """Wikipedia日本語からデータを収集"""
        logger.info("="*80)
        logger.info(f"[Wikipedia] Starting collection: {self.data_sources['wikipedia_ja']['target_samples']} samples")
        logger.info("="*80)
        
        samples = []
        target_samples = self.data_sources['wikipedia_ja']['target_samples']
        
        # WikipediaはAPIが直接利用可能なので、requestsを使用（高速）
        # MCP Chrome DevToolsは必要に応じて使用可能
        samples.extend(await self._collect_wikipedia_fallback(target_samples))
        
        logger.info(f"[Wikipedia] Collected {len(samples)} samples")
        return samples
    
    async def _collect_wikipedia_fallback(self, target_samples: int) -> List[Dict]:
        """Wikipedia収集（requestsフォールバック）"""
        samples = []
        try:
            import requests
            
            collected = 0
            while collected < target_samples:
                try:
                    # Wikipedia APIからランダムページを取得
                    api_url = "https://ja.wikipedia.org/api/rest_v1/page/random/summary"
                    response = requests.get(api_url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'title' in data and 'extract' in data:
                        sample = {
                            'text': data['extract'],
                            'title': data['title'],
                            'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            'source': 'wikipedia_ja',
                            'source_name': self.data_sources['wikipedia_ja']['name'],
                            'language': 'ja',
                            'collected_at': datetime.now().isoformat(),
                            'word_count': len(data['extract'].split())
                        }
                        
                        if sample['url'] not in self.visited_urls and len(sample['text']) > 100:
                            samples.append(sample)
                            self.visited_urls.add(sample['url'])
                            collected += 1
                            
                            if collected % 100 == 0:
                                logger.info(f"[Wikipedia] Collected {collected}/{target_samples} samples")
                    
                    await asyncio.sleep(self.delay_per_action)
                except Exception as e:
                    logger.debug(f"[Wikipedia] Request failed: {e}")
                    await asyncio.sleep(self.delay_per_action * 2)
                    continue
        except ImportError:
            logger.error("[Wikipedia] requests library not available")
        
        return samples
    
    async def collect_from_cc100(self) -> List[Dict]:
        """CC-100日本語からデータを収集"""
        logger.info("="*80)
        logger.info(f"[CC-100] Starting collection: {self.data_sources['cc100_ja']['target_samples']} samples")
        logger.info("="*80)
        
        samples = []
        target_samples = self.data_sources['cc100_ja']['target_samples']
        
        # CC-100はHugging Face datasetsライブラリを使用
        try:
            from datasets import load_dataset
            from tqdm import tqdm
            
            logger.info("[CC-100] Loading CC-100 Japanese dataset...")
            dataset = load_dataset("cc100", lang="ja", split="train", streaming=True, trust_remote_code=True)
            
            collected = 0
            for item in tqdm(dataset, desc="CC-100 Japanese", total=target_samples):
                if collected >= target_samples:
                    break
                
                text = item.get('text', '')
                if len(text) > 100:  # 最小テキスト長
                    sample = {
                        'text': text[:5000],  # 長すぎるテキストは切り詰め
                        'source': 'cc100_ja',
                        'source_name': self.data_sources['cc100_ja']['name'],
                        'language': 'ja',
                        'collected_at': datetime.now().isoformat(),
                        'word_count': len(text.split())
                    }
                    samples.append(sample)
                    collected += 1
                    
                    if collected % 1000 == 0:
                        logger.info(f"[CC-100] Collected {collected}/{target_samples} samples")
            
        except ImportError:
            logger.error("[CC-100] datasets library not available. Install with: pip install datasets tqdm")
        except Exception as e:
            logger.error(f"[CC-100] Collection failed: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"[CC-100] Collected {len(samples)} samples")
        return samples
    
    async def collect_from_mc4(self) -> List[Dict]:
        """mc4日本語からデータを収集"""
        logger.info("="*80)
        logger.info(f"[mc4] Starting collection: {self.data_sources['mc4_ja']['target_samples']} samples")
        logger.info("="*80)
        
        samples = []
        target_samples = self.data_sources['mc4_ja']['target_samples']
        
        # mc4はHugging Face datasetsライブラリを使用
        try:
            from datasets import load_dataset
            from tqdm import tqdm
            
            logger.info("[mc4] Loading mc4 Japanese dataset...")
            dataset = load_dataset("mc4", "ja", split="train", streaming=True, trust_remote_code=True)
            
            collected = 0
            for item in tqdm(dataset, desc="mc4 Japanese", total=target_samples):
                if collected >= target_samples:
                    break
                
                text = item.get('text', '')
                if len(text) > 100:  # 最小テキスト長
                    sample = {
                        'text': text[:5000],  # 長すぎるテキストは切り詰め
                        'url': item.get('url', ''),
                        'source': 'mc4_ja',
                        'source_name': self.data_sources['mc4_ja']['name'],
                        'language': 'ja',
                        'collected_at': datetime.now().isoformat(),
                        'word_count': len(text.split())
                    }
                    samples.append(sample)
                    collected += 1
                    
                    if collected % 1000 == 0:
                        logger.info(f"[mc4] Collected {collected}/{target_samples} samples")
            
        except ImportError:
            logger.error("[mc4] datasets library not available. Install with: pip install datasets tqdm")
        except Exception as e:
            logger.error(f"[mc4] Collection failed: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"[mc4] Collected {len(samples)} samples")
        return samples
    
    async def collect_all_sources(self) -> List[Dict]:
        """すべてのソースからデータを収集"""
        logger.info("="*80)
        logger.info("Starting collection from all sources")
        logger.info("="*80)
        
        all_samples = []
        
        # Wikipedia日本語
        if self.data_sources['wikipedia_ja']['enabled']:
            wikipedia_samples = await self.collect_from_wikipedia()
            all_samples.extend(wikipedia_samples)
        
        # CC-100日本語
        if self.data_sources['cc100_ja']['enabled']:
            cc100_samples = await self.collect_from_cc100()
            all_samples.extend(cc100_samples)
        
        # mc4日本語
        if self.data_sources['mc4_ja']['enabled']:
            mc4_samples = await self.collect_from_mc4()
            all_samples.extend(mc4_samples)
        
        logger.info("="*80)
        logger.info(f"Total collected: {len(all_samples)} samples")
        logger.info("="*80)
        
        return all_samples
    
    def save_dataset(self, samples: List[Dict], split_ratio: float = 0.9):
        """
        データセットを保存
        
        Args:
            samples: サンプルのリスト
            split_ratio: 訓練/検証の分割比率
        """
        logger.info(f"[SAVE] Saving {len(samples)} samples...")
        
        # ソース別の分布を確認
        source_dist = Counter(s.get('source', 'unknown') for s in samples)
        logger.info(f"[SAVE] Source distribution: {dict(source_dist)}")
        
        # 訓練/検証に分割
        import random
        random.shuffle(samples)
        split_idx = int(len(samples) * split_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # 訓練データを保存
        train_file = self.output_dir / f"japanese_training_train_{self.session_id}.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"[SAVE] Training data saved: {train_file} ({len(train_samples)} samples)")
        
        # 検証データを保存
        val_file = self.output_dir / f"japanese_training_val_{self.session_id}.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"[SAVE] Validation data saved: {val_file} ({len(val_samples)} samples)")
        
        # メタデータを保存
        metadata = {
            'total_samples': len(samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'source_distribution': dict(source_dist),
            'created_at': datetime.now().isoformat(),
            'session_id': self.session_id,
            'target_samples': {
                'wikipedia_ja': self.data_sources['wikipedia_ja']['target_samples'],
                'cc100_ja': self.data_sources['cc100_ja']['target_samples'],
                'mc4_ja': self.data_sources['mc4_ja']['target_samples']
            }
        }
        
        metadata_file = self.output_dir / f"metadata_{self.session_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"[SAVE] Metadata saved: {metadata_file}")


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Japanese Training Dataset Collector')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--use-mcp-chrome-devtools', action='store_true', default=True, help='Use MCP Chrome DevTools')
    parser.add_argument('--num-tabs', type=int, default=10, help='Number of parallel tabs')
    parser.add_argument('--delay-per-action', type=float, default=2.0, help='Delay between actions (seconds)')
    
    args = parser.parse_args()
    
    collector = JapaneseTrainingDatasetCollector(
        output_dir=args.output,
        use_mcp_chrome_devtools=args.use_mcp_chrome_devtools,
        num_tabs=args.num_tabs,
        delay_per_action=args.delay_per_action
    )
    
    all_samples = await collector.collect_all_sources()
    
    if all_samples:
        collector.save_dataset(all_samples)
        logger.info(f"[OK] Japanese training dataset collection completed: {len(all_samples)} total samples")
    else:
        logger.warning("[WARNING] No samples collected")


if __name__ == '__main__':
    asyncio.run(main())

