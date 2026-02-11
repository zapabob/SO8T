#!/usr/bin/env python3
"""
Hugging Face Dataset Discovery and Integration
HFに現存するデータセットを調査し、学習パイプラインに統合
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError as e:
    DATASETS_AVAILABLE = False
    print(f"[ERROR] datasets not installed: {e}")
    print("[INFO] Install with: pip install datasets")
    sys.exit(1)

try:
    from huggingface_hub import HfApi
    try:
        from huggingface_hub import DatasetFilter
    except ImportError:
        # DatasetFilterが利用できない場合はNoneとして扱う
        DatasetFilter = None
    HF_HUB_AVAILABLE = True
except ImportError as e:
    HF_HUB_AVAILABLE = False
    print(f"[ERROR] huggingface_hub not installed: {e}")
    print("[INFO] Install with: pip install huggingface_hub")
    print("[WARN] Dataset discovery will be limited without huggingface_hub")
    HfApi = None
    DatasetFilter = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFDatasetDiscoverer:
    """Hugging Faceデータセット発見と統合"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root
        
        self.config_dir = self.project_root / "config"
        if HF_HUB_AVAILABLE and HfApi:
            self.api = HfApi()
        else:
            self.api = None
            logger.warning("[WARN] Hugging Face Hub API not available. Dataset discovery will be limited.")
        
    def discover_popular_datasets(self, 
                                  task_types: List[str] = None,
                                  languages: List[str] = None,
                                  min_downloads: int = 1000,
                                  max_results: int = 100) -> List[Dict[str, Any]]:
        """人気のあるデータセットを発見"""
        logger.info("[DISCOVER] Discovering popular Hugging Face datasets...")
        
        if task_types is None:
            task_types = [
                "text-generation",
                "question-answering",
                "instruction-tuning",
                "reinforcement-learning-from-human-feedback",
                "mathematical-reasoning",
                "code-generation",
                "multilingual",
                "safety"
            ]
        
        if languages is None:
            languages = ["en", "ja"]
        
        discovered_datasets = []
        
        if not self.api:
            logger.error("[ERROR] Hugging Face Hub API not available. Install huggingface_hub.")
            return []
        
        try:
            # タスクタイプごとに検索
            for task_type in task_types:
                logger.info(f"[DISCOVER] Searching for {task_type} datasets...")
                
                try:
                    # データセットフィルタリング
                    if DatasetFilter:
                        filters = DatasetFilter(
                            task_categories=[task_type] if task_type != "multilingual" else None,
                            language=languages if task_type == "multilingual" else None,
                        )
                        filter_kwargs = {'filter': filters}
                    else:
                        # DatasetFilterが利用できない場合はキーワード検索
                        filter_kwargs = {}
                        logger.warning(f"[WARN] DatasetFilter not available, using basic search for {task_type}")
                    
                    # 人気順でソート
                    datasets = self.api.list_datasets(
                        **filter_kwargs,
                        sort="downloads",
                        direction=-1,
                        limit=max_results // len(task_types)
                    )
                    
                    for dataset_info in datasets:
                        if dataset_info.downloads and dataset_info.downloads >= min_downloads:
                            discovered_datasets.append({
                                'id': dataset_info.id,
                                'downloads': dataset_info.downloads,
                                'task_type': task_type,
                                'languages': languages if task_type == "multilingual" else ["en"],
                                'author': dataset_info.author if hasattr(dataset_info, 'author') else None,
                                'created_at': dataset_info.created_at.isoformat() if dataset_info.created_at else None,
                                'tags': dataset_info.tags if hasattr(dataset_info, 'tags') else [],
                            })
                            
                except Exception as e:
                    logger.warning(f"[DISCOVER] Failed to search {task_type}: {e}")
                    continue
            
            # ダウンロード数でソート
            discovered_datasets.sort(key=lambda x: x.get('downloads', 0), reverse=True)
            
            logger.info(f"[DISCOVER] Found {len(discovered_datasets)} popular datasets")
            return discovered_datasets[:max_results]
            
        except Exception as e:
            logger.error(f"[DISCOVER] Failed to discover datasets: {e}")
            return []
    
    def verify_dataset_availability(self, dataset_id: str) -> Dict[str, Any]:
        """データセットの利用可能性を検証"""
        logger.info(f"[VERIFY] Verifying dataset: {dataset_id}")
        
        try:
            # データセット情報を取得（APIが利用可能な場合）
            dataset_info = None
            if self.api:
                try:
                    dataset_info = self.api.dataset_info(dataset_id)
                except Exception as api_error:
                    logger.warning(f"[VERIFY] Could not get dataset info from API: {api_error}")
            
            # 実際に読み込んでみる（小さなサンプル、trust_remote_codeは非推奨のため削除）
            try:
                test_dataset = load_dataset(dataset_id, split='train[:1]')
                sample_count = len(test_dataset) if test_dataset else 0
                
                # splits情報を取得
                splits = ['train']
                if dataset_info and hasattr(dataset_info, 'splits'):
                    splits = list(dataset_info.splits.keys())
                
                return {
                    'id': dataset_id,
                    'available': True,
                    'splits': splits,
                    'sample_count': sample_count,
                    'size': getattr(dataset_info, 'download_size', None) if dataset_info else None,
                    'features': list(test_dataset.features.keys()) if test_dataset else [],
                    'error': None
                }
            except Exception as load_error:
                return {
                    'id': dataset_id,
                    'available': False,
                    'error': str(load_error),
                    'splits': [],
                    'sample_count': 0,
                    'features': []
                }
                
        except Exception as e:
            return {
                'id': dataset_id,
                'available': False,
                'error': str(e),
                'splits': [],
                'sample_count': 0,
                'features': []
            }
    
    def discover_and_verify_datasets(self, 
                                     task_types: List[str] = None,
                                     min_downloads: int = 1000,
                                     max_results: int = 50) -> List[Dict[str, Any]]:
        """データセットを発見し、利用可能性を検証"""
        logger.info("[DISCOVER] Starting dataset discovery and verification...")
        
        # 人気データセットを発見
        discovered = self.discover_popular_datasets(
            task_types=task_types,
            min_downloads=min_downloads,
            max_results=max_results
        )
        
        # 利用可能性を検証
        verified_datasets = []
        for dataset in discovered[:max_results]:  # 検証は最大50個まで
            logger.info(f"[VERIFY] Verifying {dataset['id']}...")
            verification = self.verify_dataset_availability(dataset['id'])
            
            if verification['available']:
                verified_datasets.append({
                    **dataset,
                    **verification
                })
                logger.info(f"[OK] {dataset['id']} is available ({verification['sample_count']} samples)")
            else:
                logger.warning(f"[SKIP] {dataset['id']} not available: {verification.get('error', 'Unknown error')}")
        
        logger.info(f"[COMPLETE] Verified {len(verified_datasets)} available datasets")
        return verified_datasets
    
    def update_dataset_config(self, verified_datasets: List[Dict[str, Any]], 
                             config_path: Optional[Path] = None):
        """データセット設定を更新"""
        if config_path is None:
            config_path = self.config_dir / "dataset.json"
        
        logger.info(f"[UPDATE] Updating dataset config: {config_path}")
        
        # 既存の設定を読み込み
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {
                "sources": [],
                "processing": {
                    "max_samples": 100000,
                    "chunk_size": 10000,
                    "max_length": 2048
                }
            }
        
        # 新しいデータセットを追加
        existing_sources = set(config.get('sources', []))
        new_sources = []
        
        for dataset in verified_datasets:
            hf_source = f"huggingface:{dataset['id']}"
            if hf_source not in existing_sources:
                new_sources.append(hf_source)
                existing_sources.add(hf_source)
        
        # 設定を更新
        config['sources'].extend(new_sources)
        
        # 重複を削除
        config['sources'] = list(dict.fromkeys(config['sources']))
        
        # バックアップを作成
        backup_path = config_path.with_suffix('.json.backup')
        if config_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)
            logger.info(f"[BACKUP] Created backup: {backup_path}")
        
        # 設定を保存
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[UPDATE] Added {len(new_sources)} new datasets to config")
        logger.info(f"[UPDATE] Total datasets: {len(config['sources'])}")
        
        return new_sources
    
    def generate_dataset_report(self, verified_datasets: List[Dict[str, Any]], 
                                report_path: Optional[Path] = None):
        """データセットレポートを生成"""
        if report_path is None:
            report_path = self.project_root / "_docs" / f"{datetime.now().strftime('%Y-%m-%d')}_HF_データセット調査レポート.md"
        
        logger.info(f"[REPORT] Generating dataset report: {report_path}")
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Hugging Face データセット調査レポート\n\n")
            f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**調査対象**: {len(verified_datasets)} データセット\n\n")
            
            f.write("## データセット一覧\n\n")
            f.write("| ID | ダウンロード数 | タスクタイプ | 利用可能 | サンプル数 | 特徴量 |\n")
            f.write("|----|--------------|------------|---------|----------|--------|\n")
            
            for dataset in verified_datasets:
                features_str = ', '.join(dataset.get('features', [])[:3])
                if len(dataset.get('features', [])) > 3:
                    features_str += "..."
                
                f.write(f"| {dataset['id']} | {dataset.get('downloads', 'N/A'):,} | "
                       f"{dataset.get('task_type', 'N/A')} | "
                       f"{'✅' if dataset.get('available') else '❌'} | "
                       f"{dataset.get('sample_count', 0)} | "
                       f"{features_str} |\n")
            
            f.write("\n## タスクタイプ別統計\n\n")
            
            # タスクタイプ別に集計
            task_stats = {}
            for dataset in verified_datasets:
                task_type = dataset.get('task_type', 'unknown')
                if task_type not in task_stats:
                    task_stats[task_type] = {'count': 0, 'total_downloads': 0}
                task_stats[task_type]['count'] += 1
                task_stats[task_type]['total_downloads'] += dataset.get('downloads', 0)
            
            for task_type, stats in sorted(task_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                f.write(f"- **{task_type}**: {stats['count']} データセット "
                       f"(合計ダウンロード数: {stats['total_downloads']:,})\n")
        
        logger.info(f"[REPORT] Report saved: {report_path}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover and integrate Hugging Face datasets')
    parser.add_argument('--task-types', nargs='+', 
                       default=['text-generation', 'question-answering', 'instruction-tuning'],
                       help='Task types to search for')
    parser.add_argument('--min-downloads', type=int, default=1000,
                       help='Minimum download count')
    parser.add_argument('--max-results', type=int, default=50,
                       help='Maximum number of datasets to discover')
    parser.add_argument('--update-config', action='store_true',
                       help='Update dataset.json config file')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate dataset discovery report')
    
    args = parser.parse_args()
    
    discoverer = HFDatasetDiscoverer()
    
    # データセットを発見・検証
    verified_datasets = discoverer.discover_and_verify_datasets(
        task_types=args.task_types,
        min_downloads=args.min_downloads,
        max_results=args.max_results
    )
    
    # 設定を更新
    if args.update_config:
        new_sources = discoverer.update_dataset_config(verified_datasets)
        print(f"\n[SUCCESS] Added {len(new_sources)} new datasets to config")
    
    # レポートを生成
    if args.generate_report:
        discoverer.generate_dataset_report(verified_datasets)
        print(f"\n[SUCCESS] Generated dataset report")
    
    print(f"\n[COMPLETE] Discovered {len(verified_datasets)} available datasets")
    print(f"[INFO] Available datasets:")
    for dataset in verified_datasets[:10]:  # 最初の10個を表示
        print(f"  - {dataset['id']} ({dataset.get('downloads', 0):,} downloads)")


if __name__ == "__main__":
    main()
