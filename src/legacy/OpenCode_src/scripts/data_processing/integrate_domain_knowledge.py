#!/usr/bin/env python3
"""
ドメイン知識データ統合スクリプト
非構造データ（PDF、白書）とNSFW/薬物検知データセットをサンセットパイプラインに統合
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    from datasets import load_dataset, Dataset, concatenate_datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[ERROR] datasets not installed")
    print("[INFO] Install with: pip install datasets")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainKnowledgeIntegrator:
    """ドメイン知識データ統合クラス"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root
        
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data" / "domain_knowledge"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_unstructured_government_data(self) -> Optional[Dataset]:
        """非構造データ（官公庁白書）を読み込み"""
        logger.info("[LOAD] Loading unstructured government data...")
        
        unstructured_dir = self.project_root / "data" / "unstructured" / "cleaned"
        
        if not unstructured_dir.exists():
            logger.warning(f"[WARN] Unstructured data directory not found: {unstructured_dir}")
            logger.info("[INFO] Run process_unstructured_data.py first to generate data")
            return None
        
        # JSONLファイルを読み込み
        jsonl_files = list(unstructured_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            logger.warning("[WARN] No JSONL files found in unstructured data directory")
            return None
        
        all_data = []
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
            except Exception as e:
                logger.error(f"[ERROR] Failed to load {jsonl_file}: {e}")
                continue
        
        if not all_data:
            return None
        
        # Datasetに変換
        dataset = Dataset.from_list(all_data)
        logger.info(f"[LOAD] Loaded {len(dataset)} items from unstructured government data")
        
        return dataset
    
    def load_arxiv_biorxiv_papers(self) -> Optional[Dataset]:
        """Arxiv/BioRxiv論文データを読み込み（科学・数学推論能力向上用）"""
        logger.info("[LOAD] Loading Arxiv/BioRxiv papers for reasoning capability improvement...")
        
        arxiv_dir = self.project_root / "data" / "arxiv_biorxiv" / "cleaned"
        
        if not arxiv_dir.exists():
            logger.warning(f"[WARN] Arxiv/BioRxiv data directory not found: {arxiv_dir}")
            logger.info("[INFO] Run process_arxiv_biorxiv.py first to generate data")
            return None
        
        # JSONLファイルを読み込み
        jsonl_files = list(arxiv_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            logger.warning("[WARN] No JSONL files found in Arxiv/BioRxiv directory")
            return None
        
        # 最新のファイルを優先
        jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        all_data = []
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
            except Exception as e:
                logger.error(f"[ERROR] Failed to load {jsonl_file}: {e}")
                continue
        
        if not all_data:
            return None
        
        # Datasetに変換
        dataset = Dataset.from_list(all_data)
        logger.info(f"[LOAD] Loaded {len(dataset)} Arxiv/BioRxiv papers for reasoning capability improvement")
        
        return dataset
    
    def load_nsfw_detection_datasets(self) -> List[Dataset]:
        """NSFW検知用データセットを読み込み"""
        logger.info("[LOAD] Loading NSFW detection datasets...")
        
        nsfw_datasets = []
        
        # HuggingFace NSFW検知データセット（利用可能なもののみ）
        hf_nsfw_sources = [
            ("allenai/real-toxicity-prompts", 1000),
            ("HuggingFaceH4/no_robots", 1000),
        ]
        
        for dataset_id, max_samples in hf_nsfw_sources:
            try:
                logger.info(f"[LOAD] Loading {dataset_id}...")
                # 利用可能なsplitを確認
                try:
                    dataset = load_dataset(dataset_id, split=f'train[:{max_samples}]')
                except:
                    # split指定なしで試行
                    dataset = load_dataset(dataset_id)
                    if hasattr(dataset, 'keys'):
                        if 'train' in dataset:
                            dataset = dataset['train']
                        else:
                            dataset = list(dataset.values())[0]
                    if len(dataset) > max_samples:
                        dataset = dataset.select(range(max_samples))
                
                nsfw_datasets.append(dataset)
                logger.info(f"[OK] Loaded {len(dataset)} samples from {dataset_id}")
            except Exception as e:
                logger.warning(f"[WARN] Failed to load {dataset_id}: {e}")
                continue
        
        # Wikipedia日本語版（NSFW検知拒否目的）- 利用可能なデータセット
        wikipedia_sources = [
            ("llm-book/japanese-bookcorpus", 5000),
        ]
        
        for dataset_id, max_samples in wikipedia_sources:
            try:
                logger.info(f"[LOAD] Loading {dataset_id}...")
                try:
                    dataset = load_dataset(dataset_id, split=f'train[:{max_samples}]')
                except:
                    dataset = load_dataset(dataset_id)
                    if hasattr(dataset, 'keys'):
                        if 'train' in dataset:
                            dataset = dataset['train']
                        else:
                            dataset = list(dataset.values())[0]
                    if len(dataset) > max_samples:
                        dataset = dataset.select(range(max_samples))
                
                nsfw_datasets.append(dataset)
                logger.info(f"[OK] Loaded {len(dataset)} samples from {dataset_id}")
            except Exception as e:
                logger.warning(f"[WARN] Failed to load {dataset_id}: {e}")
                continue
        
        logger.info(f"[LOAD] Loaded {len(nsfw_datasets)} NSFW detection datasets")
        return nsfw_datasets
    
    def load_drug_detection_datasets(self) -> List[Dataset]:
        """薬物検知用データセットを読み込み"""
        logger.info("[LOAD] Loading drug detection datasets...")
        
        drug_datasets = []
        
        # 薬物検知用データセット（利用可能なもの）
        # 注意: 実際の薬物検知データセットは限定的
        # SafeRLHFなどの安全性データセットを代替として使用
        
        # 利用可能な安全性データセット
        safety_sources = [
            ("HuggingFaceH4/no_robots", 500),  # 安全性データセット
        ]
        
        for dataset_id, max_samples in safety_sources:
            try:
                logger.info(f"[LOAD] Loading {dataset_id} for drug detection training...")
                try:
                    dataset = load_dataset(dataset_id, split=f'train[:{max_samples}]')
                except:
                    dataset = load_dataset(dataset_id)
                    if hasattr(dataset, 'keys'):
                        if 'train' in dataset:
                            dataset = dataset['train']
                        else:
                            dataset = list(dataset.values())[0]
                    if len(dataset) > max_samples:
                        dataset = dataset.select(range(max_samples))
                
                drug_datasets.append(dataset)
                logger.info(f"[OK] Loaded {len(dataset)} samples from {dataset_id}")
            except Exception as e:
                logger.warning(f"[WARN] Failed to load {dataset_id}: {e}")
                continue
        
        logger.info(f"[LOAD] Loaded {len(drug_datasets)} drug detection datasets")
        return drug_datasets
    
    def normalize_dataset_features(self, dataset: Dataset) -> Optional[Dataset]:
        """データセットの特徴量を正規化（統一形式に変換）"""
        def normalize_item(example):
            # 統一された形式に変換
            normalized = {
                'text': '',
                'domain': example.get('domain', 'unknown'),
                'source': example.get('source', 'unknown'),
                'training_purpose': example.get('training_purpose', 'general'),
                'sanitized_at': example.get('sanitized_at', datetime.now().isoformat()),
                'metadata': {}
            }
            
            # テキストフィールドを統合
            text_fields = ['text', 'content', 'prompt', 'input', 'instruction', 'question', 'answer', 'title', 'full_text', 'summary']
            text_parts = []
            
            for field in text_fields:
                if field in example:
                    value = example[field]
                    if isinstance(value, str) and value.strip():
                        # full_textやsummaryは長いので、そのまま追加（プレフィックスなし）
                        if field in ['full_text', 'summary']:
                            text_parts.append(value)
                        else:
                            text_parts.append(f"{field}: {value}")
                    elif isinstance(value, list):
                        # リストの場合は結合
                        text_parts.append(f"{field}: {' '.join(str(v) for v in value)}")
            
            # その他のフィールドをメタデータに保存
            for key, value in example.items():
                if key not in text_fields and key not in ['domain', 'source', 'training_purpose', 'sanitized_at']:
                    try:
                        # JSON serializableな値のみ保存
                        json.dumps(value)
                        normalized['metadata'][key] = value
                    except:
                        normalized['metadata'][key] = str(value)
            
            # テキストを結合
            if text_parts:
                normalized['text'] = '\n'.join(text_parts)
            elif 'text' in example:
                normalized['text'] = str(example['text'])
            
            # テキストが空の場合はスキップ
            if not normalized['text'].strip():
                return None
            
            return normalized
        
        # 正規化（Noneを除外）
        normalized_list = []
        for item in dataset:
            normalized = normalize_item(item)
            if normalized:
                normalized_list.append(normalized)
        
        if not normalized_list:
            return None
        
        # 新しいDatasetを作成
        normalized_dataset = Dataset.from_list(normalized_list)
        return normalized_dataset
    
    def sanitize_for_training(self, dataset: Dataset, purpose: str = "detection") -> Dataset:
        """学習用データにサニタイズ（正規化済みデータセット用）"""
        logger.info(f"[SANITIZE] Sanitizing dataset for {purpose} training...")
        
        def sanitize_item(example):
            import re
            
            # 正規化済みデータセットは'text'フィールドを持つ
            if 'text' in example:
                text = example['text']
                
                if isinstance(text, str):
                    # 機密情報のマスキング
                    # メールアドレス
                    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
                    # 電話番号
                    text = re.sub(r'\b\d{2,4}-\d{2,4}-\d{2,4}\b', '[PHONE]', text)
                    
                    example['text'] = text
            
            # 目的を更新
            example['training_purpose'] = purpose
            example['sanitized_at'] = datetime.now().isoformat()
            
            return example
        
        sanitized = dataset.map(sanitize_item, desc="Sanitizing")
        logger.info(f"[SANITIZE] Sanitized {len(sanitized)} items")
        
        return sanitized
    
    def integrate_all_domain_data(self) -> Optional[Dataset]:
        """全てのドメイン知識データを統合"""
        logger.info("[INTEGRATE] Integrating all domain knowledge data...")
        
        all_datasets = []
        
        # 1. 非構造データ（官公庁白書）
        gov_data = self.load_unstructured_government_data()
        if gov_data:
            normalized = self.normalize_dataset_features(gov_data)
            if normalized:
                all_datasets.append(normalized)
        
        # 2. Arxiv/BioRxiv論文（科学・数学推論能力向上用）
        arxiv_data = self.load_arxiv_biorxiv_papers()
        if arxiv_data:
            # Arxiv論文は既に構造化されているため、推論能力向上用に整形
            normalized = self.normalize_dataset_features(arxiv_data)
            if normalized:
                # 推論能力向上用の目的を追加
                def add_reasoning_purpose(example):
                    example['training_purpose'] = 'scientific_mathematical_reasoning'
                    return example
                reasoning_dataset = normalized.map(add_reasoning_purpose, desc="Adding reasoning purpose")
                all_datasets.append(reasoning_dataset)
        
        # 3. NSFW検知データセット
        nsfw_datasets = self.load_nsfw_detection_datasets()
        for ds in nsfw_datasets:
            # まず正規化してからサニタイズ
            normalized = self.normalize_dataset_features(ds)
            if normalized:
                sanitized = self.sanitize_for_training(normalized, purpose="nsfw_detection")
                all_datasets.append(sanitized)
        
        # 4. 薬物検知データセット
        drug_datasets = self.load_drug_detection_datasets()
        for ds in drug_datasets:
            # まず正規化してからサニタイズ
            normalized = self.normalize_dataset_features(ds)
            if normalized:
                sanitized = self.sanitize_for_training(normalized, purpose="drug_detection")
                all_datasets.append(sanitized)
        
        if not all_datasets:
            logger.warning("[WARN] No datasets to integrate")
            return None
        
        # 統合（特徴量が統一されているため可能）
        try:
            integrated = concatenate_datasets(all_datasets)
            logger.info(f"[INTEGRATE] Integrated {len(integrated)} total items from {len(all_datasets)} datasets")
            return integrated
        except Exception as e:
            logger.error(f"[ERROR] Failed to concatenate datasets: {e}")
            # フォールバック: リストとして統合
            all_items = []
            for ds in all_datasets:
                for item in ds:
                    all_items.append(item)
            
            if all_items:
                integrated = Dataset.from_list(all_items)
                logger.info(f"[INTEGRATE] Integrated {len(integrated)} total items (fallback method)")
                return integrated
            return None
    
    def save_integrated_dataset(self, dataset: Dataset, output_path: Optional[Path] = None):
        """統合データセットを保存"""
        if output_path is None:
            output_path = self.data_dir / f"domain_knowledge_integrated_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSONL形式で保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved integrated dataset to {output_path}")
        return output_path
    
    def update_dataset_config(self, integrated_path: Path):
        """データセット設定を更新"""
        config_path = self.config_dir / "dataset.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 新しいデータソースを追加
        new_source = f"local:domain_knowledge:{integrated_path.relative_to(self.project_root)}"
        
        if new_source not in config.get('sources', []):
            config['sources'].append(new_source)
            
            # バックアップ
            backup_path = config_path.with_suffix('.json.backup')
            import shutil
            shutil.copy2(config_path, backup_path)
            
            # 保存
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[UPDATE] Added {new_source} to dataset config")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate domain knowledge datasets into sunset pipeline')
    parser.add_argument('--update-config', action='store_true',
                       help='Update dataset.json config file')
    
    args = parser.parse_args()
    
    integrator = DomainKnowledgeIntegrator()
    
    # 統合実行
    integrated_dataset = integrator.integrate_all_domain_data()
    
    if integrated_dataset:
        # 保存
        output_path = integrator.save_integrated_dataset(integrated_dataset)
        
        # 設定更新
        if args.update_config:
            integrator.update_dataset_config(output_path)
        
        print(f"\n[SUCCESS] Integrated domain knowledge dataset")
        print(f"[OUTPUT] {output_path}")
        print(f"[STATS] Total items: {len(integrated_dataset)}")
    else:
        print("\n[WARN] No data to integrate")
        sys.exit(1)


if __name__ == "__main__":
    main()
