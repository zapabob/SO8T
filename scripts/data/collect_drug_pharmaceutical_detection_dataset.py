#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
検知用違法薬物・医薬品データセット収集スクリプト

安全判定と拒否挙動の学習を目的とした違法薬物・医薬品検知用データセットを収集します。
生成目的ではなく、検知・防止・教育を目的としています。

データソース:
- PMDA (医薬品医療機器総合機構): 日本の医薬品・医療機器情報
- GoV (Government of Victoria): オーストラリア・ビクトリア州政府の薬物情報
- UNODC (United Nations Office on Drugs and Crime): 国連薬物犯罪事務所
- EMCDDA (European Monitoring Centre for Drugs and Drug Addiction): 欧州薬物・薬物依存監視センター
- Wikipedia: 違法薬物・医薬品に関する記事

Usage:
    python scripts/data/collect_drug_pharmaceutical_detection_dataset.py --output D:/webdataset/drug_pharmaceutical_detection_dataset
"""

import sys
import json
import logging
import argparse
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from urllib.parse import urljoin, urlparse
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_drug_pharmaceutical_detection_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available")

# requestsインポート
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests/BeautifulSoup not available")

# Wikipedia APIインポート
try:
    import wikipediaapi
    WIKIPEDIA_API_AVAILABLE = True
except ImportError:
    WIKIPEDIA_API_AVAILABLE = False
    logger.warning("wikipediaapi not available")


class DrugPharmaceuticalDetectionDatasetCollector:
    """検知用違法薬物・医薬品データセット収集クラス"""
    
    def __init__(self, output_dir: Path):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データソース定義
        self.data_sources = {
            'pmda': {
                'base_url': 'https://www.pmda.go.jp',
                'name': 'PMDA (医薬品医療機器総合機構)',
                'language': 'ja',
                'enabled': True
            },
            'gov': {
                'base_url': 'https://www.health.vic.gov.au',
                'name': 'Government of Victoria (Health)',
                'language': 'en',
                'enabled': True
            },
            'who': {
                'base_url': 'https://www.who.int',
                'name': 'WHO (World Health Organization)',
                'language': 'en',
                'enabled': True
            },
            'unodc': {
                'base_url': 'https://www.unodc.org',
                'name': 'UNODC (United Nations Office on Drugs and Crime)',
                'language': 'en',
                'enabled': True
            },
            'emcdda': {
                'base_url': 'https://www.emcdda.europa.eu',
                'name': 'EMCDDA (European Monitoring Centre for Drugs and Drug Addiction)',
                'language': 'en',
                'enabled': True
            },
            'wikipedia': {
                'base_url': 'https://ja.wikipedia.org',
                'name': 'Wikipedia',
                'language': 'ja',
                'enabled': True
            }
        }
        
        # 違法薬物・医薬品カテゴリ定義（検知目的）
        self.drug_categories = {
            'illegal_drugs': {
                'keywords_ja': ['違法薬物', '麻薬', '覚醒剤', '大麻', 'コカイン', 'ヘロイン', 'MDMA', 'LSD', '幻覚剤', '向精神薬'],
                'keywords_en': ['illegal drugs', 'narcotics', 'amphetamine', 'cannabis', 'cocaine', 'heroin', 'MDMA', 'LSD', 'hallucinogen', 'psychotropic'],
                'severity': 'high',
                'legal_status': 'illegal',
                'context_required': False
            },
            'prescription_drugs_abuse': {
                'keywords_ja': ['処方薬乱用', 'オピオイド乱用', '鎮痛剤乱用', '睡眠薬乱用', '抗不安薬乱用'],
                'keywords_en': ['prescription drug abuse', 'opioid abuse', 'painkiller abuse', 'sleeping pill abuse', 'anxiety medication abuse'],
                'severity': 'high',
                'legal_status': 'illegal_when_abused',
                'context_required': True
            },
            'pharmaceuticals_legitimate': {
                'keywords_ja': ['医薬品', '薬剤', '治療薬', '処方薬', '承認薬', '臨床試験'],
                'keywords_en': ['pharmaceutical', 'medication', 'therapeutic', 'prescription drug', 'approved drug', 'clinical trial'],
                'severity': 'low',
                'legal_status': 'legal',
                'context_required': True
            },
            'controlled_substances': {
                'keywords_ja': ['指定薬物', '規制薬物', '向精神薬', '麻薬指定', '覚醒剤指定'],
                'keywords_en': ['controlled substance', 'scheduled drug', 'psychotropic substance', 'narcotic', 'amphetamine'],
                'severity': 'high',
                'legal_status': 'controlled',
                'context_required': False
            },
            'drug_trafficking': {
                'keywords_ja': ['薬物密輸', '薬物取引', '薬物販売', '薬物密売'],
                'keywords_en': ['drug trafficking', 'drug trade', 'drug dealing', 'drug smuggling'],
                'severity': 'critical',
                'legal_status': 'illegal',
                'context_required': False
            },
            'drug_manufacturing': {
                'keywords_ja': ['薬物製造', '覚醒剤製造', '麻薬製造', '違法製造'],
                'keywords_en': ['drug manufacturing', 'amphetamine manufacturing', 'narcotic manufacturing', 'illegal manufacturing'],
                'severity': 'critical',
                'legal_status': 'illegal',
                'context_required': False
            }
        }
        
        # 収集済みサンプル
        self.collected_samples = []
        
        logger.info("="*80)
        logger.info("Drug/Pharmaceutical Detection Dataset Collector Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Data sources: {len([s for s in self.data_sources.values() if s['enabled']])}")
    
    async def collect_from_pmda(self, max_samples: int = 1000) -> List[Dict]:
        """PMDAからデータを収集"""
        logger.info("[PMDA] Starting collection from PMDA...")
        
        samples = []
        
        if not REQUESTS_AVAILABLE:
            logger.warning("[PMDA] requests/BeautifulSoup not available, skipping")
            return samples
        
        try:
            # PMDAの主要ページ
            pmda_pages = [
                '/safety/surveillance-analysis/0045.html',  # 医薬品等安全性情報
                '/review-services/drug-reviews/review-information/p-drugs/0013.html',  # 未承認薬データベース
                '/safety/info-services/drug-recall/0010.html',  # 医薬品回収情報
            ]
            
            for page_path in pmda_pages:
                if len(samples) >= max_samples:
                    break
                
                url = urljoin(self.data_sources['pmda']['base_url'], page_path)
                logger.info(f"[PMDA] Fetching {url}...")
                
                try:
                    response = requests.get(url, timeout=30, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # 薬物関連キーワードをチェック
                    if self._contains_drug_keywords(text, 'ja'):
                        sample = {
                            'text': text[:5000],  # 最大5000文字
                            'url': url,
                            'source': 'pmda',
                            'source_name': self.data_sources['pmda']['name'],
                            'language': 'ja',
                            'category': self._classify_drug_category(text, 'ja'),
                            'collected_at': datetime.now().isoformat(),
                            'detection_purpose': 'safety_training'
                        }
                        samples.append(sample)
                        logger.info(f"[PMDA] Collected sample from {url}")
                
                except Exception as e:
                    logger.warning(f"[PMDA] Failed to fetch {url}: {e}")
                    continue
            
            logger.info(f"[PMDA] Collected {len(samples)} samples")
            return samples
        
        except Exception as e:
            logger.error(f"[PMDA] Error: {e}")
            return samples
    
    async def collect_from_gov(self, max_samples: int = 1000) -> List[Dict]:
        """GoV (Government of Victoria)からデータを収集"""
        logger.info("[GoV] Starting collection from Government of Victoria...")
        
        samples = []
        
        if not REQUESTS_AVAILABLE:
            logger.warning("[GoV] requests/BeautifulSoup not available, skipping")
            return samples
        
        try:
            # GoVの薬物関連ページ
            gov_pages = [
                '/aod/alcohol-and-drug-services',
                '/aod/drugs-and-alcohol',
                '/aod/alcohol-and-drug-treatment',
            ]
            
            for page_path in gov_pages:
                if len(samples) >= max_samples:
                    break
                
                url = urljoin(self.data_sources['gov']['base_url'], page_path)
                logger.info(f"[GoV] Fetching {url}...")
                
                try:
                    response = requests.get(url, timeout=30, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # 薬物関連キーワードをチェック
                    if self._contains_drug_keywords(text, 'en'):
                        sample = {
                            'text': text[:5000],
                            'url': url,
                            'source': 'gov',
                            'source_name': self.data_sources['gov']['name'],
                            'language': 'en',
                            'category': self._classify_drug_category(text, 'en'),
                            'collected_at': datetime.now().isoformat(),
                            'detection_purpose': 'safety_training'
                        }
                        samples.append(sample)
                        logger.info(f"[GoV] Collected sample from {url}")
                
                except Exception as e:
                    logger.warning(f"[GoV] Failed to fetch {url}: {e}")
                    continue
            
            logger.info(f"[GoV] Collected {len(samples)} samples")
            return samples
        
        except Exception as e:
            logger.error(f"[GoV] Error: {e}")
            return samples
    
    async def collect_from_unodc(self, max_samples: int = 1000) -> List[Dict]:
        """UNODCからデータを収集"""
        logger.info("[UNODC] Starting collection from UNODC...")
        
        samples = []
        
        if not REQUESTS_AVAILABLE:
            logger.warning("[UNODC] requests/BeautifulSoup not available, skipping")
            return samples
        
        try:
            # UNODCの薬物関連ページ
            unodc_pages = [
                '/unodc/en/drugs/index.html',
                '/unodc/en/drug-prevention-and-treatment/index.html',
                '/unodc/en/drug-trafficking/index.html',
            ]
            
            for page_path in unodc_pages:
                if len(samples) >= max_samples:
                    break
                
                url = urljoin(self.data_sources['unodc']['base_url'], page_path)
                logger.info(f"[UNODC] Fetching {url}...")
                
                try:
                    response = requests.get(url, timeout=30, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # 薬物関連キーワードをチェック
                    if self._contains_drug_keywords(text, 'en'):
                        sample = {
                            'text': text[:5000],
                            'url': url,
                            'source': 'unodc',
                            'source_name': self.data_sources['unodc']['name'],
                            'language': 'en',
                            'category': self._classify_drug_category(text, 'en'),
                            'collected_at': datetime.now().isoformat(),
                            'detection_purpose': 'safety_training'
                        }
                        samples.append(sample)
                        logger.info(f"[UNODC] Collected sample from {url}")
                
                except Exception as e:
                    logger.warning(f"[UNODC] Failed to fetch {url}: {e}")
                    continue
            
            logger.info(f"[UNODC] Collected {len(samples)} samples")
            return samples
        
        except Exception as e:
            logger.error(f"[UNODC] Error: {e}")
            return samples
    
    async def collect_from_who(self, max_samples: int = 1000) -> List[Dict]:
        """WHO (World Health Organization)からデータを収集"""
        logger.info("[WHO] Starting collection from WHO...")
        
        samples = []
        
        if not REQUESTS_AVAILABLE:
            logger.warning("[WHO] requests/BeautifulSoup not available, skipping")
            return samples
        
        try:
            # WHOの薬物関連ページ
            who_pages = [
                '/health-topics/substance-abuse',
                '/health-topics/drug-abuse',
                '/health-topics/mental-health',
                '/health-topics/pharmaceuticals',
            ]
            
            for page_path in who_pages:
                if len(samples) >= max_samples:
                    break
                
                url = urljoin(self.data_sources['who']['base_url'], page_path)
                logger.info(f"[WHO] Fetching {url}...")
                
                try:
                    response = requests.get(url, timeout=30, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # 薬物関連キーワードをチェック
                    if self._contains_drug_keywords(text, 'en'):
                        sample = {
                            'text': text[:5000],
                            'url': url,
                            'source': 'who',
                            'source_name': self.data_sources['who']['name'],
                            'language': 'en',
                            'category': self._classify_drug_category(text, 'en'),
                            'collected_at': datetime.now().isoformat(),
                            'detection_purpose': 'safety_training'
                        }
                        samples.append(sample)
                        logger.info(f"[WHO] Collected sample from {url}")
                
                except Exception as e:
                    logger.warning(f"[WHO] Failed to fetch {url}: {e}")
                    continue
            
            logger.info(f"[WHO] Collected {len(samples)} samples")
            return samples
        
        except Exception as e:
            logger.error(f"[WHO] Error: {e}")
            return samples
    
    async def collect_from_emcdda(self, max_samples: int = 1000) -> List[Dict]:
        """EMCDDAからデータを収集"""
        logger.info("[EMCDDA] Starting collection from EMCDDA...")
        
        samples = []
        
        if not REQUESTS_AVAILABLE:
            logger.warning("[EMCDDA] requests/BeautifulSoup not available, skipping")
            return samples
        
        try:
            # EMCDDAの薬物関連ページ
            emcdda_pages = [
                '/publications/html/pods/drugs_en.html',
                '/publications/html/pods/drug-prevention_en.html',
                '/publications/html/pods/drug-treatment_en.html',
            ]
            
            for page_path in emcdda_pages:
                if len(samples) >= max_samples:
                    break
                
                url = urljoin(self.data_sources['emcdda']['base_url'], page_path)
                logger.info(f"[EMCDDA] Fetching {url}...")
                
                try:
                    response = requests.get(url, timeout=30, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # 薬物関連キーワードをチェック
                    if self._contains_drug_keywords(text, 'en'):
                        sample = {
                            'text': text[:5000],
                            'url': url,
                            'source': 'emcdda',
                            'source_name': self.data_sources['emcdda']['name'],
                            'language': 'en',
                            'category': self._classify_drug_category(text, 'en'),
                            'collected_at': datetime.now().isoformat(),
                            'detection_purpose': 'safety_training'
                        }
                        samples.append(sample)
                        logger.info(f"[EMCDDA] Collected sample from {url}")
                
                except Exception as e:
                    logger.warning(f"[EMCDDA] Failed to fetch {url}: {e}")
                    continue
            
            logger.info(f"[EMCDDA] Collected {len(samples)} samples")
            return samples
        
        except Exception as e:
            logger.error(f"[EMCDDA] Error: {e}")
            return samples
    
    async def collect_from_wikipedia(self, max_samples: int = 1000) -> List[Dict]:
        """Wikipediaからデータを収集"""
        logger.info("[Wikipedia] Starting collection from Wikipedia...")
        
        samples = []
        
        # Wikipedia記事のリスト（違法薬物・医薬品関連）
        wikipedia_articles = {
            'ja': [
                '違法薬物',
                '麻薬',
                '覚醒剤',
                '大麻',
                'コカイン',
                'ヘロイン',
                'MDMA',
                'LSD',
                '向精神薬',
                '処方薬',
                '医薬品',
                '薬物依存',
                '薬物乱用',
                '薬物犯罪',
            ],
            'en': [
                'Illegal drug trade',
                'Narcotic',
                'Amphetamine',
                'Cannabis',
                'Cocaine',
                'Heroin',
                'MDMA',
                'LSD',
                'Psychotropic drug',
                'Prescription drug',
                'Pharmaceutical drug',
                'Drug addiction',
                'Drug abuse',
                'Drug crime',
            ]
        }
        
        if WIKIPEDIA_API_AVAILABLE:
            # Wikipedia APIを使用
            try:
                wiki_ja = wikipediaapi.Wikipedia('ja', user_agent='SO8T-Drug-Collector/1.0')
                wiki_en = wikipediaapi.Wikipedia('en', user_agent='SO8T-Drug-Collector/1.0')
                
                for lang, articles in wikipedia_articles.items():
                    if len(samples) >= max_samples:
                        break
                    
                    wiki = wiki_ja if lang == 'ja' else wiki_en
                    
                    for article_title in articles:
                        if len(samples) >= max_samples:
                            break
                        
                        try:
                            page = wiki.page(article_title)
                            if page.exists():
                                text = page.text
                                
                                if self._contains_drug_keywords(text, lang):
                                    sample = {
                                        'text': text[:5000],
                                        'url': page.fullurl,
                                        'source': 'wikipedia',
                                        'source_name': 'Wikipedia',
                                        'language': lang,
                                        'category': self._classify_drug_category(text, lang),
                                        'article_title': article_title,
                                        'collected_at': datetime.now().isoformat(),
                                        'detection_purpose': 'safety_training'
                                    }
                                    samples.append(sample)
                                    logger.info(f"[Wikipedia] Collected article: {article_title}")
                        
                        except Exception as e:
                            logger.warning(f"[Wikipedia] Failed to fetch article {article_title}: {e}")
                            continue
            
            except Exception as e:
                logger.warning(f"[Wikipedia] Wikipedia API error: {e}, falling back to web scraping")
        
        # Webスクレイピングフォールバック
        if not WIKIPEDIA_API_AVAILABLE or len(samples) < max_samples:
            if REQUESTS_AVAILABLE:
                try:
                    for lang, articles in wikipedia_articles.items():
                        if len(samples) >= max_samples:
                            break
                        
                        base_url = 'https://ja.wikipedia.org' if lang == 'ja' else 'https://en.wikipedia.org'
                        
                        for article_title in articles:
                            if len(samples) >= max_samples:
                                break
                            
                            try:
                                url = f"{base_url}/wiki/{article_title.replace(' ', '_')}"
                                response = requests.get(url, timeout=30, headers={
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                })
                                response.raise_for_status()
                                
                                soup = BeautifulSoup(response.content, 'html.parser')
                                
                                # メインコンテンツを抽出
                                content_div = soup.find('div', {'id': 'mw-content-text'})
                                if content_div:
                                    text = content_div.get_text(separator='\n', strip=True)
                                    
                                    if self._contains_drug_keywords(text, lang):
                                        sample = {
                                            'text': text[:5000],
                                            'url': url,
                                            'source': 'wikipedia',
                                            'source_name': 'Wikipedia',
                                            'language': lang,
                                            'category': self._classify_drug_category(text, lang),
                                            'article_title': article_title,
                                            'collected_at': datetime.now().isoformat(),
                                            'detection_purpose': 'safety_training'
                                        }
                                        samples.append(sample)
                                        logger.info(f"[Wikipedia] Collected article: {article_title}")
                            
                            except Exception as e:
                                logger.warning(f"[Wikipedia] Failed to fetch article {article_title}: {e}")
                                continue
                
                except Exception as e:
                    logger.error(f"[Wikipedia] Web scraping error: {e}")
        
        logger.info(f"[Wikipedia] Collected {len(samples)} samples")
        return samples
    
    def _contains_drug_keywords(self, text: str, language: str) -> bool:
        """テキストに薬物関連キーワードが含まれているかチェック"""
        text_lower = text.lower()
        
        for category, info in self.drug_categories.items():
            keywords = info['keywords_ja'] if language == 'ja' else info['keywords_en']
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return True
        
        return False
    
    def _classify_drug_category(self, text: str, language: str) -> str:
        """薬物カテゴリを分類"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, info in self.drug_categories.items():
            keywords = info['keywords_ja'] if language == 'ja' else info['keywords_en']
            score = 0
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'unknown'
    
    async def collect_all(self, max_samples_per_source: int = 1000) -> List[Dict]:
        """すべてのソースからデータを収集"""
        logger.info(f"[COLLECT] Starting collection from all sources (max {max_samples_per_source} per source)...")
        
        all_samples = []
        
        # PMDA
        if self.data_sources['pmda']['enabled']:
            pmda_samples = await self.collect_from_pmda(max_samples_per_source)
            all_samples.extend(pmda_samples)
        
        # GoV
        if self.data_sources['gov']['enabled']:
            gov_samples = await self.collect_from_gov(max_samples_per_source)
            all_samples.extend(gov_samples)
        
        # UNODC
        if self.data_sources['unodc']['enabled']:
            unodc_samples = await self.collect_from_unodc(max_samples_per_source)
            all_samples.extend(unodc_samples)
        
        # WHO
        if self.data_sources['who']['enabled']:
            who_samples = await self.collect_from_who(max_samples_per_source)
            all_samples.extend(who_samples)
        
        # EMCDDA
        if self.data_sources['emcdda']['enabled']:
            emcdda_samples = await self.collect_from_emcdda(max_samples_per_source)
            all_samples.extend(emcdda_samples)
        
        # Wikipedia
        if self.data_sources['wikipedia']['enabled']:
            wikipedia_samples = await self.collect_from_wikipedia(max_samples_per_source)
            all_samples.extend(wikipedia_samples)
        
        logger.info(f"[COLLECT] Total collected: {len(all_samples)} samples")
        
        # ソース別統計
        source_dist = Counter(s.get('source', 'unknown') for s in all_samples)
        logger.info(f"[COLLECT] Source distribution: {dict(source_dist)}")
        
        # カテゴリ別統計
        category_dist = Counter(s.get('category', 'unknown') for s in all_samples)
        logger.info(f"[COLLECT] Category distribution: {dict(category_dist)}")
        
        return all_samples
    
    def save_dataset(self, samples: List[Dict], split_ratio: float = 0.8):
        """データセットを保存"""
        logger.info(f"[SAVE] Saving {len(samples)} samples...")
        
        # ラベル分布を確認
        source_dist = Counter(s.get('source', 'unknown') for s in samples)
        logger.info(f"[SAVE] Source distribution: {dict(source_dist)}")
        
        # 訓練/検証に分割
        import random
        random.shuffle(samples)
        split_idx = int(len(samples) * split_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # 訓練データを保存
        train_file = self.output_dir / "drug_pharmaceutical_detection_train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"[SAVE] Training data saved: {train_file} ({len(train_samples)} samples)")
        
        # 検証データを保存
        val_file = self.output_dir / "drug_pharmaceutical_detection_val.jsonl"
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
            'category_distribution': dict(Counter(s.get('category', 'unknown') for s in samples)),
            'language_distribution': dict(Counter(s.get('language', 'unknown') for s in samples)),
            'created_at': datetime.now().isoformat(),
            'purpose': 'safety_training',  # 生成目的ではない
            'data_sources': {k: v['name'] for k, v in self.data_sources.items() if v['enabled']},
            'categories': list(self.drug_categories.keys()),
            'total_categories': len(self.drug_categories)
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"[SAVE] Metadata saved: {metadata_file}")


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Drug/Pharmaceutical Detection Dataset Collector')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--max-samples-per-source', type=int, default=1000, help='Maximum samples per source')
    parser.add_argument('--sources', nargs='+', choices=['pmda', 'gov', 'who', 'unodc', 'emcdda', 'wikipedia', 'all'],
                       default=['all'], help='Data sources to collect from')
    
    args = parser.parse_args()
    
    collector = DrugPharmaceuticalDetectionDatasetCollector(output_dir=args.output)
    
    # ソースを有効化/無効化
    if 'all' not in args.sources:
        for source in collector.data_sources.keys():
            collector.data_sources[source]['enabled'] = source in args.sources
    
    # データ収集
    samples = await collector.collect_all(max_samples_per_source=args.max_samples_per_source)
    
    if samples:
        collector.save_dataset(samples)
        logger.info(f"[OK] Drug/pharmaceutical detection dataset collection completed: {len(samples)} total samples")
    else:
        logger.warning("[WARNING] No samples collected")


if __name__ == '__main__':
    asyncio.run(main())

