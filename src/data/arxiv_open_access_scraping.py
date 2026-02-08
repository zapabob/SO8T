#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arxiv・オープンアクセス論文全自動Webスクレイピング（SO8T四重推論統制）

Arxivの全ジャンルとオープンアクセス論文をSO8Tの四重推論で統制しながら
全自動でバックグラウンドWebスクレイピングを実行します。

Usage:
    python scripts/data/arxiv_open_access_scraping.py --output D:/webdataset/processed --daemon
"""

import sys
import json
import logging
import asyncio
import argparse
import signal
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, quote, urljoin
from collections import deque
from dataclasses import dataclass

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# SO8Tモデルインポート
try:
    from so8t_mmllm.src.models.so8t_thinking_model import SO8TThinkingModel
    from so8t_mmllm.src.models.thinking_tokens import (
        add_thinking_tokens_to_tokenizer,
        build_quadruple_thinking_prompt,
        extract_quadruple_thinking
    )
    from transformers import AutoTokenizer
    SO8T_AVAILABLE = True
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "so8t_thinking_model",
            PROJECT_ROOT / "so8t-mmllm" / "src" / "models" / "so8t_thinking_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SO8TThinkingModel = module.SO8TThinkingModel
        SO8T_AVAILABLE = True
    except Exception:
        SO8T_AVAILABLE = False

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext, TimeoutError as PlaywrightTimeoutError
except ImportError:
    print("[ERROR] Playwright not installed. Install with: pip install playwright")
    sys.exit(1)

# BeautifulSoupインポート
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("[ERROR] BeautifulSoup not installed. Install with: pip install beautifulsoup4")
    sys.exit(1)

import torch

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/arxiv_open_access_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Arxiv全ジャンルカテゴリ
ARXIV_CATEGORIES = {
    'cs': {
        'name': 'Computer Science',
        'subcategories': [
            'cs.AI', 'cs.CL', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.GT', 'cs.CV', 'cs.CY',
            'cs.CR', 'cs.DS', 'cs.DB', 'cs.DL', 'cs.DM', 'cs.DC', 'cs.ET', 'cs.FL',
            'cs.GL', 'cs.GR', 'cs.AR', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO',
            'cs.MS', 'cs.MA', 'cs.MM', 'cs.NI', 'cs.NE', 'cs.NA', 'cs.OS', 'cs.OH',
            'cs.PF', 'cs.PL', 'cs.RO', 'cs.SI', 'cs.SE', 'cs.SD', 'cs.SC', 'cs.SY'
        ]
    },
    'math': {
        'name': 'Mathematics',
        'subcategories': [
            'math.AG', 'math.AT', 'math.AP', 'math.CT', 'math.CA', 'math.CO', 'math.AC',
            'math.CV', 'math.DG', 'math.DS', 'math.FA', 'math.GM', 'math.GN', 'math.GT',
            'math.GR', 'math.HO', 'math.IT', 'math.KT', 'math.LO', 'math.MP', 'math.MG',
            'math.NT', 'math.NA', 'math.OA', 'math.OC', 'math.PR', 'math.QA', 'math.RT',
            'math.RA', 'math.SP', 'math.ST', 'math.SG'
        ]
    },
    'physics': {
        'name': 'Physics',
        'subcategories': [
            'physics.acc-ph', 'physics.app-ph', 'physics.ao-ph', 'physics.atom-ph',
            'physics.atm-clus', 'physics.bio-ph', 'physics.chem-ph', 'physics.class-ph',
            'physics.comp-ph', 'physics.data-an', 'physics.flu-dyn', 'physics.gen-ph',
            'physics.geo-ph', 'physics.hist-ph', 'physics.ins-det', 'physics.med-ph',
            'physics.optics', 'physics.ed-ph', 'physics.soc-ph', 'physics.plasm-ph',
            'physics.pop-ph', 'physics.space-ph', 'physics.stat-mech', 'physics.surf-ph'
        ]
    },
    'q-bio': {
        'name': 'Quantitative Biology',
        'subcategories': ['q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN', 'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM', 'q-bio.SC', 'q-bio.TO']
    },
    'q-fin': {
        'name': 'Quantitative Finance',
        'subcategories': ['q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF', 'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.ST', 'q-fin.TR']
    },
    'stat': {
        'name': 'Statistics',
        'subcategories': ['stat.AP', 'stat.CO', 'stat.ML', 'stat.ME', 'stat.OT', 'stat.TH']
    },
    'eess': {
        'name': 'Electrical Engineering and Systems Science',
        'subcategories': ['eess.AS', 'eess.IV', 'eess.SP', 'eess.SY']
    },
    'econ': {
        'name': 'Economics',
        'subcategories': ['econ.EM', 'econ.GN', 'econ.TH']
    }
}

# オープンアクセス論文サイト
OPEN_ACCESS_SITES = [
    {
        'name': 'arXiv',
        'base_url': 'https://arxiv.org',
        'search_url': 'https://arxiv.org/search/',
        'categories': ARXIV_CATEGORIES
    },
    {
        'name': 'PubMed Central',
        'base_url': 'https://www.ncbi.nlm.nih.gov/pmc',
        'search_url': 'https://www.ncbi.nlm.nih.gov/pmc/?term=',
        'categories': {}
    },
    {
        'name': 'DOAJ',
        'base_url': 'https://doaj.org',
        'search_url': 'https://doaj.org/search',
        'categories': {}
    },
    {
        'name': 'PLOS ONE',
        'base_url': 'https://journals.plos.org/plosone',
        'search_url': 'https://journals.plos.org/plosone/search',
        'categories': {}
    },
    {
        'name': 'BioRxiv',
        'base_url': 'https://www.biorxiv.org',
        'search_url': 'https://www.biorxiv.org/search/',
        'categories': {}
    },
    {
        'name': 'medRxiv',
        'base_url': 'https://www.medrxiv.org',
        'search_url': 'https://www.medrxiv.org/search/',
        'categories': {}
    },
    {
        'name': 'HAL',
        'base_url': 'https://hal.archives-ouvertes.fr',
        'search_url': 'https://hal.archives-ouvertes.fr/search/index',
        'categories': {}
    },
    {
        'name': 'CORE',
        'base_url': 'https://core.ac.uk',
        'search_url': 'https://core.ac.uk/search',
        'categories': {}
    }
]


@dataclass
class PaperTask:
    """論文タスク"""
    site: str
    category: str
    url: str
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    language: str = 'en'


class ArxivOpenAccessScraper:
    """Arxiv・オープンアクセス論文スクレイパー（SO8T四重推論統制）"""
    
    def __init__(
        self,
        output_dir: Path,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_action: float = 2.0,
        timeout: int = 30000,
        max_papers_per_category: int = 50,
        use_so8t_control: bool = True,
        so8t_model_path: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            use_cursor_browser: Cursorブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_action: アクション間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_papers_per_category: カテゴリあたりの最大論文数
            use_so8t_control: SO8T統制を使用するか
            so8t_model_path: SO8Tモデルのパス
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_action = delay_per_action
        self.timeout = timeout
        self.max_papers_per_category = max_papers_per_category
        self.use_so8t_control = use_so8t_control
        
        self.all_papers: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # SO8Tモデル初期化
        self.so8t_model = None
        self.so8t_tokenizer = None
        if self.use_so8t_control and SO8T_AVAILABLE:
            try:
                self._initialize_so8t_model(so8t_model_path)
                logger.info("[SO8T] SO8T model initialized for paper scraping control")
            except Exception as e:
                logger.warning(f"[SO8T] Failed to initialize SO8T model: {e}")
                logger.warning("[SO8T] Continuing without SO8T control")
                self.use_so8t_control = False
        
        logger.info("="*80)
        logger.info("Arxiv Open Access Paper Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"SO8T control: {self.use_so8t_control}")
    
    def _initialize_so8t_model(self, model_path: Optional[str] = None):
        """SO8Tモデルを初期化"""
        try:
            if model_path is None:
                default_paths = [
                    "D:/webdataset/models/so8t-phi4-so8t-ja-finetuned",
                    "models/so8t-phi4-so8t-ja-finetuned",
                    "so8t-mmllm/models/so8t-phi4-so8t-ja-finetuned"
                ]
                for path in default_paths:
                    if Path(path).exists():
                        model_path = path
                        break
                
                if model_path is None:
                    raise FileNotFoundError("SO8T model not found")
            
            logger.info(f"[SO8T] Loading model from: {model_path}")
            
            self.so8t_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            from so8t_mmllm.src.models.safety_aware_so8t import SafetyAwareSO8TConfig
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
            
            self.so8t_model = SO8TThinkingModel(
                base_model_name_or_path=model_path,
                so8t_config=so8t_config,
                use_redacted_tokens=False,
                use_quadruple_thinking=True
            )
            
            self.so8t_model.set_tokenizer(self.so8t_tokenizer)
            self.so8t_model.eval()
            
            logger.info("[SO8T] Model loaded successfully")
            
        except Exception as e:
            logger.error(f"[SO8T] Failed to initialize model: {e}")
            raise
    
    async def so8t_control_paper_action(
        self,
        action_type: str,
        context: Dict[str, any]
    ) -> Dict[str, any]:
        """
        SO8Tモデルを使って論文スクレイピング動作を統制
        
        Args:
            action_type: 動作タイプ（'scrape', 'download', 'access'）
            context: コンテキスト情報（URL、タイトル、カテゴリなど）
        
        Returns:
            統制結果（'allow', 'deny', 'modify'）と推論結果
        """
        if not self.use_so8t_control or self.so8t_model is None:
            return {'decision': 'allow', 'reasoning': 'SO8T control disabled'}
        
        try:
            prompt = self._build_so8t_paper_prompt(action_type, context)
            
            result = await asyncio.to_thread(
                self.so8t_model.generate_thinking,
                self.so8t_tokenizer,
                prompt,
                max_new_tokens=256,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            if self.so8t_model.use_quadruple_thinking:
                task_text, safety_text, policy_text, final_text = extract_quadruple_thinking(
                    result.get('full_text', '')
                )
                
                decision = self._extract_decision_from_final(final_text)
                
                return {
                    'decision': decision,
                    'task_reasoning': task_text,
                    'safety_reasoning': safety_text,
                    'policy_reasoning': policy_text,
                    'final_reasoning': final_text,
                    'raw_output': result
                }
            else:
                thinking_text = result.get('thinking', '')
                final_text = result.get('final', '')
                decision = self._extract_decision_from_final(final_text)
                
                return {
                    'decision': decision,
                    'thinking': thinking_text,
                    'final_reasoning': final_text,
                    'raw_output': result
                }
                
        except Exception as e:
            logger.error(f"[SO8T] Control failed: {e}")
            return {'decision': 'allow', 'reasoning': f'SO8T control error: {e}'}
    
    def _build_so8t_paper_prompt(self, action_type: str, context: Dict[str, any]) -> str:
        """SO8T統制用プロンプトを構築"""
        url = context.get('url', 'unknown')
        title = context.get('title', 'unknown')
        category = context.get('category', 'unknown')
        site = context.get('site', 'unknown')
        
        if action_type == 'scrape':
            prompt = f"""以下の論文スクレイピング動作を評価し、実行を許可するか判断してください。

動作タイプ: 論文スクレイピング
サイト: {site}
カテゴリ: {category}
タイトル: {title}
URL: {url}

四重推論を行い、以下を判断してください：
1. <think-task>: この論文のスクレイピングがタスクに適切か、研究に有効か
2. <think-safety>: この論文が安全か、適切な学術コンテンツか
3. <think-policy>: この論文のスクレイピングがポリシーに準拠しているか、利用規約を遵守しているか
4. <final>: 実行を許可するか（allow/deny/modify）と理由

回答: <think-task>"""
        
        elif action_type == 'download':
            prompt = f"""以下の論文ダウンロード動作を評価し、実行を許可するか判断してください。

動作タイプ: 論文ダウンロード
サイト: {site}
タイトル: {title}
URL: {url}

四重推論を行い、以下を判断してください：
1. <think-task>: この論文のダウンロードがタスクに適切か
2. <think-safety>: この論文のダウンロードが安全か
3. <think-policy>: この論文のダウンロードがポリシーに準拠しているか、著作権を遵守しているか
4. <final>: 実行を許可するか（allow/deny/modify）と理由

回答: <think-task>"""
        
        else:
            prompt = f"""以下の論文アクセス動作を評価し、実行を許可するか判断してください。

動作タイプ: {action_type}
サイト: {site}
タイトル: {title}
URL: {url}

四重推論を行い、以下を判断してください：
1. <think-task>: この動作がタスクに適切か
2. <think-safety>: この動作が安全か
3. <think-policy>: この動作がポリシーに準拠しているか
4. <final>: 実行を許可するか（allow/deny/modify）と理由

回答: <think-task>"""
        
        return prompt
    
    def _extract_decision_from_final(self, final_text: str) -> str:
        """最終回答から意思決定を抽出"""
        final_lower = final_text.lower()
        
        if 'deny' in final_lower or '拒否' in final_lower or '禁止' in final_lower:
            return 'deny'
        elif 'modify' in final_lower or '修正' in final_lower or '変更' in final_lower:
            return 'modify'
        else:
            return 'allow'
    
    async def scrape_arxiv_category(
        self,
        page: Page,
        category: str,
        subcategory: Optional[str] = None
    ) -> List[Dict]:
        """Arxivカテゴリをスクレイピング"""
        papers = []
        
        try:
            # 検索URL構築
            if subcategory:
                search_url = f"https://arxiv.org/search/?query={subcategory}&searchtype=all&source=header"
            else:
                search_url = f"https://arxiv.org/search/?query={category}&searchtype=all&source=header"
            
            logger.info(f"[ARXIV] Scraping category: {subcategory or category}")
            
            # SO8T統制: アクセス動作を評価
            if self.use_so8t_control:
                access_context = {
                    'url': search_url,
                    'site': 'arXiv',
                    'category': subcategory or category,
                    'title': f"Category: {subcategory or category}"
                }
                control_result = await self.so8t_control_paper_action('scrape', access_context)
                
                if control_result['decision'] == 'deny':
                    logger.warning(f"[SO8T] Arxiv access denied: {search_url}")
                    return papers
            
            # ページに移動
            await page.goto(search_url, timeout=self.timeout, wait_until="networkidle")
            await asyncio.sleep(random.uniform(self.delay_per_action, self.delay_per_action * 2))
            
            # 論文リストを取得
            paper_links = await page.query_selector_all('p.list-title a[href*="/abs/"]')
            
            paper_count = 0
            for link in paper_links[:self.max_papers_per_category]:
                try:
                    href = await link.get_attribute('href')
                    if href:
                        paper_url = urljoin('https://arxiv.org', href)
                        
                        # 論文ページをスクレイピング
                        paper_data = await self.scrape_arxiv_paper(page, paper_url, subcategory or category)
                        if paper_data:
                            papers.append(paper_data)
                            paper_count += 1
                            
                            if paper_count >= self.max_papers_per_category:
                                break
                            
                            await asyncio.sleep(random.uniform(self.delay_per_action, self.delay_per_action * 2))
                
                except Exception as e:
                    logger.warning(f"[ARXIV] Failed to scrape paper link: {e}")
                    continue
            
            logger.info(f"[ARXIV] Scraped {len(papers)} papers from category: {subcategory or category}")
        
        except Exception as e:
            logger.error(f"[ARXIV] Failed to scrape category {subcategory or category}: {e}")
        
        return papers
    
    async def check_page_errors(self, page: Page) -> Dict[str, bool]:
        """ページのエラーをチェック"""
        errors = {
            'is_404': False,
            'is_200_empty': False,
            'has_content': False,
            'status_code': None
        }
        
        try:
            page_content = await page.content()
            text_content = await page.evaluate('() => document.body.innerText')
            
            # 404エラーの検出
            if any(pattern in page_content.lower() for pattern in [
                '404', 'not found', 'page not found', 'ページが見つかりません'
            ]):
                errors['is_404'] = True
            
            # 200エラー（コンテンツなし）の検出
            if text_content and len(text_content.strip()) < 50:
                errors['is_200_empty'] = True
            
            # コンテンツの有無をチェック
            if text_content and len(text_content.strip()) > 100:
                errors['has_content'] = True
        
        except Exception as e:
            logger.debug(f"[ERROR CHECK] Failed to check page errors: {e}")
        
        return errors
    
    async def handle_paper_error(
        self,
        page: Page,
        errors: Dict[str, bool],
        category: str
    ) -> bool:
        """論文ページエラーを処理（ブラウザバック、別カテゴリ遷移）"""
        try:
            if errors['is_404'] or errors['is_200_empty']:
                logger.warning(f"[ERROR HANDLING] Paper error detected: 404={errors['is_404']}, 200_empty={errors['is_200_empty']}")
                
                # ブラウザバック
                await page.go_back()
                await asyncio.sleep(random.uniform(1.0, 2.0))
                
                return True
        
        except Exception as e:
            logger.error(f"[ERROR HANDLING] Failed to handle paper error: {e}")
            return False
        
        return False
    
    async def scrape_arxiv_paper(self, page: Page, paper_url: str, category: str) -> Optional[Dict]:
        """Arxiv論文をスクレイピング（エラーハンドリング強化版）"""
        try:
            await page.goto(paper_url, timeout=self.timeout, wait_until="networkidle")
            await asyncio.sleep(random.uniform(self.delay_per_action, self.delay_per_action * 2))
            
            # エラーチェック
            errors = await self.check_page_errors(page)
            
            # エラーが検出された場合は処理
            if errors['is_404'] or errors['is_200_empty']:
                logger.warning(f"[ERROR] Paper error detected: 404={errors['is_404']}, 200_empty={errors['is_200_empty']}")
                await self.handle_paper_error(page, errors, category)
                return None
            
            # コンテンツがない場合はNoneを返す
            if not errors['has_content']:
                logger.warning(f"[ERROR] No content found for paper: {paper_url}")
                return None
            
            # SO8T統制: 論文スクレイピング動作を評価
            if self.use_so8t_control:
                title_element = await page.query_selector('h1.title')
                title = await title_element.inner_text() if title_element else 'Unknown'
                
                scrape_context = {
                    'url': paper_url,
                    'site': 'arXiv',
                    'category': category,
                    'title': title
                }
                control_result = await self.so8t_control_paper_action('scrape', scrape_context)
                
                if control_result['decision'] == 'deny':
                    logger.warning(f"[SO8T] Paper scraping denied: {paper_url}")
                    return None
            
            # タイトル取得
            title_element = await page.query_selector('h1.title')
            title = await title_element.inner_text() if title_element else 'Unknown'
            title = title.replace('Title:', '').strip()
            
            # 著者取得
            authors_element = await page.query_selector('div.authors')
            authors_text = await authors_element.inner_text() if authors_element else ''
            authors = [a.strip() for a in authors_text.replace('Authors:', '').split(',') if a.strip()]
            
            # アブストラクト取得
            abstract_element = await page.query_selector('blockquote.abstract')
            abstract = await abstract_element.inner_text() if abstract_element else ''
            abstract = abstract.replace('Abstract:', '').strip()
            
            # キーワード取得（Subjects）
            subjects_element = await page.query_selector('td.subjects')
            subjects_text = await subjects_element.inner_text() if subjects_element else ''
            keywords = [s.strip() for s in subjects_text.split(';') if s.strip()]
            
            # PDF URL取得
            pdf_link = await page.query_selector('a[href*="/pdf/"]')
            pdf_url = None
            if pdf_link:
                pdf_href = await pdf_link.get_attribute('href')
                if pdf_href:
                    pdf_url = urljoin('https://arxiv.org', pdf_href)
            
            paper_data = {
                'url': paper_url,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'keywords': keywords,
                'category': category,
                'site': 'arXiv',
                'pdf_url': pdf_url,
                'language': 'en',
                'crawled_at': datetime.now().isoformat(),
                'session_id': self.session_id
            }
            
            return paper_data
        
        except Exception as e:
            logger.error(f"[ARXIV] Failed to scrape paper {paper_url}: {e}")
            return None
    
    async def scrape_open_access_site(
        self,
        page: Page,
        site_config: Dict
    ) -> List[Dict]:
        """オープンアクセス論文サイトをスクレイピング"""
        papers = []
        site_name = site_config['name']
        base_url = site_config['base_url']
        search_url = site_config['search_url']
        
        try:
            logger.info(f"[OPEN ACCESS] Scraping site: {site_name}")
            
            # SO8T統制: サイトアクセス動作を評価
            if self.use_so8t_control:
                access_context = {
                    'url': search_url,
                    'site': site_name,
                    'category': 'open_access',
                    'title': f"Site: {site_name}"
                }
                control_result = await self.so8t_control_paper_action('scrape', access_context)
                
                if control_result['decision'] == 'deny':
                    logger.warning(f"[SO8T] Site access denied: {site_name}")
                    return papers
            
            # サイトにアクセス
            await page.goto(search_url, timeout=self.timeout, wait_until="networkidle")
            await asyncio.sleep(random.uniform(self.delay_per_action, self.delay_per_action * 2))
            
            # サイトごとのスクレイピングロジック（簡易版）
            # 実際の実装では、各サイトのHTML構造に合わせて調整が必要
            
            logger.info(f"[OPEN ACCESS] Scraped {len(papers)} papers from {site_name}")
        
        except Exception as e:
            logger.error(f"[OPEN ACCESS] Failed to scrape site {site_name}: {e}")
        
        return papers
    
    async def run_scraping(self):
        """スクレイピング実行"""
        logger.info("="*80)
        logger.info("Starting Arxiv Open Access Paper Scraping")
        logger.info("="*80)
        
        async with async_playwright() as playwright:
            # ブラウザ接続
            if self.use_cursor_browser:
                browser = await playwright.chromium.connect_over_cdp(
                    f"http://localhost:{self.remote_debugging_port}"
                )
            else:
                browser = await playwright.chromium.launch(headless=False)
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            try:
                # Arxiv全カテゴリをスクレイピング
                for category_key, category_info in ARXIV_CATEGORIES.items():
                    logger.info(f"[ARXIV] Processing category: {category_key} ({category_info['name']})")
                    
                    # サブカテゴリごとにスクレイピング
                    for subcategory in category_info['subcategories']:
                        papers = await self.scrape_arxiv_category(page, category_key, subcategory)
                        self.all_papers.extend(papers)
                        
                        # カテゴリ間の待機
                        await asyncio.sleep(random.uniform(self.delay_per_action * 2, self.delay_per_action * 4))
                
                # オープンアクセスサイトをスクレイピング
                for site_config in OPEN_ACCESS_SITES:
                    if site_config['name'] != 'arXiv':  # Arxivは既に処理済み
                        papers = await self.scrape_open_access_site(page, site_config)
                        self.all_papers.extend(papers)
                        
                        # サイト間の待機
                        await asyncio.sleep(random.uniform(self.delay_per_action * 2, self.delay_per_action * 4))
            
            finally:
                await page.close()
                await context.close()
                if not self.use_cursor_browser:
                    await browser.close()
        
        logger.info(f"[TOTAL] Collected {len(self.all_papers)} papers")
    
    def save_papers(self) -> Path:
        """論文データを保存"""
        output_file = self.output_dir / f"arxiv_open_access_papers_{self.session_id}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for paper in self.all_papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(self.all_papers)} papers to {output_file}")
        return output_file


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Arxiv Open Access Paper Scraping")
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Output directory'
    )
    parser.add_argument(
        '--use-cursor-browser',
        action='store_true',
        default=True,
        help='Use Cursor browser'
    )
    parser.add_argument(
        '--remote-debugging-port',
        type=int,
        default=9222,
        help='Remote debugging port'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between actions (seconds)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30000,
        help='Page load timeout (milliseconds)'
    )
    parser.add_argument(
        '--max-papers-per-category',
        type=int,
        default=50,
        help='Maximum papers per category'
    )
    parser.add_argument(
        '--use-so8t-control',
        action='store_true',
        default=True,
        help='Use SO8T model to control scraping actions'
    )
    parser.add_argument(
        '--so8t-model-path',
        type=str,
        default=None,
        help='Path to SO8T model'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        default=False,
        help='Run as daemon (background process)'
    )
    
    args = parser.parse_args()
    
    # スクレイパー作成
    scraper = ArxivOpenAccessScraper(
        output_dir=args.output,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_action=args.delay,
        timeout=args.timeout,
        max_papers_per_category=args.max_papers_per_category,
        use_so8t_control=args.use_so8t_control,
        so8t_model_path=args.so8t_model_path
    )
    
    # スクレイピング実行
    await scraper.run_scraping()
    
    # 保存
    output_file = scraper.save_papers()
    
    logger.info(f"[SUCCESS] Arxiv Open Access scraping completed. Output: {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(main())

