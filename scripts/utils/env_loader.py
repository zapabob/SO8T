#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
環境変数読み込みユーティリティ

.envファイルから環境変数を読み込み、型変換やデフォルト値の設定を行う。

Usage:
    from scripts.utils.env_loader import get_env, get_env_list, get_comprehensive_site_lists
    
    # 環境変数を取得
    output_dir = get_env('OUTPUT_DIR', 'D:/webdataset/processed')
    
    # URLリストを取得
    urls = get_env_list('WIKIPEDIA_JA_URLS', [])
    
    # COMPREHENSIVE_SITE_LISTSを取得
    site_lists = get_comprehensive_site_lists()
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# .envファイルのパス
ENV_FILE = PROJECT_ROOT / ".env"
EXAMPLE_ENV_FILE = PROJECT_ROOT / "example.env"


def load_env_file(env_file: Optional[Path] = None) -> bool:
    """
    .envファイルを読み込む
    
    Args:
        env_file: .envファイルのパス（Noneの場合はプロジェクトルートの.envを使用）
    
    Returns:
        success: 読み込み成功フラグ
    """
    if env_file is None:
        env_file = ENV_FILE
    
    if not env_file.exists():
        logger.warning(f"[ENV] .env file not found: {env_file}")
        logger.info(f"[ENV] Please copy {EXAMPLE_ENV_FILE} to {env_file} and set your values")
        return False
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # 空行やコメント行をスキップ
                if not line or line.startswith('#'):
                    continue
                
                # 環境変数の設定
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 既存の環境変数を上書きしない（システム環境変数を優先）
                    if key not in os.environ:
                        os.environ[key] = value
                    else:
                        logger.debug(f"[ENV] Skipping {key} (already set in environment)")
        
        logger.info(f"[ENV] Loaded environment variables from {env_file}")
        return True
    
    except Exception as e:
        logger.error(f"[ENV] Failed to load .env file: {e}")
        return False


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    環境変数を取得（デフォルト値対応）
    
    Args:
        key: 環境変数名
        default: デフォルト値（Noneの場合は環境変数が存在しない場合もNoneを返す）
    
    Returns:
        value: 環境変数の値
    """
    # まず.envファイルを読み込む（まだ読み込んでいない場合）
    if not hasattr(get_env, '_loaded'):
        load_env_file()
        get_env._loaded = True
    
    value = os.environ.get(key, default)
    return value


def get_env_list(key: str, default: Optional[List[str]] = None) -> List[str]:
    """
    カンマ区切り文字列をリストに変換
    
    Args:
        key: 環境変数名
        default: デフォルト値（Noneの場合は空リストを返す）
    
    Returns:
        urls: URLリスト
    """
    value = get_env(key)
    
    if value is None or value == '':
        return default if default is not None else []
    
    # カンマ区切りで分割し、空白を除去
    urls = [url.strip() for url in value.split(',') if url.strip()]
    return urls


def get_env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """
    環境変数を整数として取得
    
    Args:
        key: 環境変数名
        default: デフォルト値
    
    Returns:
        value: 整数値
    """
    value = get_env(key)
    
    if value is None or value == '':
        return default
    
    try:
        return int(value)
    except ValueError:
        logger.warning(f"[ENV] Failed to convert {key} to int: {value}, using default: {default}")
        return default


def get_env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    """
    環境変数を浮動小数点数として取得
    
    Args:
        key: 環境変数名
        default: デフォルト値
    
    Returns:
        value: 浮動小数点数値
    """
    value = get_env(key)
    
    if value is None or value == '':
        return default
    
    try:
        return float(value)
    except ValueError:
        logger.warning(f"[ENV] Failed to convert {key} to float: {value}, using default: {default}")
        return default


def get_env_bool(key: str, default: Optional[bool] = None) -> Optional[bool]:
    """
    環境変数をブール値として取得
    
    Args:
        key: 環境変数名
        default: デフォルト値
    
    Returns:
        value: ブール値
    """
    value = get_env(key)
    
    if value is None or value == '':
        return default
    
    # true/false, 1/0, yes/no などをブール値に変換
    value_lower = value.lower().strip()
    if value_lower in ('true', '1', 'yes', 'on'):
        return True
    elif value_lower in ('false', '0', 'no', 'off'):
        return False
    else:
        logger.warning(f"[ENV] Failed to convert {key} to bool: {value}, using default: {default}")
        return default


def get_comprehensive_site_lists() -> Dict[str, Dict[str, List[str]]]:
    """
    COMPREHENSIVE_SITE_LISTSを環境変数から構築
    
    Returns:
        site_lists: サイトリスト辞書
    """
    # デフォルト値（既存のハードコードされた値）
    default_lists = {
        'encyclopedia': {
            'wikipedia_ja': [
                "https://ja.wikipedia.org/wiki/メインページ",
                "https://ja.wikipedia.org/wiki/Category:コンピュータ",
                "https://ja.wikipedia.org/wiki/Category:プログラミング言語",
                "https://ja.wikipedia.org/wiki/Category:ソフトウェア",
                "https://ja.wikipedia.org/wiki/Category:軍事",
                "https://ja.wikipedia.org/wiki/Category:航空宇宙",
                "https://ja.wikipedia.org/wiki/Category:インフラ",
                "https://ja.wikipedia.org/wiki/Category:日本企業",
                "https://ja.wikipedia.org/wiki/防衛省",
                "https://ja.wikipedia.org/wiki/航空宇宙",
                "https://ja.wikipedia.org/wiki/医療",
            ],
            'wikipedia_en': [
                "https://en.wikipedia.org/wiki/Main_Page",
                "https://en.wikipedia.org/wiki/Category:Computer_science",
                "https://en.wikipedia.org/wiki/Category:Programming_languages",
                "https://en.wikipedia.org/wiki/Category:Software",
                "https://en.wikipedia.org/wiki/Category:Military",
                "https://en.wikipedia.org/wiki/Category:Aerospace",
                "https://en.wikipedia.org/wiki/Category:Infrastructure",
                "https://en.wikipedia.org/wiki/Defense",
                "https://en.wikipedia.org/wiki/Aerospace",
                "https://en.wikipedia.org/wiki/Medicine",
            ],
            'kotobank': [
                "https://kotobank.jp/",
                "https://kotobank.jp/word/プログラミング",
                "https://kotobank.jp/word/コンピュータ",
                "https://kotobank.jp/word/ソフトウェア",
                "https://kotobank.jp/word/軍事",
                "https://kotobank.jp/word/航空宇宙",
                "https://kotobank.jp/word/インフラ",
            ],
            'britannica': [
                "https://www.britannica.com/",
                "https://www.britannica.com/technology/computer",
                "https://www.britannica.com/technology/software",
                "https://www.britannica.com/technology/programming-language",
                "https://www.britannica.com/topic/military",
                "https://www.britannica.com/topic/aerospace-industry",
                "https://www.britannica.com/topic/infrastructure",
            ]
        },
        'coding': {
            'github': [
                "https://github.com/trending",
                "https://github.com/trending/python",
                "https://github.com/trending/rust",
                "https://github.com/trending/typescript",
                "https://github.com/trending/java",
                "https://github.com/trending/cpp",
                "https://github.com/trending/swift",
                "https://github.com/trending/kotlin",
                "https://github.com/trending/csharp",
                "https://github.com/trending/php",
                "https://github.com/explore",
            ],
            'stack_overflow': [
                "https://stackoverflow.com/questions/tagged/python",
                "https://stackoverflow.com/questions/tagged/rust",
                "https://stackoverflow.com/questions/tagged/typescript",
                "https://stackoverflow.com/questions/tagged/javascript",
                "https://stackoverflow.com/questions/tagged/java",
                "https://stackoverflow.com/questions/tagged/c%2b%2b",
                "https://stackoverflow.com/questions/tagged/c",
                "https://stackoverflow.com/questions/tagged/swift",
                "https://stackoverflow.com/questions/tagged/kotlin",
                "https://stackoverflow.com/questions/tagged/c%23",
                "https://stackoverflow.com/questions/tagged/unity3d",
                "https://stackoverflow.com/questions/tagged/php",
            ],
            'documentation': [
                "https://pytorch.org/",
                "https://pytorch.org/docs/stable/index.html",
                "https://pytorch.org/tutorials/",
                "https://www.tensorflow.org/",
                "https://www.tensorflow.org/api_docs",
                "https://www.tensorflow.org/tutorials",
                "https://docs.python.org/",
                "https://developer.mozilla.org/",
                "https://react.dev/",
                "https://vuejs.org/",
                "https://angular.io/",
                "https://docs.microsoft.com/en-us/dotnet/",
                "https://docs.microsoft.com/en-us/cpp/",
                "https://developer.apple.com/swift/",
                "https://kotlinlang.org/docs/home.html",
            ],
            'learning_sites': [
                "https://www.freecodecamp.org/",
                "https://www.codecademy.com/",
                "https://leetcode.com/",
                "https://www.codewars.com/",
            ],
            'tech_blogs': [
                "https://techcrunch.com/",
                "https://www.infoq.com/",
                "https://www.oreilly.com/",
            ],
            'reddit': [
                "https://www.reddit.com/r/programming/",
                "https://www.reddit.com/r/Python/",
                "https://www.reddit.com/r/rust/",
                "https://www.reddit.com/r/typescript/",
                "https://www.reddit.com/r/java/",
                "https://www.reddit.com/r/cpp/",
                "https://www.reddit.com/r/swift/",
                "https://www.reddit.com/r/Kotlin/",
                "https://www.reddit.com/r/csharp/",
                "https://www.reddit.com/r/Unity3D/",
                "https://www.reddit.com/r/PHP/",
            ],
            'hacker_news': [
                "https://news.ycombinator.com/",
            ],
            'engineer_sites': [
                "https://qiita.com/",
                "https://zenn.dev/",
                "https://dev.to/",
                "https://medium.com/tag/programming",
            ]
        },
        'nsfw_detection': {
            'fanza': [
                "https://www.fanza.co.jp/",
                "https://www.dmm.co.jp/",
                "https://www.dmm.co.jp/digital/videoa/",
                "https://www.dmm.co.jp/digital/videoc/",
                "https://www.dmm.co.jp/rental/",
                "https://www.dmm.co.jp/rental/videoa/",
            ],
            'fc2': [
                "https://live.fc2.com/",
                "https://live.fc2.com/category/",
                "https://live.fc2.com/ranking/",
            ],
            'missav': [
                "https://missav.ai/",
                "https://missav.ai/genre/",
                "https://missav.ai/ranking/",
            ],
            'adult_sites': [
                "https://www.pornhub.com/",
                "https://www.pornhub.com/video",
                "https://www.xvideos.com/",
                "https://www.xvideos.com/video",
                "https://www.xhamster.com/",
                "https://www.xhamster.com/video",
            ]
        },
        'government': {
            'japan': [
                "https://www.mod.go.jp/",
                "https://www.jaxa.jp/",
                "https://www.mhlw.go.jp/",
                "https://www.pmda.go.jp/",
            ],
            'us': [
                "https://www.defense.gov/",
                "https://www.nasa.gov/",
                "https://www.fda.gov/",
            ]
        },
        'drug_detection': {
            'egov': [
                "https://www.e-gov.go.jp/",
                "https://elaws.e-gov.go.jp/",
                "https://law.e-gov.go.jp/",
                "https://www.e-gov.go.jp/law/",
                "https://www.e-gov.go.jp/law/1/",
                "https://www.e-gov.go.jp/law/2/",
                "https://www.e-gov.go.jp/law/3/",
                "https://data.e-gov.go.jp/",
            ],
            'wikipedia_drug_ja': [
                "https://ja.wikipedia.org/wiki/Category:薬物",
                "https://ja.wikipedia.org/wiki/Category:違法薬物",
                "https://ja.wikipedia.org/wiki/麻薬",
                "https://ja.wikipedia.org/wiki/覚醒剤",
                "https://ja.wikipedia.org/wiki/大麻",
                "https://ja.wikipedia.org/wiki/MDMA",
                "https://ja.wikipedia.org/wiki/LSD",
                "https://ja.wikipedia.org/wiki/コカイン",
                "https://ja.wikipedia.org/wiki/ヘロイン",
            ],
            'wikipedia_drug_en': [
                "https://en.wikipedia.org/wiki/Category:Drugs",
                "https://en.wikipedia.org/wiki/Category:Illegal_drugs",
                "https://en.wikipedia.org/wiki/Narcotic",
                "https://en.wikipedia.org/wiki/Stimulant",
                "https://en.wikipedia.org/wiki/Cannabis",
                "https://en.wikipedia.org/wiki/MDMA",
                "https://en.wikipedia.org/wiki/LSD",
                "https://en.wikipedia.org/wiki/Cocaine",
                "https://en.wikipedia.org/wiki/Heroin",
            ],
            'who': [
                "https://www.who.int/",
            ],
            'unodc': [
                "https://www.unodc.org/",
            ],
            'emcdda': [
                "https://www.emcdda.europa.eu/",
            ]
        },
        'government_documents': {
            'modat': [
                "https://www.mod.go.jp/atla/",
                "https://www.mod.go.jp/atla/news/",
                "https://www.mod.go.jp/atla/publication/",
            ],
            'jr_railway': [
                "https://www.jreast.co.jp/",
                "https://www.jreast.co.jp/press/",
                "https://www.westjr.co.jp/",
                "https://www.westjr.co.jp/press/",
                "https://jr-central.co.jp/",
                "https://jr-central.co.jp/news/",
                "https://www.jrkyushu.co.jp/",
                "https://www.jrkyushu.co.jp/company/press/",
                "https://www.jrhokkaido.co.jp/",
                "https://www.jrhokkaido.co.jp/press/",
                "https://www.jr-shikoku.co.jp/",
                "https://www.jr-shikoku.co.jp/press/",
            ],
            'private_railway': [
                "https://www.tokyu.co.jp/",
                "https://www.tokyu.co.jp/railway/news/",
                "https://www.keikyu.co.jp/",
                "https://www.keikyu.co.jp/news/",
                "https://www.odakyu.jp/",
                "https://www.odakyu.jp/news/",
                "https://www.seibu-group.co.jp/",
                "https://www.seibu-group.co.jp/railways/news/",
                "https://www.keio.co.jp/",
                "https://www.keio.co.jp/news/",
                "https://www.hankyu.co.jp/",
                "https://www.hankyu.co.jp/news/",
                "https://www.hanshin.co.jp/",
                "https://www.hanshin.co.jp/news/",
                "https://www.kintetsu.co.jp/",
                "https://www.kintetsu.co.jp/news/",
                "https://www.meitetsu.co.jp/",
                "https://www.meitetsu.co.jp/news/",
                "https://www.nankai.co.jp/",
                "https://www.nankai.co.jp/news/",
            ],
            'infrastructure': [
                "https://www.tepco.co.jp/",
                "https://www.tepco.co.jp/press/",
                "https://www.kepco.co.jp/",
                "https://www.kepco.co.jp/press/",
                "https://www.chuden.co.jp/",
                "https://www.chuden.co.jp/press/",
                "https://www.rikuden.co.jp/",
                "https://www.rikuden.co.jp/press/",
                "https://www.tokyo-gas.co.jp/",
                "https://www.tokyo-gas.co.jp/press/",
                "https://www.osakagas.co.jp/",
                "https://www.osakagas.co.jp/press/",
                "https://www.mlit.go.jp/",
                "https://www.mlit.go.jp/report/",
                "https://www.mlit.go.jp/road/",
                "https://www.mlit.go.jp/koku/",
                "https://www.mlit.go.jp/port/",
            ],
            'government_db_jp': [
                "https://www.soumu.go.jp/",
                "https://www.soumu.go.jp/menu_news/",
                "https://www.mof.go.jp/",
                "https://www.mof.go.jp/press/",
                "https://www.meti.go.jp/",
                "https://www.meti.go.jp/press/",
                "https://www.mhlw.go.jp/",
                "https://www.mhlw.go.jp/stf/",
                "https://www.mext.go.jp/",
                "https://www.mext.go.jp/b_menu/",
                "https://www.maff.go.jp/",
                "https://www.maff.go.jp/press/",
                "https://www.mlit.go.jp/",
                "https://www.mlit.go.jp/report/",
                "https://www.env.go.jp/",
                "https://www.env.go.jp/press/",
                "https://www.mod.go.jp/",
                "https://www.mod.go.jp/j/press/",
            ],
            'government_db_us': [
                "https://www.defense.gov/",
                "https://www.defense.gov/News/",
                "https://www.energy.gov/",
                "https://www.energy.gov/news",
                "https://www.transportation.gov/",
                "https://www.transportation.gov/newsroom",
                "https://www.commerce.gov/",
                "https://www.commerce.gov/news",
                "https://www.hhs.gov/",
                "https://www.hhs.gov/about/news",
            ],
            'government_db_uk': [
                "https://www.gov.uk/government/organisations/ministry-of-defence",
                "https://www.gov.uk/government/organisations/department-for-transport",
                "https://www.gov.uk/government/organisations/department-for-business-energy-and-industrial-strategy",
                "https://www.gov.uk/government/organisations/department-of-health-and-social-care",
            ],
            'legal_db': [
                "https://elaws.e-gov.go.jp/",
                "https://law.e-gov.go.jp/",
                "https://www.courts.go.jp/app/hanrei_jp/",
                "https://www.cas.go.jp/jp/seisaku/hourei/",
            ],
            'white_papers': [
                "https://www.mod.go.jp/j/publication/wp/",
                "https://www.enecho.meti.go.jp/about/whitepaper/",
                "https://www.mlit.go.jp/hakusyo/",
                "https://www.soumu.go.jp/johotsusintokei/whitepaper/",
                "https://www.env.go.jp/policy/hakusyo/",
            ]
        },
        'tech_blogs': {
            'qiita': [
                "https://qiita.com/",
            ],
            'zenn': [
                "https://zenn.dev/",
            ],
            'note': [
                "https://note.com/tech",
            ]
        }
    }
    
    # 環境変数から読み込む
    site_lists = {}
    
    # Encyclopedia
    encyclopedia = {}
    wikipedia_ja = get_env_list('WIKIPEDIA_JA_URLS', default_lists['encyclopedia']['wikipedia_ja'])
    wikipedia_en = get_env_list('WIKIPEDIA_EN_URLS', default_lists['encyclopedia']['wikipedia_en'])
    kotobank = get_env_list('KOTOBANK_URLS', default_lists['encyclopedia']['kotobank'])
    britannica = get_env_list('BRITANNICA_URLS', default_lists['encyclopedia']['britannica'])
    
    if wikipedia_ja:
        encyclopedia['wikipedia_ja'] = wikipedia_ja
    if wikipedia_en:
        encyclopedia['wikipedia_en'] = wikipedia_en
    if kotobank:
        encyclopedia['kotobank'] = kotobank
    if britannica:
        encyclopedia['britannica'] = britannica
    
    if encyclopedia:
        site_lists['encyclopedia'] = encyclopedia
    
    # Coding
    coding = {}
    github = get_env_list('GITHUB_URLS', default_lists['coding']['github'])
    stack_overflow = get_env_list('STACK_OVERFLOW_URLS', default_lists['coding']['stack_overflow'])
    documentation = get_env_list('DOCUMENTATION_URLS', default_lists['coding']['documentation'])
    learning_sites = get_env_list('LEARNING_SITES_URLS', default_lists['coding']['learning_sites'])
    tech_blogs = get_env_list('TECH_BLOGS_URLS', default_lists['coding']['tech_blogs'])
    reddit = get_env_list('REDDIT_URLS', default_lists['coding']['reddit'])
    hacker_news = get_env_list('HACKER_NEWS_URLS', default_lists['coding']['hacker_news'])
    engineer_sites = get_env_list('ENGINEER_SITES_URLS', default_lists['coding']['engineer_sites'])
    
    if github:
        coding['github'] = github
    if stack_overflow:
        coding['stack_overflow'] = stack_overflow
    if documentation:
        coding['documentation'] = documentation
    if learning_sites:
        coding['learning_sites'] = learning_sites
    if tech_blogs:
        coding['tech_blogs'] = tech_blogs
    if reddit:
        coding['reddit'] = reddit
    if hacker_news:
        coding['hacker_news'] = hacker_news
    if engineer_sites:
        coding['engineer_sites'] = engineer_sites
    
    if coding:
        site_lists['coding'] = coding
    
    # NSFW Detection
    nsfw_detection = {}
    fanza = get_env_list('FANZA_URLS', default_lists['nsfw_detection']['fanza'])
    fc2 = get_env_list('FC2_URLS', default_lists['nsfw_detection']['fc2'])
    missav = get_env_list('MISSAV_URLS', default_lists['nsfw_detection']['missav'])
    adult_sites = get_env_list('ADULT_SITES_URLS', default_lists['nsfw_detection']['adult_sites'])
    
    if fanza:
        nsfw_detection['fanza'] = fanza
    if fc2:
        nsfw_detection['fc2'] = fc2
    if missav:
        nsfw_detection['missav'] = missav
    if adult_sites:
        nsfw_detection['adult_sites'] = adult_sites
    
    if nsfw_detection:
        site_lists['nsfw_detection'] = nsfw_detection
    
    # Government
    government = {}
    government_jp = get_env_list('GOVERNMENT_JP_URLS', default_lists['government']['japan'])
    government_us = get_env_list('GOVERNMENT_US_URLS', default_lists['government']['us'])
    
    if government_jp:
        government['japan'] = government_jp
    if government_us:
        government['us'] = government_us
    
    if government:
        site_lists['government'] = government
    
    # Drug Detection
    drug_detection = {}
    egov = get_env_list('EGOV_URLS', default_lists['drug_detection']['egov'])
    wikipedia_drug_ja = get_env_list('WIKIPEDIA_DRUG_JA_URLS', default_lists['drug_detection']['wikipedia_drug_ja'])
    wikipedia_drug_en = get_env_list('WIKIPEDIA_DRUG_EN_URLS', default_lists['drug_detection']['wikipedia_drug_en'])
    who = get_env_list('WHO_URLS', default_lists['drug_detection']['who'])
    unodc = get_env_list('UNODC_URLS', default_lists['drug_detection']['unodc'])
    emcdda = get_env_list('EMCDDA_URLS', default_lists['drug_detection']['emcdda'])
    
    if egov:
        drug_detection['egov'] = egov
    if wikipedia_drug_ja:
        drug_detection['wikipedia_drug_ja'] = wikipedia_drug_ja
    if wikipedia_drug_en:
        drug_detection['wikipedia_drug_en'] = wikipedia_drug_en
    if who:
        drug_detection['who'] = who
    if unodc:
        drug_detection['unodc'] = unodc
    if emcdda:
        drug_detection['emcdda'] = emcdda
    
    if drug_detection:
        site_lists['drug_detection'] = drug_detection
    
    # Government Documents
    government_documents = {}
    modat = get_env_list('MODAT_URLS', default_lists['government_documents']['modat'])
    jr_railway = get_env_list('JR_RAILWAY_URLS', default_lists['government_documents']['jr_railway'])
    private_railway = get_env_list('PRIVATE_RAILWAY_URLS', default_lists['government_documents']['private_railway'])
    infrastructure = get_env_list('INFRASTRUCTURE_URLS', default_lists['government_documents']['infrastructure'])
    government_db_jp = get_env_list('GOVERNMENT_DB_JP_URLS', default_lists['government_documents']['government_db_jp'])
    government_db_us = get_env_list('GOVERNMENT_DB_US_URLS', default_lists['government_documents']['government_db_us'])
    government_db_uk = get_env_list('GOVERNMENT_DB_UK_URLS', default_lists['government_documents']['government_db_uk'])
    legal_db = get_env_list('LEGAL_DB_URLS', default_lists['government_documents']['legal_db'])
    white_papers = get_env_list('WHITE_PAPERS_URLS', default_lists['government_documents']['white_papers'])
    
    if modat:
        government_documents['modat'] = modat
    if jr_railway:
        government_documents['jr_railway'] = jr_railway
    if private_railway:
        government_documents['private_railway'] = private_railway
    if infrastructure:
        government_documents['infrastructure'] = infrastructure
    if government_db_jp:
        government_documents['government_db_jp'] = government_db_jp
    if government_db_us:
        government_documents['government_db_us'] = government_db_us
    if government_db_uk:
        government_documents['government_db_uk'] = government_db_uk
    if legal_db:
        government_documents['legal_db'] = legal_db
    if white_papers:
        government_documents['white_papers'] = white_papers
    
    if government_documents:
        site_lists['government_documents'] = government_documents
    
    # Tech Blogs
    tech_blogs_category = {}
    qiita = get_env_list('QIITA_URLS', default_lists['tech_blogs']['qiita'])
    zenn = get_env_list('ZENN_URLS', default_lists['tech_blogs']['zenn'])
    note = get_env_list('NOTE_URLS', default_lists['tech_blogs']['note'])
    
    if qiita:
        tech_blogs_category['qiita'] = qiita
    if zenn:
        tech_blogs_category['zenn'] = zenn
    if note:
        tech_blogs_category['note'] = note
    
    if tech_blogs_category:
        site_lists['tech_blogs'] = tech_blogs_category
    
    # 環境変数が設定されていない場合はデフォルト値を使用
    if not site_lists:
        logger.info("[ENV] No environment variables found, using default site lists")
        return default_lists
    
    logger.info(f"[ENV] Loaded site lists from environment variables: {list(site_lists.keys())}")
    return site_lists


# モジュール読み込み時に.envファイルを自動読み込み
load_env_file()


