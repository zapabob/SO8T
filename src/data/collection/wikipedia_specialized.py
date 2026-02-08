import os
import json
import logging
from pathlib import Path
import requests
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = {
    "military": ["Military", "Weapon", "Tactics", "Nuclear_warfare", "Aviation_safety"],
    "aerospace": ["Space_exploration", "Aerospace_engineering", "Satellite", "Rocket_propulsion"],
    "intelligence": ["Intelligence_agency", "Espionage", "Cryptography", "OSINT"],
    "ai_research": ["Large_language_model", "Artificial_neural_network", "Transformer_(deep_learning)"],
    "medical_pharma": ["Pharmacology", "Virology", "Molecular_biology", "Drug_discovery"],
    "safety_policy": ["Controlled_substances", "Toxicology", "Bioethics"]
}

LANGS = ["ja", "en"]

class WikipediaSpecializedFetcher:
    """
    Fetches specialized content from Wikipedia API for targeted domains.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SO8T-Moonshot-Pipeline/4.0 (contact: zapabob)"})

    def fetch_category_members(self, lang: str, category: str, limit: int = 50) -> List[str]:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit,
            "format": "json"
        }
        titles = []
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"Error {response.status_code} fetching members for {category} in {lang}")
                return []
            if not response.content:
                logger.error(f"Empty response fetching members for {category} in {lang}")
                return []
            data = response.json()
            for member in data.get("query", {}).get("categorymembers", []):
                if member["ns"] == 0: # Article namespace
                    titles.append(member["title"])
        except json.JSONDecodeError:
            logger.error(f"JSON decode error fetching members for {category} in {lang}. Response might not be JSON.")
        except Exception as e:
            logger.error(f"Error fetching members for {category} in {lang}: {e}")
        return titles

    def fetch_article_content(self, lang: str, title: str) -> Optional[str]:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "exlimit": "1",
            "titles": title,
            "explaintext": "1",
            "format": "json"
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"Error {response.status_code} fetching content for {title} in {lang}")
                return None
            if not response.content:
                logger.error(f"Empty response fetching content for {title} in {lang}")
                return None
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for pid in pages:
                return pages[pid].get("extract")
        except json.JSONDecodeError:
            logger.error(f"JSON decode error fetching content for {title} in {lang}. Response might not be JSON.")
        except Exception as e:
            logger.error(f"Error fetching content for {title} in {lang}: {e}")
        return None

    def run_collection(self):
        for domain, categories in tqdm(DOMAINS.items(), desc="Domains"):
            domain_data = []
            logger.info(f"Collecting domain: {domain}")
            for cat in tqdm(categories, desc=f"Categories ({domain})", leave=False):
                for lang in LANGS:
                    titles = self.fetch_category_members(lang, cat)
                    for title in tqdm(titles, desc=f"Articles ({lang}:{cat})", leave=False):
                        content = self.fetch_article_content(lang, title)
                        if content:
                            domain_data.append({
                                "title": title,
                                "lang": lang,
                                "domain": domain,
                                "category": cat,
                                "text": content
                            })
            
            if domain_data:
                output_path = self.output_dir / f"wikipedia_{domain}.jsonl"
                with open(output_path, "w", encoding="utf-8") as f:
                    for entry in domain_data:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                logger.info(f"Saved {len(domain_data)} articles for {domain} to {output_path}")

if __name__ == "__main__":
    fetcher = WikipediaSpecializedFetcher(Path("data/collected_2025_2026/wikipedia_v4"))
    fetcher.run_collection()
