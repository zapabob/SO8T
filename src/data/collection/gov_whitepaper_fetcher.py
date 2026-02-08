import os
import json
import logging
from pathlib import Path
import requests
from tqdm import tqdm
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Target sources: US DOD, JP MOD, Aerospace Agencies, Financial Reports
GOV_SOURCES = [
    {"name": "DOD_Annual_Report", "url": "https://www.defense.gov/News/Publications/", "domain": "military"},
    {"name": "NASA_Tech_Reports", "url": "https://ntrs.nasa.gov/", "domain": "aerospace"},
    {"name": "MOD_Whitepaper_Japan", "url": "https://www.mod.go.jp/en/publ/w_paper/", "domain": "military"},
    {"name": "Finance_Gov_JP", "url": "https://www.mof.go.jp/english/", "domain": "finance"}
]

class GovWhitepaperFetcher:
    """
    Fetches government-published white papers and OSINT summaries.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def fetch_summaries(self, source: Dict[str, str]) -> List[Dict[str, Any]]:
        logger.info(f"Fetching from {source['name']}: {source['url']}")
        # In a real tool, this would parse HTML or use a search API
        # Mocking 3 high-value documents per source
        docs = []
        for i in range(3):
            docs.append({
                "source": source["name"],
                "url": source["url"] + f"report_{i}.pdf",
                "title": f"Future of {source['domain'].capitalize()} - Strategic Assessment {2024+i}",
                "domain": source["domain"],
                "summary": f"A comprehensive report detailing the {source['domain']} strategy for the next decade.",
                "date": f"202{4+i}-0{i+1}-01"
            })
        return docs

    def run(self):
        all_docs = []
        for source in tqdm(GOV_SOURCES, desc="Gov Sources"):
            docs = self.fetch_summaries(source)
            all_docs.extend(docs)
            
        if all_docs:
            output_path = self.output_dir / "gov_whitepaper_v4.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in all_docs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(all_docs)} whitepaper entries to {output_path}")

if __name__ == "__main__":
    fetcher = GovWhitepaperFetcher(Path("data/collected_2025_2026/osint_v4"))
    fetcher.run()
