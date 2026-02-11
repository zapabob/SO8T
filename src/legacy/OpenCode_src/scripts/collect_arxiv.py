# collect_arxiv.py
"""Collect arXiv papers for specified categories and save as plain text.

Usage:
    python collect_arxiv.py --category cs.AI --max 100 --output D:\\dataset\\arxiv
"""
import argparse
import os
import json
import time
from pathlib import Path
import requests

API_URL = "http://export.arxiv.org/api/query"

def fetch_entries(category: str, max_results: int = 100):
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.text

def parse_feed(feed_xml: str):
    # Very lightweight parsing – extract title, id, summary
    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(feed_xml)
    entries = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        id_url = entry.find("atom:id", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        entries.append({"title": title, "id": id_url, "summary": summary})
    return entries

def save_entries(entries, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(entries, 1):
        safe_title = "_".join(e["title"].split())[:100]
        filename = out_dir / f"{i:04d}_{safe_title}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(e["summary"])
        time.sleep(0.2)  # be gentle to the API

def main():
    parser = argparse.ArgumentParser(description="Download arXiv abstracts.")
    parser.add_argument("--category", type=str, required=True, help="arXiv category, e.g. cs.AI")
    parser.add_argument("--max", type=int, default=100, help="Maximum number of papers")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    feed = fetch_entries(args.category, args.max)
    entries = parse_feed(feed)
    save_entries(entries, Path(args.output))
    print(f"Saved {len(entries)} entries to {args.output}")

if __name__ == "__main__":
    main()
