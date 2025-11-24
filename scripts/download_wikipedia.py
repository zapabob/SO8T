# download_wikipedia.py
"""Download Wikipedia pages for Physics and Mathematics categories.

Usage:
    python download_wikipedia.py --output D:\\dataset\\wikipedia
"""
import argparse
import requests
from pathlib import Path
import time

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "AEGIS_Bot/1.0 (http://example.com/agiasi; agiasi@example.com)"
}

def get_category_members(category: str, max_results: int = 50):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": max_results,
        "format": "json"
    }
    response = requests.get(WIKI_API_URL, params=params, headers=HEADERS)
    data = response.json()
    return data.get("query", {}).get("categorymembers", [])

def get_page_content(pageid: int):
    params = {
        "action": "query",
        "prop": "extracts",
        "pageids": pageid,
        "explaintext": True,
        "format": "json"
    }
    response = requests.get(WIKI_API_URL, params=params, headers=HEADERS)
    data = response.json()
    page = data.get("query", {}).get("pages", {}).get(str(pageid), {})
    return page.get("extract", "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    categories = [
        # Academic
        "Physics", "Mathematics", "Linguistics", "History", "English_grammar", "Japanese_grammar",
        # Critical Infrastructure & Business
        "Military", "Aerospace_engineering", "Transport", "Public_utility", "Computer_security", "Business_software",
        # Safety, Law, Drugs, NSFW context
        "Pharmacology", "Illegal_drug_trade", "Human_sexuality", "Law", "E-government"
    ]
    for cat in categories:
        print(f"Fetching category: {cat}")
        members = get_category_members(cat)
        for member in members:
            pageid = member["pageid"]
            title = member["title"]
            print(f"  Downloading {title}...")
            content = get_page_content(pageid)
            if content:
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip()
                filename = args.output / f"{cat}_{safe_title}.txt"
                filename.write_text(content, encoding="utf-8")
            time.sleep(0.2)

if __name__ == "__main__":
    main()
