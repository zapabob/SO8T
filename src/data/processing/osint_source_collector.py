#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OSINT source collector for pop-culture and world-affairs.

Fetches sources from configured RSS feeds and GDELT queries and stores
normalized JSONL for downstream enrichment.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests
import yaml
from xml.etree import ElementTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/osint_sources.yaml")


def load_sources_config(path: Optional[Path] = None) -> Dict[str, object]:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        logger.warning("OSINT config not found: %s", cfg_path)
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to parse OSINT config: %s", exc)
        return {}


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _parse_rss_items(content: str) -> List[Dict[str, str]]:
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return []

    items: List[Dict[str, str]] = []

    # RSS 2.0
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        summary = item.findtext("description") or ""
        if link:
            items.append({"title": title, "url": link, "published": pub_date, "summary": summary})

    # Atom
    atom_ns = "{http://www.w3.org/2005/Atom}"
    for entry in root.findall(f".//{atom_ns}entry"):
        title = entry.findtext(f"{atom_ns}title") or ""
        link_el = entry.find(f"{atom_ns}link")
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        published = entry.findtext(f"{atom_ns}published") or entry.findtext(f"{atom_ns}updated") or ""
        summary = entry.findtext(f"{atom_ns}summary") or ""
        if link:
            items.append({"title": title, "url": link, "published": published, "summary": summary})

    return items


def fetch_rss(url: str, max_items: int = 200, timeout: int = 15) -> List[Dict[str, str]]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("RSS fetch failed: %s (%s)", url, exc)
        return []

    items = _parse_rss_items(resp.text)
    return items[:max_items]


def fetch_gdelt(query: str, max_records: int = 200, timespan: str = "365d", timeout: int = 15) -> List[Dict[str, str]]:
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_records,
        "format": "json",
        "timespan": timespan,
    }
    try:
        resp = requests.get(endpoint, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("GDELT fetch failed: %s", exc)
        return []

    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    items: List[Dict[str, str]] = []
    for article in articles:
        url = article.get("url") or ""
        if not url:
            continue
        items.append(
            {
                "title": article.get("title", ""),
                "url": url,
                "published": article.get("seendate", ""),
                "summary": article.get("snippet", ""),
                "source": article.get("sourceCommonName", ""),
                "domain": article.get("domain", "") or _domain_from_url(url),
                "country": article.get("sourceCountry", ""),
                "tone": str(article.get("tone", "")),
            }
        )
    return items


def _normalize_item(item: Dict[str, str], source_name: str, category: str) -> Dict[str, object]:
    url = item.get("url", "")
    return {
        "title": item.get("title") or item.get("event") or "",
        "summary": item.get("summary") or "",
        "published": item.get("published") or "",
        "url": url,
        "source": item.get("source") or source_name,
        "domain": item.get("domain") or _domain_from_url(url),
        "category": category,
    }


def collect_sources(
    domain: str,
    config: Optional[Dict[str, object]] = None,
    output_dir: Optional[Path] = None,
    max_items: int = 200,
) -> Optional[Path]:
    domain_key = domain
    cfg = config or {}
    domain_cfg = cfg.get(domain_key, {}) if isinstance(cfg, dict) else {}
    if not domain_cfg:
        logger.warning("No OSINT config for domain: %s", domain)

    output_root = output_dir or Path("data") / domain
    output_root.mkdir(parents=True, exist_ok=True)

    collected: List[Dict[str, object]] = []
    dedup = set()

    for rss in domain_cfg.get("rss", []) if isinstance(domain_cfg, dict) else []:
        url = rss.get("url") if isinstance(rss, dict) else None
        if not url:
            continue
        items = fetch_rss(url, max_items=max_items)
        source_name = rss.get("name", "rss") if isinstance(rss, dict) else "rss"
        for item in items:
            normalized = _normalize_item(item, source_name, domain_key)
            if normalized["url"] and normalized["url"] not in dedup:
                dedup.add(normalized["url"])
                collected.append(normalized)

    for gdelt_cfg in domain_cfg.get("gdelt", []) if isinstance(domain_cfg, dict) else []:
        query = gdelt_cfg.get("query") if isinstance(gdelt_cfg, dict) else None
        if not query:
            continue
        items = fetch_gdelt(
            query=query,
            max_records=int(gdelt_cfg.get("max_records", max_items)),
            timespan=gdelt_cfg.get("timespan", "365d"),
        )
        source_name = gdelt_cfg.get("name", "gdelt") if isinstance(gdelt_cfg, dict) else "gdelt"
        for item in items:
            normalized = _normalize_item(item, source_name, domain_key)
            if normalized["url"] and normalized["url"] not in dedup:
                dedup.add(normalized["url"])
                collected.append(normalized)

    if not collected:
        logger.warning("No sources collected for domain: %s", domain)
        return None

    output_path = output_root / "sources.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in collected:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary_path = output_root / "sources_summary.json"
    summary = {
        "domain": domain,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count": len(collected),
        "sources": sorted({entry.get("source") for entry in collected if entry.get("source")}),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Collected %d OSINT sources for %s -> %s", len(collected), domain, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OSINT source collector")
    parser.add_argument("--domain", choices=["pop_culture", "world_affairs", "all"], default="all")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="OSINT config YAML")
    parser.add_argument("--output-dir", default="", help="Output directory base")
    parser.add_argument("--max-items", type=int, default=int(os.getenv("SO8T_OSINT_MAX_ITEMS", "200")))
    args = parser.parse_args()

    config = load_sources_config(Path(args.config))
    output_dir = Path(args.output_dir) if args.output_dir else None

    domains = ["pop_culture", "world_affairs"] if args.domain == "all" else [args.domain]
    for domain in domains:
        collect_sources(domain, config=config, output_dir=output_dir, max_items=args.max_items)


if __name__ == "__main__":
    main()
