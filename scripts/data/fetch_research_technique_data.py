#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch real papers (ArXiv) and GitHub repos for advanced technique training data.
Targets: mHC, GRPO/DeepSeek-V3, GRAPE, SO8T/VSSI, imatrix, SakanaAI
Outputs VSSI-tagged JSONL for SFT.
"""

import json
import os
import re
import sys
import time
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests

# ArXiv search
try:
    import arxiv
    ARXIV_OK = True
except ImportError:
    ARXIV_OK = False

# PDF text extraction
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.vssi_template import render_thinking

OUTPUT_DIR = PROJECT_ROOT / "data" / "research_techniques"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"advanced_techniques_vssi_{datetime.now().strftime('%Y%m%d')}.jsonl"

# ---- ArXiv paper IDs and queries per topic ----
ARXIV_TARGETS = {
    "grpo_deepseek": {
        "ids": ["2412.19437"],  # DeepSeek-V3 Technical Report
        "queries": ["GRPO group relative policy optimization", "DeepSeek reinforcement learning math reasoning"],
        "max_query_results": 15,
    },
    "sakana_ai": {
        "ids": ["2403.13187"],  # Evolutionary Optimization of Model Merging Recipes
        "queries": ["evolutionary model merging CMA-ES", "Sakana AI model merge optimization"],
        "max_query_results": 10,
    },
    "grape_position_encoding": {
        "ids": [],
        "queries": ["rotary position encoding graph structure", "GRAPE position encoding neural network",
                     "multiplicative additive position encoding transformer"],
        "max_query_results": 10,
    },
    "manifold_harmonic": {
        "ids": [],
        "queries": ["manifold harmonic correction neural network", "Riemannian geometry latent space alignment",
                     "hyper-connections manifold constrained deep learning"],
        "max_query_results": 10,
    },
    "imatrix_quantization": {
        "ids": [],
        "queries": ["importance matrix quantization LLM", "GGUF quantization calibration llama.cpp",
                     "quantization aware importance weighting language model"],
        "max_query_results": 10,
    },
    "so8_quadrality": {
        "ids": [],
        "queries": ["SO(8) triality group theory neural", "quadrality reasoning multi-perspective inference",
                     "spinor vector integration reasoning architecture"],
        "max_query_results": 10,
    },
}

# ---- GitHub repos to fetch README + key source files ----
GITHUB_REPOS = [
    {
        "owner": "deepseek-ai", "repo": "DeepSeek-V3",
        "topic": "grpo_deepseek",
        "files": ["README.md", "README_WEIGHTS.md"],
    },
    {
        "owner": "SakanaAI", "repo": "evolutionary-model-merge",
        "topic": "sakana_ai",
        "files": ["README.md"],
    },
    {
        "owner": "ggerganov", "repo": "llama.cpp",
        "topic": "imatrix_quantization",
        "files": ["examples/imatrix/README.md", "examples/quantize/README.md"],
    },
]


def fetch_arxiv_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """Fetch single ArXiv paper metadata + PDF text by ID."""
    if not ARXIV_OK:
        return None
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        results = list(client.results(search))
        if not results:
            return None
        r = results[0]
        paper = {
            "arxiv_id": paper_id,
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "summary": r.summary,
            "published": r.published.isoformat(),
            "categories": r.categories,
            "pdf_url": r.pdf_url,
            "source": "arxiv",
        }
        # Download and extract PDF text
        text = download_and_extract_pdf(r.pdf_url, paper_id)
        if text:
            paper["full_text"] = text
        return paper
    except Exception as e:
        print(f"[WARN] Failed to fetch ArXiv ID {paper_id}: {e}")
        return None


def fetch_arxiv_by_query(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search ArXiv by query and return paper metadata."""
    if not ARXIV_OK:
        return []
    papers = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for r in client.results(search):
            papers.append({
                "arxiv_id": r.entry_id.split("/")[-1],
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "summary": r.summary,
                "published": r.published.isoformat(),
                "categories": r.categories,
                "pdf_url": r.pdf_url,
                "source": "arxiv",
            })
        time.sleep(3)  # rate limit
    except Exception as e:
        print(f"[WARN] ArXiv query failed '{query}': {e}")
    return papers


def download_and_extract_pdf(pdf_url: str, paper_id: str) -> Optional[str]:
    """Download PDF and extract text."""
    if not PDF_OK or not pdf_url:
        return None
    try:
        resp = requests.get(pdf_url, timeout=60)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        text_parts = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages[:30]:  # max 30 pages
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        os.unlink(tmp_path)
        full = "\n\n".join(text_parts)
        return full[:50000] if full else None  # cap at 50k chars
    except Exception as e:
        print(f"[WARN] PDF extraction failed for {paper_id}: {e}")
        return None


def fetch_github_file(owner: str, repo: str, filepath: str) -> Optional[str]:
    """Fetch a single file from GitHub raw."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{filepath}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text[:50000]
        # Try master branch
        url2 = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{filepath}"
        resp2 = requests.get(url2, timeout=30)
        if resp2.status_code == 200:
            return resp2.text[:50000]
    except Exception as e:
        print(f"[WARN] GitHub fetch failed {owner}/{repo}/{filepath}: {e}")
    return None


def paper_to_vssi_samples(paper: Dict[str, Any], topic: str) -> List[Dict[str, Any]]:
    """Convert a paper into multiple VSSI-tagged training samples."""
    samples = []
    title = paper.get("title", "")
    summary = paper.get("summary", "")
    full_text = paper.get("full_text", "")
    categories = paper.get("categories", [])
    authors = ", ".join(paper.get("authors", [])[:5])

    # Sample 1: Paper summarization with VSSI reasoning
    if summary:
        thinking = render_thinking(
            task_block=(
                f"[Vector_State]\n"
                f"- Paper: {title}\n"
                f"- Authors: {authors}\n"
                f"- Categories: {', '.join(categories[:3])}\n"
                f"- Topic: {topic}"
            ),
            safety_block=(
                f"[Spinor_Minus_Safety]\n"
                f"- Verify claims against established theory\n"
                f"- Check reproducibility concerns\n"
                f"- Note any limitations stated by authors"
            ),
            policy_block=(
                f"[Quadrality_Integration]\n"
                f"- Synthesize key contributions\n"
                f"- Assess practical applicability for SO8T pipeline\n"
                f"- Integration priority for training pipeline"
            ),
            analysis_block=(
                f"[Spinor_Plus_Logic]\n"
                f"- Core methodology analysis\n"
                f"- Mathematical foundations\n"
                f"- Relation to existing techniques in SO8T"
            ),
            use_quadruple=True,
            style="xml",
        )
        samples.append({
            "instruction": "Analyze and summarize the following research paper, explaining its methodology and key contributions.",
            "input": f"Title: {title}\nAuthors: {authors}\nAbstract: {summary}",
            "output": f"{thinking}\n<final>{summary}</final>",
            "metadata": {"topic": topic, "arxiv_id": paper.get("arxiv_id"), "type": "paper_summary"},
        })

    # Sample 2: Technical explanation from full text
    if full_text and len(full_text) > 500:
        # Extract a meaningful chunk (method section or intro)
        chunks = re.split(r'\n(?:Introduction|Method|Approach|Experiment|Result|Conclusion|Abstract)\s*\n', full_text, flags=re.IGNORECASE)
        for i, chunk in enumerate(chunks[:3]):
            chunk = chunk.strip()
            if len(chunk) < 200:
                continue
            chunk_text = chunk[:3000]
            thinking = render_thinking(
                task_block=f"[Vector_State]\n- Paper: {title}\n- Section {i+1} content analysis\n- Topic: {topic}",
                safety_block=f"[Spinor_Minus_Safety]\n- Verify technical accuracy\n- Flag any unsupported claims",
                policy_block=f"[Quadrality_Integration]\n- Extract actionable insights for implementation\n- Prioritize for SO8T integration",
                analysis_block=f"[Spinor_Plus_Logic]\n- Detailed technical breakdown\n- Mathematical formulation review",
                use_quadruple=True,
                style="xml",
            )
            samples.append({
                "instruction": f"Explain the technical details from this section of the paper '{title}'.",
                "input": chunk_text,
                "output": f"{thinking}\n<final>{chunk_text[:2000]}</final>",
                "metadata": {"topic": topic, "arxiv_id": paper.get("arxiv_id"), "type": "technical_detail", "section": i},
            })

    return samples


def github_to_vssi_samples(content: str, repo_info: Dict, filepath: str) -> List[Dict[str, Any]]:
    """Convert GitHub file content into VSSI training samples."""
    samples = []
    topic = repo_info["topic"]
    repo_name = f"{repo_info['owner']}/{repo_info['repo']}"

    # Split content into meaningful chunks
    sections = re.split(r'\n#{1,3}\s+', content)
    for i, section in enumerate(sections[:10]):
        section = section.strip()
        if len(section) < 100:
            continue
        chunk = section[:3000]
        thinking = render_thinking(
            task_block=f"[Vector_State]\n- Repository: {repo_name}\n- File: {filepath}\n- Section {i+1}\n- Topic: {topic}",
            safety_block=f"[Spinor_Minus_Safety]\n- Check code quality and security\n- Verify documentation accuracy",
            policy_block=f"[Quadrality_Integration]\n- Extract implementation patterns\n- Assess integration with SO8T pipeline",
            analysis_block=f"[Spinor_Plus_Logic]\n- Code structure analysis\n- API and interface design review",
            use_quadruple=True,
            style="xml",
        )
        samples.append({
            "instruction": f"Analyze the following documentation/code from the {repo_name} repository.",
            "input": chunk,
            "output": f"{thinking}\n<final>{chunk[:2000]}</final>",
            "metadata": {"topic": topic, "repo": repo_name, "file": filepath, "type": "github_doc", "section": i},
        })

    return samples


def main():
    all_samples = []
    seen_ids = set()

    print("=" * 60)
    print("  Advanced Techniques Data Collector (ArXiv + GitHub)")
    print("  mHC / GRPO / GRAPE / SO8T / imatrix / SakanaAI")
    print("=" * 60)

    # ---- Phase 1: ArXiv papers ----
    for topic, config in ARXIV_TARGETS.items():
        print(f"\n[TOPIC] {topic}")

        # Fetch by ID
        for pid in config.get("ids", []):
            if pid in seen_ids:
                continue
            print(f"  [ID] Fetching ArXiv {pid}...")
            paper = fetch_arxiv_by_id(pid)
            if paper:
                seen_ids.add(pid)
                samples = paper_to_vssi_samples(paper, topic)
                all_samples.extend(samples)
                print(f"    -> {len(samples)} samples from {paper['title'][:60]}")
            time.sleep(2)

        # Fetch by query
        for query in config.get("queries", []):
            print(f"  [QUERY] {query}")
            papers = fetch_arxiv_by_query(query, max_results=config.get("max_query_results", 10))
            for paper in papers:
                pid = paper.get("arxiv_id", "")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                samples = paper_to_vssi_samples(paper, topic)
                all_samples.extend(samples)
            print(f"    -> {len(papers)} papers found")

    # ---- Phase 2: GitHub repos ----
    print("\n[GITHUB] Fetching repository files...")
    for repo_info in GITHUB_REPOS:
        repo_name = f"{repo_info['owner']}/{repo_info['repo']}"
        for filepath in repo_info.get("files", []):
            print(f"  [FILE] {repo_name}/{filepath}")
            content = fetch_github_file(repo_info["owner"], repo_info["repo"], filepath)
            if content:
                samples = github_to_vssi_samples(content, repo_info, filepath)
                all_samples.extend(samples)
                print(f"    -> {len(samples)} samples")
            time.sleep(1)

    # ---- Phase 3: Write output ----
    print(f"\n[SAVE] Writing {len(all_samples)} samples to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[OK] Done! {len(all_samples)} VSSI training samples generated")
    print(f"[OUTPUT] {OUTPUT_FILE}")
    print(f"[SIZE] {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
