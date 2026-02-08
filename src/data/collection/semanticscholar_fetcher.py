import json
import argparse
import time
import requests
import os
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to sys.path for internal imports
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.vssi_template import render_thinking

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SemanticScholarFetcher:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,authors,citationCount,publicationDate,externalIds,venue,year,s2FieldsOfStudy"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
        self.session.headers["User-Agent"] = "SO8T-Pipeline/1.0"
        
        self.use_quadruple_tokens = os.getenv("SO8T_QUADRUPLE_TOKENS", "0") == "1"
        self.think_tag_style = os.getenv("SO8T_THINK_TAG_STYLE", "xml").strip().lower()

    def fetch_papers(self, query: str, max_papers: int = 50000, year_range: str = "2024-2026", existing_ids: set = None) -> List[Dict[str, Any]]:
        logger.info(f"[S2] Fetching Semantic Scholar papers for query: '{query}', target: {max_papers}")
        existing_ids = existing_ids or set()
        
        endpoint = f"{self.BASE_URL}/paper/search/bulk"
        params = {
            "query": query,
            "fields": self.FIELDS,
            "year": year_range,
            "limit": 1000  # Max limit per request for bulk search
        }
        
        papers = []
        token = None
        
        try:
            while len(papers) < max_papers:
                if token:
                    params["token"] = token
                
                response = self.session.get(endpoint, params=params, timeout=60)
                if response.status_code == 429:
                    logger.warning("[S2] Rate limited. Waiting 10s...")
                    time.sleep(10)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                fetched_count = 0
                for paper in data.get("data", []):
                    if paper.get("abstract"):
                        p_id = paper.get("paperId")
                        if p_id and p_id in existing_ids:
                            continue
                            
                        processed = self._format_to_vssi(paper)
                        papers.append(processed)
                        fetched_count += 1
                        
                    if len(papers) >= max_papers:
                        break
                
                logger.info(f"[S2] Collected {len(papers)} papers so far...")
                
                token = data.get("token")
                if not token:
                    logger.info("[S2] No more papers available for this query.")
                    break
                
                # S2 Bulk Search recommended delay (0.5s with API key, 1s without)
                time.sleep(0.5 if self.api_key else 1.0)
                    
        except Exception as e:
            logger.error(f"[S2] Failed to fetch papers from Semantic Scholar: {e}")
            
        return papers

    def _format_to_vssi(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Sophisticated Arxiv-style VSSI formatting."""
        title = paper.get("title", "Unknown Title")
        abstract = paper.get("abstract", "")
        authors_list = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
        citations = paper.get("citationCount", 0)
        domain = self._classify_domain(paper)
        reasoning_type = self._classify_reasoning_type(paper, abstract)
        
        # 1. Observation
        observation = {
            'phase': 'observation',
            'dimension': 'vector',
            'key_facts': {
                'title': title,
                'citation_count': citations,
                'published_year': paper.get('year'),
                'authors': authors_list[:5]
            },
            'domain': domain,
            'extracted_numbers': re.findall(r'\b\d+\.?\d*\b', abstract)[:10]
        }
        
        # 2. Deduction
        deduction = {
            'phase': 'deduction',
            'dimension': 'positive_spinor',
            'reasoning_type': reasoning_type,
            'theoretical_keywords': [kw for kw in ['algorithm', 'model', 'theorem', 'method'] if kw in abstract.lower()],
            'domain_approaches': ['theoretical', 'computational'],
            'logical_structure': 'standard_solution_path'
        }
        
        # 3. Abduction
        abduction = {
            'phase': 'abduction',
            'dimension': 'negative_spinor',
            'edge_case_keywords': [kw for kw in ['limitation', 'exception', 'assumption'] if kw in abstract.lower()],
            'alternative_approaches': ['critical_analysis'],
            'challenges_to_deduction': 'standard_review'
        }
        
        # 4. Integration
        integration = {
            'phase': 'integration',
            'dimension': 'quadrality',
            'quality_score': 0.75,
            'final_reasoning': 'grokking_synthesis_complete',
            'integrated_insights': {'synthesis': 'convergent'}
        }
        
        # Render thinking format matching process_arxiv_biorxiv.py
        task_block = (
            "[Vector_State]\n"
            f"- Title: {title}\n"
            f"- Domain: {domain}\n"
            f"- Citations: {citations}\n"
            f"- Authors: {', '.join(authors_list[:3])}"
        )
        analysis_block = (
            "[Spinor_Plus_Logic] (Deduction)\n"
            f"- Reasoning Type: {reasoning_type}\n"
            f"- Methodological Logic: Linear"
        )
        safety_block = (
            "[Spinor_Minus_Synthesis] (Abduction)\n"
            f"- Challenges: {abduction.get('challenges_to_deduction')}"
        )
        policy_block = (
            "[Quadrality_Integration] (Synthesis)\n"
            f"- VSSI Quality Score: {integration.get('quality_score'):.4f}\n"
            f"- Final Decision: Approved"
        )
        
        thinking_content = render_thinking(
            task_block,
            safety_block,
            policy_block,
            analysis_block=analysis_block,
            use_quadruple=self.use_quadruple_tokens,
            style=self.think_tag_style,
        )
        
        final_answer = abstract[:500]
        
        return {
            "instruction": f"Summarize and critique the paper using VSSI reasoning: {title}",
            "input": abstract,
            "output": f"{thinking_content}\n<final>\n{final_answer}\n</final>",
            "metadata": {
                "source": "semanticscholar",
                "title": title,
                "authors": authors_list,
                "citation_count": citations,
                "year": paper.get("year"),
                "paper_id": paper.get("paperId"),
                "domain": domain,
                "reasoning_type": reasoning_type,
                "generated_at": datetime.now().isoformat()
            }
        }

    def _classify_domain(self, paper: Dict[str, Any]) -> str:
        s2_fields = [f.get("category", "") for f in paper.get("s2FieldsOfStudy", []) if f.get("category")]
        fields_str = " ".join(s2_fields).lower()
        
        if any(kw in fields_str for kw in ["mathematics", "math"]):
            return "mathematics"
        if "physics" in fields_str:
            return "physics"
        if any(kw in fields_str for kw in ["biology", "biomedical"]):
            return "biology"
        if any(kw in fields_str for kw in ["computer science", "artificial intelligence"]):
            return "ai_ml"
        return "science"

    def _classify_reasoning_type(self, paper: Dict[str, Any], text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['proof', 'theorem', 'lemma']):
            return 'theoretical_proof'
        elif any(kw in text_lower for kw in ['algorithm', 'method', 'architecture']):
            return 'methodological'
        return 'analytical'

def main():
    parser = argparse.ArgumentParser(description='Semantic Scholar Data Fetcher')
    parser.add_argument('--query', type=str, help='Search query (e.g. "mathematics physics")')
    parser.add_argument('--max-papers', type=int, default=100, help='Max papers to fetch')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    
    args = parser.parse_args()
    
    fetcher = SemanticScholarFetcher()
    
    # Science/Math focused queries if no specific query provided
    query = args.query or "mathematics physics computer science biology"
    
    # Load existing IDs from all JSONL files in the output directory
    existing_ids = set()
    if os.getenv("SO8T_SKIP_EXISTING", "1") == "1":
        output_path = Path(args.output)
        scan_dir = output_path.parent
        if scan_dir.exists():
            logger.info(f"[S2] Scanning {scan_dir} for existing paper IDs...")
            for jsonl_file in scan_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip(): continue
                            data = json.loads(line)
                            # Handle both Arxiv-style 'id' and S2-style 'metadata.paper_id'
                            if 'id' in data:
                                existing_ids.add(data['id'])
                            elif 'metadata' in data and 'paper_id' in data['metadata']:
                                existing_ids.add(data['metadata']['paper_id'])
                except Exception as e:
                    logger.warning(f"Failed to load existing IDs from {jsonl_file}: {e}")
            logger.info(f"Loaded {len(existing_ids)} existing paper IDs from {scan_dir}")
    
    papers = fetcher.fetch_papers(query, args.max_papers, existing_ids=existing_ids)
    
    if papers:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode to handle large fetches incrementally if needed, 
        # but here we overwrite for a fresh run as per script behavior
        with open(output_path, "w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved {len(papers)} papers to {args.output}")
    else:
        logger.warning("No papers fetched.")

if __name__ == "__main__":
    main()
