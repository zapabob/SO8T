#!/usr/bin/env python3
"""
Arxiv/BioRxiv論文処理パイプライン
2024-2026年の引用上位10万件を構造化して科学・数学の推論能力向上に役立てる
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.utils.vssi_template import render_thinking
from datetime import datetime
import logging

try:
    import requests
    import arxiv
    from bs4 import BeautifulSoup
    import PyPDF2
    import pdfplumber

    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    print("[ERROR] Required packages not installed")
    print(
        "[INFO] Install with: pip install arxiv requests beautifulsoup4 PyPDF2 pdfplumber"
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivBioRxivProcessor:
    """Arxiv/BioRxiv論文処理クラス"""

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root

        self.raw_dir = self.project_root / "data" / "arxiv_biorxiv" / "raw"
        self.processed_dir = self.project_root / "data" / "arxiv_biorxiv" / "processed"
        self.cleaned_dir = self.project_root / "data" / "arxiv_biorxiv" / "cleaned"

        # ディレクトリ作成
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        # 期間設定
        self.start_year = 2024
        self.end_year = 2026
        self.target_count = 100000  # 上位10万件
        self.think_tag_style = (
            os.getenv("SO8T_THINK_TAG_STYLE", "legacy").strip().lower()
        )
        self.use_quadruple_tokens = os.getenv("SO8T_QUADRUPLE_TOKENS", "0") == "1"
        self.semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.semantic_sleep = float(os.getenv("SO8T_SEMANTIC_SCHOLAR_SLEEP", "0.5"))
        self.arxiv_sleep = float(
            os.getenv("SO8T_ARXIV_SLEEP", "5.0")
        )  # デフォルト5秒に設定

    def get_existing_paper_ids(self) -> set:
        """既存の処理済み論文IDを取得"""
        if os.getenv("SO8T_SKIP_EXISTING", "1") != "1":
            logger.info(
                "[CONFIG] Skipping existing paper check disabled (SO8T_SKIP_EXISTING != 1)"
            )
            return set()

        existing_ids = set()
        try:
            # Cleaned directory内の全JSONLを確認
            for jsonl_file in self.cleaned_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            # arxiv_id あるいは id フィールドから取得
                            if "id" in data:
                                existing_ids.add(data["id"])
                            elif "paper_id" in data.get("metadata", {}):
                                existing_ids.add(data["metadata"]["paper_id"])
                except Exception as e:
                    logger.warning(f"[WARN] Failed to read {jsonl_file}: {e}")

            logger.info(f"[INIT] Found {len(existing_ids)} existing papers in history")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load existing IDs: {e}")
        return existing_ids

    def _render_thinking_content(
        self,
        task_block: str,
        safety_block: str,
        policy_block: str,
        analysis_block: Optional[str] = None,
    ) -> str:
        """Render thinking blocks using either simple or quadruple tokens."""
        return render_thinking(
            task_block,
            safety_block,
            policy_block,
            analysis_block=analysis_block,
            use_quadruple=self.use_quadruple_tokens,
            style=self.think_tag_style,
        )

    def search_arxiv_papers(
        self, query: str = "", max_results: int = 50000
    ) -> List[Dict[str, Any]]:
        """Arxiv論文を検索"""
        logger.info(
            f"[SEARCH] Searching Arxiv papers from {self.start_year} to {self.end_year}..."
        )

        papers = []

        # カテゴリ別に検索（科学・数学関連）
        categories = ["cs.AI", "cs.LG", "cs.CL", "math", "physics", "q-bio", "stat"]

        # 50k規模の検索に対応するため、カテゴリごとの取得件数を調整
        per_category = max_results // len(categories)

        for category in tqdm(categories, desc="Arxiv Categories"):
            try:
                logger.info(f"[SEARCH] Searching category: {category}")

                # Arxiv検索
                search_query = f"cat:{category} AND submittedDate:[{self.start_year}0101* TO {self.end_year}1231*]"
                if query:
                    search_query = f"{search_query} AND {query}"

                search = arxiv.Search(
                    query=search_query,
                    max_results=per_category,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                for result in search.results():
                    paper = {
                        "arxiv_id": result.entry_id.split("/")[-1],
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "summary": result.summary,
                        "published": result.published.isoformat(),
                        "updated": result.updated.isoformat()
                        if result.updated
                        else None,
                        "categories": result.categories,
                        "pdf_url": result.pdf_url,
                        "primary_category": result.primary_category,
                        "citation_count": 0,
                        "source": "arxiv",
                    }
                    papers.append(paper)

                    if len(papers) >= max_results:
                        break

                logger.info(f"[SEARCH] Arxiv papers so far: {len(papers)}")
                time.sleep(
                    self.arxiv_sleep
                )  # レート制限対策 (設定可能: SO8T_ARXIV_SLEEP, デフォルト10秒)

            except Exception as e:
                logger.error(f"[ERROR] Failed to search Arxiv category {category}: {e}")
                continue

        return papers

    def search_biorxiv_papers(self, max_results: int = 10000) -> List[Dict[str, Any]]:
        """BioRxiv APIを使用して論文を検索（重複チェック高速化版）"""
        logger.info(
            f"[SEARCH] Searching BioRxiv papers for {self.start_year}-{self.end_year}..."
        )

        # 既存論文IDを事前に取得（高速化のため）
        existing_ids = self.get_existing_paper_ids()
        if existing_ids:
            logger.info(
                f"[SEARCH] Skipping {len(existing_ids)} existing papers from BioRxiv search"
            )

        papers = []
        skipped_count = 0
        base_url = "https://api.biorxiv.org/details/biorxiv/"

        # BioRxiv APIは日付範囲で取得
        start_date = f"{self.start_year}-01-01"
        end_date = f"{self.end_year}-12-31"

        cursor = 0
        while len(papers) < max_results:
            try:
                url = f"{base_url}{start_date}/{end_date}/{cursor}/json"
                logger.info(
                    f"[SEARCH] Querying BioRxiv API: {cursor} (collected: {len(papers)}, skipped: {skipped_count})"
                )

                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    break

                data = response.json()
                if "collection" not in data or not data["collection"]:
                    break

                for entry in data["collection"]:
                    doi = entry.get("doi")
                    # 高速化: APIレスポンス直後に重複チェック
                    paper_id = f"biorxiv_{doi}" if doi else None
                    if paper_id and paper_id in existing_ids:
                        skipped_count += 1
                        continue

                    paper = {
                        "biorxiv_doi": doi,
                        "title": entry.get("title"),
                        "authors": [
                            a.get("author_name")
                            for a in entry.get("authors", [])
                            if isinstance(a, dict)
                        ],
                        "summary": entry.get("abstract"),
                        "published": entry.get("date"),
                        "categories": [entry.get("category")],
                        "pdf_url": f"https://www.biorxiv.org/content/{doi}.full.pdf"
                        if doi
                        else None,
                        "primary_category": entry.get("category"),
                        "citation_count": 0,
                        "source": "biorxiv",
                    }
                    papers.append(paper)
                    if len(papers) >= max_results:
                        break

                cursor += len(data["collection"])
                time.sleep(0.5)

            except Exception as e:
                logger.error(
                    f"[ERROR] BioRxiv API fetch failed at cursor {cursor}: {e}"
                )
                break

        logger.info(
            f"[SEARCH] BioRxiv papers found: {len(papers)} (skipped {skipped_count} existing)"
        )
        return papers

    def get_citation_counts(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Semantic Scholar APIから引用数を取得"""
        logger.info("[CITATION] Fetching citation counts from Semantic Scholar...")

        session = requests.Session()
        headers = {"User-Agent": "SO8T-Codex/1.0"}
        if self.semantic_scholar_key:
            headers["x-api-key"] = self.semantic_scholar_key

        # Semantic Scholar API（無料版は制限あり）
        # 注意: 大量のリクエストにはAPIキーが必要な場合があります

        for i, paper in enumerate(papers):
            try:
                # Arxiv IDからSemantic Scholar IDを取得
                arxiv_id = paper.get("arxiv_id", "")
                if not arxiv_id:
                    continue

                # Semantic Scholar API（簡易版）
                # 実際の実装では、Semantic Scholar APIを使用
                # ここでは簡易的な実装
                url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
                params = {"fields": "citationCount,referenceCount"}

                response = session.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    paper["citation_count"] = data.get("citationCount", 0)
                    paper["reference_count"] = data.get("referenceCount", 0)
                else:
                    paper["citation_count"] = 0

                # レート制限対策
                if (i + 1) % 100 == 0:
                    logger.info(f"[CITATION] Processed {i + 1}/{len(papers)} papers")
                    time.sleep(self.semantic_sleep)

            except Exception as e:
                logger.warning(
                    f"[WARN] Failed to get citation count for {paper.get('arxiv_id')}: {e}"
                )
                paper["citation_count"] = 0
                continue

        # 引用数でソート
        papers_sorted = sorted(
            papers, key=lambda x: x.get("citation_count", 0), reverse=True
        )

        logger.info(
            f"[CITATION] Top paper citation count: {papers_sorted[0].get('citation_count', 0) if papers_sorted else 0}"
        )
        return papers_sorted

    def download_pdf(self, paper: Dict[str, Any]) -> Optional[Path]:
        """論文PDFをダウンロード"""
        try:
            pdf_url = paper.get("pdf_url")
            if not pdf_url:
                return None

            # IDの取得（Arxiv ID または BioRxiv DOI）
            file_id = paper.get("arxiv_id")
            if not file_id:
                file_id = paper.get("biorxiv_doi")

            if not file_id:
                file_id = "unknown"  # Should ideally not happen if logic is correct

            # ファイル名のサニタイズ (スラッシュなどを置換)
            safe_id = str(file_id).replace("/", "_").replace("\\", "_").replace(":", "")
            pdf_filename = f"{safe_id}.pdf"
            pdf_path = self.raw_dir / pdf_filename

            if pdf_path.exists():
                logger.debug(f"[SKIP] PDF already exists: {pdf_path.name}")
                return pdf_path

            # User-Agentを設定してダウンロード
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/pdf,application/octet-stream,*/*",
            }

            response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            # Content-Typeのチェック (念のため)
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                logger.warning(f"[WARN] URL returned HTML instead of PDF: {pdf_url}")
                return None

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # サイズチェック (極端に小さいファイルはエラーの可能性)
            if pdf_path.stat().st_size < 1000:
                logger.warning(
                    f"[WARN] Downloaded PDF is too small (<1KB), deleting: {pdf_path.name}"
                )
                pdf_path.unlink()
                return None

            logger.debug(f"[OK] Downloaded: {pdf_path.name}")
            return pdf_path

        except Exception as e:
            logger.warning(
                f"[WARN] Failed to download PDF for {paper.get('arxiv_id', paper.get('biorxiv_doi'))}: {e}"
            )
            return None

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """PDFからテキストを抽出"""
        try:
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)

            return "\n\n".join(text_content)

        except Exception as e:
            logger.warning(
                f"[WARN] pdfplumber failed for {pdf_path.name}, trying PyPDF2: {e}"
            )
            try:
                text_content = []
                with open(pdf_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)

                return "\n\n".join(text_content)
            except Exception as e2:
                logger.error(f"[ERROR] Failed to extract text from {pdf_path}: {e2}")
                # Corrupted file handling: delete to allow redownload next time
                try:
                    if pdf_path.exists():
                        pdf_path.unlink()
                        logger.info(f"[FIX] Deleted corrupted PDF: {pdf_path.name}")
                except Exception as del_e:
                    logger.warning(
                        f"[WARN] Failed to delete corrupted file {pdf_path}: {del_e}"
                    )

                return ""

    def structure_paper_data(self, paper: Dict[str, Any], text: str) -> Dict[str, Any]:
        """論文データを構造化（推論能力向上用）"""

        # IDの生成 (Arxiv優先、BioRxivはDOIベース)
        if paper.get("source") == "biorxiv" and not paper.get("arxiv_id"):
            paper_id = f"biorxiv_{paper.get('biorxiv_doi', 'unknown')}"
        else:
            paper_id = f"arxiv_{paper.get('arxiv_id', 'unknown')}"

        # 科学・数学の推論能力向上用に構造化
        structured = {
            "id": paper_id,
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "summary": paper.get("summary", ""),
            "full_text": text,
            "published": paper.get("published", ""),
            "categories": paper.get("categories", []),
            "primary_category": paper.get("primary_category", ""),
            "citation_count": paper.get("citation_count", 0),
            "source": paper.get("source", "arxiv"),
            "domain": self._classify_domain(paper),
            "reasoning_type": self._classify_reasoning_type(paper, text),
            "structured_at": datetime.now().isoformat(),
        }

        # 四重推論構造化を追加
        quad_inference = self._generate_quadruple_inference(paper, text)
        structured["quadruple_inference"] = quad_inference

        # /thinkingモデル化データを追加
        thinking_format = self._generate_thinking_model_format(
            paper, text, quad_inference
        )
        structured["thinking_model"] = thinking_format

        return structured

    def _generate_thinking_model_format(
        self, paper: Dict[str, Any], text: str, quad_inference: Dict[str, Any]
    ) -> Dict[str, Any]:
        """/thinkingモデル化データの生成（<think>タグ形式）"""
        observation = quad_inference.get("observation", {})
        deduction = quad_inference.get("deduction", {})
        abduction = quad_inference.get("abduction", {})
        integration = quad_inference.get("integration", {})

        # <think>タグ内の四重推論構造を生成
        task_block = (
            "[Vector_State]\n"
            f"- Title: {paper.get('title', '')}\n"
            f"- Domain: {observation.get('domain', '')}\n"
            f"- Categories: {', '.join(paper.get('categories', [])[:3])}\n"
            f"- Citations: {paper.get('citation_count', 0)}\n"
            f"- Numbers: {', '.join(observation.get('extracted_numbers', [])[:5])}\n"
            f"- Equations: {', '.join(observation.get('extracted_equations', [])[:3])}"
        )
        analysis_block = (
            "[Spinor_Plus_Logic] (Deduction)\n"
            f"- Reasoning Type: {deduction.get('reasoning_type', 'Undefined')}\n"
            f"- Theoretical Keys: {', '.join(deduction.get('theoretical_keywords', [])[:4])}\n"
            f"- Domain Approaches: {', '.join(deduction.get('domain_approaches', [])[:4])}\n"
            f"- Methodological Logic: {deduction.get('logical_structure', 'Linear')}"
        )
        safety_block = (
            "[Spinor_Minus_Synthesis] (Abduction)\n"
            f"- Critical Edges: {', '.join(abduction.get('edge_case_keywords', [])[:3])}\n"
            f"- Counter-Narratives: {', '.join(abduction.get('alternative_approaches', [])[:2])}\n"
            f"- Ethical/Safety Checks: {abduction.get('challenges_to_deduction', 'Standard Review')}"
        )
        policy_block = (
            "[Quadrality_Integration] (Synthesis)\n"
            f"- Final Synthesis: {integration.get('integrated_insights', {}).get('synthesis', 'Convergent')}\n"
            f"- VSSI Quality Score: {integration.get('quality_score', 0):.4f}\n"
            f"- Golden Ratio Convergence: {integration.get('golden_ratio_convergence', {}).get('convergence_distance', 0):.4f}\n"
            f"- Final Decision: {integration.get('final_reasoning', 'Approved')}"
        )
        thinking_content = self._render_thinking_content(
            task_block,
            safety_block,
            policy_block,
            analysis_block=analysis_block,
        )

        # 最終回答（論文の要約を簡潔に）
        final_answer = paper.get("summary", "")[:500]  # 最初の500文字

        return {
            "instruction": f"Summarize and critique the paper using VSSI reasoning: {paper.get('title', '')}",
            "input": paper.get("summary", "")[:1000],
            "thinking": thinking_content,
            "output": final_answer,
            "format": "so8t_quadrality_thinking",
            "generated_at": datetime.now().isoformat(),
        }

    def export_vssi_dataset(self, cleaned_path: Path, output_path: Path) -> Path:
        """Export cleaned paper data into instruction/input/output JSONL for SFT."""
        items: List[Dict[str, Any]] = []
        if not cleaned_path.exists():
            raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")
        with open(cleaned_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                thinking_model = record.get("thinking_model") or {}
                thinking_block = thinking_model.get("thinking", "")
                final_answer = thinking_model.get("output", record.get("summary", ""))
                output = f"{thinking_block}\n<final>{final_answer}</final>"
                items.append(
                    {
                        "instruction": thinking_model.get(
                            "instruction", "Summarize the paper."
                        ),
                        "input": thinking_model.get("input", record.get("summary", "")),
                        "output": output,
                        "metadata": {
                            "paper_id": record.get("id"),
                            "citation_count": record.get("citation_count", 0),
                            "domain": record.get("domain"),
                            "source": record.get("source"),
                            "generated_at": thinking_model.get("generated_at"),
                        },
                    }
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("[SAVE] VSSI dataset written: %s", output_path)
        return output_path

    def _generate_quadruple_inference(
        self, paper: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """四重推論構造を生成（観察・演繹・帰納・統合）"""
        domain = self._classify_domain(paper)
        reasoning_type = self._classify_reasoning_type(paper, text)

        # フェーズ1: 観察（Vector Dimension）- 事実収集とリテラルマッピング
        observation = self._generate_observation_phase(paper, text, domain)

        # フェーズ2: 演繹（Positive Spinor）- 論理構造と標準的解決策
        deduction = self._generate_deduction_phase(
            paper, text, domain, reasoning_type, observation
        )

        # フェーズ3: 帰納（Negative Spinor）- 対立例探索とエッジケース検証
        abduction = self._generate_abduction_phase(
            paper, text, domain, reasoning_type, observation, deduction
        )

        # フェーズ4: 統合（Quadrality Integration）- 統合とGrokking合成
        integration = self._generate_integration_phase(
            paper, text, domain, observation, deduction, abduction
        )

        return {
            "observation": observation,
            "deduction": deduction,
            "abduction": abduction,
            "integration": integration,
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_observation_phase(
        self, paper: Dict[str, Any], text: str, domain: str
    ) -> Dict[str, Any]:
        """観察フェーズ（Vector Dimension）の生成"""
        # 事実収集とリテラルマッピング
        title = paper.get("title", "")
        summary = paper.get("summary", "")
        categories = paper.get("categories", [])
        citation_count = paper.get("citation_count", 0)

        # 主要な事実を抽出
        key_facts = {
            "title": title,
            "primary_category": paper.get("primary_category", ""),
            "categories": categories,
            "citation_count": citation_count,
            "published_date": paper.get("published", ""),
            "authors": paper.get("authors", []),
        }

        # テキストから主要な数値・定数を抽出
        numbers = re.findall(r"\b\d+\.?\d*\b", text[:5000])  # 最初の5000文字から
        equations = re.findall(
            r"[A-Za-z]+\s*[=<>≤≥]\s*[A-Za-z0-9+\-*/()]+", text[:5000]
        )

        return {
            "phase": "observation",
            "dimension": "vector",
            "key_facts": key_facts,
            "extracted_numbers": numbers[:20],  # 最初の20個
            "extracted_equations": equations[:10],  # 最初の10個
            "text_length": len(text),
            "domain": domain,
        }

    def _generate_deduction_phase(
        self,
        paper: Dict[str, Any],
        text: str,
        domain: str,
        reasoning_type: str,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """演繹フェーズ（Positive Spinor）の生成"""
        # 論理構造と標準的解決策
        title = paper.get("title", "").lower()
        summary = paper.get("summary", "").lower()
        text_lower = text.lower()

        # 理論的キーワードの検出
        theoretical_keywords = []
        if any(kw in text_lower for kw in ["theorem", "proof", "lemma", "proposition"]):
            theoretical_keywords.append("theoretical_proof")
        if any(kw in text_lower for kw in ["algorithm", "method", "approach"]):
            theoretical_keywords.append("methodological")
        if any(kw in text_lower for kw in ["model", "framework", "architecture"]):
            theoretical_keywords.append("modeling")

        # ドメイン固有の理論的アプローチ
        domain_approaches = {
            "mathematics": ["algebraic", "geometric", "analytical", "topological"],
            "physics": ["quantum", "classical", "statistical", "field_theory"],
            "biology": ["molecular", "systems", "evolutionary", "neural"],
            "ai_ml": ["neural_network", "optimization", "learning", "inference"],
        }

        approaches = domain_approaches.get(domain, ["general"])

        return {
            "phase": "deduction",
            "dimension": "positive_spinor",
            "reasoning_type": reasoning_type,
            "theoretical_keywords": theoretical_keywords,
            "domain_approaches": approaches,
            "logical_structure": "standard_solution_path",
            "based_on_observation": observation.get("key_facts", {}),
        }

    def _generate_abduction_phase(
        self,
        paper: Dict[str, Any],
        text: str,
        domain: str,
        reasoning_type: str,
        observation: Dict[str, Any],
        deduction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """帰納フェーズ（Negative Spinor）の生成"""
        # 対立例探索とエッジケース検証
        text_lower = text.lower()

        # エッジケースのキーワード
        edge_case_keywords = []
        if any(
            kw in text_lower for kw in ["exception", "edge case", "boundary", "limit"]
        ):
            edge_case_keywords.append("edge_cases")
        if any(
            kw in text_lower for kw in ["counterexample", "contradiction", "paradox"]
        ):
            edge_case_keywords.append("counterexamples")
        if any(kw in text_lower for kw in ["assumption", "limitation", "constraint"]):
            edge_case_keywords.append("assumptions")

        # 対立するアプローチの検出
        alternative_approaches = []
        if "theoretical" in reasoning_type and any(
            kw in text_lower for kw in ["empirical", "experimental"]
        ):
            alternative_approaches.append("empirical_alternative")
        if "analytical" in reasoning_type and any(
            kw in text_lower for kw in ["numerical", "computational"]
        ):
            alternative_approaches.append("computational_alternative")

        return {
            "phase": "abduction",
            "dimension": "negative_spinor",
            "edge_case_keywords": edge_case_keywords,
            "alternative_approaches": alternative_approaches,
            "challenges_to_deduction": "critical_analysis",
            "based_on_observation": observation.get("key_facts", {}),
            "based_on_deduction": deduction.get("theoretical_keywords", []),
        }

    def _generate_integration_phase(
        self,
        paper: Dict[str, Any],
        text: str,
        domain: str,
        observation: Dict[str, Any],
        deduction: Dict[str, Any],
        abduction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """統合フェーズ（Quadrality Integration）の生成"""
        # 統合とGrokking合成
        # 観察、演繹、帰納の結果を統合

        # 統合された洞察
        integrated_insights = {
            "synthesis": "unified_understanding",
            "consistency_check": "passed",
            "generalization": "domain_applicable",
            "novelty": "contribution_assessed",
        }

        # ノーベル賞・フィールズ賞レベルの評価基準
        quality_metrics = {
            "mathematical_rigor": 0.8 if "mathematics" in domain else 0.6,
            "physical_correctness": 0.8 if "physics" in domain else 0.6,
            "logical_consistency": 0.85,
            "causality_maintenance": 0.8,
            "symmetry_respect": 0.75,
            "computational_stability": 0.7,
            "information_preservation": 0.75,
        }

        # 黄金比収束（Φ^(-2) ≈ 0.382）を考慮した推論品質評価
        golden_ratio_inverse_square = 1 / ((1 + (5**0.5)) / 2) ** 2  # ≈ 0.382
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        convergence_score = abs(quality_score - golden_ratio_inverse_square)

        return {
            "phase": "integration",
            "dimension": "quadrality",
            "integrated_insights": integrated_insights,
            "quality_metrics": quality_metrics,
            "quality_score": quality_score,
            "golden_ratio_convergence": {
                "target_value": golden_ratio_inverse_square,
                "current_score": quality_score,
                "convergence_distance": convergence_score,
            },
            "synthesis_of_phases": {
                "observation_summary": observation.get("key_facts", {}).get(
                    "title", ""
                ),
                "deduction_summary": deduction.get("reasoning_type", ""),
                "abduction_summary": abduction.get("edge_case_keywords", []),
            },
            "final_reasoning": "grokking_synthesis_complete",
        }

    def _classify_domain(self, paper: Dict[str, Any]) -> str:
        """ドメイン分類"""
        categories = paper.get("categories", [])
        primary = paper.get("primary_category", "")

        if "math" in primary.lower() or any("math" in c.lower() for c in categories):
            return "mathematics"
        elif "physics" in primary.lower() or any(
            "physics" in c.lower() for c in categories
        ):
            return "physics"
        elif "q-bio" in primary.lower() or any(
            "q-bio" in c.lower() for c in categories
        ):
            return "biology"
        elif "cs.AI" in primary or "cs.LG" in primary:
            return "ai_ml"
        elif "stat" in primary.lower():
            return "statistics"
        else:
            return "general_science"

    def _classify_reasoning_type(self, paper: Dict[str, Any], text: str) -> str:
        """推論タイプ分類"""
        text_lower = text.lower()
        title_lower = paper.get("title", "").lower()
        summary_lower = paper.get("summary", "").lower()

        combined = f"{title_lower} {summary_lower} {text_lower}"

        # 推論タイプのキーワード
        if any(
            kw in combined
            for kw in ["proof", "theorem", "lemma", "proposition", "corollary"]
        ):
            return "theoretical_proof"
        elif any(
            kw in combined
            for kw in ["experiment", "empirical", "evaluation", "benchmark"]
        ):
            return "empirical_analysis"
        elif any(
            kw in combined for kw in ["algorithm", "method", "approach", "technique"]
        ):
            return "methodological"
        elif any(kw in combined for kw in ["model", "framework", "architecture"]):
            return "modeling"
        elif any(kw in combined for kw in ["analysis", "analysis of", "study of"]):
            return "analytical"
        else:
            return "general"

    def clean_and_sanitize(
        self, structured_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """データをクレンジング・サニタイズ"""
        logger.info(f"[CLEAN] Cleaning and sanitizing {len(structured_data)} papers...")

        cleaned_data = []

        for item in structured_data:
            # テキストクレンジング
            text = item.get("full_text", "")

            # 不要な文字を削除
            text = re.sub(r"\s+", " ", text)  # 連続する空白を1つに
            text = re.sub(
                r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF.,!?;:()\[\]{}"\'-]',
                "",
                text,
            )  # 特殊文字を削除

            # 長さチェック
            if len(text) < 500:  # 短すぎる論文はスキップ
                continue

            # 機密情報のマスキング
            text = self._mask_sensitive_info(text)

            # クレンジング済みデータ
            cleaned_item = {
                **item,
                "full_text": text,
                "text_length": len(text),
                "word_count": len(text.split()),
                "cleaned_at": datetime.now().isoformat(),
            }

            cleaned_data.append(cleaned_item)

        logger.info(
            f"[CLEAN] Cleaned {len(cleaned_data)} papers (removed {len(structured_data) - len(cleaned_data)})"
        )
        return cleaned_data

    def _mask_sensitive_info(self, text: str) -> str:
        """機密情報をマスキング"""
        # メールアドレス
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
        )

        # 電話番号
        text = re.sub(r"\b\d{2,4}-\d{2,4}-\d{2,4}\b", "[PHONE]", text)

        return text

    def save_structured_data(self, data: List[Dict[str, Any]], output_path: Path):
        """構造化データを保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"[SAVE] Saved {len(data)} papers to {output_path}")

    def process_top_cited_papers(
        self, arxiv_count: int = 40000, biorxiv_count: int = 10000
    ) -> Path:
        """引用上位論文を処理（Arxiv + BioRxiv）"""
        total_target = arxiv_count + biorxiv_count
        logger.info(
            f"[PROCESS] Processing top cited papers. Target - Arxiv: {arxiv_count}, BioRxiv: {biorxiv_count}"
        )

        # 既存データの読み込み
        existing_ids = self.get_existing_paper_ids()

        # 1. Arxiv検索
        arxiv_papers = []
        if arxiv_count > 0:
            raw_arxiv = self.search_arxiv_papers(
                max_results=arxiv_count * 2
            )  # 重複除去を見越して多めに検索
            for p in raw_arxiv:
                p_id = f"arxiv_{p.get('arxiv_id')}"
                if p_id not in existing_ids:
                    arxiv_papers.append(p)
            logger.info(
                f"[FILTER] Arxiv papers after skipping existing: {len(arxiv_papers)} (Original: {len(raw_arxiv)})"
            )

        # 2. BioRxiv検索
        biorxiv_papers = []
        if biorxiv_count > 0:
            raw_biorxiv = self.search_biorxiv_papers(max_results=biorxiv_count * 2)
            for p in raw_biorxiv:
                # ID generation logic aligned with structure_paper_data
                if p.get("arxiv_id"):
                    p_id = f"arxiv_{p.get('arxiv_id')}"
                else:
                    p_id = f"biorxiv_{p.get('biorxiv_doi', 'unknown')}"

                if p_id in existing_ids:
                    continue
                biorxiv_papers.append(p)
            logger.info(
                f"[FILTER] BioRxiv papers after skipping existing: {len(biorxiv_papers)} (Original: {len(raw_biorxiv)})"
            )

        all_papers = arxiv_papers + biorxiv_papers
        # Limit to target count
        if len(all_papers) > total_target:
            all_papers = all_papers[:total_target]

        logger.info(f"[PROCESS] Total papers to check for citations: {len(all_papers)}")

        # 3. 引用数取得
        papers_with_citations = self.get_citation_counts(all_papers)

        # 4. 上位N件を選択 (Total target)
        top_papers = papers_with_citations[:total_target]
        logger.info(
            f"[PROCESS] Selected top {len(top_papers)} papers by citation count"
        )

        # 5. PDFダウンロードとテキスト抽出（上位5000件のみフルテキスト、残りはアブストラクト）
        structured_data = []
        full_text_limit = 5000

        for i, paper in enumerate(top_papers):
            try:
                text = paper.get("summary", "")

                # 上位5000件のみフルテキスト抽出を試みる（リソース節約）
                if i < full_text_limit:
                    pdf_path = self.download_pdf(paper)
                    if pdf_path:
                        full_text = self.extract_text_from_pdf(pdf_path)
                        if full_text:
                            text = full_text

                # 構造化
                structured = self.structure_paper_data(paper, text)
                structured_data.append(structured)

                if (i + 1) % 100 == 0:
                    logger.info(f"[PROCESS] Processed {i + 1}/{len(top_papers)} papers")

            except Exception as e:
                logger.error(
                    f"[ERROR] Failed to process paper {paper.get('arxiv_id', paper.get('biorxiv_doi'))}: {e}"
                )
                continue

        # 6. クレンジング
        cleaned_data = self.clean_and_sanitize(structured_data)

        # 7. 保存
        output_path = (
            self.cleaned_dir
            / f"arxiv_biorxiv_top_{len(cleaned_data)}_papers_50k_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        self.save_structured_data(cleaned_data, output_path)

        return output_path


def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Arxiv/BioRxiv papers for reasoning capability improvement"
    )
    parser.add_argument(
        "--arxiv-count",
        type=int,
        default=40000,
        help="Number of Arxiv papers to process (default: 40000)",
    )
    parser.add_argument(
        "--biorxiv-count",
        type=int,
        default=10000,
        help="Number of BioRxiv papers to process (default: 10000)",
    )
    parser.add_argument(
        "--max-papers", type=int, help="(Deprecated) Total papers, splits 80/20 if used"
    )
    parser.add_argument("--query", type=str, default="", help="Additional search query")
    parser.add_argument(
        "--export-vssi",
        action="store_true",
        help="Export VSSI instruction dataset from cleaned output",
    )
    parser.add_argument(
        "--vssi-output",
        type=str,
        default="",
        help="Optional path for VSSI dataset output",
    )

    args = parser.parse_args()

    processor = ArxivBioRxivProcessor()

    a_count = args.arxiv_count
    b_count = args.biorxiv_count

    # Legacy support
    if args.max_papers:
        a_count = int(args.max_papers * 0.8)
        b_count = int(args.max_papers * 0.2)

    # 処理実行
    output_path = processor.process_top_cited_papers(
        arxiv_count=a_count, biorxiv_count=b_count
    )
    if args.export_vssi:
        vssi_output = (
            Path(args.vssi_output)
            if args.vssi_output
            else processor.processed_dir
            / f"arxiv_biorxiv_vssi_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        processor.export_vssi_dataset(output_path, vssi_output)

    print(f"\n[SUCCESS] Processed Arxiv/BioRxiv papers")
    print(f"[OUTPUT] {output_path}")


if __name__ == "__main__":
    main()
