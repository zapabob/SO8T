#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: Multi-domain data enrichment
- Academic scaling (arXiv/BioRxiv + Semantic Scholar)
- Pop-culture / anime / film critique data
- World affairs primary-source mapping
- Pharma safety enrichment
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from src.data.processing.process_arxiv_biorxiv import ArxivBioRxivProcessor
except ImportError:
    ArxivBioRxivProcessor = None  # type: ignore[misc, assignment]

try:
    from src.data.osint_source_collector import collect_sources, load_sources_config
except ImportError:
    collect_sources = None  # type: ignore[assignment]
    load_sources_config = None  # type: ignore[assignment]

# enrich_pharma_safety module not yet implemented
enrich_pharma_dataset = None  # type: ignore[assignment]

try:
    from src.utils.vssi_template import render_thinking
except ImportError:
    def render_thinking(*args, **kwargs):  # type: ignore[misc]
        return ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(paths: Iterable[Path]) -> List[Dict[str, object]]:
    data: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def write_jsonl(path: Path, items: List[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def build_pop_culture(items: List[Dict[str, object]], use_quadruple: bool, style: Optional[str]) -> List[Dict[str, object]]:
    results = []
    for entry in items:
        title = entry.get("title") or entry.get("work") or "Untitled"
        year = entry.get("year") or "unknown"
        critique = entry.get("critique") or entry.get("review") or ""
        narrative = entry.get("narrative") or entry.get("story") or ""
        artistic = entry.get("artistic_merit") or entry.get("direction") or ""

        task_block = (
            "[Vector_State]\n"
            f"- Title: {title}\n"
            f"- Year: {year}\n"
            f"- Medium: {entry.get('type', 'anime/film')}\n"
            f"- Key Themes: {entry.get('themes', '')}"
        )
        analysis_block = (
            "[Spinor_Plus_Logic]\n"
            f"- Narrative logic: {narrative or 'not specified'}\n"
            f"- Critical lens: {critique or 'not specified'}\n"
            f"- Style markers: {entry.get('style', '')}"
        )
        safety_block = (
            "[Spinor_Minus_Synthesis]\n"
            f"- Narrative gaps: {entry.get('narrative_gaps', 'none noted')}\n"
            f"- Continuity risks: {entry.get('continuity', 'unknown')}"
        )
        policy_block = (
            "[Quadrality_Integration]\n"
            f"- Artistic merit: {artistic}\n"
            f"- Narrative logic: {narrative}\n"
            f"- Critical stance: {critique}"
        )
        thinking = render_thinking(
            task_block,
            safety_block,
            policy_block,
            analysis_block=analysis_block,
            use_quadruple=use_quadruple,
            style=style,
        )
        final = (
            f"EN: {title} ({year}) review — {critique}\n"
            f"JP: {title}（{year}年）レビュー — {critique}"
        )
        results.append(
            {
                "instruction": f"Provide a critical review of {title} focusing on artistic merit and narrative logic.",
                "input": f"Summary: {entry.get('summary', '')}\nNotes: {entry.get('notes', '')}",
                "output": f"{thinking}\n<final>{final}</final>",
                "metadata": {
                    "domain": "pop_culture",
                    "title": title,
                    "year": year,
                    "source": entry.get("source"),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                },
            }
        )
    return results


def build_world_affairs(items: List[Dict[str, object]], use_quadruple: bool, style: Optional[str]) -> List[Dict[str, object]]:
    results = []
    for entry in items:
        event = entry.get("event") or entry.get("title") or "Event"
        date = entry.get("date") or entry.get("year") or "unknown"
        region = entry.get("region") or entry.get("country") or "global"
        sources = entry.get("sources") or entry.get("primary_sources") or []
        source_text = ", ".join(sources) if isinstance(sources, list) else str(sources)

        task_block = (
            "[Vector_State]\n"
            f"- Event: {event}\n"
            f"- Date: {date}\n"
            f"- Region: {region}\n"
            f"- Primary sources: {source_text}"
        )
        analysis_block = (
            "[Spinor_Plus_Logic]\n"
            f"- Stakeholders: {entry.get('stakeholders', 'unknown')}\n"
            f"- Trigger factors: {entry.get('triggers', 'unknown')}\n"
            f"- Confidence: {entry.get('confidence', 'medium')}"
        )
        safety_block = (
            "[Spinor_Minus_Synthesis]\n"
            f"- Uncertainties: {entry.get('uncertainty', 'unknown')}\n"
            f"- Cross-check: {entry.get('cross_check', 'pending')}"
        )
        policy_block = (
            "[Quadrality_Integration]\n"
            f"- Impact summary: {entry.get('impact', '')}\n"
            f"- Technology angle: {entry.get('tech_milestone', '')}"
        )
        thinking = render_thinking(
            task_block,
            safety_block,
            policy_block,
            analysis_block=analysis_block,
            use_quadruple=use_quadruple,
            style=style,
        )
        final = (
            f"EN: {event} ({date}) — {entry.get('summary', '')}\n"
            f"JP: {event}（{date}年）— {entry.get('summary', '')}"
        )
        results.append(
            {
                "instruction": f"Map the event '{event}' to primary sources and summarize implications.",
                "input": f"Summary: {entry.get('summary', '')}\nSources: {source_text}",
                "output": f"{thinking}\n<final>{final}</final>",
                "metadata": {
                    "domain": "world_affairs",
                    "event": event,
                    "date": date,
                    "region": region,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                },
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-domain enrichment pipeline")
    parser.add_argument("--academic", action="store_true", help="Enable academic scaling (arXiv/BioRxiv)")
    parser.add_argument("--pop-culture", action="store_true", help="Enable pop-culture enrichment")
    parser.add_argument("--world-affairs", action="store_true", help="Enable world affairs enrichment")
    parser.add_argument("--pharma", action="store_true", help="Enable pharma safety enrichment")
    parser.add_argument("--auto-sources", action="store_true", help="Auto collect OSINT sources for pop/world")
    parser.add_argument("--max-papers", type=int, default=int(os.getenv("SO8T_MAX_PAPERS", "100000")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quadruple", action="store_true")
    parser.add_argument("--think-tag-style", default=os.getenv("SO8T_THINK_TAG_STYLE", "legacy"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "data" / "multi_domain_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)

    enable_academic = args.academic or os.getenv("SO8T_ENRICH_ACADEMIC", "1") == "1"
    enable_pop = args.pop_culture or os.getenv("SO8T_ENRICH_POP", "1") == "1"
    enable_world = args.world_affairs or os.getenv("SO8T_ENRICH_WORLD", "1") == "1"
    enable_pharma = args.pharma or os.getenv("SO8T_ENRICH_PHARMA", "1") == "1"

    use_quadruple = args.quadruple or os.getenv("SO8T_QUADRUPLE_TOKENS", "0") == "1"
    style = args.think_tag_style
    auto_sources = args.auto_sources or os.getenv("SO8T_OSINT_AUTO_SOURCES", "0") == "1"

    manifest: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": {},
    }

    if enable_academic:
        processor = ArxivBioRxivProcessor(project_root)
        cleaned_files = list((project_root / "data" / "arxiv_biorxiv" / "cleaned").glob("*.jsonl"))
        cleaned_path = max(cleaned_files, key=lambda p: p.stat().st_mtime) if cleaned_files else None
        if not cleaned_path and not args.dry_run:
            cleaned_path = processor.process_top_cited_papers(max_papers=args.max_papers)
        if cleaned_path:
            vssi_path = out_dir / f"academic_vssi_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            processor.export_vssi_dataset(cleaned_path, vssi_path)
            manifest["outputs"]["academic"] = str(vssi_path)
        else:
            logger.warning("No academic data found. Skipping academic enrichment.")

    if enable_pop:
        pop_sources_paths = [
            project_root / "data" / "pop_culture" / "sources.jsonl",
            project_root / "data" / "pop_culture" / "raw" / "pop_culture.jsonl",
        ]
        if auto_sources and not args.dry_run:
            config = load_sources_config(project_root / "config" / "osint_sources.yaml")
            auto_output = collect_sources(
                domain="pop_culture",
                config=config,
                output_dir=project_root / "data" / "pop_culture",
                max_items=int(os.getenv("SO8T_OSINT_MAX_ITEMS", "200")),
            )
            if auto_output:
                pop_sources_paths.insert(0, auto_output)
                manifest["outputs"]["pop_culture_sources"] = str(auto_output)
        pop_sources = load_jsonl(pop_sources_paths)
        pop_items = build_pop_culture(pop_sources, use_quadruple, style)
        pop_path = out_dir / "pop_culture_vssi.jsonl"
        write_jsonl(pop_path, pop_items)
        manifest["outputs"]["pop_culture"] = str(pop_path)

    if enable_world:
        world_sources_paths = [
            project_root / "data" / "world_affairs" / "timeline.jsonl",
            project_root / "data" / "world_affairs" / "raw" / "events.jsonl",
        ]
        if auto_sources and not args.dry_run:
            config = load_sources_config(project_root / "config" / "osint_sources.yaml")
            auto_output = collect_sources(
                domain="world_affairs",
                config=config,
                output_dir=project_root / "data" / "world_affairs",
                max_items=int(os.getenv("SO8T_OSINT_MAX_ITEMS", "200")),
            )
            if auto_output:
                world_sources_paths.insert(0, auto_output)
                manifest["outputs"]["world_affairs_sources"] = str(auto_output)
        world_sources = load_jsonl(world_sources_paths)
        world_items = build_world_affairs(world_sources, use_quadruple, style)
        world_path = out_dir / "world_affairs_vssi.jsonl"
        write_jsonl(world_path, world_items)
        manifest["outputs"]["world_affairs"] = str(world_path)

    if enable_pharma:
        pharma_output = out_dir / "pharma_safety_vssi.jsonl"
        enrich_pharma_dataset(
            [
                project_root / "data" / "pharma" / "raw" / "pharma_sources.jsonl",
                project_root / "data" / "drug_pharma" / "raw" / "drug_sources.jsonl",
            ],
            pharma_output,
            use_quadruple,
            style,
        )
        manifest["outputs"]["pharma"] = str(pharma_output)

    manifest_path = project_root / "results" / "multi_domain_enrichment_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[MANIFEST] %s", manifest_path)


if __name__ == "__main__":
    main()
