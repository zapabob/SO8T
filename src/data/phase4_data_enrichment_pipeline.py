#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: 高度データ拡充 統合パイプライン

薬物・NSFW検知、arXiv論文、地政学OSINT、科学・数学CoTデータを統合し、
SO8T四重推論モデルの事後学習に適したデータセットを構築します。

Features:
- HF CLI経由での既存データセット取得
- 薬物/NSFW検知データセットの統合
- 地政学OSINTデータの自動収集
- 科学・数学・CoTデータの統合
- MCP/Skill (Tool-calling) データの生成
- Phase 5 SFT/GRPOパイプラインへのシームレスな接続
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "phase4_data_enrichment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase4DataEnrichmentPipeline:
    """
    Phase 4 高度データ拡充パイプライン

    統合対象:
    1. 薬物検知データセット (研究・検知目的)
    2. NSFW検知データセット (モデレーション目的)
    3. 地政学OSINTデータ (2024-2026年国際情勢)
    4. 科学・数学・CoTデータ (HF CLIより取得)
    5. MCP/Skill (Tool-calling) データ
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.project_root = PROJECT_ROOT
        self.output_dir = output_dir or self.project_root / "data" / "phase4_enriched"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = config_path or self.project_root / "src" / "infrastructure" / "config" / "osint_sources.yaml"
        self.collected_samples: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            "drug_detection": 0,
            "nsfw_detection": 0,
            "geopolitics_osint": 0,
            "culture_anime": 0, # 追加
            "science_math_cot": 0,
            "mcp_skill": 0,
            "sakana_science": 0,
            "sakana_osint": 0,
            "total": 0,
        }
        logger.info("Phase 4 Data Enrichment Pipeline initialized.")
        logger.info(f"Output directory: {self.output_dir}")

    def collect_drug_detection_data(self, max_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        薬物検知データの収集 (研究・検知目的)
        既存の `collect_drug_pharmaceutical_detection_dataset.py` を呼び出す。
        """
        logger.info(f"Collecting drug detection data (max: {max_samples})...")
        samples: List[Dict[str, Any]] = []
        try:
            from src.data.collect_drug_pharmaceutical_detection_dataset import DrugPharmaceuticalDetectionDatasetCollector
            collector = DrugPharmaceuticalDetectionDatasetCollector(output_dir=self.output_dir / "drug_detection")
            # Collect from all sources
            collected = collector.collect_all(max_samples_per_source=max_samples // 5)
            samples.extend(collected)
            logger.info(f"Collected {len(collected)} drug detection samples.")
        except ImportError as e:
            logger.warning(f"Drug detection module not available: {e}")
        except Exception as e:
            logger.error(f"Error collecting drug detection data: {e}")
        
        self.stats["drug_detection"] = len(samples)
        return samples

    def collect_nsfw_detection_data(self, max_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        NSFW検知データの収集 (モデレーション目的)
        既存データセットの読み込みと統合。
        """
        logger.info(f"Collecting NSFW detection data (max: {max_samples})...")
        samples: List[Dict[str, Any]] = []
        
        # Existing NSFW datasets
        nsfw_datasets = [
            self.project_root / "src" / "data" / "datasets" / "final_integrated_nsfw_dataset.jsonl",
            self.project_root / "src" / "data" / "datasets" / "drug_nsfw_fiction_dataset.jsonl",
        ]
        
        for ds_path in nsfw_datasets:
            if ds_path.exists():
                try:
                    with open(ds_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if len(samples) >= max_samples:
                                break
                            try:
                                sample = json.loads(line.strip())
                                sample["source"] = "nsfw_detection"
                                sample["domain"] = "safety"
                                samples.append(sample)
                            except json.JSONDecodeError:
                                continue
                    logger.info(f"Loaded {len(samples)} samples from {ds_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {ds_path}: {e}")
        
        self.stats["nsfw_detection"] = len(samples)
        return samples

    def collect_geopolitics_osint_data(self, max_samples: int = 10000) -> List[Dict[str, Any]]:
        """
        地政学OSINTデータの収集 (2024-2026年国際情勢)
        既存の `osint_source_collector.py` を呼び出す。
        """
        logger.info(f"Collecting geopolitics OSINT data (max: {max_samples})...")
        samples: List[Dict[str, Any]] = []
        try:
            from src.data.processing.osint_source_collector import OSINTSourceCollector
            
            if self.config_path.exists():
                collector = OSINTSourceCollector(config_path=self.config_path)
                collected = collector.collect_all(max_items=max_samples)
                for item in collected:
                    item["source"] = "geopolitics_osint"
                    item["domain"] = "intelligence"
                    samples.append(item)
                logger.info(f"Collected {len(collected)} OSINT samples.")
            else:
                logger.warning(f"OSINT config not found: {self.config_path}")
        except ImportError as e:
            logger.warning(f"OSINT module not available: {e}")
        except Exception as e:
            logger.error(f"Error collecting OSINT data: {e}")
        
        self.stats["geopolitics_osint"] = len(samples)
        return samples

    def collect_culture_anime_data(self, max_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        文化・ガンダムデータの収集。
        """
        logger.info(f"Collecting culture/Gundam data (max: {max_samples})...")
        samples: List[Dict[str, Any]] = []
        try:
            from src.data.culture_gundam_collector import GundamCultureDataCollector
            collector = GundamCultureDataCollector(output_dir=self.output_dir / "culture_anime")
            seed_samples = collector.generate_gundam_reasoning_samples()
            for s in seed_samples:
                samples.append({
                    "conversations": [
                        {"from": "human", "value": s["prompt"]},
                        {"from": "gpt", "value": s["response"]},
                    ],
                    "source": "culture_gundam_seed",
                    "domain": "culture"
                })
            logger.info(f"Collected {len(samples)} culture samples.")
        except Exception as e:
            logger.error(f"Error collecting culture data: {e}")
        
        self.stats["culture_anime"] = len(samples)
        return samples

    def collect_science_math_cot_data(self, max_samples: int = 20000) -> List[Dict[str, Any]]:
        """
        科学・数学・CoTデータの収集 (HF CLI経由)
        arXiv論文、数学推論データセットから取得。
        """
        logger.info(f"Collecting science/math/CoT data (max: {max_samples})...")
        samples: List[Dict[str, Any]] = []
        
        # Try loading from HF datasets using huggingface_hub
        try:
            from huggingface_hub import hf_hub_download, list_datasets
            
            # Pre-defined high-quality datasets for CoT
            hf_datasets = [
                ("lighteval/MATH", "test"),
                ("gsm8k", "main"),
                ("allenai/ai2_arc", "ARC-Challenge"),
            ]
            
            for ds_name, subset in hf_datasets:
                try:
                    # Attempt to download and load
                    logger.info(f"Attempting to load {ds_name}/{subset}...")
                    # This is a placeholder - actual HF dataset loading would require datasets library
                except Exception as e:
                    logger.warning(f"Could not load {ds_name}: {e}")
        except ImportError:
            logger.warning("huggingface_hub not available. Skipping HF dataset loading.")
        
        # Load existing local science/math datasets
        local_datasets = list(self.project_root.glob("src/data/datasets/*science*.jsonl"))
        local_datasets.extend(self.project_root.glob("src/data/datasets/*math*.jsonl"))
        local_datasets.extend(self.project_root.glob("src/data/datasets/*cot*.jsonl"))
        
        for ds_path in local_datasets:
            try:
                with open(ds_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(samples) >= max_samples:
                            break
                        try:
                            sample = json.loads(line.strip())
                            sample["source"] = "science_math_cot"
                            sample["domain"] = "academic"
                            samples.append(sample)
                        except json.JSONDecodeError:
                            continue
                logger.info(f"Loaded samples from {ds_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {ds_path}: {e}")
        
        self.stats["science_math_cot"] = len(samples)
        return samples

    def collect_mcp_skill_data(self, max_samples: int = 5000) -> List[Dict[str, Any]]:
        """
        MCP/Skill (Tool-calling) データの収集
        Function calling, tool use のパターンを含むデータセット。
        """
        logger.info(f"Collecting MCP/Skill data (max: {max_samples})...")
        samples: List[Dict[str, Any]] = []
        
        # Load existing tool-calling datasets
        skill_datasets = list(self.project_root.glob("src/data/datasets/*tool*.jsonl"))
        skill_datasets.extend(self.project_root.glob("src/data/datasets/*function*.jsonl"))
        skill_datasets.extend(self.project_root.glob("src/data/datasets/*mcp*.jsonl"))
        skill_datasets.extend(self.project_root.glob("src/infrastructure/skills/**/*.jsonl"))
        
        for ds_path in skill_datasets:
            try:
                with open(ds_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(samples) >= max_samples:
                            break
                        try:
                            sample = json.loads(line.strip())
                            sample["source"] = "mcp_skill"
                            sample["domain"] = "tool_use"
                            samples.append(sample)
                        except json.JSONDecodeError:
                            continue
                logger.info(f"Loaded samples from {ds_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {ds_path}: {e}")
        
        self.stats["mcp_skill"] = len(samples)
        return samples

    def collect_sakana_science_data(self, max_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Sakana AI Science Agentによる研究データ生成
        """
        logger.info(f"Collecting Sakana AI Science data (max: {max_samples})...")
        samples: List[Dict[str, Any]] = []
        try:
            from src.agents.sakana_ai_integrated_agent import AIScientistAgent
            agent = AIScientistAgent(project_root=self.project_root)
            
            topics = [
                "Quantum Machine Learning", "Climate Modeling AI", "Bio-mimetic Robotics",
                "Fusion Energy Control", "Neuromorphic Computing", "Space Debris Removal",
                "Personalized Medicine AI", "Ethical AI Frameworks", "Dark Matter Detection",
                "Ocean Acidification Reversal"
            ]
            
            for topic in topics:
                if len(samples) >= max_samples:
                    break
                
                try:
                    # 1. Generate Idea
                    ideas = agent.generate_ideas(topic, num_ideas=1)
                    if not ideas: continue
                    idea = ideas[0]
                    
                    # 2. Conduct Experiment (Simulation)
                    result = agent.conduct_experiment(idea)
                    
                    # 3. Write Paper (This generates the training content with <think> tags)
                    paper_content = agent.write_paper(idea, result)
                    
                    # Create SFT Sample
                    samples.append({
                        "conversations": [
                            {"from": "human", "value": f"Conduct a comprehensive research cycle on the topic: {topic}. Generate a research paper draft including abstract, methodology, and results."},
                            {"from": "gpt", "value": paper_content}
                        ],
                        "source": "sakana_science",
                        "domain": "academic"
                    })
                    logger.info(f"Generated science sample for: {topic}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate science data for {topic}: {e}")
                    
        except ImportError as e:
            logger.warning(f"Sakana Agent not available: {e}")
        except Exception as e:
            logger.error(f"Error initializing Sakana Science Agent: {e}")
            
        self.stats["sakana_science"] = len(samples)
        return samples

    def collect_sakana_osint_data(self, max_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Sakana AI OSINT Agentによるインテリジェンスデータ生成
        """
        logger.info(f"Collecting Sakana AI OSINT data (max: {max_samples})...")
        samples: List[Dict[str, Any]] = []
        try:
            from src.agents.sakana_ai_integrated_agent import OSINTAIAgent
            agent = OSINTAIAgent(project_root=self.project_root)
            
            topics = [
                "Global Cybersecurity Threats 2026", "Arctic Trade Route Stability",
                "Rare Earth Supply Chain Resilience", "Deepfake Disinformation Campaigns",
                "Hypersonic Missile Development", "Space Weaponization Trends",
                "AI Regulation International Standards", "Cryptocurrency Money Laundering",
                "Autonomous Drone Swarm Tactics", "Quantum Encryption Vulnerabilities"
            ]
            
            for topic in topics:
                if len(samples) >= max_samples:
                    break
                    
                try:
                    # 1. Collect Intelligence (Simulation/API)
                    intelligence = agent.collect_intelligence(topic)
                    
                    # 2. Generate Analysis (This generates the training content with <think> tags)
                    analysis_report = agent.generate_analysis(topic, intelligence, use_quadrality=True)
                    
                    # Create SFT Sample
                    samples.append({
                        "conversations": [
                            {"from": "human", "value": f"Analyze the current situation regarding: {topic}. Provide a comprehensive OSINT report with security and policy considerations."},
                            {"from": "gpt", "value": analysis_report}
                        ],
                        "source": "sakana_osint",
                        "domain": "intelligence"
                    })
                    logger.info(f"Generated OSINT sample for: {topic}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate OSINT data for {topic}: {e}")

        except ImportError as e:
            logger.warning(f"Sakana Agent not available: {e}")
        except Exception as e:
            logger.error(f"Error initializing Sakana OSINT Agent: {e}")

        self.stats["sakana_osint"] = len(samples)
        return samples

    def format_for_sft(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        SFT学習用フォーマットへ変換。
        ShareGPT形式（conversations）に統一。
        """
        logger.info(f"Formatting {len(samples)} samples for SFT...")
        formatted: List[Dict[str, Any]] = []
        
        for sample in samples:
            try:
                # Already in conversation format
                if "conversations" in sample:
                    formatted.append(sample)
                    continue
                
                # Convert from prompt/response format
                if "prompt" in sample and "response" in sample:
                    formatted.append({
                        "conversations": [
                            {"from": "human", "value": sample["prompt"]},
                            {"from": "gpt", "value": sample["response"]},
                        ],
                        "source": sample.get("source", "unknown"),
                        "domain": sample.get("domain", "unknown"),
                    })
                    continue
                
                # Convert from input/output format
                if "input" in sample and "output" in sample:
                    formatted.append({
                        "conversations": [
                            {"from": "human", "value": sample["input"]},
                            {"from": "gpt", "value": sample["output"]},
                        ],
                        "source": sample.get("source", "unknown"),
                        "domain": sample.get("domain", "unknown"),
                    })
                    continue
                
                # Convert from question/answer format
                if "question" in sample and "answer" in sample:
                    formatted.append({
                        "conversations": [
                            {"from": "human", "value": sample["question"]},
                            {"from": "gpt", "value": sample["answer"]},
                        ],
                        "source": sample.get("source", "unknown"),
                        "domain": sample.get("domain", "unknown"),
                    })
                    continue
                
                # Text only - create a simple format
                if "text" in sample:
                    formatted.append({
                        "text": sample["text"],
                        "source": sample.get("source", "unknown"),
                        "domain": sample.get("domain", "unknown"),
                    })
            except Exception as e:
                logger.warning(f"Failed to format sample: {e}")
        
        logger.info(f"Formatted {len(formatted)} samples for SFT.")
        return formatted

    def save_dataset(self, samples: List[Dict[str, Any]], filename: str = "phase4_enriched_dataset.jsonl") -> Path:
        """
        データセットをJSONL形式で保存。
        """
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(samples)} samples to {output_path}")
        return output_path

    def run(
        self,
        drug_max: int = 5000,
        nsfw_max: int = 5000,
        osint_max: int = 10000,
        science_max: int = 20000,
        mcp_max: int = 5000,
        sakana_max: int = 200,
    ) -> Path:
        """
        Phase 4 高度データ拡充パイプラインを実行。
        """
        logger.info("=" * 60)
        logger.info("Starting Phase 4: Advanced Data Enrichment Pipeline")
        logger.info("=" * 60)
        
        all_samples: List[Dict[str, Any]] = []
        
        # 1. Drug detection data
        drug_samples = self.collect_drug_detection_data(max_samples=drug_max)
        all_samples.extend(drug_samples)
        
        # 2. NSFW detection data
        nsfw_samples = self.collect_nsfw_detection_data(max_samples=nsfw_max)
        all_samples.extend(nsfw_samples)
        
        # 3. Geopolitics OSINT data
        osint_samples = self.collect_geopolitics_osint_data(max_samples=osint_max)
        all_samples.extend(osint_samples)
        
        # 3.5 Culture/Anime data
        culture_samples = self.collect_culture_anime_data(max_samples=5000)
        all_samples.extend(culture_samples)
        
        # 4. Science/Math/CoT data
        science_samples = self.collect_science_math_cot_data(max_samples=science_max)
        all_samples.extend(science_samples)
        
        # 5. MCP/Skill data
        mcp_samples = self.collect_mcp_skill_data(max_samples=mcp_max)
        all_samples.extend(mcp_samples)

        # 6. Sakana AI Science data
        sakana_science_samples = self.collect_sakana_science_data(max_samples=sakana_max // 2)
        all_samples.extend(sakana_science_samples)

        # 7. Sakana AI OSINT data
        sakana_osint_samples = self.collect_sakana_osint_data(max_samples=sakana_max // 2)
        all_samples.extend(sakana_osint_samples)
        
        # Update total stats
        self.stats["total"] = len(all_samples)
        
        # Format for SFT
        formatted_samples = self.format_for_sft(all_samples)
        
        # Save dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.save_dataset(formatted_samples, f"phase4_enriched_{timestamp}.jsonl")
        
        # Save stats
        stats_path = self.output_dir / f"phase4_stats_{timestamp}.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 60)
        logger.info("Phase 4 Data Enrichment Complete!")
        logger.info(f"Total samples: {self.stats['total']}")
        logger.info(f"  - Drug detection: {self.stats['drug_detection']}")
        logger.info(f"  - NSFW detection: {self.stats['nsfw_detection']}")
        logger.info(f"  - Geopolitics OSINT: {self.stats['geopolitics_osint']}")
        logger.info(f"  - Culture/Anime: {self.stats['culture_anime']}")
        logger.info(f"  - Science/Math/CoT: {self.stats['science_math_cot']}")
        logger.info(f"  - MCP/Skill: {self.stats['mcp_skill']}")
        logger.info(f"  - Sakana Science: {self.stats['sakana_science']}")
        logger.info(f"  - Sakana OSINT: {self.stats['sakana_osint']}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)
        
        return output_path


def main() -> None:
    """Main entry point."""
    pipeline = Phase4DataEnrichmentPipeline()
    output_path = pipeline.run()
    print(f"\nPhase 4 complete. Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
