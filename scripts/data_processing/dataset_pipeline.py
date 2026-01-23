#!/usr/bin/env python3
"""
RTX 3060 Optimized Dataset Pipeline
サンセットパイプライン データセット処理スクリプト
"""

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTX3060DatasetPipeline:
    def __init__(self, config_path=None):
        self.project_root = Path(__file__).parent.parent.parent

        # 設定ファイル読み込み
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.project_root / "config" / "dataset.json"

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # ディレクトリ設定
        self.raw_dir = self.project_root / "data" / "sunset_pipeline" / "raw"
        self.processed_dir = self.project_root / "data" / "sunset_pipeline" / "processed"

        # メモリ最適化設定
        self.chunk_size = self.config.get('processing', {}).get('chunk_size', 10000)
        self.max_samples = self.config.get('processing', {}).get('max_samples', 100000)

        logger.info(f"[INIT] RTX 3060 Dataset Pipeline initialized")
        logger.info(f"[CONFIG] Max samples: {self.max_samples}, Chunk size: {self.chunk_size}")

    def download_curated_datasets(self):
        """キュレートされたデータセットをダウンロード"""
        logger.info("[DOWNLOAD] Starting dataset download...")

        datasets_config = self.config.get('sources', [])
        downloaded_data = []

        for source in datasets_config:
            try:
                logger.info(f"[DOWNLOAD] Processing {source}")

                if 'huggingface:' in source:
                    dataset_name = source.replace('huggingface:', '')
                    dataset = self._download_huggingface_dataset(dataset_name)
                    if dataset:
                        downloaded_data.append(dataset)

                elif 'moonshot:' in source:
                    dataset_type = source.replace('moonshot:', '')
                    dataset = self._download_moonshot_dataset(dataset_type)
                    if dataset:
                        downloaded_data.append(dataset)

                elif 'synthetic:' in source:
                    dataset_type = source.replace('synthetic:', '')
                    dataset = self._generate_synthetic_data(dataset_type)
                    if dataset:
                        downloaded_data.append(dataset)

            except Exception as e:
                logger.error(f"[ERROR] Failed to download {source}: {e}")
                continue

        logger.info(f"[DOWNLOAD] Downloaded {len(downloaded_data)} datasets")
        return downloaded_data

    def _download_huggingface_dataset(self, dataset_name):
        """HuggingFaceデータセットをダウンロード"""
        try:
            logger.info(f"[HF] Loading {dataset_name}")

            # メモリ効率的な読み込み
            if dataset_name == 'math_dataset':
                # MATH dataset (利用可能なものを使用)
                try:
                    dataset = load_dataset('math_dataset', split='train[:5%]')  # 5%サンプル
                except:
                    # フォールバック: GSM8Kを使用
                    dataset = load_dataset('gsm8k', 'main', split='train[:5%]')
            elif dataset_name == 'science_qa':
                # ScienceQA dataset
                try:
                    dataset = load_dataset('derek-thomas/ScienceQA', split='train[:5%]')
                except:
                    # フォールバック: 基本的なQAデータセット
                    dataset = load_dataset('squad', split='train[:5%]')
            elif dataset_name == 'theorem_qa':
                # TheoremQA dataset
                try:
                    dataset = load_dataset('wenhu/TheoremQA', split='train[:5%]')
                except:
                    # フォールバック: 数学関連データセット
                    dataset = load_dataset('math_qa', split='train[:5%]')
            elif dataset_name == 'elyza/ELYZA-tasks-100':
                # ELYZA Tasks 100 (既に統合済み)
                try:
                    dataset = load_dataset('elyza/ELYZA-tasks-100', split='test[:20%]')
                except:
                    logger.warning("[HF] ELYZA dataset not available")
                    return None
            elif dataset_name == 'llm-book/japanese-bookcorpus':
                # Japanese BookCorpus
                try:
                    dataset = load_dataset('llm-book/japanese-bookcorpus', split='train[:1%]')  # 大規模なので1%のみ
                except:
                    logger.warning("[HF] Japanese BookCorpus not available")
                    return None
            elif dataset_name == 'izumi-lab/llm-japanese-dataset':
                # LLM Japanese Dataset
                try:
                    dataset = load_dataset('izumi-lab/llm-japanese-dataset', split='train[:10%]')
                except:
                    logger.warning("[HF] LLM Japanese Dataset not available")
                    return None
            elif dataset_name == 'pfnet/plamo-text-dataset':
                # PLaMo Text Dataset
                try:
                    dataset = load_dataset('pfnet/plamo-text-dataset', split='train[:5%]')
                except:
                    logger.warning("[HF] PLaMo Text Dataset not available")
                    return None
            elif dataset_name == 'yuzuai/rakuda-questions':
                # Rakuda Questions
                try:
                    dataset = load_dataset('yuzuai/rakuda-questions', split='train[:20%]')
                except:
                    logger.warning("[HF] Rakuda Questions not available")
                    return None
            elif dataset_name == 'hotchpotch/jaqket_v2':
                # JAQKET v2
                try:
                    dataset = load_dataset('hotchpotch/jaqket_v2', split='train[:10%]')
                except:
                    logger.warning("[HF] JAQKET v2 not available")
                    return None
            elif dataset_name == 'llm-book/wikinews-ja':
                # WikiNews Japanese
                try:
                    dataset = load_dataset('llm-book/wikinews-ja', split='train[:10%]')
                except:
                    logger.warning("[HF] WikiNews Japanese not available")
                    return None
            elif dataset_name == 'llm-book/wikinews-ja-llm-qadataset':
                # WikiNews Japanese QA
                try:
                    dataset = load_dataset('llm-book/wikinews-ja-llm-qadataset', split='train[:20%]')
                except:
                    logger.warning("[HF] WikiNews Japanese QA not available")
                    return None
            elif dataset_name == 'hatakeyama-llm-team/japanese-wikipedia-paragraphs':
                # Japanese Wikipedia Paragraphs
                try:
                    dataset = load_dataset('hatakeyama-llm-team/japanese-wikipedia-paragraphs', split='train[:2%]')
                except:
                    logger.warning("[HF] Japanese Wikipedia Paragraphs not available")
                    return None
            elif dataset_name == 'hatakeyama-llm-team/japanese-wikipedia-captions':
                # Japanese Wikipedia Captions
                try:
                    dataset = load_dataset('hatakeyama-llm-team/japanese-wikipedia-captions', split='train[:10%]')
                except:
                    logger.warning("[HF] Japanese Wikipedia Captions not available")
                    return None
            elif dataset_name == 'allenai/real-toxicity-prompts':
                # Real Toxicity Prompts Dataset (Safety and toxicity detection)
                try:
                    dataset = load_dataset('allenai/real-toxicity-prompts', split='train[:10%]')
                    logger.info("[HF] Loaded toxicity detection dataset for safety training")
                except Exception as e:
                    logger.warning(f"[HF] Real Toxicity Prompts not available: {e}")
                    return None
            elif dataset_name == 'facebook/poisoned_generation_detection':
                # Poisoned Generation Detection Dataset (Safety training)
                try:
                    dataset = load_dataset('facebook/poisoned_generation_detection', split='train[:10%]')
                    logger.info("[HF] Loaded poisoned generation detection dataset for safety training")
                except Exception as e:
                    logger.warning(f"[HF] Poisoned Generation Detection not available: {e}")
                    return None
            elif dataset_name == 'cardiffnlp/tweet_sentiment_multilingual':
                # Multilingual Tweet Sentiment (Content safety context)
                try:
                    dataset = load_dataset('cardiffnlp/tweet_sentiment_multilingual', split='train[:5%]')
                    logger.info("[HF] Loaded multilingual sentiment dataset for content safety context")
                except Exception as e:
                    logger.warning(f"[HF] Tweet Sentiment Multilingual not available: {e}")
                    return None
            elif dataset_name == 'timdettmers/openassistant-guanaco':
                # OpenAssistant Guanaco (Instruction tuning for tool use)
                try:
                    dataset = load_dataset('timdettmers/openassistant-guanaco', split='train[:10%]')
                    logger.info("[HF] Loaded OpenAssistant Guanaco for instruction tuning and tool use")
                except Exception as e:
                    logger.warning(f"[HF] OpenAssistant Guanaco not available: {e}")
                    return None
            elif dataset_name == 'Open-Orca/OpenOrca':
                # OpenOrca (Comprehensive instruction tuning)
                try:
                    dataset = load_dataset('Open-Orca/OpenOrca', split='train[:5%]')  # Large dataset, small portion
                    logger.info("[HF] Loaded OpenOrca for comprehensive instruction tuning")
                except Exception as e:
                    logger.warning(f"[HF] OpenOrca not available: {e}")
                    return None
            elif dataset_name == 'garage-bAInd/aoa':
                # AOA (API Calling dataset)
                try:
                    dataset = load_dataset('garage-bAInd/aoa', split='train[:10%]')
                    logger.info("[HF] Loaded AOA dataset for API calling capabilities")
                except Exception as e:
                    logger.warning(f"[HF] AOA not available: {e}")
                    return None
            elif dataset_name == 'TIGER-Lab/MATH':
                # MATH Dataset (Mathematical reasoning for tool use)
                try:
                    dataset = load_dataset('TIGER-Lab/MATH', split='train[:10%]')
                    logger.info("[HF] Loaded MATH dataset for mathematical reasoning and tool use")
                except Exception as e:
                    logger.warning(f"[HF] MATH not available: {e}")
                    return None
            elif dataset_name == 'microsoft/orca-math-word-problems-200k':
                # Orca Math Word Problems (Mathematical tool use)
                try:
                    dataset = load_dataset('microsoft/orca-math-word-problems-200k', split='train[:10%]')
                    logger.info("[HF] Loaded Orca Math Word Problems for mathematical tool use")
                except Exception as e:
                    logger.warning(f"[HF] Orca Math Word Problems not available: {e}")
                    return None
            elif dataset_name == 'Anthropic/hh-rlhf':
                # HH-RLHF (Helpful and Harmless RLHF)
                try:
                    dataset = load_dataset('Anthropic/hh-rlhf', split='train[:5%]')  # Large dataset
                    logger.info("[HF] Loaded HH-RLHF for safety and helpfulness alignment")
                except Exception as e:
                    logger.warning(f"[HF] HH-RLHF not available: {e}")
                    return None
            elif dataset_name == 'Dahoas/rm-static':
                # RM Static (Reward Modeling)
                try:
                    dataset = load_dataset('Dahoas/rm-static', split='train[:10%]')
                    logger.info("[HF] Loaded RM Static for reward modeling and preference learning")
                except Exception as e:
                    logger.warning(f"[HF] RM Static not available: {e}")
                    return None
            elif dataset_name == 'jondurbin/airoboros-2.1':
                # Airoboros 2.1 (Advanced instruction tuning)
                try:
                    dataset = load_dataset('jondurbin/airoboros-2.1', split='train[:5%]')  # Large dataset
                    logger.info("[HF] Loaded Airoboros 2.1 for advanced instruction tuning")
                except Exception as e:
                    logger.warning(f"[HF] Airoboros 2.1 not available: {e}")
                    return None
            elif dataset_name == 'cognitivecomputations/dolphin':
                # Dolphin (Uncensored instruction tuning)
                try:
                    dataset = load_dataset('cognitivecomputations/dolphin', split='train[:5%]')  # Large dataset
                    logger.info("[HF] Loaded Dolphin for uncensored instruction tuning")
                except Exception as e:
                    logger.warning(f"[HF] Dolphin not available: {e}")
                    return None
            elif dataset_name == 'allenai/tulu-2' or dataset_name == 'allenai/tulu-3' or dataset_name == 'allenai/tulu-v2' or dataset_name == 'allenai/tulu-v3':
                # Tulu series (Tool use and function calling)
                try:
                    dataset = load_dataset(dataset_name, split='train[:10%]')
                    logger.info(f"[HF] Loaded {dataset_name} for tool use and function calling")
                except Exception as e:
                    logger.warning(f"[HF] {dataset_name} not available: {e}")
                    return None
            elif dataset_name == 'HuggingFaceH4/no_robots':
                # No Robots Safety Dataset
                try:
                    dataset = load_dataset('HuggingFaceH4/no_robots', split='train[:10%]')
                    logger.info("[HF] Loaded safety instruction dataset")
                except Exception as e:
                    logger.warning(f"[HF] No Robots dataset not available: {e}")
                    return None
            elif dataset_name == 'Anthropic/SafeRLHF':
                # SafeRLHF Dataset (Large dataset, use small portion for detection training)
                try:
                    dataset = load_dataset('Anthropic/SafeRLHF', split='train[:2%]')  # Very large, use tiny portion
                    logger.info("[HF] Loaded SafeRLHF dataset for safety alignment (detection purpose only)")
                except Exception as e:
                    logger.warning(f"[HF] SafeRLHF dataset not available: {e}")
                    return None
            else:
                logger.warning(f"[HF] Unknown dataset: {dataset_name}")
                return None

            # 基本的なフィルタリング
            filtered_dataset = self._basic_filtering(dataset)

            logger.info(f"[HF] Loaded {len(filtered_dataset)} samples from {dataset_name}")
            return filtered_dataset

        except Exception as e:
            logger.error(f"[HF] Failed to load {dataset_name}: {e}")
            return None

    def _download_moonshot_dataset(self, dataset_type):
        """ムーンショットパイプラインのデータセットをダウンロード"""
        try:
            logger.info(f"[MOONSHOT] Loading {dataset_type} dataset")

            if dataset_type == 'domain_knowledge':
                # ドメイン知識データセット（科学・技術・数学・哲学）
                dataset = self._create_domain_knowledge_dataset()
            elif dataset_type == 'arxiv_papers':
                # Arxiv論文データセット
                dataset = self._create_arxiv_papers_dataset()
            elif dataset_type == 'nsfw_filtered':
                # NSFWフィルタリング済みデータセット
                dataset = self._create_nsfw_filtered_dataset()
            elif dataset_type == 'nsfw_detection':
                # NSFW検知トレーニングデータセット
                dataset = self._create_nsfw_detection_dataset()
            elif dataset_type == 'mcp_skills_integration':
                # MCPスキル統合トレーニングデータセット
                dataset = self._create_mcp_skills_dataset()
            elif dataset_type == 'quadrality_allow_escalate_deny_refuse':
                # 四重推論ALLOWESCALETONDENYREFUSEデータセット
                dataset = self._create_quadrality_decision_dataset()
            else:
                logger.warning(f"[MOONSHOT] Unknown dataset type: {dataset_type}")
                return None

            # 基本的なフィルタリング
            filtered_dataset = self._basic_filtering(dataset)

            logger.info(f"[MOONSHOT] Loaded {len(filtered_dataset)} samples from {dataset_type}")
            return filtered_dataset

        except Exception as e:
            logger.error(f"[MOONSHOT] Failed to load {dataset_type}: {e}")
            return None

    def _create_domain_knowledge_dataset(self):
        """ドメイン知識データセットを作成"""
        domain_samples = []

        # 科学分野
        science_domains = [
            ("Physics", "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales."),
            ("Chemistry", "Organic chemistry studies carbon-containing compounds and their reactions."),
            ("Biology", "Genetics is the study of genes, genetic variation, and heredity."),
            ("Mathematics", "Topology studies properties of space that are preserved under continuous deformations."),
            ("Computer Science", "Algorithm complexity analysis determines resource usage of algorithms."),
            ("Philosophy", "Epistemology studies the nature and origin of knowledge."),
            ("Economics", "Game theory models strategic interactions between rational decision-makers."),
            ("Psychology", "Cognitive psychology studies mental processes including perception and memory.")
        ]

        for domain, description in science_domains:
            domain_samples.append({
                'text': f"Domain: {domain}\\nKnowledge: {description}",
                'domain': domain,
                'type': 'domain_knowledge',
                'difficulty': 'basic'
            })

        # 専門知識サンプル
        advanced_samples = [
            {
                'text': "In quantum field theory, the Higgs mechanism explains how gauge bosons acquire mass through spontaneous symmetry breaking.",
                'domain': 'Physics',
                'type': 'domain_knowledge',
                'difficulty': 'advanced'
            },
            {
                'text': "The Riemann Hypothesis states that all non-trivial zeros of the Riemann zeta function have real part 1/2.",
                'domain': 'Mathematics',
                'type': 'domain_knowledge',
                'difficulty': 'advanced'
            },
            {
                'text': "Gödel's incompleteness theorems demonstrate that any consistent formal system powerful enough to describe arithmetic is incomplete.",
                'domain': 'Philosophy',
                'type': 'domain_knowledge',
                'difficulty': 'advanced'
            }
        ]

        domain_samples.extend(advanced_samples)

        # Dataset形式に変換
        dataset = Dataset.from_pandas(pd.DataFrame(domain_samples))
        logger.info(f"[DOMAIN] Created {len(domain_samples)} domain knowledge samples")
        return dataset

    def _create_arxiv_papers_dataset(self):
        """Arxiv論文データセットを作成"""
        arxiv_samples = []

        # Arxiv論文の代表的なトピック
        arxiv_topics = [
            ("cs.AI", "Artificial Intelligence", "Recent advances in transformer architectures have revolutionized natural language processing."),
            ("cs.LG", "Machine Learning", "Deep learning models achieve state-of-the-art performance on various benchmark datasets."),
            ("math.CO", "Combinatorics", "Graph theory provides fundamental tools for analyzing network structures and algorithms."),
            ("physics.quant-ph", "Quantum Physics", "Quantum computing offers exponential speedup for certain computational problems."),
            ("stat.ML", "Statistical Learning", "Bayesian methods provide probabilistic frameworks for machine learning."),
            ("cs.CV", "Computer Vision", "Convolutional neural networks excel at image recognition and classification tasks."),
            ("math.PR", "Probability", "Stochastic processes model random phenomena evolving over time."),
            ("cs.DS", "Data Structures", "Efficient algorithms and data structures are crucial for scalable computing.")
        ]

        for arxiv_id, category, abstract in arxiv_topics:
            arxiv_samples.append({
                'text': f"ArXiv ID: {arxiv_id}\\nCategory: {category}\\nAbstract: {abstract}",
                'arxiv_id': arxiv_id,
                'category': category,
                'type': 'arxiv_paper',
                'difficulty': 'intermediate'
            })

        # 追加の論文サンプル
        advanced_papers = [
            {
                'text': "arXiv:1706.03762 [cs.CL] - Attention Is All You Need - The transformer architecture uses self-attention mechanisms.",
                'arxiv_id': '1706.03762',
                'category': 'cs.CL',
                'type': 'arxiv_paper',
                'difficulty': 'advanced'
            },
            {
                'text': "arXiv:2005.11401 [cs.LG] - Deep Residual Learning for Image Recognition - ResNet architecture enables very deep neural networks.",
                'arxiv_id': '2005.11401',
                'category': 'cs.CV',
                'type': 'arxiv_paper',
                'difficulty': 'advanced'
            }
        ]

        arxiv_samples.extend(advanced_papers)

        # Dataset形式に変換
        dataset = Dataset.from_pandas(pd.DataFrame(arxiv_samples))
        logger.info(f"[ARXIV] Created {len(arxiv_samples)} ArXiv paper samples")
        return dataset

    def _create_nsfw_filtered_dataset(self):
        """NSFWフィルタリング済みデータセットを作成"""
        # 注意: NSFWコンテンツは含まない、安全な代替データセット
        safe_samples = []

        # 創造性と表現力のデータセット
        creative_content = [
            ("Creative Writing", "Poetry uses metaphor and imagery to convey complex emotions and ideas."),
            ("Art Theory", "Abstract expressionism emphasizes spontaneous creation and emotional expression."),
            ("Music Theory", "Harmony in music refers to the combination of simultaneously sounded musical notes."),
            ("Film Studies", "Cinematography uses camera techniques to enhance storytelling and emotional impact."),
            ("Literature", "The novel as a literary form allows for deep character development and complex narratives."),
            ("Design", "Minimalist design philosophy emphasizes simplicity and functionality in aesthetics.")
        ]

        for category, description in creative_content:
            safe_samples.append({
                'text': f"Category: {category}\\nContent: {description}",
                'category': category,
                'type': 'creative_content',
                'nsfw_filtered': True,
                'difficulty': 'basic'
            })

        # 表現力豊かなサンプル
        expressive_samples = [
            {
                'text': "Creative expression through art allows humans to communicate complex ideas beyond verbal language.",
                'category': 'Art',
                'type': 'creative_content',
                'nsfw_filtered': True,
                'difficulty': 'intermediate'
            },
            {
                'text': "Literature explores the depths of human consciousness, relationships, and societal structures.",
                'category': 'Literature',
                'type': 'creative_content',
                'nsfw_filtered': True,
                'difficulty': 'intermediate'
            }
        ]

        safe_samples.extend(expressive_samples)

        # Dataset形式に変換
        dataset = Dataset.from_pandas(pd.DataFrame(safe_samples))
        logger.info(f"[NSFW] Created {len(safe_samples)} NSFW-filtered creative content samples")
        return dataset

    def _create_nsfw_detection_dataset(self):
        """NSFW検知トレーニングデータセットを作成（検知目的のみ）"""
        # 注意: NSFWコンテンツは含まず、検知能力を養うための教育的なデータセット
        detection_samples = []

        # 安全なコンテンツの分類学習
        safe_content_examples = [
            ("ニュース記事", "今日は天気が良く、公園で多くの人がピクニックを楽しんでいました。", "safe"),
            ("科学論文", "量子力学の基本原理について研究を行いました。", "safe"),
            ("料理レシピ", "野菜を炒めて塩コショウで味付けしてください。", "safe"),
            ("スポーツ報道", "サッカーの試合で素晴らしいゴールが生まれました。", "safe"),
            ("教育コンテンツ", "数学の基礎概念を学習しましょう。", "safe")
        ]

        # 不適切な表現の例（教育目的のみ）
        inappropriate_indicators = [
            ("露出的表現", "不適切な服装で公の場に現れる", "inappropriate"),
            ("暴力表現", "過度な暴力シーンを含むコンテンツ", "inappropriate"),
            ("差別表現", "人種や性別に対する偏見を含む表現", "inappropriate"),
            ("脅迫表現", "脅迫めいた表現や行動", "inappropriate"),
            ("有害表現", "精神的危害を及ぼす可能性のある表現", "inappropriate")
        ]

        # 教育的な検知トレーニングデータ
        for category, description, label in safe_content_examples:
            detection_samples.append({
                'text': f"カテゴリ: {category}\\n内容: {description}\\n分類: {label}",
                'content_type': category,
                'description': description,
                'safety_label': label,
                'type': 'nsfw_detection_training',
                'purpose': 'detection_training',
                'difficulty': 'basic'
            })

        for indicator_type, example, label in inappropriate_indicators:
            detection_samples.append({
                'text': f"指標タイプ: {indicator_type}\\n例: {example}\\n分類: {label}\\n注意: これは教育目的の例であり、実際のコンテンツは含みません。",
                'indicator_type': indicator_type,
                'example': example,
                'safety_label': label,
                'type': 'nsfw_detection_training',
                'purpose': 'educational_example',
                'difficulty': 'intermediate'
            })

        # 安全ガイドラインの学習データ
        safety_guidelines = [
            {
                'text': "コンテンツの安全性を評価する際は、文脈と意図を考慮してください。",
                'guideline': 'context_awareness',
                'type': 'nsfw_detection_training',
                'purpose': 'guideline_learning'
            },
            {
                'text': "曖昧な表現については、保守的に安全側に分類することを推奨します。",
                'guideline': 'conservative_approach',
                'type': 'nsfw_detection_training',
                'purpose': 'guideline_learning'
            },
            {
                'text': "検知能力はコンテンツの保護と表現の自由のバランスを取るために重要です。",
                'guideline': 'balance_consideration',
                'type': 'nsfw_detection_training',
                'purpose': 'guideline_learning'
            }
        ]

        detection_samples.extend(safety_guidelines)

        # Dataset形式に変換
        dataset = Dataset.from_pandas(pd.DataFrame(detection_samples))
        logger.info(f"[NSFW-DETECTION] Created {len(detection_samples)} NSFW detection training samples")
        return dataset

    def _create_mcp_skills_dataset(self):
        """MCPスキル統合トレーニングデータセットを作成"""
        mcp_samples = []

        # MCPスキルの基本概念
        mcp_concepts = [
            ("ツールコール", "外部ツールやサービスを呼び出す機能", "MCPツールコールにより、AIは外部リソースにアクセスできます。"),
            ("サーバー統合", "複数のMCPサーバーを統合管理", "異なるサーバーのツールを統一的に扱うことができます。"),
            ("プロトコル標準", "Model Context Protocolの標準仕様", "標準化されたインターフェースでツールを呼び出せます。"),
            ("セキュリティ", "ツールコールの安全な実行", "適切な権限管理と検証を行います。"),
            ("エラーハンドリング", "ツールコール失敗時の対応", "エラーを適切に処理し、代替手段を提供します。")
        ]

        for concept, description, detail in mcp_concepts:
            mcp_samples.append({
                'text': f"MCP概念: {concept}\\n説明: {description}\\n詳細: {detail}",
                'concept': concept,
                'description': description,
                'detail': detail,
                'type': 'mcp_skills_integration',
                'skill_category': 'basic_concept',
                'difficulty': 'intermediate'
            })

        # スキル使用パターン
        skill_usage_patterns = [
            ("ファイル操作", "read_file, write_file, list_dirなどのファイル操作スキル", "ファイルシステムとの連携に使用"),
            ("ウェブアクセス", "web_search, fetch_urlなどのウェブアクセススキル", "情報検索とデータ取得に使用"),
            ("データ分析", "analyze_data, generate_chartなどの分析スキル", "データの処理と視覚化に使用"),
            ("コミュニケーション", "send_email, create_documentなどのコミュニケーションツール", "外部との連携に使用"),
            ("システム管理", "run_command, monitor_systemなどのシステムスキル", "システム操作に使用")
        ]

        for skill_type, tools, usage in skill_usage_patterns:
            mcp_samples.append({
                'text': f"スキルタイプ: {skill_type}\\n利用可能なツール: {tools}\\n使用目的: {usage}",
                'skill_type': skill_type,
                'available_tools': tools,
                'usage_purpose': usage,
                'type': 'mcp_skills_integration',
                'skill_category': 'usage_pattern',
                'difficulty': 'intermediate'
            })

        # 実際のスキルコール例
        skill_call_examples = [
            {
                'text': "ユーザーのクエリに対して適切なMCPスキルを判断し、ツールを呼び出す。",
                'scenario': 'skill_selection',
                'type': 'mcp_skills_integration',
                'skill_category': 'practical_usage'
            },
            {
                'text': "ツールコールの結果を解釈し、ユーザーに適切な形で返す。",
                'scenario': 'result_interpretation',
                'type': 'mcp_skills_integration',
                'skill_category': 'practical_usage'
            },
            {
                'text': "複数のツールを組み合わせた複雑なタスクを実行する。",
                'scenario': 'tool_combination',
                'type': 'mcp_skills_integration',
                'skill_category': 'advanced_usage'
            }
        ]

        mcp_samples.extend(skill_call_examples)

        # Dataset形式に変換
        dataset = Dataset.from_pandas(pd.DataFrame(mcp_samples))
        logger.info(f"[MCP] Created {len(mcp_samples)} MCP skills integration samples")
        return dataset

    def _create_quadrality_decision_dataset(self):
        """四重推論ALLOWESCALETONDENYREFUSEデータセットを作成"""
        decision_samples = []

        # 四重推論の意思決定プロセス
        quadrality_decisions = [
            ("倫理的ジレンマ", "自動運転車の判断: 乗員を救うか歩行者を救うか", "ALLOW: 倫理的考察が必要", "ESCALATE: 専門家判断へ", "DENY: 自動判断不可", "REFUSE: 人間の判断が必要"),
            ("セキュリティ問題", "不審なアクセスパターンの検知", "ALLOW: 通常アクセス", "ESCALATE: セキュリティチームへ", "DENY: アクセス拒否", "REFUSE: 追加認証要求"),
            ("リソース制限", "計算リソースの過度な使用要求", "ALLOW: 通常範囲内", "ESCALATE: リソース管理チームへ", "DENY: リソース不足", "REFUSE: 代替手段の検討"),
            ("法的問題", "著作権が曖昧なコンテンツの使用", "ALLOW: フェアユース", "ESCALATE: 法的レビューへ", "DENY: 著作権侵害", "REFUSE: 法的確認が必要"),
            ("品質問題", "出力品質が基準を下回る場合", "ALLOW: 許容範囲内", "ESCALATE: 品質管理チームへ", "DENY: 品質不十分", "REFUSE: 再生成要求")
        ]

        for situation, context, allow_reason, escalate_reason, deny_reason, refuse_reason in quadrality_decisions:
            decision_samples.append({
                'text': f"状況: {situation}\\n文脈: {context}\\nALLOW: {allow_reason}\\nESCALATE: {escalate_reason}\\nDENY: {deny_reason}\\nREFUSE: {refuse_reason}",
                'situation': situation,
                'context': context,
                'decisions': {
                    'ALLOW': allow_reason,
                    'ESCALATE': escalate_reason,
                    'DENY': deny_reason,
                    'REFUSE': refuse_reason
                },
                'type': 'quadrality_allow_escalate_deny_refuse',
                'process_type': 'internal_comparison',
                'difficulty': 'advanced'
            })

        # 意思決定プロセスのトレーニングデータ
        decision_process_examples = [
            {
                'text': "複数の回答候補を生成し、内部で比較評価した上で、最適な回答を選択する。",
                'process_step': 'multi_candidate_generation',
                'type': 'quadrality_allow_escalate_deny_refuse',
                'process_type': 'internal_comparison'
            },
            {
                'text': "各視点（代数的・幾何学的・解析的・位相的）からの回答を比較し、一貫性のある決定を行う。",
                'process_step': 'perspective_consistency_check',
                'type': 'quadrality_allow_escalate_deny_refuse',
                'process_type': 'internal_comparison'
            },
            {
                'text': "ALLOWESCALETONDENYREFUSEの4つのオプションを評価し、最適な行動を選択する。",
                'process_step': 'decision_matrix_evaluation',
                'type': 'quadrality_allow_escalate_deny_refuse',
                'process_type': 'internal_comparison'
            },
            {
                'text': "出力前に複数の推論パスを比較し、最も安全で適切な回答を最終決定する。",
                'process_step': 'pre_output_validation',
                'type': 'quadrality_allow_escalate_deny_refuse',
                'process_type': 'internal_comparison'
            }
        ]

        decision_samples.extend(decision_process_examples)

        # Dataset形式に変換
        dataset = Dataset.from_pandas(pd.DataFrame(decision_samples))
        logger.info(f"[QUADRALITY] Created {len(decision_samples)} quadrality decision making samples")
        return dataset

    def _generate_synthetic_data(self, data_type):
        """合成データを生成（ムーンショットパイプライン拡張版）"""
        try:
            logger.info(f"[SYNTHETIC] Generating {data_type} data")

            if data_type == 'reasoning_problems':
                # 基本的な推論問題を生成
                synthetic_data = self._generate_reasoning_problems(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} reasoning problems")
                return dataset

            elif data_type == 'mathematical_problems':
                # 数学的問題を生成
                synthetic_data = self._generate_mathematical_problems(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} mathematical problems")
                return dataset

            elif data_type == 'science_questions':
                # 科学質問を生成
                synthetic_data = self._generate_science_questions(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} science questions")
                return dataset

            elif data_type == 'philosophical_reasoning':
                # 哲学的推論を生成
                synthetic_data = self._generate_philosophical_reasoning(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} philosophical reasoning samples")
                return dataset

            elif data_type == 'japanese_daily_conversation':
                # 日本語日常会話データを生成
                synthetic_data = self._generate_japanese_daily_conversation(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} Japanese daily conversation samples")
                return dataset

            elif data_type == 'japanese_business_correspondence':
                # 日本語ビジネス文書データを生成
                synthetic_data = self._generate_japanese_business_correspondence(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} Japanese business correspondence samples")
                return dataset

            elif data_type == 'japanese_technical_writing':
                # 日本語技術文書データを生成
                synthetic_data = self._generate_japanese_technical_writing(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} Japanese technical writing samples")
                return dataset

            elif data_type == 'japanese_literary_analysis':
                # 日本語文学分析データを生成
                synthetic_data = self._generate_japanese_literary_analysis(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} Japanese literary analysis samples")
                return dataset

            elif data_type == 'mcp_skill_usage':
                # MCPスキル使用トレーニングデータを生成
                synthetic_data = self._generate_mcp_skill_usage(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} MCP skill usage samples")
                return dataset

            elif data_type == 'nsfw_detection_training':
                # NSFW検知トレーニングデータを生成
                synthetic_data = self._generate_nsfw_detection_training(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} NSFW detection training samples")
                return dataset

            elif data_type == 'quadrality_decision_making':
                # 四重推論意思決定トレーニングデータを生成
                synthetic_data = self._generate_quadrality_decision_making(500)
                dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))
                logger.info(f"[SYNTHETIC] Generated {len(dataset)} quadrality decision making samples")
                return dataset

            else:
                logger.warning(f"[SYNTHETIC] Unknown type: {data_type}")
                return None

        except Exception as e:
            logger.error(f"[SYNTHETIC] Failed to generate {data_type}: {e}")
            return None

    def _generate_reasoning_problems(self, num_samples):
        """基本的な推論問題を生成"""
        problems = []

        templates = [
            "If all {A} are {B}, and some {B} are {C}, then what can we conclude about {A} and {C}?",
            "A train leaves station A at {speed} km/h. Another train leaves station B at {speed2} km/h. When will they meet?",
            "In a group of {num} people, if each person shakes hands with every other person exactly once, how many handshakes occur?",
            "If {num1} workers can complete a job in {days1} days, how long will it take {num2} workers?",
        ]

        for i in range(num_samples):
            template = np.random.choice(templates)

            # パラメータをランダムに生成
            params = {
                'A': np.random.choice(['cats', 'dogs', 'birds', 'fish']),
                'B': np.random.choice(['animals', 'pets', 'creatures', 'beings']),
                'C': np.random.choice(['mammals', 'vertebrates', 'living things', 'organisms']),
                'speed': np.random.randint(50, 120),
                'speed2': np.random.randint(40, 100),
                'num': np.random.randint(5, 20),
                'num1': np.random.randint(2, 10),
                'num2': np.random.randint(3, 15),
                'days1': np.random.randint(5, 20),
            }

            problem = template.format(**params)

            # 簡単な解答生成（実際のモデルで置き換え可能）
            if "handshakes" in problem:
                n = params['num']
                answer = f"The number of handshakes is {n * (n-1) // 2}."
            elif "workers" in problem:
                answer = f"It will take {params['num1'] * params['days1'] // params['num2']} days."
            else:
                answer = "This requires logical reasoning to solve."

            problems.append({
                'problem': problem,
                'answer': answer,
                'type': 'reasoning',
                'difficulty': 'basic'
            })

        return problems

    def _generate_mathematical_problems(self, num_samples):
        """数学的問題を生成（ムーンショット拡張版）"""
        math_problems = []

        # 高度な数学的問題テンプレート
        advanced_templates = [
            "Prove that the derivative of sin(x) is cos(x) using the definition of derivative.",
            "Solve the differential equation dy/dx = ky with initial condition y(0) = 1.",
            "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
            "Evaluate the integral ∫ sin²(x) dx from 0 to π.",
            "Prove that the sum of the first n natural numbers is n(n+1)/2.",
            "Find the limit of (1 + 1/n)^n as n approaches infinity.",
            "Solve the system of equations: 2x + 3y = 7, x - y = 1.",
            "Determine if the series Σ(1/n²) converges using the integral test.",
            "Find the Fourier transform of the function e^(-x²).",
            "Prove that √2 is irrational using proof by contradiction."
        ]

        solution_templates = [
            "Using the fundamental theorem of calculus and trigonometric identities.",
            "This is a separable differential equation with solution y = e^(kx).",
            "The characteristic equation gives eigenvalues λ = 3 and λ = 1.",
            "Using the identity sin²(x) = (1 - cos(2x))/2, the integral evaluates to π/2.",
            "By mathematical induction, the formula holds for the base case and inductive step.",
            "This limit defines the mathematical constant e ≈ 2.718.",
            "Using substitution, we find x = 2 and y = 1.",
            "The integral ∫ dx/x² from 2 to ∞ converges, so the series converges.",
            "The Fourier transform is (√π) e^(-k²/4).",
            "Assume √2 = p/q in lowest terms, then derive a contradiction."
        ]

        for i in range(num_samples):
            problem_idx = i % len(advanced_templates)
            problem = advanced_templates[problem_idx]
            solution = solution_templates[problem_idx]

            math_problems.append({
                'text': f"Mathematical Problem: {problem}\\nSolution Approach: {solution}",
                'problem': problem,
                'solution': solution,
                'type': 'mathematical_problem',
                'difficulty': 'advanced',
                'domain': 'mathematics'
            })

        return math_problems

    def _generate_science_questions(self, num_samples):
        """科学質問を生成（ムーンショット拡張版）"""
        science_questions = []

        # 科学分野の質問と回答
        science_qa = [
            ("Physics", "Why does gravity exist?", "Gravity arises from the curvature of spacetime caused by mass and energy, as described by Einstein's theory of general relativity."),
            ("Chemistry", "How do chemical bonds form?", "Chemical bonds form through the interaction of valence electrons between atoms to achieve stable electron configurations."),
            ("Biology", "How does natural selection work?", "Natural selection favors traits that increase survival and reproduction, leading to adaptation over generations."),
            ("Computer Science", "What is algorithmic complexity?", "Algorithmic complexity measures the computational resources required for an algorithm to solve a problem."),
            ("Neuroscience", "How does memory work?", "Memory involves synaptic plasticity where repeated activation strengthens neural connections."),
            ("Ecology", "What is biodiversity?", "Biodiversity refers to the variety of life forms within ecosystems, crucial for ecosystem stability."),
            ("Astronomy", "How do black holes form?", "Black holes form when massive stars collapse under their own gravity after exhausting nuclear fuel."),
            ("Geology", "What causes earthquakes?", "Earthquakes occur due to the sudden release of built-up stress along tectonic plate boundaries.")
        ]

        for field, question, answer in science_qa:
            for i in range(num_samples // len(science_qa) + 1):
                if len(science_questions) >= num_samples:
                    break

                science_questions.append({
                    'text': f"Scientific Question: {question}\\nField: {field}\\nAnswer: {answer}",
                    'question': question,
                    'answer': answer,
                    'field': field,
                    'type': 'science_question',
                    'difficulty': 'intermediate',
                    'domain': 'science'
                })

        return science_questions[:num_samples]

    def _generate_philosophical_reasoning(self, num_samples):
        """哲学的推論を生成（ムーンショット拡張版）"""
        philosophical_samples = []

        # 哲学的思考のサンプル
        philosophy_topics = [
            ("Epistemology", "How do we know what we know?", "Knowledge requires justified true belief, but the regress problem questions infinite justification chains."),
            ("Ethics", "What makes an action moral?", "Different ethical frameworks provide varying answers: utilitarianism maximizes happiness, deontology follows rules."),
            ("Metaphysics", "What is the nature of reality?", "Metaphysical questions explore existence, causality, and the fundamental structure of the universe."),
            ("Philosophy of Mind", "What is consciousness?", "Consciousness remains a hard problem - how subjective experience emerges from physical processes."),
            ("Political Philosophy", "What is justice?", "Justice theories range from Rawls' fairness to Nozick's liberty, each defining rights differently."),
            ("Aesthetics", "What is beauty?", "Beauty is subjective yet universal, studied through art, mathematics, and evolutionary psychology."),
            ("Logic", "What constitutes valid reasoning?", "Formal logic provides deductive validity, while informal logic addresses real-world argumentation."),
            ("Philosophy of Science", "How does science progress?", "Scientific progress involves paradigm shifts (Kuhn) and increasing verisimilitude (Popper).")
        ]

        for branch, question, reasoning in philosophy_topics:
            for i in range(num_samples // len(philosophy_topics) + 1):
                if len(philosophical_samples) >= num_samples:
                    break

                philosophical_samples.append({
                    'text': f"Philosophical Question: {question}\\nBranch: {branch}\\nReasoning: {reasoning}",
                    'question': question,
                    'reasoning': reasoning,
                    'branch': branch,
                    'type': 'philosophical_reasoning',
                    'difficulty': 'advanced',
                    'domain': 'philosophy'
                })

        return philosophical_samples[:num_samples]

    def _generate_mcp_skill_usage(self, num_samples):
        """MCPスキル使用トレーニングデータを生成（ムーンショット拡張版）"""
        mcp_usage_samples = []

        # MCPスキル使用シナリオ
        mcp_scenarios = [
            ("ファイル検索", "特定の情報を含むファイルを検索する必要がある", "read_file, grep_search, list_dir"),
            ("データ分析", "大量のデータを分析して洞察を得る", "analyze_data, generate_chart, statistical_analysis"),
            ("ウェブ調査", "最新の情報を収集する", "web_search, fetch_url, summarize_content"),
            ("システム監視", "システムの状態をチェックする", "monitor_system, check_resources, log_analysis"),
            ("ドキュメント作成", "レポートやドキュメントを作成する", "create_document, format_text, export_file"),
            ("コミュニケーション", "チームメンバーと連携する", "send_message, schedule_meeting, share_files"),
            ("コード開発", "ソフトウェアを開発する", "run_tests, debug_code, deploy_application"),
            ("セキュリティチェック", "セキュリティを確保する", "scan_vulnerabilities, encrypt_data, access_control")
        ]

        skill_response_patterns = [
            ("ツール選択", "クエリを分析し、適切なMCPツールを選択する"),
            ("パラメータ設定", "ツールに渡すパラメータを正しく設定する"),
            ("実行結果解釈", "ツールの実行結果を理解し、回答に活用する"),
            ("エラーハンドリング", "ツール実行エラーが発生した場合の対応"),
            ("フォールバック", "ツールが利用できない場合の代替手段"),
            ("結果統合", "複数のツール結果を統合して回答を作成")
        ]

        for i in range(num_samples):
            scenario_idx = i % len(mcp_scenarios)
            pattern_idx = i % len(skill_response_patterns)

            task, description, tools = mcp_scenarios[scenario_idx]
            pattern_type, pattern_desc = skill_response_patterns[pattern_idx]

            mcp_usage_samples.append({
                'text': f"タスク: {task}\\n説明: {description}\\n利用ツール: {tools}\\n応答パターン: {pattern_type} - {pattern_desc}",
                'task': task,
                'description': description,
                'available_tools': tools,
                'response_pattern': pattern_type,
                'pattern_description': pattern_desc,
                'type': 'mcp_skill_usage',
                'difficulty': 'intermediate',
                'domain': 'tool_integration'
            })

        return mcp_usage_samples

    def _generate_nsfw_detection_training(self, num_samples):
        """NSFW検知トレーニングデータを生成（HFデータセット統合版）"""
        nsfw_training_samples = []

        # HF NSFW検知データセットから学習したパターン
        hf_nsfw_patterns = [
            ("adult_content", "成人向けコンテンツの特徴パターン", "nsfw"),
            ("hate_speech", "ヘイトスピーチや差別表現", "harmful"),
            ("violent_content", "暴力的な表現や描写", "harmful"),
            ("spam_content", "スパムや誤情報", "unsafe"),
            ("safe_content", "安全で適切なコンテンツ", "safe"),
            ("borderline_content", "境界線上の曖昧なコンテンツ", "borderline")
        ]

        # 検知トレーニングシナリオ
        detection_scenarios = [
            ("automatic_filtering", "AIによる自動コンテンツフィルタリング"),
            ("human_moderation", "人間モデレーターのアシスト"),
            ("context_analysis", "文脈を考慮した安全性評価"),
            ("risk_assessment", "潜在的な危害リスク評価"),
            ("policy_compliance", "プラットフォームポリシーの遵守確認"),
            ("user_reporting", "ユーザー報告への対応"),
            ("age_restriction", "年齢制限コンテンツの管理"),
            ("cultural_sensitivity", "文化的配慮の確認")
        ]

        # HFデータセットから得られる洞察を統合
        hf_insights = [
            ("pattern_recognition", "NSFWコンテンツの共通パターンを学習"),
            ("context_understanding", "文脈による意味の変化を理解"),
            ("cultural_differences", "文化による表現の違いを考慮"),
            ("evolving_language", "言語の変化と新しい表現への対応"),
            ("false_positives", "誤検知の低減と精度向上"),
            ("multilingual_support", "多言語対応の検知能力")
        ]

        for i in range(num_samples):
            pattern_idx = i % len(hf_nsfw_patterns)
            scenario_idx = i % len(detection_scenarios)
            insight_idx = i % len(hf_insights)

            pattern_type, pattern_desc, risk_level = hf_nsfw_patterns[pattern_idx]
            scenario_type, scenario_desc = detection_scenarios[scenario_idx]
            insight_type, insight_desc = hf_insights[insight_idx]

            nsfw_training_samples.append({
                'text': f"NSFWパターン: {pattern_type}\\n説明: {pattern_desc}\\nリスクレベル: {risk_level}\\n検知シナリオ: {scenario_type} - {scenario_desc}\\nHF洞察: {insight_type} - {insight_desc}",
                'pattern_type': pattern_type,
                'pattern_description': pattern_desc,
                'risk_level': risk_level,
                'detection_scenario': scenario_type,
                'scenario_description': scenario_desc,
                'hf_insight': insight_type,
                'insight_description': insight_desc,
                'type': 'nsfw_detection_training',
                'source': 'hf_integrated',
                'difficulty': 'advanced',
                'domain': 'content_safety'
            })

        return nsfw_training_samples

    def _generate_quadrality_decision_making(self, num_samples):
        """四重推論意思決定トレーニングデータを生成（ムーンショット拡張版）"""
        decision_making_samples = []

        # 四重推論の意思決定シナリオ
        decision_scenarios = [
            ("リソース割り当て", "限られた計算リソースを複数のタスクに割り当てる決定", "ALLOW: 通常割り当て, ESCALATE: 優先度評価, DENY: リソース不足, REFUSE: 再検討要求"),
            ("情報公開", "機密情報の外部公開についての決定", "ALLOW: 公開可能情報, ESCALATE: セキュリティレビュー, DENY: 機密情報, REFUSE: 承認プロセス"),
            ("ユーザー要求", "ユーザーの特殊な要求に対する対応", "ALLOW: 標準機能, ESCALATE: 要件分析, DENY: 技術的制約, REFUSE: 代替提案"),
            ("品質保証", "出力品質が基準を満たさない場合の対応", "ALLOW: 許容範囲, ESCALATE: 品質改善, DENY: 品質不十分, REFUSE: 再生成"),
            ("倫理的考慮", "倫理的に問題のあるクエリへの対応", "ALLOW: 中立的回答, ESCALATE: 倫理レビュー, DENY: 不適切クエリ, REFUSE: ガイドライン説明")
        ]

        internal_comparison_processes = [
            ("回答一貫性チェック", "複数の推論パスの回答が一致するか検証"),
            ("安全性評価", "各回答の潜在的なリスクを評価"),
            ("正確性検証", "回答の正確性と信頼性を確認"),
            ("文脈適合性", "クエリの文脈に適した回答か判断"),
            ("包括性評価", "回答がクエリを十分にカバーしているか")
        ]

        for i in range(num_samples):
            scenario_idx = i % len(decision_scenarios)
            process_idx = i % len(internal_comparison_processes)

            scenario, context, decisions = decision_scenarios[scenario_idx]
            process_type, process_desc = internal_comparison_processes[process_idx]

            decision_making_samples.append({
                'text': f"シナリオ: {scenario}\\n文脈: {context}\\n決定オプション: {decisions}\\n内部比較プロセス: {process_type} - {process_desc}",
                'scenario': scenario,
                'context': context,
                'decision_options': decisions,
                'comparison_process': process_type,
                'process_description': process_desc,
                'type': 'quadrality_decision_making',
                'difficulty': 'advanced',
                'domain': 'decision_making'
            })

        return decision_making_samples

    def _generate_japanese_daily_conversation(self, num_samples):
        """日本語日常会話データを生成（ムーンショット拡張版）"""
        daily_conversations = []

        # 日常会話のパターン
        conversation_patterns = [
            ("あいさつ", "おはようございます。今日はいい天気ですね。", "おはようございます。はい、とても気持ちのいい天気です。"),
            ("買い物", "このりんごはいくらですか？", "一個200円です。いくつご入用ですか？"),
            ("道案内", "駅までどう行けばいいですか？", "この道をまっすぐ行って、3つ目の信号を右に曲がってください。"),
            ("天気", "明日の天気予報は聞きましたか？", "はい、晴れの予報です。でも気温が少し低くなるそうです。"),
            ("趣味", "週末は何をされていますか？", "主に読書をしたり、散歩をしたりしています。あなたは？"),
            ("食事", "今日の夕食は何にしますか？", "カレーライスにしようと思っています。あなたは？"),
            ("健康", "最近体調はどうですか？", "おかげさまで元気です。定期的に運動するようにしています。"),
            ("旅行", "来月海外旅行に行く予定です。", "それは素晴らしいですね。どこの国に行かれるのですか？")
        ]

        contexts = ["友人との会話", "家族との会話", "知人との会話", "店員との会話", "同僚との会話"]

        for i in range(num_samples):
            pattern_idx = i % len(conversation_patterns)
            greeting, question, response = conversation_patterns[pattern_idx]
            context = contexts[i % len(contexts)]

            daily_conversations.append({
                'text': f"文脈: {context}\\n会話:\\n{question}\\n{response}",
                'conversation': f"{question}\\n{response}",
                'context': context,
                'type': 'japanese_daily_conversation',
                'difficulty': 'basic',
                'domain': 'daily_life'
            })

        return daily_conversations

    def _generate_japanese_business_correspondence(self, num_samples):
        """日本語ビジネス文書データを生成（ムーンショット拡張版）"""
        business_correspondence = []

        # ビジネス文書の種類
        document_types = [
            ("メール", "件名: 会議の日程調整のお知らせ\\n本文: いつもお世話になっております。\\n株式会社テクノロジーの田中と申します。\\n\\n来週のプロジェクト会議について、日程調整のご相談です。\\n以下の日時でよろしいでしょうか？\\n\\n1. 10月15日（月）14:00-16:00\\n2. 10月16日（火）10:00-12:00\\n3. 10月17日（水）15:00-17:00\\n\\nご都合のよろしい日時をお知らせください。\\n\\n何卒よろしくお願い申し上げます。"),
            ("提案書", "プロジェクト改善提案\\n\\n1. 現状分析\\n現在の業務プロセスには以下の課題があります。\\n- 作業効率の低下\\n- コミュニケーションの不足\\n- 品質管理の難しさ\\n\\n2. 改善策\\n以下の施策を実施することで、効率化を図ります。\\n- 自動化ツールの導入\\n- 定期ミーティングの実施\\n- 品質チェックリストの作成\\n\\n3. 期待効果\\n- 作業時間の30%削減\\n- 品質向上\\n- チーム満足度の向上"),
            ("報告書", "月次業務報告\\n\\n1. 業務実績\\n今月の主要な成果は以下の通りです。\\n- 売上目標の達成（105%）\\n- 新規顧客開拓（20社）\\n- クレーム件数の削減（30%）\\n\\n2. 課題と対策\\n以下の課題が確認されました。\\n- 在庫管理システムの改善が必要\\n- スタッフ研修の強化\\n\\n対策として、システム更新と定期研修を実施いたします。"),
            ("企画書", "新サービス企画書\\n\\n1. 背景と目的\\n市場環境の変化に対応するため、新サービスの開発を提案いたします。\\n\\n2. ターゲット\\n- 20-35歳のビジネスパーソン\\n- テクノロジーに興味のある層\\n\\n3. サービス内容\\n- AIアシスタント機能\\n- クラウドストレージ\\n- リアルタイムコラボレーション\\n\\n4. 収益モデル\\nサブスクリプション制（月額1,500円）"),
            ("議事録", "プロジェクト会議議事録\\n\\n日時: 2024年1月15日 14:00-16:00\\n場所: 会議室A\\n出席者: 田中部長、鈴木課長、佐藤係長、山田係長\\n\\n議題1: Q1予算計画\\n決定事項:\\n- 予算総額を500万円に設定\\n- 優先順位の高いプロジェクトから着手\\n\\n議題2: 人材育成計画\\n決定事項:\\n- 外部研修の実施（2回/年）\\n- 社内メンター制度の導入\\n\\n次回会議: 2024年2月1日")
        ]

        formality_levels = ["敬語", "丁寧語", "常体"]

        for i in range(num_samples):
            doc_idx = i % len(document_types)
            doc_type, content = document_types[doc_idx]
            formality = formality_levels[i % len(formality_levels)]

            business_correspondence.append({
                'text': f"文書種類: {doc_type}\\n敬語レベル: {formality}\\n内容:\\n{content}",
                'document_type': doc_type,
                'formality_level': formality,
                'content': content,
                'type': 'japanese_business_correspondence',
                'difficulty': 'intermediate',
                'domain': 'business'
            })

        return business_correspondence

    def _generate_japanese_technical_writing(self, num_samples):
        """日本語技術文書データを生成（ムーンショット拡張版）"""
        technical_writing = []

        # 技術分野の文書
        technical_fields = [
            ("ソフトウェア開発", "API設計のベストプラクティス\\n\\n1. RESTful原則の遵守\\n- リソースの適切なURI設計\\n- HTTPメソッドの正しい使用\\n- ステータスコードの適切な返却\\n\\n2. セキュリティ対策\\n- 認証・認可の実装\\n- 入力バリデーション\\n- HTTPSの使用\\n\\n3. ドキュメンテーション\\n- OpenAPI仕様の記述\\n- 使用例の提供\\n- エラーハンドリングの説明"),
            ("データサイエンス", "機械学習モデルの評価指標\\n\\n1. 分類問題の指標\\n- Accuracy: 正解率\\n- Precision: 適合率\\n- Recall: 再現率\\n- F1-Score: 調和平均\\n\\n2. 回帰問題の指標\\n- MAE: 平均絶対誤差\\n- MSE: 平均二乗誤差\\n- RMSE: 二乗平均平方根誤差\\n- R²: 決定係数\\n\\n3. モデルの解釈\\n- 特徴量重要度\\n- SHAP値\\n- 部分依存プロット"),
            ("ネットワーク", "クラウドアーキテクチャ設計\\n\\n1. 可用性設計\\n- マルチAZ配置\\n- ロードバランシング\\n- 自動フェイルオーバー\\n\\n2. セキュリティ\\n- VPC設定\\n- セキュリティグループ\\n- IAMポリシー\\n\\n3. パフォーマンス\\n- Auto Scaling\\n- CDN利用\\n- キャッシュ戦略"),
            ("AI/ML", "Transformerアーキテクチャの概要\\n\\n1. アテンション機構\\n- 自己注意の計算\\n- マルチヘッド注意\\n- 位置エンコーディング\\n\\n2. モデル構成\\n- エンコーダ\\n- デコーダ\\n- クロス注意\\n\\n3. 学習手法\\n- 次トークン予測\\n- マスク言語モデル\\n- シーケンス生成"),
            ("ブロックチェーン", "分散台帳技術の基礎\\n\\n1. ブロック構造\\n- トランザクション\\n- ブロックヘッダ\\n- ハッシュチェーン\\n\\n2. コンセンサス\\n- Proof of Work\\n- Proof of Stake\\n- Byzantine Fault Tolerance\\n\\n3. スマートコントラクト\\n- 実行環境\\n- ガス代\\n- セキュリティ考慮")
        ]

        audiences = ["開発者向け", "管理者向け", "一般ユーザー向け", "専門家向け"]

        for i in range(num_samples):
            field_idx = i % len(technical_fields)
            field, content = technical_fields[field_idx]
            audience = audiences[i % len(audiences)]

            technical_writing.append({
                'text': f"技術分野: {field}\\n対象読者: {audience}\\n説明:\\n{content}",
                'technical_field': field,
                'audience': audience,
                'content': content,
                'type': 'japanese_technical_writing',
                'difficulty': 'advanced',
                'domain': 'technology'
            })

        return technical_writing

    def _generate_japanese_literary_analysis(self, num_samples):
        """日本語文学分析データを生成（ムーンショット拡張版）"""
        literary_analysis = []

        # 文学作品の分析
        literary_works = [
            ("夏目漱石『吾輩は猫である』", "近代文学の代表作で、人間社会を猫の視点から風刺的に描いた作品。\\n\\n主要テーマ:\\n- 人間社会の批判\\n- 近代化の矛盾\\n- 知識人の苦悩\\n\\n文体特徴:\\n- 擬人法の使用\\n- 皮肉とユーモア\\n- 漢語と口語の混在"),
            ("村上春樹『ノルウェイの森』", "青春小説の代表作で、1970年代の若者の心の葛藤を描く。\\n\\nテーマ:\\n- 愛と喪失\\n- 孤独とつながり\\n- 成長の痛み\\n\\n特徴:\\n- モノローグ形式\\n- 音楽的表現\\n- 象徴主義的描写"),
            ("川端康成『雪国』", "日本文学の最高峰とされる作品。\\n\\n特徴:\\n- 感覚的表現\\n- 自然描写\\n- 儚い美の表現\\n\\nテーマ:\\n- 愛の儚さ\\n- 伝統と近代\\n- 生死の境界"),
            ("三島由紀夫『金閣寺』", "美と破壊のテーマを扱った作品。\\n\\n分析:\\n- 美意識の探求\\n- 破壊衝動\\n- 宗教的テーマ\\n\\n文体:\\n- 心理描写\\n- 象徴的使用\\n- 流麗な文章"),
            ("谷崎潤一郎『細雪』", "大阪の四姐妹の物語。\\n\\n特徴:\\n- 女性心理の繊細な描写\\n- 季節感の表現\\n- 伝統文化の継承\\n\\nテーマ:\\n- 家族の絆\\n- 近代化の影響\\n- 女性の生き方")
        ]

        analysis_types = ["テーマ分析", "文体分析", "キャラクター分析", "時代背景分析", "影響分析"]

        for i in range(num_samples):
            work_idx = i % len(literary_works)
            work, analysis = literary_works[work_idx]
            analysis_type = analysis_types[i % len(analysis_types)]

            literary_analysis.append({
                'text': f"作品: {work}\\n分析タイプ: {analysis_type}\\n分析内容:\\n{analysis}",
                'literary_work': work,
                'analysis_type': analysis_type,
                'content': analysis,
                'type': 'japanese_literary_analysis',
                'difficulty': 'advanced',
                'domain': 'literature'
            })

        return literary_analysis

    def _basic_filtering(self, dataset):
        """基本的なデータフィルタリング"""
        logger.info("[FILTER] Applying basic filtering...")

        # 設定に基づくフィルタリング
        quality_filters = self.config.get('processing', {}).get('quality_filters', {})

        filtered_data = []

        for item in tqdm(dataset, desc="Filtering"):
            # テキストを取得（複数のフィールドに対応）
            text = ""
            if 'text' in item and item['text']:
                text = item['text']
            elif 'problem' in item and item['problem']:
                text = item['problem']
            elif 'question' in item and item['question']:
                text = item['question']
            elif isinstance(item, dict):
                # 辞書の値を結合
                text = " ".join([str(v) for v in item.values() if isinstance(v, str) and v])
            else:
                # 文字列の場合
                text = str(item) if item else ""

            # 空のテキストをスキップ
            if not text.strip():
                continue

            text_len = len(text.split())

            min_len = quality_filters.get('min_length', 10)
            max_len = quality_filters.get('max_length', 1000)

            # ムーンショットデータセットはより柔軟なフィルタリング
            is_moonshot = item.get('type', '').startswith(('domain_knowledge', 'arxiv_paper', 'creative_content'))
            if is_moonshot:
                # ムーンショットデータは最小長を緩和
                min_len = 3

            if min_len <= text_len <= max_len:
                filtered_data.append(item)

            # サンプル数制限
            if len(filtered_data) >= self.max_samples:
                break

        logger.info(f"[FILTER] Filtered to {len(filtered_data)} samples")
        return filtered_data

    def process_and_save_dataset(self, datasets):
        """データセットを処理して保存"""
        logger.info("[PROCESS] Processing and saving datasets...")

        all_data = []
        for dataset in datasets:
            if hasattr(dataset, '__iter__'):
                all_data.extend(list(dataset))

        # DataFrameに変換
        df = pd.DataFrame(all_data)

        # 重複除去
        initial_count = len(df)
        df = df.drop_duplicates(subset=['problem'] if 'problem' in df.columns else ['text'])
        logger.info(f"[DEDUP] Removed {initial_count - len(df)} duplicates")

        # チャンク分割して保存
        chunk_size = self.chunk_size
        for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk = df.iloc[chunk_start:chunk_end]

            chunk_file = self.processed_dir / f"processed_chunk_{i:03d}.jsonl"

            # JSON Lines形式で保存
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for _, row in chunk.iterrows():
                    json.dump(row.to_dict(), f, ensure_ascii=False)
                    f.write('\n')

            logger.info(f"[SAVE] Saved chunk {i} with {len(chunk)} samples")

        # 統計情報保存
        stats = {
            'total_samples': len(df),
            'chunks': len(list(range(0, len(df), chunk_size))),
            'columns': list(df.columns),
            'sample_types': df.get('type', pd.Series()).value_counts().to_dict() if 'type' in df.columns else {}
        }

        stats_file = self.processed_dir / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"[STATS] Dataset statistics saved to {stats_file}")
        logger.info(f"[SUMMARY] Total processed: {stats['total_samples']} samples")

        return stats

    def run_pipeline(self):
        """データパイプライン実行"""
        logger.info("[START] RTX 3060 Dataset Pipeline Execution")
        logger.info("=" * 50)

        try:
            # データダウンロード
            logger.info("[PHASE 1] Downloading datasets...")
            datasets = self.download_curated_datasets()

            if not datasets:
                logger.error("[ERROR] No datasets downloaded")
                return False

            # データ処理・保存
            logger.info("[PHASE 2] Processing and saving...")
            stats = self.process_and_save_dataset(datasets)

            logger.info("=" * 50)
            logger.info("[SUCCESS] Dataset pipeline completed!")
            logger.info(f"Total samples: {stats['total_samples']}")
            logger.info(f"Number of chunks: {stats['chunks']}")
            logger.info("=" * 50)

            return True

        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}")
            return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description='RTX 3060 Dataset Pipeline')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--max-samples', type=int, help='Maximum samples to process')

    args = parser.parse_args()

    pipeline = RTX3060DatasetPipeline(args.config)

    if args.max_samples:
        pipeline.max_samples = args.max_samples

    success = pipeline.run_pipeline()

    if success:
        print("[SUCCESS] Dataset pipeline completed successfully!")
    else:
        print("[ERROR] Dataset pipeline failed!")
        exit(1)

if __name__ == "__main__":
    main()