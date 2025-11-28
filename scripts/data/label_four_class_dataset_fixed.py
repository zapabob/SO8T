#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四値分類ラベル付けスクリプト

ALLOW/ESCALATION/DENY/REFUSEの自動ラベル付けを実行
Usage:
    python scripts/label_four_class_dataset.py --input data/cleaned --output data/labeled
    python scripts/label_four_class_dataset.py --huggingface --input D:/webdataset/datasets --output D:/webdataset/labeled --test-size 0.2 --val-size 0.1
"""

import json
import logging
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FourClassLabeler:
    """
    LLM PPOベストプラクティスに基づく四値分類ラベル付けクラス

    PPO学習のための分類:
    - ALLOW: 安全で有用な応答（学習に適したポジティブサンプル）
    - ESCALATION: 複雑で専門的な判断が必要な応答
    - DENY: 安全上問題のある内容（ネガティブサンプル）
    - REFUSE: 明確に拒否すべき危険な内容
    """

    # PPOベストプラクティスに基づくキーワード定義
    ALLOW_KEYWORDS = [
        # 安全で有用な応答を示すキーワード
        # 日本語
        "説明します", "教えます", "お手伝いします", "ご案内します", "分かりました",
        "理解しました", "お答えします", "ご紹介します", "ご説明します", "お勧めします",
        "安全です", "問題ありません", "許可します", "可能です", "利用可能です",
        "役立ちます", "参考になります", "役に立つ", "便利です", "効果的です",
        # 英語
        "explain", "help", "assist", "guide", "understand", "answer", "recommend",
        "safe", "allowed", "possible", "useful", "helpful", "effective", "beneficial",
        "appropriate", "acceptable", "valid", "correct", "right", "good"
    ]

    ESCALATION_KEYWORDS = [
        # より高度な判断や専門知識が必要な内容
        # 日本語
        "専門家", "医師", "弁護士", "コンサルタント", "専門知識", "専門的な",
        "複雑", "高度", "詳細", "精密", "正確", "慎重", "注意深く", "十分な検討",
        "調査", "研究", "分析", "評価", "査定", "審査", "検証", "確認",
        "専門的なアドバイス", "医療的な", "法的", "技術的な", "科学的な",
        # 英語
        "expert", "specialist", "professional", "complex", "advanced", "detailed",
        "careful", "thorough", "investigate", "research", "analyze", "evaluate",
        "verify", "confirm", "medical", "legal", "technical", "scientific",
        "consultation", "assessment", "review", "validation"
    ]

    DENY_KEYWORDS = [
        # 安全上問題のある内容を示すキーワード
        # 日本語
        "危険", "有害", "不適切", "不安全", "問題", "懸念", "リスク", "不安",
        "誤り", "間違い", "不正確", "誤解を招く", "誤情報", "フェイク", "デマ",
        "偏見", "差別", "攻撃的", "攻撃性", "脅威", "脅迫", "嫌がらせ", "ハラスメント",
        "プライバシー侵害", "個人情報漏洩", "機密情報", "セキュリティ問題",
        # 英語
        "dangerous", "harmful", "inappropriate", "unsafe", "problem", "concern",
        "risk", "worrying", "wrong", "incorrect", "inaccurate", "misleading",
        "false", "biased", "discriminatory", "offensive", "threatening",
        "harassment", "privacy violation", "security issue", "confidential"
    ]

    REFUSE_KEYWORDS = [
        # 明確に拒否すべき内容を示すキーワード
        # 日本語
        "違法", "犯罪", "禁止", "禁止されている", "違反", "無効", "無許可",
        "拒否", "断る", "お断り", "対応不可", "応答不可", "お手伝いできません",
        "協力できません", "関与できません", "参加できません", "許可できません",
        "不可能", "実行不可", "利用不可", "アクセス不可", "提供不可",
        "自殺", "殺人", "暴力", "テロ", "麻薬", "違法薬物", "児童虐待",
        # 英語
        "illegal", "criminal", "prohibited", "forbidden", "violation", "invalid",
        "unauthorized", "refuse", "decline", "cannot", "unable", "impossible",
        "suicide", "murder", "violence", "terrorism", "drugs", "child abuse",
        "hacking", "exploit", "malware", "virus", "attack", "breach"
    ]
    
    def __init__(self, balance_classes: bool = True):
        """
        Args:
            balance_classes: クラスバランスを調整するかどうか
        """
        self.balance_classes = balance_classes
        logger.info("Four Class Labeler initialized")
    
    def classify_text(self, text: str) -> str:
        """
        LLM PPOベストプラクティスに基づく四値分類

        以下の基準で分類:
        - REFUSE: 明確に拒否すべき危険・違法内容
        - DENY: 安全上問題のある内容
        - ESCALATION: 専門判断が必要な内容
        - ALLOW: 安全で有用な内容
        """
        if not text or len(text.strip()) == 0:
            return "DENY"

        text_lower = text.lower()
        text_length = len(text.strip())

        # PPOベストプラクティス: Safety-Firstアプローチ
        # 1. REFUSE判定（最優先: 明確な危険信号）
        refuse_indicators = [
            # 違法・危険活動
            any(kw in text_lower for kw in ["kill", "murder", "suicide", "bomb", "terrorism"]),
            # 違法薬物・犯罪
            any(kw in text_lower for kw in ["drugs", "cocaine", "heroin", "meth", "illegal drugs"]),
            # 児童関連の危険
            any(kw in text_lower for kw in ["child abuse", "pedophilia", "child exploitation"]),
            # ハッキング・セキュリティ侵害
            any(kw in text_lower for kw in ["hack", "exploit", "breach", "malware", "virus"]),
            # 明確な拒否表現
            any(kw in text_lower for kw in ["cannot comply", "will not help", "refuse to"])
        ]

        if any(refuse_indicators):
            return "REFUSE"

        # 2. DENY判定（安全上問題のある内容）
        deny_indicators = [
            # 有害コンテンツ
            any(kw in text_lower for kw in ["harmful", "dangerous", "unsafe", "threat"]),
            # 不適切表現
            any(kw in text_lower for kw in ["offensive", "abusive", "harassment", "bullying"]),
            # 誤情報・偏見
            any(kw in text_lower for kw in ["misinformation", "biased", "discriminatory"]),
            # プライバシー侵害
            any(kw in text_lower for kw in ["personal information", "confidential", "private"]),
            # 過度に短い拒否応答
            text_length < 20 and any(kw in text_lower for kw in ["no", "cannot", "unable"])
        ]

        if any(deny_indicators):
            return "DENY"

        # 3. ESCALATION判定（専門知識が必要）
        escalation_indicators = [
            # 専門領域
            any(kw in text_lower for kw in ["medical", "legal", "financial", "technical"]),
            # 複雑な判断
            any(kw in text_lower for kw in ["complex", "advanced", "specialized", "expert"]),
            # 調査・確認が必要
            any(kw in text_lower for kw in ["research", "investigate", "verify", "confirm"]),
            # 質問形式（複雑なクエリ）
            ("?" in text or "？" in text) and text_length > 100,
            # 複数の条件判断
            sum(1 for kw in ["if", "when", "how", "why", "what"] if kw in text_lower) >= 3
        ]

        if any(escalation_indicators):
            return "ESCALATION"

        # 4. ALLOW判定（安全で有用な内容）
        allow_indicators = [
            # 有用な情報提供
            any(kw in text_lower for kw in ["explain", "help", "guide", "teach", "inform"]),
            # 肯定的・建設的な表現
            any(kw in text_lower for kw in ["useful", "helpful", "beneficial", "good", "safe"]),
            # 教育・学習コンテンツ
            any(kw in text_lower for kw in ["learn", "understand", "knowledge", "education"]),
            # 適切な長さと構造
            50 <= text_length <= 2000,
            # 構造化された応答
            any(struct in text for struct in ["•", "-", "1.", "2.", "Step", "First"])
        ]

        if any(allow_indicators):
            return "ALLOW"

        # デフォルト判定
        # 過度に長い応答はESCALATION（複雑すぎる可能性）
        if text_length > 2000:
            return "ESCALATION"
        # 質問はESCALATION
        elif "?" in text or "？" in text:
            return "ESCALATION"
        # それ以外はALLOW（安全側に倒す）
        else:
            return "ALLOW"

    def balance_dataset(self, samples: List[Dict]) -> List[Dict]:
        """データセットのクラスバランスを調整"""
        if not getattr(self, "balance_classes", False):
            return samples
        
        # クラス別に振り分け
        class_samples = {"ALLOW": [], "ESCALATION": [], "DENY": [], "REFUSE": []}
        for sample in samples:
            label = sample.get("label", "ALLOW")
            if label in class_samples:
                class_samples[label].append(sample)
        
        # 各クラスのサンプル数
        class_counts = {k: len(v) for k, v in class_samples.items()}
        logger.info(f"Class distribution before balancing: {class_counts}")

        # 最小サンプル数に合わせる
        min_count = min(class_counts.values()) if class_counts.values() else 0
        if min_count == 0:
            logger.warning("Some classes have no samples, skipping balancing")
            return samples

        # 各クラスから最小サンプル数をランダムサンプリング
        balanced_samples = []
        for label, samples_list in class_samples.items():
            if len(samples_list) > min_count:
                balanced_samples.extend(random.sample(samples_list, min_count))
            else:
                balanced_samples.extend(samples_list)

        # シャッフル
        random.shuffle(balanced_samples)
        
        class_counts_after = Counter(s["label"] for s in balanced_samples)
        logger.info(f"Class distribution after balancing: {dict(class_counts_after)}")
        
        return balanced_samples
    
    def label_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        huggingface_mode: bool = False,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, int]:
        """データセットにラベル付け"""
        logger.info("="*80)
        logger.info("Four Class Labeling")
        logger.info("="*80)

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if huggingface_mode:
            return self._label_huggingface_datasets(input_dir, output_dir, test_size, val_size)
        else:
            return self._label_jsonl_files(input_dir, output_dir, test_size, val_size)

    def _label_jsonl_files(self, input_dir: Path, output_dir: Path, test_size: float, val_size: float) -> Dict[str, int]:
            
        # 入力ファイル検索
        input_files = list(input_dir.glob("*.jsonl"))
        json_files = list(input_dir.glob("*.json"))

        if not input_files and not json_files and input_dir.is_file():
            # 単一ファイルが指定された場吁E            if input_dir.suffix == '.jsonl':
                input_files = [input_dir]
            elif input_dir.suffix == '.json':
                json_files = [input_dir]

        if not input_files and not json_files:
            logger.error(f"No JSONL or JSON files found in {input_dir}")
            return {}

        all_files = input_files + json_files
        logger.info(f"Found {len(all_files)} input files ({len(input_files)} JSONL, {len(json_files)} JSON)")
        return self._process_files(all_files, output_dir, "JSON", test_size, val_size)

    def _label_huggingface_datasets(self, input_dir: Path, output_dir: Path, test_size: float, val_size: float) -> Dict[str, int]:
        """HuggingFaceデータセットからラベル付け"""
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return {}

        # チE�EタセチE��チE��レクトリを検索
        dataset_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not dataset_dirs:
            logger.error(f"No dataset directories found in {input_dir}")
            return {}

        logger.info(f"Found {len(dataset_dirs)} dataset directories")

        # 吁E��ータセチE��からファイルを収雁E        all_files = []
        for dataset_dir in dataset_dirs:
            # JSON/JSONLファイルを検索
            json_files = list(dataset_dir.glob("*.json")) + list(dataset_dir.glob("*.jsonl"))
            if json_files:
                all_files.extend(json_files)
                logger.info(f"  {dataset_dir.name}: {len(json_files)} files")

        if not all_files:
            logger.error("No JSON/JSONL files found in any dataset directory")
            return {}

        logger.info(f"Total files to process: {len(all_files)}")
        return self._process_files(all_files, output_dir, "HuggingFace", test_size, val_size)

    def _process_files(self, input_files: List[Path], output_dir: Path, source_type: str,
                      test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Dict[str, int]:
        """共通�Eファイル処琁E��ジチE��"""
        # 統訁E        stats = {
            "total": 0,
            "ALLOW": 0,
            "ESCALATION": 0,
            "DENY": 0,
            "REFUSE": 0
        }

        labeled_samples: List[Dict] = []

        # 全ファイルを処理
        for input_file in input_files:
            logger.info(f"Processing {input_file.name}...")

            try:
                if input_file.suffix == ".json":
                    # JSONファイルの場合
                    with open(input_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        for sample in tqdm(data, desc=f"Labeling {input_file.name}"):
                            self._process_sample(sample, stats, labeled_samples)
                    elif isinstance(data, dict):
                        # SO8Tデータセット形式の場合
                        if 'training_data' in data:
                            for sample in tqdm(data['training_data'], desc=f"Labeling {input_file.name}"):
                                self._process_sample(sample, stats, labeled_samples)
                        elif 'data' in data:
                            # 別のデータセット形式
                            for sample in tqdm(data['data'], desc=f"Labeling {input_file.name}"):
                                self._process_sample(sample, stats, labeled_samples)
                        else:
                            # 単一サンプルとして扱う
                            self._process_sample(data, stats, labeled_samples)

                elif input_file.suffix == ".jsonl":
                    # JSONLファイルの場合
                    with open(input_file, 'r', encoding='utf-8') as f:
                        for line in tqdm(f, desc=f"Labeling {input_file.name}"):
                            try:
                                sample = json.loads(line.strip())
                                self._process_sample(sample, stats, labeled_samples)
                            except json.JSONDecodeError:
                                logger.warning(f"JSON decode error in file {input_file.name}, skipping line.")
                                continue

            except Exception as e:
                logger.warning(f"Error processing file {input_file}: {e}")
                continue

        # クラスバランス調整
        if self.balance_classes:
            logger.info("Balancing classes...")
            labeled_samples = self.balance_dataset(labeled_samples)

        # データ分割 (train/val/test)
        total_split_size = test_size + val_size
        min_samples_for_split = 100  # 分割に必要な最小サンプル数

        if total_split_size > 0 and total_split_size < 1.0 and len(labeled_samples) >= min_samples_for_split:
            logger.info(f"PPOデータ分割を実行 (test_size={test_size}, val_size={val_size})...")

            # stratify用のラベルを取得
            labels = [s["label"] for s in labeled_samples]

            # 各クラスの最小サンプル数をチェック
            from collections import Counter
            label_counts = Counter(labels)
            min_class_count = min(label_counts.values())

            if min_class_count >= 2:
                # stratifyを使用可能
                use_stratify = True
                stratify_param = labels
            else:
                # stratifyを使用できない場合
                use_stratify = False
                stratify_param = None
                logger.warning(f"Some classes have too few samples for stratification (min: {min_class_count}), disabling stratify")

            if val_size > 0:
                # train/val/testに3分割
                # まずtrainと(temp = val + test)に分割
                train_samples, temp_samples = train_test_split(
                    labeled_samples,
                    test_size=test_size + val_size,
                    stratify=stratify_param,
                    random_state=random_state
                )

                # 次にtempをvalとtestに分割
                temp_labels = [s["label"] for s in temp_samples] if use_stratify else None
                val_ratio = val_size / (test_size + val_size)
                val_samples, test_samples = train_test_split(
                    temp_samples,
                    test_size=1 - val_ratio,  # test_sizeは残りのうちの割合
                    stratify=temp_labels,
                    random_state=random_state
                )
            else:
                # train/testに2分割
                train_samples, test_samples = train_test_split(
                    labeled_samples,
                    test_size=test_size,
                    stratify=stratify_param,
                    random_state=random_state
                )
                val_samples = []

            logger.info(f"Split results: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

            # 分割データを保存
            self._save_split_data(train_samples, output_dir, f"train_{source_type.lower()}.jsonl")
            if val_samples:
                self._save_split_data(val_samples, output_dir, f"val_{source_type.lower()}.jsonl")
            if test_samples:
                self._save_split_data(test_samples, output_dir, f"test_{source_type.lower()}.jsonl")
        else:
            if len(labeled_samples) < min_samples_for_split:
                logger.info(f"Dataset too small for splitting ({len(labeled_samples)} < {min_samples_for_split}), skipping split")
            # 分割なしで保存
            output_file = output_dir / f"labeled_four_class_dataset_{source_type.lower()}.jsonl"
            logger.info(f"Saving labeled dataset to {output_file}...")

            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in labeled_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 統計レポ�EチE        logger.info("="*80)
        logger.info("Labeling Statistics")
        logger.info("="*80)
        logger.info(f"Total samples: {stats['total']:,}")
        logger.info(f"ALLOW: {stats['ALLOW']:,}")
        logger.info(f"ESCALATION: {stats['ESCALATION']:,}")
        logger.info(f"DENY: {stats['DENY']:,}")
        logger.info(f"REFUSE: {stats['REFUSE']:,}")

        if test_size > 0 or val_size > 0:
            logger.info(f"Data split: test_size={test_size}, val_size={val_size}")
            logger.info("Output files:")
            logger.info(f"  - train_{source_type.lower()}.jsonl")
            if val_size > 0:
                logger.info(f"  - val_{source_type.lower()}.jsonl")
            if test_size > 0:
                logger.info(f"  - test_{source_type.lower()}.jsonl")
        else:
            logger.info(f"Output file: labeled_four_class_dataset_{source_type.lower()}.jsonl")

        logger.info("="*80)

        return stats

    def _save_split_data(self, samples: List[Dict], output_dir: Path, filename: str):
        """データセットを保存"""
        output_file = output_dir / filename
        logger.info(f"Saving {len(samples)} samples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    def _load_json_file(self, file_path: Path) -> List[Dict]:
        """JSONファイルを読み込み、サンプルリストに変換"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # JSONファイルの構造に応じてチE�Eタを抽出
            if isinstance(data, list):
                # 直接リスト形弁E                samples = data
            elif isinstance(data, dict) and 'training_data' in data:
                # SO8TチE�EタセチE��形弁E                samples = data['training_data']
            elif isinstance(data, dict) and 'data' in data:
                # 別のチE�EタセチE��形弁E                samples = data['data']
            else:
                # そ�E他�E構造は単一サンプルとして扱ぁE                samples = [data]

            logger.info(f"Loaded {len(samples)} samples from JSON file: {file_path}")
            return samples

        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return []

    def _process_sample(self, sample: Dict, stats: Dict[str, int], labeled_samples: List[Dict]):
        """個別のサンプルを処理"""
        stats["total"] += 1

        # テキスト抽出 - 複数のフィールドから優先順位で
        text = ""
        text_fields = ["chosen", "rejected", "text", "content", "instruction", "input", "output", "response", "prompt"]
        for field in text_fields:
            if field in sample and isinstance(sample[field], str) and sample[field].strip():
                text = sample[field].strip()
                break  # 最初に見つかったフィールドを使用

        if not text:
            return

        # 既存のラベルがある場合はそれを使用
        existing_label = sample.get("four_class_label", "").upper()
        if existing_label in ["ALLOW", "ESCALATION", "DENY", "REFUSE"]:
            label = existing_label
        else:
            # 新規ラベル付け
            label = self.classify_text(text)

        stats[label] += 1

        # ラベル付きサンプル
        labeled_sample = {
            **sample,
            "label": label,
            "original_text": text[:500]  # デバッグ用にテキストの一部を保存
        }
        labeled_samples.append(labeled_sample)


def main():
    """LLM PPOベストプラクティスに基づく四値分類メイン関数"""
    parser = argparse.ArgumentParser(
        description="LLM PPO用四値分類ラベリング (ALLOW/ESCALATION/DENY/REFUSE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LLM PPOベストプラクティスに基づく分類:
- ALLOW: 安全で有用な応答（学習に適したポジティブサンプル）
- ESCALATION: 専門判断が必要な複雑なクエリ
- DENY: 安全上問題のある内容（ネガティブサンプル）
- REFUSE: 明確に拒否すべき危険な内容

使用例:
  python script.py --input data/ --output labeled/ --test-size 0.2 --val-size 0.1
  python script.py --huggingface --input datasets/ --output output/
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="入力データセットのパス（JSON/JSONLファイルまたはディレクトリ）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="ラベル付きデータセットの出力ディレクトリ"
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="クラスバランス調整を無効化（PPO学習では有効推奨）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード（再現性のため、default: 42）"
    )
    parser.add_argument(
        "--huggingface",
        action="store_true",
        help="HuggingFaceデータセット形式を処理"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="テストセットの割合（PPO評価用、default: 0.2）"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="検証セットの割合（PPOチューニング用、default: 0.1）"
    )
    
    args = parser.parse_args()
    
    # シード設宁E    random.seed(args.seed)
    
    # ラベル付け実衁E    labeler = FourClassLabeler(balance_classes=not args.no_balance)
    
    try:
        stats = labeler.label_dataset(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            huggingface_mode=args.huggingface,
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        logger.info("[SUCCESS] Dataset labeling completed")
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] Dataset labeling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
