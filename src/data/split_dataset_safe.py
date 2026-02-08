"""
安全なデータセット分割スクリプト

scikit-learnを使用して、ドメイン別に層化分割し、
情報リークを防止（同一URL/同一ソースからtrain/test両方に入らないよう制御）。
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from urllib.parse import urlparse
import sys

from sklearn.model_selection import train_test_split
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def extract_domain_key(sample: Dict[str, Any]) -> str:
    """
    サンプルからドメインキーを抽出（情報リーク防止用）
    
    Args:
        sample: サンプル辞書
    
    Returns:
        ドメインキー（URLまたはソース識別子）
    """
    # URLから抽出
    url = sample.get("url", "")
    if url:
        parsed = urlparse(url)
        # ドメイン + パスの最初の部分（同一記事を識別）
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) > 0:
            return f"{parsed.netloc}/{path_parts[0]}"
        return parsed.netloc
    
    # ソース識別子から抽出
    source = sample.get("source", sample.get("domain", ""))
    if source:
        return source
    
    # フォールバック: コンテンツのハッシュ（同一コンテンツを識別）
    content = sample.get("content", sample.get("text", ""))
    if content:
        return str(hash(content[:100]))  # 最初の100文字のハッシュ
    
    return "unknown"


def split_dataset_safe(
    input_file: Path,
    train_file: Path,
    val_file: Path,
    test_file: Optional[Path] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: str = "safety_label",
    prevent_leakage: bool = True,
    random_state: int = 42,
):
    """
    データセットを安全に分割
    
    Args:
        input_file: 入力データセット（JSONL）
        train_file: 訓練データセット出力ファイル
        val_file: 検証データセット出力ファイル
        test_file: テストデータセット出力ファイル（オプション）
        train_ratio: 訓練データの割合
        val_ratio: 検証データの割合
        test_ratio: テストデータの割合
        stratify_by: 層化の基準フィールド
        prevent_leakage: 情報リークを防止するか（同一ドメインキーを同じ側に入れる）
        random_state: 乱数シード
    """
    # データをロード
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    
    print(f"[INFO] Loaded {len(samples)} samples")
    
    if prevent_leakage:
        # ドメインキーでグループ化
        domain_groups: Dict[str, List[int]] = {}
        for idx, sample in enumerate(samples):
            domain_key = extract_domain_key(sample)
            if domain_key not in domain_groups:
                domain_groups[domain_key] = []
            domain_groups[domain_key].append(idx)
        
        print(f"[INFO] Grouped into {len(domain_groups)} domain groups")
        
        # グループ単位で分割
        group_keys = list(domain_groups.keys())
        
        # 層化のためのラベルを取得（各グループの代表サンプルのラベルを使用）
        group_labels = []
        for key in group_keys:
            group_indices = domain_groups[key]
            # グループの最初のサンプルのラベルを使用
            sample = samples[group_indices[0]]
            label = sample.get(stratify_by, "unknown")
            group_labels.append(label)
        
        # グループを分割
        if test_file:
            # train / val / test に分割
            train_groups, temp_groups = train_test_split(
                group_keys,
                test_size=(val_ratio + test_ratio),
                stratify=group_labels if stratify_by else None,
                random_state=random_state,
            )
            
            val_groups, test_groups = train_test_split(
                temp_groups,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=[group_labels[group_keys.index(k)] for k in temp_groups] if stratify_by else None,
                random_state=random_state,
            )
            
            train_indices = [idx for key in train_groups for idx in domain_groups[key]]
            val_indices = [idx for key in val_groups for idx in domain_groups[key]]
            test_indices = [idx for key in test_groups for idx in domain_groups[key]]
        else:
            # train / val に分割
            train_groups, val_groups = train_test_split(
                group_keys,
                test_size=val_ratio,
                stratify=group_labels if stratify_by else None,
                random_state=random_state,
            )
            
            train_indices = [idx for key in train_groups for idx in domain_groups[key]]
            val_indices = [idx for key in val_groups for idx in domain_groups[key]]
            test_indices = []
    else:
        # 通常の分割（情報リーク防止なし）
        labels = [s.get(stratify_by, "unknown") for s in samples] if stratify_by else None
        
        if test_file:
            train_indices, temp_indices = train_test_split(
                range(len(samples)),
                test_size=(val_ratio + test_ratio),
                stratify=labels,
                random_state=random_state,
            )
            
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=[labels[i] for i in temp_indices] if labels else None,
                random_state=random_state,
            )
        else:
            train_indices, val_indices = train_test_split(
                range(len(samples)),
                test_size=val_ratio,
                stratify=labels,
                random_state=random_state,
            )
            test_indices = []
    
    # データセットを保存
    train_file.parent.mkdir(parents=True, exist_ok=True)
    with open(train_file, 'w', encoding='utf-8') as f:
        for idx in train_indices:
            f.write(json.dumps(samples[idx], ensure_ascii=False) + '\n')
    
    val_file.parent.mkdir(parents=True, exist_ok=True)
    with open(val_file, 'w', encoding='utf-8') as f:
        for idx in val_indices:
            f.write(json.dumps(samples[idx], ensure_ascii=False) + '\n')
    
    if test_file and test_indices:
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            for idx in test_indices:
                f.write(json.dumps(samples[idx], ensure_ascii=False) + '\n')
    
    print(f"[SUCCESS] Split completed:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    if test_indices:
        print(f"  Test: {len(test_indices)} samples")


def main():
    parser = argparse.ArgumentParser(description="Safe dataset splitting")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset file (JSONL)",
    )
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Train dataset output file",
    )
    parser.add_argument(
        "--val",
        type=Path,
        required=True,
        help="Validation dataset output file",
    )
    parser.add_argument(
        "--test",
        type=Path,
        help="Test dataset output file (optional)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="safety_label",
        help="Field to stratify by",
    )
    parser.add_argument(
        "--no-leakage-prevention",
        action="store_true",
        help="Disable leakage prevention",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state",
    )
    
    args = parser.parse_args()
    
    split_dataset_safe(
        input_file=args.input,
        train_file=args.train,
        val_file=args.val,
        test_file=args.test,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_by=args.stratify_by,
        prevent_leakage=not args.no_leakage_prevention,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

