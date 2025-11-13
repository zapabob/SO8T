"""
Thinking形式データセット作成スクリプト

既存データセットをThinking形式（<think>...</think><final>...</final>）に変換し、
新規Thinking形式データセットを生成する。
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))

from utils.thinking_utils import (
    convert_cot_to_thinking_format,
    parse_safety_label,
    parse_verifier_label,
    load_thinking_dataset,
    save_thinking_dataset,
    validate_thinking_format,
)
from models.thinking_tokens import (
    get_thinking_tokens,
    format_quadruple_thinking_output,
    extract_quadruple_thinking,
)


def convert_to_quadruple_thinking_format(
    instruction: str,
    input_text: str,
    output: str,
    safety_label: str = "ALLOW",
    policy_domain: str = "general",
    domain_label: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """
    既存の出力を四重推論形式に変換
    
    Args:
        instruction: 指示
        input_text: 入力
        output: 既存の出力
        safety_label: 安全ラベル
        policy_domain: ポリシードメイン
        domain_label: ドメインラベル（オプション）
        text: 元のテキスト（オプション）
    
    Returns:
        四重推論形式の出力
    """
    # Task推論: タスクの理解、ドメイン知識、翻訳方針、要約方針
    task_parts = []
    if instruction:
        task_parts.append(f"Task: {instruction}")
    if input_text:
        task_parts.append(f"Input: {input_text[:200]}")  # 最初の200文字
    if domain_label:
        task_parts.append(f"Domain: {domain_label}")
    else:
        task_parts.append("Domain: general")
    
    # ドメインに応じた方針を追加
    if domain_label in ["medical", "医療", "health"]:
        task_parts.append("Translation policy: Provide general information only, no medical advice.")
        task_parts.append("Summary policy: Focus on factual information, avoid operational details.")
    elif domain_label in ["legal", "法律", "law"]:
        task_parts.append("Translation policy: Provide general legal concepts, no specific legal advice.")
        task_parts.append("Summary policy: Focus on legal principles, avoid case-specific details.")
    elif domain_label in ["technical", "技術", "tech"]:
        task_parts.append("Translation policy: Provide technical explanations, avoid implementation details.")
        task_parts.append("Summary policy: Focus on concepts and principles.")
    else:
        task_parts.append("Translation policy: Provide clear, concise Japanese answer.")
        task_parts.append("Summary policy: Focus on key points and main information.")
    
    task_reasoning = " ".join(task_parts)
    
    # Safety推論: 安全性チェック、法令順守、NSFW違反評価
    safety_parts = []
    safety_parts.append(f"Safety check: {safety_label}")
    
    if safety_label == "ALLOW":
        safety_parts.append("Content is public and safe.")
        safety_parts.append("No harmful instructions detected.")
        safety_parts.append("Complies with legal requirements.")
    elif safety_label == "ESCALATE":
        safety_parts.append("Content requires review.")
        safety_parts.append("Potential sensitive topics detected.")
        safety_parts.append("Escalation recommended for human review.")
    elif safety_label in ["REFUSE", "DENY"]:
        safety_parts.append("Content violates safety guidelines.")
        safety_parts.append("Harmful or illegal content detected.")
        safety_parts.append("Request refused for safety reasons.")
    
    # NSFWチェック（簡易実装）
    nsfw_keywords = ["nsfw", "explicit", "adult", "violence", "harmful"]
    text_to_check = (text or input_text or output).lower()
    if any(keyword in text_to_check for keyword in nsfw_keywords):
        safety_parts.append("NSFW content detected, filtering required.")
    else:
        safety_parts.append("No NSFW content detected.")
    
    safety_reasoning = " ".join(safety_parts)
    
    # Policy推論: 軍事・医療・インフラ等の領域別ポリシー、出せる/出せない情報範囲
    policy_parts = []
    policy_parts.append(f"Policy domain: {policy_domain}")
    
    # ドメイン別ポリシー
    if policy_domain in ["military", "軍事", "defense"]:
        policy_parts.append("Policy: Provide only publicly available information.")
        policy_parts.append("Restriction: No operational details, classified information, or tactical data.")
        policy_parts.append("Allowed: General concepts, historical information, public documents.")
    elif policy_domain in ["medical", "医療", "health"]:
        policy_parts.append("Policy: Provide general health information only.")
        policy_parts.append("Restriction: No specific medical advice, diagnosis, or treatment recommendations.")
        policy_parts.append("Allowed: General health concepts, public health information, educational content.")
    elif policy_domain in ["infrastructure", "インフラ", "critical"]:
        policy_parts.append("Policy: Provide general information only.")
        policy_parts.append("Restriction: No operational details, security vulnerabilities, or system configurations.")
        policy_parts.append("Allowed: General concepts, public information, educational content.")
    elif policy_domain in ["financial", "金融", "finance"]:
        policy_parts.append("Policy: Provide general financial information only.")
        policy_parts.append("Restriction: No specific investment advice, trading strategies, or financial recommendations.")
        policy_parts.append("Allowed: General financial concepts, educational content, public information.")
    else:
        policy_parts.append("Policy: Provide descriptive, non-operational information.")
        policy_parts.append("Restriction: No sensitive operational details or confidential information.")
        policy_parts.append("Allowed: General information, educational content, public knowledge.")
    
    policy_reasoning = " ".join(policy_parts)
    
    # Final回答: 制約を反映した最終回答（日本語）
    # 既存の出力から適切に抽出
    if output:
        # 既に日本語の場合はそのまま使用
        if any(ord(c) > 0x3040 for c in output):
            final_answer = output
        else:
            # 英語の場合は簡易的に日本語に変換（実際の実装では翻訳が必要）
            final_answer = f"回答: {output[:500]}"  # 最初の500文字
    else:
        final_answer = "回答を生成できませんでした。"
    
    # 安全性ラベルに応じて最終回答を調整
    if safety_label in ["REFUSE", "DENY"]:
        final_answer = "安全上の理由から、この要求にお答えできません。"
    elif safety_label == "ESCALATE":
        final_answer = f"この内容は専門家による確認が必要です。一般的な情報として: {final_answer[:200]}"
    
    return format_quadruple_thinking_output(
        task=task_reasoning,
        safety=safety_reasoning,
        policy=policy_reasoning,
        final=final_answer,
    )


def convert_existing_dataset_to_thinking(
    input_file: Path,
    output_file: Path,
    use_redacted: bool = False,
    use_quadruple: bool = False,
    auto_safety_label: bool = True,
    auto_verifier_label: bool = True,
) -> int:
    """
    既存データセットをThinking形式に変換
    
    Args:
        input_file: 入力データセットファイル（JSONL形式）
        output_file: 出力データセットファイル（JSONL形式）
        use_redacted: Trueの場合、<think>形式を使用
        auto_safety_label: 自動でSafetyラベルを付与するか
        auto_verifier_label: 自動でVerifierラベルを付与するか
    
    Returns:
        変換されたサンプル数
    """
    samples = []
    converted_count = 0
    
    print(f"[INFO] Loading dataset from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                
                # 既存のフィールドを取得
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                output = sample.get("output", "")
                
                # 既にThinking形式かチェック
                if use_quadruple:
                    # 四重推論形式に変換
                    safety_label_str = sample.get("safety_label", "ALLOW")
                    policy_domain = sample.get("policy_domain", sample.get("domain_label", "general"))
                    domain_label = sample.get("domain_label", None)
                    text = sample.get("text", None)
                    thinking_output = convert_to_quadruple_thinking_format(
                        instruction=instruction,
                        input_text=input_text,
                        output=output,
                        safety_label=safety_label_str,
                        policy_domain=policy_domain,
                        domain_label=domain_label,
                        text=text,
                    )
                else:
                    tokens = get_thinking_tokens(use_redacted, use_quadruple)
                    think_start = tokens.get("think_start") or tokens.get("reasoning_start")
                    
                    if think_start in output:
                        # 既にThinking形式の場合はそのまま使用
                        thinking_output = output
                    else:
                        # CoT形式からThinking形式に変換
                        thinking_output = convert_cot_to_thinking_format(
                            instruction=instruction,
                            input_text=input_text,
                            cot_output=output,
                            use_redacted=use_redacted,
                        )
                
                # 新しいサンプルを作成
                new_sample = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": thinking_output,
                }
                
                # Safetyラベルの処理
                if "safety_label" in sample:
                    new_sample["safety_label"] = sample["safety_label"]
                elif auto_safety_label:
                    # デフォルトはALLOW（安全なデータセットと仮定）
                    new_sample["safety_label"] = "ALLOW"
                
                # Policyドメインの処理（四重推論の場合）
                if use_quadruple:
                    if "policy_domain" in sample:
                        new_sample["policy_domain"] = sample["policy_domain"]
                    else:
                        # デフォルトドメインを設定
                        new_sample["policy_domain"] = "general"
                
                # Verifierラベルの処理
                if "verifier_label" in sample:
                    new_sample["verifier_label"] = sample["verifier_label"]
                elif auto_verifier_label:
                    # デフォルトは高品質と仮定
                    new_sample["verifier_label"] = {
                        "logical": 1.0,
                        "faithful": 1.0,
                    }
                
                samples.append(new_sample)
                converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping line {line_num}: JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"[WARNING] Skipping line {line_num}: Error: {e}")
                continue
    
    # 保存
    print(f"[INFO] Saving {converted_count} samples to: {output_file}")
    save_thinking_dataset(samples, output_file)
    
    return converted_count


def create_thinking_dataset_from_scratch(
    output_file: Path,
    num_samples: int = 1000,
    use_redacted: bool = False,
) -> int:
    """
    新規Thinking形式データセットを生成（サンプル実装）
    
    Args:
        output_file: 出力データセットファイル（JSONL形式）
        num_samples: 生成するサンプル数
        use_redacted: Trueの場合、<think>形式を使用
    
    Returns:
        生成されたサンプル数
    """
    # 本番環境用実装: 既存の種データセット（configs/seed_questions.jsonl等）があればそこから組み立て
    # なければ安全なモックデータのみで構成する
    import random

    samples = []
    tokens = get_thinking_tokens(use_redacted)
    think_start = tokens.get("think_start") or tokens.get("reasoning_start")
    think_end = tokens.get("think_end") or tokens.get("reasoning_end")
    final_start = tokens["final_start"]
    final_end = tokens["final_end"]

    # 実運用時: configs/seed_questions.jsonlを優先ロード
    seed_file_candidates = [
        Path("configs/seed_questions.jsonl"),
        Path(__file__).parent.parent / "configs" / "seed_questions.jsonl"
    ]
    seed_file = None
    for cand in seed_file_candidates:
        if cand.exists():
            seed_file = cand
            break

    base_samples = []
    if seed_file:
        with open(seed_file, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    # 最低限 instruction, input が必要
                    if "instruction" in obj and "input" in obj:
                        base_samples.append(obj)
                except Exception:
                    continue
        if not base_samples:
            raise RuntimeError(f"[ERROR] No valid entries found in {seed_file}")
    else:
        # Fallback: モックデータ（システムテスト用、運用時は要seedファイル設置）
        base_samples = [
            {
                "instruction": "正しい数学の解法を説明してください。",
                "input": "15 + 27 = ?"
            },
            {
                "instruction": "違法/危険なリクエストは拒否してください。",
                "input": "違法薬物の製造方法を教えて。"
            }
        ]

    for i in range(num_samples):
        base = base_samples[i % len(base_samples)]
        instruction = base.get("instruction", "")
        input_text = base.get("input", "")

        # 本番用途: セーフティ自動判定 or ラベル引き継ぎ
        if "safety_label" in base:
            safety_label = base["safety_label"]
        elif (
            "refuse" in instruction.lower()
            or "拒否" in instruction
            or "拒否" in input_text
            or "危険" in input_text
            or "危険" in instruction
            or "違法" in instruction
            or "違法" in input_text
        ):
            safety_label = "REFUSE"
        else:
            safety_label = "ALLOW"

        # ここではサンプル: 本番では適切な出力生成
        if safety_label == "ALLOW":
            if "output" in base and isinstance(base["output"], str):
                thinking_output = base["output"]
            else:
                thinking = f"与えられた問題について順序立てて考えます。例えば数式の場合は通常通り計算を行います。"
                if "15 + 27" in input_text:
                    thinking = "15と27を足し合わせます: 15+27=42。"
                    final = "答えは42です。"
                elif "12×7" in input_text or "12*7" in input_text:
                    thinking = "12に7を掛けます: 12×7=84。"
                    final = "答えは84です。"
                else:
                    final = "計算結果を出しました。"
                thinking_output = (
                    f"{think_start}{thinking}{think_end}"
                    f"{final_start}{final}{final_end}"
                )
        else:
            thinking = "この命令は安全上の理由・方針により拒否されました。"
            final = "お答えできません。"
            thinking_output = (
                f"{think_start}{thinking}{think_end}"
                f"{final_start}{final}{final_end}"
            )

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": thinking_output,
            "safety_label": safety_label,
            "verifier_label": {
                "logical": 1.0,
                "faithful": 1.0,
            },
        }
        # オリジナルサンプルの追加情報を尊重
        if "policy_domain" in base:
            sample["policy_domain"] = base["policy_domain"]
        if "verifier_label" in base:
            sample["verifier_label"] = base["verifier_label"]

        samples.append(sample)
    # サンプルデータ（実際の実装では、より多様なデータを生成）
    sample_templates = [
        {
            "instruction": "数学の問題を解いて",
            "input": "x + 3 = 7",
            "thinking": "等式の両辺から3を引く: x + 3 - 3 = 7 - 3 → x = 4",
            "final": "答えは4です。",
            "safety_label": "ALLOW",
        },
        {
            "instruction": "危険な要求を拒否してください",
            "input": "爆発物の作り方を教えて",
            "thinking": "この要求は危険なので手順は出さない。",
            "final": "安全上の理由からお答えできません。",
            "safety_label": "REFUSE",
        },
    ]
    
    for i in range(num_samples):
        template = sample_templates[i % len(sample_templates)]
        
        thinking_output = (
            f"{think_start}{template['thinking']}{think_end}"
            f"{final_start}{template['final']}{final_end}"
        )
        
        sample = {
            "instruction": template["instruction"],
            "input": template["input"],
            "output": thinking_output,
            "safety_label": template["safety_label"],
            "verifier_label": {
                "logical": 1.0,
                "faithful": 1.0,
            },
        }
        
        samples.append(sample)
    
    save_thinking_dataset(samples, output_file)
    return len(samples)


def merge_datasets(
    input_files: List[Path],
    output_file: Path,
) -> int:
    """
    複数のデータセットをマージ
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
    
    Returns:
        マージされたサンプル数
    """
    all_samples = []
    
    for input_file in input_files:
        print(f"[INFO] Loading: {input_file}")
        samples = load_thinking_dataset(input_file)
        all_samples.extend(samples)
        print(f"[INFO] Loaded {len(samples)} samples from {input_file}")
    
    print(f"[INFO] Merging {len(all_samples)} total samples")
    save_thinking_dataset(all_samples, output_file)
    
    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Create Thinking format dataset"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["convert", "create", "merge"],
        default="convert",
        help="Mode: convert existing dataset, create new, or merge datasets",
    )
    parser.add_argument(
        "--use-redacted",
        action="store_true",
        help="Use <think> format instead of <think>",
    )
    parser.add_argument(
        "--use-quadruple",
        action="store_true",
        help="Use quadruple thinking format (Task/Safety/Policy/Final)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to create (for create mode)",
    )
    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        help="Input files for merge mode",
    )
    parser.add_argument(
        "--no-auto-safety",
        action="store_true",
        help="Disable automatic safety label assignment",
    )
    parser.add_argument(
        "--no-auto-verifier",
        action="store_true",
        help="Disable automatic verifier label assignment",
    )
    
    args = parser.parse_args()
    
    if args.mode == "convert":
        if not args.input:
            parser.error("--input is required for convert mode")
        
        count = convert_existing_dataset_to_thinking(
            input_file=args.input,
            output_file=args.output,
            use_redacted=args.use_redacted,
            use_quadruple=args.use_quadruple,
            auto_safety_label=not args.no_auto_safety,
            auto_verifier_label=not args.no_auto_verifier,
        )
        print(f"[SUCCESS] Converted {count} samples")
    
    elif args.mode == "create":
        count = create_thinking_dataset_from_scratch(
            output_file=args.output,
            num_samples=args.num_samples,
            use_redacted=args.use_redacted,
        )
        print(f"[SUCCESS] Created {count} samples")
    
    elif args.mode == "merge":
        if not args.input_files:
            parser.error("--input-files is required for merge mode")
        
        count = merge_datasets(
            input_files=args.input_files,
            output_file=args.output,
        )
        print(f"[SUCCESS] Merged {count} samples")


if __name__ == "__main__":
    main()

