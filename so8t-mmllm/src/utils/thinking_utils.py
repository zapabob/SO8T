"""
Thinking関連ユーティリティ関数

Thinking/Final抽出、Safety判定、Verifierスコア計算、データ変換などの
ヘルパー関数を提供する。
"""

from typing import Optional, Dict, Any, Tuple, List
import hashlib
import json
from pathlib import Path
import sys

# 相対インポートエラー回避のため、絶対インポートを使用
# sys.pathにso8t-mmllm/srcが追加されていることを前提とする
try:
    from models.thinking_tokens import (
        extract_thinking_and_final,
        format_thinking_output,
        get_thinking_tokens,
    )
except ImportError:
    # フォールバック: 直接パスを指定
    from pathlib import Path
    import importlib.util
    _file_path = Path(__file__)
    _models_path = _file_path.parent.parent / "models" / "thinking_tokens.py"
    spec = importlib.util.spec_from_file_location("thinking_tokens", _models_path)
    thinking_tokens_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(thinking_tokens_module)
    extract_thinking_and_final = thinking_tokens_module.extract_thinking_and_final
    format_thinking_output = thinking_tokens_module.format_thinking_output
    get_thinking_tokens = thinking_tokens_module.get_thinking_tokens


def compute_text_hash(text: str) -> str:
    """
    テキストのハッシュ値を計算（監査ログ用）
    
    Args:
        text: ハッシュ化するテキスト
    
    Returns:
        SHA256ハッシュ値（16進数文字列）
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def extract_thinking_safely(
    text: str,
    use_redacted: bool = False,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    テキストからThinkingとFinalを安全に抽出
    
    Args:
        text: 抽出対象のテキスト
        use_redacted: Trueの場合、<think>形式を使用
    
    Returns:
        (thinking_text, final_text, full_text) のタプル
    """
    thinking, final = extract_thinking_and_final(text, use_redacted)
    
    # フォールバック: 抽出できない場合は全体を返す
    if thinking is None and final is None:
        return None, None, text
    
    return thinking, final, text


def validate_thinking_format(
    text: str,
    use_redacted: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Thinking形式が正しいか検証
    
    Args:
        text: 検証対象のテキスト
        use_redacted: Trueの場合、<think>形式を使用
    
    Returns:
        (is_valid, error_message) のタプル
    """
    tokens = get_thinking_tokens(use_redacted)
    
    think_start = tokens.get("think_start") or tokens.get("reasoning_start")
    think_end = tokens.get("think_end") or tokens.get("reasoning_end")
    final_start = tokens["final_start"]
    final_end = tokens["final_end"]
    
    # Thinkingタグのチェック
    has_think_start = think_start in text
    has_think_end = think_end in text
    
    if has_think_start and not has_think_end:
        return False, f"Missing closing tag: {think_end}"
    if has_think_end and not has_think_start:
        return False, f"Missing opening tag: {think_start}"
    
    # Finalタグのチェック
    has_final_start = final_start in text
    has_final_end = final_end in text
    
    if has_final_start and not has_final_end:
        return False, f"Missing closing tag: {final_end}"
    if has_final_end and not has_final_start:
        return False, f"Missing opening tag: {final_start}"
    
    # 順序のチェック
    if has_think_start and has_final_start:
        think_start_idx = text.find(think_start)
        final_start_idx = text.find(final_start)
        if final_start_idx < think_start_idx:
            return False, f"{final_start} must come after {think_start}"
    
    return True, None


def convert_cot_to_thinking_format(
    instruction: str,
    input_text: str,
    cot_output: str,
    use_redacted: bool = False,
) -> str:
    """
    CoT形式の出力をThinking形式に変換
    
    Args:
        instruction: 指示
        input_text: 入力
        cot_output: CoT形式の出力（段階的推論を含む）
        use_redacted: Trueの場合、<think>形式を使用
    
    Returns:
        Thinking形式の出力
    """
    # 簡易実装: CoT出力をThinkingとして扱い、最後の行をFinalとする
    lines = cot_output.strip().split('\n')
    
    # 最後の行をFinal、それ以外をThinkingとする
    if len(lines) > 1:
        thinking = '\n'.join(lines[:-1])
        final = lines[-1]
    else:
        thinking = cot_output
        final = cot_output.split('.')[0] + '.' if '.' in cot_output else cot_output
    
    return format_thinking_output(thinking, final, use_redacted)


def parse_safety_label(label: str) -> int:
    """
    安全ラベル文字列を数値に変換
    
    Args:
        label: 安全ラベル（"ALLOW", "ESCALATE", "REFUSE"）
    
    Returns:
        ラベルID（0=ALLOW, 1=ESCALATE, 2=REFUSE）
    """
    label_map = {
        "ALLOW": 0,
        "ESCALATE": 1,
        "REFUSE": 2,
    }
    label_upper = label.upper()
    if label_upper not in label_map:
        raise ValueError(f"Unknown safety label: {label}")
    return label_map[label_upper]


def parse_verifier_label(
    verifier_dict: Dict[str, float],
    default_logical: float = 1.0,
    default_faithful: float = 1.0,
) -> Tuple[float, float]:
    """
    Verifierラベル辞書を数値タプルに変換
    
    Args:
        verifier_dict: Verifierラベル辞書（{"logical": 1.0, "faithful": 1.0}等）
        default_logical: デフォルトの論理性スコア
        default_faithful: デフォルトの信頼度スコア
    
    Returns:
        (logical_score, faithful_score) のタプル
    """
    logical = verifier_dict.get("logical", default_logical)
    faithful = verifier_dict.get("faithful", default_faithful)
    return float(logical), float(faithful)


def load_thinking_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """
    Thinking形式データセットをロード
    
    Args:
        file_path: データセットファイルパス（JSONL形式）
    
    Returns:
        データセットのリスト
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            samples.append(sample)
    return samples


def save_thinking_dataset(
    samples: List[Dict[str, Any]],
    file_path: Path,
) -> None:
    """
    Thinking形式データセットを保存
    
    Args:
        samples: データセットのリスト
        file_path: 保存先ファイルパス（JSONL形式）
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

