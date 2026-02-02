"""
Thinking特殊トークン定義とトークナイザー拡張

<think>, </think>, <final>, </final> の特殊トークンを定義し、
トークナイザーへの追加機能を提供する。
"""

from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


# Thinking特殊トークン定義
THINKING_SPECIAL_TOKENS = {
    "think_start": "<think>",
    "think_end": "</think>",
    "final_start": "<final>",
    "final_end": "</final>",
}

# ユーザーが提示した形式（<think>）もサポート
REDACTED_REASONING_TOKENS = {
    "reasoning_start": "<think>",
    "reasoning_end": "</think>",
    "final_start": "<final>",
    "final_end": "</final>",
}

# 四重推論形式（Task/Safety/Policy/Final）
QUADRUPLE_THINKING_TOKENS = {
    "think_task_start": "<think-task>",
    "think_task_end": "</think-task>",
    "think_safety_start": "<think-safety>",
    "think_safety_end": "</think-safety>",
    "think_policy_start": "<think-policy>",
    "think_policy_end": "</think-policy>",
    "final_start": "<final>",
    "final_end": "</final>",
}

# デフォルトは<think>形式を使用
DEFAULT_SPECIAL_TOKENS = THINKING_SPECIAL_TOKENS


def get_thinking_tokens(use_redacted: bool = False, use_quadruple: bool = False) -> Dict[str, str]:
    """
    Thinking特殊トークンの辞書を取得
    
    Args:
        use_redacted: Trueの場合、<think>形式を使用
        use_quadruple: Trueの場合、四重推論形式（Task/Safety/Policy/Final）を使用
    
    Returns:
        特殊トークンの辞書
    """
    if use_quadruple:
        return QUADRUPLE_THINKING_TOKENS
    if use_redacted:
        return REDACTED_REASONING_TOKENS
    return DEFAULT_SPECIAL_TOKENS


def add_thinking_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    use_redacted: bool = False,
    use_quadruple: bool = False,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    トークナイザーにThinking特殊トークンを追加
    
    Args:
        tokenizer: 追加対象のトークナイザー
        use_redacted: Trueの場合、<think>形式を使用
        use_quadruple: Trueの場合、四重推論形式を使用
    
    Returns:
        特殊トークンが追加されたトークナイザー
    """
    tokens = get_thinking_tokens(use_redacted, use_quadruple)
    
    # 特殊トークンのリストを作成
    special_tokens_list = list(tokens.values())
    
    # 既存の特殊トークンとマージ
    existing_special_tokens = tokenizer.special_tokens_map
    additional_tokens = []
    
    for token in special_tokens_list:
        if token not in existing_special_tokens.values():
            if token not in additional_tokens:
                additional_tokens.append(token)
    
    if additional_tokens:
        # 特殊トークンを追加
        tokenizer.add_special_tokens({
            "additional_special_tokens": additional_tokens
        })
    
    return tokenizer


def get_token_ids(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    use_redacted: bool = False,
) -> Dict[str, int]:
    """
    Thinking特殊トークンのIDを取得
    
    Args:
        tokenizer: トークナイザー
        use_redacted: Trueの場合、<think>形式を使用
    
    Returns:
        トークン名とIDのマッピング
    """
    tokens = get_thinking_tokens(use_redacted)
    token_ids = {}
    
    for key, token_str in tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id == tokenizer.unk_token_id:
            # 特殊トークンが追加されていない場合
            raise ValueError(
                f"Token '{token_str}' not found in tokenizer. "
                "Call add_thinking_tokens_to_tokenizer() first."
            )
        token_ids[key] = token_id
    
    return token_ids


def extract_quadruple_thinking(
    text: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    テキストから四重推論（Task/Safety/Policy/Final）を抽出
    
    Args:
        text: 抽出対象のテキスト
    
    Returns:
        (task_text, safety_text, policy_text, final_text) のタプル
    """
    tokens = QUADRUPLE_THINKING_TOKENS
    
    # Task部分の抽出
    task_text = None
    if tokens["think_task_start"] in text and tokens["think_task_end"] in text:
        start_idx = text.find(tokens["think_task_start"])
        end_idx = text.find(tokens["think_task_end"])
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            task_text = text[
                start_idx + len(tokens["think_task_start"]):end_idx
            ].strip()
    
    # Safety部分の抽出
    safety_text = None
    if tokens["think_safety_start"] in text and tokens["think_safety_end"] in text:
        start_idx = text.find(tokens["think_safety_start"])
        end_idx = text.find(tokens["think_safety_end"])
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            safety_text = text[
                start_idx + len(tokens["think_safety_start"]):end_idx
            ].strip()
    
    # Policy部分の抽出
    policy_text = None
    if tokens["think_policy_start"] in text and tokens["think_policy_end"] in text:
        start_idx = text.find(tokens["think_policy_start"])
        end_idx = text.find(tokens["think_policy_end"])
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            policy_text = text[
                start_idx + len(tokens["think_policy_start"]):end_idx
            ].strip()
    
    # Final部分の抽出
    final_text = None
    if tokens["final_start"] in text and tokens["final_end"] in text:
        start_idx = text.find(tokens["final_start"])
        end_idx = text.find(tokens["final_end"])
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            final_text = text[
                start_idx + len(tokens["final_start"]):end_idx
            ].strip()
    
    return task_text, safety_text, policy_text, final_text


def extract_thinking_and_final(
    text: str,
    use_redacted: bool = False,
    use_quadruple: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    テキストからThinking部分とFinal部分を抽出
    
    Args:
        text: 抽出対象のテキスト
        use_redacted: Trueの場合、<think>形式を使用
        use_quadruple: Trueの場合、四重推論形式を使用
    
    Returns:
        (thinking_text, final_text) のタプル
    """
    if use_quadruple:
        task, safety, policy, final = extract_quadruple_thinking(text)
        # 四重推論の場合は、すべてを結合してthinking_textとする
        thinking_parts = [p for p in [task, safety, policy] if p]
        thinking_text = "\n".join(thinking_parts) if thinking_parts else None
        return thinking_text, final
    
    tokens = get_thinking_tokens(use_redacted, use_quadruple)
    
    # Thinking部分の抽出
    think_start_tag = tokens.get("think_start") or tokens.get("reasoning_start")
    think_end_tag = tokens.get("think_end") or tokens.get("reasoning_end")
    
    thinking_text = None
    if think_start_tag in text and think_end_tag in text:
        start_idx = text.find(think_start_tag)
        end_idx = text.find(think_end_tag)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            thinking_text = text[
                start_idx + len(think_start_tag):end_idx
            ].strip()
    
    # Final部分の抽出
    final_start_tag = tokens["final_start"]
    final_end_tag = tokens["final_end"]
    
    final_text = None
    if final_start_tag in text and final_end_tag in text:
        start_idx = text.find(final_start_tag)
        end_idx = text.find(final_end_tag)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            final_text = text[
                start_idx + len(final_start_tag):end_idx
            ].strip()
    
    return thinking_text, final_text


def format_quadruple_thinking_output(
    task: str,
    safety: str,
    policy: str,
    final: str,
) -> str:
    """
    四重推論を特殊トークンで囲んだ形式にフォーマット
    
    Args:
        task: Task推論テキスト
        safety: Safety推論テキスト
        policy: Policy推論テキスト
        final: 最終回答テキスト
    
    Returns:
        フォーマット済みテキスト
    """
    tokens = QUADRUPLE_THINKING_TOKENS
    return (
        f"{tokens['think_task_start']}{task}{tokens['think_task_end']}"
        f"{tokens['think_safety_start']}{safety}{tokens['think_safety_end']}"
        f"{tokens['think_policy_start']}{policy}{tokens['think_policy_end']}"
        f"{tokens['final_start']}{final}{tokens['final_end']}"
    )


def format_thinking_output(
    thinking: str,
    final: str,
    use_redacted: bool = False,
    use_quadruple: bool = False,
    task: Optional[str] = None,
    safety: Optional[str] = None,
    policy: Optional[str] = None,
) -> str:
    """
    ThinkingとFinalを特殊トークンで囲んだ形式にフォーマット
    
    Args:
        thinking: 内部推論テキスト（四重推論でない場合）
        final: 最終回答テキスト
        use_redacted: Trueの場合、<think>形式を使用
        use_quadruple: Trueの場合、四重推論形式を使用
        task: Task推論テキスト（四重推論の場合）
        safety: Safety推論テキスト（四重推論の場合）
        policy: Policy推論テキスト（四重推論の場合）
    
    Returns:
        フォーマット済みテキスト
    """
    if use_quadruple and task and safety and policy:
        return format_quadruple_thinking_output(task, safety, policy, final)
    
    tokens = get_thinking_tokens(use_redacted, use_quadruple)
    
    think_start_tag = tokens.get("think_start") or tokens.get("reasoning_start")
    think_end_tag = tokens.get("think_end") or tokens.get("reasoning_end")
    final_start_tag = tokens["final_start"]
    final_end_tag = tokens["final_end"]
    
    return (
        f"{think_start_tag}{thinking}{think_end_tag}"
        f"{final_start_tag}{final}{final_end_tag}"
    )


def build_quadruple_thinking_prompt(user_query: str) -> str:
    """
    四重推論生成用のプロンプトを構築
    
    Args:
        user_query: ユーザークエリ
    
    Returns:
        プロンプトテキスト
    """
    prompt = (
        "以下の問題に対して、四段階の内部推論を行い、その後<final>で日本語で回答してください。\n"
        "1. <think-task>: タスク推論（英語で、ドメイン知識・翻訳方針・要約方針を考える）\n"
        "2. <think-safety>: 安全性推論（英語で、安全性・法令順守・NSFW違反を評価）\n"
        "3. <think-policy>: ポリシー推論（英語で、軍事・医療・インフラ等の領域別ポリシーに沿って、出せる/出せない情報範囲を決める）\n"
        "4. <final>: 最終回答（日本語で、制約を反映した最終回答のみ出力）\n"
        "内部推論はユーザーに公開されないことを前提に、正確に考えてください。\n"
        "問題: {query}\n"
        "答え: <think-task>"
    ).format(query=user_query)
    
    return prompt


def build_thinking_prompt(
    user_query: str,
    use_redacted: bool = False,
    use_quadruple: bool = False,
) -> str:
    """
    Thinking生成用のプロンプトを構築
    
    Args:
        user_query: ユーザークエリ
        use_redacted: Trueの場合、<think>形式を使用
        use_quadruple: Trueの場合、四重推論形式を使用
    
    Returns:
        プロンプトテキスト
    """
    if use_quadruple:
        return build_quadruple_thinking_prompt(user_query)
    
    tokens = get_thinking_tokens(use_redacted, use_quadruple)
    think_start_tag = tokens.get("think_start") or tokens.get("reasoning_start")
    
    prompt = (
        "以下の問題に対して、まず{think_tag}で内部推論を書き、"
        "その後<final>で短く回答してください。\n"
        "内部推論はユーザーに公開されないことを前提に、正確に考えてください。\n"
        "問題: {query}\n"
        "答え: {think_tag}"
    ).format(
        think_tag=think_start_tag,
        query=user_query
    )
    
    return prompt

