# AEGIS v2.5 評価条件標準化ドキュメント

## 概要

AEGIS v2.5の評価条件を標準化し、v2.4との差異を明確化することで、性能比較の信頼性を確保するドキュメントです。

## 1. 評価条件の変遷分析

### v2.4 vs v2.5 の主な変更点

#### GSM8K (数学的推論)
- **v2.4**: 98.2% (8-shot CoT)
- **v2.5**: 77.0% (8-shot CoT)
- **変化**: -21.2pt (-21.6%)
- **推定原因**:
  - 答え抽出ロジックの厳格化
  - few-shotプロンプトの変更
  - 評価スクリプトの更新

#### MATH (競技数学)
- **v2.4**: 32.1% (0-shot CoT)
- **v2.5**: 43.0% (0-shot CoT)
- **変化**: +10.9pt (+34.0%)
- **推定原因**:
  - GRPOの効果
  - SO8T四重推論の改善
  - 数式的証明支援の強化

#### ARC-Challenge (科学的推論)
- **v2.4**: 45.3% (10-shot)
- **v2.5**: 74.0% (10-shot)
- **変化**: +28.7pt (+63.4%)
- **推定原因**:
  - A/B/C/D形式の強制抽出実装
  - mHC多様体アーキテクチャの効果
  - 評価プロンプトの改善

#### ELYZA Tasks 100 (日本語理解)
- **v2.4**: 85.4% (4-5スケール)
- **v2.5**: 83.0% (4-5スケール)
- **変化**: -2.4pt (-2.8%)
- **推定原因**:
  - 安定した性能維持
  - SO8T日本語適応の効果

## 2. 標準化された評価プロトコル

### 共通設定
```python
# 全ベンチマーク共通の設定
EVALUATION_CONFIG = {
    "seeds": [42, 123, 456, 789, 999],  # 5-seed統計
    "temperature": 0.1,                  # 決定論的生成
    "max_tokens": {
        "gsm8k": 512,
        "math": 1024,
        "arc": 256,
        "elyza": 256
    },
    "do_sample": False,                   # 決定論的サンプリング
    "num_return_sequences": 1
}
```

### GSM8K評価プロトコル
```python
def evaluate_gsm8k_standard(model, tokenizer, questions):
    """標準化されたGSM8K評価"""

    for question in questions:
        # 標準プロンプト形式
        prompt = f"""Solve this math problem step by step, showing your work:

{question}

Step-by-step solution:"""

        inputs = tokenizer(prompt, return_tensors="pt")

        # 標準生成設定
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.1,
            do_sample=False,
            num_return_sequences=1
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 標準答え抽出ロジック
        final_answer = extract_final_answer_standard(response)

        # 正解判定
        is_correct = check_answer_standard(final_answer, question)
```

### MATH評価プロトコル
```python
def evaluate_math_standard(model, tokenizer, problems):
    """標準化されたMATH評価"""

    for problem in problems:
        # 0-shot CoTプロンプト
        prompt = f"""Solve this mathematics problem. Provide a complete solution with reasoning.

Problem: {problem}

Solution:"""

        inputs = tokenizer(prompt, return_tensors="pt")

        # 長い回答を許容
        outputs = model.generate(
            **inputs,
            max_length=1024,
            temperature=0.1,
            do_sample=False
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 数学的正解判定
        is_correct = verify_math_solution(response, problem)
```

### ARC-Challenge評価プロトコル
```python
def evaluate_arc_standard(model, tokenizer, questions):
    """標準化されたARC評価"""

    for item in questions:
        # 10-shot形式（簡易版）
        prompt = f"""Question: {item['question']}

Choices:
{chr(10).join(item['choices'])}

Answer with only the letter of the correct choice (A, B, C, or D):"""

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_length=256,
            temperature=0.1,
            do_sample=False
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # A/B/C/Dの1文字強制抽出
        predicted = extract_arc_answer_standard(response)

        is_correct = (predicted == item['correct'])
```

### ELYZA Tasks 100評価プロトコル
```python
def evaluate_elyza_standard(model, tokenizer, tasks):
    """標準化されたELYZA評価"""

    for task in tasks:
        # 日本語プロンプト
        prompt = f"""以下の質問に対して、正確で役立つ回答を提供してください。

質問: {task}

回答:"""

        inputs = tokenizer(prompt, return_tensors="pt")

        # 創造性を許容
        outputs = model.generate(
            **inputs,
            max_length=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 4-5点スケールでの自動評価
        score = score_elyza_response_standard(response, task)
```

## 3. 答え抽出の標準化

### GSM8K答え抽出標準
```python
def extract_final_answer_standard(response):
    """GSM8Kの標準的な答え抽出"""

    # 最終的な数値パターンを検索
    import re

    # まず最終行から検索
    lines = response.strip().split('\n')
    for line in reversed(lines):
        # 「答えはX」パターン
        match = re.search(r'答えは[:\s]*([+-]?\d+(?:\.\d+)?)', line)
        if match:
            return match.group(1)

        # 「= X」パターン
        match = re.search(r'[=\s]+([+-]?\d+(?:\.\d+)?)$', line)
        if match:
            return match.group(1)

        # 単独の数値
        numbers = re.findall(r'\b\d+\b', line)
        if numbers:
            return numbers[-1]

    # 全体から最後の数値を検索
    all_numbers = re.findall(r'\b\d+\b', response)
    return all_numbers[-1] if all_numbers else "0"
```

### ARC答え抽出標準
```python
def extract_arc_answer_standard(response):
    """ARCの標準的な答え抽出（A/B/C/Dのみ）"""

    import re

    # 大文字変換
    response_upper = response.upper()

    # 明確なパターン検索
    patterns = [
        r'\b([A-D])\b',                    # A, B, C, D
        r'answer:\s*([A-D])',             # answer: A
        r'choice:\s*([A-D])',             # choice: A
        r'([A-D])\s*\.',                  # A.
        r'([A-D])\s*\)',                  # A)
    ]

    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)

    # 最初のA/B/C/D文字を検索
    for char in response_upper:
        if char in 'ABCD':
            return char

    return None
```

### MATH正解検証標準
```python
def verify_math_solution(response, problem):
    """MATHの標準的な正解検証"""

    # 問題タイプに応じた検証
    if "solve for" in problem.lower() or "find" in problem.lower():
        # 方程式解
        return verify_equation_solution(response, problem)
    elif "derivative" in problem.lower():
        # 微分
        return verify_derivative_solution(response, problem)
    elif "integral" in problem.lower():
        # 積分
        return verify_integral_solution(response, problem)
    else:
        # 一般的な数学的検証
        return verify_general_math(response, problem)
```

## 4. 統計分析の標準化

### 信頼区間計算標準
```python
def calculate_confidence_interval(scores, confidence=0.95):
    """標準化された95%信頼区間計算"""

    import numpy as np
    from scipy import stats

    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)  # サンプル標準偏差

    if n < 2:
        return mean, mean, 0

    # t分布を使用（小サンプルサイズ対応）
    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_value * std / np.sqrt(n)

    lower = mean - margin
    upper = mean + margin

    return lower, upper, margin
```

### 有意性検定標準
```python
def perform_significance_test(scores, baseline, alpha=0.05):
    """標準化された統計的有意性検定"""

    from scipy import stats

    # 片側t検定（改善を検定）
    t_stat, p_value = stats.ttest_1samp(scores, baseline, alternative='greater')

    # Cohen's d効果量
    import numpy as np
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    cohens_d = (mean_score - baseline) / std_score if std_score > 0 else 0

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
        "effect_size_interpretation": interpret_cohens_d(cohens_d)
    }
```

## 5. 評価環境の標準化

### ハードウェア仕様
- **GPU**: NVIDIA RTX 3080以上 (12GB+ VRAM)
- **RAM**: 64GB+ システムメモリ
- **ストレージ**: 50GB+ 空き容量
- **CUDA**: 12.0+

### ソフトウェア環境
- **Python**: 3.11+
- **PyTorch**: 2.0.1+
- **Transformers**: 4.30.0+
- **PEFT**: 0.6.0+
- **Datasets**: 2.14.0+

### 再現性確保
```python
def set_evaluation_seed(seed):
    """評価の再現性を確保"""
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # CuDNNの決定論的動作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## 6. バージョン管理と追跡

### 評価スクリプトのバージョン
- **v2.5.0**: 初期標準化
- **変更履歴**: 全変更を追跡
- **互換性**: 過去バージョンとの比較を可能

### 結果の永続化
```python
def save_evaluation_results(results, metadata):
    """評価結果の標準化保存"""

    output = {
        "metadata": {
            "model_version": "AEGIS v2.5",
            "evaluation_version": "2.5.0",
            "timestamp": datetime.now().isoformat(),
            "hardware": get_hardware_info(),
            "software": get_software_versions()
        },
        "configuration": EVALUATION_CONFIG,
        "results": results,
        "statistics": calculate_statistics(results)
    }

    # JSON保存
    with open(f"evaluation_results_{timestamp}.json", 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # マークダウンレポート生成
    generate_evaluation_report(output)
```

## 7. 品質保証チェックリスト

### 前評価チェック
- [ ] モデルが正しく読み込まれている
- [ ] トークナイザーが一致している
- [ ] GPUメモリが十分
- [ ] 評価データが正しい

### 中評価チェック
- [ ] 生成パラメータが標準設定
- [ ] 答え抽出が正しく動作
- [ ] エラーハンドリングが適切
- [ ] メモリリークなし

### 後評価チェック
- [ ] 統計計算が正しい
- [ ] 信頼区間が適切
- [ ] 有意性検定が有効
- [ ] 結果が再現可能

## 8. トラブルシューティング

### 一般的な問題
1. **メモリ不足**: バッチサイズを小さくする
2. **タイムアウト**: 生成長を制限する
3. **エンコーディングエラー**: UTF-8を確保する
4. **結果のばらつき**: シードを固定する

### ベンチマーク固有の問題
- **GSM8K**: 答え抽出ロジックを確認
- **MATH**: タイムアウト設定を調整
- **ARC**: A/B/C/D抽出を検証
- **ELYZA**: 日本語エンコーディングを確認

---

*このドキュメントにより、AEGIS v2.5の評価の透明性と再現性が保証されます。*