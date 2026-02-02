"""
Safety-Aware SO8T Model 使用例

基本的な使用方法とデモコード
"""

import torch
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.safety_aware_so8t import SafetyAwareSO8TConfig, SafetyAwareSO8TModel
from src.agents.thinking_router import ThinkingRouter


def main():
    """メイン関数"""
    base_name = "Qwen/Qwen2-1.5B"  # 好きなCausalLMに変更可
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 設定を作成
    so8t_cfg = SafetyAwareSO8TConfig(
        pet_lambda=0.1,
        alpha_safety=2.0,
        beta_danger_penalty=8.0,
        gamma_safe_allow_reward=1.0,
        delta_escalate_penalty=0.5,
        safety_conf_threshold=0.7,
        use_verifier_head=True,
        mu_norm=0.01,
        nu_orth=0.01,
        rho_iso=0.01,
    )
    
    # モデルとトークナイザーをロード
    print("[INFO] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = SafetyAwareSO8TModel(base_name, so8t_cfg).to(device)
    
    # ThinkingRouterを作成
    router = ThinkingRouter(model, tokenizer, device=device)
    
    # テストプロンプト
    prompt = "Explain the concept of SO(8) rotational symmetry in simple terms."
    
    print(f"\n[INFO] Processing prompt: {prompt}")
    print("[INFO] Using /thinking mode...")
    
    # /thinkingモードで処理
    result = router.process(
        f"/thinking {prompt}",
        do_self_verification=True,
        num_paths=3,
    )
    
    # 結果を表示
    print(f"\n[RESULT] Decision: {result['decision']}")
    print(f"[RESULT] Confidence: {result['confidence']:.4f}")
    print(f"[RESULT] Answer:\n{result['answer']}")
    
    if result.get("thinking"):
        print(f"\n[THINKING] Generated {len(result['thinking'])} reasoning paths")
        for i, path in enumerate(result['thinking']):
            print(f"\n[PATH {i}]:\n{path[:200]}...")
    
    # 通常モードでも試す
    print("\n" + "="*80)
    print("[INFO] Using normal mode...")
    
    result_normal = router.process(
        prompt,
        do_self_verification=False,
    )
    
    print(f"\n[RESULT] Decision: {result_normal['decision']}")
    print(f"[RESULT] Confidence: {result_normal['confidence']:.4f}")
    print(f"[RESULT] Answer:\n{result_normal['answer']}")


if __name__ == "__main__":
    main()

