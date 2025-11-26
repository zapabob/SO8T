"""
SO8T Thinking Model Implementation

SafetyAwareSO8TModelを継承し、Thinking出力形式（<think>...</think><final>...</final>）を
サポートするモデル。内部推論と最終回答を分離し、Safety/Verifierヘッドで安全ゲートを行う。
"""

from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .safety_aware_so8t import SafetyAwareSO8TModel, SafetyAwareSO8TConfig
from .thinking_tokens import (
    add_thinking_tokens_to_tokenizer,
    extract_thinking_and_final,
    extract_quadruple_thinking,
    format_thinking_output,
    format_quadruple_thinking_output,
    build_thinking_prompt,
    get_token_ids,
)

# ドメインラベル定義
DOMAIN_LABELS = [
    "defense_public",
    "aerospace",
    "medical_reg",
    "law_policy",
    "wikipedia_ja_en",
    "nsfw_adult",
    "nsfw_block",
    "general",
]
NUM_DOMAIN_LABELS = len(DOMAIN_LABELS)


class SO8TThinkingModel(SafetyAwareSO8TModel):
    """
    SO8T Thinking Model
    
    SafetyAwareSO8TModelを拡張し、Thinking出力形式をサポート。
    - <think>...</think> で内部推論を生成
    - <final>...</final> で最終回答を生成
    - Safety/Verifierヘッドで安全ゲート
    """
    
    def __init__(
        self,
        base_model_name_or_path: str,
        so8t_config: SafetyAwareSO8TConfig,
        use_redacted_tokens: bool = False,
        use_quadruple_thinking: bool = False,
        quantization_config: Optional[Any] = None,
    ):
        """
        Args:
            base_model_name_or_path: ベースモデル名またはパス
            so8t_config: SO8T設定
            use_redacted_tokens: Trueの場合、<think>形式を使用（デフォルトは<think>）
            use_quadruple_thinking: Trueの場合、四重推論形式（Task/Safety/Policy/Final）を使用
            quantization_config: 量子化設定（BitsAndBytesConfig等）
        """
        super().__init__(base_model_name_or_path, so8t_config, quantization_config=quantization_config)
        self.use_redacted_tokens = use_redacted_tokens
        self.use_quadruple_thinking = use_quadruple_thinking
        
        # Domainヘッドを追加（Spinor-成分からドメイン分類）
        hidden_size = self.base_model.config.hidden_size
        _, _, d_S_minus, _ = so8t_config.compute_role_dimensions(hidden_size)
        
        self.domain_head = nn.Sequential(
            nn.Linear(d_S_minus, d_S_minus // 2),
            nn.GELU(),
            nn.Linear(d_S_minus // 2, NUM_DOMAIN_LABELS),
        )
        
        # トークナイザーに特殊トークンを追加（後で設定される）
        self._tokenizer = None
    
    def set_tokenizer(self, tokenizer: AutoTokenizer):
        """
        トークナイザーを設定し、特殊トークンを追加
        
        Args:
            tokenizer: トークナイザー
        """
        self._tokenizer = add_thinking_tokens_to_tokenizer(
            tokenizer,
            use_redacted=self.use_redacted_tokens,
            use_quadruple=self.use_quadruple_thinking
        )
        # ベースモデルの埋め込み層をリサイズ
        if hasattr(self.base_model, 'resize_token_embeddings'):
            self.base_model.resize_token_embeddings(len(self._tokenizer))
    
    @torch.no_grad()
    def generate_thinking(
        self,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Thinking形式でテキストを生成
        
        Args:
            tokenizer: トークナイザー
            prompt: 入力プロンプト
            max_new_tokens: 最大生成トークン数
            temperature: サンプリング温度
            top_p: Top-pサンプリング
            do_sample: サンプリングを使用するか
            device: デバイス
        
        Returns:
            生成結果の辞書
        """
        # プロンプトをエンコード
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # 生成
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # デコード
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=False
        )
        
        # プロンプト部分を除去
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):].strip()
        
        # ThinkingとFinalを抽出
        if self.use_quadruple_thinking:
            task_text, safety_text, policy_text, final_text = extract_quadruple_thinking(generated_text)
            # 四重推論の場合は、すべてを結合してthinking_textとする
            thinking_parts = [p for p in [task_text, safety_text, policy_text] if p]
            thinking_text = "\n".join(thinking_parts) if thinking_parts else None
        else:
            thinking_text, final_text = extract_thinking_and_final(
                generated_text,
                use_redacted=self.use_redacted_tokens,
                use_quadruple=False
            )
        
        return {
            "full_text": generated_text,
            "thinking": thinking_text,
            "final": final_text,
            "raw_output": generated_text,
        }
    
    @torch.no_grad()
    def evaluate_safety_and_verifier(
        self,
        tokenizer: AutoTokenizer,
        thinking_text: str,
        final_text: str,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        ThinkingとFinalに対してSafety/Domain/Verifier評価を実行
        
        Args:
            tokenizer: トークナイザー
            thinking_text: 内部推論テキスト
            final_text: 最終回答テキスト
            device: デバイス
        
        Returns:
            評価結果の辞書
        """
        # 評価用テキスト（Thinking + Final）
        eval_text = f"{thinking_text}\n{final_text}" if thinking_text else final_text
        
        # エンコード
        inputs = tokenizer(
            eval_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        
        # Forward pass
        outputs = self.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
        )
        
        # 隠れ状態からDomainヘッド用の特徴を取得
        hidden_states = outputs.get("hidden_states")
        if hidden_states is not None:
            last_hidden = hidden_states[-1]  # [1, T, H]
            # 最終トークンのSpinor-成分を取得
            _, _, h_S_minus, _ = self.split_hidden_states(last_hidden)
            pooled_S_minus = h_S_minus[:, -1, :]  # [1, d_S_minus]
            
            # Domain判定
            domain_logits = self.domain_head(pooled_S_minus)  # [1, NUM_DOMAIN_LABELS]
            domain_probs = torch.softmax(domain_logits, dim=-1)
            domain_pred = int(torch.argmax(domain_probs, dim=-1).item())
            domain_conf = float(domain_probs[0, domain_pred].item())
            domain_label = DOMAIN_LABELS[domain_pred]
        else:
            domain_label = "general"
            domain_conf = 0.0
            domain_logits = None
        
        # Safety判定
        safety_logits = outputs["safety_logits"]  # [1, 3]
        safety_probs = torch.softmax(safety_logits, dim=-1)
        safety_pred = int(torch.argmax(safety_probs, dim=-1).item())
        safety_conf = float(safety_probs[0, safety_pred].item())
        
        safety_labels = ["ALLOW", "ESCALATE", "REFUSE"]
        safety_label = safety_labels[safety_pred]
        
        # Verifierスコア
        verifier_scores = None
        plausibility = None
        self_confidence = None
        
        if outputs["verifier_scores"] is not None:
            verifier_scores = outputs["verifier_scores"][0]  # [num_verifier_dims]
            if verifier_scores.size(0) >= 1:
                plausibility = float(verifier_scores[0].item())
            if verifier_scores.size(0) >= 2:
                self_confidence = float(verifier_scores[1].item())
            elif verifier_scores.size(0) == 1:
                # 1次元の場合、それを自己信頼度として使用
                self_confidence = plausibility
        
        return {
            "safety_label": safety_label,
            "safety_confidence": safety_conf,
            "safety_logits": safety_logits.cpu().numpy(),
            "domain_label": domain_label,
            "domain_confidence": domain_conf,
            "domain_logits": domain_logits.cpu().numpy() if domain_logits is not None else None,
            "verifier_plausibility": plausibility,
            "verifier_self_confidence": self_confidence,
            "verifier_scores": verifier_scores.cpu().numpy() if verifier_scores is not None else None,
        }
    
    @torch.no_grad()
    def evaluate_safety_and_verifier(
        self,
        tokenizer: AutoTokenizer,
        thinking_text: str,
        final_text: str,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        後方互換性のためのエイリアス
        """
        return self.evaluate_safety_domain_and_verifier(tokenizer, thinking_text, final_text, device)
    
    @torch.no_grad()
    def generate_with_safety_gate(
        self,
        tokenizer: AutoTokenizer,
        user_query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Thinking生成 → Safety/Verifier評価 → Final抽出の完全フロー
        
        Args:
            tokenizer: トークナイザー
            user_query: ユーザークエリ
            max_new_tokens: 最大生成トークン数
            temperature: サンプリング温度
            top_p: Top-pサンプリング
            device: デバイス
        
        Returns:
            完全な結果辞書
        """
        # 1. Thinking生成用プロンプトを構築
        prompt = build_thinking_prompt(
            user_query,
            use_redacted=self.use_redacted_tokens,
            use_quadruple=self.use_quadruple_thinking
        )
        
        # 2. Thinking形式で生成
        generation_result = self.generate_thinking(
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        
        thinking_text = generation_result["thinking"]
        final_text = generation_result["final"]
        
        # 3. Safety/Domain/Verifier評価
        if thinking_text and final_text:
            eval_result = self.evaluate_safety_domain_and_verifier(
                tokenizer=tokenizer,
                thinking_text=thinking_text,
                final_text=final_text,
                device=device,
            )
        else:
            # Thinking/Finalが抽出できない場合のフォールバック
            eval_result = {
                "safety_label": "REFUSE",
                "safety_confidence": 0.0,
                "verifier_plausibility": 0.0,
                "verifier_self_confidence": 0.0,
            }
        
        # 4. 結果を統合
        result = {
            "user_query": user_query,
            "thinking": thinking_text,
            "final": final_text,
            "full_generation": generation_result["full_text"],
            "safety_label": eval_result["safety_label"],
            "safety_confidence": eval_result["safety_confidence"],
            "domain_label": eval_result.get("domain_label", "general"),
            "domain_confidence": eval_result.get("domain_confidence", 0.0),
            "verifier_plausibility": eval_result.get("verifier_plausibility"),
            "verifier_self_confidence": eval_result.get("verifier_self_confidence"),
            "escalated": eval_result["safety_label"] == "ESCALATE",
            "refused": eval_result["safety_label"] == "REFUSE",
        }
        
        return result

