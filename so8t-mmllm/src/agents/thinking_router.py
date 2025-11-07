"""
/thinkingモード用エージェント風ルータ

Safety Gate判定後のルーティングとSelf-Verification統合を行う。
"""

from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer

from ..models.safety_aware_so8t import SafetyAwareSO8TModel


class ThinkingRouter:
    """
    /thinkingモード用エージェント風ルータ
    
    Safety Gate判定後のルーティングとSelf-Verification統合を行う。
    """
    
    def __init__(
        self,
        model: SafetyAwareSO8TModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        """
        Args:
            model: SafetyAwareSO8TModelインスタンス
            tokenizer: トークナイザー
            device: デバイス
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def route(
        self,
        prompt: str,
        thinking_mode: bool = False,
        do_self_verification: bool = True,
        num_paths: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        プロンプトをルーティング
        
        Args:
            prompt: 入力プロンプト
            thinking_mode: /thinkingモードかどうか
            do_self_verification: Self-Verificationを行うか
            num_paths: Self-Verificationのパス数
            **kwargs: その他の引数
        
        Returns:
            ルーティング結果の辞書
        """
        # Safety Gateで判定
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        sg = self.model.safety_gate(**inputs)
        decision = sg["decisions"][0]
        conf = sg["confidence"][0].item()
        
        # REFUSEの場合
        if decision == "REFUSE":
            return {
                "decision": "REFUSE",
                "reason": "safety_head_refuse",
                "confidence": conf,
                "answer": "I'm sorry, but I can't assist with that request.",
                "thinking": None,
            }
        
        # ESCALATEの場合
        if decision == "ESCALATE":
            return {
                "decision": "ESCALATE",
                "reason": "safety_head_escalate_or_low_conf",
                "confidence": conf,
                "answer": "This request requires human review or higher-level approval.",
                "thinking": None,
            }
        
        # ALLOWの場合
        if thinking_mode:
            # /thinkingモード: Self-Verificationを必ず実行
            result = self.model.generate_answer(
                tokenizer=self.tokenizer,
                prompt=prompt,
                do_self_verification=True,
                num_paths=num_paths,
                device=self.device,
                **kwargs,
            )
            result["thinking"] = result.get("paths", [])
            return result
        else:
            # 通常モード: Self-Verificationはオプション
            result = self.model.generate_answer(
                tokenizer=self.tokenizer,
                prompt=prompt,
                do_self_verification=do_self_verification,
                num_paths=num_paths if do_self_verification else 1,
                device=self.device,
                **kwargs,
            )
            if do_self_verification:
                result["thinking"] = result.get("paths", [])
            else:
                result["thinking"] = None
            return result
    
    def parse_thinking_command(self, text: str) -> Dict[str, Any]:
        """
        /thinkingコマンドをパース
        
        Args:
            text: 入力テキスト（/thinkingコマンドを含む可能性がある）
        
        Returns:
            パース結果の辞書
        """
        thinking_mode = False
        clean_text = text
        
        # /thinkingコマンドの検出
        if text.strip().startswith("/thinking"):
            thinking_mode = True
            # /thinkingコマンドを除去
            clean_text = text.replace("/thinking", "").strip()
            if clean_text.startswith("\n"):
                clean_text = clean_text[1:]
        
        return {
            "thinking_mode": thinking_mode,
            "clean_text": clean_text,
        }
    
    def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        テキストを処理（/thinkingコマンド対応）
        
        Args:
            text: 入力テキスト
            **kwargs: その他の引数
        
        Returns:
            処理結果の辞書
        """
        # /thinkingコマンドをパース
        parsed = self.parse_thinking_command(text)
        thinking_mode = parsed["thinking_mode"]
        clean_text = parsed["clean_text"]
        
        # ルーティング
        return self.route(
            prompt=clean_text,
            thinking_mode=thinking_mode,
            **kwargs,
        )


