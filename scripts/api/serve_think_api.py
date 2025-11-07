"""
SO8T /think エンドポイント実装

内部推論（非公開）→安全評価→要約した最終回答というフローを提供する
本番運用可能なREST API実装
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer

# プロジェクトルートをパスに追加
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))

# SO8Tモデルのインポート（パスを追加済み）
from models.safety_aware_so8t import (
    SafetyAwareSO8TModel,
    SafetyAwareSO8TConfig,
)
from safety_sql.sqlmm import SQLMemoryManager

app = FastAPI(
    title="SO8T Think API",
    description="SO8T model with internal reasoning and safety evaluation",
    version="1.0.0",
)

# グローバル変数
model: Optional[SafetyAwareSO8TModel] = None
tokenizer: Optional[AutoTokenizer] = None
db_manager: Optional[SQLMemoryManager] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# 安全ラベルのマッピング
SAFETY_LABELS = ["ALLOW", "ESCALATE", "REFUSE"]


class ThinkRequest(BaseModel):
    """/thinkエンドポイントのリクエストモデル"""
    user_id: str = Field(..., description="ユーザーID", min_length=1, max_length=256)
    query: str = Field(..., min_length=1, max_length=10000, description="ユーザークエリ")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="最大生成トークン数")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="サンプリング温度")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-pサンプリング")
    
    @validator("query")
    def validate_query(cls, v):
        """クエリの検証"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
    @validator("user_id")
    def validate_user_id(cls, v):
        """ユーザーIDの検証"""
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty or whitespace only")
        # セキュリティ: SQLインジェクション対策
        if any(char in v for char in ["'", '"', ";", "--", "/*", "*/"]):
            raise ValueError("User ID contains invalid characters")
        return v.strip()


class ThinkResponse(BaseModel):
    """/thinkエンドポイントのレスポンスモデル"""
    answer: str = Field(..., description="最終回答")
    safety_label: str = Field(..., description="安全判定ラベル")
    safety_conf: float = Field(..., ge=0.0, le=1.0, description="安全判定の信頼度")
    verifier_plausibility: Optional[float] = Field(None, description="Verifier妥当性スコア")
    verifier_self_confidence: Optional[float] = Field(None, description="Verifier自己信頼度")
    escalated: bool = Field(False, description="エスカレーションが必要かどうか")
    internal_reasoning_hash: Optional[str] = Field(None, description="内部推論のハッシュ値（監査用）")


def load_model(
    base_model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    model_path: Optional[str] = None,
) -> tuple[SafetyAwareSO8TModel, AutoTokenizer]:
    """
    SO8Tモデルとトークナイザーをロード
    
    Args:
        base_model_name: ベースモデル名
        model_path: カスタムモデルパス（オプション）
    
    Returns:
        (model, tokenizer)のタプル
    
    Raises:
        FileNotFoundError: モデルファイルが見つからない場合
        RuntimeError: モデルのロードに失敗した場合
    """
    model_name_or_path = model_path if model_path else base_model_name
    print(f"[INFO] Loading model: {model_name_or_path}")
    
    try:
        # SO8T設定
        so8t_config = SafetyAwareSO8TConfig(
            num_safety_labels=3,
            num_verifier_dims=2,  # plausibility, self_confidence
            use_verifier_head=True,
            use_strict_so8_rotation=True,
        )
        
        # モデルをロード
        try:
            model = SafetyAwareSO8TModel(
                base_model_name_or_path=model_name_or_path,
                so8t_config=so8t_config,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SafetyAwareSO8TModel: {e}") from e
        
        model.eval()
        model.to(device)
        
        # トークナイザーをロード
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}") from e
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[INFO] Model loaded successfully on {device}")
        return model, tokenizer
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {model_name_or_path}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading model: {e}") from e


def build_internal_think_prompt(user_query: str) -> str:
    """
    内部推論用プロンプトを構築
    
    Args:
        user_query: ユーザークエリ
    
    Returns:
        内部推論用プロンプト
    """
    return (
        "You are SO8T, an aligned multi-role model with internal reasoning.\n"
        "You will now think step by step to answer the user's request.\n"
        "Do not produce the final user-facing answer yet.\n"
        "Mark your internal reasoning between <think> and </think> tags.\n"
        "User query:\n"
        f"{user_query}\n"
        "Begin internal reasoning:\n"
        "<think>"
    )


def build_external_answer_prompt(user_query: str, internal_reasoning: str) -> str:
    """
    最終回答生成用プロンプトを構築
    
    Args:
        user_query: ユーザークエリ
        internal_reasoning: 内部推論テキスト
    
    Returns:
        最終回答生成用プロンプト
    """
    return (
        "You are SO8T.\n"
        "You have the following internal reasoning (not to be shown verbatim to the user):\n"
        f"{internal_reasoning}\n"
        "Now produce a concise, safe, and helpful final answer to the user.\n"
        "Do not reveal the <think> content or any hidden chain-of-thought.\n"
        "If the reasoning indicates the request is disallowed, respond with a brief refusal.\n"
        "User query:\n"
        f"{user_query}\n"
        "Final answer:"
    )


def extract_internal_reasoning(text: str) -> str:
    """
    内部推論テキストを抽出（<think>...</think>タグから）
    
    Args:
        text: 生成されたテキスト
    
    Returns:
        抽出された内部推論テキスト
    """
    start = text.find("<think>")
    end = text.find("</think>")
    if start != -1 and end != -1 and end > start:
        return text[start + len("<think>"):end].strip()
    return text


def compute_text_hash(text: str) -> str:
    """
    テキストのSHA-256ハッシュを計算（監査ログ用）
    
    Args:
        text: ハッシュ化するテキスト
    
    Returns:
        ハッシュ値（16進数文字列）
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@app.on_event("startup")
async def startup() -> None:
    """アプリケーション起動時の初期化"""
    global model, tokenizer, db_manager
    
    # モデルをロード
    base_model_name = os.getenv("SO8T_BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
    model_path = os.getenv("SO8T_MODEL_PATH", None)
    
    try:
        model, tokenizer = load_model(base_model_name, model_path)
    except FileNotFoundError as e:
        print(f"[ERROR] Model file not found: {e}")
        print("[WARNING] API will start but /think endpoint will return errors until model is loaded")
        model, tokenizer = None, None
    except RuntimeError as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[WARNING] API will start but /think endpoint will return errors until model is loaded")
        model, tokenizer = None, None
    except Exception as e:
        print(f"[ERROR] Unexpected error during model loading: {e}")
        import traceback
        traceback.print_exc()
        print("[WARNING] API will start but /think endpoint will return errors until model is loaded")
        model, tokenizer = None, None
    
    # データベースマネージャーを初期化
    db_path = Path("database/so8t_think_audit.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        db_manager = SQLMemoryManager(db_path)
        print(f"[INFO] Database initialized: {db_path}")
    except Exception as e:
        print(f"[WARNING] Failed to initialize database: {e}")
        import traceback
        traceback.print_exc()
        db_manager = None
        print("[WARNING] API will continue without audit logging")


@app.on_event("shutdown")
async def shutdown() -> None:
    """アプリケーション終了時のクリーンアップ"""
    global model, tokenizer
    model = None
    tokenizer = None
    print("[INFO] Model unloaded")


def run_so8t_safety_and_verifier(
    context_text: str,
) -> tuple[str, float, Optional[float], Optional[float]]:
    """
    SO8Tモデルを使用して安全評価とVerifier評価を実行
    
    Args:
        context_text: 評価するテキスト（内部推論または最終回答）
    
    Returns:
        (safety_label, safety_conf, plausibility, self_confidence)のタプル
    
    Raises:
        HTTPException: モデルがロードされていない場合、または評価に失敗した場合
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not context_text or not context_text.strip():
        raise ValueError("Context text cannot be empty")
    
    try:
        # トークナイザーでエンコード
        inputs = tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 安全評価を実行
        with torch.no_grad():
            try:
                safety_result = model.safety_gate(**inputs)
            except Exception as e:
                raise RuntimeError(f"Safety gate evaluation failed: {e}") from e
            
            if "decisions" not in safety_result or "confidence" not in safety_result:
                raise RuntimeError("Invalid safety gate result format")
            
            safety_probs = safety_result["safety_probs"][0]
            safety_label = safety_result["decisions"][0]
            safety_conf = safety_result["confidence"][0].item()
            
            # Verifier評価を実行
            try:
                outputs = model.forward(
                    **inputs,
                    output_hidden_states=True,
                )
            except Exception as e:
                raise RuntimeError(f"Model forward pass failed: {e}") from e
            
            verifier_scores = outputs.get("verifier_scores")
            plausibility = None
            self_confidence = None
            
            if verifier_scores is not None:
                # verifier_scores: [batch, num_verifier_dims]
                if verifier_scores.shape[1] >= 2:
                    plausibility = float(verifier_scores[0, 0].item())
                    self_confidence = float(verifier_scores[0, 1].item())
                elif verifier_scores.shape[1] == 1:
                    self_confidence = float(verifier_scores[0, 0].item())
            
            return safety_label, safety_conf, plausibility, self_confidence
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during evaluation: {str(e)}")


def generate_text(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    テキストを生成（lm_headのみ使用）
    
    Args:
        prompt: プロンプト
        max_new_tokens: 最大生成トークン数
        temperature: サンプリング温度
        top_p: Top-pサンプリング
    
    Returns:
        生成されたテキスト
    
    Raises:
        HTTPException: モデルがロードされていない場合、または生成に失敗した場合
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                output_ids = model.base_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except Exception as e:
                raise RuntimeError(f"Text generation failed: {e}") from e
        
        # 生成されたテキストをデコード
        try:
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        except Exception as e:
            raise RuntimeError(f"Token decoding failed: {e}") from e
        
        # プロンプト部分を除去
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].lstrip()
        
        return generated_text
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during generation: {str(e)}")


@app.post("/think", response_model=ThinkResponse)
async def think_endpoint(req: ThinkRequest, request: Request) -> ThinkResponse:
    """
    /thinkエンドポイント: 内部推論→安全評価→最終回答生成
    
    フロー:
    1. 内部推論プロンプトを構築し、内部推論を生成
    2. 内部推論に対して安全評価とVerifier評価を実行
    3. 安全判定がALLOWの場合のみ、内部推論を要約した最終回答を生成
    4. 最終回答にも安全チェックを実行
    5. 監査ログを記録
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    user_id = req.user_id
    query = req.query  # すでにvalidatorで検証済み
    
    try:
        # 1. 内部推論プロンプトを構築
        internal_prompt = build_internal_think_prompt(query)
        
        # 2. 内部推論を生成
        internal_full = generate_text(
            internal_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        
        # 3. 内部推論テキストを抽出
        internal_reasoning = extract_internal_reasoning(internal_full)
        if not internal_reasoning:
            internal_reasoning = internal_full
        
        # 4. 内部推論のハッシュを計算（監査ログ用）
        internal_reasoning_hash = compute_text_hash(internal_reasoning)
        
        # 5. 安全評価とVerifier評価を実行
        safety_label, safety_conf, plaus, self_conf = run_so8t_safety_and_verifier(
            internal_reasoning
        )
        
        # 6. 監査ログを記録（内部推論ハッシュ、安全判定、Verifierスコア）
        if db_manager is not None:
            try:
                # 内部推論の詳細ログ
                db_manager.log_audit_event(
                    event_type="think_internal_reasoning",
                    payload={
                        "user_id": user_id,
                        "query": query,
                        "internal_reasoning_hash": internal_reasoning_hash,
                        "internal_reasoning_length": len(internal_reasoning),
                        "safety_label": safety_label,
                        "safety_conf": float(safety_conf),
                        "verifier_plausibility": float(plaus) if plaus is not None else None,
                        "verifier_self_confidence": float(self_conf) if self_conf is not None else None,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
                
                # decision_logにも記録（互換性のため）
                db_manager.log_decision(
                    conversation_id=user_id,
                    user_input=query,
                    model_output=f"[Internal Reasoning Hash: {internal_reasoning_hash}]",
                    decision=safety_label,
                    verifier_score=float(plaus) if plaus is not None else float(self_conf) if self_conf is not None else 0.0,
                )
            except Exception as e:
                print(f"[WARNING] Failed to log audit event: {e}")
                import traceback
                traceback.print_exc()
        
        # 7. 安全判定に基づいて処理
        if safety_label == "REFUSE":
            final_answer = "申し訳ありませんが、このご要望にはお応えできません。"
            return ThinkResponse(
                answer=final_answer,
                safety_label=safety_label,
                safety_conf=safety_conf,
                verifier_plausibility=plaus,
                verifier_self_confidence=self_conf,
                escalated=False,
                internal_reasoning_hash=internal_reasoning_hash,
            )
        
        if safety_label == "ESCALATE":
            final_answer = "このリクエストは追加の確認が必要です。担当者にエスカレーションします。"
            
            # 監査ログを記録
            if db_manager is not None:
                try:
                    db_manager.log_audit_event(
                        event_type="think_escalation",
                        payload={
                            "user_id": user_id,
                            "query": query,
                            "internal_reasoning_hash": internal_reasoning_hash,
                            "safety_label": safety_label,
                            "safety_conf": safety_conf,
                        },
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to log escalation event: {e}")
            
            return ThinkResponse(
                answer=final_answer,
                safety_label=safety_label,
                safety_conf=safety_conf,
                verifier_plausibility=plaus,
                verifier_self_confidence=self_conf,
                escalated=True,
                internal_reasoning_hash=internal_reasoning_hash,
            )
        
        # 8. ALLOWの場合: 内部推論を要約した最終回答を生成
        external_prompt = build_external_answer_prompt(query, internal_reasoning)
        final_answer = generate_text(
            external_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        
        # 9. 最終回答にも安全チェックを実行
        final_safety_label, final_safety_conf, final_plaus, final_self_conf = (
            run_so8t_safety_and_verifier(final_answer)
        )
        
        if final_safety_label != "ALLOW":
            # 最終生成で逸脱した場合は安全側に倒す
            safe_fallback = "申し訳ありませんが、安全上の理由からこの回答は提供できません。"
            
            # 監査ログを記録
            if db_manager is not None:
                try:
                    db_manager.log_audit_event(
                        event_type="think_final_answer_rejected",
                        payload={
                            "user_id": user_id,
                            "query": query,
                            "internal_reasoning_hash": internal_reasoning_hash,
                            "final_safety_label": final_safety_label,
                            "final_safety_conf": final_safety_conf,
                        },
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to log final answer rejection: {e}")
            
            return ThinkResponse(
                answer=safe_fallback,
                safety_label=final_safety_label,
                safety_conf=final_safety_conf,
                verifier_plausibility=final_plaus,
                verifier_self_confidence=final_self_conf,
                escalated=(final_safety_label == "ESCALATE"),
                internal_reasoning_hash=internal_reasoning_hash,
            )
        
        # 10. 正常な最終回答を返す
        # 監査ログを記録
        if db_manager is not None:
            try:
                # 最終回答の監査ログ
                db_manager.log_audit_event(
                    event_type="think_final_answer",
                    payload={
                        "user_id": user_id,
                        "query": query,
                        "internal_reasoning_hash": internal_reasoning_hash,
                        "final_answer": final_answer,
                        "final_answer_length": len(final_answer),
                        "safety_label": safety_label,
                        "safety_conf": float(safety_conf),
                        "verifier_plausibility": float(plaus) if plaus is not None else None,
                        "verifier_self_confidence": float(self_conf) if self_conf is not None else None,
                    },
                )
                
                # decision_logにも記録
                db_manager.log_decision(
                    conversation_id=user_id,
                    user_input=query,
                    model_output=final_answer,
                    decision="ALLOW",
                    verifier_score=float(plaus) if plaus is not None else float(self_conf) if self_conf is not None else 0.0,
                )
            except Exception as e:
                print(f"[WARNING] Failed to log decision: {e}")
                import traceback
                traceback.print_exc()
        
        return ThinkResponse(
            answer=final_answer,
            safety_label=safety_label,
            safety_conf=safety_conf,
            verifier_plausibility=plaus,
            verifier_self_confidence=self_conf,
            escalated=False,
            internal_reasoning_hash=internal_reasoning_hash,
        )
    
    except Exception as e:
        print(f"[ERROR] Error in think_endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": device,
    }


def run() -> None:
    """サーバーを起動"""
    import uvicorn
    
    host = os.getenv("SO8T_API_HOST", "0.0.0.0")
    port = int(os.getenv("SO8T_API_PORT", "8000"))
    
    uvicorn.run(
        "scripts.serve_think_api:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run()

