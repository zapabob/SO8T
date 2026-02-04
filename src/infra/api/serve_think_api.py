"""
SO8T /think エンドポイント実装（完全置き換え版）

SO8TThinkingModelを使用し、Thinking生成 → Safety/Verifier評価 → Final抽出の
完全フローを提供する本番運用可能なREST API。
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))

# SO8T Thinking Modelのインポート
from models.so8t_thinking_model import SO8TThinkingModel
from models.safety_aware_so8t import SafetyAwareSO8TConfig
from utils.thinking_utils import compute_text_hash
from safety_sql.sqlmm import SQLMemoryManager

app = FastAPI(
    title="SO8T Think API",
    description="SO8T Thinking Model with internal reasoning and safety evaluation",
    version="2.0.0",
)

# グローバル変数
model: Optional[SO8TThinkingModel] = None
tokenizer: Optional[AutoTokenizer] = None
db_manager: Optional[SQLMemoryManager] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# 安全ラベルのマッピング
SAFETY_LABELS = ["ALLOW", "ESCALATE", "REFUSE"]


class ThinkRequest(BaseModel):
    """/thinkエンドポイントのリクエストモデル"""
    user_id: str = Field(..., description="ユーザーID", min_length=1, max_length=256)
    query: str = Field(..., min_length=1, max_length=10000, description="ユーザークエリ")
    max_new_tokens: int = Field(512, ge=1, le=2048, description="最大生成トークン数")
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
    answer: str = Field(..., description="最終回答（Final部分のみ）")
    safety_label: str = Field(..., description="安全判定ラベル")
    safety_conf: float = Field(..., ge=0.0, le=1.0, description="安全判定の信頼度")
    domain_label: Optional[str] = Field(None, description="ドメインラベル")
    domain_conf: Optional[float] = Field(None, ge=0.0, le=1.0, description="ドメイン判定の信頼度")
    verifier_plausibility: Optional[float] = Field(None, description="Verifier妥当性スコア")
    verifier_self_confidence: Optional[float] = Field(None, description="Verifier自己信頼度")
    escalated: bool = Field(False, description="エスカレーションが必要かどうか")
    internal_reasoning_hash: Optional[str] = Field(None, description="内部推論のハッシュ値（監査用）")


def load_model(
    base_model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    model_path: Optional[str] = None,
    use_redacted: bool = False,
    use_quadruple: bool = True,
) -> tuple[SO8TThinkingModel, AutoTokenizer]:
    """
    SO8TThinkingModelとトークナイザーをロード
    
    Args:
        base_model_name: ベースモデル名
        model_path: カスタムモデルパス（オプション）
        use_redacted: Trueの場合、<think>形式を使用
    
    Returns:
        (model, tokenizer)のタプル
    
    Raises:
        FileNotFoundError: モデルファイルが見つからない場合
        RuntimeError: モデルのロードに失敗した場合
    """
    model_name_or_path = model_path if model_path else base_model_name
    print(f"[INFO] Loading SO8T Thinking Model: {model_name_or_path}")
    
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
            model = SO8TThinkingModel(
                base_model_name_or_path=model_name_or_path,
                so8t_config=so8t_config,
                use_redacted_tokens=use_redacted,
                use_quadruple_thinking=use_quadruple,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SO8TThinkingModel: {e}") from e
        
        model.eval()
        model.to(device)
        
        # トークナイザーをロード
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}") from e
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 特殊トークンを追加
        model.set_tokenizer(tokenizer)
        
        print(f"[INFO] Model loaded successfully on {device}")
        return model, tokenizer
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {model_name_or_path}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading model: {e}") from e


@app.on_event("startup")
async def startup():
    """アプリケーション起動時の初期化"""
    global model, tokenizer, db_manager
    
    # 環境変数から設定を取得
    model_path = os.getenv("SO8T_MODEL_PATH")
    base_model = os.getenv("SO8T_BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
    use_redacted = os.getenv("SO8T_USE_REDACTED", "false").lower() == "true"
    use_quadruple = os.getenv("SO8T_USE_QUADRUPLE", "true").lower() == "true"
    db_path = os.getenv("SO8T_DB_PATH", "database/so8t_compliance.db")
    
    # モデルをロード
    model, tokenizer = load_model(
        base_model_name=base_model,
        model_path=model_path,
        use_redacted=use_redacted,
        use_quadruple=use_quadruple,
    )
    
    # データベースマネージャーを初期化
    try:
        db_manager = SQLMemoryManager(db_path)
        print(f"[INFO] Database manager initialized: {db_path}")
    except Exception as e:
        print(f"[WARNING] Failed to initialize database manager: {e}")
        db_manager = None


@app.post("/think", response_model=ThinkResponse)
async def think_endpoint(req: ThinkRequest, request: Request) -> ThinkResponse:
    """
    /thinkエンドポイント: Thinking生成 → Safety/Verifier評価 → Final抽出
    
    フロー:
    1. Thinking生成用プロンプトを構築し、Thinking形式で生成
    2. ThinkingとFinalを抽出
    3. Safety/Verifier評価を実行
    4. 安全ゲート: REFUSE/ESCALATE時の適切な処理
    5. 監査ログを記録（Thinkingハッシュ、Safety判定、Verifierスコア）
    6. Finalのみをユーザーに返す（Thinkingは非公開）
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    user_id = req.user_id
    query = req.query
    
    try:
        # 1. Thinking生成 → Safety/Verifier評価 → Final抽出の完全フロー
        result = model.generate_with_safety_gate(
            tokenizer=tokenizer,
            user_query=query,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            device=device,
        )
        
        thinking_text = result.get("thinking")
        final_text = result.get("final")
        safety_label = result.get("safety_label", "REFUSE")
        safety_conf = result.get("safety_confidence", 0.0)
        domain_label = result.get("domain_label", "general")
        domain_conf = result.get("domain_confidence", 0.0)
        plausibility = result.get("verifier_plausibility")
        self_confidence = result.get("verifier_self_confidence")
        escalated = result.get("escalated", False)
        refused = result.get("refused", False)
        
        # 2. Thinkingのハッシュを計算（監査ログ用）
        thinking_hash = None
        if thinking_text:
            thinking_hash = compute_text_hash(thinking_text)
        
        # 3. 安全ゲート処理
        if refused:
            # REFUSEの場合
            return ThinkResponse(
                answer="申し訳ありませんが、このご要望にはお応えできません。",
                safety_label=safety_label,
                safety_conf=safety_conf,
                domain_label=domain_label,
                domain_conf=domain_conf,
                verifier_plausibility=plausibility,
                verifier_self_confidence=self_confidence,
                escalated=False,
                internal_reasoning_hash=thinking_hash,
            )
        
        if escalated:
            # ESCALATEの場合
            return ThinkResponse(
                answer="このリクエストは追加の確認が必要です。担当者にエスカレーションします。",
                safety_label=safety_label,
                safety_conf=safety_conf,
                domain_label=domain_label,
                domain_conf=domain_conf,
                verifier_plausibility=plausibility,
                verifier_self_confidence=self_confidence,
                escalated=True,
                internal_reasoning_hash=thinking_hash,
            )
        
        # 4. ALLOWの場合: Finalを返す（Thinkingは非公開）
        final_answer = final_text if final_text else "回答を生成できませんでした。"
        
        # 5. 監査ログを記録
        if db_manager is not None:
            try:
                db_manager.log_decision(
                    conversation_id=user_id,
                    user_input=query,
                    model_output=final_answer,
                    decision=safety_label,
                    verifier_score=float(self_confidence) if self_confidence else 0.0,
                )
            except Exception as e:
                print(f"[WARNING] Failed to log decision: {e}")
        
        return ThinkResponse(
            answer=final_answer,
            safety_label=safety_label,
            safety_conf=safety_conf,
            domain_label=domain_label,
            domain_conf=domain_conf,
            verifier_plausibility=plausibility,
            verifier_self_confidence=self_confidence,
            escalated=False,
            internal_reasoning_hash=thinking_hash,
        )
    
    except Exception as e:
        print(f"[ERROR] Error in /think endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": device,
    }


def run():
    """APIサーバーを起動"""
    import uvicorn
    
    port = int(os.getenv("SO8T_API_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
