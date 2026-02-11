"""
FastAPI service exposing SO8T inference with audit logging.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from safety_sql.sqlmm import SQLMemoryManager
from scripts.demo_infer import infer

app = FastAPI(title="SO8T Inference Service")
db_manager: Optional[SQLMemoryManager] = None


class InferRequest(BaseModel):
    text: str
    checkpoint: Optional[str] = None


class InferResponse(BaseModel):
    decision: str
    score: float
    probabilities: list[float]


@app.on_event("startup")
async def startup() -> None:
    global db_manager
    db_path = Path("database/so8t_audit.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_manager = SQLMemoryManager(db_path)


@app.post("/infer", response_model=InferResponse)
async def infer_endpoint(payload: InferRequest) -> InferResponse:
    checkpoint = Path(payload.checkpoint) if payload.checkpoint else None
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, infer, payload.text, checkpoint)
    if db_manager is not None:
        db_manager.log_decision(
            conversation_id="api",
            user_input=payload.text,
            model_output=str(result),
            decision=result["decision"],
            verifier_score=float(result["score"]),
        )
    return InferResponse(**result)


def run() -> None:
    import uvicorn

    uvicorn.run("scripts.serve_fastapi:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
