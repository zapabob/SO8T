#!/usr/bin/env python3
"""
SO8T Safe Agent API Server

FastAPI-based REST API server for the SO8T Safe Agent.
Provides HTTP endpoints for safe decision-making and monitoring.

Usage:
    python examples/api_server.py
    python examples/api_server.py --host 0.0.0.0 --port 8000
    uvicorn examples.api_server:app --host 0.0.0.0 --port 8000
"""

import argparse
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from inference.agent_runtime import SO8TAgentRuntime, run_agent
from inference.logging_middleware import LoggingMiddleware


# Pydantic models
class AgentRequest(BaseModel):
    """Agent request model."""
    context: str = Field(..., description="Context for the request")
    user_request: str = Field(..., description="User request")
    request_id: Optional[str] = Field(None, description="Optional request ID")
    config_path: Optional[str] = Field(None, description="Optional config path")


class AgentResponse(BaseModel):
    """Agent response model."""
    request_id: str = Field(..., description="Request ID")
    decision: str = Field(..., description="Decision (ALLOW/REFUSE/ESCALATE)")
    rationale: str = Field(..., description="Rationale for the decision")
    confidence: float = Field(..., description="Confidence score (0-1)")
    human_required: bool = Field(..., description="Whether human intervention is required")
    task_response: Optional[str] = Field(None, description="Task response if applicable")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class BatchRequest(BaseModel):
    """Batch request model."""
    requests: List[AgentRequest] = Field(..., description="List of requests")
    max_workers: int = Field(4, description="Maximum number of workers")


class BatchResponse(BaseModel):
    """Batch response model."""
    results: List[AgentResponse] = Field(..., description="List of responses")
    total_requests: int = Field(..., description="Total number of requests")
    total_time: float = Field(..., description="Total processing time")
    success_count: int = Field(..., description="Number of successful requests")
    error_count: int = Field(..., description="Number of failed requests")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")


class StatsResponse(BaseModel):
    """Statistics response model."""
    total_requests: int = Field(..., description="Total number of requests")
    average_processing_time: float = Field(..., description="Average processing time")
    throughput: float = Field(..., description="Requests per second")
    error_rate: float = Field(..., description="Error rate")
    decision_counts: Dict[str, int] = Field(..., description="Decision counts")
    uptime: float = Field(..., description="Uptime in seconds")


# Global variables
app = FastAPI(
    title="SO8T Safe Agent API",
    description="Safe Operation 8-Task Agent API for secure decision-making",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global runtime and middleware
runtime: Optional[SO8TAgentRuntime] = None
logging_middleware: Optional[LoggingMiddleware] = None
start_time = time.time()

# Statistics
stats = {
    "total_requests": 0,
    "total_time": 0.0,
    "decision_counts": {"ALLOW": 0, "REFUSE": 0, "ESCALATE": 0},
    "error_count": 0
}


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


# Dependency functions
def get_runtime() -> SO8TAgentRuntime:
    """Get SO8T runtime instance."""
    if runtime is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return runtime


def get_logging_middleware() -> LoggingMiddleware:
    """Get logging middleware instance."""
    if logging_middleware is None:
        raise HTTPException(status_code=503, detail="Logging middleware not initialized")
    return logging_middleware


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    global runtime, logging_middleware
    
    try:
        # Initialize runtime
        runtime = SO8TAgentRuntime()
        
        # Initialize logging middleware
        config = {
            "log_dir": "logs",
            "max_log_size": 100 * 1024 * 1024,  # 100MB
            "max_backup_count": 5
        }
        logging_middleware = LoggingMiddleware(config)
        
        logging.info("SO8T Safe Agent API server started successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logging.info("SO8T Safe Agent API server shutting down")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "SO8T Safe Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0",
        uptime=uptime
    )


@app.post("/agent/process", response_model=AgentResponse)
async def process_request(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    runtime: SO8TAgentRuntime = Depends(get_runtime),
    middleware: LoggingMiddleware = Depends(get_logging_middleware)
):
    """Process a single request."""
    start_time = time.time()
    
    try:
        # Process request
        response = runtime.process_request(
            context=request.context,
            user_request=request.user_request,
            request_id=request.request_id
        )
        
        processing_time = time.time() - start_time
        
        # Update statistics
        stats["total_requests"] += 1
        stats["total_time"] += processing_time
        stats["decision_counts"][response["decision"]] += 1
        
        # Log request in background
        background_tasks.add_task(
            log_request,
            request,
            response,
            processing_time,
            middleware
        )
        
        return AgentResponse(
            request_id=response.get("request_id", request.request_id or "unknown"),
            decision=response["decision"],
            rationale=response["rationale"],
            confidence=response["confidence"],
            human_required=response["human_required"],
            task_response=response.get("task_response"),
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        # Update error statistics
        stats["error_count"] += 1
        
        # Log error
        middleware.log_error(
            level="ERROR",
            message=f"Request processing failed: {e}",
            exception=e
        )
        
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/batch", response_model=BatchResponse)
async def process_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    runtime: SO8TAgentRuntime = Depends(get_runtime),
    middleware: LoggingMiddleware = Depends(get_logging_middleware)
):
    """Process multiple requests in batch."""
    start_time = time.time()
    results = []
    success_count = 0
    error_count = 0
    
    try:
        # Process requests
        for agent_request in request.requests:
            try:
                # Process single request
                response = runtime.process_request(
                    context=agent_request.context,
                    user_request=agent_request.user_request,
                    request_id=agent_request.request_id
                )
                
                processing_time = time.time() - start_time
                
                # Update statistics
                stats["total_requests"] += 1
                stats["total_time"] += processing_time
                stats["decision_counts"][response["decision"]] += 1
                success_count += 1
                
                # Create response
                agent_response = AgentResponse(
                    request_id=response.get("request_id", agent_request.request_id or "unknown"),
                    decision=response["decision"],
                    rationale=response["rationale"],
                    confidence=response["confidence"],
                    human_required=response["human_required"],
                    task_response=response.get("task_response"),
                    processing_time=processing_time,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                results.append(agent_response)
                
            except Exception as e:
                error_count += 1
                stats["error_count"] += 1
                
                # Create error response
                error_response = AgentResponse(
                    request_id=agent_request.request_id or "unknown",
                    decision="ESCALATE",
                    rationale=f"エラーが発生しました: {str(e)}",
                    confidence=0.0,
                    human_required=True,
                    processing_time=0.0,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                results.append(error_response)
        
        total_time = time.time() - start_time
        
        # Log batch request in background
        background_tasks.add_task(
            log_batch_request,
            request,
            results,
            total_time,
            middleware
        )
        
        return BatchResponse(
            results=results,
            total_requests=len(request.requests),
            total_time=total_time,
            success_count=success_count,
            error_count=error_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/stats", response_model=StatsResponse)
async def get_statistics():
    """Get service statistics."""
    uptime = time.time() - start_time
    
    # Calculate derived statistics
    average_processing_time = 0.0
    throughput = 0.0
    error_rate = 0.0
    
    if stats["total_requests"] > 0:
        average_processing_time = stats["total_time"] / stats["total_requests"]
        throughput = stats["total_requests"] / uptime if uptime > 0 else 0
        error_rate = stats["error_count"] / stats["total_requests"]
    
    return StatsResponse(
        total_requests=stats["total_requests"],
        average_processing_time=average_processing_time,
        throughput=throughput,
        error_rate=error_rate,
        decision_counts=stats["decision_counts"].copy(),
        uptime=uptime
    )


@app.get("/agent/decisions")
async def get_decision_history(
    limit: int = 100,
    decision: Optional[str] = None,
    middleware: LoggingMiddleware = Depends(get_logging_middleware)
):
    """Get decision history."""
    try:
        # This would typically query a database
        # For now, return a placeholder response
        return {
            "message": "Decision history endpoint",
            "limit": limit,
            "decision_filter": decision,
            "note": "This would typically query a database for decision history"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/reset_stats")
async def reset_statistics():
    """Reset service statistics."""
    global stats
    stats = {
        "total_requests": 0,
        "total_time": 0.0,
        "decision_counts": {"ALLOW": 0, "REFUSE": 0, "ESCALATE": 0},
        "error_count": 0
    }
    
    return {"message": "Statistics reset successfully"}


# Background task functions
async def log_request(
    request: AgentRequest,
    response: Dict[str, Any],
    processing_time: float,
    middleware: LoggingMiddleware
):
    """Log request in background."""
    try:
        middleware.log_audit(
            request_id=response.get("request_id", request.request_id or "unknown"),
            context=request.context,
            user_request=request.user_request,
            decision=response["decision"],
            rationale=response["rationale"],
            confidence=response["confidence"],
            human_required=response["human_required"],
            processing_time_ms=processing_time * 1000,
            model_version="so8t_v1.0"
        )
    except Exception as e:
        logging.error(f"Failed to log request: {e}")


async def log_batch_request(
    request: BatchRequest,
    results: List[AgentResponse],
    total_time: float,
    middleware: LoggingMiddleware
):
    """Log batch request in background."""
    try:
        # Log batch summary
        middleware.log_audit(
            request_id=f"batch_{int(time.time() * 1000)}",
            context="Batch Processing",
            user_request=f"Processed {len(request.requests)} requests",
            decision="BATCH",
            rationale=f"Batch processing completed: {len(results)} results",
            confidence=1.0,
            human_required=False,
            processing_time_ms=total_time * 1000,
            model_version="so8t_v1.0"
        )
    except Exception as e:
        logging.error(f"Failed to log batch request: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logging.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SO8T Safe Agent API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run server
    uvicorn.run(
        "examples.api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()