"""
api/main.py — FastAPI + WebSocket server for Real-time Sentiment Dashboard.

Endpoints:
    POST /classify           — classify one text
    POST /classify/batch     — classify up to 64 texts
    WS   /ws/stream          — WebSocket stream: send text, receive emotion JSON
    GET  /health             — health check
"""

import os
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ml.classifier import init_classifier, get_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.getenv("MODEL_NAME", "SamLowe/roberta-base-go_emotions")
    logger.info("Loading classifier: %s", model_name)
    init_classifier(model_name)
    logger.info("Ready.")
    yield


app = FastAPI(
    title="Real-time Sentiment Intelligence Dashboard",
    description="Live 8-class emotion classification via REST and WebSocket.",
    version="1.0.0",
    lifespan=lifespan,
)


# Schemas

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class ClassifyResponse(BaseModel):
    text: str
    top_emotion: str
    top_score: float
    all_scores: dict[str, float]
    latency_ms: float


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


#  Endpoints 

@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("MODEL_NAME", "SamLowe/roberta-base-go_emotions")}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    try:
        clf = get_classifier()
        result = clf.classify(req.text)
        return ClassifyResponse(**result.__dict__)
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/classify/batch", response_model=list[ClassifyResponse])
async def classify_batch(req: BatchClassifyRequest):
    try:
        clf = get_classifier()
        results = clf.classify_batch(req.texts)
        return [ClassifyResponse(**r.__dict__) for r in results]
    except Exception as exc:
        logger.exception("Batch classification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    WebSocket endpoint: client sends text strings, server responds with emotion JSON.
    Disconnect to end session.
    """
    await ws.accept()
    clf = get_classifier()
    try:
        while True:
            text = await ws.receive_text()
            if not text.strip():
                continue
            result = clf.classify(text)
            await ws.send_text(json.dumps({
                "text": result.text,
                "top_emotion": result.top_emotion,
                "top_score": result.top_score,
                "all_scores": result.all_scores,
                "latency_ms": result.latency_ms,
            }))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
