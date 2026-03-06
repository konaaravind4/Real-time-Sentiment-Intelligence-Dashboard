"""
FastAPI + WebSocket server for real-time sentiment dashboard.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.emotion_classifier import EmotionClassifier, EMOTIONS

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
WS_PUSH_INTERVAL = float(os.getenv("WS_PUSH_INTERVAL", "2.0"))  # seconds

redis_client: Optional[aioredis.Redis] = None
classifier: Optional[EmotionClassifier] = None
connected_clients: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, classifier
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    classifier = EmotionClassifier()
    asyncio.create_task(broadcast_loop())
    yield
    await redis_client.close()


app = FastAPI(title="Sentiment Intelligence Dashboard API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── REST Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "clients": len(connected_clients)}


@app.get("/stats/emotions")
async def emotion_stats() -> dict:
    """Return current emotion counts from Redis."""
    counts = await redis_client.hgetall("global:emotion:counts")
    total = sum(int(v) for v in counts.values()) or 1
    return {
        emotion: {
            "count": int(counts.get(emotion, 0)),
            "percentage": round(int(counts.get(emotion, 0)) / total * 100, 1),
        }
        for emotion in EMOTIONS
    }


@app.get("/stats/platforms")
async def platform_stats() -> dict:
    platforms = ["twitter", "reddit", "instagram", "linkedin"]
    result = {}
    for platform in platforms:
        emotions = await redis_client.hgetall(f"platform:{platform}:emotions")
        result[platform] = {k: int(v) for k, v in emotions.items()}
    return result


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/analyze")
async def analyze_text(req: AnalyzeRequest) -> dict:
    """Classify a single text on demand."""
    result = classifier.classify(req.text)
    return {
        "top_emotion": result.top_emotion,
        "top_score": result.top_score,
        "sentiment": result.sentiment,
        "scores": result.scores,
    }


# ── WebSocket live feed ───────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info("WS client connected. Total: %d", len(connected_clients))
    try:
        while True:
            await asyncio.sleep(WS_PUSH_INTERVAL)
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        logger.info("WS client disconnected. Total: %d", len(connected_clients))


async def broadcast_loop() -> None:
    """Push emotion stats to all connected WebSocket clients every WS_PUSH_INTERVAL."""
    while True:
        await asyncio.sleep(WS_PUSH_INTERVAL)
        if not connected_clients:
            continue
        try:
            counts = await redis_client.hgetall("global:emotion:counts")
            total = sum(int(v) for v in counts.values()) or 1
            payload = json.dumps({
                "type": "emotion_update",
                "timestamp": time.time(),
                "emotions": {
                    e: {
                        "count": int(counts.get(e, 0)),
                        "pct": round(int(counts.get(e, 0)) / total * 100, 1),
                    }
                    for e in EMOTIONS
                },
            })
            dead = set()
            for ws in connected_clients.copy():
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            connected_clients -= dead
        except Exception as e:
            logger.warning("Broadcast error: %s", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=False)
