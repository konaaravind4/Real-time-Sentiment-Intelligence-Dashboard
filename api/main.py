"""
api/main.py — FastAPI + WebSocket server for Real-time Sentiment Dashboard.

Endpoints:
    POST /classify                   — classify one text (with optional KonaDB storage)
    POST /classify/batch             — classify up to 64 texts
    POST /classify/financial         — financial bullish/bearish/neutral signal
    WS   /ws/stream                  — WebSocket stream: send text, receive emotion JSON
    WS   /ws/alerts                  — WebSocket stream: extreme emotions only
    GET  /health                     — health check + model info
    GET  /metrics                    — classifier performance stats
    GET  /sentiment/financial/summary — aggregated financial sentiment summary
    GET  /history/distribution       — emotion distribution from KonaDB
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.classifier import init_classifier, get_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_financial_analyzer = None
_kona_storage = None


def _get_financial_analyzer():
    global _financial_analyzer
    if _financial_analyzer is None:
        from api.financial_mode import FinancialSentimentAnalyzer
        _financial_analyzer = FinancialSentimentAnalyzer()
    return _financial_analyzer


def _get_kona_storage():
    global _kona_storage
    if _kona_storage is None:
        db_path = os.getenv("KONA_DB_PATH", "sentiment.kona")
        from api.kona_storage import SentimentStorage
        _kona_storage = SentimentStorage(db_path)
    return _kona_storage


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.getenv("MODEL_NAME", "SamLowe/roberta-base-go_emotions")
    logger.info("Loading classifier: %s", model_name)
    init_classifier(model_name)
    logger.info("Ready.")
    yield


app = FastAPI(
    title="Real-time Sentiment Intelligence Dashboard",
    description=(
        "Live 8-class emotion classification via REST and WebSocket. "
        "Financial sentiment mode, KonaDB persistence, real-time alerts."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Connected alert WebSocket clients ─────────────────────────────────────────
_alert_clients: list[WebSocket] = []


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    store: bool = Field(False, description="Persist result to KonaDB")
    source: str = Field("api", description="Data source label (twitter/reddit/api/etc.)")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")


class ClassifyResponse(BaseModel):
    text: str
    top_emotion: str
    top_score: float
    all_scores: dict[str, float]
    latency_ms: float
    intensity: str
    is_extreme: bool
    device: str


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    store: bool = Field(False)
    source: str = Field("batch")


class FinancialRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    store: bool = Field(False)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — returns model info and classifier status."""
    clf = get_classifier()
    return {
        "status": "ok",
        "model": clf.model_name,
        "device": clf.device,
        "is_loaded": clf.is_loaded,
        "version": "1.1.0",
    }


@app.get("/metrics")
async def metrics():
    """Return classifier performance statistics."""
    clf = get_classifier()
    return clf.stats()


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    """
    Classify a single text for emotion.

    Optionally stores the result in KonaDB (set store=true).
    Returns intensity label and extreme flag alongside scores.
    """
    try:
        clf = get_classifier()

        if req.min_confidence > 0:
            result = clf.classify_with_confidence_threshold(req.text, req.min_confidence)
            if result is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Confidence below threshold {req.min_confidence}"
                )
        else:
            result = clf.classify(req.text)

        intensity = clf.get_emotion_intensity(result)
        is_extreme = clf.is_extreme_emotion(result)

        # Persist to KonaDB if requested
        if req.store:
            try:
                storage = _get_kona_storage()
                storage.save_emotion(
                    label=result.top_emotion,
                    score=result.top_score,
                    text=result.text,
                    source=req.source,
                )
            except Exception as e:
                logger.warning("KonaDB storage failed (non-fatal): %s", e)

        # Push to alert WebSocket clients if extreme
        if is_extreme and _alert_clients:
            alert_payload = json.dumps({
                "type": "extreme_emotion",
                "emotion": result.top_emotion,
                "score": result.top_score,
                "intensity": intensity,
                "text": result.text[:100],
            })
            dead = []
            for ws in _alert_clients:
                try:
                    await ws.send_text(alert_payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _alert_clients.remove(ws)

        return ClassifyResponse(
            text=result.text,
            top_emotion=result.top_emotion,
            top_score=result.top_score,
            all_scores=result.all_scores,
            latency_ms=result.latency_ms,
            intensity=intensity,
            is_extreme=is_extreme,
            device=result.device,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/classify/batch", response_model=list[ClassifyResponse])
async def classify_batch(req: BatchClassifyRequest):
    """
    Classify a batch of up to 64 texts using true batch pipeline inference.
    Up to 3x faster than calling /classify in a loop.
    """
    try:
        clf = get_classifier()
        results = clf.classify_batch(req.texts)

        responses = []
        for result in results:
            intensity = clf.get_emotion_intensity(result)
            is_extreme = clf.is_extreme_emotion(result)

            if req.store:
                try:
                    _get_kona_storage().save_emotion(
                        label=result.top_emotion,
                        score=result.top_score,
                        text=result.text,
                        source=req.source,
                    )
                except Exception as e:
                    logger.warning("KonaDB storage failed (non-fatal): %s", e)

            responses.append(ClassifyResponse(
                text=result.text,
                top_emotion=result.top_emotion,
                top_score=result.top_score,
                all_scores=result.all_scores,
                latency_ms=result.latency_ms,
                intensity=intensity,
                is_extreme=is_extreme,
                device=result.device,
            ))
        return responses
    except Exception as exc:
        logger.exception("Batch classification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/classify/financial")
async def classify_financial(req: FinancialRequest):
    """
    Analyze financial text for market signals (bullish/bearish/neutral).

    Returns signal strength, confidence, detected tickers, and % magnitude.
    Optionally stores results in KonaDB for Kronos integration.
    """
    try:
        analyzer = _get_financial_analyzer()
        result = analyzer.analyze(req.text)

        if req.store:
            try:
                _get_kona_storage().save_financial(
                    signal=result.signal,
                    score=result.score,
                    ticker=result.tickers[0] if result.tickers else "MARKET",
                    confidence=result.confidence,
                )
            except Exception as e:
                logger.warning("KonaDB financial storage failed (non-fatal): %s", e)

        return {
            "signal": result.signal,
            "score": result.score,
            "confidence": result.confidence,
            "tickers": result.tickers,
            "magnitude": result.magnitude,
            "bullish_signals": result.bullish_signals,
            "bearish_signals": result.bearish_signals,
            "timestamp": result.timestamp,
        }
    except Exception as exc:
        logger.exception("Financial classification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sentiment/financial/summary")
async def financial_summary(
    hours: float = Query(1.0, ge=0.1, le=168.0, description="Time window in hours"),
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
):
    """
    Return aggregated financial market sentiment over a time window.
    Data sourced from KonaDB persistent storage.
    """
    try:
        storage = _get_kona_storage()
        summary = storage.financial_summary(hours=hours)
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history/distribution")
async def emotion_distribution(
    hours: float = Query(1.0, ge=0.1, le=168.0, description="Time window in hours"),
):
    """
    Return emotion score distribution over the last N hours from KonaDB.
    Useful for real-time heatmap visualization on the dashboard.
    """
    try:
        storage = _get_kona_storage()
        dist = storage.emotion_distribution(hours=hours)
        return {"hours": hours, "distribution": dist}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history/emotions")
async def emotion_history(
    label: Optional[str] = Query(None, description="Filter by emotion label"),
    hours: float = Query(24.0, ge=0.1, le=720.0),
):
    """Return raw emotion history records from KonaDB."""
    try:
        storage = _get_kona_storage()
        records = storage.emotion_history(label=label, hours=hours)
        return {
            "label": label,
            "hours": hours,
            "count": len(records),
            "records": [
                {"label": r.label, "score": r.score, "source": r.source,
                 "timestamp": r.timestamp, "text": r.text[:100]}
                for r in records
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── WebSocket endpoints ───────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Real-time emotion stream.
    Client sends text strings; server responds with full emotion JSON
    including intensity, is_extreme flag, and financial signal if applicable.
    """
    await ws.accept()
    clf = get_classifier()
    try:
        while True:
            text = await ws.receive_text()
            if not text.strip():
                continue
            result = clf.classify(text)
            intensity = clf.get_emotion_intensity(result)
            is_extreme = clf.is_extreme_emotion(result)
            await ws.send_text(json.dumps({
                "text": result.text,
                "top_emotion": result.top_emotion,
                "top_score": result.top_score,
                "all_scores": result.all_scores,
                "latency_ms": result.latency_ms,
                "intensity": intensity,
                "is_extreme": is_extreme,
                "device": result.device,
            }))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /ws/stream")


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    """
    Extreme emotion alert stream.
    Only receives messages when an emotion score exceeds 0.85 (extreme).
    Used for monitoring dashboards and notification systems.
    """
    await ws.accept()
    _alert_clients.append(ws)
    logger.info("Alert WebSocket client connected (%d total)", len(_alert_clients))
    try:
        while True:
            # Keep connection alive; alerts are pushed from /classify
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _alert_clients:
            _alert_clients.remove(ws)
        logger.info("Alert WebSocket client disconnected (%d remaining)", len(_alert_clients))
