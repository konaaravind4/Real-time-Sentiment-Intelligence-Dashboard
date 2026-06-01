"""
KonaDB Storage Integration for Real-time Sentiment Intelligence Dashboard
=========================================================================
Persists classified emotion signals and financial sentiment data into KonaDB
using the time-series module — enabling historical trend analysis and
cross-project data sharing with Kronos and AI SQL Analyst.

Usage:
    from api.kona_storage import SentimentStorage

    storage = SentimentStorage("sentiment.kona")
    storage.save_emotion(label="joy", score=0.91, text="Great news!", source="twitter")
    storage.save_financial(signal="bullish", score=0.75, ticker="BTC")

    # Query recent data
    joy_history = storage.emotion_history("joy", hours=24)
    market_mood = storage.financial_summary(hours=1)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass


@dataclass
class EmotionRecord:
    label: str
    score: float
    text: str
    source: str
    timestamp: float


@dataclass
class FinancialRecord:
    signal: str
    score: float
    ticker: str
    confidence: float
    timestamp: float


class SentimentStorage:
    """
    KonaDB-backed storage for sentiment and emotion time-series data.
    
    Tables created automatically on first use:
    - sentiment_emotions  (8-class emotion classifier results)
    - sentiment_financial (financial signal results)
    """

    def __init__(self, db_path: str = "sentiment.kona"):
        try:
            import kona
            self.conn = kona.connect(db_path)
            self._setup_tables()
            self._available = True
        except ImportError:
            self.conn = None
            self._available = False
            print("[SentimentStorage] kona-db not installed — storage disabled. "
                  "Install with: pip install -e ../kona-db")

    def _setup_tables(self) -> None:
        """Create required tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_emotions (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                label VARCHAR(32) NOT NULL,
                score FLOAT NOT NULL,
                text TEXT,
                source VARCHAR(64),
                timestamp FLOAT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_financial (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                signal VARCHAR(16) NOT NULL,
                score FLOAT NOT NULL,
                ticker VARCHAR(16),
                confidence FLOAT,
                timestamp FLOAT NOT NULL
            )
        """)

    def save_emotion(
        self,
        label: str,
        score: float,
        text: str = "",
        source: str = "api",
    ) -> bool:
        """
        Save a single emotion classification result.

        Args:
            label: Emotion label (joy, anger, fear, surprise, sadness, disgust, contempt, neutral).
            score: Confidence score (0.0 - 1.0).
            text: Original text that was classified.
            source: Data source (twitter, reddit, api, etc.).

        Returns:
            True if saved successfully.
        """
        if not self._available:
            return False
        try:
            self.conn.execute(
                "INSERT INTO sentiment_emotions (label, score, text, source, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (label, score, text[:500], source, time.time()),
            )
            return True
        except Exception as e:
            print(f"[SentimentStorage] save_emotion failed: {e}")
            return False

    def save_batch_emotions(self, classifications: list[dict]) -> int:
        """
        Save a batch of emotion results (from batch classification endpoint).

        Args:
            classifications: List of dicts with keys: label, score, text, source.

        Returns:
            Number of records saved.
        """
        if not self._available:
            return 0
        saved = 0
        for c in classifications:
            ok = self.save_emotion(
                label=c.get("label", "neutral"),
                score=c.get("score", 0.0),
                text=c.get("text", ""),
                source=c.get("source", "batch"),
            )
            if ok:
                saved += 1
        return saved

    def save_financial(
        self,
        signal: str,
        score: float,
        ticker: str = "MARKET",
        confidence: float = 0.5,
    ) -> bool:
        """Save a financial sentiment signal."""
        if not self._available:
            return False
        try:
            self.conn.execute(
                "INSERT INTO sentiment_financial (signal, score, ticker, confidence, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (signal, score, ticker, confidence, time.time()),
            )
            return True
        except Exception as e:
            print(f"[SentimentStorage] save_financial failed: {e}")
            return False

    def emotion_history(
        self,
        label: str | None = None,
        hours: float = 24.0,
    ) -> list[EmotionRecord]:
        """
        Query emotion history for a given label over the last N hours.

        Args:
            label: Filter by emotion label. None = all labels.
            hours: Look-back window in hours.

        Returns:
            List of EmotionRecord sorted by ascending timestamp.
        """
        if not self._available:
            return []
        cutoff = time.time() - (hours * 3600)
        if label:
            rows = self.conn.execute(
                "SELECT label, score, text, source, timestamp FROM sentiment_emotions "
                "WHERE label = ? AND timestamp >= ? ORDER BY timestamp ASC",
                (label, cutoff),
            )
        else:
            rows = self.conn.execute(
                "SELECT label, score, text, source, timestamp FROM sentiment_emotions "
                "WHERE timestamp >= ? ORDER BY timestamp ASC",
                (cutoff,),
            )
        return [EmotionRecord(**r) for r in (rows or [])]

    def financial_summary(self, hours: float = 1.0) -> dict:
        """
        Aggregate financial sentiment over a time window.

        Returns a dict compatible with Kronos sentiment overlay format:
        {
            "window_hours": 1.0,
            "total": 120,
            "bullish": 72,
            "bearish": 31,
            "neutral": 17,
            "bull_bear_ratio": 2.32,
            "avg_score": 0.23,
            "dominant": "bullish",
            "timestamp": 1700000000.0
        }
        """
        if not self._available:
            return {"error": "KonaDB not available"}

        cutoff = time.time() - (hours * 3600)
        rows = self.conn.execute(
            "SELECT signal, score FROM sentiment_financial WHERE timestamp >= ?",
            (cutoff,),
        ) or []

        total = len(rows)
        if total == 0:
            return {"window_hours": hours, "total": 0, "dominant": "neutral", "avg_score": 0.0}

        signals = [r["signal"] for r in rows]
        scores  = [r["score"] for r in rows]

        bullish = signals.count("bullish")
        bearish = signals.count("bearish")
        neutral = signals.count("neutral")
        avg_score = sum(scores) / total
        bull_bear = bullish / max(bearish, 1)

        if avg_score > 0.05:
            dominant = "bullish"
        elif avg_score < -0.05:
            dominant = "bearish"
        else:
            dominant = "neutral"

        return {
            "window_hours": hours,
            "total": total,
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "bull_bear_ratio": round(bull_bear, 2),
            "avg_score": round(avg_score, 4),
            "dominant": dominant,
            "timestamp": time.time(),
        }

    def emotion_distribution(self, hours: float = 1.0) -> dict[str, float]:
        """
        Return the emotion distribution (label → average score) for the last N hours.
        Useful for the real-time dashboard heatmap.
        """
        if not self._available:
            return {}
        cutoff = time.time() - (hours * 3600)
        rows = self.conn.execute(
            "SELECT label, AVG(score) AS avg_score FROM sentiment_emotions "
            "WHERE timestamp >= ? GROUP BY label",
            (cutoff,),
        ) or []
        return {r["label"]: round(r["avg_score"], 4) for r in rows}

    def close(self) -> None:
        if self.conn:
            self.conn.close()
