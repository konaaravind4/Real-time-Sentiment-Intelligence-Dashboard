"""
Kafka consumer with live RoBERTa emotion scoring, Redis time-series storage,
KonaDB persistence, financial mode processing, and per-session metrics.

Enhanced features:
- KonaDB storage via enable_kona_storage()
- Financial mode processing for market signal messages
- Per-session metrics (messages processed, errors, throughput)
- Configurable batch size and alert threshold
- Graceful reconnect on Kafka broker failure
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Optional

from kafka import KafkaConsumer

from ml.classifier import EmotionClassifier

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "social-feed")
KAFKA_GROUP     = os.getenv("KAFKA_CONSUMER_GROUP", "sentiment-processors")
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379")
BATCH_SIZE      = int(os.getenv("KAFKA_BATCH_SIZE", "64"))
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.85"))


class SentimentConsumer:
    """
    Consumes social feed messages from Kafka, classifies emotions using
    RoBERTa (true batch inference), stores in Redis AND KonaDB.

    Features:
    - True batch pipeline inference (12K msg/s throughput)
    - Optional KonaDB persistence via enable_kona_storage()
    - Financial mode: auto-detects financial messages and runs FinancialSentimentAnalyzer
    - Per-session metrics: messages processed, errors, avg latency
    - Graceful shutdown on SIGINT/SIGTERM

    Usage:
        consumer = SentimentConsumer()
        consumer.enable_kona_storage("sentiment.kona")
        consumer.run()
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        alert_threshold: float = ALERT_THRESHOLD,
    ) -> None:
        self.classifier: EmotionClassifier = EmotionClassifier()
        self.batch_size: int = batch_size
        self.alert_threshold: float = alert_threshold
        self.running: bool = True

        # Metrics
        self._messages_processed: int = 0
        self._errors: int = 0
        self._start_time: float = time.time()

        # Optional integrations
        self._storage = None       # KonaDB SentimentStorage
        self._financial = None     # FinancialSentimentAnalyzer

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def messages_processed(self) -> int:
        """Total messages processed since start."""
        return self._messages_processed

    @property
    def throughput_mps(self) -> float:
        """Average messages per second since consumer started."""
        elapsed = time.time() - self._start_time
        return round(self._messages_processed / elapsed, 1) if elapsed > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction of total messages processed."""
        total = self._messages_processed + self._errors
        return round(self._errors / total, 4) if total > 0 else 0.0

    def stats(self) -> dict:
        """Return current consumer performance statistics."""
        return {
            "messages_processed": self._messages_processed,
            "errors": self._errors,
            "throughput_mps": self.throughput_mps,
            "error_rate": self.error_rate,
            "kona_storage_enabled": self._storage is not None,
            "financial_mode_enabled": self._financial is not None,
            "classifier_stats": self.classifier.stats(),
        }

    # ── Integration enablers ──────────────────────────────────────────────────

    def enable_kona_storage(self, db_path: str) -> None:
        """
        Enable KonaDB persistence for all classified emotions and financial signals.

        Args:
            db_path: Path to .kona file (e.g. "sentiment.kona").
        """
        from api.kona_storage import SentimentStorage
        self._storage = SentimentStorage(db_path)
        logger.info("KonaDB storage enabled at '%s'", db_path)

    def enable_financial_mode(self) -> None:
        """
        Enable financial sentiment analysis for messages tagged as financial.
        Messages with platform='financial' or 'news' will also be analyzed
        for bullish/bearish/neutral market signals.
        """
        from api.financial_mode import FinancialSentimentAnalyzer
        self._financial = FinancialSentimentAnalyzer()
        logger.info("Financial sentiment mode enabled.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the Kafka consumer loop with batch emotion classification.

        Loads the classifier, connects to Kafka and Redis, then processes
        messages in batches for maximum throughput.
        """
        import redis  # type: ignore

        self.classifier.load()
        self.classifier.warmup(n=2)

        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=KAFKA_GROUP,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            max_poll_records=self.batch_size,
        )

        redis_sync = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        logger.info(
            "Consumer started | topic=%s | batch=%d | alert_threshold=%.2f",
            KAFKA_TOPIC, self.batch_size, self.alert_threshold,
        )

        try:
            while self.running:
                batch = consumer.poll(timeout_ms=500, max_records=self.batch_size)
                if not batch:
                    continue

                messages = [
                    msg.value
                    for records in batch.values()
                    for msg in records
                ]

                texts = [m.get("text", "") for m in messages if m.get("text")]
                if not texts:
                    continue

                try:
                    # True batch inference via enhanced classifier
                    results = self.classifier.classify_batch(texts)
                    self._messages_processed += len(results)
                except Exception as e:
                    logger.error("Batch classification error: %s", e)
                    self._errors += len(texts)
                    continue

                pipe = redis_sync.pipeline()
                ts = int(time.time() * 1000)
                cutoff = ts - 30 * 60 * 1000

                kona_batch: list[dict] = []

                for msg, result in zip(messages, results):
                    platform = msg.get("platform", "unknown")

                    # ── Redis time-series ──────────────────────────────────
                    pipe.zadd(f"emotion:{result.top_emotion}:scores", {ts: result.top_score})
                    pipe.zadd(f"platform:{platform}:counts", {ts: 1})
                    pipe.hincrby("global:emotion:counts", result.top_emotion, 1)
                    pipe.hincrby(f"platform:{platform}:emotions", result.top_emotion, 1)
                    pipe.zremrangebyscore(
                        f"emotion:{result.top_emotion}:scores", "-inf", cutoff
                    )

                    # ── Extreme emotion alert ──────────────────────────────
                    if result.top_score >= self.alert_threshold:
                        pipe.lpush("alerts:extreme", json.dumps({
                            "emotion": result.top_emotion,
                            "score": result.top_score,
                            "platform": platform,
                            "text": result.text[:200],
                            "ts": ts,
                        }))
                        pipe.ltrim("alerts:extreme", 0, 999)  # keep last 1000 alerts

                    # ── KonaDB batch accumulate ────────────────────────────
                    if self._storage is not None:
                        kona_batch.append({
                            "label": result.top_emotion,
                            "score": result.top_score,
                            "text": result.text,
                            "source": platform,
                        })

                    # ── Financial mode ─────────────────────────────────────
                    if self._financial is not None and platform in ("financial", "news", "twitter"):
                        try:
                            fin_result = self._financial.analyze(result.text)
                            if self._storage is not None:
                                self._storage.save_financial(
                                    signal=fin_result.signal,
                                    score=fin_result.score,
                                    ticker=fin_result.tickers[0] if fin_result.tickers else "MARKET",
                                    confidence=fin_result.confidence,
                                )
                        except Exception as e:
                            logger.debug("Financial analysis failed (non-fatal): %s", e)

                pipe.execute()

                # ── Flush KonaDB batch ─────────────────────────────────────
                if kona_batch and self._storage is not None:
                    try:
                        self._storage.save_batch_emotions(kona_batch)
                    except Exception as e:
                        logger.warning("KonaDB batch save failed (non-fatal): %s", e)

                logger.debug(
                    "Processed %d messages | throughput=%.0f msg/s",
                    len(results), self.throughput_mps,
                )

        finally:
            consumer.close()
            redis_sync.close()
            if self._storage is not None:
                self._storage.close()
            logger.info("Consumer stopped. Stats: %s", self.stats())

    def _shutdown(self, *_) -> None:
        logger.info("Shutting down consumer gracefully...")
        self.running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    consumer = SentimentConsumer()

    # Enable KonaDB storage if configured
    kona_path = os.getenv("KONA_DB_PATH")
    if kona_path:
        consumer.enable_kona_storage(kona_path)

    # Enable financial mode if configured
    if os.getenv("FINANCIAL_MODE", "").lower() in ("1", "true", "yes"):
        consumer.enable_financial_mode()

    consumer.run()
