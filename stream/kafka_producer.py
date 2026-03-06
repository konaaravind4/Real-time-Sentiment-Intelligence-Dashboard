"""
Kafka producer — simulates social media feed ingestion.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime

from kafka import KafkaProducer

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "social-feed")

SAMPLE_TEXTS = [
    "This product is absolutely amazing, I love it!",
    "Terrible service, completely disappointed.",
    "Just had the best meal of my life!",
    "I'm really worried about the upcoming changes.",
    "Surprised by how quickly they responded.",
    "This is utterly disgusting behavior.",
    "Could care less about this honestly.",
    "The new feature is quite interesting.",
    "We're breaking records today! 🚀",
    "Awful experience, would not recommend.",
    "Feeling grateful for all the support.",
    "What a shocking announcement this morning.",
]

PLATFORMS = ["twitter", "reddit", "instagram", "linkedin"]
LANGUAGES = ["en"] * 8 + ["es", "fr", "de", "pt"]


class SocialFeedProducer:
    """
    Publishes mock social media messages to a Kafka topic.
    In production, replace with real API connectors (Twitter v2, Reddit PRAW, etc.)
    """

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP, topic: str = TOPIC):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )

    def publish(self, text: str, platform: str = "twitter", metadata: dict | None = None) -> None:
        """Publish a single message to the Kafka topic."""
        payload = {
            "id": f"{platform}-{int(time.time() * 1000)}",
            "text": text,
            "platform": platform,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.producer.send(self.topic, value=payload)

    def simulate(self, rate_per_second: float = 5.0, duration_seconds: float | None = None) -> None:
        """Simulate a live social feed at given rate for optional duration."""
        logger.info("Simulating social feed at %.1f msg/s on topic '%s'", rate_per_second, self.topic)
        t0 = time.monotonic()
        interval = 1.0 / rate_per_second
        count = 0

        while True:
            text = random.choice(SAMPLE_TEXTS)
            platform = random.choice(PLATFORMS)
            self.publish(text, platform=platform)
            count += 1

            if duration_seconds and (time.monotonic() - t0) >= duration_seconds:
                break
            time.sleep(interval)

        self.producer.flush()
        logger.info("Published %d messages.", count)

    def close(self) -> None:
        self.producer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    producer = SocialFeedProducer()
    producer.simulate(rate_per_second=10.0)
