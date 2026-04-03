"""
ml/classifier.py — Real-time emotion classification using fine-tuned RoBERTa.
"""

import os
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "joy", "anger", "fear", "surprise",
    "sadness", "disgust", "contempt", "neutral",
]


@dataclass
class EmotionResult:
    text: str
    top_emotion: str
    top_score: float
    all_scores: dict[str, float]
    latency_ms: float


class EmotionClassifier:
    """
    Wraps a HuggingFace text-classification pipeline for 8-class emotion detection.
    Designed to be loaded once and reused across messages.
    """

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        self._model_name = model_name
        self._pipeline = None

    def load(self) -> None:
        """Load model from HuggingFace Hub. Call once at startup."""
        from transformers import pipeline  # type: ignore

        logger.info("Loading emotion classifier: %s", self._model_name)
        self._pipeline = pipeline(
            "text-classification",
            model=self._model_name,
            top_k=None,
            truncation=True,
            max_length=512,
        )
        logger.info("Classifier ready.")

    def classify(self, text: str) -> EmotionResult:
        """Classify a single text and return emotion scores."""
        if self._pipeline is None:
            raise RuntimeError("Classifier not loaded. Call .load() first.")

        t0 = time.perf_counter()
        raw: list[list[dict]] = self._pipeline(text)
        latency_ms = (time.perf_counter() - t0) * 1000

        scores_list: list[dict] = raw[0] if isinstance(raw[0], list) else raw
        all_scores = {item["label"]: round(item["score"], 4) for item in scores_list}
        top = max(scores_list, key=lambda x: x["score"])

        return EmotionResult(
            text=text,
            top_emotion=top["label"],
            top_score=round(top["score"], 4),
            all_scores=all_scores,
            latency_ms=round(latency_ms, 1),
        )

    def classify_batch(self, texts: list[str]) -> list[EmotionResult]:
        return [self.classify(t) for t in texts]


# Module-level singleton
_classifier: Optional[EmotionClassifier] = None


def get_classifier() -> EmotionClassifier:
    global _classifier
    if _classifier is None:
        raise RuntimeError("Classifier not initialised.")
    return _classifier


def init_classifier(model_name: str) -> None:
    global _classifier
    _classifier = EmotionClassifier(model_name)
    _classifier.load()
