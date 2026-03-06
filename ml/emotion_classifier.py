"""
RoBERTa-based 8-class emotion classifier for real-time sentiment analysis.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from transformers import pipeline, Pipeline

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("SENTIMENT_MODEL", "j-hartmann/emotion-english-distilroberta-base")
EMOTIONS = ["joy", "anger", "fear", "surprise", "sadness", "disgust", "contempt", "neutral"]


@dataclass
class EmotionResult:
    text: str
    top_emotion: str
    top_score: float
    scores: dict[str, float]
    sentiment: str  # positive | negative | neutral


POSITIVE_EMOTIONS = {"joy", "surprise"}
NEGATIVE_EMOTIONS = {"anger", "fear", "sadness", "disgust", "contempt"}


class EmotionClassifier:
    """
    Fine-tuned DistilRoBERTa classifier for 8-class emotion detection.
    Wraps HuggingFace pipeline with batch processing support.
    """

    def __init__(self, model_id: str = MODEL_ID, device: int = -1):
        logger.info("Loading emotion classifier: %s", model_id)
        self.classifier: Pipeline = pipeline(
            "text-classification",
            model=model_id,
            top_k=None,
            device=device,  # -1=CPU, 0=CUDA:0
            truncation=True,
            max_length=512,
        )
        logger.info("Emotion classifier ready.")

    def classify(self, text: str) -> EmotionResult:
        """Classify a single text string."""
        results = self.classifier(text)[0]
        scores = {r["label"].lower(): round(r["score"], 4) for r in results}
        top = max(results, key=lambda x: x["score"])
        top_label = top["label"].lower()
        top_score = round(top["score"], 4)

        if top_label in POSITIVE_EMOTIONS:
            sentiment = "positive"
        elif top_label in NEGATIVE_EMOTIONS:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return EmotionResult(
            text=text[:200],
            top_emotion=top_label,
            top_score=top_score,
            scores=scores,
            sentiment=sentiment,
        )

    def classify_batch(self, texts: list[str], batch_size: int = 32) -> list[EmotionResult]:
        """Classify multiple texts efficiently with batching."""
        all_results = self.classifier(texts, batch_size=batch_size)
        output = []
        for text, results in zip(texts, all_results):
            scores = {r["label"].lower(): round(r["score"], 4) for r in results}
            top = max(results, key=lambda x: x["score"])
            top_label = top["label"].lower()
            sentiment = (
                "positive" if top_label in POSITIVE_EMOTIONS
                else "negative" if top_label in NEGATIVE_EMOTIONS
                else "neutral"
            )
            output.append(EmotionResult(
                text=text[:200],
                top_emotion=top_label,
                top_score=round(top["score"], 4),
                scores=scores,
                sentiment=sentiment,
            ))
        return output

    @staticmethod
    def emotion_color(emotion: str) -> str:
        colors = {
            "joy": "#22c55e", "surprise": "#3b82f6",
            "anger": "#ef4444", "fear": "#a855f7",
            "sadness": "#64748b", "disgust": "#f59e0b",
            "contempt": "#ec4899", "neutral": "#94a3b8",
        }
        return colors.get(emotion, "#ffffff")
