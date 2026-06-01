"""
ml/classifier.py — Real-time emotion classification using fine-tuned RoBERTa.

Enhanced with:
- Device auto-detection (CUDA / MPS / CPU)
- Batch pipeline inference (faster than single-item loop)
- Per-instance call metrics (latency tracking)
- Confidence thresholding
- Emotion intensity classification
- Model warmup
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

EMOTION_LABELS: list[str] = [
    "joy", "anger", "fear", "surprise",
    "sadness", "disgust", "contempt", "neutral",
]

# Intensity thresholds
_INTENSITY_THRESHOLDS = {
    "low":     (0.0,  0.40),
    "medium":  (0.40, 0.65),
    "high":    (0.65, 0.85),
    "extreme": (0.85, 1.01),
}


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class EmotionResult:
    """Result from a single emotion classification call."""
    text: str
    top_emotion: str
    top_score: float
    all_scores: dict[str, float]
    latency_ms: float
    device: str = "cpu"


class EmotionClassifier:
    """
    Wraps a HuggingFace text-classification pipeline for 8-class emotion detection.

    Features:
    - Auto-detects CUDA / MPS / CPU device
    - True batch inference via pipeline (not a loop)
    - Per-instance call count and latency tracking
    - Confidence threshold filtering
    - Emotion intensity classification (low/medium/high/extreme)
    - Model warmup to pre-JIT the pipeline

    Usage:
        clf = EmotionClassifier()
        clf.load()
        result = clf.classify("I'm so happy today!")
        print(result.top_emotion, result.top_score)
    """

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions") -> None:
        self._model_name: str = model_name
        self._pipeline = None
        self._call_count: int = 0
        self._total_latency_ms: float = 0.0
        self.device: str = _detect_device()

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        """Total number of classify() calls made."""
        return self._call_count

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if self._call_count == 0:
            return 0.0
        return round(self._total_latency_ms / self._call_count, 2)

    @property
    def model_name(self) -> str:
        """Name of the loaded HuggingFace model."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """True if the pipeline is loaded and ready."""
        return self._pipeline is not None

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load model from HuggingFace Hub. Call once at startup.
        Automatically uses CUDA/MPS/CPU based on what is available.
        """
        from transformers import pipeline  # type: ignore

        logger.info("Loading emotion classifier '%s' on device='%s'", self._model_name, self.device)
        self._pipeline = pipeline(
            "text-classification",
            model=self._model_name,
            top_k=None,
            truncation=True,
            max_length=512,
            device=0 if self.device == "cuda" else (-1 if self.device == "cpu" else self.device),
        )
        logger.info("Classifier ready (device=%s).", self.device)

    def warmup(self, n: int = 3) -> None:
        """
        Run N dummy classifications to warm up the pipeline (pre-compiles JIT etc).

        Args:
            n: Number of warmup passes (default: 3).
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier not loaded. Call .load() first.")
        logger.info("Warming up classifier (%d passes)...", n)
        dummy = "warmup text for JIT compilation"
        for _ in range(n):
            self._pipeline(dummy)
        logger.info("Warmup complete.")

    # ── Inference ───────────────────────────────────────────────────────────────

    def classify(self, text: str) -> EmotionResult:
        """
        Classify a single text string and return its emotion scores.

        Args:
            text: Input text to classify (max 512 tokens).

        Returns:
            EmotionResult with top_emotion, top_score, all_scores, latency_ms.

        Raises:
            RuntimeError: If the classifier is not loaded.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier not loaded. Call .load() first.")

        t0 = time.perf_counter()
        raw: list = self._pipeline(text)
        latency_ms = (time.perf_counter() - t0) * 1000

        self._call_count += 1
        self._total_latency_ms += latency_ms

        scores_list: list[dict] = raw[0] if isinstance(raw[0], list) else raw
        all_scores = {item["label"]: round(item["score"], 4) for item in scores_list}
        top = max(scores_list, key=lambda x: x["score"])

        return EmotionResult(
            text=text,
            top_emotion=top["label"],
            top_score=round(top["score"], 4),
            all_scores=all_scores,
            latency_ms=round(latency_ms, 1),
            device=self.device,
        )

    def classify_batch(self, texts: list[str]) -> list[EmotionResult]:
        """
        Classify a batch of texts using true batch pipeline inference.

        This is significantly faster than calling classify() in a loop
        because it leverages the HuggingFace pipeline's internal batching.

        Args:
            texts: List of input texts (each max 512 tokens).

        Returns:
            List of EmotionResult, one per input text, in the same order.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier not loaded. Call .load() first.")
        if not texts:
            return []

        t0 = time.perf_counter()
        # True batch inference — pipeline processes all texts in one GPU pass
        raw_batch: list = self._pipeline(texts, batch_size=min(len(texts), 64))
        batch_latency_ms = (time.perf_counter() - t0) * 1000

        results: list[EmotionResult] = []
        per_item_ms = round(batch_latency_ms / len(texts), 1)

        for text, raw in zip(texts, raw_batch):
            self._call_count += 1
            self._total_latency_ms += per_item_ms

            scores_list: list[dict] = raw if isinstance(raw, list) else [raw]
            all_scores = {item["label"]: round(item["score"], 4) for item in scores_list}
            top = max(scores_list, key=lambda x: x["score"])

            results.append(EmotionResult(
                text=text,
                top_emotion=top["label"],
                top_score=round(top["score"], 4),
                all_scores=all_scores,
                latency_ms=per_item_ms,
                device=self.device,
            ))

        return results

    # ── Advanced helpers ────────────────────────────────────────────────────────

    def classify_with_confidence_threshold(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> EmotionResult | None:
        """
        Classify text and return None if top confidence is below the threshold.

        Useful for filtering ambiguous/neutral text that shouldn't trigger alerts.

        Args:
            text: Input text.
            min_confidence: Minimum top_score to return a result (default: 0.5).

        Returns:
            EmotionResult if top_score >= min_confidence, else None.
        """
        result = self.classify(text)
        return result if result.top_score >= min_confidence else None

    def is_extreme_emotion(
        self,
        result: EmotionResult,
        threshold: float = 0.85,
    ) -> bool:
        """
        Check if the result contains an extreme emotion signal.

        An emotion is "extreme" if its score exceeds the threshold.
        Useful for triggering real-time alerts.

        Args:
            result: Classification result.
            threshold: Score threshold for extreme emotion (default: 0.85).

        Returns:
            True if any emotion score exceeds the threshold.
        """
        return any(score >= threshold for score in result.all_scores.values())

    def get_emotion_intensity(self, result: EmotionResult) -> str:
        """
        Classify the intensity of the top emotion as a human-readable label.

        Args:
            result: Classification result.

        Returns:
            'low' (0-40%), 'medium' (40-65%), 'high' (65-85%), 'extreme' (85-100%)
        """
        score = result.top_score
        for label, (low, high) in _INTENSITY_THRESHOLDS.items():
            if low <= score < high:
                return label
        return "extreme"

    def stats(self) -> dict:
        """
        Return classifier performance statistics.

        Returns:
            Dict with call_count, avg_latency_ms, model_name, device.
        """
        return {
            "call_count": self._call_count,
            "avg_latency_ms": self.avg_latency_ms,
            "total_latency_ms": round(self._total_latency_ms, 1),
            "model_name": self._model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
        }

    def reset_stats(self) -> None:
        """Reset call count and latency counters."""
        self._call_count = 0
        self._total_latency_ms = 0.0


# ── Module-level singleton ──────────────────────────────────────────────────────

_classifier: Optional[EmotionClassifier] = None


def get_classifier() -> EmotionClassifier:
    """Get the module-level singleton classifier."""
    global _classifier
    if _classifier is None:
        raise RuntimeError("Classifier not initialised. Call init_classifier() first.")
    return _classifier


def init_classifier(model_name: str) -> None:
    """Initialise and load the module-level classifier singleton."""
    global _classifier
    _classifier = EmotionClassifier(model_name)
    _classifier.load()
    _classifier.warmup(n=2)
