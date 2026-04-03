"""
tests/test_classifier.py — Unit tests for emotion classifier (no model download).
"""

import pytest
from unittest.mock import MagicMock


class TestEmotionResult:
    def test_fields(self):
        from ml.classifier import EmotionResult
        r = EmotionResult(
            text="I love this!",
            top_emotion="joy",
            top_score=0.95,
            all_scores={"joy": 0.95, "neutral": 0.05},
            latency_ms=12.3,
        )
        assert r.top_emotion == "joy"
        assert r.top_score == 0.95


class TestEmotionClassifier:
    def test_classify_with_mock_pipeline(self):
        from ml.classifier import EmotionClassifier

        clf = EmotionClassifier()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [[
            {"label": "joy", "score": 0.92},
            {"label": "neutral", "score": 0.05},
            {"label": "anger", "score": 0.03},
        ]]
        clf._pipeline = mock_pipeline

        result = clf.classify("This is amazing!")
        assert result.top_emotion == "joy"
        assert result.top_score == 0.92
        assert "joy" in result.all_scores

    def test_classify_raises_when_not_loaded(self):
        from ml.classifier import EmotionClassifier
        clf = EmotionClassifier()
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.classify("test")

    def test_classify_batch(self):
        from ml.classifier import EmotionClassifier

        clf = EmotionClassifier()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [[{"label": "joy", "score": 0.9}, {"label": "neutral", "score": 0.1}]]
        clf._pipeline = mock_pipeline

        results = clf.classify_batch(["text1", "text2"])
        assert len(results) == 2


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        from fastapi.testclient import TestClient
        import api.main as m
        import ml.classifier as clf_module

        mock_clf = MagicMock()
        clf_module._classifier = mock_clf

        client = TestClient(m.app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
