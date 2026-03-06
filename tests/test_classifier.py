"""
Tests for Sentiment Dashboard — emotion classifier logic and stream consumer.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestEmotionClassifier:
    @patch("ml.emotion_classifier.pipeline")
    def test_classify_returns_emotion_result(self, mock_pipeline_fn):
        from ml.emotion_classifier import EmotionClassifier, EmotionResult

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [[
            {"label": "joy", "score": 0.85},
            {"label": "neutral", "score": 0.10},
            {"label": "anger", "score": 0.05},
        ]]
        mock_pipeline_fn.return_value = mock_pipeline

        classifier = EmotionClassifier()
        result = classifier.classify("I love this!")

        assert isinstance(result, EmotionResult)
        assert result.top_emotion == "joy"
        assert result.sentiment == "positive"
        assert result.top_score == pytest.approx(0.85)

    @patch("ml.emotion_classifier.pipeline")
    def test_classify_anger_is_negative(self, mock_pipeline_fn):
        from ml.emotion_classifier import EmotionClassifier

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [[
            {"label": "anger", "score": 0.90},
            {"label": "joy", "score": 0.05},
            {"label": "neutral", "score": 0.05},
        ]]
        mock_pipeline_fn.return_value = mock_pipeline

        classifier = EmotionClassifier()
        result = classifier.classify("This is terrible!")
        assert result.sentiment == "negative"
        assert result.top_emotion == "anger"

    @patch("ml.emotion_classifier.pipeline")
    def test_classify_batch(self, mock_pipeline_fn):
        from ml.emotion_classifier import EmotionClassifier

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            [{"label": "joy", "score": 0.9}, {"label": "neutral", "score": 0.1}],
            [{"label": "sadness", "score": 0.8}, {"label": "joy", "score": 0.2}],
        ]
        mock_pipeline_fn.return_value = mock_pipeline

        classifier = EmotionClassifier()
        results = classifier.classify_batch(["Great!", "Terrible."])
        assert len(results) == 2
        assert results[0].top_emotion == "joy"
        assert results[1].top_emotion == "sadness"

    def test_emotion_color_returns_string(self):
        from ml.emotion_classifier import EmotionClassifier
        color = EmotionClassifier.emotion_color("joy")
        assert color.startswith("#")
        unknown = EmotionClassifier.emotion_color("unknown_emotion")
        assert unknown.startswith("#")
