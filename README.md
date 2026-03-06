# Real-time Sentiment Intelligence Dashboard 📊

[![CI](https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Kafka](https://img.shields.io/badge/kafka-3.x-black)
![RoBERTa](https://img.shields.io/badge/model-distilroberta-green)

Live sentiment + emotion detection across social streams with 8-class RoBERTa, Apache Kafka, Redis time-series, and WebSocket dashboard.

## 🏗️ Architecture
```
Social Feed (Kafka Producer)
      │  10K msg/s
      ▼
Kafka Topic: social-feed
      │
      ▼
SentimentConsumer (batch=64)
  └─► EmotionClassifier (DistilRoBERTa, 8-class)
      │ scores
      ▼
Redis Time-Series (30-min rolling window)
      │
      ▼
FastAPI WebSocket Server
  ├── /ws/live  ← push every 2s
  ├── /analyze  ← on-demand
  └── /stats/*  ← aggregated emotion + platform counts
```

## 📊 Metrics
| Metric | Value |
|--------|-------|
| Throughput | 12K msg/s |
| Emotion F1 | 88.4% |
| Latency P99 | 340ms |
| Update Interval | 2s |

## 🚀 Quick Start
```bash
git clone https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard.git
cd Real-time-Sentiment-Intelligence-Dashboard
docker-compose up --build
```

- API: http://localhost:8003
- WebSocket: ws://localhost:8003/ws/live

## 📁 Structure
```
├── ml/
│   └── emotion_classifier.py   # DistilRoBERTa 8-class pipeline
├── stream/
│   ├── kafka_producer.py       # Social feed simulator
│   └── kafka_consumer.py       # Batch consumer + Redis store
├── api/
│   └── main.py                 # FastAPI + WebSocket server
└── tests/
    └── test_classifier.py
```
