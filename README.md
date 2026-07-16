# Real-time Sentiment Intelligence Dashboard 

> **Live 8-class emotion classification across social streams — now with Financial Sentiment Mode, KonaDB persistence, and Kronos market overlay integration.**

[![CI](https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![Model](https://img.shields.io/badge/model-RoBERTa--base--go__emotions-purple)](https://huggingface.co/SamLowe/roberta-base-go_emotions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard?style=social)](https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard)

Live 8-class emotion classification across social streams using a fine-tuned **RoBERTa** model, with real-time WebSocket updates, Kafka stream processing, **financial sentiment mode for market signal generation**, and **KonaDB-backed time-series persistence**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **8-class emotion detection** | joy, anger, fear, surprise, sadness, disgust, contempt, neutral |
| **Real-time WebSocket stream** | Live emotion scoring with <340ms P99 latency |
| **12K msg/s throughput** | Kafka-backed stream processing |
| **FastAPI REST + WebSocket** | `/classify`, `/classify/batch`, `/ws/stream` |
| **Financial Sentiment Mode** | Bullish/bearish/neutral market signals from financial text |
| **KonaDB Time-Series** | Persistent emotion history with windowed aggregations |
| **Kronos Integration** | Feed market sentiment signals directly into Kronos forecasts |
| **Alert System** | WebSocket alerts when extreme sentiment detected |
| **Docker-ready** | One-command deployment |

---

##  Architecture

```
Social API / News Feed
        │
        ▼
Kafka Producer (12K msg/s)
        │
        ▼
┌───────────────────────────┐
│    RoBERTa Classifier     │ ← SamLowe/roberta-base-go_emotions
│  (8-class emotion output) │
└───────────────────────────┘
        │                   │
        ▼                   ▼
Redis Time-Series    KonaDB TimeSeries (NEW)
        │                   │
        ▼                   ▼
WebSocket Stream     Kronos Integration (NEW)
        │
        ▼
Real-time Dashboard
```

### Financial Mode Pipeline (New!)

```
Financial Text (news/tweets)
        │
        ▼
FinancialSentimentAnalyzer
  ├─ Keyword scoring (bullish/bearish/neutral)
  ├─ Ticker extraction (BTC, AAPL, SPY…)
  └─ Magnitude detection (±% changes)
        │
        ▼
Signal: bullish | bearish | neutral
Score:  -1.0 to +1.0
        │
        ▼
KonaDB Storage → Kronos Overlay → Market Forecast
```

---

## Quick Start

```bash
git clone https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard
cd Real-time-Sentiment-Intelligence-Dashboard
cp .env.example .env
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t sentiment-dashboard .
docker run -p 8000:8000 \
  -e MODEL_NAME=SamLowe/roberta-base-go_emotions \
  -e KONA_DB_PATH=/data/sentiment.kona \
  -v $(pwd)/data:/data \
  sentiment-dashboard
```

---

## API Reference

### Emotion Classification

```bash
# Single text
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I cannot believe this just happened!", "store": true}'

# Response
{
  "label": "surprise",
  "score": 0.923,
  "all_emotions": {
    "surprise": 0.923, "joy": 0.042, "fear": 0.019, "neutral": 0.016,
    "anger": 0.0, "sadness": 0.0, "disgust": 0.0, "contempt": 0.0
  },
  "latency_ms": 48
}
```

```bash
# Batch classification
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love this!", "This is terrible", "Interesting news"]}'
```

### Financial Sentiment (New!)

```bash
# Analyze financial text for market signals
curl -X POST http://localhost:8000/classify/financial \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin surges 15% after SEC approves spot ETF!"}'

# Response
{
  "signal": "bullish",
  "score": 0.82,
  "confidence": 0.91,
  "tickers": ["BTC"],
  "magnitude": 15.0,
  "bullish_signals": ["surges", "approval"],
  "bearish_signals": []
}

# Market summary (last N hours)
curl http://localhost:8000/sentiment/financial/summary?hours=1

# Response
{
  "window_hours": 1.0,
  "total": 243,
  "bullish": 142,
  "bearish": 67,
  "neutral": 34,
  "bull_bear_ratio": 2.12,
  "avg_score": 0.24,
  "dominant": "bullish"
}
```

### WebSocket Real-time Stream

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/stream");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data = { label, score, text, timestamp, financial_signal? }
  updateDashboard(data);
};

// Subscribe to financial alerts only
ws.send(JSON.stringify({ filter: "financial", min_score: 0.8 }));
```

### Emotion History (KonaDB)

```bash
# Get joy trend over last 24h (1-hour buckets)
curl "http://localhost:8000/history/emotions?label=joy&hours=24&window=3600"

# Get all emotion distribution for last hour
curl "http://localhost:8000/history/distribution?hours=1"
```

---

## Emotion Classes

| Emotion | Description | Example Text |
|---------|-------------|--------------|
| **joy** | Happiness, delight, excitement | "This is the best day ever!" |
| **anger** | Frustration, rage, annoyance | "This is completely unacceptable" |
| **fear** | Anxiety, worry, dread | "I'm terrified about what comes next" |
| **surprise** | Astonishment, shock | "I can't believe this just happened!" |
| **sadness** | Grief, disappointment | "I'm devastated by this loss" |
| **disgust** | Revulsion, contempt | "This is absolutely disgusting" |
| **contempt** | Disdain, scorn | "What a pathetic excuse" |
| **neutral** | Factual, no emotion | "The meeting starts at 3pm" |

---

##  KonaDB Integration (New!)

Emotion and financial signals are automatically persisted for historical analysis:

```python
from api.kona_storage import SentimentStorage

storage = SentimentStorage("sentiment.kona")

# Manually save (automatic when using API with store=true)
storage.save_emotion(label="joy", score=0.91, text="Great earnings!", source="twitter")
storage.save_financial(signal="bullish", score=0.75, ticker="AAPL")

# Query history
joy_last_24h = storage.emotion_history("joy", hours=24)
market_mood  = storage.financial_summary(hours=1)
distribution = storage.emotion_distribution(hours=6)
```

---

##  Alert System (New!)

Get WebSocket alerts when extreme sentiment is detected:

```python
# Configure alert thresholds in .env
ALERT_FEAR_THRESHOLD=0.85      # alert when fear score > 85%
ALERT_ANGER_THRESHOLD=0.80
ALERT_BULLISH_THRESHOLD=0.90   # financial alerts
ALERT_BEARISH_THRESHOLD=0.85

# Connect to alert stream
ws = new WebSocket("ws://localhost:8000/ws/alerts");
# Receives only extreme sentiment events
```

---

##  Ecosystem Integration

```
Real-time-Sentiment-Intelligence-Dashboard
        │
        ├── Emotion stream ──────────────────────► Dashboard UI (WebSocket)
        │
        ├── Financial signals ───────────────────► Kronos Reproduction
        │   (bullish/bearish/neutral + score)         (sentiment overlay on forecasts)
        │
        ├── KonaDB Time-Series ──────────────────► kona-db
        │   (persistent emotion history)              (shared data layer)
        │
        └── REST API ────────────────────────────► AI SQL Analyst
            (/history/emotions, /summary)             (query trends in natural language)
```

**Connect with [Kronos](https://github.com/konaaravind4/kronos-reproduction)**:

```python
from api.kona_storage import SentimentStorage

# In Kronos forecasting script:
storage = SentimentStorage("sentiment.kona")
market_mood = storage.financial_summary(hours=1)

# Adjust forecast based on market sentiment
sentiment_factor = 1 + (market_mood["avg_score"] * 0.1)  # ±10% influence
adjusted_forecast = kronos_forecast * sentiment_factor
```

**Query in plain English** with [AI SQL Analyst](https://github.com/konaaravind4/AI-SQL-Data-Analyst):

```bash
curl -X POST http://ai-sql:8000/query \
  -d '{"question": "What was the average joy score during BTC price spikes last week?"}'
```

---

##  Related Projects

| Project | Integration |
|---------|-------------|
| [kona-db](https://github.com/konaaravind4/kona-db) | Emotion & financial time-series persistence |
| [kronos-reproduction](https://github.com/konaaravind4/kronos-reproduction) | Sentiment overlay on market forecasts |
| [AI-SQL-Data-Analyst](https://github.com/konaaravind4/AI-SQL-Data-Analyst) | Query emotion trends in natural language |
| [RAG-GraphRAG-Knowledge-Engine](https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine) | Contextualize emotions with current events |

---

## License

MIT © [konaaravind4](https://github.com/konaaravind4)
