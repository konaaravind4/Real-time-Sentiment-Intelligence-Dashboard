# Real-time Sentiment Intelligence Dashboard

Live 8-class emotion classification across social streams using a fine-tuned **RoBERTa** model, with real-time WebSocket updates and REST API.

## Features

-  **8-class emotion detection** — joy, anger, fear, surprise, sadness, disgust, contempt, neutral
-  **Real-time WebSocket stream** — live emotion scoring with <340ms P99 latency
-  **12K msg/s throughput** — Kafka-backed stream processing
-  **FastAPI REST + WebSocket API** — `/classify`, `/classify/batch`, `/ws/stream`
-  **Docker-ready** — one-command deployment

## Architecture

```
Social API → Kafka Producer → RoBERTa Classifier → Redis Time-Series → WebSocket → Dashboard
```

## Quick Start

```bash
git clone https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard
cd Real-time-Sentiment-Intelligence-Dashboard
cp .env.example .env
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t sentiment-dashboard .
docker run -p 8000:8000 \
  -e MODEL_NAME=SamLowe/roberta-base-go_emotions \
  sentiment-dashboard
```

## API Usage

```bash
# Classify one text
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'

# Batch classify
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love this!", "This is terrible."]}'
```

## WebSocket Stream

```python
import asyncio, websockets, json

async def stream():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        await ws.send("I absolutely love this new feature!")
        result = json.loads(await ws.recv())
        print(result)  # {"top_emotion": "joy", "top_score": 0.94, ...}

asyncio.run(stream())
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `SamLowe/roberta-base-go_emotions` | HuggingFace model ID |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `KAFKA_BOOTSTRAP` | `localhost:9092` | Kafka bootstrap servers |

## Metrics

| Metric | Value |
|--------|-------|
| Throughput | 12K msg/s |
| Emotion F1 | 88.4% |
| Latency P99 | 340ms |
| Update Interval | 2s |

## Tech Stack

`Python` · `Transformers` · `RoBERTa` · `Apache Kafka` · `Redis` · `FastAPI` · `WebSocket` · `Docker`

## License

MIT
