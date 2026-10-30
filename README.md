# Real-Time Adaptive Multilingual Sentiment Analyzer

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Endpoints](#api-endpoints)
- [Streaming Example](#streaming-example)
- [Testing](#testing)
- [Performance Monitoring](#performance-monitoring)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
The **Real-Time Adaptive Multilingual Sentiment Analyzer** is a production‑ready system that ingests text streams (e.g., social media posts, chat messages, or customer reviews), detects the language on‑the‑fly, and returns sentiment scores (positive, neutral, negative) in real time.  

Key aspects:
- **Multilingual**: Supports over 30 languages using Hugging Face transformer models.
- **Adaptive Learning**: Continuously updates the underlying model with River’s incremental learning algorithms to adapt to concept drift.
- **Low‑Latency API**: Exposes a FastAPI endpoint that can handle thousands of requests per second when run behind Uvicorn/Gunicorn.
- **Scalable**: Stateless design enables horizontal scaling behind a load balancer.

---

## Features
- **Language Detection** – Fast, lightweight detection using `langdetect`.
- **Transformer‑based Sentiment** – Pre‑trained multilingual BERT models fine‑tuned for sentiment.
- **Online Model Updates** – River’s `HoeffdingTreeClassifier` updates with each labeled request.
- **Batch & Streaming Modes** – Process single messages or bulk CSV/JSON payloads.
- **Metrics & Logging** – Integrated with Prometheus metrics and structured JSON logs.
- **Docker Support** – Official Dockerfile for containerized deployment.

---

## Architecture
```
┌─────────────────────┐
│   FastAPI (Uvicorn) │
│   /sentiment endpoint│
└───────┬─────────────┘
        │
        ▼
┌─────────────────────┐   ┌─────────────────────┐
│ Language Detector   │   │ Sentiment Model      │
│ (langdetect)        │   │ (transformers)       │
└───────┬─────────────┘   └───────┬─────────────┘
        │                     │
        ▼                     ▼
┌─────────────────────┐   ┌─────────────────────┐
│ River Online Learner│   │ Metrics & Logging    │
│ (HoeffdingTree)     │   │ (Prometheus, Loguru)│
└─────────────────────┘   └─────────────────────┘
```

---

## Tech Stack
- **Python 3.11+**
- **FastAPI** – high‑performance API framework
- **Uvicorn** – ASGI server
- **Transformers** – Hugging Face BERT models
- **River** – online machine learning
- **Pandas** – data manipulation for batch jobs
- **scikit‑learn** – utilities for preprocessing & evaluation
- **langdetect** – language identification
- **Prometheus client** – metrics exposition
- **Loguru** – structured logging
- **Docker** – containerization

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourorg/real-time-adaptive-multilingual-sentiment-analyzer.git
cd real-time-adaptive-multilingual-sentiment-analyzer
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Install the optional GPU version of Transformers
If you have a CUDA‑enabled GPU:
```bash
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## Configuration
Configuration is managed via environment variables. Create a `.env` file in the project root (or export variables in your deployment environment).

| Variable               | Description                                    | Default |
|------------------------|------------------------------------------------|---------|
| `HOST`                 | Host address for Uvicorn                       | `0.0.0.0` |
| `PORT`                 | Port for the API server                        | `8000` |
| `MODEL_NAME`           | Hugging Face model identifier (e.g., `nlptown/bert-base-multilingual-uncased-sentiment`) | `nlptown/bert-base-multilingual-uncased-sentiment` |
| `MAX_BATCH_SIZE`       | Maximum number of records per batch request    | `128` |
| `LOG_LEVEL`            | Loguru logging level (`DEBUG`, `INFO`, …)     | `INFO` |
| `PROMETHEUS_PORT`      | Port for Prometheus metrics endpoint           | `8001` |
| `RIVER_LEARNING_RATE`  | Learning rate for River online learner         | `0.01` |

Load the variables automatically with `python-dotenv` (already included in `requirements.txt`).

```bash
export HOST=0.0.0.0
export PORT=8000
export MODEL_NAME=nlptown/bert-base-multilingual-uncased-sentiment
```

---

## Running the Service

### Development mode
```bash
uvicorn app.main:app --reload --host $HOST --port $PORT
```

### Production mode (recommended)
```bash
gunicorn app.main:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind ${HOST}:${PORT} \
  --log-level $LOG_LEVEL
```

### Docker
```bash
docker build -t sentiment-analyzer .
docker run -d -p 8000:8000 -p 8001:8001 --env-file .env sentiment-analyzer
```

---

## API Endpoints

| Method | Path                | Description                                 | Request Body                                    | Response |
|--------|---------------------|---------------------------------------------|-------------------------------------------------|----------|
| `POST` | `/sentiment`        | Analyze a single text message               | `{ "text": "Your message", "label": "positive" (optional) }` | `{ "language": "en", "sentiment": "positive", "score": 0.92, "model_version": "v1.0.0" }` |
| `POST` | `/sentiment/batch`  | Analyze a batch of messages (JSON array)    | `[ { "text": "...", "label": "...?" }, ... ]`   | `{ "results": [ ... ], "batch_id": "uuid" }` |
| `GET`  | `/metrics`          | Prometheus metrics endpoint                 | –                                               | Prometheus‑compatible text |
| `GET`  | `/health`           | Liveness probe                              | –                                               | `{ "status": "ok" }` |

### Request example (single)
```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this product!"}'
```

### Request example (batch)
```bash
curl -X POST http://localhost:8000/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '[
        {"text":"I love this product!"},
        {"text":"Este producto es terrible."},
        {"text":"C’est un bon service."}
      ]'
```

---

## Streaming Example
Below is a minimal Python client that streams tweets (or any async source) to the API and prints live sentiment updates.

```python
import asyncio
import httpx
import json

API_URL = "http://localhost:8000/sentiment"

async def stream_messages(messages):
    async with httpx.AsyncClient() as client:
        for msg in messages:
            payload = {"text": msg}
            resp = await client.post(API_URL, json=payload, timeout=10.0)
            data = resp.json()
            print(f"[{data['language']}] {msg} → {data['sentiment']} ({data['score']:.2f})")
            await asyncio.sleep(0.1)  # simulate real‑time pacing

if __name__ == "__main__":
    sample = [
        "I love the new update!",
        "Este es un desastre total.",
        "C’est incroyable!",
        "Das ist nicht gut."
    ]
    asyncio.run(stream_messages(sample))
```

---

## Testing
The project includes a comprehensive test suite using `pytest`. Tests cover language detection, transformer inference, River online updates, and API contract validation.

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run the test suite
pytest -vv
```

### Sample test (located in `tests/test_sentiment.py`)
```python
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_single_sentiment():
    response = client.post(
        "/sentiment",
        json={"text": "I absolutely adore this!"},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["language"] == "en"
    assert payload["sentiment"] in {"positive", "neutral", "negative"}
    assert 0.0 <= payload["score"] <= 1.0
```

All tests should pass on Python 3.11+ with the default `requirements.txt`.

---

## Performance Monitoring
- **Prometheus**: The `/metrics` endpoint exposes request latency, error counters, and model update rates.
- **Grafana Dashboard**: A sample Grafana JSON dashboard is provided in `monitoring/grafana-dashboard.json`.
- **Logging**: Structured logs (JSON) are emitted via Loguru. Configure log aggregation (e.g., Loki, ELK) by setting `LOG_LEVEL` and `LOG_OUTPUT` environment variables.

---

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Write code adhering to the existing style (PEP 8, type hints, docstrings).
4. Add or update tests.
5. Ensure all tests pass (`pytest -q`).
6. Open a Pull Request with a clear description of the changes.

Please see `CONTRIBUTING.md` for detailed guidelines, code of conduct, and branch naming conventions.

---

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for full terms.