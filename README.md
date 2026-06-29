# French Sentiment Analysis — CamemBERT fine-tuned on Allociné

Binary sentiment classifier (positive / negative) for French movie reviews,
built by fine-tuning CamemBERT on the Allociné dataset, served via a FastAPI
REST API, containerized with Docker, tracked with MLflow, and demoed with
Streamlit.

## Results

| Epoch | train_loss | val_loss | val_accuracy |
|-------|-----------|---------|-------------|
| 1     | 0.1050    | 0.0792  | 97.38%      |
| 2     | 0.0566    | 0.0775  | 97.50%      |
| 3     | 0.0293    | 0.0888  | 97.50%      |

Best checkpoint: epoch 2 (`val_loss=0.0775`, `val_accuracy=97.50%`).

| Metric   | Score  |
|----------|--------|
| Accuracy | 97.58% |
| F1-score | 0.9758 |

*Evaluated on the Allociné test set (20 000 reviews).*

## Dataset

[Allociné](https://huggingface.co/datasets/tblard/allocine) — 200 000 French movie reviews scraped from Allociné.fr.

| Split      | Size    |
|------------|---------|
| Train      | 160 000 |
| Validation |  20 000 |
| Test       |  20 000 |

Labels: `0` = negative, `1` = positive.

## Model

- **Base model:** [`camembert-base`](https://huggingface.co/camembert/camembert-base)
- **Architecture:** CamemBERT encoder + linear classification head (768 → 2)
- **Fine-tuning:** 3 epochs, AdamW (lr=2e-5), linear lr decay

## Limitations

The model has never seen out-of-distribution input (gibberish, non-French
text, random characters) during training. On such inputs, it still outputs
a confident-looking probability for one of the two classes, since softmax
always sums to 100% — this confidence is not meaningful in that case. A
confidence threshold or a language-detection pre-check would be needed to
flag unreliable predictions in production.

## MLOps & Serving

This project goes beyond training a model — it covers the full lifecycle to
production:

- **Experiment tracking (MLflow):** parameters, per-epoch metrics, and the
  trained model are logged for every training run, replacing manual
  copy-pasting of results.
- **REST API (FastAPI):** the model is exposed via a `/predict` endpoint,
  with request/response validation handled by Pydantic. The model is loaded
  once at startup (via FastAPI's lifespan) rather than reloaded on every
  request.
- **Containerization (Docker):** the API is packaged into a portable image,
  runnable identically on any machine with Docker installed.
- **Demo (Streamlit):** a simple web UI lets anyone test the model without
  writing code or crafting HTTP requests. The demo calls the FastAPI service
  over HTTP rather than loading the model directly, keeping presentation and
  inference logic decoupled.

### API

**Endpoints:**

| Method | Path       | Description                                |
|--------|-----------|---------------------------------------------|
| GET    | `/health`  | Health check — confirms the API and model are up |
| POST   | `/predict` | Returns sentiment label, score, and class probabilities |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ce film est vraiment génial"}'
```

**Example response:**

```json
{
  "label": "POSITIVE",
  "score": 0.9959,
  "probabilities": {
    "NEGATIVE": 0.0041,
    "POSITIVE": 0.9959
  }
}
```

Interactive API docs are available at `http://localhost:8000/docs` once the
API is running.

## Project Structure

```
sentiment-camembert/
├── data.py          # dataset loading and tokenization
├── train.py         # fine-tuning pipeline, instrumented with MLflow
├── evaluate.py      # metrics on test set
├── predict.py       # quick CLI inference (reloads the model each call)
├── inference.py     # SentimentPredictor class — loads the model once,
│                     # used by the API for fast repeated inference
├── main.py          # FastAPI app exposing /health and /predict
├── app.py           # Streamlit demo, calls the FastAPI service over HTTP
├── Dockerfile        # containerizes the FastAPI service
├── .dockerignore
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/your-username/sentiment-camembert
cd sentiment-camembert
pip install -r requirements.txt
```

## Usage

**Train (logs to MLflow):**
```bash
python train.py
```

View tracked experiments:
```bash
mlflow ui
```

**Evaluate on test set:**
```bash
python evaluate.py
```

**Quick CLI prediction:**
```python
from predict import predict

result = predict("Ce film est absolument magnifique !")
print(result)  # {'label': 'POSITIVE', 'score': 0.9987}
```

**Run the API:**
```bash
uvicorn main:app --reload
```

**Run the API in Docker:**
```bash
docker build -t sentiment-camembert .
docker run -p 8000:8000 sentiment-camembert
```

**Run the Streamlit demo** (requires the API running on port 8000):
```bash
streamlit run app.py
```

## License

MIT
