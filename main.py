from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference import SentimentPredictor


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à analyser (avis de film en français)")


class PredictResponse(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float]


predictor: SentimentPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("Loading model...", flush=True)
    predictor = SentimentPredictor()
    print("Model loaded, API ready.", flush=True)
    yield
    print("Shutting down API.", flush=True)


app = FastAPI(
    title="French Sentiment Analysis API",
    description="API de classification de sentiment (positif/négatif) pour avis de films français, basée sur CamemBERT fine-tuné.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    result = predictor.predict(request.text)
    return result
