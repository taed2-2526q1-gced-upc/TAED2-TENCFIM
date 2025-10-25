"""FastAPI API for emotion classification using a Hugging Face pipeline."""

from collections import Counter
from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from transformers.pipelines import pipeline
import torch

from src.api.schemas import (
    PredictionRequest,
    SingleClassResponse,
    MultiClassResponse,
    CountsResponse,
)
from src.config import MODELS_DIR, PROD_MODEL


MAP_EMOTIONS = {
    "Happiness": "Positive",
    "Love": "Positive",
    "Desire": "Positive",
    "Surprise": "Neutral",
    "Sadness": "Negative",
    "Anger": "Negative",
    "Fear": "Negative",
    "Disgust": "Negative",
    "Shame": "Negative",
    "Guilt": "Negative",
    "Sarcasm": "Negative",
    "Confusion": "Neutral",
    "Neutral": "Neutral",
}


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    FastAPI lifespan manager.

    Loads the HF pipeline at startup and clears it on shutdown,
    storing it in `app.state.pipe`.
    """
    model_path = MODELS_DIR / PROD_MODEL
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise RuntimeError(f"Model not found: {model_path}")

    device = 0 if torch.cuda.is_available() else -1
    logger.info(
        f"Loading emotion classification pipeline from {model_path} (device={device})"
    )

    fastapi_app.state.pipe = pipeline(
        task="text-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        device=device,
    )
    logger.success("Model pipeline loaded successfully.")

    try:
        yield
    finally:
        logger.info("Cleaning up pipeline from memory.")
        fastapi_app.state.pipe = None


app = FastAPI(title="Sentiment analysis from text || API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    """Root endpoint that returns a basic welcome message."""
    return {
        "message": "Welcome to the Emotion Classification API!",
        "description": (
            "This API predicts emotions from text using a fine-tuned RoBERTa model."
        ),
        "labels": (
            "The available labels are: 'Happiness', 'Sadness', 'Neutral', 'Anger', "
            "'Love', 'Fear', 'Disgust', 'Confusion', 'Surprise', 'Shame', 'Guilt', "
            "'Sarcasm' and 'Desire'."
        ),
    }


@app.get("/health",)
async def health_check(request: Request):
    """
    Health check endpoint.
    Returns basic status info, including whether the model pipeline is loaded.
    """
    pipe = getattr(request.app.state, "pipe", None)

    model_loaded = pipe is not None

    status = {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "version": "1.0.0"
    }

    return status


@app.post("/prediction/single-class", response_model=List[SingleClassResponse])
async def predict_emotion(
    request: Request, body: PredictionRequest
) -> List[SingleClassResponse]:
    """
    Predict the emotion(s) expressed in one or more text inputs.

    Returns a list of (text, label, score) for each input.
    """
    pipe = request.app.state.pipe
    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(
            status_code=503, detail="Model not loaded yet. Try again later."
        )

    try:
        if body.texts is not None:
            texts = [item.text for item in body.texts]
        else:
            raise ValueError("No text provided in the request.")
        if not texts:
            raise ValueError("No valid texts found for prediction.")

        predictions: Any = pipe(texts)
        if isinstance(predictions, dict):
            predictions = [predictions]

        response_list: List[SingleClassResponse] = []
        for text, pred in zip(texts, predictions):
            label = pred.get("label", "Unknown") if isinstance(pred, dict) else "Unknown"
            score = float(pred.get("score", 0.00)) if isinstance(pred, dict) else 0.00
            score = round(score, 2)
            response_list.append(SingleClassResponse(text=text, label=label, score=score))
        return response_list

    except ValueError as ve:
        logger.warning(f"Invalid input: {ve}")
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid input: "
                f"{ve}. Expected a JSON body with 'text' or 'texts' fields."
            ),
        ) from ve


@app.post("/prediction/multi-class", response_model=List[MultiClassResponse])
async def predict_emotion_multiclass(
    request: Request, body: PredictionRequest
) -> List[MultiClassResponse]:
    """Predict the top 3 emotions for each input text with confidence scores."""
    pipe = request.app.state.pipe
    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(
            status_code=503, detail="Model not loaded yet. Try again later."
        )

    try:
        if body.texts is not None:
            texts = [item.text for item in body.texts]
        else:
            raise ValueError("No text provided in the request.")
        if not texts:
            raise ValueError("No valid texts found for prediction.")

        predictions: Any = pipe(texts, return_all_scores=True)
        if isinstance(predictions, dict):
            predictions = [predictions]

        response_list: List[MultiClassResponse] = []
        for text, pred_list in zip(texts, predictions):
            top3 = sorted(pred_list, key=lambda x: x["score"], reverse=True)[:3]
            label_scores = {p["label"]: round(float(p["score"]), 2) for p in top3}
            response_list.append(MultiClassResponse(text=text, predictions=label_scores))
        return response_list

    except ValueError as ve:
        logger.warning(f"Invalid input: {ve}")
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid input: "
                f"{ve}. Expected a JSON body with 'text' or 'texts' fields."
            ),
        ) from ve


@app.post("/prediction/counter", response_model=CountsResponse)
async def get_counts(request: Request, body: PredictionRequest) -> CountsResponse:
    """Count the occurrences of each predicted emotion across inputs."""
    pipe = request.app.state.pipe
    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(
            status_code=503, detail="Model not loaded yet. Try again later."
        )

    try:
        if body.texts is not None:
            texts = [item.text for item in body.texts]
        else:
            raise ValueError("No text provided in the request.")
        if not texts:
            raise ValueError("No valid texts found for prediction.")

        predictions: Any = pipe(texts)
        if isinstance(predictions, dict):
            predictions = [predictions]

        labels = [p["label"] for p in predictions]
        counts = dict(Counter(labels))
        return CountsResponse(counts=counts, total=len(texts))

    except ValueError as ve:
        logger.warning(f"Invalid input: {ve}")
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid input: "
                f"{ve}. Expected a JSON body with 'text' or 'texts' fields."
            ),
        ) from ve


@app.post("/prediction/basic", response_model=List[SingleClassResponse])
async def predict_emotion_basic(
    request: Request, body: PredictionRequest
) -> List[SingleClassResponse]:
    """Predict simplified categories (Positive/Negative/Neutral) for each input."""
    pipe = request.app.state.pipe
    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(
            status_code=503, detail="Model not loaded yet. Try again later."
        )

    try:
        if body.texts is not None:
            texts = [item.text for item in body.texts]
        else:
            raise ValueError("No text provided in the request.")
        if not texts:
            raise ValueError("No valid texts found for prediction.")

        predictions: Any = pipe(texts)
        if isinstance(predictions, dict):
            predictions = [predictions]

        response_list: List[SingleClassResponse] = []
        for text, pred in zip(texts, predictions):
            original_label = pred.get("label", "Unknown") if isinstance(pred, dict) else "Unknown"
            score = float(pred.get("score", 0.0)) if isinstance(pred, dict) else 0.0
            score = round(score, 2)
            basic_label = MAP_EMOTIONS.get(original_label, "Neutral")
            response_list.append(SingleClassResponse(text=text, label=basic_label, score=score))
        return response_list

    except ValueError as ve:
        logger.warning(f"Invalid input: {ve}")
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid input: "
                f"{ve}. Expected a JSON body with 'text' or 'texts' fields."
            ),
        ) from ve


@app.post("/prediction/basic-counter", response_model=CountsResponse)
async def get_counts_basic(request: Request, body: PredictionRequest) -> CountsResponse:
    """Count occurrences of Positive/Negative/Neutral across inputs."""
    pipe = request.app.state.pipe
    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(
            status_code=503, detail="Model not loaded yet. Try again later."
        )

    try:
        if body.texts is not None:
            texts = [item.text for item in body.texts]
        else:
            raise ValueError("No text provided in the request.")
        if not texts:
            raise ValueError("No valid texts found for prediction.")

        predictions: Any = pipe(texts)
        if isinstance(predictions, dict):
            predictions = [predictions]

        labels = [MAP_EMOTIONS.get(p["label"], "Neutral") for p in predictions]
        counts = dict(Counter(labels))
        return CountsResponse(counts=counts, total=len(texts))

    except ValueError as ve:
        logger.warning(f"Invalid input: {ve}")
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid input: "
                f"{ve}. Expected a JSON body with 'text' or 'texts' fields."
            ),
        ) from ve