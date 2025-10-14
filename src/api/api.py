from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import FastAPI, HTTPException
from loguru import logger
from transformers.pipelines import pipeline
import torch

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import MODELS_DIR, PROD_MODEL

pipe = None

EMOTION_LABELS = [
    "Happiness",
    "Sadness",
    "Neutral",
    "Anger",
    "Love",
    "Fear",
    "Disgust",
    "Confusion",
    "Surprise",
    "Shame",
    "Guilt",
    "Sarcasm",
    "Desire"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the Hugging Face pipeline at startup and cleans it up on shutdown.
    """
    global pipe

    model_path = MODELS_DIR / PROD_MODEL
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise RuntimeError(f"Model not found: {model_path}")

    try:
        # Use GPU if available, otherwise fall back to CPU
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading emotion classification pipeline from {model_path} (device={device})")

        # Initialize Hugging Face text classification pipeline
        pipe = pipeline(
            task="text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=device,
        )
        logger.success("Model pipeline loaded successfully.")
    except Exception as e:
        logger.exception(f"Failed to load model pipeline: {e}")
        raise  # Prevent the app from starting with a broken model

    try:
        yield
    finally:
        # Cleanup resources when the app stops
        try:
            logger.info("Cleaning up pipeline from memory.")
            pipe = None
        except Exception:
            logger.exception("Error while cleaning up pipeline.")


# Create the FastAPI application
app = FastAPI(title="Sentiment Analysis API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    """
    Root endpoint that returns a basic welcome message.
    """
    return {
        "message": "Welcome to the Emotion Classification API!",
        "description": "This API predicts emotions from text using a fine-tuned RoBERTa model.",
        "available_labels": EMOTION_LABELS,
    }


@app.post("/prediction", response_model=List[PredictionResponse])
async def predict_emotion(request: PredictionRequest) -> List[PredictionResponse]:
    """
    Predict the emotion(s) expressed in one or more text inputs.

    Parameters
    ----------
    request : PredictionRequest
        Input request containing either a single text or a list of text objects.

    Returns
    -------
    list[PredictionResponse]
        List of objects containing:
        - `text`: original text input
        - `label`: predicted emotion
        - `score`: model confidence score (0-1)

    Raises
    ------
    HTTPException
        400 - Invalid input format.
        503 - Model not loaded yet.
        500 - Unexpected server error.
    """
    global pipe

    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again later.")

    try:
        # Extract texts depending on input
        if request.texts is not None:
            texts = [item.text for item in request.texts]
        elif request.text is not None:
            texts = [request.text]
        else:
            raise ValueError("No text provided in the request.")

        if not texts:
            raise ValueError("No valid texts found for prediction.")

        # Perform emotion prediction
        predictions: Any = pipe(texts)

        # Ensure predictions is always a list
        if isinstance(predictions, dict):
            predictions = [predictions]

        # Construct response objects
        response_list: List[PredictionResponse] = []
        for text, pred in zip(texts, predictions):
            label = pred.get("label", "Unknown") if isinstance(pred, dict) else "Unknown"
            score = float(pred.get("score", 0.0)) if isinstance(pred, dict) else 0.0
            response_list.append(PredictionResponse(text=text, label=label, score=score))

        return response_list

    except ValueError as ve:
        logger.warning(f"Invalid input: {ve}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {ve}. Expected a JSON body with 'text' or 'texts' fields."
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unexpected error during prediction: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}") from exc