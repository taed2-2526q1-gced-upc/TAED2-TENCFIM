from fastapi import FastAPI, HTTPException
from loguru import logger
from typing import List
from src.api.schemas import PredictionRequest, PredictionResponse
from src.services import predict_emotions

EMOTION_LABELS = [
    "Happiness", "Sadness", "Neutral", "Anger", "Love", "Fear",
    "Disgust", "Confusion", "Surprise", "Shame", "Guilt", "Sarcasm", "Desire"
]

app = FastAPI(title="Emotion Classification API", version="1.0.0")


@app.get("/")
def root():
    """
    Root endpoint returning a welcome message and available emotion labels.
    """
    return {
        "message": "Welcome to the Emotion Classification API!",
        "description": "Predict emotions from text using a fine-tuned RoBERTa model.",
        "available_labels": EMOTION_LABELS,
    }


@app.post("/prediction", response_model=List[PredictionResponse])
async def prediction(request: PredictionRequest):
    """
    Predict emotions for a list of text inputs.
    """
    texts = [s.text for s in request.texts]
    try:
        predictions = predict_emotions(texts)
        return [PredictionResponse(**pred) for pred in predictions]
    except ValueError as ve:
        logger.warning(f"Invalid input: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")