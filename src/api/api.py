from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import FastAPI, HTTPException
from loguru import logger
from transformers.pipelines import pipeline
import torch

from src.api.schemas import PredictionRequest, SingleClassResponse, MultiClassResponse, CountsResponse
from src.config import MODELS_DIR, PROD_MODEL

from collections import Counter

pipe = None

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
    "Neutral": "Neutral"
}

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



# Create a FastAPI instance
app = FastAPI(title="IMDB Reviews API", version="1.0.0", lifespan=lifespan)


# Root route to return basic information
@app.get("/")
def root():
    """
    Root endpoint that returns a basic welcome message.
    """
    return {
        "message": "Welcome to the Emotion Classification API!",
        "description": "This API predicts emotions from text using a fine-tuned RoBERTa model.",
        "labels": "The available labels are: 'Happiness', 'Sadness', 'Neutral', 'Anger', 'Love', 'Fear', 'Disgust', 'Confusion', 'Surprise', 'Shame', 'Guilt', 'Sarcasm' and 'Desire'."
    }


@app.post("/prediction/single-class", response_model=List[SingleClassResponse])
async def predict_emotion(request: PredictionRequest) -> List[SingleClassResponse]:
    """
    Predict the emotion(s) expressed in one or more text inputs.

    This endpoint accepts one or more text inputs and returns the predicted emotion label
    and confidence score for each input. The input texts are validated using Pydantic models, 
    and the predictions are made using a pre-trained Hugging Face pipeline loaded at app startup.

    Parameters
    ----------
    request : PredictionRequest -- The input request containing the text(s) to be predicted.
    
    The request is validated using Pydantic models to ensure the input format is correct.

    Returns
    -------
    list[SingleClassResponse] -- List of SingleClassResponse objects containing:

        text: original text input

        label: predicted emotion

        score: model confidence score (0-1)

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
        response_list: List[SingleClassResponse] = []
        for text, pred in zip(texts, predictions):
            label = pred.get("label", "Unknown") if isinstance(pred, dict) else "Unknown"
            score = float(pred.get("score", 0.00)) if isinstance(pred, dict) else 0.00
            score=round(score, 2) # show only 2 decimals
            response_list.append(SingleClassResponse(text=text, label=label, score=score))

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

@app.post("/prediction/multi-class", response_model=List[MultiClassResponse])
async def predict_emotion_multiclass(request: PredictionRequest) -> List[MultiClassResponse]:
    """
    Predict the top 3 emotions expressed in one or multiple text inputs along with their scores.

    This endpoint accepts one or more text inputs and returns the top 3 predicted emotions
    for each input, including their confidence scores. The input texts are validated using 
    Pydantic models, and the predictions are made using a pre-trained Hugging Face pipeline 
    loaded at app startup.

    Parameters
    ----------
    request : PredictionRequest -- The input request containing the text(s) to be predicted.
    
    The request is validated using Pydantic models to ensure the input format is correct.

    Returns
    -------
    list[MultiClassResponse] -- List of MultiClassResponse objects containing:

        text: the original text input

        predictions: a dictionary mapping the top 3 predicted emotion labels to their confidence scores.

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
        if request.texts is not None:
            texts = [item.text for item in request.texts]
        else:
            raise ValueError("No text provided in the request.")

        if not texts:
            raise ValueError("No valid texts found for prediction.")

        predictions: Any = pipe(texts, return_all_scores=True) # we include to return all the labels with his scores
                                                               # not only the best

        if isinstance(predictions, dict):
            predictions = [predictions]

        response_list: List[MultiClassResponse] = []
        for text, pred_list in zip(texts, predictions):
            # ordenar por score descendente y quedarse con top 3
            top3 = sorted(pred_list, key=lambda x: x['score'], reverse=True)[:3]
            label_scores = {p['label']: round(float(p['score']), 2) for p in top3}
            response_list.append(MultiClassResponse(text=text, predictions=label_scores))

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


@app.post("/prediction/counter", response_model=CountsResponse)
async def get_counts(request: PredictionRequest) -> CountsResponse:
    """
    Count the occurrences of each emotion in one or multiple text inputs.

    This endpoint accepts one or more text inputs and returns a count of predicted
    emotion labels across all inputs. The input texts are validated using Pydantic models,
    and the predictions are made using a pre-trained Hugging Face pipeline loaded at app startup.

    Parameters
    ----------
    request : PredictionRequest -- The input request containing the text(s) to be predicted.
    
    The request is validated using Pydantic models to ensure the input format is correct.

    Returns
    -------
    CountsResponse -- An object containing:

        counts: a dictionary mapping each predicted emotion to its number of occurrences.

        total: the total number of texts processed.
    
    Raises
    ------
    HTTPException

        400 - If no text is provided or if the input format is invalid.

        503 - If the model pipeline is not loaded yet.

        500 - If an unexpected error occurs during prediction.
    """
    global pipe

    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again later.")

    try:
    
        if request.texts is not None:
            texts = [item.text for item in request.texts]
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
            detail=f"Invalid input: {ve}. Expected a JSON body with 'text' or 'texts' fields."
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unexpected error during prediction: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}") from exc


@app.post("/prediction/basic", response_model=List[SingleClassResponse])
async def predict_emotion_basic(request: PredictionRequest) -> List[SingleClassResponse]:
    """
    Predict simplified emotion categories (Positive, Negative, Neutral) for one or multiple text inputs.

    This endpoint accepts one or more text inputs and returns the predicted emotion label
    and confidence score for each input. The input texts are validated using Pydantic models, 
    and the predictions are made using a pre-trained Hugging Face pipeline loaded at app startup.

    Parameters
    ----------
    request : PredictionRequest -- The input request containing the text(s) to be predicted.
    
    The request is validated using Pydantic models to ensure the input format is correct.

    Returns
    -------
    list[SingleClassResponse] -- List of SingleClassResponse objects containing:

        text: original text input

        label: predicted emotion

        score: model confidence score (0-1)

    Raises
    ------
    HTTPException

        400 - Invalid input format.

        503 - Model not loaded yet.

        500 - Unexpected server error.
    """
    global pipe, MAP_EMOTIONS

    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again later.")

    try:
        if request.texts is not None:
            texts = [item.text for item in request.texts]
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
            score=round(score, 2) # show only 2 decimals

            # map emotion predicted to postive/negative/neutral case
            basic_label = MAP_EMOTIONS.get(original_label, "Neutral")
            response_list.append(SingleClassResponse(text=text, label=basic_label, score=score))

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


@app.post("/prediction/basic-counter", response_model=CountsResponse)
async def get_counts_basic(request: PredictionRequest) -> CountsResponse:
    """
    Count the occurrences of each simplified emotion categories (Positive, Negative, Neutral) for one 
    or multiple text inputs.

    This endpoint accepts one or more text inputs and returns a count of predicted
    emotion labels across all inputs. The input texts are validated using Pydantic models,
    and the predictions are made using a pre-trained Hugging Face pipeline loaded at app startup.

    Parameters
    ----------
    request : PredictionRequest -- The input request containing the text(s) to be predicted.
    
    The request is validated using Pydantic models to ensure the input format is correct.

    Returns
    -------
    CountsResponse -- An object containing:

        counts: a dictionary mapping each predicted emotion to its number of occurrences.

        total: the total number of texts processed.
    
     Raises
    ------
    HTTPException

        400 - If no text is provided or if the input format is invalid.

        503 - If the model pipeline is not loaded yet.

        500 - If an unexpected error occurs during prediction.

    """
    global pipe, MAP_EMOTIONS

    if pipe is None:
        logger.error("Prediction requested but model pipeline is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again later.")

    try:
    
        if request.texts is not None:
            texts = [item.text for item in request.texts]
        else:
            raise ValueError("No text provided in the request.")

        if not texts:
            raise ValueError("No valid texts found for prediction.")

        
        predictions: Any = pipe(texts)
        if isinstance(predictions, dict):
            predictions = [predictions]
        
        # map emotion predicted to postive/negative/neutral case
        labels = [MAP_EMOTIONS.get(p["label"], "Neutral") for p in predictions]
        counts = dict(Counter(labels))

        return CountsResponse(counts=counts, total=len(texts))

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