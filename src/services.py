from pathlib import Path
from typing import List
import torch
from transformers import pipeline, AutoTokenizer
from loguru import logger
from src.config import MODELS_DIR, PROD_MODEL

MAX_TEXT_TOKENS = 512

# Load model and tokenizer at app startup
device = 0 if torch.cuda.is_available() else -1
model_path: Path = MODELS_DIR / PROD_MODEL

try:
    logger.info(f"Loading model from {model_path} on device={device}")
    pipe = pipeline("text-classification", model=str(model_path), tokenizer=str(model_path), device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.success("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")


def validate_text_length(text: str):
    """Validate that text does not exceed the maximum token length."""
    length = len(tokenizer.encode(text))
    if length > MAX_TEXT_TOKENS:
        raise ValueError(f"Text with {length} tokens exceeds the maximum allowed ({MAX_TEXT_TOKENS}).")


def predict_emotions(texts: List[str]) -> List[dict]:
    """Predict emotions for a list of texts."""
    # Validate text length
    for text in texts:
        validate_text_length(text)

    # Make predictions
    predictions = pipe(texts)
    if isinstance(predictions, dict):
        predictions = [predictions]

    # Normalize output
    results = []
    for text, pred in zip(texts, predictions):
        label = pred.get("label", "Unknown") if isinstance(pred, dict) else "Unknown"
        score = float(pred.get("score", 0.0)) if isinstance(pred, dict) else 0.0
        results.append({"text": text, "label": label, "score": score})
    return results