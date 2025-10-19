"""Pydantic schemas for the Emotion Classification API."""

from typing import Dict, List, Union

from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer

from src.config import MODELS_DIR, PROD_MODEL

MAX_TEXT_TOKENS = 512


class Sentence(BaseModel):
    """
    Schema representing a single text input (sentence) to be classified
    by the emotion model.
    """

    text: str = Field(..., example="Input text")

    @field_validator("text")
    @classmethod
    def check_string_length(cls, value: str) -> str:
        """
        Ensure that the input text does not exceed the maximum number of tokens.
        """
        tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / PROD_MODEL)
        input_len = len(tokenizer.encode(value))

        if input_len > MAX_TEXT_TOKENS:
            raise ValueError(
                f"The input text contains {input_len} tokens, which exceeds the maximum "
                f"allowed ({MAX_TEXT_TOKENS})."
            )
        return value


class PredictionRequest(BaseModel):
    """
    Input schema for emotion prediction requests.

    Accepts a list of text objects:
    {"texts": [{"text": "I love this!"}, {"text": "This is awful"}]}
    """

    texts: Union[List[Sentence], None] = Field(
        None, example=[{"text": "Input text1"}, {"text": "Input text2"}]
    )

    @field_validator("texts", mode="before")
    @classmethod
    def ensure_non_empty(cls, value):
        """Validate that at least one text is provided."""
        if value is None or (isinstance(value, list) and not value):
            raise ValueError("At least one text must be provided.")
        return value


class SingleClassResponse(BaseModel):
    """
    Output schema for single-label emotion classification predictions.
    """

    text: str = Field(..., example="Input text")
    label: str = Field(..., example="Emotion predicted")
    score: float = Field(..., example=0.95)


class MultiClassResponse(BaseModel):
    """
    Output schema for multi-label emotion classification predictions.
    """

    text: str = Field(..., example="Input text")
    predictions: Dict[str, float] = Field(
        ..., example={"Emotion1": 0.7, "Emotion2": 0.2, "Emotion3": 0.1}
    )


class CountsResponse(BaseModel):
    """
    Output schema for emotion classification predictions counter.
    """

    counts: Dict[str, int] = Field(
        ...,
        example={"Emotion1": 5, "Emotion2": 2, "Emotion3": 1, "Emotion4": 2},
    )
    total: int = Field(..., example=10)
