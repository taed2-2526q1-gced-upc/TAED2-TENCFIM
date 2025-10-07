from pydantic import BaseModel, field_validator
from typing import List, Union
from transformers import AutoTokenizer
from src.config import MODELS_DIR, PROD_MODEL

MAX_TEXT_TOKENS = 512


class Review(BaseModel):
    """
    Schema representing a single text input (sentence)
    to be classified by the emotion model.
    """
    text: str

    @field_validator("text")
    @classmethod
    def check_string_length(cls, input: str) -> str:
        """
        Ensure that the input text does not exceed the maximum number of tokens.

        Parameters
        ----------
        input : str
            The input text.
        Returns
        -------
        str
            The validated input string.

        Raises
        ------
        ValueError
            If the input exceeds MAX_TEXT_TOKENS after tokenization.
        """
        tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / PROD_MODEL)
        input_len = len(tokenizer.encode(input))

        if input_len > MAX_TEXT_TOKENS:
            raise ValueError(
                f"The input text contains {input_len} tokens, "
                f"which exceeds the maximum allowed ({MAX_TEXT_TOKENS})."
            )
        return input


class PredictionRequest(BaseModel):
    """
    Input schema for emotion prediction requests.

    Accepts either:
    - A single text object: {"text": "I love this!"}
    - Or a list of text objects: {"texts": ["I love this!", "This is awful"]}
    """

    text: Union[str, None] = None
    texts: Union[List[Review], None] = None

    @field_validator("texts", "text", mode="before")
    @classmethod
    def ensure_non_empty(cls, value):
        if value is None or (isinstance(value, list) and not value):
            raise ValueError("At least one text must be provided.")
        return value


class PredictionResponse(BaseModel):
    """
    Output schema for emotion classification predictions.

    Attributes
    -----------
    text (str): Original input text.
    label (str): Predicted emotion label (e.g. 'Happiness', 'Sadness', etc.).
    score (float): Model confidence score between 0 and 1.
    """

    text: str
    label: str
    score: float