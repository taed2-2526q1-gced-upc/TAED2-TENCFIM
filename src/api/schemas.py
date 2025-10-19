from pydantic import BaseModel, field_validator, Field
from typing import List, Union, Dict
from transformers import AutoTokenizer

from src.config import MODELS_DIR, PROD_MODEL

MAX_TEXT_TOKENS = 512

class Sentence(BaseModel):
    """
    Schema representing a single text input (sentence)
    to be classified by the emotion model.
    """
    text: str = Field(..., example="Input text")

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

    Accepts a list of text objects: {"texts": ["I love this!", "This is awful"]}
    """
    
    texts: Union[List[Sentence], None] = Field(None,example=[{"text": "Input text1"}, {"text": "Input text2"}])

    @field_validator("texts", mode="before")
    @classmethod
    def ensure_non_empty(cls, value):
        if value is None or (isinstance(value, list) and not value):
            raise ValueError("At least one text must be provided.")
        return value


class SingleClassResponse(BaseModel):
    """
    Output schema for single-label emotion classification predictions.

    Attributes
    ----------
    text : str
        Original input text.
    label : str
        Predicted emotion label (e.g. 'Happiness', 'Sadness', etc.).
    score : float
        Model confidence score between 0 and 1.
    """

    text: str = Field(..., example="Input text")
    label: str = Field(..., example="Emotion predicted")
    score: float = Field(..., example=0.95)



class MultiClassResponse(BaseModel):
    """
    Output schema for multi-label emotion classification predictions.

    Attributes
    ----------
    text : str
        Original input text.
    predictions : dict[str, float]
        Dictionary mapping each predicted emotion label to its confidence score
        between 0 and 1.
    """
    text: str = Field(..., example="Input text")
    predictions: dict[str, float] = Field(
        ...,
        example={
            "Emotion1": 0.7,
            "Emotion2": 0.2,
            "Emotion3": 0.1
        }
    )


from pydantic import BaseModel, Field
from typing import Dict

class CountsResponse(BaseModel):
    """
    Output schema for emotion classification predictions counter.

    Attributes
    ----------
    counts : Dict[str, int]
        A dictionary mapping each emotion label to the number of text inputs classified under it.
    total : int
        Number of text inputs processed.
    """

    counts: Dict[str, int] = Field(
        ...,
        example={
            "Emotion1": 5,
            "Emotion2": 2,
            "Emotion3": 1,
            "Emotion4": 2
        }
    )
    total: int = Field(..., example=10)
