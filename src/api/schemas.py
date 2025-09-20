from pydantic import BaseModel, field_validator
from transformers import AutoTokenizer

from src.config import MODELS_DIR, PROD_MODEL

MAX_REVIEW_TOKENS = 512


class Review(BaseModel):
    review: str

    @field_validator("review")
    @classmethod
    def check_string_length(cls, input: str) -> str:
        """
        Validator to ensure that the input review string does not exceed the
        maximum ammount of tokens.

        Parameters
        ----------
        input : str
            The input review string to be validated.
        Returns
        -------
        str
            The validated input review string.

        Raises
        ------
        ValueError
            If the input review string exceeds the maximum number of tokens.
        """

        tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / PROD_MODEL)
        input_len = len(tokenizer.encode(input))
        # input_len = len(input)

        if input_len > MAX_REVIEW_TOKENS:
            raise ValueError(
                f"The input review exceeds with {input_len} the " + f"maximum number of {MAX_REVIEW_TOKENS} tokens."
            )

        return input


class PredictionRequest(BaseModel):
    """
    A Pydantic model that represents the input schema for the review to be predicted.

    Attributes
    -----------
        reviews list[Review]: A list of Review objects containing the review text to be predicted.

    """

    reviews: list[Review]


class PredictionResponse(BaseModel):
    """
    A Pydantic model that represents the output schema for the review prediction.

    Attributes
    -----------
        review (str): The review text provided by the user.
        label (str): The predicted sentiment label ('Positive' or 'Negative').
        score (float): The confidence score of the prediction.
    """

    review: str
    label: str
    score: float
