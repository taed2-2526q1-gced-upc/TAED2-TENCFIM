from pydantic import BaseModel, field_validator
from typing import List

class Sentence(BaseModel):
    text: str

class PredictionRequest(BaseModel):
    texts: List[Sentence]

    @field_validator("texts", mode="before")
    @classmethod
    def ensure_non_empty(cls, value):
        if not value:
            raise ValueError("At least one text must be provided for prediction.")
        return value

class PredictionResponse(BaseModel):
    text: str
    label: str
    score: float