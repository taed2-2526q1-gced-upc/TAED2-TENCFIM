import pytest
from pathlib import Path
from datasets import load_dataset
import evaluate
from transformers import pipeline

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED

MODEL_NAME = "roberta-emotions-13"   # adapt to your trained model name
ACC_THRESHOLD = 0.70                 # realistic threshold for 13-class task


# Emotion label set of boltuix/emotions-dataset (13 classes)
LABEL_LIST = [
    "Happiness", "Sadness", "Neutral", "Anger", "Love", "Fear", "Disgust",
    "Confusion", "Surprise", "Shame", "Guilt", "Sarcasm", "Desire"
]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}


@pytest.fixture(scope="module")
def pipe():
    """Load trained text classification pipeline."""
    model_path = Path(MODELS_DIR) / MODEL_NAME
    if not model_path.exists():
        pytest.skip(f"Local model not found at {model_path}; skipping model tests.")

    # Prefer loading from local files only to avoid hub validation/download attempts
    return pipeline(
        "text-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        local_files_only=True,
    )


@pytest.fixture(scope="function")
def test_ds():
    """Load a subset of test dataset."""
    ds = (
        load_dataset(
            "parquet",
            data_files={"test": str(PROCESSED_DATA_DIR / "test.parquet")},
        )["test"]
        .shuffle(SEED)
        .select(range(min(500, len(_["test"]))))
    )
    return ds


def test_model_accuracy(pipe, test_ds):
    """
    Evaluate model accuracy on the test set.
    """
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    eval = evaluate.evaluator("text-classification")
    result = eval.compute(
        model_or_pipeline=pipe,
        data=test_ds,
        metric=[accuracy, f1],
        label_column="labels",
        label_mapping=LABEL2ID,
    )

    acc_score = result["accuracy"]
    f1_macro = result["f1_macro"]

    assert acc_score > ACC_THRESHOLD, (
        f"Model accuracy {acc_score:.2f} is below threshold {ACC_THRESHOLD}."
    )
    assert f1_macro > 0.60, f"Macro F1 {f1_macro:.2f} is too low."


@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("I am feeling so happy today!", "Happiness"),
        ("I am really sad about this.", "Sadness"),
        ("That was disgusting to watch.", "Disgust"),
        ("I love spending time with you.", "Love"),
        ("This is confusing and unclear.", "Confusion"),
    ],
)
def test_model_predictions(pipe, text, expected_label):
    """
    Check model predictions on sample sentences.
    """
    result = pipe(text)
    predicted = result[0]["label"]
    assert predicted == expected_label, (
        f"Expected {expected_label}, but got {predicted} for '{text}'."
    )
