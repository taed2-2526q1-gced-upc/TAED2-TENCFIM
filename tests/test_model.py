import pytest
from datasets import load_dataset
import evaluate
from transformers import pipeline

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED, PROD_MODEL

ACC_THRESHOLD = 0.70
TEST_SAMPLE_SIZE = 1000


EMOTIONS_LABELS = [
    "Happiness", "Sadness", "Neutral", "Anger", "Love",
    "Fear", "Disgust", "Confusion", "Surprise", "Shame",
    "Guilt", "Sarcasm", "Desire",
]


@pytest.fixture(scope="module")
def sentiment_pipeline():
    """Load the sentiment analysis pipeline once per module."""

    model_path = MODELS_DIR / PROD_MODEL
    assert model_path.exists(), f"Model does not exist at {model_path}"
    return pipeline("sentiment-analysis", str(model_path))


@pytest.fixture(scope="function")
def test_dataset():
    """Load a small, shuffled subset of the test dataset."""

    ds = load_dataset(
        "parquet",
        data_files={"test": str(PROCESSED_DATA_DIR / "test.parquet")}
    )["test"]

    ds = ds.shuffle(seed=SEED).select(range(TEST_SAMPLE_SIZE))
    assert "text" in ds.column_names and "labels" in ds.column_names, "Columns 'text' and 'labels' must be present in the dataset."

    return ds


def test_model_accuracy(sentiment_pipeline, test_dataset):
    """Test the model's accuracy on a small test subset."""

    accuracy = evaluate.load("accuracy")
    label2id = {emotion.capitalize(): i for i, emotion in enumerate(EMOTIONS_LABELS)}

    if isinstance(test_dataset["labels"][0], str):
        test_dataset = test_dataset.map(
            lambda x: {"labels": label2id.get(x["labels"].capitalize(), -1)}
        )

    evaluator = evaluate.evaluator("sentiment-analysis")
    result_acc = evaluator.compute(
        model_or_pipeline=sentiment_pipeline,
        data=test_dataset,
        metric=accuracy,
        label_column="labels",
        label_mapping=label2id,
    )

    acc_score = result_acc["accuracy"]
    assert acc_score > ACC_THRESHOLD, (
        f"Model accuracy is {acc_score:.2f}, below threshold {ACC_THRESHOLD}."
    )


# Prediction tests
@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("I am very happy", "Happiness"),
        ("I am so sad", "Sadness"),
        ("I dislike the plot and the characters.", "Disgust"),
        ("I love the plot and the characters.", "Love"),
        ("I am scared of the dark.", "Fear"),
        ("This is a neutral statement.", "Neutral"),
        ("I am furious about the delay!", "Anger"),
        ("What a surprising turn of events!", "Surprise"),
        ("I feel so guilty about what happened.", "Guilt"),
        ("This is so confusing to me.", "Confusion"),
        ("I can't believe how much I desire that.", "Desire"),
        ("Oh great, another rainy day. Just what I needed.", "Sarcasm"),
    ],
)


def test_model_predictions(sentiment_pipeline, text, expected_label):
    """Check that the model predicts the correct label for specific examples."""

    result = sentiment_pipeline(text)
    predicted_label = result[0]["label"]

    assert predicted_label == expected_label, (
        f"Expected '{expected_label}' but got '{predicted_label}' for text: '{text}'."
    )


def test_model_reproducibility(sentiment_pipeline):
    """The same input text should always yield the same predicted label."""

    text = "I am very happy with this result!"
    result1 = sentiment_pipeline(text)[0]["label"]
    result2 = sentiment_pipeline(text)[0]["label"]

    assert result1 == result2, (
        f"Predictions are not reproducible: '{result1}' vs '{result2}'"
    )


def test_batch_inference_output(sentiment_pipeline):
    """Ensure batch inference returns one prediction per input with valid labels."""

    texts = ["I love this!", "I am angry right now.", "This is confusing.", "Totally neutral."]
    results = sentiment_pipeline(texts)

    assert len(results) == len(texts), "Number of predictions doesn't match input size"
    invalid = [r["label"] for r in results if r["label"] not in EMOTIONS_LABELS]
    assert not invalid, f"Found invalid labels: {invalid}"


def test_model_handles_empty_input(sentiment_pipeline):
    """The model should handle empty input gracefully (no crash or invalid output)."""
    
    result = sentiment_pipeline("")

    assert result is not None, "Model returned None for empty input"
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 0 or all("label" in r for r in result), f"Unexpected output format: {result}"


def test_model_handles_long_input(sentiment_pipeline):
    """The model should crash when processing inputs longer than 512 tokens."""

    long_text = "This is a very long text. " * 1000
    try:
        result = sentiment_pipeline(long_text)
        assert isinstance(result, list) and len(result) == 1, "Unexpected output format for long input"
        assert "label" in result[0] and "score" in result[0], "Output missing expected keys"
    except Exception as e:
        pytest.fail(f"Model failed on long input: {e}")


def test_model_handles_special_characters(sentiment_pipeline):
    """The model should handle special characters and emojis without errors."""

    special_text = "I love this! 😍👍 #excited @friend"

    try:
        result = sentiment_pipeline(special_text)
    except Exception as e:
        pytest.fail(f"Model failed on special character input: {e}")

    assert isinstance(result, list) and len(result) == 1, "Unexpected output format for special character input"

    output = result[0]
    assert "label" in output and "score" in output, "Output missing expected keys ('label', 'score')"
    assert output["label"] in EMOTIONS_LABELS, "Predicted label is not valid"
    assert isinstance(output["score"], float), "Score must be a float"
    assert 0.0 <= output["score"] <= 1.0, "Score out of valid range [0, 1]"


def test_model_handles_non_english_input(sentiment_pipeline):
    """The model should handle non-English texts without crashing or invalid output."""

    non_english_text = "Me encuentro muy bien!"  # Spanish for "I feel really well!"
    try:
        result = sentiment_pipeline(non_english_text)
    except Exception as e:
        pytest.fail(f"Model failed on non-English input: {e}")

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 1, "Unexpected number of outputs for non-English input"

    output = result[0]
    assert "label" in output and "score" in output, "Output missing expected keys"
    assert output["label"] in EMOTIONS_LABELS, f"Predicted label is not valid: {output['label']}"
    assert isinstance(output["score"], float), "Score must be a float"
    assert 0.0 <= output["score"] <= 1.0, f"Score out of range: {output['score']}"