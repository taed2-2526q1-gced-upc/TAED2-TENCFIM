from http import HTTPStatus

from fastapi.testclient import TestClient
import pytest

from src.api.api import app
from src.api.schemas import MAX_REVIEW_TOKENS
from src.config import TESTS_DIR


@pytest.fixture(scope="module", autouse=True)
def client():
    """Fixture that returns a TestClient for the FastAPI app."""
    # Use the TestClient with a `with` statement to trigger the lifespan events.
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def long_review():
    """Fixture that returns a payload with long review."""
    with open(TESTS_DIR / "aux_files" / "long-review.txt", "r") as file:
        review = file.read()
    return {"reviews": [{"review": review}]}


def test_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    json = response.json()
    assert response.status_code == HTTPStatus.OK
    assert json["message"] == "Welcome to the IMDB reviews app!"


def test_review_too_long(client, long_review):
    """Test that the API returns a 422 error when the review is too long."""
    response = client.post("/prediction", json=long_review)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    json = response.json()
    assert (
        json["detail"][0]["msg"] == "Value error, The input review exceeds with 898 the maximum number of 512 tokens."
    )


def test_single_review(client):
    """Test the /predict endpoint with a single review."""
    response = client.post(
        "/prediction",
        json={"reviews": [{"review": "This is a great movie!"}]},
    )
    assert response.status_code == HTTPStatus.OK
    json = response.json()
    assert json[0]["review"] == "This is a great movie!"
    assert json[0]["label"] == "positive"
    assert isinstance(json[0]["score"], float)