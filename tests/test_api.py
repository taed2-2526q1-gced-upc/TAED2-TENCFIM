from http import HTTPStatus
from fastapi.testclient import TestClient
import pytest

from src.api.api import app
from src.config import TESTS_DIR

client = TestClient(app)


@pytest.fixture
def valid_request():
    """Fixture: payload valid for tests."""

    return {"texts": [{"text": "I love this movie!"}]}


@pytest.fixture
def multi_texts_request():
    """Fixture: payload with multiple texts."""

    with open(TESTS_DIR / "aux_files" / "multiple-texts.txt", "r") as file:
        texts = file.readlines()
    return {"texts": [{"text": text.strip()} for text in texts]}


@pytest.fixture
def long_text():
    """Fixture: text that is too long and should exceed the token limit."""

    with open(TESTS_DIR / "aux_files" / "long-text.txt", "r") as file:
        text = file.read()
    return {"texts": [{"text": text}]}


def test_root_endpoint():
    """Checks that the root endpoint responds correctly."""

    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert "Welcome to the Emotion Classification API!" in data["message"]
    assert "labels" in data


def test_single_class_prediction(valid_request):
    """Tests the /prediction/single-class endpoint with valid text."""

    with TestClient(app) as client:
        response = client.post("/prediction/single-class", json=valid_request)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "text" in data[0]
        assert "label" in data[0]
        assert "score" in data[0]
        assert isinstance(data[0]["score"], float)


def test_single_class_prediction_invalid():
    """Tests that the /prediction/single-class endpoint fails with empty input."""

    response = client.post("/prediction/single-class", json={"texts": []})
    assert response.status_code in (HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY)


def test_single_class_prediction_long_input(long_text):
    """Tests that the schema rejects a text that is too long."""

    response = client.post("/prediction/single-class", json=long_text)
    assert response.status_code in (HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY)
    json_data = response.json()
    assert "tokens" in str(json_data) or "exceeds" in str(json_data)


def test_multi_class_prediction(valid_request):
    """Tests the /prediction/multi-class endpoint."""

    with TestClient(app) as client:
        response = client.post("/prediction/multi-class", json=valid_request)
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert isinstance(data, list)
        assert "predictions" in data[0]
        assert isinstance(data[0]["predictions"], dict)
        assert len(data[0]["predictions"]) <= 3


def test_basic_prediction(valid_request):
    """Tests the /prediction/basic endpoint."""

    with TestClient(app) as client:
        response = client.post("/prediction/basic", json=valid_request)
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert data[0]["label"] in ["Positive", "Negative", "Neutral"]


def test_basic_counter(multi_texts_request):
    """Tests the /prediction/basic-counter endpoint."""

    with TestClient(app) as client:
        response = client.post("/prediction/basic-counter", json=multi_texts_request)
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert "counts" in data
        assert "total" in data
        assert isinstance(data["counts"], dict)
        assert isinstance(data["total"], int)


def test_full_counter(multi_texts_request):
    """Tests the /prediction/counter endpoint."""
    
    with TestClient(app) as client:
        response = client.post("/prediction/counter", json=multi_texts_request)
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert "counts" in data
        assert "total" in data
        assert isinstance(data["counts"], dict)
        assert isinstance(data["total"], int)
