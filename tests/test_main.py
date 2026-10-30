import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# The FastAPI application is expected to be defined in the project's entry point.
# Adjust the import path if the app lives elsewhere (e.g., `app.main`).
try:
    from main import app  # type: ignore
except ImportError:  # pragma: no cover
    from app.main import app  # type: ignore

client = TestClient(app)


def _load_fixture(name: str) -> Dict[str, Any]:
    """Utility to load JSON fixtures from the ``tests/fixtures`` directory.

    Args:
        name: Name of the fixture file without extension.

    Returns:
        Parsed JSON content as a Python ``dict``.
    """
    fixture_path = Path(__file__).with_name("fixtures") / f"{name}.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture {fixture_path} does not exist.")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "endpoint,expected_status,expected_body",
    [
        ("/health", 200, {"status": "ok"}),
    ],
)
def test_get_endpoints(endpoint: str, expected_status: int, expected_body: Dict[str, Any]) -> None:
    """Basic sanity checks for simple GET endpoints.

    The health endpoint should return a JSON payload indicating the service is alive.
    """
    response = client.get(endpoint)
    assert response.status_code == expected_status, f"Unexpected status for {endpoint}"
    assert response.json() == expected_body, f"Unexpected body for {endpoint}"


def test_analyze_endpoint_success() -> None:
    """POST /analyze should return a well‑formed sentiment payload.

    The test does not assert a specific sentiment label because the underlying
    model may evolve. Instead it checks that:
    - The request succeeds (HTTP 200).
    - The response JSON contains the required keys.
    - The confidence ``score`` is a float in the range ``[-1.0, 1.0]``.
    """
    payload = {"text": "I love this product! It works wonderfully."}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200, "Analyze endpoint did not return 200 OK"

    data = response.json()
    # Expected keys
    for key in ("sentiment", "language", "score"):
        assert key in data, f"Missing key '{key}' in response"

    # Validate types
    assert isinstance(data["sentiment"], str), "Sentiment should be a string"
    assert isinstance(data["language"], str), "Language should be a string"
    assert isinstance(data["score"], float), "Score should be a float"

    # Score must be within the typical sentiment range
    assert -1.0 <= data["score"] <= 1.0, "Score out of expected range"


def test_analyze_endpoint_missing_text() -> None:
    """A request without the required ``text`` field should be rejected with 422."""
    response = client.post("/analyze", json={})
    assert response.status_code == 422, "Expected validation error for missing text"


def test_analyze_endpoint_multilingual() -> None:
    """Verify that the API can handle a non‑English input and returns a language code."""
    payload = {"text": "Este producto es increíble, me encanta!"}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200, "Multilingual request failed"

    data = response.json()
    # Language code should be a two‑letter ISO 639‑1 string
    language = data.get("language")
    assert isinstance(language, str) and len(language) == 2, "Invalid language code returned"

    # The sentiment field must still be present and correctly typed
    assert "sentiment" in data and isinstance(data["sentiment"], str)
    assert "score" in data and isinstance(data["score"], float)


def test_analyze_endpoint_batch_processing(tmp_path: Path) -> None:
    """Send a batch of texts via a temporary JSON file to emulate bulk analysis.

    The endpoint is expected to accept a JSON payload with a ``batch`` key
    containing a list of strings. The response should be a list of sentiment
    objects matching the input order.
    """
    batch = [
        "I am very happy with the service.",
        "This is the worst experience ever.",
        "It was okay, nothing special."
    ]
    payload = {"batch": batch}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200, "Batch analysis request failed"

    results = response.json()
    assert isinstance(results, list) and len(results) == len(batch), "Batch size mismatch"

    for result in results:
        for key in ("sentiment", "language", "score"):
            assert key in result, f"Missing key '{key}' in batch result"
        assert isinstance(result["sentiment"], str)
        assert isinstance(result["language"], str) and len(result["language"]) == 2
        assert isinstance(result["score"], float)
        assert -1.0 <= result["score"] <= 1.0
