"""
test_api.py — Integration tests for the FastAPI backend.

Uses FastAPI's TestClient (no server needed) to test:
  - GET /            → 200
  - GET /health      → schema validation
  - GET /classes     → 4 class names
  - POST /predict    → with a real tiny image → correct response schema
  - POST /predict    → with bad file types   → 400
  - POST /predict    → with oversized file   → 413
"""
import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# We need to patch the model so tests don't require a real .pth file
# ---------------------------------------------------------------------------

import unittest.mock as mock

# Patch TumorPredictor before importing app so the lifespan doesn't crash
with mock.patch("src.inference.predict.TumorPredictor") as MockPredictor:
    from app.backend.main import app, _predictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_bytes(size=(64, 64), fmt="JPEG") -> bytes:
    """Create a minimal in-memory image and return its bytes."""
    img = Image.new("RGB", size, color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_mock_predictor():
    """Return a MagicMock that behaves like a loaded TumorPredictor."""
    pred = mock.MagicMock()
    pred.predict.return_value = {
        "class_name": "glioma",
        "class_idx": 0,
        "confidence": 0.92,
        "probabilities": [0.92, 0.04, 0.02, 0.02],
    }
    # model.model.conv_head exists (needed by gradcam auto-detect)
    pred.model.model.conv_head = mock.MagicMock()
    return pred


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with a mocked predictor injected into the app state."""
    import app.backend.main as backend_module

    mock_pred = _make_mock_predictor()

    with mock.patch.object(backend_module, "_predictor", mock_pred):
        with mock.patch(
            "app.backend.main.generate_gradcam",
            return_value=(None, Image.new("RGB", (224, 224))),
        ):
            with mock.patch(
                "app.backend.main.overlay_to_base64",
                return_value="FAKEB64",
            ):
                with TestClient(app) as c:
                    yield c


@pytest.fixture
def client_no_model():
    """TestClient where no model is loaded (predictor = None)."""
    import app.backend.main as backend_module

    with mock.patch.object(backend_module, "_predictor", None):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_get_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_message(self, client):
        resp = client.get("/")
        assert "message" in resp.json()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "classes" in data
        assert isinstance(data["classes"], list)

    def test_health_model_loaded_true(self, client):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True

    def test_health_no_model(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.json()["model_loaded"] is False


class TestClassesEndpoint:
    def test_classes_returns_list(self, client):
        resp = client.get("/classes")
        assert resp.status_code == 200
        classes = resp.json()["classes"]
        assert isinstance(classes, list)
        assert len(classes) == 4

    def test_classes_contains_expected_names(self, client):
        resp = client.get("/classes")
        classes = resp.json()["classes"]
        expected = {"glioma", "meningioma", "notumor", "pituitary"}
        assert expected == set(classes)


class TestPredictEndpoint:
    def test_predict_valid_jpeg_returns_200(self, client):
        img_bytes = _make_image_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("mri.jpg", img_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        img_bytes = _make_image_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("mri.jpg", img_bytes, "image/jpeg")},
        )
        data = resp.json()
        assert "class_name" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "heatmap_base64" in data

    def test_predict_probabilities_has_4_classes(self, client):
        img_bytes = _make_image_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("mri.jpg", img_bytes, "image/jpeg")},
        )
        probs = resp.json()["probabilities"]
        assert len(probs) == 4

    def test_predict_confidence_in_range(self, client):
        img_bytes = _make_image_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("mri.jpg", img_bytes, "image/jpeg")},
        )
        conf = resp.json()["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_predict_invalid_extension_returns_400(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("scan.pdf", b"fake pdf content", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_predict_oversized_file_returns_413(self, client):
        # Create a file > 10 MB
        big_bytes = b"0" * (11 * 1024 * 1024)
        resp = client.post(
            "/predict",
            files={"file": ("big.jpg", big_bytes, "image/jpeg")},
        )
        assert resp.status_code == 413

    def test_predict_no_model_returns_503(self, client_no_model):
        img_bytes = _make_image_bytes()
        resp = client_no_model.post(
            "/predict",
            files={"file": ("mri.jpg", img_bytes, "image/jpeg")},
        )
        assert resp.status_code == 503
