import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app
from backend.utils.image_preprocessing import prepare_image

client = TestClient(app)

# -------------------------
# Datos de ejemplo válidos (como strings)
# -------------------------
valid_text_data = {
    "clump_thickness": "5",
    "uniformity_cell_size": "1",
    "uniformity_cell_shape": "1",
    "marginal_adhesion": "1",
    "single_epithelial_size": "2",
    "bare_nuclei": "1",
    "bland_chromatin": "3",
    "normal_nucleoli": "1",
    "mitoses": "1"
}

# -------------------------
# Imagen de prueba válida
# -------------------------
def create_test_image():
    img = Image.fromarray(np.uint8(np.random.rand(224,224,3)*255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# -------------------------
# Test endpoint multimodal válido
# -------------------------
def test_multimodal_valid():
    img_bytes = create_test_image()

    response = client.post(
        "/predict/multimodal",
        data=valid_text_data,
        files={"file": ("test.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200
    json_data = response.json()
    assert "text_prediction" in json_data
    assert "image_prediction" in json_data
    assert "final_prediction" in json_data
    assert 0.0 <= json_data["text_probability"] <= 1.0
    assert 0.0 <= json_data["image_probability"] <= 1.0
    assert 0.0 <= json_data["final_probability"] <= 1.0

# -------------------------
# Test endpoint con datos faltantes
# -------------------------
def test_multimodal_missing_field():
    incomplete_data = valid_text_data.copy()
    incomplete_data.pop("bare_nuclei")

    img_bytes = create_test_image()

    response = client.post(
        "/predict/multimodal",
        data=incomplete_data,
        files={"file": ("test.png", img_bytes, "image/png")}
    )

    assert response.status_code == 422  # validación Pydantic falla

# -------------------------
# Test endpoint con archivo inválido
# -------------------------
def test_multimodal_invalid_file():
    invalid_file = io.BytesIO(b"not an image")

    response = client.post(
        "/predict/multimodal",
        data=valid_text_data,
        files={"file": ("test.txt", invalid_file, "text/plain")}
    )

    assert response.status_code == 400  # debe fallar por imagen no válida

# -------------------------
# Test mínimo para pytest
# -------------------------
def test_dummy():
    assert 1 + 1 == 2
