import sys
import os

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

# Utils y schemas
from backend.schemas.predict_text_schema import TextInput
from backend.utils.text_preprocessing import prepare_text_data
from backend.utils.image_preprocessing import prepare_image

import joblib
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Cancer Detection API",
    description="API con predicción por TEXTO e IMAGEN",
    version="2.1.0"
)

# ---------------- CORS -----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend operativo (solo texto / imagen) 👍"}


# --------------- CARGA DE MODELOS -----------------

text_model_path = os.path.join("backend", "models", "xgboost_breast_cancer.joblib")
image_model_path = os.path.join("backend", "models", "cnn_breast_keras_10epochs.keras")

try:
    text_model = joblib.load(text_model_path)
    print("✔ Modelo de TEXTO cargado correctamente")
except:
    text_model = None
    print("⚠ Error cargando modelo de texto")

try:
    image_model = load_model(image_model_path)
    print("✔ Modelo de IMAGEN cargado correctamente")
except:
    image_model = None
    print("⚠ Error cargando modelo de imagen")


# -------------------------------------------------
#           ENDPOINT: SOLO TEXTO
# -------------------------------------------------

@app.post("/predict/text")
async def predict_text_endpoint(data: TextInput):

    if text_model is None:
        raise HTTPException(status_code=500, detail="Modelo de texto no disponible.")

    try:
        processed_text = prepare_text_data(data)
        prob = float(text_model.predict_proba(processed_text)[0][1])
        pred = 1 if prob >= 0.5 else 0

        return {
            "prediction": pred,
            "probability": prob
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicción texto: {e}")


# -------------------------------------------------
#           ENDPOINT: SOLO IMAGEN
# -------------------------------------------------

@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)):

    if image_model is None:
        raise HTTPException(status_code=500, detail="Modelo de imagen no disponible.")

    try:
        img_array = prepare_image(file.file)
        prob = float(image_model.predict(img_array)[0][0])
        pred = 1 if prob >= 0.5 else 0

        return {
            "prediction": pred,
            "probability": prob
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Archivo no es una imagen válida.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicción imagen: {e}")
