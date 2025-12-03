import sys
import os

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

# Utils y schemas
from backend.schemas.predict_text_schema import TextInput
from backend.utils.text_preprocessing import prepare_text_data
from backend.utils.image_preprocessing import prepare_image

import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Cancer Detection API",
    description="API para predicción multimodal (datos clínicos + imágenes)",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend operativo para predicción MULTIMODAL 👍"}

# =======================================
#        CARGA DE MODELOS
# =======================================

text_model_path = os.path.join("backend", "models", "xgboost_breast_cancer.joblib")
image_model_path = os.path.join("backend", "models", "cnn_breast_keras_10epochs.keras")

try:
    text_model = joblib.load(text_model_path)
    print("✔ Modelo de TEXTO cargado correctamente")
except Exception as e:
    text_model = None
    print("⚠️ Error cargando el modelo de texto:", e)

try:
    image_model = load_model(image_model_path)
    print("✔ Modelo de IMAGEN cargado correctamente")
except Exception as e:
    image_model = None
    print("⚠️ Error cargando el modelo de imagen:", e)

# =======================================
#        ENDPOINT MULTIMODAL
# =======================================

@app.post("/predict/multimodal")
async def predict_multimodal(
    clump_thickness: int = Form(...),
    uniformity_cell_size: int = Form(...),
    uniformity_cell_shape: int = Form(...),
    marginal_adhesion: int = Form(...),
    single_epithelial_size: int = Form(...),
    bare_nuclei: int = Form(...),
    bland_chromatin: int = Form(...),
    normal_nucleoli: int = Form(...),
    mitoses: int = Form(...),
    file: UploadFile = File(...)
):

    if text_model is None or image_model is None:
        raise HTTPException(status_code=500, detail="Uno o ambos modelos NO están disponibles.")

    # ----------- TEXTO -----------
    text_data = TextInput(
        clump_thickness=clump_thickness,
        uniformity_cell_size=uniformity_cell_size,
        uniformity_cell_shape=uniformity_cell_shape,
        marginal_adhesion=marginal_adhesion,
        single_epithelial_size=single_epithelial_size,
        bare_nuclei=bare_nuclei,
        bland_chromatin=bland_chromatin,
        normal_nucleoli=normal_nucleoli,
        mitoses=mitoses
    )
    processed_text = prepare_text_data(text_data)
    try:
        prob_text = float(text_model.predict_proba(processed_text)[0][1])
        pred_text = 1 if prob_text >= 0.5 else 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción de texto: {e}")

    # ----------- IMAGEN -----------
    try:
        img_array = prepare_image(file.file)
        # Obtener predicción del modelo
        prediction = image_model.predict(img_array, verbose=0)
        
        # Debug: verificar la forma de la salida
        # El modelo tiene Dense(1, activation='sigmoid'), así que debería ser shape (1, 1)
        # Si la salida es (1, 1), entonces [0][0] es correcto
        # Si la salida es diferente, necesitamos ajustar el índice
        
        # Extraer probabilidad
        if prediction.ndim == 2:
            # Si es (batch, 1) o (batch, 2)
            if prediction.shape[1] == 1:
                prob_image = float(prediction[0][0])
            elif prediction.shape[1] == 2:
                # Si tiene 2 salidas, tomar la probabilidad de la clase positiva (índice 1)
                prob_image = float(prediction[0][1])
            else:
                prob_image = float(prediction[0][0])
        else:
            # Si es un array 1D
            prob_image = float(prediction[0])
        
        # Debug temporal: imprimir valores para diagnóstico
        print(f"🔍 DEBUG imagen - Shape: {prediction.shape}, Valor raw: {prediction}, Probabilidad: {prob_image}")
        
        # Asegurar que esté en el rango [0, 1]
        prob_image = max(0.0, min(1.0, prob_image))
        pred_image = 1 if prob_image >= 0.5 else 0
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Archivo subido no es una imagen válida.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción de imagen: {e}")

    # ----------- FUSIÓN -----------
    prob_final = (prob_text + prob_image) / 2
    pred_final = 1 if prob_final >= 0.5 else 0

    return {
        "text_prediction": pred_text,
        "text_probability": prob_text,
        "image_prediction": pred_image,
        "image_probability": prob_image,
        "final_prediction": pred_final,
        "final_probability": prob_final
    }
