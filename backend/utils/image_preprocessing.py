from PIL import Image, UnidentifiedImageError
import numpy as np

def prepare_image(file):
    """
    Convierte un archivo subido en un array listo para el modelo CNN.
    - Redimensiona a 224x224
    - Normaliza a valores entre 0 y 1
    - Devuelve shape (1,224,224,3)
    """
    try:
        image = Image.open(file).convert("RGB")
    except UnidentifiedImageError:
        raise UnidentifiedImageError("Archivo no es una imagen válida")
    except Exception as e:
        raise Exception(f"Error al abrir la imagen: {e}")

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
