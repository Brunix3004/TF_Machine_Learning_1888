from pydantic import BaseModel

class ImageInput(BaseModel):
    """
    Schema opcional si quisieras validar metadatos de la imagen
    (por ejemplo, tamaño original, tipo de archivo, etc.).
    Actualmente no lo usamos directamente, pero sirve para futuras mejoras.
    """
    filename: str
    content_type: str
