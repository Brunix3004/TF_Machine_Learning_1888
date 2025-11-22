import numpy as np

def prepare_text_data(text_data):
    """
    Convierte el objeto Pydantic recibido a un array 2D para XGBoost.
    """
    arr = [
        text_data.clump_thickness,
        text_data.uniformity_cell_size,
        text_data.uniformity_cell_shape,
        text_data.marginal_adhesion,
        text_data.single_epithelial_size,
        text_data.bare_nuclei,
        text_data.bland_chromatin,
        text_data.normal_nucleoli,
        text_data.mitoses
    ]
    return np.array([arr])  # shape (1, 9)
