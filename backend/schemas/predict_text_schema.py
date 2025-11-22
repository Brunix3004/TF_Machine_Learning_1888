from pydantic import BaseModel

class TextInput(BaseModel):
    clump_thickness: int
    uniformity_cell_size: int
    uniformity_cell_shape: int
    marginal_adhesion: int
    single_epithelial_size: int
    bare_nuclei: int
    bland_chromatin: int
    normal_nucleoli: int
    mitoses: int
