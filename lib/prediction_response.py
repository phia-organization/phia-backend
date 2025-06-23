from pydantic import BaseModel
import uuid

class RGB(BaseModel):
    r: int
    g: int
    b: int

class RGBAreas(BaseModel):
    Q1: RGB
    Q2: RGB
    Q3: RGB
    Q4: RGB

class PredictionResponse(BaseModel):
    predicted_ph: float
    identifier: uuid.UUID
    rgbs: RGBAreas