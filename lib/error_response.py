from pydantic import BaseModel

class ErrorResponse(BaseModel):
    detail: str
    error: str = None  