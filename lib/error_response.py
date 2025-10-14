from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    status_code: int
    detail: str
    error: str | None = None


def raise_http_500(detail: str, error: Exception):
    content = ErrorResponse(status_code=500, detail=detail, error=str(error)).dict()
    return JSONResponse(status_code=500, content=content)
