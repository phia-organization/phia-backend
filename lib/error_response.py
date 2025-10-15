from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any


class ErrorResponse(BaseModel):
    status_code: int
    detail: str
    error: str | None = None


def raise_http_500(detail: Any, error: Exception):
    content = ErrorResponse(status_code=500, detail=detail, error=str(error)).dict()
    return JSONResponse(status_code=500, content=content)
