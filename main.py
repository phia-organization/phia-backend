import cv2
import numpy as np
import time

from lib import (
    PingResponse,
    ErrorResponse,
    PredictionResponse,
    raise_http_500,
    MAX_IMAGE_SIZE_MB,
)
from classes import PredictionClass

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import imghdr

app = FastAPI()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

prediction_class = PredictionClass()
y_measures_cm = {"Q1": 0.7, "Q2": 1.8, "Q3": 3.0, "Q4": 4.2, "total": 12.6}


@app.get("/health-check", response_model=PingResponse)
def read_root() -> PingResponse:
    """
    A simple endpoint that returns a greeting.
    Returns:
        dict: A dictionary with a greeting message.
    """
    return PingResponse(
        ping="pong",
    )


@app.post(
    "/predict",
    responses={400: {"model": ErrorResponse}, 200: {"model": PredictionResponse}},
)
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and predict pH strip value.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    filename = file.filename.lower()
    extension = filename.split(".")[-1]
    dir_identifier = str(time.time()).replace(".", "-")

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "description": "Invalid file type. Only PNG and JPEG images are allowed.",
                "identifier": dir_identifier,
            },
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail={
                "description": "File too large. Maximum allowed size is 10 MB.",
                "identifier": dir_identifier,
            },
        )

    if imghdr.what(None, contents) not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "description": "Uploaded file is not a valid image.",
                "identifier": dir_identifier,
            },
        )

    image_np = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail={
                "description": "Failed to decode the image. Ensure it's a valid image file.",
                "identifier": dir_identifier,
            },
        )

    prediction_class.set_file_info(filename, extension, dir_identifier)

    try:
        strip_withouth_bg = prediction_class.remove_background(image)
    except Exception as e:
        return raise_http_500(
            {"description": "Background removal failed", "identifier": dir_identifier},
            e,
        )

    try:
        cropped_strip = prediction_class.crop(strip_withouth_bg)
        cropped_strip = cv2.cvtColor(cropped_strip, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return raise_http_500(
            {"description": "Cropping the strip failed", "identifier": dir_identifier},
            e,
        )

    try:
        median_rgbs = prediction_class.extract_median_rgbs(cropped_strip)
    except Exception as e:
        return raise_http_500(
            {"description": "RGB extraction failed", "identifier": dir_identifier},
            e,
        )

    try:
        predicted_ph_interval = prediction_class.predict_ph_interval(median_rgbs)
    except Exception as e:
        return raise_http_500(
            {"description": "pH prediction failed", "identifier": dir_identifier},
            e,
        )

    print(f"\n     🎯 Prediction for {dir_identifier}: {predicted_ph_interval}")

    return JSONResponse(
        content={
            "predicted_ph_interval": predicted_ph_interval,
            "identifier": dir_identifier,
            "rgbs": {
                "Q1": median_rgbs[0].tolist(),
                "Q2": median_rgbs[1].tolist(),
                "Q3": median_rgbs[2].tolist(),
                "Q4": median_rgbs[3].tolist(),
            },
        }
    )
