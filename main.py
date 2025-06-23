import cv2
import uuid
import numpy as np

from lib import PingResponse, ErrorResponse, PredictionResponse, raise_http_500, MAX_IMAGE_SIZE_MB, ENVIRONMENT, IS_PRODUCTION
from classes import PredictionClass

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import imghdr

app = FastAPI()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

prediction_class = PredictionClass()
y_measures_cm = {
    'Q1': 0.7,
    'Q2': 1.8,
    'Q3': 3.0,
    'Q4': 4.2,
    'total': 12.6
}


@app.get("/health-check", response_model=PingResponse)
def read_root() -> PingResponse:
    """
    A simple endpoint that returns a greeting.
    Returns:
        dict: A dictionary with a greeting message.
    """
    return {"ping": "pong!"}

@app.post("/predict", responses={400: {"model": ErrorResponse}, 200: {"model": PredictionResponse}})
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and predict pH strip value.
    """

    filename = file.filename.lower()
    extension = filename.split(".")[-1]
    dir_identifier = uuid.uuid4().hex
    
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG and JPEG images are allowed.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 10 MB.")

    if imghdr.what(None, contents) not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    image_np = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode the image. Ensure it's a valid image file.")


    prediction_class.set_file_info(filename, extension, dir_identifier)
    

    try:
        strip_withouth_bg = prediction_class.remove_background(image)
    except Exception as e:
        return raise_http_500("Background removal failed", e)

    try:
        strip_rotated = prediction_class.rotate_vertically(strip_withouth_bg)
    except Exception as e:
        return raise_http_500("Rotation failed", e)

    try:
        strip_cropped = prediction_class.crop(strip_rotated)
    except Exception as e:
        return raise_http_500("Cropping failed", e)

    try:
        strip_white_balanced = prediction_class.white_balance_gray_world(strip_cropped)
    except Exception as e:
        return raise_http_500("White balancing failed", e)

    try:
        df_rgb = prediction_class.extract_rgbs(strip_white_balanced, y_measures_cm)
    except Exception as e:
        return raise_http_500("RGB extraction failed", e)

    try:
        predicted_ph = prediction_class.predict_ph(df_rgb)
    except Exception as e:
        return raise_http_500("pH prediction failed", e)


    return JSONResponse(
        content={
            "predicted_ph": predicted_ph,
            "identifier": dir_identifier,
            "rgbs": {
                "Q1": df_rgb.loc['Q1'].to_dict(),
                "Q2": df_rgb.loc['Q2'].to_dict(),
                "Q3": df_rgb.loc['Q3'].to_dict(),
                "Q4": df_rgb.loc['Q4'].to_dict()
            }
        }
    )