from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", 10))
MODEL_URL = os.getenv("MODEL_URL")
SCALER_URL = os.getenv("SCALER_URL")