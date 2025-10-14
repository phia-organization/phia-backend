FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install gdown

WORKDIR /app

COPY . .

ARG MODEL_ID
RUN mkdir -p models && \
    gdown --fuzzy "https://drive.google.com/uc?id=${MODEL_ID}" -O models/RandomForestClassifier.sav

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.u2net && \
    wget -O /root/.u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

ENV PORT=8000
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
