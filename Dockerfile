FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PORT=8000

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
