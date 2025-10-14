FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências necessárias
RUN pip install --no-cache-dir gdown python-dotenv

WORKDIR /app

# Copia o código e o .env
COPY . .

# Lê o MODEL_ID do .env e baixa o modelo do Google Drive
RUN python -c "import os; from dotenv import load_dotenv; load_dotenv(); import subprocess; mid=os.getenv('MODEL_ID'); \
    os.makedirs('models', exist_ok=True); \
    subprocess.run(['gdown', '--fuzzy', f'https://drive.google.com/uc?id={mid}', '-O', 'models/RandomForestClassifier.sav'], check=True)"


# Instala dependências do projeto
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PORT=8000

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
