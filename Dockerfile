FROM python:3.11-slim

WORKDIR /home/user/app

# Dépendances système pour PyAV / ffmpeg / audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Étape 1 : dépendances légères (toujours fonctionnent)
RUN pip install --no-cache-dir \
    pydantic \
    pydantic-settings \
    sqlalchemy \
    numpy \
    soundfile \
    loguru \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    streamlit \
    httpx \
    python-dotenv \
    scipy \
    pydub \
    librosa \
    pyloudnorm

# Étape 2 : torch CPU-only (beaucoup plus léger, pas besoin de CUDA sur HF Spaces si on configure le hardware après)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Étape 3 : transformers et accélération
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    diffusers \
    safetensors

# Étape 4 : audiocraft (optionnel, peut échouer sans GPU mais le Dockerfile continuera)
RUN pip install --no-cache-dir audiocraft || echo "⚠️ audiocraft skipped (needs GPU build)"

# Étape 5 : tâches optionnelles
RUN pip install --no-cache-dir celery redis python-jose passlib minio pytest pytest-asyncio || true

# Copier le code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY hf_app.py .

EXPOSE 7860

ENV API_URL=http://localhost:8000
ENV DEVICE=cpu

CMD ["python", "hf_app.py"]
