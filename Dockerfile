FROM python:3.11-slim

WORKDIR /home/user/app

# Dépendances système pour le traitement audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 pkg-config build-essential \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pydantic sqlalchemy numpy soundfile \
    loguru fastapi "uvicorn[standard]" python-multipart \
    streamlit httpx python-dotenv scipy pydub librosa \
    torch torchaudio transformers accelerate diffusers safetensors \
    huggingface_hub

# audiocraft optionnel (peut échouer sans GPU)
RUN pip install --no-cache-dir audiocraft 2>/dev/null || echo "⚠️ audiocraft non installé"

# Tâches optionnelles
RUN pip install --no-cache-dir celery redis python-jose passlib minio 2>/dev/null || true

# Copier le code
COPY . .

EXPOSE 7860

ENV API_URL=http://localhost:8000
ENV DEVICE=cpu

CMD ["python", "app.py"]
