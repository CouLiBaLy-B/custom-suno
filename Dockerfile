FROM python:3.11-slim

WORKDIR /home/user/app

# Dépendances système pour l'audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 pkg-config build-essential \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer les dépendances Python
COPY requirements.txt .

# Installer les dépendances légères en premier (échec rapide si problème)
RUN pip install --no-cache-dir \
    pydantic sqlalchemy numpy soundfile \
    loguru fastapi "uvicorn[standard]" python-multipart \
    streamlit httpx python-dotenv scipy pydub librosa \
    torch torchaudio transformers accelerate diffusers safetensors \
    huggingface_hub && \
    pip install --no-cache-dir audiocraft 2>/dev/null || echo "⚠️ audiocraft skipped"

# Copier le code du projet
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p storage models

EXPOSE 7860

ENV API_URL=""
ENV DEVICE=cpu
ENV PORT=7860

CMD ["python", "app.py"]
