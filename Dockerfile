FROM python:3.11-slim

WORKDIR /home/user/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    build-essential \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pydantic pydantic-settings sqlalchemy numpy soundfile \
    loguru fastapi "uvicorn[standard]" python-multipart \
    streamlit httpx python-dotenv scipy pydub librosa \
    torch torchaudio transformers accelerate diffusers safetensors \
    huggingface_hub

RUN pip install --no-cache-dir audiocraft || echo "⚠️ audiocraft skipped"
RUN pip install --no-cache-dir celery redis python-jose passlib minio || true

COPY . .

RUN chmod +x app.py

EXPOSE 7860

ENV API_URL=http://localhost:8000
ENV DEVICE=cpu

CMD ["python", "app.py"]
