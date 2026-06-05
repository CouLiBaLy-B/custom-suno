FROM python:3.11-slim

WORKDIR /home/user/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY hf_app.py .

EXPOSE 7860

ENV API_URL=http://localhost:8000
ENV DEVICE=cpu

CMD ["python", "hf_app.py"]
