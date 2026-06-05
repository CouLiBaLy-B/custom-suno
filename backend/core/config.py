"""Configuration - sans pydantic-settings pour compatibilité HF Spaces"""
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Paramètres depuis variables d'environnement."""
    host: str = os.environ.get("HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8000"))
    streamlit_port: int = int(os.environ.get("STREAMLIT_PORT", "8501"))
    debug: bool = os.environ.get("DEBUG", "false").lower() == "true"
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///./musicstudio.db")
    celery_broker_url: str = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    storage_path: str = os.environ.get("STORAGE_PATH", "./storage")
    model_cache_dir: str = os.environ.get("MODEL_CACHE_DIR", "./models")
    device: str = os.environ.get("DEVICE", "cpu")  # CPU par défaut pour Spaces
    use_half_precision: bool = os.environ.get("USE_HALF_PRECISION", "true").lower() == "true"
    musicgen_model: str = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
    stable_audio_model: str = os.environ.get("STABLE_AUDIO_MODEL", "stabilityai/stable-audio-open-1.0")
    bark_model: str = os.environ.get("BARK_MODEL", "suno/bark-small")
    secret_key: str = os.environ.get("SECRET_KEY", "dev-secret-key")

    @property
    def storage_dir(self) -> Path:
        return Path(self.storage_path).resolve()

    @property
    def model_dir(self) -> Path:
        return Path(self.model_cache_dir).resolve()


_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
