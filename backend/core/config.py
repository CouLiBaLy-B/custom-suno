"""Configuration centrale - AI Music Studio"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres globaux de l'application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    debug: bool = False
    database_url: str = "sqlite:///./musicstudio.db"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    storage_path: str = "./storage"
    model_cache_dir: str = "./models"
    device: str = "cuda"
    use_half_precision: bool = True
    musicgen_model: str = "facebook/musicgen-small"
    stable_audio_model: str = "stabilityai/stable-audio-open-1.0"
    bark_model: str = "suno/bark-small"
    secret_key: str = "dev-secret-key"

    @property
    def storage_dir(self) -> Path:
        return Path(self.storage_path).resolve()

    @property
    def model_dir(self) -> Path:
        return Path(self.model_cache_dir).resolve()


@lru_cache()
def get_settings() -> Settings:
    return Settings()
