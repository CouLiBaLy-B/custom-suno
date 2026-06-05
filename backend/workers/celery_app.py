"""Configuration Celery pour le traitement asynchrone des tâches de génération."""
from __future__ import annotations
from celery import Celery
from backend.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ai_music_studio",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "backend.workers.tasks.musicgen_task",
        "backend.workers.tasks.stable_audio_task",
        "backend.workers.tasks.bark_task",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Paris",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

celery_app.conf.worker_concurrency = 1
