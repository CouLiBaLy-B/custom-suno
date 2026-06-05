"""Tâche Celery pour la génération MusicGen."""
from __future__ import annotations
from celery import current_app
from loguru import logger
from backend.services.musicgen_service import MusicGenService
from backend.core.config import get_settings

settings = get_settings()


@current_app.task(bind=True, name="tasks.musicgen.generate", max_retries=2, default_retry_delay=30)
def generate_music(
    self,
    task_id: str,
    prompt: str,
    duration: int = 15,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
    cfg_coef: float = 3.0,
    num_variations: int = 1,
    seed: int = -1,
):
    """Génère de la musique avec MusicGen."""
    logger.info(f"🎸 Démarrage MusicGen: task_id={task_id}, prompt='{prompt[:50]}...'")

    try:
        service = MusicGenService()

        def progress_callback(progress: int):
            self.update_state(state="PROGRESS", meta={"progress": progress, "step": f"Génération {progress}%"})

        audio, sample_rate = service.generate(
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
            num_variations=num_variations,
            seed=seed,
            progress_callback=progress_callback,
        )

        import soundfile as sf
        import numpy as np
        from pathlib import Path

        storage_dir = Path(settings.storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio)
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)

        file_path = storage_dir / f"{task_id}.wav"
        sf.write(str(file_path), audio_np.T, sample_rate)

        duration_sec = audio_np.shape[-1] / sample_rate

        logger.info(f"✅ MusicGen terminé: {file_path}, durée={duration_sec:.2f}s")

        return {
            "task_id": task_id,
            "status": "completed",
            "audio_path": str(file_path),
            "audio_url": f"/api/audio/{task_id}.wav",
            "audio_duration": duration_sec,
            "sample_rate": sample_rate,
        }

    except Exception as e:
        logger.error(f"❌ Échec MusicGen: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        self.retry(exc=e)
