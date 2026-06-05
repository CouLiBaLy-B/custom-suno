"""
Service de génération d'effets sonores avec Stable Audio Open (Stability AI).
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np
import torch
# logger = print  # replaced with print
from backend.core.config import get_settings

settings = get_settings()


class StableAudioService:
    """Service singleton pour Stable Audio Open 1.0."""
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_pipeline(self) -> None:
        """Charge le pipeline Stable Audio."""
        if self._pipeline is not None:
            return

        print(f"Chargement Stable Audio: {settings.stable_audio_model}")
        from diffusers import StableAudioPipeline
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipeline = StableAudioPipeline.from_pretrained(
            settings.stable_audio_model, torch_dtype=dtype
        )
        self._pipeline = self._pipeline.to(self.device)
        print("✅ Stable Audio chargé")

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        duration: float = 30.0,
        num_variations: int = 1,
        seed: int = -1,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Génère un effet sonore / sample audio.
        """
        self.load_pipeline()
        if progress_callback:
            progress_callback(20)

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None

        if progress_callback:
            progress_callback(40)

        print(f"Génération Stable Audio: '{prompt[:50]}...' ({duration}s)")

        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=200,
            audio_end_in_s=min(duration, 47.0),
            num_waveforms_per_prompt=num_variations,
            generator=generator,
        )

        if progress_callback:
            progress_callback(90)

        audio = result.audios[0].cpu().numpy() if num_variations == 1 else result.audios.cpu().numpy()
        sample_rate = self._pipeline.vae.sampling_rate

        if progress_callback:
            progress_callback(100)

        print(f"✅ Stable Audio terminé: shape={audio.shape}, sr={sample_rate}")
        return audio, sample_rate
