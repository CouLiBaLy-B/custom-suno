"""
Service de génération vocale avec Bark (Suno).
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np
import torch
# logger = print  # replaced with print
from backend.core.config import get_settings

settings = get_settings()


class BarkService:
    """Service singleton pour Bark."""
    _instance = None
    _model = None
    _processor = None

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

    def load_model(self) -> None:
        """Charge le modèle Bark."""
        if self._model is not None:
            return

        print(f"Chargement Bark: {settings.bark_model}")
        from transformers import AutoProcessor, BarkModel
        self._processor = AutoProcessor.from_pretrained(settings.bark_model)
        self._model = BarkModel.from_pretrained(settings.bark_model)
        self._model = self._model.to(self.device)
        print("✅ Bark chargé")

    async def generate(
        self,
        text: str,
        voice_preset: str = "v2/fr_speaker_1",
        duration: Optional[int] = None,
        num_variations: int = 1,
        seed: int = -1,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Génère de l'audio vocal à partir de texte.
        """
        self.load_model()
        if progress_callback:
            progress_callback(20)

        inputs = self._processor(text=text, voice_preset=voice_preset)

        if progress_callback:
            progress_callback(40)

        print(f"Génération Bark: '{text[:50]}...'")

        audios = []
        for i in range(num_variations):
            if seed >= 0:
                torch.manual_seed(seed + i)
            audio = self._model.generate(
                **{k: v.to(self.device) for k, v in inputs.items()},
                do_sample=True,
            )
            audios.append(audio.cpu().numpy())

        if progress_callback:
            progress_callback(90)

        sample_rate = 24000
        result = audios[0] if num_variations == 1 else np.concatenate(audios, axis=-1)

        if progress_callback:
            progress_callback(100)

        print(f"✅ Bark terminé: shape={result.shape}, sr={sample_rate}")
        return result, sample_rate
