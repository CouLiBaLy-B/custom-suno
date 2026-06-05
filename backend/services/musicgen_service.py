"""
Service de génération musicale avec MusicGen (Meta).
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np
import torch
# logger = print  # replaced with print
from backend.core.config import get_settings

settings = get_settings()


class MusicGenService:
    """Service singleton pour MusicGen avec lazy loading."""
    _instance = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def device(self) -> str:
        if settings.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        if settings.use_half_precision and self.device == "cuda":
            return torch.float16
        return torch.float32

    def load_model(self) -> None:
        """Charge le modèle MusicGen en mémoire."""
        if self._model is not None:
            return

        model_name = settings.musicgen_model
        print(f"Chargement MusicGen: {model_name}")

        try:
            from audiocraft.models import MusicGen as MC
            self._model = MC.get_pretrained(model_name)
            self._model = self._model.to(self.device)
            print(f"✅ MusicGen chargé via audiocraft: {model_name}")
        except ImportError:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=self.dtype
            )
            self._model = self._model.to(self.device)
            print(f"✅ MusicGen chargé via transformers: {model_name}")

    async def generate(
        self,
        prompt: str,
        duration: int = 15,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.0,
        num_variations: int = 1,
        seed: int = -1,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Génère de la musique à partir d'un prompt textuel.
        """
        self.load_model()
        if progress_callback:
            progress_callback(20)

        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if progress_callback:
            progress_callback(40)

        print(f"Génération MusicGen: '{prompt[:50]}...' ({duration}s)")

        if hasattr(self._model, 'set_generation_params'):
            # API audiocraft
            self._model.set_generation_params(
                duration=min(duration, 30),
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                cfg_coef=cfg_coef,
            )
            prompts = [prompt] * num_variations
            wav = self._model.generate(prompts)
            audio = wav.cpu().numpy()
            sample_rate = int(self._model.sample_rate)
        else:
            # API transformers
            inputs = self._processor(text=[prompt] * num_variations, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            audio_values = self._model.generate(
                **inputs,
                max_new_tokens=int(duration * 50),
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            audio = audio_values.cpu().numpy()
            sample_rate = self._model.config.audio_encoder_sample_rate

        if progress_callback:
            progress_callback(100)

        print(f"✅ MusicGen terminé: shape={audio.shape}, sr={sample_rate}")
        return audio, sample_rate
