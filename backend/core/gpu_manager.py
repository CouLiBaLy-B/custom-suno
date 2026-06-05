"""
AI Music Studio - Gestionnaire GPU
Lazy loading, cache LRU, quantification, monitoring VRAM.
"""
from __future__ import annotations
import gc
import threading
from contextlib import contextmanager
from typing import Dict, Optional
import torch
# logger = print  # replaced with print
from backend.core.config import get_settings

settings = get_settings()


class GPUManager:
    """
    Gestionnaire singleton des ressources GPU.
    - Lazy loading des modèles
    - LRU cache
    - Gestion mémoire GPU
    - Support quantification 8-bit/4-bit
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._loaded_models: Dict[str, object] = {}
        self._model_usage: Dict[str, int] = {}
        self._max_cached_models = 2
        self._device = self._detect_device()
        self._dtype = self._detect_dtype()
        print(f"GPUManager initialisé - Device: {self._device}, Dtype: {self._dtype}")

    def _detect_device(self) -> str:
        if settings.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU détecté: {gpu_name} ({gpu_memory:.1f} GB)")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS (Apple Silicon) détecté")
            return "mps"
        else:
            print("Aucun GPU détecté, utilisation du CPU")
            return "cpu"

    def _detect_dtype(self) -> torch.dtype:
        if self._device == "cuda" and settings.use_half_precision:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_vram_usage(self) -> dict:
        """Retourne l'utilisation VRAM actuelle."""
        if self._device != "cuda" or not torch.cuda.is_available():
            return {"used_gb": 0, "total_gb": 0, "free_gb": 0}

        used = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        return {
            "used_gb": round(used, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
        }

    def load_model(self, model_id: str, model_loader: callable, use_quantization: bool = False) -> object:
        """Charge un modèle en mémoire avec gestion de cache LRU."""
        if model_id in self._loaded_models:
            self._model_usage[model_id] += 1
            print(f"Modèle '{model_id}' déjà chargé en cache")
            return self._loaded_models[model_id]

        self._evict_least_used()
        self._free_memory()

        print(f"Chargement du modèle '{model_id}' (quantization: {use_quantization})")

        try:
            model = model_loader()

            if self._device == "cuda":
                model = model.to(self.dtype)

            self._loaded_models[model_id] = model
            self._model_usage[model_id] = 1
            print(f"✅ Modèle '{model_id}' chargé avec succès")

            return model

        except Exception as e:
            print(f"❌ Échec du chargement du modèle '{model_id}': {e}")
            raise

    def unload_model(self, model_id: str) -> None:
        """Décharge un modèle spécifique de la mémoire."""
        if model_id in self._loaded_models:
            print(f"Déchargement du modèle '{model_id}'")
            del self._loaded_models[model_id]
            del self._model_usage[model_id]
            self._free_memory()

    def _evict_least_used(self) -> None:
        """Évince le modèle le moins utilisé du cache."""
        if len(self._loaded_models) >= self._max_cached_models:
            least_used = min(self._model_usage, key=self._model_usage.get)
            print(f"Éviction du modèle '{least_used}' du cache")
            self.unload_model(least_used)

    def _free_memory(self) -> None:
        """Libère la mémoire GPU."""
        gc.collect()
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"VRAM après nettoyage: {self.get_vram_usage()}")

    def get_model(self, model_id: str) -> Optional[object]:
        """Récupère un modèle du cache."""
        if model_id in self._loaded_models:
            self._model_usage[model_id] += 1
            return self._loaded_models[model_id]
        return None

    def clear_cache(self) -> None:
        """Vide complètement le cache de modèles."""
        print("Vidage du cache de modèles")
        self._loaded_models.clear()
        self._model_usage.clear()
        self._free_memory()

    @contextmanager
    def model_context(self, model_id: str, model_loader: callable, **kwargs):
        """Context manager pour charger/unload un modèle automatiquement."""
        model = self.load_model(model_id, model_loader, **kwargs)
        try:
            yield model
        finally:
            self._model_usage[model_id] = max(0, self._model_usage.get(model_id, 1) - 1)


# Singleton global
gpu_manager = GPUManager()
