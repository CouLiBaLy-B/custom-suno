"""Tests unitaires pour AI Music Studio."""
import pytest


def test_imports():
    """Teste que les imports fonctionnent."""
    from backend.core.config import get_settings
    settings = get_settings()
    assert settings.api_port == 8000
    assert settings.streamlit_port == 8501


def test_database():
    """Teste l'initialisation de la base de données."""
    from backend.core.database import init_db, engine
    init_db()
    assert engine is not None


def test_models():
    """Teste les modèles SQLAlchemy."""
    from backend.api.models.models import User, Project, Generation
    assert User.__tablename__ == "users"
    assert Project.__tablename__ == "projects"
    assert Generation.__tablename__ == "generations"


def test_gpu_manager():
    """Teste le GPU Manager."""
    from backend.core.gpu_manager import GPUManager
    mgr = GPUManager()
    usage = mgr.get_vram_usage()
    assert "used_gb" in usage
    assert "total_gb" in usage


def test_musicgen_service():
    """Teste l'instanciation du service MusicGen."""
    from backend.services.musicgen_service import MusicGenService
    svc = MusicGenService()
    assert svc is not None
    assert svc.device in ("cuda", "mps", "cpu")


def test_stable_audio_service():
    """Teste l'instanciation du service Stable Audio."""
    from backend.services.stable_audio_service import StableAudioService
    svc = StableAudioService()
    assert svc is not None


def test_bark_service():
    """Teste l'instanciation du service Bark."""
    from backend.services.bark_service import BarkService
    svc = BarkService()
    assert svc is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
