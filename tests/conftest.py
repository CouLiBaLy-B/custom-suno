"""
Configuration partagée pour tous les tests.
"""
import os
import sys
import tempfile
import shutil
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture(autouse=True)
def clean_env():
    """Nettoyer les variables d'environnement avant chaque test."""
    os.environ["DEVICE"] = "cpu"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["STORAGE_PATH"] = tempfile.mkdtemp()
    os.environ["MODEL_CACHE_DIR"] = tempfile.mkdtemp()
    os.environ["USE_HALF_PRECISION"] = "false"
    yield
    # Cleanup
    shutil.rmtree(os.environ.get("STORAGE_PATH", ""), ignore_errors=True)
    shutil.rmtree(os.environ.get("MODEL_CACHE_DIR", ""), ignore_errors=True)


@pytest.fixture
def tmp_storage():
    """Fixture pour un dossier de stockage temporaire."""
    tmpdir = tempfile.mkdtemp()
    os.environ["STORAGE_PATH"] = tmpdir
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_audio_1d():
    """Fixture: audio mono 1D (44100 échantillons, 1 seconde à 44100Hz)."""
    import numpy as np
    t = np.linspace(0, 1, 44100, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sample_audio_2d():
    """Fixture: audio stereo 2D (2 canaux, 44100 échantillons)."""
    import numpy as np
    t = np.linspace(0, 1, 44100, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    return np.array([left, right])


@pytest.fixture
def sample_audio_3d():
    """Fixture: audio 3D comme MusicGen (1, 1, 48000)."""
    import numpy as np
    return np.random.randn(1, 1, 48000).astype(np.float32)


@pytest.fixture
def sample_audio_4d():
    """Fixture: audio 4D extrême (1, 1, 1, 48000)."""
    import numpy as np
    return np.random.randn(1, 1, 1, 48000).astype(np.float32)
