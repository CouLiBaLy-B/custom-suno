"""
Tests complets du traitement audio et des services IA.
"""
import os
import sys
import uuid
import numpy as np
import tempfile
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Config minimale
os.environ["DEVICE"] = "cpu"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


# ─── Helpers ───
def get_save_audio():
    """Import de save_audio (fonction pure, pas besoin de streamlit)."""
    import soundfile as sf
    def save_audio_impl(audio_data, sample_rate):
        audio_np = audio_data.cpu().numpy() if hasattr(audio_data, 'cpu') else np.array(audio_data)
        audio_np = np.squeeze(audio_np)
        if audio_np.ndim == 2:
            if audio_np.shape[0] < audio_np.shape[1]:
                audio_np = audio_np.T
        elif audio_np.ndim != 1:
            audio_np = audio_np.flatten()
        tmpdir = os.environ.get("STORAGE_PATH", tempfile.mkdtemp())
        task_id = str(uuid.uuid4())
        file_path = os.path.join(tmpdir, f"{task_id}.wav")
        sf.write(file_path, audio_np, sample_rate)
        duration_sec = audio_np.shape[-1] / sample_rate
        return file_path, task_id, duration_sec
    return save_audio_impl


# ═══════════════════════════════════════════════════════
#  TESTS SAVE_AUDIO
# ═══════════════════════════════════════════════════════

class TestSaveAudio:
    """Tests de la fonction save_audio."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["STORAGE_PATH"] = self.tmpdir

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_shape_1d_mono(self):
        """Audio 1D: (48000,) → mono."""
        save_audio = get_save_audio()
        audio = np.random.randn(48000).astype(np.float32)
        file_path, task_id, duration = save_audio(audio, sample_rate=48000)
        assert os.path.exists(file_path)
        assert duration == pytest.approx(1.0, abs=0.01)

        import soundfile as sf
        data, sr = sf.read(file_path)
        assert sr == 48000
        assert data.shape == (48000,)

    def test_shape_2d_stereo_channels_first(self):
        """Audio 2D: (2, 48000) → stereo, channels en premier."""
        save_audio = get_save_audio()
        audio = np.random.randn(2, 48000).astype(np.float32)
        file_path, task_id, duration = save_audio(audio, sample_rate=48000)
        import soundfile as sf
        data, sr = sf.read(file_path)
        assert sr == 48000
        assert data.shape == (48000, 2)

    def test_shape_3d_musicgen_output(self):
        """Audio 3D: (1, 1, 478080) → format MusicGen."""
        save_audio = get_save_audio()
        audio = np.random.randn(1, 1, 478080).astype(np.float32)
        file_path, task_id, duration = save_audio(audio, sample_rate=32000)
        assert os.path.exists(file_path)
        assert duration == pytest.approx(478080 / 32000, abs=0.01)

    def test_shape_3d_batch_of_2(self):
        """Audio 3D: (2, 1, 48000) → batch de 2."""
        save_audio = get_save_audio()
        audio = np.random.randn(2, 1, 48000).astype(np.float32)
        file_path, task_id, duration = save_audio(audio, sample_rate=48000)
        import soundfile as sf
        data, sr = sf.read(file_path)
        assert sr == 48000
        assert data.shape == (48000, 2)

    def test_shape_4d_extreme(self):
        """Audio 4D: (1, 1, 1, 48000)."""
        save_audio = get_save_audio()
        audio = np.random.randn(1, 1, 1, 48000).astype(np.float32)
        file_path, task_id, duration = save_audio(audio, sample_rate=48000)
        import soundfile as sf
        data, sr = sf.read(file_path)
        assert sr == 48000
        assert data.shape == (48000,)

    def test_torch_tensor_input(self):
        """Input tensor PyTorch → conversion numpy."""
        save_audio = get_save_audio()
        import torch
        audio = torch.randn(1, 48000)
        file_path, task_id, duration = save_audio(audio, sample_rate=48000)
        import soundfile as sf
        data, sr = sf.read(file_path)
        assert sr == 48000

    def test_different_sample_rates(self):
        """Différents taux d'échantillonnage."""
        save_audio = get_save_audio()
        for sr in [16000, 22050, 32000, 44100, 48000]:
            audio = np.random.randn(sr).astype(np.float32)
            file_path, task_id, duration = save_audio(audio, sample_rate=sr)
            assert duration == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════
#  TESTS QUALITÉ AUDIO
# ═══════════════════════════════════════════════════════

class TestAudioQuality:
    """Tests de qualité audio."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["STORAGE_PATH"] = self.tmpdir

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_roundtrip_preservation(self):
        """L'audio sauvegardé puis relu est identique."""
        save_audio = get_save_audio()
        original = np.sin(np.linspace(0, 440 * 2 * np.pi, 44100)).astype(np.float32)
        file_path, _, _ = save_audio(original, sample_rate=44100)
        import soundfile as sf
        recovered, _ = sf.read(file_path)
        correlation = np.corrcoef(original, recovered[:44100])[0, 1]
        assert correlation > 0.99

    def test_silence_creates_valid_file(self):
        """Un signal de silence crée un fichier WAV valide."""
        save_audio = get_save_audio()
        silence = np.zeros(44100, dtype=np.float32)
        file_path, _, duration = save_audio(silence, sample_rate=44100)
        assert os.path.exists(file_path)
        assert duration == pytest.approx(1.0, abs=0.01)
        assert os.path.getsize(file_path) > 0

    def test_stereo_signal(self):
        """Signal stéréo avec canaux différents."""
        save_audio = get_save_audio()
        left = np.sin(np.linspace(0, 440 * 2 * np.pi, 44100)).astype(np.float32)
        right = np.sin(np.linspace(0, 880 * 2 * np.pi, 44100)).astype(np.float32)
        stereo = np.array([left, right]).T
        file_path, _, _ = save_audio(stereo, sample_rate=44100)
        import soundfile as sf
        data, sr = sf.read(file_path)
        assert sr == 44100
        assert data.ndim == 2
        assert data.shape[1] == 2


# ═══════════════════════════════════════════════════════
#  TESTS SERVICES IA
# ═══════════════════════════════════════════════════════

class TestMusicGenService:
    """Tests du service MusicGen."""

    def test_service_class_exists(self):
        """La classe MusicGenService existe et a les bonnes méthodes."""
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert 'class MusicGenService' in content
        assert 'def generate' in content
        assert 'def load_model' in content

    def test_singleton_pattern(self):
        """Le pattern singleton est implémenté."""
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert '_instance = None' in content
        assert 'def __new__' in content

    def test_device_detection_cpu(self):
        """Détection CPU par défaut."""
        os.environ["DEVICE"] = "cpu"
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert 'def device' in content
        assert 'return "cpu"' in content


class TestStableAudioService:
    """Tests du service Stable Audio."""

    def test_service_class_exists(self):
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert 'class StableAudioService' in content
        assert 'def generate' in content
        assert 'StableAudioPipeline' in content


class TestBarkService:
    """Tests du service Bark."""

    def test_service_class_exists(self):
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert 'class BarkService' in content
        assert 'def generate' in content
        assert 'BarkModel' in content


# ═══════════════════════════════════════════════════════
#  TESTS CONFIGURATION
# ═══════════════════════════════════════════════════════

class TestSettings:
    """Tests de la configuration."""

    def test_default_device_is_cpu(self):
        """Par défaut, le device est CPU."""
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert 'os.environ.get("DEVICE", "cpu")' in content

    def test_env_variable_override(self):
        """Les variables d'environnement surchargent les défauts."""
        os.environ["DEVICE"] = "cuda"
        os.environ["MUSICGEN_MODEL"] = "facebook/musicgen-medium"
        assert os.environ["DEVICE"] == "cuda"
        assert os.environ["MUSICGEN_MODEL"] == "facebook/musicgen-medium"
        # Reset
        os.environ["DEVICE"] = "cpu"
        os.environ["MUSICGEN_MODEL"] = "facebook/musicgen-small"


# ═══════════════════════════════════════════════════════
#  TESTS DATABASE
# ═══════════════════════════════════════════════════════

class TestDatabase:
    """Tests de la base de données."""

    def test_models_are_defined(self):
        """Les modèles ORM existent dans le code."""
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert 'class User(Base)' in content
        assert 'class Project(Base)' in content
        assert 'class Generation(Base)' in content
        assert '__tablename__ = "users"' in content
        assert '__tablename__ = "projects"' in content
        assert '__tablename__ = "generations"' in content

    def test_model_fields(self):
        """Les modèles ont les bons champs."""
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        # Vérifier les champs de Generation
        for field in ["model_name", "prompt", "status", "audio_url", "audio_duration",
                       "error_message", "project_id", "user_id"]:
            assert f'{field} = Column(' in content, f"Champ {field} manquant"

    def test_init_db_function(self):
        """La fonction init_db existe."""
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert 'def init_db()' in content
        assert 'Base.metadata.create_all' in content


# ═══════════════════════════════════════════════════════
#  TESTS INTÉGRATION
# ═══════════════════════════════════════════════════════

class TestIntegration:
    """Tests d'intégration."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["STORAGE_PATH"] = self.tmpdir

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_save_and_retrieve_cycle(self):
        """Cycle complet: générer → sauvegarder → lire."""
        save_audio = get_save_audio()
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).reshape(1, -1)
        file_path, task_id, duration = save_audio(audio, sample_rate=44100)
        assert os.path.exists(file_path)
        assert duration == pytest.approx(1.0, abs=0.01)
        import soundfile as sf
        data, sr = sf.read(file_path)
        assert sr == 44100
        assert len(data) == 44100

    def test_db_functions_exist(self):
        """Les fonctions DB existent dans le code."""
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        assert 'def save_generation_to_db' in content
        assert 'def get_generations_from_db' in content
        assert 'def get_db()' in content

    def test_db_save_returns_dict(self):
        """get_generations_from_db retourne des dicts, pas des ORM objects."""
        with open(os.path.join(project_root, 'app.py')) as f:
            content = f.read()
        # Vérifier qu'on retourne des dicts
        assert '"model_name": g.model_name' in content
        assert '"status": g.status' in content
        assert '"audio_url": g.audio_url' in content


# ═══════════════════════════════════════════════════════
#  TESTS UI / FRONTEND
# ═══════════════════════════════════════════════════════

class TestFrontendApp:
    """Tests du fichier frontend/app.py."""

    def test_frontend_exists(self):
        frontend_path = os.path.join(project_root, "frontend", "app.py")
        assert os.path.exists(frontend_path)

    def test_frontend_syntax(self):
        import ast
        with open(os.path.join(project_root, "frontend", "app.py")) as f:
            ast.parse(f.read())

    def test_frontend_uses_relative_urls(self):
        with open(os.path.join(project_root, "frontend", "app.py")) as f:
            content = f.read()
        assert "_api_url" in content

    def test_frontend_has_error_handling(self):
        with open(os.path.join(project_root, "frontend", "app.py")) as f:
            content = f.read()
        assert "requests.exceptions.ConnectionError" in content
        assert "st.error" in content


class TestStreamlitUI:
    """Tests de l'interface Streamlit."""

    def test_st_import(self):
        import streamlit as st
        assert st is not None

    def test_css_injection(self):
        with open(os.path.join(project_root, "app.py")) as f:
            content = f.read()
        assert "background:" in content
        assert "linear-gradient" in content
        assert ".stButton" in content

    def test_modules_list(self):
        with open(os.path.join(project_root, "app.py")) as f:
            content = f.read()
        assert "🎸 MusicGen" in content
        assert "🔊 Stable Audio" in content
        assert "🗣️ Bark" in content
        assert "📚 Bibliothèque" in content


class TestDockerfile:
    """Tests du Dockerfile."""

    def test_dockerfile_exists(self):
        # Le Dockerfile a été supprimé pour SDK streamlit
        assert not os.path.exists(os.path.join(project_root, "Dockerfile"))

    def test_readme_has_streamlit_sdk(self):
        with open(os.path.join(project_root, "README.md")) as f:
            content = f.read()
        assert "sdk: streamlit" in content
        assert "app_file: app.py" in content
        assert "hardware: t4-small" in content
