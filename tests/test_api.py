"""
Tests unitaires pour AI Music Studio API.
Tests des fichiers backend sans dépendances lourdes (torch, loguru).
"""
import os
import sys
import ast
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBackendStructure:
    """Tests de la structure du backend."""

    def test_backend_dir_exists(self):
        assert os.path.isdir(os.path.join(project_root, "backend"))

    def test_backend_init_exists(self):
        path = os.path.join(project_root, "backend", "__init__.py")
        assert os.path.exists(path)

    def test_api_dir_exists(self):
        assert os.path.isdir(os.path.join(project_root, "backend", "api"))

    def test_api_main_exists(self):
        path = os.path.join(project_root, "backend", "api", "main.py")
        assert os.path.exists(path)

    def test_core_dir_exists(self):
        assert os.path.isdir(os.path.join(project_root, "backend", "core"))

    def test_services_dir_exists(self):
        assert os.path.isdir(os.path.join(project_root, "backend", "services"))


class TestConfig:
    """Tests de la configuration."""

    def test_config_file_exists(self):
        assert os.path.exists(os.path.join(project_root, "backend", "core", "config.py"))

    def test_config_has_settings_class(self):
        with open(os.path.join(project_root, "backend", "core", "config.py")) as f:
            content = f.read()
        assert "class Settings" in content

    def test_config_has_get_settings(self):
        with open(os.path.join(project_root, "backend", "core", "config.py")) as f:
            content = f.read()
        assert "def get_settings" in content

    def test_config_has_storage_dir(self):
        with open(os.path.join(project_root, "backend", "core", "config.py")) as f:
            content = f.read()
        assert "storage_dir" in content
        assert "model_dir" in content


class TestDatabase:
    """Tests de la configuration de la base de données."""

    def test_database_file_exists(self):
        assert os.path.exists(os.path.join(project_root, "backend", "core", "database.py"))

    def test_database_has_engine(self):
        with open(os.path.join(project_root, "backend", "core", "database.py")) as f:
            content = f.read()
        assert "create_engine" in content
        assert "SessionLocal" in content

    def test_database_has_get_db(self):
        with open(os.path.join(project_root, "backend", "core", "database.py")) as f:
            content = f.read()
        assert "def get_db()" in content

    def test_database_has_init_db(self):
        with open(os.path.join(project_root, "backend", "core", "database.py")) as f:
            content = f.read()
        assert "def init_db()" in content


class TestModels:
    """Tests des modèles ORM."""

    def test_models_file_exists(self):
        assert os.path.exists(os.path.join(project_root, "backend", "api", "models", "models.py"))

    def test_user_model(self):
        with open(os.path.join(project_root, "backend", "api", "models", "models.py")) as f:
            content = f.read()
        assert "class User(Base)" in content
        assert "username" in content
        assert "email" in content
        assert "hashed_password" in content

    def test_project_model(self):
        with open(os.path.join(project_root, "backend", "api", "models", "models.py")) as f:
            content = f.read()
        assert "class Project(Base)" in content
        assert "owner_id" in content
        assert "ForeignKey" in content

    def test_generation_model(self):
        with open(os.path.join(project_root, "backend", "api", "models", "models.py")) as f:
            content = f.read()
        assert "class Generation(Base)" in content
        assert "model_name" in content
        assert "audio_url" in content
        assert "audio_duration" in content


class TestMusicGenService:
    """Tests du service MusicGen (analyse statique du code)."""

    def test_service_file_exists(self):
        assert os.path.exists(os.path.join(project_root, "backend", "services", "musicgen_service.py"))

    def test_service_class_definition(self):
        with open(os.path.join(project_root, "backend", "services", "musicgen_service.py")) as f:
            content = f.read()
        assert "class MusicGenService" in content
        assert "def generate" in content
        assert "def load_model" in content

    def test_service_has_device_property(self):
        with open(os.path.join(project_root, "backend", "services", "musicgen_service.py")) as f:
            content = f.read()
        assert "def device" in content

    def test_service_has_cuda_fallback(self):
        with open(os.path.join(project_root, "backend", "services", "musicgen_service.py")) as f:
            content = f.read()
        assert "cpu" in content
        assert "mps" in content

    def test_no_loguru_dependency(self):
        with open(os.path.join(project_root, "backend", "services", "musicgen_service.py")) as f:
            content = f.read()
        assert "from loguru import logger" not in content


class TestStableAudioService:
    """Tests du service Stable Audio."""

    def test_service_file_exists(self):
        assert os.path.exists(os.path.join(project_root, "backend", "services", "stable_audio_service.py"))

    def test_service_class_definition(self):
        with open(os.path.join(project_root, "backend", "services", "stable_audio_service.py")) as f:
            content = f.read()
        assert "class StableAudioService" in content
        assert "def generate" in content
        assert "def load_pipeline" in content

    def test_no_loguru_dependency(self):
        with open(os.path.join(project_root, "backend", "services", "stable_audio_service.py")) as f:
            content = f.read()
        assert "from loguru import logger" not in content


class TestBarkService:
    """Tests du service Bark."""

    def test_service_file_exists(self):
        assert os.path.exists(os.path.join(project_root, "backend", "services", "bark_service.py"))

    def test_service_class_definition(self):
        with open(os.path.join(project_root, "backend", "services", "bark_service.py")) as f:
            content = f.read()
        assert "class BarkService" in content
        assert "def generate" in content
        assert "def load_model" in content

    def test_no_loguru_dependency(self):
        with open(os.path.join(project_root, "backend", "services", "bark_service.py")) as f:
            content = f.read()
        assert "from loguru import logger" not in content


class TestGPUManager:
    """Tests du GPU Manager."""

    def test_gpu_manager_file_exists(self):
        assert os.path.exists(os.path.join(project_root, "backend", "core", "gpu_manager.py"))

    def test_gpu_manager_class_definition(self):
        with open(os.path.join(project_root, "backend", "core", "gpu_manager.py")) as f:
            content = f.read()
        assert "class GPUManager" in content
        assert "def load_model" in content
        assert "def get_vram_usage" in content

    def test_no_loguru_dependency(self):
        with open(os.path.join(project_root, "backend", "core", "gpu_manager.py")) as f:
            content = f.read()
        assert "from loguru import logger" not in content


class TestMainAPI:
    """Tests de l'API principale."""

    def test_main_file_syntax(self):
        with open(os.path.join(project_root, "backend", "api", "main.py")) as f:
            ast.parse(f.read())

    def test_main_has_fastapi_app(self):
        with open(os.path.join(project_root, "backend", "api", "main.py")) as f:
            content = f.read()
        assert "FastAPI" in content
        assert "app = FastAPI" in content

    def test_main_has_generate_endpoint(self):
        with open(os.path.join(project_root, "backend", "api", "main.py")) as f:
            content = f.read()
        assert "@app.post(\"/api/generate\"" in content

    def test_main_has_health_endpoint(self):
        with open(os.path.join(project_root, "backend", "api", "main.py")) as f:
            content = f.read()
        assert "@app.get(\"/api/health\"" in content

    def test_main_has_audio_endpoint(self):
        with open(os.path.join(project_root, "backend", "api", "main.py")) as f:
            content = f.read()
        assert "@app.get(\"/api/audio/" in content

    def test_main_has_websocket(self):
        with open(os.path.join(project_root, "backend", "api", "main.py")) as f:
            content = f.read()
        assert "@app.websocket(\"/ws/generation/" in content


class TestDockerConfig:
    """Tests de la configuration Docker."""

    def test_dockerfile_api_exists(self):
        assert os.path.exists(os.path.join(project_root, "Dockerfile.api"))

    def test_dockerfile_worker_exists(self):
        assert os.path.exists(os.path.join(project_root, "Dockerfile.worker"))

    def test_dockerfile_frontend_exists(self):
        assert os.path.exists(os.path.join(project_root, "Dockerfile.frontend"))

    def test_docker_compose_exists(self):
        assert os.path.exists(os.path.join(project_root, "docker-compose.yml"))

    def test_docker_compose_has_services(self):
        with open(os.path.join(project_root, "docker-compose.yml")) as f:
            content = f.read()
        assert "api:" in content
        assert "frontend:" in content
        assert "worker:" in content


class TestCIWorkflow:
    """Tests du workflow CI/CD."""

    def test_workflow_file_exists(self):
        assert os.path.exists(os.path.join(project_root, ".github", "workflows", "deploy-hf.yml"))

    def test_workflow_has_hf_upload(self):
        with open(os.path.join(project_root, ".github", "workflows", "deploy-hf.yml")) as f:
            content = f.read()
        assert "upload_folder" in content
        assert "huggingface_hub" in content

    def test_workflow_has_secret_check(self):
        with open(os.path.join(project_root, ".github", "workflows", "deploy-hf.yml")) as f:
            content = f.read()
        assert "HF_TOKEN" in content
        assert "HF_USERNAME" in content
        assert "HF_SPACE_NAME" in content
