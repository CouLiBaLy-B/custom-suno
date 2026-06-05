"""
AI Music Studio — Single-file Application
Tout dans un seul fichier : modèles, services, API, frontend launcher.
Zero problème d'import. Fonctionne sur HF Spaces, Docker, et local.
"""
from __future__ import annotations

# ── CRITIQUE : sys.path en TOUT premier ──
import os, sys
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.chdir(app_dir)

import asyncio
import json
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════

class Settings:
    host: str = os.environ.get("HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8000"))
    streamlit_port: int = int(os.environ.get("STREAMLIT_PORT", "8501"))
    debug: bool = os.environ.get("DEBUG", "false").lower() == "true"
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///./musicstudio.db")
    celery_broker_url: str = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    storage_path: str = os.environ.get("STORAGE_PATH", "./storage")
    model_cache_dir: str = os.environ.get("MODEL_CACHE_DIR", "./models")
    device: str = os.environ.get("DEVICE", "cpu")
    use_half_precision: bool = os.environ.get("USE_HALF_PRECISION", "true").lower() == "true"
    musicgen_model: str = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
    stable_audio_model: str = os.environ.get("STABLE_AUDIO_MODEL", "stabilityai/stable-audio-open-1.0")
    bark_model: str = os.environ.get("BARK_MODEL", "suno/bark-small")
    secret_key: str = os.environ.get("SECRET_KEY", "dev-secret-key")

    @property
    def storage_dir(self):
        return os.path.abspath(self.storage_path)

    @property
    def model_dir(self):
        return os.path.abspath(self.model_cache_dir)

settings = Settings()

# ═══════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)

# ═══════════════════════════════════════════════════════
#  MODELS ORM
# ═══════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    generations = relationship("Generation", back_populates="user", cascade="all, delete-orphan")

class Project(Base):
    __tablename__ = "projects"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), default="Nouveau projet")
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    owner = relationship("User", back_populates="projects")
    generations = relationship("Generation", back_populates="project", cascade="all, delete-orphan")

class Generation(Base):
    __tablename__ = "generations"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    model_name = Column(String(64), nullable=False)
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, nullable=True)
    genre_tags = Column(Text, nullable=True)
    lyrics = Column(Text, nullable=True)
    parameters = Column(Text, nullable=True)
    status = Column(String(20), default="pending")
    error_message = Column(Text, nullable=True)
    audio_url = Column(String(512), nullable=True)
    audio_duration = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    project = relationship("Project", back_populates="generations")
    user = relationship("User", back_populates="generations")

# ═══════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════

class GenerationRequest(BaseModel):
    model_name: str = Field("musicgen")
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = None
    duration: int = Field(15, ge=3, le=300)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(250, ge=1, le=1000)
    top_p: float = Field(0.0, ge=0.0, le=1.0)
    cfg_coef: float = Field(3.0, ge=0.0, le=10.0)
    num_variations: int = Field(1, ge=1, le=4)
    seed: int = Field(-1)
    genre_tags: Optional[str] = None
    lyrics: Optional[str] = None
    project_id: Optional[str] = None

class GenerationResponse(BaseModel):
    task_id: str
    status: str = "pending"
    estimated_time_seconds: int = 60
    message: str = "Génération lancée"

class GenerationStatus(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    current_step: Optional[str] = None
    eta: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[str] = None

class ProjectCreate(BaseModel):
    name: str = "Nouveau projet"
    description: str = ""

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    generations: list = []

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: str
    username: str
    email: str

# ═══════════════════════════════════════════════════════
#  TASK STORE (in-memory pour le MVP)
# ═══════════════════════════════════════════════════════

_task_store: dict[str, dict] = {}

def _update_task(task_id: str, **kwargs):
    if task_id in _task_store:
        _task_store[task_id].update(kwargs)

# ═══════════════════════════════════════════════════════
#  AI SERVICES
# ═══════════════════════════════════════════════════════

class MusicGenService:
    _instance = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def device(self):
        import torch
        if settings.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        if self._model is not None:
            return
        import torch
        print(f"📥 Loading MusicGen: {settings.musicgen_model}")
        try:
            from audiocraft.models import MusicGen as MC
            self._model = MC.get_pretrained(settings.musicgen_model)
            self._model = self._model.to(self.device)
            print("✅ MusicGen loaded via audiocraft")
        except ImportError:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            dtype = torch.float16 if self.device == "cuda" and settings.use_half_precision else torch.float32
            self._processor = AutoProcessor.from_pretrained(settings.musicgen_model)
            self._model = MusicgenForConditionalGeneration.from_pretrained(settings.musicgen_model, torch_dtype=dtype)
            self._model = self._model.to(self.device)
            print("✅ MusicGen loaded via transformers")

    async def generate(self, prompt: str, duration: int = 15, temperature: float = 1.0,
                       top_k: int = 250, top_p: float = 0.0, cfg_coef: float = 3.0,
                       num_variations: int = 1, seed: int = -1,
                       progress_callback=None):
        import torch
        self.load_model()
        if progress_callback: progress_callback(20)
        if seed >= 0:
            torch.manual_seed(seed)
        if progress_callback: progress_callback(40)
        print(f"🎸 Generating: '{prompt[:50]}...' ({duration}s)")
        if hasattr(self._model, 'set_generation_params'):
            self._model.set_generation_params(
                duration=min(duration, 30), top_k=top_k, top_p=top_p,
                temperature=temperature, cfg_coef=cfg_coef,
            )
            wav = self._model.generate([prompt] * num_variations)
            audio = wav.cpu().numpy()
            sample_rate = int(self._model.sample_rate)
        else:
            inputs = self._processor(text=[prompt] * num_variations, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            audio_values = self._model.generate(
                **inputs, max_new_tokens=int(duration * 50),
                do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            audio = audio_values.cpu().numpy()
            sample_rate = self._model.config.audio_encoder_sample_rate
        if progress_callback: progress_callback(100)
        print(f"✅ MusicGen done: shape={audio.shape}, sr={sample_rate}")
        return audio, sample_rate


class StableAudioService:
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def device(self):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_pipeline(self):
        if self._pipeline is not None:
            return
        import torch
        print(f"📥 Loading Stable Audio: {settings.stable_audio_model}")
        from diffusers import StableAudioPipeline
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipeline = StableAudioPipeline.from_pretrained(
            settings.stable_audio_model, torch_dtype=dtype
        )
        self._pipeline = self._pipeline.to(self.device)
        print("✅ Stable Audio loaded")

    async def generate(self, prompt: str, negative_prompt: str = None,
                       duration: float = 30.0, num_variations: int = 1,
                       seed: int = -1, progress_callback=None):
        import torch
        self.load_pipeline()
        if progress_callback: progress_callback(20)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None
        if progress_callback: progress_callback(40)
        print(f"🔊 Generating: '{prompt[:50]}...' ({duration}s)")
        result = self._pipeline(
            prompt=prompt, negative_prompt=negative_prompt or "",
            num_inference_steps=200, audio_end_in_s=min(duration, 47.0),
            num_waveforms_per_prompt=num_variations, generator=generator,
        )
        if progress_callback: progress_callback(90)
        audio = result.audios[0].cpu().numpy() if num_variations == 1 else result.audios.cpu().numpy()
        sample_rate = self._pipeline.vae.sampling_rate
        if progress_callback: progress_callback(100)
        print(f"✅ Stable Audio done: shape={audio.shape}, sr={sample_rate}")
        return audio, sample_rate


class BarkService:
    _instance = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def device(self):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        if self._model is not None:
            return
        print(f"📥 Loading Bark: {settings.bark_model}")
        from transformers import AutoProcessor, BarkModel
        self._processor = AutoProcessor.from_pretrained(settings.bark_model)
        self._model = BarkModel.from_pretrained(settings.bark_model)
        self._model = self._model.to(self.device)
        print("✅ Bark loaded")

    async def generate(self, text: str, voice_preset: str = "v2/fr_speaker_1",
                       duration: int = None, num_variations: int = 1,
                       seed: int = -1, progress_callback=None):
        import torch
        self.load_model()
        if progress_callback: progress_callback(20)
        inputs = self._processor(text=text, voice_preset=voice_preset)
        if progress_callback: progress_callback(40)
        print(f"🗣️ Generating: '{text[:50]}...'")
        audios = []
        for i in range(num_variations):
            if seed >= 0:
                torch.manual_seed(seed + i)
            audio = self._model.generate(
                **{k: v.to(self.device) for k, v in inputs.items()},
                do_sample=True,
            )
            audios.append(audio.cpu().numpy())
        if progress_callback: progress_callback(90)
        sample_rate = 24000
        result = audios[0] if num_variations == 1 else np.concatenate(audios, axis=-1)
        if progress_callback: progress_callback(100)
        print(f"✅ Bark done: shape={result.shape}, sr={sample_rate}")
        return result, sample_rate

# ═══════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    os.makedirs(settings.storage_dir, exist_ok=True)
    os.makedirs(settings.model_dir, exist_ok=True)
    init_db()
    print("✅ AI Music Studio API démarrée")
    yield
    print("🛑 AI Music Studio API arrêtée")

api_app = FastAPI(
    title="AI Music Studio",
    description="Plateforme open source de génération musicale par IA",
    version="0.1.0",
    lifespan=lifespan,
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api_app.post("/api/generate", response_model=GenerationResponse)
async def generate_music(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    task_id = str(uuid.uuid4())
    gen = Generation(
        id=task_id, project_id=request.project_id, user_id="anonymous",
        model_name=request.model_name, prompt=request.prompt,
        negative_prompt=request.negative_prompt, genre_tags=request.genre_tags,
        lyrics=request.lyrics,
        parameters=json.dumps({
            "duration": request.duration, "temperature": request.temperature,
            "top_k": request.top_k, "top_p": request.top_p,
            "cfg_coef": request.cfg_coef, "seed": request.seed,
            "num_variations": request.num_variations,
        }),
        status="pending", created_at=datetime.utcnow(),
    )
    db.add(gen)
    db.commit()
    _task_store[task_id] = {
        "task_id": task_id, "status": "pending", "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(_run_generation, task_id=task_id, request=request)
    return GenerationResponse(
        task_id=task_id, estimated_time_seconds=request.duration * 2,
        message=f"Génération '{request.model_name}' lancée: '{request.prompt[:50]}...'",
    )

@api_app.get("/api/generate/{task_id}", response_model=GenerationStatus)
async def get_status(task_id: str):
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    task = _task_store[task_id]
    status = task["status"]
    resp = GenerationStatus(
        task_id=task_id, status=status,
        progress=task.get("progress", 0),
        current_step=task.get("current_step"),
    )
    if status == "completed":
        resp.result = {
            "task_id": task_id,
            "audio_url": task.get("audio_url"),
            "audio_duration": task.get("audio_duration"),
        }
    elif status == "failed":
        resp.error = task.get("error", "Erreur inconnue")
    return resp

@api_app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(settings.storage_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    return FileResponse(file_path, media_type="audio/wav")

@api_app.websocket("/ws/generation/{task_id}")
async def ws_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            if task_id not in _task_store:
                await websocket.send_json({"error": "Tâche introuvable"})
                break
            task = _task_store[task_id]
            await websocket.send_json({
                "task_id": task_id, "status": task["status"],
                "progress": task.get("progress", 0),
                "current_step": task.get("current_step"),
            })
            if task["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

@api_app.get("/api/projects", response_model=list[ProjectResponse])
async def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).all()
    return [
        ProjectResponse(
            id=p.id, name=p.name, description=p.description,
            created_at=p.created_at.isoformat(),
        )
        for p in projects
    ]

@api_app.post("/api/projects", response_model=ProjectResponse)
async def create_project(proj: ProjectCreate, db: Session = Depends(get_db)):
    new_proj = Project(
        id=str(uuid.uuid4()), owner_id="anonymous",
        name=proj.name, description=proj.description,
        created_at=datetime.utcnow(),
    )
    db.add(new_proj)
    db.commit()
    db.refresh(new_proj)
    return ProjectResponse(
        id=new_proj.id, name=new_proj.name,
        description=new_proj.description,
        created_at=new_proj.created_at.isoformat(),
    )

@api_app.get("/api/health")
async def health():
    return {"status": "ok", "service": "AI Music Studio", "version": "0.1.0"}

@api_app.post("/api/auth/login", response_model=TokenResponse)
async def login(creds: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == creds.username).first()
    if not user:
        user = User(
            id=str(uuid.uuid4()), username=creds.username,
            email=f"{creds.username}@example.com",
            hashed_password=creds.password,
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return TokenResponse(access_token=str(uuid.uuid4()))

@api_app.get("/api/auth/me", response_model=UserResponse)
async def get_me():
    return UserResponse(id="anon", username="anonymous", email="anon@example.com")

# ═══════════════════════════════════════════════════════
#  BACKGROUND GENERATION WORKER
# ═══════════════════════════════════════════════════════

async def _run_generation(task_id: str, request: GenerationRequest):
    """Exécute la génération en arrière-plan."""
    _update_task(task_id, status="processing", progress=10)
    try:
        if request.model_name == "musicgen":
            audio, sr = await MusicGenService().generate(
                prompt=request.prompt, duration=request.duration,
                temperature=request.temperature, top_k=request.top_k,
                top_p=request.top_p, cfg_coef=request.cfg_coef,
                num_variations=request.num_variations, seed=request.seed,
                progress_callback=lambda p: _update_task(task_id, progress=p),
            )
        elif request.model_name == "stable_audio":
            audio, sr = await StableAudioService().generate(
                prompt=request.prompt, negative_prompt=request.negative_prompt,
                duration=request.duration, num_variations=request.num_variations,
                seed=request.seed,
                progress_callback=lambda p: _update_task(task_id, progress=p),
            )
        elif request.model_name == "bark":
            audio, sr = await BarkService().generate(
                text=request.prompt, duration=request.duration,
                num_variations=request.num_variations, seed=request.seed,
                progress_callback=lambda p: _update_task(task_id, progress=p),
            )
        else:
            raise ValueError(f"Modèle inconnu: {request.model_name}")

        # Sauvegarder l'audio
        audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio)
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)

        file_path = os.path.join(settings.storage_dir, f"{task_id}.wav")
        sf.write(file_path, audio_np.T, sr)
        duration_sec = audio_np.shape[-1] / sr

        with get_db_context() as db:
            gen = db.query(Generation).filter(Generation.id == task_id).first()
            if gen:
                gen.status = "completed"
                gen.audio_url = f"/api/audio/{task_id}.wav"
                gen.audio_duration = duration_sec
                gen.completed_at = datetime.utcnow()
                db.commit()

        _update_task(
            task_id, status="completed", progress=100,
            audio_url=f"/api/audio/{task_id}.wav",
            audio_duration=duration_sec,
            completed_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        print(f"Échec génération {task_id}: {e}")
        _update_task(task_id, status="failed", error=str(e))
        with get_db_context() as db:
            gen = db.query(Generation).filter(Generation.id == task_id).first()
            if gen:
                gen.status = "failed"
                gen.error_message = str(e)
                gen.completed_at = datetime.utcnow()
                db.commit()

# ═══════════════════════════════════════════════════════
#  LAUNCHER (quand exécuté directement via python app.py)
# ═══════════════════════════════════════════════════════

def launch_api():
    """Lance l'API FastAPI en arrière-plan."""
    import subprocess
    print("🚀 Starting FastAPI backend on port 8000...")
    env = os.environ.copy()
    env["PYTHONPATH"] = app_dir + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "uvicorn", "app:api_app",
         "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
        env=env, cwd=app_dir,
    )
    if result.returncode != 0:
        print(f"⚠️ API exited with code: {result.returncode}")

def launch_frontend():
    """Lance le frontend Streamlit."""
    import subprocess
    import time
    # Attendre que l'API démarre
    print("⏳ Waiting for API to start...")
    time.sleep(5)

    print("🚀 Starting Streamlit frontend on port 7860...")
    env = os.environ.copy()
    env["API_URL"] = "http://localhost:8000"
    env["PYTHONPATH"] = app_dir + ":" + env.get("PYTHONPATH", "")
    frontend_file = os.path.join(app_dir, "frontend", "app.py")
    if os.path.exists(frontend_file):
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", frontend_file,
             "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"],
            env=env, cwd=app_dir,
        )
    else:
        print(f"❌ frontend/app.py not found at {frontend_file}")
        sys.exit(1)

if __name__ == "__main__":
    import threading
    print("=" * 50)
    print("  🎵 AI Music Studio - HF Spaces")
    print("=" * 50)
    print(f"  App directory: {app_dir}")
    print(f"  Python: {sys.version}")
    print(f"  Device: {settings.device}")
    print("=" * 50)

    # Lancer l'API dans un thread
    api_thread = threading.Thread(target=launch_api, daemon=True)
    api_thread.start()

    # Lancer le frontend (bloque)
    launch_frontend()
