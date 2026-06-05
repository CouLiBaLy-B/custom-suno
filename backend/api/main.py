"""
AI Music Studio — Application principale FastAPI
Endpoints REST pour la génération musicale asynchrone.
"""
from __future__ import annotations

# ── CRITIQUE : sys.path en premier (avant tout import custom) ──
import sys, os
_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.core.database import get_db, get_db_context, init_db
from backend.api.models.models import Generation, Project, User

settings = get_settings()
_task_store: dict[str, dict] = {}


def _update_task(task_id: str, **kwargs):
    if task_id in _task_store:
        _task_store[task_id].update(kwargs)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    try:
        init_db()
    except Exception:
        pass
    print("✅ AI Music Studio API démarrée")
    yield
    print("🛑 AI Music Studio API arrêtée")


app = FastAPI(
    title="AI Music Studio",
    description="Plateforme open source de génération musicale par IA",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/generate", response_model=GenerationResponse)
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


@app.get("/api/generate/{task_id}", response_model=GenerationStatus)
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


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    file_path = settings.storage_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    return FileResponse(str(file_path), media_type="audio/wav")


@app.websocket("/ws/generation/{task_id}")
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


@app.get("/api/projects", response_model=list[ProjectResponse])
async def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).all()
    return [
        ProjectResponse(id=p.id, name=p.name, description=p.description,
                        created_at=p.created_at.isoformat())
        for p in projects
    ]


@app.post("/api/projects", response_model=ProjectResponse)
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "AI Music Studio", "version": "0.1.0"}


@app.post("/api/auth/login", response_model=TokenResponse)
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


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me():
    return UserResponse(id="anon", username="anonymous", email="anon@example.com")


async def _run_generation(task_id: str, request: GenerationRequest):
    _update_task(task_id, status="processing", progress=10)
    try:
        if request.model_name == "musicgen":
            from backend.services.musicgen_service import MusicGenService
            audio, sr = await MusicGenService().generate(
                prompt=request.prompt, duration=request.duration,
                temperature=request.temperature, top_k=request.top_k,
                top_p=request.top_p, cfg_coef=request.cfg_coef,
                num_variations=request.num_variations, seed=request.seed,
                progress_callback=lambda p: _update_task(task_id, progress=p),
            )
        elif request.model_name == "stable_audio":
            from backend.services.stable_audio_service import StableAudioService
            audio, sr = await StableAudioService().generate(
                prompt=request.prompt, negative_prompt=request.negative_prompt,
                duration=request.duration, num_variations=request.num_variations,
                seed=request.seed,
                progress_callback=lambda p: _update_task(task_id, progress=p),
            )
        elif request.model_name == "bark":
            from backend.services.bark_service import BarkService
            audio, sr = await BarkService().generate(
                text=request.prompt, duration=request.duration,
                num_variations=request.num_variations, seed=request.seed,
                progress_callback=lambda p: _update_task(task_id, progress=p),
            )
        else:
            raise ValueError(f"Modèle inconnu: {request.model_name}")

        audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio)
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        file_path = settings.storage_dir / f"{task_id}.wav"
        sf.write(str(file_path), audio_np.T, sr)
        duration_sec = audio_np.shape[-1] / sr

        with get_db_context() as db:
            gen = db.query(Generation).filter(Generation.id == task_id).first()
            if gen:
                gen.status = "completed"
                gen.audio_url = f"/api/audio/{task_id}.wav"
                gen.audio_duration = duration_sec
                gen.completed_at = datetime.utcnow()
                db.commit()

        _update_task(task_id, status="completed", progress=100,
                     audio_url=f"/api/audio/{task_id}.wav",
                     audio_duration=duration_sec,
                     completed_at=datetime.utcnow().isoformat())

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
