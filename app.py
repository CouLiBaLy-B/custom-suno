"""
AI Music Studio — Single-file Streamlit App for Hugging Face Spaces
Architecture : tout dans un seul fichier, appels directs aux services IA.
Zero subprocess, zero FastAPI séparé, zero problème de port.
"""
from __future__ import annotations

import os
import sys
import json
import uuid
import numpy as np
import soundfile as sf
import streamlit as st
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

# ── sys.path ──
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.chdir(app_dir)

# ── SQLAlchemy imports ──
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════

class Settings:
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///./musicstudio.db")
    storage_path: str = os.environ.get("STORAGE_PATH", "./storage")
    model_cache_dir: str = os.environ.get("MODEL_CACHE_DIR", "./models")
    device: str = os.environ.get("DEVICE", "cpu")
    use_half_precision: bool = os.environ.get("USE_HALF_PRECISION", "true").lower() == "true"
    musicgen_model: str = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
    stable_audio_model: str = os.environ.get("STABLE_AUDIO_MODEL", "stabilityai/stable-audio-open-1.0")
    bark_model: str = os.environ.get("BARK_MODEL", "suno/bark-small")

    @property
    def storage_dir(self):
        d = os.path.abspath(self.storage_path)
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def model_dir(self):
        d = os.path.abspath(self.model_cache_dir)
        os.makedirs(d, exist_ok=True)
        return d

settings = Settings()

# ═══════════════════════════════════════════════════════
#  DATABASE (SQLite local)
# ═══════════════════════════════════════════════════════

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

@contextmanager
def get_db():
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

class Project(Base):
    __tablename__ = "projects"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), default="Nouveau projet")
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

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

# ═══════════════════════════════════════════════════════
#  AI SERVICES (lazy, thread-safe)
# ═══════════════════════════════════════════════════════

import threading

class MusicGenService:
    _instance = None
    _model = None
    _processor = None
    _lock = threading.Lock()

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
        with self._lock:
            if self._model is not None:
                return
            import torch
            st.toast(f"📥 Chargement MusicGen: {settings.musicgen_model}", icon="⏳")
            try:
                from audiocraft.models import MusicGen as MC
                self._model = MC.get_pretrained(settings.musicgen_model)
                self._model = self._model.to(self.device)
                st.toast("✅ MusicGen chargé (audiocraft)", icon="🎸")
            except ImportError:
                from transformers import AutoProcessor, MusicgenForConditionalGeneration
                dtype = torch.float16 if self.device == "cuda" and settings.use_half_precision else torch.float32
                self._processor = AutoProcessor.from_pretrained(settings.musicgen_model)
                self._model = MusicgenForConditionalGeneration.from_pretrained(
                    settings.musicgen_model, torch_dtype=dtype
                )
                self._model = self._model.to(self.device)
                st.toast("✅ MusicGen chargé (transformers)", icon="🎸")

    def generate(self, prompt: str, duration: int = 15, temperature: float = 1.0,
                 top_k: int = 250, top_p: float = 0.0, cfg_coef: float = 3.0,
                 num_variations: int = 1, seed: int = -1):
        import torch
        self.load_model()
        if seed >= 0:
            torch.manual_seed(seed)
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
        print(f"✅ MusicGen done: shape={audio.shape}, sr={sample_rate}")
        return audio, sample_rate


class StableAudioService:
    _instance = None
    _pipeline = None
    _lock = threading.Lock()

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
        with self._lock:
            if self._pipeline is not None:
                return
            import torch
            st.toast(f"📥 Chargement Stable Audio...", icon="⏳")
            from diffusers import StableAudioPipeline
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._pipeline = StableAudioPipeline.from_pretrained(
                settings.stable_audio_model, torch_dtype=dtype
            )
            self._pipeline = self._pipeline.to(self.device)
            st.toast("✅ Stable Audio chargé", icon="🔊")

    def generate(self, prompt: str, negative_prompt: str = None,
                 duration: float = 30.0, num_variations: int = 1,
                 seed: int = -1):
        import torch
        self.load_pipeline()
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None
        print(f"🔊 Generating: '{prompt[:50]}...' ({duration}s)")
        result = self._pipeline(
            prompt=prompt, negative_prompt=negative_prompt or "",
            num_inference_steps=200, audio_end_in_s=min(duration, 47.0),
            num_waveforms_per_prompt=num_variations, generator=generator,
        )
        audio = result.audios[0].cpu().numpy() if num_variations == 1 else result.audios.cpu().numpy()
        sample_rate = self._pipeline.vae.sampling_rate
        print(f"✅ Stable Audio done: shape={audio.shape}, sr={sample_rate}")
        return audio, sample_rate


class BarkService:
    _instance = None
    _model = None
    _processor = None
    _lock = threading.Lock()

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
        with self._lock:
            if self._model is not None:
                return
            st.toast(f"📥 Chargement Bark...", icon="⏳")
            from transformers import AutoProcessor, BarkModel
            self._processor = AutoProcessor.from_pretrained(settings.bark_model)
            self._model = BarkModel.from_pretrained(settings.bark_model)
            self._model = self._model.to(self.device)
            st.toast("✅ Bark chargé", icon="🗣️")

    def generate(self, text: str, voice_preset: str = "v2/fr_speaker_1",
                 duration: int = None, num_variations: int = 1,
                 seed: int = -1):
        import torch
        self.load_model()
        inputs = self._processor(text=text, voice_preset=voice_preset)
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
        sample_rate = 24000
        result = audios[0] if num_variations == 1 else np.concatenate(audios, axis=-1)
        print(f"✅ Bark done: shape={result.shape}, sr={sample_rate}")
        return result, sample_rate

# ═══════════════════════════════════════════════════════
#  SAVE & LOAD HELPERS
# ═══════════════════════════════════════════════════════

def save_audio(audio_data, sample_rate: int) -> str:
    """Sauvegarde l'audio et retourne le chemin du fichier."""
    audio_np = audio_data.cpu().numpy() if hasattr(audio_data, 'cpu') else np.array(audio_data)
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)
    task_id = str(uuid.uuid4())
    file_path = os.path.join(settings.storage_dir, f"{task_id}.wav")
    sf.write(file_path, audio_np.T, sample_rate)
    duration_sec = audio_np.shape[-1] / sample_rate
    return file_path, task_id, duration_sec

def save_generation_to_db(model_name: str, prompt: str, status: str,
                          audio_url: str = None, duration: float = None,
                          error: str = None):
    """Enregistre la génération dans la base de données."""
    try:
        with get_db() as db:
            gen = Generation(
                id=str(uuid.uuid4()),
                user_id="anonymous",
                model_name=model_name,
                prompt=prompt,
                status=status,
                audio_url=audio_url,
                audio_duration=duration,
                error_message=error,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow() if status != "pending" else None,
            )
            db.add(gen)
    except Exception as e:
        print(f"⚠️ Erreur sauvegarde DB: {e}")

def get_generations_from_db(limit: int = 20):
    """Récupère l'historique des générations."""
    try:
        with get_db() as db:
            from sqlalchemy import desc
            return db.query(Generation).order_by(desc(Generation.created_at)).limit(limit).all()
    except Exception:
        return []

# ═══════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════

# Init DB
init_db()
os.makedirs(settings.storage_dir, exist_ok=True)
os.makedirs(settings.model_dir, exist_ok=True)

# ─── CSS ───
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 25px;
        width: 100%;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown('<h1 class="main-header">🎵 AI Music Studio</h1>', unsafe_allow_html=True)
st.caption("Génération musicale IA open source — MusicGen · Stable Audio · Bark")

# ─── Sidebar ───
page = st.sidebar.radio("🎛️ Module", [
    "🎸 MusicGen",
    "🔊 Stable Audio",
    "🗣️ Bark",
    "📚 Bibliothèque",
])

# ═══════════════════════════════════════════════════════
#  PAGE: MusicGen
# ═══════════════════════════════════════════════════════
if page == "🎸 MusicGen":
    st.header("🎸 Génération Instrumentale (MusicGen)")
    c1, c2 = st.columns([2, 1])

    with c1:
        prompt = st.text_area(
            "🎼 Décrivez votre musique",
            placeholder="epic orchestral battle music with dramatic strings",
            height=100,
        )

        st.markdown("### 💡 Prompts d'inspiration")
        presets = {
            "🎻 Orchestral": "epic orchestral battle music",
            "🎸 Rock": "rock with guitars and drums",
            "🎹 Piano": "calm piano melody",
            "🎧 Electro": "electronic dance music",
        }
        cols = st.columns(4)
        for i, (label, text) in enumerate(presets.items()):
            with cols[i]:
                if st.button(label, use_container_width=True):
                    prompt = text
                    st.rerun()

    with c2:
        st.subheader("⚙️ Paramètres")
        dur = st.slider("Durée (s)", 3, 30, 15)
        temp = st.slider("Créativité", 0.1, 2.0, 1.0, 0.1)
        nvar = st.slider("Variations", 1, 4, 1)
        seed = st.number_input("Seed (-1 = aléatoire)", -1, 99999, -1)

    st.markdown("---")
    if st.button("🎵 Générer (MusicGen)", type="primary", use_container_width=True):
        if prompt:
            with st.spinner("🎵 Chargement du modèle et génération..."):
                try:
                    svc = MusicGenService()
                    audio, sr = svc.generate(
                        prompt=prompt, duration=dur, temperature=temp,
                        top_k=250, top_p=0.0, cfg_coef=3.0,
                        num_variations=nvar, seed=seed,
                    )
                    file_path, task_id, duration_sec = save_audio(audio, sr)
                    save_generation_to_db("musicgen", prompt, "completed",
                                          audio_url=f"/api/audio/{task_id}.wav",
                                          duration=duration_sec)

                    st.success(f"✅ Généré ! ({duration_sec:.1f}s, {sr}Hz)")
                    st.audio(file_path, format="audio/wav")
                    with open(file_path, "rb") as f:
                        st.download_button("💾 Télécharger WAV", data=f,
                                          file_name=f"{task_id}.wav", mime="audio/wav")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    save_generation_to_db("musicgen", prompt, "failed", error=str(e))
        else:
            st.error("❌ Entrez une description pour votre musique.")

# ═══════════════════════════════════════════════════════
#  PAGE: Stable Audio
# ═══════════════════════════════════════════════════════
elif page == "🔊 Stable Audio":
    st.header("🔊 Effets Sonores (Stable Audio Open)")
    c1, c2 = st.columns([2, 1])

    with c1:
        prompt = st.text_area(
            "🎼 Décrivez le son",
            placeholder="warm analog synthesizer arpeggio with reverb",
            height=100,
        )
        neg = st.text_area("🚫 Negative prompt", placeholder="low quality, distorted", height=60)

        st.markdown("### 💡 Exemples")
        examples = {
            "🥁 Beat": "hard hitting trap drum beat with 808 bass",
            "🎸 Guitare": "distorted electric guitar riff in D minor",
            "🌊 Ambient": "ocean waves with soft synthesizer pads",
            "🎹 Piano": "classical piano melody with soft reverb",
        }
        cols = st.columns(4)
        for i, (label, text) in enumerate(examples.items()):
            with cols[i]:
                if st.button(label, use_container_width=True):
                    prompt = text
                    st.rerun()

    with c2:
        st.subheader("⚙️ Paramètres")
        dur = st.slider("Durée (s)", 1, 47, 10)
        nvar = st.slider("Variations", 1, 4, 1)
        seed = st.number_input("Seed (-1 = aléatoire)", -1, 99999, -1, key="sa_seed")

    st.markdown("---")
    if st.button("🔊 Générer (Stable Audio)", type="primary", use_container_width=True):
        if prompt:
            with st.spinner("🔊 Génération en cours..."):
                try:
                    svc = StableAudioService()
                    audio, sr = svc.generate(
                        prompt=prompt, negative_prompt=neg, duration=dur,
                        num_variations=nvar, seed=seed,
                    )
                    file_path, task_id, duration_sec = save_audio(audio, sr)
                    save_generation_to_db("stable_audio", prompt, "completed",
                                          audio_url=f"/api/audio/{task_id}.wav",
                                          duration=duration_sec)
                    st.success(f"✅ Généré ! ({duration_sec:.1f}s, {sr}Hz)")
                    st.audio(file_path, format="audio/wav")
                    with open(file_path, "rb") as f:
                        st.download_button("💾 Télécharger WAV", data=f,
                                          file_name=f"{task_id}.wav", mime="audio/wav")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    save_generation_to_db("stable_audio", prompt, "failed", error=str(e))
        else:
            st.error("❌ Entrez une description pour le son.")

# ═══════════════════════════════════════════════════════
#  PAGE: Bark
# ═══════════════════════════════════════════════════════
elif page == "🗣️ Bark":
    st.header("🗣️ Voix & Jingles (Bark)")
    c1, c2 = st.columns([2, 1])

    with c1:
        text = st.text_area(
            "📝 Texte ou paroles",
            placeholder="♪ La la la, c'est ma chanson ♪\n\nUtilisez ♪ autour des paroles pour le chant",
            height=150,
        )

    with c2:
        st.subheader("⚙️ Paramètres")
        voice = st.selectbox(
            "Voix",
            ["v2/fr_speaker_1", "v2/fr_speaker_2", "v2/en_speaker_1",
             "v2/en_speaker_2", "v2/de_speaker_1", "v2/es_speaker_1"],
        )
        nvar = st.slider("Variations", 1, 4, 1)
        seed = st.number_input("Seed (-1 = aléatoire)", -1, 99999, -1, key="bark_seed")

    st.markdown("---")
    if st.button("🗣️ Générer (Bark)", type="primary", use_container_width=True):
        if text:
            with st.spinner("🗣️ Génération vocale en cours..."):
                try:
                    svc = BarkService()
                    audio, sr = svc.generate(
                        text=text, voice_preset=voice, num_variations=nvar, seed=seed,
                    )
                    file_path, task_id, duration_sec = save_audio(audio, sr)
                    save_generation_to_db("bark", text, "completed",
                                          audio_url=f"/api/audio/{task_id}.wav",
                                          duration=duration_sec)
                    st.success(f"✅ Généré ! ({duration_sec:.1f}s, {sr}Hz)")
                    st.audio(file_path, format="audio/wav")
                    with open(file_path, "rb") as f:
                        st.download_button("💾 Télécharger WAV", data=f,
                                          file_name=f"{task_id}.wav", mime="audio/wav")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    save_generation_to_db("bark", text, "failed", error=str(e))
        else:
            st.error("❌ Entrez du texte ou des paroles.")

# ═══════════════════════════════════════════════════════
#  PAGE: Bibliothèque
# ═══════════════════════════════════════════════════════
elif page == "📚 Bibliothèque":
    st.header("📚 Bibliothèque de Générations")

    generations = get_generations_from_db()
    if generations:
        for gen in generations:
            with st.container():
                cols = st.columns([1, 3, 1, 1])
                status_icon = {"completed": "✅", "failed": "❌", "pending": "⏳"}.get(gen.status, "⏳")
                with cols[0]:
                    st.markdown(f"{status_icon} **{gen.model_name}**")
                with cols[1]:
                    st.caption(f"`{gen.prompt[:60]}...`" if len(gen.prompt) > 60 else f"`{gen.prompt}`")
                with cols[2]:
                    if gen.audio_duration:
                        st.caption(f"⏱ {gen.audio_duration:.1f}s")
                with cols[3]:
                    if gen.status == "completed" and gen.audio_url:
                        task_id = gen.audio_url.split("/")[-1].replace(".wav", "")
                        file_path = os.path.join(settings.storage_dir, f"{task_id}.wav")
                        if os.path.exists(file_path):
                            st.audio(file_path, format="audio/wav")
            st.divider()
    else:
        st.info("📂 Aucune génération enregistrée. Commencez par créer de la musique !")

# ─── Footer ───
st.markdown("---")
st.caption(f"🎵 **AI Music Studio** v0.1.0 | Device: {settings.device} | Storage: {settings.storage_dir}")
