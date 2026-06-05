"""
AI Music Studio — Interface Streamlit complète
"""
from __future__ import annotations
import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="🎵 AI Music Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ───
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
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 25px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─── Fonctions utilitaires ───
def check_api():
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def submit_generation(req: dict):
    r = requests.post(f"{API_URL}/api/generate", json=req, timeout=10)
    return r.json() if r.status_code == 200 else None


def get_status(task_id: str):
    r = requests.get(f"{API_URL}/api/generate/{task_id}", timeout=5)
    return r.json() if r.status_code == 200 else None


def download_audio(task_id: str):
    r = requests.get(f"{API_URL}/api/audio/{task_id}.wav", timeout=30)
    return r.content if r.status_code == 200 else None


# ─── Header ───
st.markdown('<h1 class="main-header">🎵 AI Music Studio</h1>', unsafe_allow_html=True)
st.caption("Génération musicale IA open source — MusicGen · Stable Audio · Bark")

api_ok = check_api()
if not api_ok:
    st.warning("⚠️ API backend non disponible. Les appels de génération échoueront.")

# ─── Sidebar ───
page = st.sidebar.radio("🎛️ Module", [
    "🎸 MusicGen",
    "🔊 Stable Audio",
    "🗣️ Bark",
    "📚 Bibliothèque",
])

# ─── Page: MusicGen ───
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
            "🎷 Jazz": "smooth jazz with saxophone",
            "🌿 Ambient": "ambient nature soundscape",
        }

        cols = st.columns(3)
        for i, (label, text) in enumerate(presets.items()):
            with cols[i % 3]:
                if st.button(label, use_container_width=True):
                    st.session_state["mg_prompt"] = text
                    st.rerun()

        prompt = prompt or st.session_state.get("mg_prompt", "")

    with c2:
        st.subheader("⚙️ Paramètres")
        dur = st.slider("Durée (s)", 3, 30, 15)
        temp = st.slider("Créativité", 0.1, 2.0, 1.0, 0.1)
        nvar = st.slider("Variations", 1, 4, 1)
        seed = st.number_input("Seed (-1 = aléatoire)", -1, 99999, -1)

    st.markdown("---")
    if st.button("🎵 Générer (MusicGen)", type="primary", use_container_width=True):
        if prompt:
            res = submit_generation({
                "model_name": "musicgen",
                "prompt": prompt,
                "duration": dur,
                "temperature": temp,
                "num_variations": nvar,
                "seed": seed,
            })
            if res:
                st.session_state["task"] = res["task_id"]
                st.success(f"✅ Lancé ! Task ID: `{res['task_id']}`")
        else:
            st.error("❌ Entrez une description pour votre musique.")

# ─── Page: Stable Audio ───
elif page == "🔊 Stable Audio":
    st.header("🔊 Effets Sonores (Stable Audio Open)")
    c1, c2 = st.columns([2, 1])

    with c1:
        prompt = st.text_area(
            "🎼 Décrivez le son",
            placeholder="warm analog synthesizer arpeggio with reverb",
            height=100,
        )
        neg = st.text_area(
            "🚫 Negative prompt (ce qu'on ne veut PAS)",
            placeholder="low quality, distorted, noisy",
            height=60,
        )

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
                    st.session_state["sa_prompt"] = text
                    st.rerun()

        prompt = prompt or st.session_state.get("sa_prompt", "")

    with c2:
        st.subheader("⚙️ Paramètres")
        dur = st.slider("Durée (s)", 1, 47, 10)
        nvar = st.slider("Variations", 1, 4, 1)
        seed = st.number_input("Seed (-1 = aléatoire)", -1, 99999, -1)

    st.markdown("---")
    if st.button("🔊 Générer (Stable Audio)", type="primary", use_container_width=True):
        if prompt:
            res = submit_generation({
                "model_name": "stable_audio",
                "prompt": prompt,
                "negative_prompt": neg,
                "duration": dur,
                "num_variations": nvar,
                "seed": seed,
            })
            if res:
                st.session_state["task"] = res["task_id"]
                st.success(f"✅ Lancé ! Task ID: `{res['task_id']}`")
        else:
            st.error("❌ Entrez une description pour le son.")

# ─── Page: Bark ───
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
             "v2/en_speaker_2", "v2/de_speaker_1", "v2/es_speaker_1",
             "v2/hi_speaker_1", "v2/zh_speaker_1", "v2/ja_speaker_1"],
        )
        nvar = st.slider("Variations", 1, 4, 1)
        seed = st.number_input("Seed (-1 = aléatoire)", -1, 99999, -1)

    st.markdown("---")
    if st.button("🗣️ Générer (Bark)", type="primary", use_container_width=True):
        if text:
            res = submit_generation({
                "model_name": "bark",
                "prompt": text,
                "num_variations": nvar,
                "seed": seed,
            })
            if res:
                st.session_state["task"] = res["task_id"]
                st.success(f"✅ Lancé ! Task ID: `{res['task_id']}`")
        else:
            st.error("❌ Entrez du texte ou des paroles.")

# ─── Page: Bibliothèque ───
elif page == "📚 Bibliothèque":
    st.header("📚 Bibliothèque de Générations")

    tid = st.session_state.get("task")
    if tid:
        status = get_status(tid)
        if status:
            st.write(f"**Task ID:** `{tid}`")
            st.write(f"**Statut:** `{status['status']}`")
            st.write(f"**Progression:** {status['progress']}%")
            st.progress(status["progress"] / 100)

            if status.get("current_step"):
                st.info(status["current_step"])

            if status["status"] == "completed":
                st.success("✅ Génération terminée !")
                data = download_audio(tid)
                if data:
                    st.audio(data, format="audio/wav")
                    st.download_button(
                        label="💾 Télécharger en WAV",
                        data=data,
                        file_name=f"{tid}.wav",
                        mime="audio/wav",
                    )
            elif status["status"] == "failed":
                st.error(f"❌ Échec: {status.get('error', 'Erreur inconnue')}")
    else:
        st.info("📂 Aucune génération en cours. Allez dans un module pour créer de la musique !")

# ─── Footer ───
st.markdown("---")
st.caption("🎵 **AI Music Studio** — Open Source | MIT License")
