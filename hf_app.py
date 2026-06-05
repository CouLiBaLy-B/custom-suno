"""
AI Music Studio — Hugging Face Spaces Entry Point
Lance le backend FastAPI ET le frontend Streamlit.
"""
import os
import sys
import subprocess
import threading
import time

# Ajouter le dossier courant au PYTHONPATH
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

os.chdir(app_dir)

def launch_api():
    """Lance l'API FastAPI en arrière-plan."""
    print("🚀 Launching FastAPI backend on port 8000...")
    env = os.environ.copy()
    env["PYTHONPATH"] = app_dir + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "backend.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info",
        ],
        env=env,
        cwd=app_dir,
    )
    print(f"⚠️ API process exited with code: {result.returncode}")

def launch_frontend():
    """Lance le frontend Streamlit."""
    # Attendre que l'API démarre
    print("⏳ Attente du démarrage de l'API...")
    time.sleep(5)
    
    print("🚀 Launching Streamlit frontend on port 7860...")
    env = os.environ.copy()
    env["API_URL"] = "http://localhost:8000"
    env["PYTHONPATH"] = app_dir + ":" + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            os.path.join(app_dir, "frontend", "app.py"),
            "--server.port=7860",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
        env=env,
        cwd=app_dir,
    )

if __name__ == "__main__":
    print("=" * 50)
    print("  🎵 AI Music Studio - HF Spaces")
    print("=" * 50)
    print(f"  App directory: {app_dir}")
    print(f"  Python: {sys.version}")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    print("=" * 50)
    
    # Lancer l'API dans un thread
    api_thread = threading.Thread(target=launch_api, daemon=True)
    api_thread.start()
    
    # Lancer le frontend (bloque)
    launch_frontend()
