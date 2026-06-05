"""
AI Music Studio — Hugging Face Spaces Entry Point
Architecture bulletproof : PYTHONPATH + imports absolus
"""
import os
import sys

# ── CRITIQUE : PYTHONPATH avant TOUT autre import ──
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.chdir(app_dir)
os.environ["PYTHONPATH"] = app_dir + ":" + os.environ.get("PYTHONPATH", "")

print(f"📁 App dir: {app_dir}")
print(f"📁 sys.path[0]: {sys.path[0]}")
print(f"📁 backend exists: {os.path.isdir(os.path.join(app_dir, 'backend'))}")
print(f"📁 backend/api exists: {os.path.isdir(os.path.join(app_dir, 'backend', 'api'))}")
print(f"📁 backend/api/models exists: {os.path.isdir(os.path.join(app_dir, 'backend', 'api', 'models'))}")

# ── Lancer l'API FastAPI ──
def launch_api():
    """Lance l'API FastAPI en arrière-plan."""
    import subprocess
    print("🚀 Starting FastAPI backend on port 8000...")
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
    if result.returncode != 0:
        print(f"⚠️ API exited with code: {result.returncode}")

# ── Lancer le frontend Streamlit ──
def launch_frontend():
    """Lance le frontend Streamlit."""
    import subprocess
    import time
    time.sleep(5)  # Attendre que l'API démarre
    
    print("🚀 Starting Streamlit frontend on port 7860...")
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

# ── Main ──
if __name__ == "__main__":
    import threading
    print("=" * 50)
    print("  🎵 AI Music Studio - HF Spaces")
    print("=" * 50)
    
    api_thread = threading.Thread(target=launch_api, daemon=True)
    api_thread.start()
    
    launch_frontend()
