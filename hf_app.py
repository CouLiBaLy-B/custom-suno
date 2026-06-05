"""
AI Music Studio — Hugging Face Spaces Entry Point
Point d'entrée principal pour le déploiement sur HF Spaces.
"""
import os
import subprocess
import sys
import threading

def install_deps():
    """Installe les dépendances manquantes."""
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        print("📦 Installing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
            check=False,
        )
        print("✅ Dependencies installed")

def launch_api():
    """Lance l'API FastAPI en arrière-plan."""
    api_file = os.path.join(os.path.dirname(__file__), "backend", "api", "main.py")
    if os.path.exists(api_file):
        print("🚀 Launching FastAPI backend on port 8000...")
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "backend.api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
            ],
            cwd=os.path.dirname(__file__),
        )

def launch_frontend():
    """Lance le frontend Streamlit."""
    frontend_file = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
    if os.path.exists(frontend_file):
        print(f"🚀 Launching Streamlit frontend on port 7860...")
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                frontend_file,
                "--server.port=7860",
                "--server.address=0.0.0.0",
                "--server.headless=true",
            ],
            env={**os.environ, "API_URL": "http://localhost:8000"},
        )
    else:
        print("❌ frontend/app.py not found")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 50)
    print("  🎵 AI Music Studio - Hugging Face Spaces")
    print("=" * 50)
    
    # Installer les dépendances
    install_deps()
    
    # Lancer l'API en arrière-plan
    api_thread = threading.Thread(target=launch_api, daemon=True)
    api_thread.start()
    
    # Attendre que l'API démarre
    import time
    time.sleep(5)
    
    # Lancer le frontend (bloque)
    launch_frontend()
