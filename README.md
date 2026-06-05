# 🎵 AI Music Studio

> **Plateforme open source de génération musicale par IA** — Alternative à Suno.ai

## 🧠 Modèles supportés

| Modèle | Type | Durée Max | VRAM Min | License |
|--------|------|-----------|----------|---------|
| **MusicGen** | Instrumental | 30s | 8GB | CC-BY-NC 4.0 |
| **Stable Audio Open** | SFX/Samples | 47s | 8GB+ | Community |
| **Bark** | Voix/Jingles | ~13s | 8GB | MIT |

## 🚀 Démarrage rapide

```bash
# 1. Cloner et configurer
cd suno-clone
cp .env.example .env

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'API backend
uvicorn backend.api.main:app --reload --port 8000

# 4. Dans un autre terminal, lancer le frontend
streamlit run frontend/app.py --server.port 8501
```

Ou utilisez le script:

```bash
./run.sh api      # API seule
./run.sh frontend # Frontend seul
./run.sh all      # Tout lancer
./run.sh status   # Vérifier les services
```

## 🏗️ Architecture

```
suno-clone/
├── backend/
│   ├── api/main.py              # FastAPI (396 lignes)
│   ├── api/models/models.py     # SQLAlchemy ORM
│   ├── core/config.py           # Configuration Pydantic
│   ├── core/database.py         # SQLAlchemy setup
│   ├── core/gpu_manager.py      # Gestion GPU (LRU, quantization)
│   ├── services/
│   │   ├── musicgen_service.py   # MusicGen (Meta)
│   │   ├── stable_audio_service.py # Stable Audio (Stability AI)
│   │   └── bark_service.py       # Bark (Suno)
│   └── workers/
│       ├── celery_app.py         # Configuration Celery
│       └── tasks/
│           └── musicgen_task.py  # Tâche MusicGen
├── frontend/app.py               # Streamlit (~300 lignes)
├── tests/test_api.py             # Tests unitaires
├── docker-compose.yml            # Orchestration Docker
├── Dockerfile.api                # Image API
├── Dockerfile.worker             # Image Worker
├── Dockerfile.frontend           # Image Frontend
├── requirements.txt              # Dépendances Python
├── run.sh                        # Script de lancement
├── .env.example                  # Variables d'environnement
└── README.md                     # Ce fichier
```

## 📖 API Reference

### Endpoints principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/generate` | Lancer une génération musicale |
| `GET` | `/api/generate/{task_id}` | Statut d'une tâche |
| `GET` | `/api/audio/{filename}` | Télécharger un fichier audio |
| `GET` | `/api/health` | Santé de l'API |
| `GET` | `/api/projects` | Lister les projets |
| `POST` | `/api/projects` | Créer un projet |
| `POST` | `/api/auth/login` | Authentification |
| `GET` | `/api/auth/me` | Utilisateur courant |
| `WS` | `/ws/generation/{task_id}` | Progression temps réel |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "musicgen",
    "prompt": "epic orchestral battle music",
    "duration": 30,
    "temperature": 1.0,
    "cfg_coef": 3.0,
    "num_variations": 2,
    "seed": -1
  }'
```

### Exemple de réponse

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "estimated_time_seconds": 60,
  "message": "Génération 'musicgen' lancée: 'epic orchestral battle music...'"
}
```

## 🧪 Tests

```bash
pytest tests/ -v
pytest tests/test_api.py -v
```

## 📋 Roadmap

- [x] **Phase 1 (MVP)** — Infrastructure, MusicGen, Streamlit
- [ ] **Phase 2** — YuE (lyrics-to-song), Celery, GPU Manager avancé
- [ ] **Phase 3** — Stable Audio Open, Bark, post-traitement
- [ ] **Phase 4** — Auth JWT, bibliothèque complète, optimisations

## 📜 Licence

MIT pour le code. Licences respectives des modèles IA:
- MusicGen: CC-BY-NC 4.0
- Stable Audio Open: Community License  
- Bark: MIT
