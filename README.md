---
title: AI Music Studio
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
hardware: gpu
license: mit
---

# 🎵 AI Music Studio

> **Plateforme open source de génération musicale par IA** — Alternative à Suno.ai

## 🧠 Modèles supportés

| Modèle | Type | Durée Max | VRAM Min |
|--------|------|-----------|----------|
| **MusicGen** | Instrumental | 30s | 8GB |
| **Stable Audio Open** | SFX/Samples | 47s | 8GB+ |
| **Bark** | Voix/Jingles | ~13s | 8GB |

## 🚀 Utilisation

1. Sélectionnez un modèle dans la sidebar
2. Entrez votre prompt/description
3. Ajustez les paramètres (durée, créativité, variations)
4. Cliquez sur "Générer"
5. Écoutez et téléchargez le résultat !

## ⚙️ Configuration

Définissez les variables d'environnement dans les Settings du Space :

| Variable | Description | Défaut |
|----------|-------------|--------|
| `API_URL` | URL de l'API backend | `http://localhost:8000` |
| `DEVICE` | Device pour l'inférence | `cuda` |
| `USE_HALF_PRECISION` | Utiliser float16 | `true` |

## 🏗️ Architecture

- **Frontend :** Streamlit
- **Backend :** FastAPI
- **IA :** MusicGen (Meta), Stable Audio (Stability AI), Bark (Suno)
- **GPU Manager :** LRU cache, lazy loading, monitoring VRAM

## 📜 Licence

MIT pour le code source. Licences respectives des modèles IA.
