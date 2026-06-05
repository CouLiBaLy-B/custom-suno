# 🚀 Guide de Déploiement CI/CD - Hugging Face Spaces

Ce document explique comment configurer le déploiement automatique de **AI Music Studio** sur Hugging Face Spaces via GitHub Actions.

---

## 📋 Prérequis

1. ✅ **Compte GitHub** — Le repo `CouLiBaLy-B/custom-suno` existe
2. ✅ **Compte Hugging Face** — Avec un token d'accès
3. ✅ **GPU disponible** — Hugging Face offre des GPU T4 gratuits

---

## 🔑 Étape 1 : Créer un token Hugging Face

1. Allez sur [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Cliquez sur **"New token"**
3. Type : **"Write"**
4. Nommez-le : `github-actions-deploy`
5. Copiez le token (format `hf_xxxxxxxxxxxxxxxxxxxx`)

---

## ⚙️ Étape 2 : Configurer les Secrets & Variables GitHub Actions

### A. Secret HF_TOKEN

1. Repo GitHub → **Settings** → **Secrets and variables** → **Actions**
2. Onglet **Secrets** → **"New repository secret"**
   - **Name** : `HF_TOKEN`
   - **Secret** : `hf_votre_token_ici`

### B. Variables

Onglet **Variables** → Ajoutez :

| Name | Value |
|------|-------|
| `HF_USERNAME` | `CouLiBaLy-B` |
| `HF_SPACE_NAME` | `ai-music-studio` |

---

## 🚀 Étape 3 : Déclencher le déploiement

### Automatique
Chaque push sur `main` déclenche le déploiement.

### Manuel
1. Repo GitHub → **Actions** → **"🚀 Deploy to Hugging Face Spaces"**
2. Cliquez sur **"Run workflow"**

---

## 🌐 Étape 4 : Accéder au Space

URL : `https://huggingface.co/spaces/CouLiBaLy-B/ai-music-studio`

### Configuration du Space
1. **Settings** du Space → **Hardware** → Sélectionnez **T4 Small** (gratuit)
2. **Variables** : `API_URL=http://localhost:8000`, `DEVICE=cuda`
3. **Reboot**

---

## 🔄 Cycle de développement

```
Code local → git push → GitHub Actions (tests) → HF Spaces → Live !
```

---

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| Workflow échoue "HF_TOKEN not configured" | Vérifiez le secret dans Settings |
| Space ne démarre pas | Vérifiez les logs du Space, activez le GPU |
| Erreur "No space left" | Vérifiez le .gitignore (exclut models/, storage/, *.db) |
| Modèle ne charge pas | Activez GPU T4 dans les Settings du Space |

---

## 📊 Coûts

| Ressource | Prix |
|-----------|------|
| GitHub Actions | 🆓 2000 min/mois |
| HF Space T4 | 🆓 Gratuit |
| HF Space A10G | 💰 ~$1.50/h |

Pour le MVP, le **T4 gratuit** suffit !
