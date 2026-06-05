#!/bin/bash
# ============================================================
# AI Music Studio - Script de lancement
# ============================================================
# Usage: ./run.sh [all|api|frontend|worker|stop|status|logs]
# ============================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

API_PORT=8000
STREAMLIT_PORT=8501

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_requirements() {
    log_info "Vérification des prérequis..."
    command -v python3 &>/dev/null || { log_error "Python3 requis"; exit 1; }
    command -v pip3 &>/dev/null || { log_error "pip3 requis"; exit 1; }
    command -v ffmpeg &>/dev/null || log_warning "ffmpeg non installé (requis pour l'audio)"
    log_success "Prérequis vérifiés"
}

install_deps() {
    log_info "Installation des dépendances..."
    [ -d ".venv" ] || python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    log_success "Dépendances installées"
}

start_api() {
    log_info "Démarrage API FastAPI sur port $API_PORT..."
    source .venv/bin/activate
    uvicorn backend.api.main:app --host 0.0.0.0 --port $API_PORT --reload
}

start_frontend() {
    log_info "Démarrage Streamlit sur port $STREAMLIT_PORT..."
    source .venv/bin/activate
    streamlit run frontend/app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0
}

start_worker() {
    log_info "Démarrage worker Celery..."
    source .venv/bin/activate
    celery -A backend.workers.celery_app worker --loglevel=info --concurrency=1
}

start_all() {
    check_requirements
    install_deps
    [ -f ".env" ] || cp .env.example .env 2>/dev/null || true

    log_info "Démarrage de tous les services..."
    start_api &
    sleep 3
    start_frontend &

    echo ""
    log_info "═══════════════════════════════════════"
    log_success "🎵 AI Music Studio est prêt !"
    log_info "═══════════════════════════════════════"
    log_info "Frontend: http://localhost:$STREAMLIT_PORT"
    log_info "API:      http://localhost:$API_PORT/docs"
    log_info "Swagger:  http://localhost:$API_PORT/redoc"
    log_info "═══════════════════════════════════════"
    echo ""
    wait
}

stop_all() {
    log_info "Arrêt de tous les services..."
    pkill -f "uvicorn backend.api.main" 2>/dev/null || true
    pkill -f "streamlit run" 2>/dev/null || true
    pkill -f "celery.*worker" 2>/dev/null || true
    log_success "Services arrêtés"
}

status() {
    log_info "Statut des services:"
    echo ""
    pgrep -f "uvicorn backend.api.main" &>/dev/null && log_success "✅ API: en cours" || log_error "❌ API: arrêtée"
    pgrep -f "streamlit run" &>/dev/null && log_success "✅ Frontend: en cours" || log_error "❌ Frontend: arrêté"
    pgrep -f "celery.*worker" &>/dev/null && log_success "✅ Worker: en cours" || log_error "❌ Worker: arrêté"
    echo ""
}

case "${1:-all}" in
    all) start_all ;;
    api) check_requirements; install_deps; start_api ;;
    frontend) check_requirements; start_frontend ;;
    worker) check_requirements; install_deps; start_worker ;;
    stop) stop_all ;;
    status) status ;;
    install) check_requirements; install_deps ;;
    help|--help|-h)
        echo "Usage: $0 {all|api|frontend|worker|stop|status|install|help}"
        ;;
    *) log_error "Commande inconnue: $1" ;;
esac
