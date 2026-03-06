#!/bin/bash
# =============================================================================
# K8s AI Ops Assistant — Startup Script (CentOS 8)
# Starts: Ollama, PostgreSQL, FastAPI backend, React frontend
#
# Usage:
#   ./start.sh          — start all services
#   ./start.sh stop     — stop all services
#   ./start.sh status   — show service status
#   ./start.sh logs     — tail all logs
#   ./start.sh restart  — stop then start all
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
LOG_DIR="$BACKEND_DIR/logs"
PID_DIR="$SCRIPT_DIR/.pids"

BACKEND_PORT=8000
FRONTEND_PORT=5173
OLLAMA_PORT=11434
POSTGRES_PORT=5432

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────

log()     { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }
header()  { echo -e "\n${BOLD}$*${NC}"; echo "$(echo "$*" | sed 's/./-/g')"; }

mkdir -p "$LOG_DIR" "$PID_DIR"

wait_for_port() {
    local name=$1 port=$2 timeout=${3:-30}
    log "Waiting for $name on port $port..."
    for i in $(seq 1 $timeout); do
        if ss -tlnp | grep -q ":${port} " 2>/dev/null || \
           curl -sf "http://localhost:${port}" >/dev/null 2>&1; then
            success "$name is up (port $port)"
            return 0
        fi
        sleep 1
    done
    error "$name did not start within ${timeout}s"
    return 1
}

save_pid() {
    local name=$1 pid=$2
    echo "$pid" > "$PID_DIR/${name}.pid"
}

read_pid() {
    local name=$1
    local pidfile="$PID_DIR/${name}.pid"
    [[ -f "$pidfile" ]] && cat "$pidfile" || echo ""
}

is_running() {
    local name=$1
    local pid
    pid=$(read_pid "$name")
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_service() {
    local name=$1
    local pid
    pid=$(read_pid "$name")
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Stopping $name (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$PID_DIR/${name}.pid"
        success "$name stopped"
    else
        warn "$name is not running"
        rm -f "$PID_DIR/${name}.pid"
    fi
}

# ── Start functions ───────────────────────────────────────────────────────

start_ollama() {
    header "Starting Ollama"

    if command -v ollama &>/dev/null; then
        if ss -tlnp | grep -q ":${OLLAMA_PORT} "; then
            success "Ollama already running on port $OLLAMA_PORT"
            return 0
        fi
        log "Starting Ollama server..."
        nohup ollama serve \
            >> "$LOG_DIR/ollama.log" 2>&1 &
        save_pid "ollama" $!
        wait_for_port "Ollama" "$OLLAMA_PORT" 30
    else
        warn "Ollama not found in PATH — skipping"
        warn "Install: curl -fsSL https://ollama.com/install.sh | sh"
    fi
}

start_postgres() {
    header "Starting PostgreSQL"

    # Check if already running
    if ss -tlnp | grep -q ":${POSTGRES_PORT} "; then
        success "PostgreSQL already running on port $POSTGRES_PORT"
        return 0
    fi

    # Try systemctl (CentOS 8 native postgres)
    if systemctl is-enabled postgresql &>/dev/null 2>&1; then
        log "Starting PostgreSQL via systemctl..."
        sudo systemctl start postgresql
        success "PostgreSQL started via systemctl"
        return 0
    fi

    # Try Docker pgvector
    if command -v docker &>/dev/null; then
        if docker ps -a --format '{{.Names}}' | grep -q "^pgvector$"; then
            log "Starting existing pgvector container..."
            docker start pgvector >> "$LOG_DIR/postgres.log" 2>&1
        else
            log "Creating new pgvector container..."
            # Read credentials from env file
            source <(grep -E '^POSTGRES_' "$BACKEND_DIR/env" 2>/dev/null || true)
            docker run -d --name pgvector \
                -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}" \
                -e POSTGRES_DB="${POSTGRES_DB:-k8sops}" \
                -e POSTGRES_USER="${POSTGRES_USER:-postgres}" \
                -p "${POSTGRES_PORT}:5432" \
                pgvector/pgvector:pg16 \
                >> "$LOG_DIR/postgres.log" 2>&1
        fi
        wait_for_port "PostgreSQL" "$POSTGRES_PORT" 30
        success "PostgreSQL (Docker) is up"
    else
        error "Neither systemctl postgresql nor Docker found"
        error "Install Docker: dnf install docker -y && systemctl start docker"
        exit 1
    fi
}

start_backend() {
    header "Starting FastAPI Backend"

    if is_running "backend"; then
        success "Backend already running (PID $(read_pid backend))"
        return 0
    fi

    if [[ ! -f "$BACKEND_DIR/main.py" ]]; then
        error "main.py not found in $BACKEND_DIR"
        exit 1
    fi

    # Check Python
    if ! command -v python3 &>/dev/null; then
        error "python3 not found"
        exit 1
    fi

    log "Starting FastAPI backend on port $BACKEND_PORT..."
    cd "$BACKEND_DIR"

    nohup python3 -m uvicorn main:app \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --log-level info \
        --access-log \
        >> "$LOG_DIR/uvicorn.log" 2>&1 &

    save_pid "backend" $!
    cd "$SCRIPT_DIR"

    wait_for_port "FastAPI Backend" "$BACKEND_PORT" 60
    success "Backend started (PID $(read_pid backend))"
    success "API docs: http://localhost:${BACKEND_PORT}/docs"
}

start_frontend() {
    header "Starting React Frontend"

    if is_running "frontend"; then
        success "Frontend already running (PID $(read_pid frontend))"
        return 0
    fi

    if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
        error "package.json not found in $FRONTEND_DIR"
        exit 1
    fi

    # Check Node
    if ! command -v node &>/dev/null; then
        error "Node.js not found"
        error "Install: dnf module install nodejs:18 -y"
        exit 1
    fi

    # Install deps if needed
    if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
        log "Installing npm dependencies..."
        cd "$FRONTEND_DIR"
        npm install >> "$LOG_DIR/npm.log" 2>&1
        cd "$SCRIPT_DIR"
        success "npm dependencies installed"
    fi

    log "Starting React frontend on port $FRONTEND_PORT..."
    cd "$FRONTEND_DIR"

    nohup npm run dev -- --host \
        >> "$LOG_DIR/frontend.log" 2>&1 &

    save_pid "frontend" $!
    cd "$SCRIPT_DIR"

    wait_for_port "React Frontend" "$FRONTEND_PORT" 30
    success "Frontend started (PID $(read_pid frontend))"
    success "UI: http://localhost:${FRONTEND_PORT}"
}

# ── Status ────────────────────────────────────────────────────────────────

show_status() {
    header "Service Status"

    # Ollama and Postgres may be started externally (systemctl, docker, etc.)
    # so we check their port directly rather than relying on a PID file.
    # Backend and Frontend are always managed by start.sh so PID file is used,
    # with a port fallback in case they were started manually.

    _port_up() { ss -tlnp 2>/dev/null | grep -q ":${1} "; }

    # Ollama — port check only
    if _port_up "$OLLAMA_PORT"; then
        echo -e "  ${GREEN}●${NC} Ollama — running (:$OLLAMA_PORT)"
    else
        echo -e "  ${RED}●${NC} Ollama — stopped"
    fi

    # Postgres — port check only
    if _port_up "$POSTGRES_PORT"; then
        echo -e "  ${GREEN}●${NC} Postgres — running (:$POSTGRES_PORT)"
    else
        echo -e "  ${RED}●${NC} Postgres — stopped"
    fi

    # Backend — PID file preferred, fall back to port check
    if is_running "backend"; then
        pid=$(read_pid "backend")
        echo -e "  ${GREEN}●${NC} Backend (PID $pid) — running (:$BACKEND_PORT)"
    elif _port_up "$BACKEND_PORT"; then
        echo -e "  ${GREEN}●${NC} Backend — running (:$BACKEND_PORT, external)"
    else
        echo -e "  ${RED}●${NC} Backend — stopped"
    fi

    # Frontend — PID file preferred, fall back to port check
    if is_running "frontend"; then
        pid=$(read_pid "frontend")
        echo -e "  ${GREEN}●${NC} Frontend (PID $pid) — running (:$FRONTEND_PORT)"
    elif _port_up "$FRONTEND_PORT"; then
        echo -e "  ${GREEN}●${NC} Frontend — running (:$FRONTEND_PORT, external)"
    else
        echo -e "  ${RED}●${NC} Frontend — stopped"
    fi

    echo ""
    log "Log files in $LOG_DIR:"
    ls -lh "$LOG_DIR"/*.log 2>/dev/null || echo "  (none yet)"
}

# ── Logs ──────────────────────────────────────────────────────────────────

tail_logs() {
    header "Tailing all logs (Ctrl+C to stop)"
    log "Log files: $LOG_DIR"
    tail -f \
        "$LOG_DIR/uvicorn.log" \
        "$LOG_DIR/agent.log" \
        "$LOG_DIR/k8s.log" \
        "$LOG_DIR/rag.log" \
        "$LOG_DIR/frontend.log" \
        2>/dev/null || warn "Some log files not yet created"
}

# ── Stop all ──────────────────────────────────────────────────────────────

stop_all() {
    header "Stopping all services"
    stop_service "frontend"
    stop_service "backend"
    # Only stop ollama/postgres if we started them (pid file exists)
    [[ -f "$PID_DIR/ollama.pid" ]]   && stop_service "ollama"
    [[ -f "$PID_DIR/postgres.pid" ]] && stop_service "postgres" || true
    success "All services stopped"
}

# ── Main ──────────────────────────────────────────────────────────────────

COMMAND="${1:-start}"

case "$COMMAND" in
    start)
        echo -e "\n${BOLD}${CYAN}K8s AI Ops Assistant — Starting${NC}"
        echo "=================================================="
        start_ollama
        start_postgres
        start_backend
        start_frontend
        echo ""
        echo "=================================================="
        success "All services started"
        echo ""
        echo -e "  ${CYAN}Frontend${NC}  → http://localhost:${FRONTEND_PORT}"
        echo -e "  ${CYAN}API${NC}       → http://localhost:${BACKEND_PORT}"
        echo -e "  ${CYAN}API Docs${NC}  → http://localhost:${BACKEND_PORT}/docs"
        echo -e "  ${CYAN}Health${NC}    → http://localhost:${BACKEND_PORT}/health"
        echo ""
        echo -e "  ${CYAN}Logs${NC}      → $LOG_DIR"
        echo -e "  Tail logs : ./start.sh logs"
        echo ""
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        exec "$0" start
        ;;
    status)
        show_status
        ;;
    logs)
        tail_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
