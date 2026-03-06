# K8s AI Ops Assistant — Phase 1 & 2

An air-gapped Kubernetes operations assistant powered by a local LLM.
No internet connection required after initial setup. No GPU required.

---

## Architecture

```
User (React Chat)
      ↓
FastAPI Backend
      ↓
LangGraph Agent (Reason → Act → Observe → Synthesize)
      ↓               ↓                    ↓
 K8s Tool         RAG Tool            Memory Tool
 (kubectl)     (pgvector search)    (incident history)
      ↓               ↓
 Kubernetes      PostgreSQL + pgvector
  Cluster        (vectors + logs)
                      ↑
               nomic-embed-text
               (local via Ollama)
```

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Qwen2.5 7B via Ollama | CPU inference, no GPU needed |
| Embeddings | nomic-embed-text via Ollama | 274M params, fast on CPU |
| Vector DB | PostgreSQL + pgvector | Single DB for vectors + logs |
| Agent | LangGraph | Fully local Python, no cloud |
| K8s access | kubernetes Python client | Read-only by default |
| Doc parsing | pypdf + markdown-it-py | Python 3.12 compatible |
| API | FastAPI | REST + CORS |
| Frontend | React + Vite | Connects to FastAPI on port 8000 |

---

## Phases

### Phase 1 — K8s Health Check Bot
- Connects to your Kubernetes cluster (in-cluster or via kubeconfig)
- Read-only operations: pod status, node health, logs, events, deployments
- LLM reasons directly over live cluster data
- No document retrieval — pure cluster diagnostics

### Phase 2 — RAG (Known Issues + Runbooks)
- Ingests your markdown/PDF documents into pgvector
- At query time, retrieves the most relevant chunks semantically
- LLM cross-references live K8s observations with known issues
- Cites document source in every diagnosis

Set `PHASE=1` or `PHASE=2` in the `env` file to toggle.

---

## CPU Inference — No GPU Required

This stack runs entirely on CPU. Here is what to expect:

| Component | CPU Performance | Notes |
|---|---|---|
| nomic-embed-text | Fast (~100–200ms per chunk) | 274M params, single forward pass |
| pgvector search | Instant | Pure SQL, no model involved |
| Qwen2.5 7B inference | ~6–10 tok/s | Slow but workable for ops use |

**Ingestion (`ingest.py`) does not require a GPU.** Embedding is a simple
forward pass — no autoregressive generation. A full document set of
1,000 chunks takes roughly 1–3 minutes on CPU and only needs to run once.

### Hardware Sizing Guide

| CPU | RAM | Recommended Model | Speed |
|---|---|---|---|
| 8-core / 16GB | 16GB | `llama3.2:3b` | ~10–15 tok/s |
| 16-core / 32GB | 32GB | `qwen2.5:7b` ← default | ~6–10 tok/s |
| 32-core / 64GB | 64GB | `qwen2.5:14b` | ~3–5 tok/s |
| 64-core / 128GB | 128GB | `qwen2.5:32b` | ~1–2 tok/s |

---

## Project Structure

```
k8s-ai-ops/
├── start.sh                 # ← Start / stop / status / logs (CentOS 8)
├── .gitignore
├── README.md
│
├── backend/
│   ├── main.py              # FastAPI app — /chat, /ingest, /health
│   ├── ingest.py            # CLI to ingest docs into pgvector
│   ├── requirements.txt     # Python dependencies
│   ├── env                  # Environment config (visible, not hidden)
│   ├── env.example          # Template — copy to env and edit
│   │
│   ├── core/
│   │   └── logger.py        # Centralised logging — all components import this
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   └── agent.py         # LangGraph agent loop
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   └── k8s_tool.py      # Phase 1 — K8s read-only tools
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   └── rag_tool.py      # Phase 2 — embedding + pgvector retrieval
│   │
│   └── logs/                # Auto-created on first run
│       ├── api.log          # FastAPI requests and startup
│       ├── agent.log        # LLM iterations and tool decisions
│       ├── k8s.log          # Every kubectl call and response time
│       ├── rag.log          # Embedding, ingestion, retrieval
│       ├── ingest.log       # Document ingestion CLI
│       ├── uvicorn.log      # Uvicorn server output (start.sh only)
│       ├── frontend.log     # React/Vite output (start.sh only)
│       └── ollama.log       # Ollama server output (start.sh only)
│
├── docs/
│   ├── known-issues.md      # Sample known issues — replace with yours
│   └── dos-and-donts.md     # Sample ops guidelines — replace with yours
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx          # React chat UI
        └── main.jsx         # Entry point
```

---

## Quick Start

### Step 0 — CentOS 8 prerequisites

```bash
# Python 3.12
dnf install python3.12 python3.12-pip -y

# Node.js 18
dnf module install nodejs:18 -y

# Docker (for pgvector container)
dnf install docker -y
systemctl enable --now docker

# Make startup script executable
chmod +x start.sh
```

### Step 1 — Pull models (do this BEFORE air-gapping)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models while internet is available
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

For faster CPU responses on lower-spec hardware:

```bash
ollama pull llama3.2:3b
# then set LLM_MODEL=llama3.2:3b in backend/env
```

### Step 2 — Configure environment

```bash
cd backend
cp env.example env
```

Edit `env` and update:
- `KUBECONFIG_PATH` — path to your kubeconfig file (leave blank if running inside cluster)
- `POSTGRES_PASSWORD` — your Postgres password
- `NUM_THREAD` — set to your physical CPU core count (not hyperthreaded)
- `LLM_MODEL` — change if using a different model
- `LOG_LEVEL` — `DEBUG` for verbose, `INFO` for normal

### Step 3 — Install Python dependencies

```bash
cd backend
pip3.12 install -r requirements.txt
```

### Step 4 — Ingest documents (Phase 2 only)

Add your markdown or PDF runbooks to the `docs/` folder, then:

```bash
cd backend
python3 ingest.py --docs-dir ../docs/
```

Runs entirely on **CPU — no GPU needed**. Safe to re-run — unchanged files
are skipped by hash comparison.

```bash
python3 ingest.py --docs-dir ../docs/ --force   # force re-ingest all
python3 ingest.py --stats                        # show what's ingested
```

### Step 5 — Start everything

```bash
# From the project root
./start.sh
```

That's it. The script starts Ollama, PostgreSQL, FastAPI, and React in order,
waiting for each to be healthy before starting the next.

```
Frontend  → http://localhost:5173
API       → http://localhost:8000
API Docs  → http://localhost:8000/docs
Health    → http://localhost:8000/health
```

---

## start.sh Commands

| Command | Description |
|---|---|
| `./start.sh` | Start all services |
| `./start.sh stop` | Stop all services |
| `./start.sh restart` | Stop then start all services |
| `./start.sh status` | Show running status + port check |
| `./start.sh logs` | Tail all log files live |

---

## Logging

All components write structured logs to `backend/logs/`. Each component
has its own file to make debugging easier.

| Log file | What it contains |
|---|---|
| `api.log` | HTTP requests (method, path, status, ms), startup config |
| `agent.log` | LLM iterations, tool calls decided, total elapsed time |
| `k8s.log` | Every kubectl API call with args and response time |
| `rag.log` | Chunk embedding time, retrieval results, ingestion progress |
| `ingest.log` | Document ingestion CLI output |
| `uvicorn.log` | Raw uvicorn server output (when using start.sh) |
| `frontend.log` | React/Vite dev server output |
| `ollama.log` | Ollama server output |
| `app.log` | Catch-all for anything not covered above |

**Log rotation:** 10MB per file, 5 backups kept automatically.

**Tail all logs live:**
```bash
./start.sh logs

# Or tail a specific component:
tail -f backend/logs/agent.log
tail -f backend/logs/k8s.log
tail -f backend/logs/rag.log
```

**Change log level** in `backend/env`:
```
LOG_LEVEL=DEBUG    # verbose — shows every embed call, chunk count, etc.
LOG_LEVEL=INFO     # normal  — shows tool calls, timings, errors
LOG_LEVEL=WARNING  # quiet   — errors and warnings only
```

---

## Environment Variables

All settings live in `backend/env` (plain filename, not hidden).
Loaded via `load_dotenv("env")` in every component.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `qwen2.5:7b` | LLM model name |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `NUM_THREAD` | `16` | CPU threads — match physical core count |
| `NUM_CTX` | `4096` | Context window size |
| `NUM_GPU` | `0` | GPU layers — 0 = CPU only |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `k8sops` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `KUBECONFIG_PATH` | `~/.kube/config` | Path to kubeconfig file |
| `K8S_NAMESPACE` | `default` | Default K8s namespace |
| `PHASE` | `2` | `1` = K8s only · `2` = K8s + RAG |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend status, model, phase |
| POST | `/chat` | Send query, receive diagnosis + tool trace |
| POST | `/ingest` | Trigger doc ingestion into pgvector |
| GET | `/docs/stats` | Show ingested document stats |
| GET | `/namespaces` | List available K8s namespaces |

---

## Adding Your Own Documents

Drop `.md` or `.pdf` files into the `docs/` folder and re-run `ingest.py`.
The agent automatically uses them during diagnosis.

Suggested documents to add:
- Known issues and their resolutions
- Runbooks for common failure scenarios
- Do's and don'ts for your specific cluster
- Post-mortem reports

Document types are auto-detected from filename keywords:

| Keyword in filename | Tagged as |
|---|---|
| `known`, `issue`, `bug`, `error` | `known_issue` |
| `runbook`, `playbook`, `procedure` | `runbook` |
| `dos`, `donts`, `guidelines`, `policy` | `dos_donts` |
| anything else | `general` |

---

## Security Notes

- The K8s tool is **read-only by default**. Write operations require a
  separate confirmation gate (planned for Phase 4).
- Never expose the FastAPI backend publicly — it has direct cluster access.
- Restrict env file permissions in production:
  ```bash
  chmod 600 backend/env
  ```
