# Cloudera ECS AI Ops

An air-gapped Kubernetes operations assistant powered by a local LLM and ChromaDB.
No internet connection required after initial setup.

---

## Architecture

```
User (Browser Chat UI)
      ↓
FastAPI Backend  (app.py)
      ↓
LangGraph Agent  (Reason → Act → Observe → Synthesize)
      ↓               ↓
 K8s Tools        RAG Tool
 (tools_k8s.py)  (ChromaDB + SentenceTransformers)
      ↓               ↓
 Kubernetes       ChromaDB (local, embedded)
  Cluster         + Embedding model (local dir or Ollama)
```

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Qwen2.5 7B — local dir or Ollama | GPU or CPU inference |
| Embeddings | SentenceTransformers (local dir) or Ollama | GPU-accelerated if available |
| Vector DB | ChromaDB (embedded) | Zero external dependencies |
| Agent | LangGraph | Fully local Python, no cloud |
| K8s access | kubernetes Python client | Read-only by default |
| Doc parsing | pypdf + markdown-it-py | Python 3.12 compatible |
| API | FastAPI | REST + CORS |
| Frontend | Single-file HTML/JS | Served directly by FastAPI |

---

## Phases

### Phase 1 — K8s Health Check Bot
- Connects to your Kubernetes cluster (in-cluster or via kubeconfig)
- Read-only: pod status, node health, logs, events, deployments, storage, RBAC
- LLM reasons directly over live cluster data
- No document retrieval — pure cluster diagnostics

### Phase 2 — RAG (Known Issues + Runbooks)
- Ingests your markdown/PDF documents into ChromaDB
- At query time, retrieves the most relevant chunks semantically
- LLM cross-references live K8s observations with known issues
- Cites document source in every diagnosis

Set `PHASE=1` or `PHASE=2` in the `env` file to toggle.

---

## K8s Tools (20 total)

| Category | Tools |
|---|---|
| Pods | `get_pod_status`, `get_pod_logs`, `describe_pod` |
| Nodes | `get_node_health` |
| Events | `get_events` |
| Workloads | `get_deployment_status`, `get_daemonset_status`, `get_statefulset_status`, `get_job_status`, `get_hpa_status` |
| Storage | `get_pvc_status`, `get_persistent_volumes` |
| Networking | `get_service_status`, `get_ingress_status` |
| Config | `get_configmap_list`, `get_resource_quotas`, `get_limit_ranges` |
| RBAC | `get_service_accounts`, `get_cluster_role_bindings` |
| Namespaces | `get_namespace_status` |

---

## Project Structure

```
k8s-ai-ops/
├── app.py               # FastAPI + LangGraph agent + RAG (single file)
├── tools_k8s.py         # All 20 kubectl tool functions + registry
├── index.html           # Chat UI (served by FastAPI at /)
├── requirements.txt     # Python dependencies
├── start.sh             # Start / stop / status helper
├── env                  # Environment config (create from example below)
├── docs/                # Drop .md / .pdf / .txt runbooks here
│   ├── known-issues.md
│   ├── dos-and-donts.md
│   └── longhorn.md
├── static/
│   ├── k8s-logo.svg
│   └── rancher-logo.svg
├── chromadb/            # Auto-created on first run
└── logs/                # Auto-created on first run
    └── app.log
```

---

## Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

For NVIDIA GPU (embedding model on GPU):
```bash
# Replace CPU torch with CUDA build
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Configure environment

Create an `env` file in the same directory as `app.py`:

```ini
# Kubernetes
KUBECONFIG_PATH=~/kubeconfig      # leave blank for in-cluster

# LLM (Ollama, default)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b
EMBED_MODEL=nomic-embed-text

# Or: local model directories (fully air-gapped)
# LLM_MODEL_DIR=/models/qwen2.5-7b
# EMBED_DIR=/models/all-MiniLM-L6-v2

# Performance
NUM_THREAD=16
NUM_CTX=4096
NUM_GPU=1          # 0 = CPU only

# App
PHASE=2
LOG_LEVEL=INFO
CHROMA_DIR=./chromadb
```

### Step 3 — Ingest documents (Phase 2 only)

```bash
python3 app.py --ingest ./docs
python3 app.py --ingest ./docs --force   # re-ingest all
```

### Step 4 — Start the server

```bash
# Ollama (default)
python3 app.py

# Local LLM + local embedding model (fully air-gapped)
python3 app.py --model-dir /models/qwen2.5-7b --embed-dir /models/all-MiniLM-L6-v2

# Custom port
python3 app.py --port 9000
```

Open `http://localhost:8000` in your browser.

---

## Hardware Sizing Guide

| Setup | RAM | LLM | Speed |
|---|---|---|---|
| CPU only, 8-core | 16 GB | `llama3.2:3b` | ~10–15 tok/s |
| CPU only, 16-core | 32 GB | `qwen2.5:7b` (default) | ~6–10 tok/s |
| GPU, 8 GB VRAM | 32 GB | `qwen2.5:7b` local | ~30–60 tok/s |
| GPU, 24 GB VRAM | 64 GB | `qwen2.5:14b` local | ~20–40 tok/s |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend status, model, phase, ChromaDB chunk count |
| POST | `/chat` | Send query, receive diagnosis + tool trace |
| POST | `/ingest` | Trigger doc ingestion into ChromaDB |
| GET | `/docs/stats` | Show ingested document stats |
| GET | `/namespaces` | List available K8s namespaces |
| GET | `/metrics` | CPU, RAM, GPU metrics for the UI |

---

## Adding Your Own Documents

Drop `.md`, `.pdf`, or `.txt` files into `docs/` and run:

```bash
python3 app.py --ingest ./docs
```

Document types are auto-detected from filename keywords:

| Keyword in filename | Tagged as |
|---|---|
| `known`, `issue`, `bug`, `error` | `known_issue` |
| `runbook`, `playbook`, `procedure` | `runbook` |
| `dos`, `donts`, `guidelines` | `dos_donts` |
| anything else | `general` |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `qwen2.5:7b` | Ollama LLM model name |
| `LLM_MODEL_DIR` | — | Path to local LLM directory (overrides Ollama) |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBED_DIR` | — | Path to local SentenceTransformers directory |
| `NUM_THREAD` | `16` | CPU threads |
| `NUM_CTX` | `4096` | Context window size |
| `NUM_GPU` | auto | GPU count (0 = CPU only) |
| `KUBECONFIG_PATH` | `~/kubeconfig` | Path to kubeconfig |
| `PHASE` | `2` | `1` = K8s only · `2` = K8s + RAG |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `CHROMA_DIR` | `./chromadb` | ChromaDB persistent storage path |
| `CUSTOM_RULES` | see env | Site-specific rules injected into system prompt |

---

## Security Notes

- The K8s tools are **read-only**. No write operations (restart, delete, scale) are performed.
- Never expose the FastAPI backend publicly — it has direct cluster access.
- Restrict the env file in production:
  ```bash
  chmod 600 env
  ```
