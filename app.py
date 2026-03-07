#!/usr/bin/env python3
"""
Cloudera ECS AI Ops — Single-file application
==============================================
Runs the complete stack from one Python file.
No Node.js, no PostgreSQL, no external vector DB required.

Usage:
    python3 app.py                                    # start on port 8000
    python3 app.py --port 9000                        # custom port
    python3 app.py --host 0.0.0.0                     # bind address
    python3 app.py --model-dir  /models/qwen2.5-7b    # LLM from local dir
    python3 app.py --embed-dir  /models/nomic-embed   # embeddings from local dir
    python3 app.py --ingest ./docs                    # ingest docs then start
    python3 app.py --ingest ./docs --force            # re-ingest all docs
    python3 app.py --reload                           # dev auto-reload

Dependencies:
    pip install fastapi "uvicorn[standard]" langchain-ollama langgraph \\
                langchain-core kubernetes python-dotenv psutil         \\
                chromadb sentence-transformers pypdf markdown-it-py

Optional (GPU monitoring):
    pip install nvidia-ml-py

Optional (local LLM without Ollama):
    pip install -U langchain-huggingface transformers torch accelerate

Configuration: edit ENV DEFAULTS below, or set environment variables,
or create an 'env' file alongside this script (key=value format).
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — BOOTSTRAP
# CLI args are parsed early so --model-dir / --embed-dir override env
# before any model is loaded.
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ── Parse CLI early (no --help yet; we re-parse fully at __main__) ────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--model-dir", default=None)
_pre.add_argument("--embed-dir", default=None)
_pre.add_argument("--ingest",    default=None)
_pre.add_argument("--force",     action="store_true")
_pre.add_argument("--port",      type=int, default=8000)
_pre.add_argument("--host",      default="0.0.0.0")
_pre.add_argument("--reload",    action="store_true")
_ARGS, _ = _pre.parse_known_args()

# ── Load env file ─────────────────────────────────────────────────────────────
_env_file = _HERE / "env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

# ── ENV DEFAULTS ──────────────────────────────────────────────────────────────
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("LLM_MODEL",       "qwen2.5:7b")
os.environ.setdefault("EMBED_MODEL",     "nomic-embed-text")
os.environ.setdefault("NUM_THREAD",      "16")
os.environ.setdefault("NUM_CTX",         "4096")
os.environ.setdefault("KUBECONFIG_PATH", "~/kubeconfig")
os.environ.setdefault("PHASE",           "2")
os.environ.setdefault("LOG_LEVEL",       "INFO")
os.environ.setdefault("CHROMA_DIR",      str(_HERE / "chromadb"))
os.environ.setdefault("CUSTOM_RULES",
    "- Do NOT recommend migrating to cgroupv2. This environment uses cgroupv1.")

# ── Apply CLI overrides → env vars (CLI wins over env file) ──────────────────
if _ARGS.model_dir:
    os.environ["LLM_MODEL_DIR"] = _ARGS.model_dir   # signals use of local dir
if _ARGS.embed_dir:
    os.environ["EMBED_DIR"] = _ARGS.embed_dir        # signals local sentence-transformers

# ── Read resolved config ──────────────────────────────────────────────────────
PHASE           = int(os.getenv("PHASE",           "2"))
LLM_MODEL       = os.getenv("LLM_MODEL",            "qwen2.5:7b")
EMBED_MODEL     = os.getenv("EMBED_MODEL",           "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",       "http://localhost:11434")
NUM_THREAD      = int(os.getenv("NUM_THREAD",        "16"))
NUM_CTX         = int(os.getenv("NUM_CTX",           "4096"))
CHROMA_DIR      = os.getenv("CHROMA_DIR",            str(_HERE / "chromadb"))
CUSTOM_RULES    = os.getenv("CUSTOM_RULES",          "").strip()


# ── GPU auto-detection ────────────────────────────────────────────────────────
def _detect_gpu_count() -> int:
    explicit = os.getenv("NUM_GPU")
    if explicit is not None:
        return int(explicit)
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return n
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL)
        return len([l for l in out.decode().strip().splitlines() if l.strip()])
    except Exception:
        pass
    return 0

NUM_GPU = _detect_gpu_count()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOGGING
# ─────────────────────────────────────────────────────────────────────────────

import logging, logging.handlers, time

_LOG_DIR = _HERE / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LEVEL   = getattr(logging, os.getenv("LOG_LEVEL","INFO").upper(), logging.INFO)
_FMT_CON = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
_FMT_FIL = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(filename)s:%(lineno)d  %(message)s"
_DATE    = "%Y-%m-%d %H:%M:%S"
_cfg_set: set = set()

def get_logger(name: str) -> logging.Logger:
    if name in _cfg_set:
        return logging.getLogger(name)
    log = logging.getLogger(name)
    log.setLevel(_LEVEL)
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(_LEVEL)
        ch.setFormatter(logging.Formatter(_FMT_CON, datefmt=_DATE))
        log.addHandler(ch)
        fh = logging.handlers.RotatingFileHandler(
            _LOG_DIR / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
        fh.setLevel(_LEVEL)
        fh.setFormatter(logging.Formatter(_FMT_FIL, datefmt=_DATE))
        log.addHandler(fh)
        log.propagate = False
    _cfg_set.add(name)
    return log

for _noisy in ["httpx","httpcore","urllib3","kubernetes.client",
               "langchain","langsmith","openai","watchfiles","chromadb"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger    = get_logger("app")
_log_rag  = get_logger("rag")
_log_k8s  = get_logger("k8s")
_log_ag   = get_logger("agent")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — KUBERNETES TOOLS
# ─────────────────────────────────────────────────────────────────────────────

from kubernetes import client as _k8s, config as _k8s_cfg
from kubernetes.client.rest import ApiException


def _load_k8s():
    kc = os.getenv("KUBECONFIG_PATH", "")
    try:
        if kc and Path(os.path.expanduser(kc)).exists():
            _k8s_cfg.load_kube_config(config_file=os.path.expanduser(kc))
            _log_k8s.info(f"Loaded kubeconfig: {kc}")
        else:
            _k8s_cfg.load_incluster_config()
            _log_k8s.info("Loaded in-cluster config")
    except Exception as e:
        _log_k8s.error(f"K8s config failed: {e}")
        raise RuntimeError(f"K8s config: {e}")

_load_k8s()
_core = _k8s.CoreV1Api()
_apps = _k8s.AppsV1Api()


def get_pod_status(namespace: str = "all") -> str:
    try:
        pods = _core.list_pod_for_all_namespaces() if namespace == "all" \
               else _core.list_namespaced_pod(namespace=namespace)
        if not pods.items:
            return f"No pods found in '{namespace}'."
        lines = [f"Pods in '{namespace}':"]
        skipped = 0
        for pod in pods.items:
            phase    = pod.status.phase or "Unknown"
            restarts = sum(cs.restart_count for cs in (pod.status.container_statuses or []))
            ready    = sum(1 for cs in (pod.status.container_statuses or []) if cs.ready)
            total    = len(pod.spec.containers)
            if phase in ("Succeeded","Completed"): skipped += 1; continue
            if phase == "Running" and ready == total and restarts == 0: skipped += 1; continue
            bad = [f"{c.type}={c.status}" for c in (pod.status.conditions or []) if c.status != "True"]
            lines.append(f"  {pod.metadata.name}: {phase} | Ready {ready}/{total} | Restarts:{restarts}"
                         + (f" [{', '.join(bad)}]" if bad else ""))
        if skipped:
            lines.append(f"  ({skipped} healthy/completed pods omitted)")
        return "\n".join(lines) if len(lines) > 1 else f"All pods healthy in '{namespace}'."
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_node_health() -> str:
    try:
        nodes = _core.list_node()
        if not nodes.items: return "No nodes found."
        lines = ["Node health:"]
        for node in nodes.items:
            roles    = [k.replace("node-role.kubernetes.io/","")
                        for k in (node.metadata.labels or {})
                        if k.startswith("node-role.kubernetes.io/")] or ["worker"]
            conds    = {c.type: c.status for c in (node.status.conditions or [])}
            pressure = [t for t in ["MemoryPressure","DiskPressure","PIDPressure"]
                        if conds.get(t) == "True"]
            alloc    = node.status.allocatable or {}
            lines.append(f"  {node.metadata.name} [{','.join(roles)}]: Ready={conds.get('Ready','?')}"
                         + (f" ⚠ {','.join(pressure)}" if pressure else "")
                         + f" | CPU:{alloc.get('cpu','n/a')} Mem:{alloc.get('memory','n/a')}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_pod_logs(pod_name: str, namespace: str = "default", tail_lines: int = 50) -> str:
    tail_lines = min(tail_lines, 100)
    try:
        logs = _core.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, tail_lines=tail_lines, timestamps=True)
        return f"Last {tail_lines} lines of '{pod_name}':\n{logs}" if logs.strip() \
               else f"No logs for '{pod_name}'."
    except ApiException as e:
        return f"Pod '{pod_name}' not found." if e.status == 404 else f"K8s error: {e.reason}"


def get_events(namespace: str = "all", warning_only: bool = True) -> str:
    try:
        fs = "type=Warning" if warning_only else ""
        ev = _core.list_event_for_all_namespaces(field_selector=fs) if namespace == "all" \
             else _core.list_namespaced_event(namespace=namespace, field_selector=fs)
        if not ev.items: return f"No {'warning ' if warning_only else ''}events in '{namespace}'."
        sev   = sorted(ev.items, key=lambda e: e.last_timestamp or e.event_time or "", reverse=True)[:20]
        lines = [f"Recent events in '{namespace}':"]
        for e in sev:
            lines.append(f"  [{e.type}] {e.involved_object.kind}/{e.involved_object.name}: "
                         f"{e.reason} — {e.message} (x{e.count or 1})")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_deployment_status(namespace: str = "all") -> str:
    try:
        deps = _apps.list_deployment_for_all_namespaces() if namespace == "all" \
               else _apps.list_namespaced_deployment(namespace=namespace)
        if not deps.items: return f"No deployments in '{namespace}'."
        lines = [f"Deployments in '{namespace}':"]
        for dep in deps.items:
            desired = dep.spec.replicas or 0
            ready   = dep.status.ready_replicas or 0
            avail   = dep.status.available_replicas or 0
            lines.append(f"  {dep.metadata.name}: "
                         f"{'✓ Healthy' if ready==desired and desired>0 else '⚠ Degraded'} "
                         f"| Desired:{desired} Ready:{ready} Available:{avail}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def describe_pod(pod_name: str, namespace: str = "default") -> str:
    try:
        pod   = _core.read_namespaced_pod(name=pod_name, namespace=namespace)
        lines = [f"Pod:{pod.metadata.name}", f"Namespace:{pod.metadata.namespace}",
                 f"Phase:{pod.status.phase}", "Conditions:"]
        for c in (pod.status.conditions or []):
            lines.append(f"  {c.type}:{c.status}" + (f" — {c.message}" if c.message else ""))
        lines.append("Containers:")
        for cs in (pod.status.container_statuses or []):
            sk = list(cs.state.to_dict().keys())[0] if cs.state else "unknown"
            lines.append(f"  {cs.name}: ready={cs.ready} restarts={cs.restart_count} state={sk}")
            if cs.last_state and cs.last_state.terminated:
                lt = cs.last_state.terminated
                lines.append(f"    Last terminated: exit={lt.exit_code} reason={lt.reason}")
        for c in pod.spec.containers:
            if c.resources:
                req = c.resources.requests or {}
                lim = c.resources.limits   or {}
                lines.append(f"  {c.name} resources: "
                             f"req=cpu:{req.get('cpu','none')}/mem:{req.get('memory','none')} "
                             f"lim=cpu:{lim.get('cpu','none')}/mem:{lim.get('memory','none')}")
        return "\n".join(lines)
    except ApiException as e:
        return f"Pod '{pod_name}' not found." if e.status == 404 else f"K8s error: {e.reason}"


K8S_TOOLS = {
    "get_pod_status":        {"fn": get_pod_status,        "description": "List pods and status. Use namespace='all' to scan entire cluster.", "parameters": {"namespace":{"type":"string","default":"all"}}},
    "get_node_health":       {"fn": get_node_health,       "description": "Check node health, CPU/memory pressure, and ready status.",        "parameters": {}},
    "get_pod_logs":          {"fn": get_pod_logs,          "description": "Fetch recent logs from a specific pod.",                           "parameters": {"pod_name":{"type":"string"},"namespace":{"type":"string","default":"default"},"tail_lines":{"type":"integer","default":50}}},
    "get_events":            {"fn": get_events,            "description": "Fetch recent K8s warning events. Always the first step for diagnosing issues.", "parameters": {"namespace":{"type":"string","default":"all"},"warning_only":{"type":"boolean","default":True}}},
    "get_deployment_status": {"fn": get_deployment_status, "description": "Check deployment replica counts and health.",                      "parameters": {"namespace":{"type":"string","default":"all"}}},
    "describe_pod":          {"fn": describe_pod,          "description": "Get detailed info about a specific pod including container states.", "parameters": {"pod_name":{"type":"string"},"namespace":{"type":"string","default":"default"}}},
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RAG: ChromaDB (embedded, zero external deps) + flexible embeddings
# ─────────────────────────────────────────────────────────────────────────────

import re, hashlib
from typing import Optional

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K         = 5

# Module-level singletons — initialised lazily
_chroma_client     = None
_chroma_collection = None
_embedder_fn       = None   # callable: str -> list[float]


# ── Embedder factory ──────────────────────────────────────────────────────────
def _get_embedder():
    """
    Build and cache the embedding function.

    Priority:
      1. EMBED_DIR env / --embed-dir CLI
         → SentenceTransformers loaded from local directory (fully offline)
      2. Fallback
         → Ollama /api/embeddings endpoint (model = EMBED_MODEL)
    """
    global _embedder_fn
    if _embedder_fn is not None:
        return _embedder_fn

    embed_dir = os.getenv("EMBED_DIR", "").strip()
    if embed_dir and Path(embed_dir).exists():
        _log_rag.info(f"[Embed] Loading SentenceTransformer from: {embed_dir}")
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(embed_dir)

        def _local(text: str) -> list:
            return _st.encode(text, normalize_embeddings=True).tolist()

        _embedder_fn = _local
        _log_rag.info("[Embed] Local SentenceTransformer ready")
    else:
        _log_rag.info(f"[Embed] Using Ollama: {OLLAMA_BASE_URL}  model={EMBED_MODEL}")
        import httpx

        def _ollama(text: str) -> list:
            r = httpx.post(f"{OLLAMA_BASE_URL}/api/embeddings",
                           json={"model": EMBED_MODEL, "prompt": text}, timeout=60.0)
            r.raise_for_status()
            return r.json()["embedding"]

        _embedder_fn = _ollama

    return _embedder_fn


def embed_text(text: str) -> list:
    return _get_embedder()(text)


# ── ChromaDB factory ──────────────────────────────────────────────────────────
def _get_chroma():
    """Return (client, collection), opening the persistent store on first call."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_client, _chroma_collection

    import chromadb
    from chromadb.config import Settings

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    _log_rag.info(f"[ChromaDB] Opening persistent store: {CHROMA_DIR}")

    _chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    _chroma_collection = _chroma_client.get_or_create_collection(
        name="k8s_docs",
        metadata={"hnsw:space": "cosine"},
    )
    n = _chroma_collection.count()
    _log_rag.info(f"[ChromaDB] Ready — {n} chunk(s) already stored")
    return _chroma_client, _chroma_collection


def init_db():
    """Ensure ChromaDB directory + collection exist. Called at startup."""
    _get_chroma()


# ── Text helpers ──────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list:
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            pb = text.rfind("\n\n", start, end)
            if pb > start + CHUNK_SIZE // 2:
                end = pb
            else:
                sb = max(text.rfind(". ", start, end), text.rfind(".\n", start, end))
                if sb > start + CHUNK_SIZE // 2:
                    end = sb + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def _doc_type(filename: str) -> str:
    n = filename.lower()
    if any(k in n for k in ["known","issue","bug","error"]):    return "known_issue"
    if any(k in n for k in ["runbook","playbook","procedure"]): return "runbook"
    if any(k in n for k in ["dos","donts","guidelines"]):       return "dos_donts"
    return "general"


# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest_file(file_path: str, force: bool = False) -> dict:
    path  = Path(file_path)
    fhash = hashlib.md5(path.read_bytes()).hexdigest()
    _, col = _get_chroma()

    # Skip unchanged files unless force=True
    if not force:
        existing = col.get(where={"source": str(path)}, limit=1, include=["metadatas"])
        if existing["ids"] and existing["metadatas"]:
            if existing["metadatas"][0].get("file_hash","") == fhash:
                _log_rag.info(f"[RAG] Skip (unchanged): {path.name}")
                return {"file": path.name, "status": "skipped", "chunks": 0}

    # Parse document
    try:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        elif suffix == ".md":
            from markdown_it import MarkdownIt
            html = MarkdownIt().render(path.read_text(encoding="utf-8"))
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        else:
            text = path.read_text(encoding="utf-8")
    except Exception as e:
        _log_rag.error(f"[RAG] Read error {path.name}: {e}")
        return {"file": path.name, "status": "error", "chunks": 0, "error": str(e)}

    if not text.strip():
        return {"file": path.name, "status": "empty", "chunks": 0}

    chunks   = chunk_text(text)
    doc_type = _doc_type(path.name)
    _log_rag.info(f"[RAG] {path.name}: {len(chunks)} chunks  type={doc_type}")

    # Delete old entries for this source, then insert fresh
    try:
        col.delete(where={"source": str(path)})
    except Exception:
        pass  # empty collection raises on delete — safe to ignore

    ids        = [f"{fhash}_{i}" for i in range(len(chunks))]
    metadatas  = [{"source": str(path), "doc_type": doc_type,
                   "chunk_index": i, "file_hash": fhash}
                  for i in range(len(chunks))]
    embeddings = []
    for i, ch in enumerate(chunks):
        embeddings.append(embed_text(ch))
        if (i + 1) % 10 == 0:
            _log_rag.info(f"[RAG]   {i+1}/{len(chunks)} chunks embedded…")

    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    _log_rag.info(f"[RAG] ✓ {path.name} — stored {len(chunks)} chunks")
    return {"file": path.name, "status": "ingested", "chunks": len(chunks), "doc_type": doc_type}


def ingest_directory(docs_dir: str, force: bool = False) -> list:
    p     = Path(docs_dir)
    files = sorted(p.glob("**/*.md")) + sorted(p.glob("**/*.pdf")) + sorted(p.glob("**/*.txt"))
    if not files:
        _log_rag.warning(f"[RAG] No .md / .pdf / .txt files found in {docs_dir}")
        return []
    _log_rag.info(f"[RAG] Ingesting {len(files)} files from {docs_dir}")
    return [ingest_file(str(f), force=force) for f in files]


# ── Retrieval ─────────────────────────────────────────────────────────────────
def rag_retrieve(query: str, top_k: int = TOP_K, doc_type: Optional[str] = None) -> str:
    _, col = _get_chroma()
    n      = col.count()
    if n == 0:
        return "No documents ingested yet. Run: python3 app.py --ingest ./docs"

    qe    = embed_text(query)
    where = {"doc_type": doc_type} if doc_type else None
    try:
        res = col.query(
            query_embeddings=[qe],
            n_results=min(top_k, n),
            where=where,
            include=["documents","metadatas","distances"],
        )
    except Exception as e:
        return f"RAG query failed: {e}"

    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not docs:
        return "No relevant documentation found."

    lines = [f"Retrieved {len(docs)} relevant chunks:\n"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        sim = round(1 - dist, 3)
        src = Path(meta.get("source","?")).name
        dt  = meta.get("doc_type","general")
        lines.append(f"[{i}] {src} | {dt} | relevance:{sim}\n{doc}\n")
    return "\n".join(lines)


def get_doc_stats() -> str:
    _, col = _get_chroma()
    total  = col.count()
    if total == 0:
        return "No documents ingested yet."
    from collections import Counter
    all_meta = col.get(include=["metadatas"])["metadatas"]
    by_type  = Counter(m.get("doc_type","general") for m in all_meta)
    by_src   = Counter(m.get("source","?")        for m in all_meta)
    lines    = [f"ChromaDB: {total} chunks | {len(by_src)} source files"]
    for dt, cnt in sorted(by_type.items()):
        lines.append(f"  {dt}: {cnt} chunks")
    return "\n".join(lines)


RAG_TOOLS = {
    "search_documentation": {
        "fn": rag_retrieve,
        "description": (
            "Search the internal knowledge base for known issues, runbooks, and guidelines. "
            "Always search before finalising a diagnosis. "
            "Cross-reference live cluster data with documentation and cite the source."
        ),
        "parameters": {
            "query":    {"type": "string"},
            "top_k":    {"type": "integer", "default": 5},
            "doc_type": {"type": "string",  "default": None,
                         "description": "Filter by type: known_issue | runbook | dos_donts | general"},
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LANGGRAPH AGENT
# ─────────────────────────────────────────────────────────────────────────────

import json
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

SYSTEM_PROMPT = """You are an expert Kubernetes operations assistant running in an air-gapped environment.

LANGUAGE RULE:
- ALWAYS respond in English only, regardless of what language the user writes in.

ENVIRONMENT CONTEXT:
- Production Kubernetes cluster managed by Cloudera ECS (Embedded Container Service).
- Longhorn is the distributed block storage for persistent volumes (longhorn-system namespace).
  Common Longhorn issues: replica rebuilding, volume degraded, node disk pressure, engine image upgrades.
- Default storage class: 'longhorn'.

TOOL USE — MANDATORY:
You have access to live Kubernetes tools. You MUST call them to answer questions.
DO NOT list kubectl commands. DO NOT explain what you would do. DO NOT say "I would run...".
INSTEAD: call the appropriate tool immediately, then report what it returned.

Available tools and when to use them:
- get_node_health          → always call for cluster health or node questions
- get_pod_status           → always call for pod/workload health (use namespace="all")
- get_events               → always call for warning events or crashloops
- get_deployment_status    → call for deployment replica issues
- get_pod_logs             → call when a specific pod name is known and logs are needed
- describe_pod             → call for detailed diagnosis of a specific pod
- search_documentation     → call to cross-reference known issues and runbooks
{rag_instruction}

CRITICAL RULES:
1. ALWAYS call tools first — never answer from memory alone.
2. NEVER fabricate data — only report what tools actually returned.
3. Be specific — name the exact pod, node, or deployment with the issue.
4. NEVER suggest write operations (restart, delete, scale) — diagnose only.
5. When asked about cluster health, call get_node_health AND get_pod_status AND get_events.
6. For storage issues, ALWAYS check longhorn-system namespace.

SITE-SPECIFIC RULES:
{custom_rules}

RESPONSE FORMAT:
- Concise bullet points only. No lengthy paragraphs.
- State what you found (from tool results), what it means, what to investigate next.
- Skip sections with nothing to report.
- Max ~300 words unless genuinely complex.
"""

RAG_INSTRUCTION = """
7. ALWAYS search documentation before finalising a diagnosis.
8. Cross-reference live data with documentation. Cite the source and recommended fix.
"""


def _make_tool(name: str, cfg: dict):
    fn, desc, params = cfg["fn"], cfg["description"], cfg.get("parameters", {})
    if not params:
        @tool(name, description=desc)
        def _t() -> str:
            return fn()
        return _t
    full_desc = desc + "\nParameters: " + ", ".join(
        f"{k}:{v.get('type','str')}(default={v.get('default','required')})"
        for k, v in params.items())
    @tool(name, description=full_desc)
    def _t(tool_input: str) -> str:
        try:
            kwargs = json.loads(tool_input) if tool_input.strip().startswith("{") else {}
        except json.JSONDecodeError:
            kwargs = {}
        for k, v in params.items():
            if k not in kwargs and "default" in v:
                kwargs[k] = v["default"]
        return fn(**kwargs)
    return _t


class AgentState(TypedDict):
    messages:        Annotated[list, add_messages]
    tool_calls_made: list
    iteration:       int
    status_updates:  list


def _build_llm():
    """
    Build the LangChain LLM.

    Priority:
      1. --model-dir / LLM_MODEL_DIR env  → load from local directory (fully air-gapped)
         Uses langchain-huggingface ChatHuggingFace which supports bind_tools / function calling.
         Falls back to Ollama with the path as model name if langchain-huggingface not installed.
      2. No --model-dir                   → standard Ollama by model name
    """
    model_dir = os.getenv("LLM_MODEL_DIR","").strip()

    if model_dir and Path(model_dir).exists():
        _log_ag.info(f"[LLM] Local directory: {model_dir}")

        # ── Option A: langchain-huggingface ChatHuggingFace ────────────────
        # ChatHuggingFace wraps transformers pipeline and supports bind_tools.
        # Install: pip install -U langchain-huggingface transformers torch accelerate
        try:
            from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
            import transformers, torch
            _log_ag.info("[LLM] Loading HuggingFacePipeline from local dir…")
            pipe = transformers.pipeline(
                "text-generation",
                model=model_dir,
                tokenizer=model_dir,
                max_new_tokens=1024,
                temperature=0.1,
                repetition_penalty=1.1,
                device_map="auto" if NUM_GPU > 0 else "cpu",
                torch_dtype=torch.float16 if NUM_GPU > 0 else torch.float32,
            )
            llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
            _log_ag.info("[LLM] ChatHuggingFace ready (supports tool calling)")
            return llm
        except ImportError:
            _log_ag.warning("[LLM] langchain-huggingface / transformers not installed — "
                            "install with: pip install -U langchain-huggingface transformers torch accelerate\n"
                            "       Falling back to Ollama with local path.")
        except Exception as e:
            _log_ag.warning(f"[LLM] HuggingFace load failed ({e}) — "
                            "falling back to Ollama with local path")

        # ── Option B: Ollama pointing at the local path ─────────────────────
        from langchain_ollama import ChatOllama
        _log_ag.info(f"[LLM] Ollama(local path): {model_dir}")
        return ChatOllama(
            model=model_dir, base_url=OLLAMA_BASE_URL,
            temperature=0.1, num_ctx=NUM_CTX,
            num_thread=NUM_THREAD, num_gpu=NUM_GPU, repeat_penalty=1.1,
        )

    # ── Standard Ollama by model name ──────────────────────────────────────
    from langchain_ollama import ChatOllama
    _log_ag.info(f"[LLM] Ollama: model={LLM_MODEL} ctx={NUM_CTX} "
                 f"threads={NUM_THREAD} gpu={NUM_GPU}")
    return ChatOllama(
        model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0.1, num_ctx=NUM_CTX,
        num_thread=NUM_THREAD, num_gpu=NUM_GPU, repeat_penalty=1.1,
    )


def build_agent():
    all_tools = {**K8S_TOOLS}
    if PHASE >= 2:
        all_tools.update(RAG_TOOLS)

    lc_tools = [_make_tool(n, c) for n, c in all_tools.items()]
    tool_map  = {t.name: t for t in lc_tools}
    prompt    = SYSTEM_PROMPT.format(
        rag_instruction=RAG_INSTRUCTION if PHASE >= 2 else "",
        custom_rules=CUSTOM_RULES or "None.",
    )
    llm = _build_llm().bind_tools(lc_tools)

    TOOL_LABELS = {
        "get_pod_status":        "📦 Checking pod status",
        "get_node_health":       "🖥️  Checking node health",
        "get_pod_logs":          "📋 Fetching pod logs",
        "get_events":            "⚠️  Fetching cluster events",
        "get_deployment_status": "🚀 Checking deployments",
        "describe_pod":          "🔍 Describing pod",
        "search_documentation":  "📚 Searching knowledge base",
    }

    # ── Default tool calls for common queries when model skips tool calling ──
    QUERY_DEFAULTS = [
        (["health","status","check","overview","summary","problem","issue"],
         [("get_node_health",{}), ("get_pod_status",{"namespace":"all"}), ("get_events",{"namespace":"all"})]),
        (["pod","crash","restart","oomkill","pending","failed"],
         [("get_pod_status",{"namespace":"all"}), ("get_events",{"namespace":"all"})]),
        (["node","pressure","memory","disk","resource"],
         [("get_node_health",{}), ("get_events",{"namespace":"all"})]),
        (["deploy","deployment","replica","rollout"],
         [("get_deployment_status",{"namespace":"all"}), ("get_pod_status",{"namespace":"all"})]),
        (["event","warning","alert"],
         [("get_events",{"namespace":"all","warning_only":False})]),
        (["longhorn","storage","volume","pvc","pv"],
         [("get_pod_status",{"namespace":"longhorn-system"}), ("get_events",{"namespace":"longhorn-system"})]),
        (["log"],
         [("get_pod_status",{"namespace":"all"}), ("get_events",{"namespace":"all"})]),
    ]

    def _default_tools_for(user_msg: str):
        """Return fallback tool calls when model does not emit any."""
        lm = user_msg.lower()
        for keywords, calls in QUERY_DEFAULTS:
            if any(k in lm for k in keywords):
                return calls
        return [("get_node_health",{}), ("get_pod_status",{"namespace":"all"})]

    def llm_node(state: AgentState):
        itr     = state.get("iteration", 0) + 1
        updates = list(state.get("status_updates", []))
        updates.append(f"🧠 Reasoning… (iteration {itr})" if itr == 1
                       else f"🔄 Synthesising findings… (iteration {itr})")
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        tcs      = getattr(response, "tool_calls", []) or []

        # ── Fallback: if model returned NO tool calls on iteration 1,
        #    inject default tool calls based on the user query ──────────────
        if not tcs and itr == 1:
            user_msg = ""
            for m in reversed(state["messages"]):
                if hasattr(m, "type") and m.type == "human":
                    user_msg = m.content; break
                elif isinstance(m, HumanMessage):
                    user_msg = m.content; break
            default_calls = _default_tools_for(user_msg)
            _log_ag.info(f"[agent] Model returned no tool calls — injecting defaults: "
                         f"{[n for n,_ in default_calls]}")
            # Build synthetic tool_calls on the response so router sends to tool_node
            import uuid
            synthetic_tcs = []
            for tname, targs in default_calls:
                if tname in tool_map:
                    synthetic_tcs.append({
                        "name": tname, "args": targs,
                        "id": f"auto_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    })
            if synthetic_tcs:
                response.tool_calls = synthetic_tcs
                tcs = synthetic_tcs
                updates.append("⚙️  Auto-invoking live cluster tools…")

        if tcs:
            updates.append(f"🔧 Calling: {', '.join(tc['name'] for tc in tcs)}")
        return {"messages":[response], "tool_calls_made":state.get("tool_calls_made",[]),
                "iteration":itr, "status_updates":updates}

    def tool_node(state: AgentState):
        last         = state["messages"][-1]
        results      = []
        tools_called = list(state.get("tool_calls_made", []))
        updates      = list(state.get("status_updates", []))
        tcs          = getattr(last, "tool_calls", []) or []
        for tc in tcs:
            name = tc["name"]; args = tc.get("args", {})
            tools_called.append(name)
            label = TOOL_LABELS.get(name, f"⚙️ {name}")
            ns    = args.get("namespace","")
            if ns and ns not in ("default","all"): label += f" ({ns})"
            updates.append(label)
            try:
                fn  = tool_map.get(name)
                out = fn.invoke(json.dumps(args) if args else "{}") if fn else f"Tool '{name}' not found."
            except Exception as e:
                out = f"Tool '{name}' failed: {e}"
            results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
        return {"messages":results, "tool_calls_made":tools_called,
                "iteration":state.get("iteration",0), "status_updates":updates}

    def router(state: AgentState) -> Literal["tools","end"]:
        last = state["messages"][-1]
        if state.get("iteration",0) >= 8: return "end"
        tcs = getattr(last, "tool_calls", None)
        return "tools" if tcs else "end"

    g = StateGraph(AgentState)
    g.add_node("llm",   llm_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", router, {"tools":"tools","end":END})
    g.add_edge("tools", "llm")
    return g.compile()


_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def _clean_response(text: str) -> str:
    """Strip chat-template tokens that some models (e.g. Qwen2.5) leak into output."""
    import re
    # Remove complete <|im_start|>role....<|im_end|> blocks
    text = re.sub(r'<\|im_start\|>\w+\s*\n?[\s\S]*?<\|im_end\|>\n?', '', text)
    # If a stray opening token remains, keep only the text after the last one
    if '<|im_start|>' in text:
        last = text.split('<|im_start|>')[-1]
        last = re.sub(r'^\w+\s*\n', '', last, count=1)
        text = last
    # Strip residual end tokens and BOS/EOS markers from other model families
    for tok in ['<|im_end|>', '<s>', '</s>', '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>']:
        text = text.replace(tok, '')
    return text.strip()


async def run_agent(user_message: str) -> dict:
    agent = get_agent()
    t0    = time.time()
    final = await agent.ainvoke({
        "messages":        [HumanMessage(content=user_message)],
        "tool_calls_made": [],
        "iteration":       0,
        "status_updates":  [f"🤖 Model: {os.getenv('LLM_MODEL_DIR', LLM_MODEL)}"],
    })
    elapsed = time.time() - t0
    last    = final["messages"][-1]
    raw     = last.content if hasattr(last, "content") else str(last)
    updates = final.get("status_updates", [])
    updates.append(f"✅ Done in {elapsed:.0f}s")
    return {
        "response":        _clean_response(raw),
        "tools_used":      final.get("tool_calls_made", []),
        "iterations":      final.get("iteration", 0),
        "phase":           PHASE,
        "status_updates":  updates,
        "elapsed_seconds": round(elapsed, 1),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

import psutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"Cloudera ECS AI Ops — Phase {PHASE}")
    model_src = os.getenv("LLM_MODEL_DIR") or f"Ollama:{LLM_MODEL}"
    embed_src = os.getenv("EMBED_DIR")      or f"Ollama:{EMBED_MODEL}"
    logger.info(f"  LLM       : {model_src}")
    logger.info(f"  Embed     : {embed_src}")
    gpu_info = f"{NUM_GPU} GPU(s) — GPU inference" if NUM_GPU > 0 else "No GPU — CPU inference"
    logger.info(f"  GPU       : {gpu_info}")
    logger.info(f"  Threads   : {NUM_THREAD}   CTX: {NUM_CTX}")
    logger.info(f"  ChromaDB  : {CHROMA_DIR}")
    logger.info("=" * 60)

    if PHASE >= 2:
        try:
            init_db()
            _log_rag.info("[ChromaDB] Ready")
        except Exception as e:
            _log_rag.error(f"[ChromaDB] Init failed — RAG unavailable: {e}")

    logger.info("[Agent] Pre-warming LLM…")
    t0 = time.time()
    get_agent()
    logger.info(f"[Agent] Ready in {time.time()-t0:.1f}s")
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Cloudera ECS AI Ops", version="2.0.0", lifespan=_lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def _log_req(request: Request, call_next):
    t0  = time.time()
    res = await call_next(request)
    logger.info(f"[HTTP] {request.method} {request.url.path} → {res.status_code} "
                f"({(time.time()-t0)*1000:.0f}ms)")
    return res


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str;        tools_used: list
    iterations: int;      phase: int
    status_updates: list; elapsed_seconds: float

class IngestRequest(BaseModel):
    docs_dir: str;  force: bool = False

class IngestResponse(BaseModel):
    results: list;  total_files: int;  total_chunks: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "phase":        PHASE,
        "model":        os.getenv("LLM_MODEL_DIR") or LLM_MODEL,
        "model_source": "local_dir" if os.getenv("LLM_MODEL_DIR") else "ollama",
        "embed_source": "local_dir" if os.getenv("EMBED_DIR")      else "ollama",
        "num_gpu":      NUM_GPU,
        "chroma_dir":   CHROMA_DIR,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    try:
        return ChatResponse(**await run_agent(req.message))
    except Exception as e:
        logger.error(f"[Chat] {e}", exc_info=True)
        raise HTTPException(500, f"Agent failed: {e}")


def _gpu_metrics() -> list:
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes): name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except: temp = 0
            try:    pw   = round(pynvml.nvmlDeviceGetPowerUsage(h)/1000.0, 1)
            except: pw   = None
            gpus.append({"index":i,"name":name,"util_pct":util.gpu,
                         "mem_used_gb":round(mem.used/1e9,1),
                         "mem_total_gb":round(mem.total/1e9,1),
                         "mem_pct":round(mem.used/mem.total*100,1),
                         "temp_c":temp,"power_w":pw})
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return gpus


@app.get("/metrics")
async def metrics():
    cpu_per = psutil.cpu_percent(interval=0.2, percpu=True)
    mem     = psutil.virtual_memory()
    freq    = psutil.cpu_freq()
    return {
        "cpu_total":    round(psutil.cpu_percent(interval=None), 1),
        "cpu_per_core": [round(p,1) for p in cpu_per],
        "cpu_count":    psutil.cpu_count(logical=True),
        "freq_mhz":     round(freq.current) if freq else 0,
        "load_avg":     [round(x,2) for x in psutil.getloadavg()],
        "mem_total_gb": round(mem.total/1e9, 1),
        "mem_used_gb":  round(mem.used/1e9,  1),
        "mem_pct":      mem.percent,
        "gpus":         _gpu_metrics(),
        "num_gpu":      NUM_GPU,
    }


@app.get("/namespaces")
async def namespaces():
    try:
        return {"namespaces": [ns.metadata.name for ns in _core.list_namespace().items]}
    except Exception as e:
        return {"namespaces": ["default"], "error": str(e)}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_api(req: IngestRequest):
    if PHASE < 2:
        raise HTTPException(400, "Requires PHASE=2")
    results      = ingest_directory(req.docs_dir, force=req.force)
    total_chunks = sum(r.get("chunks", 0) for r in results)
    return IngestResponse(results=results, total_files=len(results), total_chunks=total_chunks)


@app.get("/docs/stats")
async def doc_stats():
    if PHASE < 2:
        return {"message": "Phase 1 — RAG not enabled"}
    return {"stats": get_doc_stats()}


# ── Serve HTML UI + logos ─────────────────────────────────────────────────────
_INDEX  = _HERE / "index.html"
_STATIC = _HERE / "static"

if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

@app.get("/", response_class=FileResponse)
async def serve_ui():
    if _INDEX.exists():
        return FileResponse(str(_INDEX), media_type="text/html")
    return {"error": "index.html not found"}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cloudera ECS AI Ops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard — Ollama running locally
  python3 app.py

  # Load LLM from local directory (no internet needed)
  python3 app.py --model-dir /models/qwen2.5-7b

  # Load both LLM and embedding model from local directories (fully air-gapped)
  python3 app.py --model-dir /models/qwen2.5-7b --embed-dir /models/nomic-embed-text

  # Ingest docs into ChromaDB, then start
  python3 app.py --ingest ./docs

  # Force re-ingest all docs (even unchanged)
  python3 app.py --ingest ./docs --force

  # Ingest + local models in one command
  python3 app.py --model-dir /models/qwen2.5-7b --embed-dir /models/nomic-embed-text --ingest ./docs

  # Custom host + port
  python3 app.py --host 0.0.0.0 --port 9000
""")

    parser.add_argument("--host",      default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",      type=int, default=8000,
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--model-dir", metavar="PATH", default=None,
                        help="Path to a local LLM directory "
                             "(HuggingFace safetensors or GGUF). "
                             "If provided, skips Ollama model pull.")
    parser.add_argument("--embed-dir", metavar="PATH", default=None,
                        help="Path to a local sentence-transformers model directory. "
                             "If provided, embeddings are computed locally — "
                             "no Ollama embeddings endpoint needed.")
    parser.add_argument("--ingest",    metavar="DOCS_DIR", default=None,
                        help="Directory of .md / .pdf / .txt files to ingest "
                             "into ChromaDB before starting the server.")
    parser.add_argument("--force",     action="store_true",
                        help="Re-ingest all files even if unchanged (use with --ingest).")
    parser.add_argument("--reload",    action="store_true",
                        help="Enable uvicorn auto-reload (development mode).")
    args = parser.parse_args()

    # Apply CLI overrides (idempotent if already set by pre-parse above)
    if args.model_dir:
        os.environ["LLM_MODEL_DIR"] = args.model_dir
    if args.embed_dir:
        os.environ["EMBED_DIR"] = args.embed_dir

    # ── Optional: ingest before starting ────────────────────────────────────
    if args.ingest:
        if PHASE < 2:
            print("ERROR: --ingest requires PHASE=2 (set in env file)")
            sys.exit(1)
        print(f"\n📂 Ingesting documents from: {args.ingest}  (force={args.force})")
        init_db()
        _get_embedder()   # warm up embedder so first file doesn't stall silently
        results = ingest_directory(args.ingest, force=args.force)
        total   = sum(r.get("chunks", 0) for r in results)
        print(f"\n✅  {len(results)} file(s)  |  {total} total chunks stored in ChromaDB\n")
        for r in results:
            icon = "✓" if r["status"] == "ingested" \
                   else "—" if r["status"] == "skipped" else "✗"
            print(f"  {icon}  {r['file']:<42} {r['status']:<10} ({r['chunks']} chunks)")
        print()

    # ── Print banner ─────────────────────────────────────────────────────────
    import uvicorn
    model_src = args.model_dir or LLM_MODEL
    embed_src = args.embed_dir or f"Ollama:{EMBED_MODEL}"
    gpu_str   = f"{NUM_GPU} GPU(s) — GPU inference" if NUM_GPU > 0 else "None — CPU inference"

    print(f"""
╔════════════════════════════════════════════════════════════╗
║            Cloudera ECS AI Ops  v2.0                       ║
╠════════════════════════════════════════════════════════════╣
║  Phase    : {PHASE}                                              ║
║  LLM      : {model_src:<46} ║
║  Embed    : {embed_src:<46} ║
║  GPU      : {gpu_str:<46} ║
║  ChromaDB : {CHROMA_DIR:<46} ║
║  Server   : http://{args.host}:{args.port:<38} ║
╚════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
