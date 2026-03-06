#!/usr/bin/env python3
"""
ECS AI Ops — Single-file application
=====================================
Runs the complete stack from one Python file:
  - FastAPI backend (chat, metrics, ingest, health)
  - LangGraph agent with K8s tools + RAG
  - Static frontend served from ./frontend/dist/

Usage:
    python3 app.py                   # start server on port 8000
    python3 app.py --port 9000       # custom port
    python3 app.py --host 0.0.0.0    # bind address (default: 0.0.0.0)
    python3 app.py --ingest ./docs   # ingest docs then start

Dependencies (pip install):
    fastapi uvicorn[standard] langchain-ollama langgraph langchain-core
    kubernetes python-dotenv psutil psycopg2-binary httpx pypdf markdown-it-py

No Node.js or npm required — UI is pure HTML/JS served directly by FastAPI.

Configuration: edit the ENV DEFAULTS section below, or set environment variables,
or create an 'env' file alongside this script (same key=value format).
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — ENV / CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Load 'env' file from same directory if it exists (overrides below defaults)
_env_file = _HERE / "env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

# ── ENV DEFAULTS (edit here or override via env file / environment variables)
os.environ.setdefault("OLLAMA_BASE_URL",    "http://localhost:11434")
os.environ.setdefault("LLM_MODEL",          "qwen2.5:7b")
os.environ.setdefault("EMBED_MODEL",        "nomic-embed-text")
os.environ.setdefault("NUM_THREAD",         "16")
os.environ.setdefault("NUM_CTX",            "4096")
os.environ.setdefault("POSTGRES_HOST",      "localhost")
os.environ.setdefault("POSTGRES_PORT",      "5432")
os.environ.setdefault("POSTGRES_DB",        "k8sops")
os.environ.setdefault("POSTGRES_USER",      "postgres")
os.environ.setdefault("POSTGRES_PASSWORD",  "postgres")
os.environ.setdefault("KUBECONFIG_PATH",    "~/.kube/config")
os.environ.setdefault("PHASE",              "2")
os.environ.setdefault("LOG_LEVEL",          "INFO")
os.environ.setdefault("CUSTOM_RULES",
    "- Do NOT recommend migrating to cgroupv2. This environment uses cgroupv1.")


def _detect_gpu_count() -> int:
    """Auto-detect NVIDIA GPU count. Returns 0 if none found or drivers missing."""
    # Honour explicit override from env/env-file first
    explicit = os.getenv("NUM_GPU")
    if explicit is not None:
        return int(explicit)
    # Try pynvml (nvidia-ml-py) — fast and accurate
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return n
    except Exception:
        pass
    # Fallback: try nvidia-smi subprocess
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL)
        return len([l for l in out.decode().strip().splitlines() if l.strip()])
    except Exception:
        pass
    return 0


PHASE           = int(os.getenv("PHASE", "2"))
LLM_MODEL       = os.getenv("LLM_MODEL",       "qwen2.5:7b")
EMBED_MODEL     = os.getenv("EMBED_MODEL",      "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
NUM_THREAD      = int(os.getenv("NUM_THREAD",   "16"))
NUM_CTX         = int(os.getenv("NUM_CTX",      "4096"))
NUM_GPU         = _detect_gpu_count()   # auto-detected; override via NUM_GPU env var
CUSTOM_RULES    = os.getenv("CUSTOM_RULES",     "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOGGING
# ─────────────────────────────────────────────────────────────────────────────

import logging, logging.handlers

_LOG_DIR = _HERE / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
_FMT_CONSOLE = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
_FMT_FILE    = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(filename)s:%(lineno)d  %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

_configured_loggers: set = set()

def get_logger(name: str) -> logging.Logger:
    if name in _configured_loggers:
        return logging.getLogger(name)
    log = logging.getLogger(name)
    log.setLevel(_LEVEL)
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(_LEVEL)
        ch.setFormatter(logging.Formatter(_FMT_CONSOLE, datefmt=_DATE_FMT))
        log.addHandler(ch)
        fh = logging.handlers.RotatingFileHandler(
            _LOG_DIR / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
        fh.setLevel(_LEVEL)
        fh.setFormatter(logging.Formatter(_FMT_FILE, datefmt=_DATE_FMT))
        log.addHandler(fh)
        log.propagate = False
    _configured_loggers.add(name)
    return log

# Silence noisy libs
for _noisy in ["httpx","httpcore","urllib3","kubernetes.client",
               "langchain","langsmith","openai","watchfiles"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = get_logger("app")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — KUBERNETES TOOLS
# ─────────────────────────────────────────────────────────────────────────────

import time
from kubernetes import client as k8s_client, config as k8s_config
from kubernetes.client.rest import ApiException

_log_k8s = get_logger("k8s")

def _load_k8s():
    kubeconfig = os.getenv("KUBECONFIG_PATH", "")
    try:
        if kubeconfig and Path(os.path.expanduser(kubeconfig)).exists():
            k8s_config.load_kube_config(config_file=os.path.expanduser(kubeconfig))
            _log_k8s.info(f"Loaded kubeconfig: {kubeconfig}")
        else:
            k8s_config.load_incluster_config()
            _log_k8s.info("Loaded in-cluster config")
    except Exception as e:
        _log_k8s.error(f"K8s config failed: {e}")
        raise RuntimeError(f"K8s config: {e}")

_load_k8s()
_core = k8s_client.CoreV1Api()
_apps = k8s_client.AppsV1Api()


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
            if phase in ("Succeeded", "Completed"):
                skipped += 1; continue
            if phase == "Running" and ready == total and restarts == 0:
                skipped += 1; continue
            bad = [f"{c.type}={c.status}" for c in (pod.status.conditions or []) if c.status != "True"]
            lines.append(f"  {pod.metadata.name}: {phase} | Ready {ready}/{total} | Restarts: {restarts}"
                         + (f" [{', '.join(bad)}]" if bad else ""))
        if skipped:
            lines.append(f"  ({skipped} healthy/completed pods omitted)")
        if len(lines) == 1:
            return f"All pods healthy in '{namespace}'."
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_node_health() -> str:
    try:
        nodes = _core.list_node()
        if not nodes.items:
            return "No nodes found."
        lines = ["Node health:"]
        for node in nodes.items:
            roles = [k.replace("node-role.kubernetes.io/","")
                     for k in (node.metadata.labels or {})
                     if k.startswith("node-role.kubernetes.io/")] or ["worker"]
            conds    = {c.type: c.status for c in (node.status.conditions or [])}
            pressure = [t for t in ["MemoryPressure","DiskPressure","PIDPressure"]
                        if conds.get(t) == "True"]
            alloc    = node.status.allocatable or {}
            lines.append(
                f"  {node.metadata.name} [{','.join(roles)}]: Ready={conds.get('Ready','?')}"
                + (f" ⚠ {','.join(pressure)}" if pressure else "")
                + f" | CPU:{alloc.get('cpu','n/a')} Mem:{alloc.get('memory','n/a')}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_pod_logs(pod_name: str, namespace: str = "default", tail_lines: int = 50) -> str:
    tail_lines = min(tail_lines, 100)
    try:
        logs = _core.read_namespaced_pod_log(name=pod_name, namespace=namespace,
                                             tail_lines=tail_lines, timestamps=True)
        return f"Last {tail_lines} lines of '{pod_name}':\n{logs}" if logs.strip() \
               else f"No logs for '{pod_name}'."
    except ApiException as e:
        return f"Pod '{pod_name}' not found." if e.status == 404 else f"K8s error: {e.reason}"


def get_events(namespace: str = "all", warning_only: bool = True) -> str:
    try:
        fs = "type=Warning" if warning_only else ""
        events = _core.list_event_for_all_namespaces(field_selector=fs) if namespace == "all" \
                 else _core.list_namespaced_event(namespace=namespace, field_selector=fs)
        if not events.items:
            return f"No {'warning ' if warning_only else ''}events in '{namespace}'."
        sorted_ev = sorted(events.items,
                           key=lambda e: e.last_timestamp or e.event_time or "", reverse=True)[:20]
        lines = [f"Recent events in '{namespace}':"]
        for ev in sorted_ev:
            lines.append(f"  [{ev.type}] {ev.involved_object.kind}/{ev.involved_object.name}: "
                         f"{ev.reason} — {ev.message} (x{ev.count or 1})")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def get_deployment_status(namespace: str = "all") -> str:
    try:
        deps = _apps.list_deployment_for_all_namespaces() if namespace == "all" \
               else _apps.list_namespaced_deployment(namespace=namespace)
        if not deps.items:
            return f"No deployments in '{namespace}'."
        lines = [f"Deployments in '{namespace}':"]
        for dep in deps.items:
            desired   = dep.spec.replicas or 0
            ready     = dep.status.ready_replicas or 0
            available = dep.status.available_replicas or 0
            status    = "✓ Healthy" if ready == desired and desired > 0 else "⚠ Degraded"
            lines.append(f"  {dep.metadata.name}: {status} | Desired:{desired} Ready:{ready} Available:{available}")
        return "\n".join(lines)
    except ApiException as e:
        return f"K8s API error: {e.reason}"


def describe_pod(pod_name: str, namespace: str = "default") -> str:
    try:
        pod   = _core.read_namespaced_pod(name=pod_name, namespace=namespace)
        lines = [f"Pod: {pod.metadata.name}", f"Namespace: {pod.metadata.namespace}",
                 f"Phase: {pod.status.phase}", "Conditions:"]
        for c in (pod.status.conditions or []):
            lines.append(f"  {c.type}: {c.status}" + (f" — {c.message}" if c.message else ""))
        lines.append("Containers:")
        for cs in (pod.status.container_statuses or []):
            state_key = list(cs.state.to_dict().keys())[0] if cs.state else "unknown"
            lines.append(f"  {cs.name}: ready={cs.ready} restarts={cs.restart_count} state={state_key}")
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
    "get_pod_status":        {"fn": get_pod_status,        "description": "List pods and status. Use namespace='all' to scan entire cluster.", "parameters": {"namespace": {"type":"string","default":"all"}}},
    "get_node_health":       {"fn": get_node_health,       "description": "Check node health, CPU/memory pressure, and ready status.",        "parameters": {}},
    "get_pod_logs":          {"fn": get_pod_logs,          "description": "Fetch recent logs from a specific pod.",                           "parameters": {"pod_name":{"type":"string"},"namespace":{"type":"string","default":"default"},"tail_lines":{"type":"integer","default":50}}},
    "get_events":            {"fn": get_events,            "description": "Fetch recent K8s warning events. First step for diagnosing.",      "parameters": {"namespace":{"type":"string","default":"all"},"warning_only":{"type":"boolean","default":True}}},
    "get_deployment_status": {"fn": get_deployment_status, "description": "Check deployment replica counts and health.",                      "parameters": {"namespace":{"type":"string","default":"all"}}},
    "describe_pod":          {"fn": describe_pod,          "description": "Get detailed info about a specific pod.",                          "parameters": {"pod_name":{"type":"string"},"namespace":{"type":"string","default":"default"}}},
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RAG (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

import re, hashlib
from typing import Optional

_log_rag = get_logger("rag")

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K         = 5


def _db_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST","localhost"),
        port=int(os.getenv("POSTGRES_PORT",5432)),
        dbname=os.getenv("POSTGRES_DB","k8sops"),
        user=os.getenv("POSTGRES_USER","postgres"),
        password=os.getenv("POSTGRES_PASSWORD","postgres"),
    )


def init_db():
    _log_rag.info("Initialising pgvector schema...")
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY, source TEXT NOT NULL, doc_type TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                    embedding vector(768), created_at TIMESTAMPTZ DEFAULT NOW());""")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id SERIAL PRIMARY KEY, file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL, ingested_at TIMESTAMPTZ DEFAULT NOW());""")
        conn.commit()
        _log_rag.info("Schema ready")
    finally:
        conn.close()


def embed_text(text: str) -> list:
    import httpx
    r = httpx.post(f"{OLLAMA_BASE_URL}/api/embeddings",
                   json={"model": EMBED_MODEL, "prompt": text}, timeout=30.0)
    r.raise_for_status()
    return r.json()["embedding"]


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


def ingest_file(file_path: str, force: bool = False) -> dict:
    from psycopg2.extras import execute_values
    path      = Path(file_path)
    file_hash = hashlib.md5(path.read_bytes()).hexdigest()
    conn      = _db_conn()
    try:
        if not force:
            with conn.cursor() as cur:
                cur.execute("SELECT file_hash FROM ingestion_log WHERE file_path=%s", (str(path),))
                row = cur.fetchone()
                if row and row[0] == file_hash:
                    return {"file": path.name, "status": "skipped", "chunks": 0}
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            text = "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        elif path.suffix.lower() == ".md":
            from markdown_it import MarkdownIt
            html = MarkdownIt().render(path.read_text(encoding="utf-8"))
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        else:
            text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {"file": path.name, "status": "empty", "chunks": 0}
        chunks   = chunk_text(text)
        doc_type = _doc_type(path.name)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE source=%s", (str(path),))
        rows = [(str(path), doc_type, i, ch, embed_text(ch)) for i, ch in enumerate(chunks)]
        with conn.cursor() as cur:
            execute_values(cur,
                "INSERT INTO documents (source,doc_type,chunk_index,content,embedding) VALUES %s",
                rows, template="(%s,%s,%s,%s,%s::vector)")
            cur.execute("""INSERT INTO ingestion_log (file_path,file_hash) VALUES (%s,%s)
                           ON CONFLICT (file_path) DO UPDATE
                           SET file_hash=EXCLUDED.file_hash, ingested_at=NOW()""",
                        (str(path), file_hash))
        conn.commit()
        _log_rag.info(f"Ingested {path.name} — {len(chunks)} chunks")
        return {"file": path.name, "status": "ingested", "chunks": len(chunks), "doc_type": doc_type}
    except Exception as e:
        _log_rag.error(f"Ingest failed {path.name}: {e}", exc_info=True)
        return {"file": path.name, "status": "error", "chunks": 0, "error": str(e)}
    finally:
        conn.close()


def ingest_directory(docs_dir: str, force: bool = False) -> list:
    p     = Path(docs_dir)
    files = list(p.glob("**/*.md")) + list(p.glob("**/*.pdf"))
    _log_rag.info(f"Ingesting {len(files)} files from {docs_dir}")
    return [ingest_file(str(f), force=force) for f in sorted(files)]


def rag_retrieve(query: str, top_k: int = TOP_K, doc_type: Optional[str] = None) -> str:
    qe   = embed_text(query)
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            if doc_type:
                cur.execute("""SELECT source,doc_type,content,1-(embedding<=>%s::vector) AS sim
                               FROM documents WHERE doc_type=%s
                               ORDER BY embedding<=>%s::vector LIMIT %s""",
                            (qe, doc_type, qe, top_k))
            else:
                cur.execute("""SELECT source,doc_type,content,1-(embedding<=>%s::vector) AS sim
                               FROM documents ORDER BY embedding<=>%s::vector LIMIT %s""",
                            (qe, qe, top_k))
            rows = cur.fetchall()
        if not rows:
            return "No relevant documentation found."
        lines = [f"Retrieved {len(rows)} relevant chunks:\n"]
        for i, (src, dt, content, sim) in enumerate(rows, 1):
            lines.append(f"[{i}] {Path(src).name} | {dt} | relevance:{sim:.2f}\n{content}\n")
        return "\n".join(lines)
    finally:
        conn.close()


def get_doc_stats() -> str:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_type,COUNT(*),COUNT(DISTINCT source) FROM documents GROUP BY doc_type")
            rows = cur.fetchall()
        if not rows:
            return "No documents ingested yet."
        return "\n".join(["Doc stats:"] + [f"  {dt}: {f} files, {c} chunks" for dt,c,f in rows])
    finally:
        conn.close()


RAG_TOOLS = {
    "search_documentation": {
        "fn": rag_retrieve,
        "description": ("Search internal knowledge base for known issues, runbooks, guidelines. "
                        "Cross-reference live data with documentation before diagnosing."),
        "parameters": {
            "query":    {"type":"string"},
            "top_k":    {"type":"integer","default":5},
            "doc_type": {"type":"string","default":None},
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LANGGRAPH AGENT
# ─────────────────────────────────────────────────────────────────────────────

import json
from typing import Annotated, TypedDict, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

_log_agent = get_logger("agent")

SYSTEM_PROMPT = """You are an expert Kubernetes operations assistant running in an air-gapped environment.

LANGUAGE RULE:
- ALWAYS respond in English only, regardless of what language the user writes in.

ENVIRONMENT CONTEXT:
- Production Kubernetes cluster managed by Cloudera ECS (Embedded Container Service).
- Longhorn is the distributed block storage for persistent volumes (longhorn-system namespace).
  Common Longhorn issues: replica rebuilding, volume degraded, node disk pressure, engine image upgrades.
- Default storage class: 'longhorn'.

CRITICAL RULES:
1. NEVER fabricate data — only report what tools actually returned.
2. Cite the tool result you are reasoning from.
3. Be specific — name the exact pod, node, or deployment with the issue.
4. NEVER suggest write operations (restart, delete, scale) — diagnose only.
5. When asked about cluster health, ALWAYS scan ALL namespaces (namespace='all').
6. For storage issues, ALWAYS check longhorn-system namespace.
{rag_instruction}

SITE-SPECIFIC RULES:
{custom_rules}

RESPONSE FORMAT:
- Concise bullet points only. No lengthy paragraphs.
- State what you found, what it means, what to do — nothing more.
- Skip sections with nothing to report.
- Max ~300 words unless genuinely complex.
"""

RAG_INSTRUCTION = """
7. ALWAYS search documentation before finalising a diagnosis.
8. Cross-reference live data with documentation. Cite the source and fix when matched.
"""


def _make_tool(name: str, cfg: dict):
    fn, desc, params = cfg["fn"], cfg["description"], cfg.get("parameters", {})
    if not params:
        @tool(name, description=desc)
        def _t() -> str:
            return fn()
        return _t
    else:
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


def build_agent():
    all_tools = {**K8S_TOOLS}
    if PHASE >= 2:
        all_tools.update(RAG_TOOLS)

    lc_tools = [_make_tool(n, c) for n, c in all_tools.items()]
    tool_map  = {t.name: t for t in lc_tools}

    prompt = SYSTEM_PROMPT.format(
        rag_instruction=RAG_INSTRUCTION if PHASE >= 2 else "",
        custom_rules=CUSTOM_RULES or "None.",
    )

    llm = ChatOllama(
        model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0.1, num_ctx=NUM_CTX,
        num_thread=NUM_THREAD, num_gpu=NUM_GPU, repeat_penalty=1.1,
    ).bind_tools(lc_tools)

    TOOL_LABELS = {
        "get_pod_status":        "📦 Checking pod status",
        "get_node_health":       "🖥️  Checking node health",
        "get_pod_logs":          "📋 Fetching pod logs",
        "get_events":            "⚠️  Fetching cluster events",
        "get_deployment_status": "🚀 Checking deployments",
        "describe_pod":          "🔍 Describing pod",
        "search_documentation":  "📚 Searching knowledge base",
    }

    def llm_node(state: AgentState):
        itr     = state.get("iteration", 0) + 1
        updates = list(state.get("status_updates", []))
        updates.append(f"🧠 Reasoning... (iteration {itr})" if itr == 1
                       else f"🔄 Synthesising findings... (iteration {itr})")
        messages  = [SystemMessage(content=prompt)] + state["messages"]
        response  = llm.invoke(messages)
        tcs       = getattr(response, "tool_calls", [])
        if tcs:
            updates.append(f"🔧 Calling: {', '.join(tc['name'] for tc in tcs)}")
        return {"messages":[response], "tool_calls_made":state.get("tool_calls_made",[]),
                "iteration":itr, "status_updates":updates}

    def tool_node(state: AgentState):
        last         = state["messages"][-1]
        results      = []
        tools_called = list(state.get("tool_calls_made", []))
        updates      = list(state.get("status_updates", []))
        for tc in last.tool_calls:
            name = tc["name"]
            args = tc["args"]
            tools_called.append(name)
            label = TOOL_LABELS.get(name, f"⚙️ {name}")
            ns    = args.get("namespace", "")
            if ns and ns not in ("default", "all"):
                label += f" ({ns})"
            updates.append(label)
            try:
                fn  = tool_map.get(name)
                out = fn.invoke(json.dumps(args) if args else {}) if fn else f"Tool '{name}' not found."
            except Exception as e:
                out = f"Tool '{name}' failed: {e}"
            results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
        return {"messages":results, "tool_calls_made":tools_called,
                "iteration":state.get("iteration",0), "status_updates":updates}

    def router(state: AgentState) -> Literal["tools","end"]:
        last = state["messages"][-1]
        if state.get("iteration",0) >= 8:
            return "end"
        return "tools" if hasattr(last,"tool_calls") and last.tool_calls else "end"

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


async def run_agent(user_message: str) -> dict:
    agent = get_agent()
    t0    = time.time()
    state = {
        "messages":        [HumanMessage(content=user_message)],
        "tool_calls_made": [],
        "iteration":       0,
        "status_updates":  [f"🤖 Model: {LLM_MODEL}"],
    }
    final   = await agent.ainvoke(state)
    elapsed = time.time() - t0
    last    = final["messages"][-1]
    updates = final.get("status_updates", [])
    updates.append(f"✅ Done in {elapsed:.0f}s")
    return {
        "response":        last.content if hasattr(last,"content") else str(last),
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
    logger.info(f"  Model     : {LLM_MODEL}")
    logger.info(f"  Ollama    : {OLLAMA_BASE_URL}")
    gpu_info = f"{NUM_GPU} GPU(s) detected — inference on GPU" if NUM_GPU > 0 else "No GPU — CPU inference"
    logger.info(f"  Threads   : {NUM_THREAD}  |  CTX: {NUM_CTX}  |  GPU: {gpu_info}")
    logger.info("=" * 60)

    if PHASE >= 2:
        try:
            logger.info("[DB] Initialising pgvector...")
            init_db()
            logger.info("[DB] pgvector ready")
        except Exception as e:
            logger.error(f"[DB] Init failed (RAG unavailable): {e}")

    logger.info("[Agent] Pre-warming LLM...")
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


# ── Pydantic models ───────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    tools_used: list
    iterations: int
    phase: int
    status_updates: list
    elapsed_seconds: float

class IngestRequest(BaseModel):
    docs_dir: str
    force: bool = False

class IngestResponse(BaseModel):
    results: list
    total_files: int
    total_chunks: int


# ── API routes ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status":"ok","phase":PHASE,"model":LLM_MODEL,
            "ollama_url":OLLAMA_BASE_URL,"num_gpu":NUM_GPU,"gpu_auto_detected": NUM_GPU > 0}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    try:
        result = await run_agent(req.message)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"[Chat] Agent error: {e}", exc_info=True)
        raise HTTPException(500, f"Agent failed: {e}")


def _gpu_metrics() -> list:
    """Return per-GPU utilisation, memory, temp. Empty list if no GPUs."""
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            try:
                power_w = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                power_w = None
            gpus.append({
                "index":        i,
                "name":         name,
                "util_pct":     util.gpu,
                "mem_used_gb":  round(mem.used  / 1e9, 1),
                "mem_total_gb": round(mem.total / 1e9, 1),
                "mem_pct":      round(mem.used / mem.total * 100, 1),
                "temp_c":       temp,
                "power_w":      power_w,
            })
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
        "gpus":         _gpu_metrics(),          # [] when no GPU present
        "num_gpu":      NUM_GPU,
    }


@app.get("/namespaces")
async def namespaces():
    try:
        ns_list = [ns.metadata.name for ns in _core.list_namespace().items]
        return {"namespaces": ns_list}
    except Exception as e:
        return {"namespaces":["default"], "error":str(e)}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    if PHASE < 2:
        raise HTTPException(400, "Requires Phase 2")
    results      = ingest_directory(req.docs_dir, force=req.force)
    total_chunks = sum(r.get("chunks",0) for r in results)
    return IngestResponse(results=results, total_files=len(results), total_chunks=total_chunks)


@app.get("/docs/stats")
async def doc_stats():
    if PHASE < 2:
        return {"message":"Phase 1: RAG not enabled"}
    return {"stats": get_doc_stats()}


# ── Serve HTML UI + static assets (no Node.js / npm needed) ─────────────
# index.html lives alongside app.py
# Static assets (logos) are served from ./static/
_INDEX  = _HERE / "index.html"
_STATIC = _HERE / "static"

if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

@app.get("/", response_class=FileResponse)
async def serve_ui():
    """Serve the single-file HTML UI."""
    if _INDEX.exists():
        return FileResponse(str(_INDEX), media_type="text/html")
    return {"error": "index.html not found alongside app.py"}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloudera ECS AI Ops")
    parser.add_argument("--port",   type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host",   type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--ingest", type=str, default=None, metavar="DOCS_DIR",
                        help="Ingest documents from directory before starting")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    # Optional: pre-ingest docs before starting
    if args.ingest:
        if PHASE < 2:
            print("ERROR: --ingest requires PHASE=2")
            sys.exit(1)
        print(f"Ingesting docs from: {args.ingest}")
        init_db()
        results = ingest_directory(args.ingest)
        total   = sum(r.get("chunks",0) for r in results)
        print(f"Ingested {len(results)} files, {total} chunks")
        for r in results:
            print(f"  {r['file']}: {r['status']} ({r['chunks']} chunks)")
        print()

    import uvicorn
    _gpu_str = f"{NUM_GPU} GPU(s) — GPU inference" if NUM_GPU > 0 else "None — CPU inference"
    print(f"""
╔══════════════════════════════════════════════════════╗
║          Cloudera ECS AI Ops  v2.0                   ║
╠══════════════════════════════════════════════════════╣
║  Phase  : {PHASE}                                         ║
║  Model  : {LLM_MODEL:<42} ║
║  GPU    : {_gpu_str:<42} ║
║  Server : http://{args.host}:{args.port:<34} ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",   # uvicorn's own logs — app logs via our handler
    )
