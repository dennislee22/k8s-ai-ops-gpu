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
    pip install fastapi "uvicorn[standard]" langchain-ollama langgraph \
                langchain-core kubernetes python-dotenv psutil          \
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
    os.environ["LLM_MODEL_DIR"] = _ARGS.model_dir
if _ARGS.embed_dir:
    os.environ["EMBED_DIR"] = _ARGS.embed_dir

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
_LEVEL   = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
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
            _LOG_DIR / "app.log", maxBytes=10*1024*1024, backupCount=5,
            encoding="utf-8")
        fh.setLevel(_LEVEL)
        fh.setFormatter(logging.Formatter(_FMT_FIL, datefmt=_DATE))
        log.addHandler(fh)
        log.propagate = False
    _cfg_set.add(name)
    return log

for _noisy in ["httpx", "httpcore", "urllib3", "kubernetes.client",
               "langchain", "langsmith", "openai", "watchfiles", "chromadb"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger   = get_logger("app")
_log_rag = get_logger("rag")
_log_ag  = get_logger("agent")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — KUBERNETES TOOLS
# ─────────────────────────────────────────────────────────────────────────────

from tools_k8s import K8S_TOOLS, _core   # _core re-exported for /namespaces route


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RAG: ChromaDB + flexible embeddings (Ollama or local GPU)
# ─────────────────────────────────────────────────────────────────────────────

import re, hashlib
from typing import Optional

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K         = 5

# Module-level singletons — initialised at startup, not lazily
_chroma_client     = None
_chroma_collection = None
_embedder_fn       = None   # callable: str -> list[float]


def _get_embedder():
    """
    Build and cache the embedding function.

    Priority:
      1. EMBED_DIR env / --embed-dir CLI
         → SentenceTransformer loaded from local directory.
         → Explicitly placed on GPU if one is available (device="cuda").
      2. Fallback → Ollama /api/embeddings endpoint (model = EMBED_MODEL)
    """
    global _embedder_fn
    if _embedder_fn is not None:
        return _embedder_fn

    embed_dir = os.getenv("EMBED_DIR", "").strip()
    if embed_dir and Path(embed_dir).exists():
        _log_rag.info(f"[Embed] Loading SentenceTransformer from: {embed_dir}")
        from sentence_transformers import SentenceTransformer

        # Place the model on GPU if available, otherwise CPU.
        device = "cpu"
        if NUM_GPU > 0:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    _log_rag.info(f"[Embed] GPU detected — loading onto device={device}")
            except ImportError:
                pass

        _st = SentenceTransformer(embed_dir, device=device)
        _log_rag.info(f"[Embed] SentenceTransformer ready on device={device}")

        def _local(text: str) -> list:
            return _st.encode(text, normalize_embeddings=True).tolist()

        _embedder_fn = _local
    else:
        _log_rag.info(f"[Embed] Using Ollama: {OLLAMA_BASE_URL}  model={EMBED_MODEL}")
        import httpx

        def _ollama(text: str) -> list:
            r = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60.0)
            r.raise_for_status()
            return r.json()["embedding"]

        _embedder_fn = _ollama

    return _embedder_fn


def embed_text(text: str) -> list:
    return _get_embedder()(text)


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
    """Open ChromaDB and warm up the embedder. Called eagerly at startup."""
    _get_chroma()
    _get_embedder()   # warm up now so first query has no cold-start delay


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
    if any(k in n for k in ["known", "issue", "bug", "error"]):    return "known_issue"
    if any(k in n for k in ["runbook", "playbook", "procedure"]):  return "runbook"
    if any(k in n for k in ["dos", "donts", "guidelines"]):        return "dos_donts"
    return "general"


def ingest_file(file_path: str, force: bool = False) -> dict:
    path  = Path(file_path)
    fhash = hashlib.md5(path.read_bytes()).hexdigest()
    _, col = _get_chroma()

    if not force:
        existing = col.get(where={"source": str(path)}, limit=1,
                           include=["metadatas"])
        if existing["ids"] and existing["metadatas"]:
            if existing["metadatas"][0].get("file_hash", "") == fhash:
                _log_rag.info(f"[RAG] Skip (unchanged): {path.name}")
                return {"file": path.name, "status": "skipped", "chunks": 0}

    try:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n\n".join(
                p.extract_text() or "" for p in PdfReader(str(path)).pages)
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

    try:
        col.delete(where={"source": str(path)})
    except Exception:
        pass

    ids       = [f"{fhash}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": str(path), "doc_type": doc_type,
                  "chunk_index": i, "file_hash": fhash}
                 for i in range(len(chunks))]
    embeddings = []
    for i, ch in enumerate(chunks):
        embeddings.append(embed_text(ch))
        if (i + 1) % 10 == 0:
            _log_rag.info(f"[RAG]   {i+1}/{len(chunks)} chunks embedded…")

    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    _log_rag.info(f"[RAG] ✓ {path.name} — stored {len(chunks)} chunks")
    return {"file": path.name, "status": "ingested",
            "chunks": len(chunks), "doc_type": doc_type}


def ingest_directory(docs_dir: str, force: bool = False) -> list:
    p     = Path(docs_dir)
    files = (sorted(p.glob("**/*.md")) +
             sorted(p.glob("**/*.pdf")) +
             sorted(p.glob("**/*.txt")))
    if not files:
        _log_rag.warning(f"[RAG] No .md / .pdf / .txt files found in {docs_dir}")
        return []
    _log_rag.info(f"[RAG] Ingesting {len(files)} files from {docs_dir}")
    return [ingest_file(str(f), force=force) for f in files]


def rag_retrieve(query: str, top_k: int = TOP_K,
                 doc_type: Optional[str] = None) -> str:
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
            include=["documents", "metadatas", "distances"],
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
        src = Path(meta.get("source", "?")).name
        dt  = meta.get("doc_type", "general")
        lines.append(f"[{i}] {src} | {dt} | relevance:{sim}\n{doc}\n")
    return "\n".join(lines)


def get_doc_stats() -> dict:
    _, col = _get_chroma()
    total  = col.count()
    if total == 0:
        return {"total_chunks": 0, "files": 0, "by_type": {}}
    from collections import Counter
    all_meta = col.get(include=["metadatas"])["metadatas"]
    by_type  = dict(Counter(m.get("doc_type", "general") for m in all_meta))
    by_src   = Counter(m.get("source", "?") for m in all_meta)
    return {"total_chunks": total, "files": len(by_src), "by_type": by_type}


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
                         "description": "Filter: known_issue | runbook | dos_donts | general"},
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

NAMESPACE DISCOVERY RULE (critical):
- NEVER assume a namespace. For any workload, query its known namespace directly.
- For Vault: ALWAYS use "kubectl get pods -n vault-system" first.
  NEVER use "kubectl get pods -A | grep -i vault" — this matches unrelated pods
  such as backup jobs and CronJobs whose names contain 'vault'.
  If vault-system has no pods, try: kubectl get pods -A --no-headers | awk '$1~/^vault/'
- Apply the same principle to any workload: query its specific namespace directly.

POD LISTING RULE:
- get_pod_status with show_all=false (default) returns ONLY unhealthy pods.
- If the user asks "how many pods", "list pods", "what pods are running", or any
  question requiring a count or full inventory, you MUST set show_all=true.
- If get_pod_status returns "All X pods healthy" or a small/empty list and the
  user asked for a count, trust the count — do not re-ask or hallucinate.

Available tools and when to use them:
Pods & Nodes:
- get_node_health            → always call for cluster health or node questions
- get_pod_status             → call for pod/workload health; set show_all=true for counts/listings
- get_pod_logs               → call when a specific pod name is known and logs are needed
- describe_pod               → call for detailed diagnosis of a specific pod
- get_events                 → always call for warning events or crashloops

Workloads:
- get_deployment_status      → call for Deployment replica issues
- get_daemonset_status       → call for node-level agent issues (CNI, Longhorn engine)
- get_statefulset_status     → call for stateful workload issues (databases, Longhorn)
- get_job_status             → call for batch job failures
- get_hpa_status             → call when scaling behaviour is unexpected

Storage:
- get_pvc_status             → call for Pending/Lost PVCs; use namespace="longhorn-system" for Longhorn
- get_persistent_volumes     → call for PV-level phase or reclaim policy questions

Networking:
- get_service_status         → call for connectivity or missing-endpoint issues
- get_ingress_status         → call for ingress routing or load balancer questions

Config & Resources:
- get_configmap_list         → call to inspect configuration in a namespace
- get_resource_quotas        → call when pods fail to schedule (quota exhaustion)
- get_limit_ranges           → call when containers hit unexpected CPU/memory caps
- get_namespace_status       → call for namespace-level questions

RBAC:
- get_service_accounts       → call for permission or identity questions
- get_cluster_role_bindings  → call to audit broad cluster-wide permissions

kubectl (flexible):
- kubectl_exec               → use for ANY kubectl operation not covered above:
                               namespace discovery ("kubectl get pods -A | grep -i <name>"),
                               CRDs, Longhorn volumes/replicas/engines, rollout history,
                               top nodes/pods, auth can-i, api-resources, diff, and
                               any ad-hoc read-only diagnostic command.
                               Always pass the full command: 'kubectl get pods -n default'.
                               Write operations are blocked unless the operator enables them.

RAG:
- search_documentation       → call to cross-reference known issues and runbooks
{rag_instruction}

CRITICAL RULES:
1. ALWAYS call tools first — never answer from memory alone.
2. NEVER fabricate data — only report what tools actually returned.
3. Be specific — name the exact pod, node, or deployment with the issue.
4. NEVER suggest write operations (restart, delete, scale) — diagnose only.
5. When asked about cluster health, call get_node_health AND get_pod_status AND get_events.
6. For storage issues, ALWAYS check longhorn-system namespace.
7. NEVER mention cgroup, cgroupv1, cgroupv2, or suggest migrating cgroup versions.
   These are environment-level constraints — treat them as background noise, not actionable issues.
8. DO NOT output a "Next Steps" section. Report findings only — what you found and what it means.
   If follow-up investigation is needed, state it as a single inline observation, not a numbered list.
9. DO NOT begin your response with "Based on the tool results" or reference tool call IDs.
   Start directly with the findings.
10. When you have tool results, answer IMMEDIATELY and CONCISELY. Do not repeat the question.
    Do not write the question again as part of your answer. One answer only.

SITE-SPECIFIC RULES:
{custom_rules}

RESPONSE FORMAT:
- Concise bullet points only. No lengthy paragraphs.
- State what you found and what it means.
- Skip sections with nothing to report.
- Max ~300 words unless genuinely complex.
- DO NOT add a "Next Steps" or "Investigation Steps" section.
- DO NOT repeat or restate the user's question in your response.
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
        # Parse kwargs from JSON or empty dict
        try:
            kwargs = json.loads(tool_input) if tool_input.strip().startswith("{") else {}
        except json.JSONDecodeError:
            kwargs = {}

        # Apply defaults for any missing parameter
        for k, v in params.items():
            if k not in kwargs and "default" in v:
                kwargs[k] = v["default"]

        # Type-coerce booleans: LLM sometimes sends "true"/"false" strings
        for k, v in params.items():
            if k in kwargs and v.get("type") == "boolean":
                val = kwargs[k]
                if isinstance(val, str):
                    kwargs[k] = val.lower() in ("true", "1", "yes")

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
         Uses langchain-huggingface ChatHuggingFace which supports bind_tools.
         Falls back to Ollama with the path as model name if not installed.
      2. No --model-dir → standard Ollama by model name
    """
    model_dir = os.getenv("LLM_MODEL_DIR", "").strip()

    if model_dir and Path(model_dir).exists():
        _log_ag.info(f"[LLM] Local directory: {model_dir}")
        try:
            from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
            import transformers, torch
            _log_ag.info("[LLM] Loading HuggingFacePipeline from local dir…")

            # Keep generation params out of pipeline() to avoid the
            # "Passing generation_config together with generation-related
            #  arguments is deprecated" warning. Set them on the pipeline
            # object after construction instead.
            pipe = transformers.pipeline(
                "text-generation",
                model=model_dir,
                tokenizer=model_dir,
                device_map="auto" if NUM_GPU > 0 else "cpu",
                dtype=torch.float16 if NUM_GPU > 0 else torch.float32,
            )
            # Override generation defaults cleanly via GenerationConfig
            from transformers import GenerationConfig
            pipe.model.generation_config = GenerationConfig(
                max_new_tokens=1024,
                temperature=0.1,
                repetition_penalty=1.1,
                do_sample=True,
            )
            llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
            _log_ag.info("[LLM] ChatHuggingFace ready (supports tool calling)")
            return llm
        except ImportError:
            _log_ag.warning(
                "[LLM] langchain-huggingface / transformers not installed — "
                "install with: pip install -U langchain-huggingface transformers torch accelerate\n"
                "       Falling back to Ollama with local path.")
        except Exception as e:
            _log_ag.warning(f"[LLM] HuggingFace load failed ({e}) — "
                            "falling back to Ollama with local path")

        from langchain_ollama import ChatOllama
        _log_ag.info(f"[LLM] Ollama(local path): {model_dir}")
        return ChatOllama(
            model=model_dir, base_url=OLLAMA_BASE_URL,
            temperature=0.1, num_ctx=NUM_CTX,
            num_thread=NUM_THREAD, num_gpu=NUM_GPU, repeat_penalty=1.1,
        )

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
        # Pods
        "get_pod_status":            "📦 Checking pod status",
        "get_pod_logs":              "📋 Fetching pod logs",
        "describe_pod":              "🔍 Describing pod",
        # Nodes
        "get_node_health":           "🖥️  Checking node health",
        # Events
        "get_events":                "⚠️  Fetching cluster events",
        # Workloads
        "get_deployment_status":     "🚀 Checking deployments",
        "get_daemonset_status":      "🔄 Checking DaemonSets",
        "get_statefulset_status":    "🗄️  Checking StatefulSets",
        "get_job_status":            "⏱️  Checking Jobs",
        "get_hpa_status":            "📈 Checking HPAs",
        # Storage
        "get_pvc_status":            "💾 Checking PVCs",
        "get_persistent_volumes":    "💿 Checking PersistentVolumes",
        # Networking
        "get_service_status":        "🌐 Checking Services",
        "get_ingress_status":        "🔀 Checking Ingresses",
        # Config & Resources
        "get_configmap_list":        "📄 Listing ConfigMaps",
        "get_resource_quotas":       "📊 Checking ResourceQuotas",
        "get_limit_ranges":          "📏 Checking LimitRanges",
        # RBAC
        "get_service_accounts":      "🔑 Listing ServiceAccounts",
        "get_cluster_role_bindings": "🛡️  Checking RBAC bindings",
        # Namespaces
        "get_namespace_status":      "🗂️  Checking namespaces",
        # RAG
        "search_documentation":      "📚 Searching knowledge base",
        # kubectl_exec
        "kubectl_exec":              "⚡ Running kubectl command",
    }

    # ── Default tool calls for common queries when model skips tool calling ──
    #
    # Vault-aware routing: "is vault ok?" → pods + PVCs + events in vault namespace.
    # Matched BEFORE generic health/status to avoid being swallowed by that rule.
    # Namespace search: checks both 'vault' and 'hashicorp' namespace patterns.
    def _vault_namespace(user_msg: str) -> str:
        """Return the vault namespace hint from the query, or empty string."""
        lm = user_msg.lower()
        # Explicit namespace hint: "vault in vault-system" etc.
        import re
        m = re.search(r"vault[- ](?:in[- ]|namespace[- ]|ns[- ])?([a-z][a-z0-9-]*)", lm)
        if m:
            ns = m.group(1)
            if ns not in ("ok", "doing", "is", "pod", "pvc", "status", "health"):
                return ns
        return "vault"  # default vault namespace name

    QUERY_DEFAULTS = [
        # ── Vault: precise namespace-scoped queries ───────────────────────────
        # Use two kubectl_exec calls: first discover the vault namespace, then
        # query it directly. grep -i vault on -A output is too broad — it
        # matches backup jobs and other pods whose names/events mention "vault".
        # Instead: list pods in known vault namespaces explicitly.
        (["vault", "hashicorp", "unseal", "secret engine"],
         [("kubectl_exec", {"command": "kubectl get pods -n vault-system --no-headers 2>/dev/null || kubectl get pods -A --no-headers | awk '$1~/vault/{print}'"}),
          ("kubectl_exec", {"command": "kubectl get pvc -n vault-system --no-headers 2>/dev/null || echo 'No vault-system namespace'"}),
          ("get_events",   {"namespace": "vault-system", "warning_only": False})]),

        # ── "how many pods / list all pods" — always show_all=true ───────────
        (["how many pod", "list all pod", "list pod", "all pods", "count pod"],
         [("get_pod_status", {"namespace": "all", "show_all": True})]),

        # ── Generic health / status ───────────────────────────────────────────
        (["health", "status", "check", "overview", "summary", "problem", "issue"],
         [("get_node_health", {}),
          ("get_pod_status", {"namespace": "all"}),
          ("get_events", {"namespace": "all"})]),
        (["pod", "crash", "restart", "oomkill", "pending", "failed"],
         [("get_pod_status", {"namespace": "all"}),
          ("get_events", {"namespace": "all"})]),
        (["node", "pressure", "memory", "disk"],
         [("get_node_health", {}),
          ("get_events", {"namespace": "all"})]),
        (["deploy", "deployment", "replica", "rollout"],
         [("get_deployment_status", {"namespace": "all"}),
          ("get_pod_status", {"namespace": "all"})]),
        (["daemonset", "daemon"],
         [("get_daemonset_status", {"namespace": "all"}),
          ("get_events", {"namespace": "all"})]),
        (["statefulset", "stateful"],
         [("get_statefulset_status", {"namespace": "all"}),
          ("get_events", {"namespace": "all"})]),
        (["job", "cronjob", "batch"],
         [("get_job_status", {"namespace": "all"}),
          ("get_events", {"namespace": "all"})]),
        (["scale", "hpa", "autoscal"],
         [("get_hpa_status", {"namespace": "all"}),
          ("get_deployment_status", {"namespace": "all"})]),
        (["event", "warning", "alert"],
         [("get_events", {"namespace": "all", "warning_only": False})]),
        (["longhorn", "storage", "volume", "pvc", "pv", "persistent"],
         [("get_pvc_status", {"namespace": "all"}),
          ("get_pod_status", {"namespace": "longhorn-system", "show_all": True}),
          ("get_events", {"namespace": "longhorn-system"})]),
        (["service", "svc", "endpoint", "connect", "network", "ingress", "dns"],
         [("get_service_status", {"namespace": "all"}),
          ("get_ingress_status", {"namespace": "all"})]),
        (["quota", "limit", "schedule", "resource"],
         [("get_resource_quotas", {"namespace": "all"}),
          ("get_limit_ranges", {"namespace": "all"})]),
        (["rbac", "permission", "role", "binding", "access", "serviceaccount"],
         [("get_cluster_role_bindings", {}),
          ("get_service_accounts", {"namespace": "default"})]),
        (["namespace", "ns"],
         [("get_namespace_status", {})]),
        (["log"],
         [("get_pod_status", {"namespace": "all"}),
          ("get_events", {"namespace": "all"})]),
        # kubectl_exec-backed defaults — CRDs, Longhorn volumes, resource usage
        (["longhorn volume", "volume.longhorn", "replica", "engine image"],
         [("kubectl_exec", {"command": "kubectl get volumes.longhorn.io -n longhorn-system"}),
          ("get_events",   {"namespace": "longhorn-system"})]),
        (["rollout history", "revision", "previous version", "rollback"],
         [("kubectl_exec", {"command": "kubectl rollout history deployments --all-namespaces=true"})]),
        (["top node", "top pod", "cpu usage", "memory usage", "resource usage"],
         [("kubectl_exec", {"command": "kubectl top nodes"}),
          ("get_node_health", {})]),
    ]

    def _default_tools_for(user_msg: str):
        lm = user_msg.lower()
        for keywords, calls in QUERY_DEFAULTS:
            if any(k in lm for k in keywords):
                return calls
        return [("get_node_health", {}), ("get_pod_status", {"namespace": "all"})]

    def _prepare_messages_for_hf(msgs: list) -> list:
        """
        Rewrite trailing ToolMessages into a clean HumanMessage so that
        apply_chat_template gets a valid human/assistant sequence.

        Applied for ALL model types: Ollama Qwen also echoes the instruction
        string when it appears inside a HumanMessage, causing the visible
        "Using only the tool results above..." loop in responses.

        Strategy:
          - HumanMessage 1: original user question (unchanged)
          - HumanMessage 2: tool results ONLY — no inline instructions
            (the SystemMessage already contains all formatting rules)
        """
        if not msgs:
            return msgs

        # Collect trailing ToolMessages
        tool_results = []
        head = list(msgs)
        while head and isinstance(head[-1], ToolMessage):
            tool_results.insert(0, head.pop())

        if not tool_results:
            # No tool results yet — strip AIMessages to avoid template echo
            human_only = [m for m in msgs if isinstance(m, HumanMessage)]
            return human_only if human_only else msgs

        # Original user question
        original_question = next(
            (m.content for m in msgs if isinstance(m, HumanMessage)
             and isinstance(m.content, str)), "")

        # Tool results ONLY — no instruction text (SystemMessage handles that)
        parts = []
        for tr in tool_results:
            body = (tr.content if len(tr.content) <= 3000
                    else tr.content[:3000] + "\n...[truncated]")
            parts.append(f"Tool result:\n{body}")

        tool_summary = HumanMessage(content="\n\n".join(parts))

        # Return: [original question] + [tool results]
        # The SystemMessage prepended by llm_node carries all formatting rules.
        return [HumanMessage(content=original_question), tool_summary]

    # Apply _prepare_messages_for_hf for ALL model types — Ollama Qwen echoes
    # instruction strings in HumanMessages the same way ChatHuggingFace does.
    _log_ag.info("[LLM] Message rewriting: always ON (prevents Qwen echo loops)")

    def llm_node(state: AgentState):
        itr      = state.get("iteration", 0) + 1
        msgs     = state["messages"]
        updates  = list(state.get("status_updates", []))
        sys_msg  = SystemMessage(content=prompt)

        # Always rewrite ToolMessages into clean HumanMessage pairs.
        # Prevents Ollama Qwen from echoing instruction strings back.
        invoke_msgs = _prepare_messages_for_hf(msgs)
        response = llm.invoke([sys_msg] + invoke_msgs)
        tcs      = getattr(response, "tool_calls", None) or []

        if not tcs and itr == 1:
            user_msg     = next(
                (m.content for m in reversed(msgs) if hasattr(m, "content")
                 and isinstance(m.content, str)), "")
            default_calls = _default_tools_for(user_msg)
            import uuid
            synthetic_tcs = []
            for tname, targs in default_calls:
                if tname in tool_map:
                    synthetic_tcs.append({
                        "name": tname, "args": targs,
                        "id":   f"auto_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    })
            if synthetic_tcs:
                response.tool_calls = synthetic_tcs
                tcs = synthetic_tcs
                updates.append("⚙️  Auto-invoking live cluster tools…")

        if tcs:
            updates.append(f"🔧 Calling: {', '.join(tc['name'] for tc in tcs)}")
        return {"messages": [response],
                "tool_calls_made": state.get("tool_calls_made", []),
                "iteration": itr,
                "status_updates": updates}

    def tool_node(state: AgentState):
        last         = state["messages"][-1]
        results      = []
        tools_called = list(state.get("tool_calls_made", []))
        updates      = list(state.get("status_updates", []))
        tcs          = getattr(last, "tool_calls", []) or []
        for tc in tcs:
            name = tc["name"]
            args = tc.get("args", {})
            tools_called.append(name)
            label = TOOL_LABELS.get(name, f"⚙️ {name}")
            ns    = args.get("namespace", "")
            if ns and ns not in ("default", "all"):
                label += f" ({ns})"
            # For kubectl_exec: surface the actual command in the status log
            if name == "kubectl_exec" and "command" in args:
                updates.append(f"$ {args['command']}")
            else:
                updates.append(label)
            try:
                fn  = tool_map.get(name)
                out = fn.invoke(json.dumps(args) if args else "{}") if fn \
                      else f"Tool '{name}' not found."
            except Exception as e:
                out = f"Tool '{name}' failed: {e}"
            results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
        return {"messages": results,
                "tool_calls_made": tools_called,
                "iteration": state.get("iteration", 0),
                "status_updates": updates}

    def router(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if state.get("iteration", 0) >= 8:
            return "end"
        tcs = getattr(last, "tool_calls", None)
        return "tools" if tcs else "end"

    g = StateGraph(AgentState)
    g.add_node("llm",   llm_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", router, {"tools": "tools", "end": END})
    g.add_edge("tools", "llm")
    return g.compile()


_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def _clean_response(text: str, user_question: str = "") -> str:
    """
    Strip chat-template tokens and conversation-echo artifacts.

    ChatHuggingFace / local models sometimes:
      • Echo im_start/im_end template tokens
      • Repeat the user question multiple times (echo loop)
      • Repeat partial lines with garbage suffixes ("?assed", "?ccording")
      • Dump an entire conversation history instead of a clean answer

    This function applies layered cleanup to extract the real assistant answer.
    """
    # ── 1. Strip complete im_start/im_end blocks ──────────────────────────────
    text = re.sub(r'<\|im_start\|>\w+\s*\n?[\s\S]*?<\|im_end\|>\n?', '', text)
    if '<|im_start|>' in text:
        last = text.split('<|im_start|>')[-1]
        last = re.sub(r'^\w+\s*\n', '', last, count=1)
        text = last
    for tok in ['<|im_end|>', '<s>', '</s>', '[INST]', '[/INST]',
                '<<SYS>>', '<</SYS>>']:
        text = text.replace(tok, '')

    # ── 2. Strip conversation-echo: repeated user question lines ─────────────
    if user_question:
        q_stripped = user_question.strip()
        escaped    = re.escape(q_stripped)
        # Remove ALL occurrences of the question (exact + trailing punct)
        text = re.sub(r'(?i)(\s*' + escaped + r'[?!.]?\s*){2,}', ' ', text)
        # Remove a single leading echo
        text = re.sub(r'(?i)^\s*' + escaped + r'[?!.]?\s*\n', '', text)

    # ── 3. Detect and remove partial-echo lines ───────────────────────────────
    # Pattern: "how many pods in vault-system namespace?assed"
    # The question fragment appears inside a line followed by garbage text.
    if user_question:
        # Build a short prefix (first 20 chars) to catch truncated echoes
        q_prefix = re.escape(user_question.strip()[:20])
        lines_in  = text.split('\n')
        lines_out = []
        for line in lines_in:
            # Drop lines that start with the question fragment but are garbled
            if re.match(r'(?i)\s*' + q_prefix, line) and '?' in line and len(line) > len(user_question) + 5:
                continue
            lines_out.append(line)
        text = '\n'.join(lines_out)

    # ── 4. Line-level deduplication (repeated lines 3+ times are truncated) ───
    lines  = text.split('\n')
    seen:  dict = {}
    deduped = []
    for line in lines:
        key = line.strip().lower()
        if not key:
            deduped.append(line)
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 2:
            deduped.append(line)
        # Lines seen 3+ times are silently dropped
    text = '\n'.join(deduped)

    # ── 5. If response looks like a raw conversation dump, extract final answer
    if text.count('\n') > 40:
        for marker in ['assistant\n', 'ASSISTANT:\n', 'Assistant:\n']:
            if marker in text:
                text = text.split(marker)[-1].strip()
                break

    # ── 6. Collapse excessive blank lines ────────────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)

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
        "response":        _clean_response(raw, user_message),
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


def _run_startup_checks():
    """
    Run kubectl tool smoke-tests and log results.
    Failures are logged as warnings — they do NOT abort startup.
    """
    from tools_k8s import K8S_TOOLS as _tools

    SMOKE_TESTS = [
        ("get_node_health",    {}),
        ("get_namespace_status", {}),
        ("get_pod_status",     {"namespace": "all"}),
        ("get_events",         {"namespace": "all", "warning_only": True}),
    ]

    logger.info("[Self-test] Running kubectl tool smoke-tests…")
    all_ok = True
    for name, kwargs in SMOKE_TESTS:
        cfg = _tools.get(name)
        if cfg is None:
            logger.warning(f"[Self-test] ⚠ Tool not found: {name}")
            all_ok = False
            continue
        try:
            result = cfg["fn"](**kwargs)
            # Treat API error strings as failures
            if result.startswith("K8s API error") or result.startswith("K8s error"):
                logger.warning(f"[Self-test] ⚠ {name}: {result[:120]}")
                all_ok = False
            else:
                preview = result.replace("\n", " ")[:80]
                logger.info(f"[Self-test] ✓ {name}: {preview}…")
        except Exception as e:
            logger.warning(f"[Self-test] ⚠ {name} raised: {e}")
            all_ok = False

    if all_ok:
        logger.info("[Self-test] All kubectl tools OK ✓")
    else:
        logger.warning(
            "[Self-test] Some kubectl tools failed — check KUBECONFIG_PATH "
            "and cluster connectivity.")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"Cloudera ECS AI Ops — Phase {PHASE}")
    model_src = os.getenv("LLM_MODEL_DIR") or f"Ollama:{LLM_MODEL}"
    embed_src = os.getenv("EMBED_DIR")      or f"Ollama:{EMBED_MODEL}"
    gpu_info  = (f"{NUM_GPU} GPU(s) — GPU inference"
                 if NUM_GPU > 0 else "No GPU — CPU inference")
    logger.info(f"  LLM      : {model_src}")
    logger.info(f"  Embed    : {embed_src}")
    logger.info(f"  GPU      : {gpu_info}")
    logger.info(f"  Threads  : {NUM_THREAD}   CTX: {NUM_CTX}")
    logger.info(f"  ChromaDB : {CHROMA_DIR}")
    logger.info(f"  Tools    : {len(K8S_TOOLS)} kubectl tools registered")
    logger.info("=" * 60)

    # ── 1. kubectl self-test ──────────────────────────────────────────────────
    _run_startup_checks()

    # ── 2. ChromaDB + embedder (always initialised, even if PHASE=1) ─────────
    try:
        _log_rag.info("[ChromaDB] Initialising persistent store…")
        init_db()   # opens ChromaDB AND warms the embedder (with GPU if available)
        stats = get_doc_stats()
        _log_rag.info(
            f"[ChromaDB] Ready — {stats['total_chunks']} chunks "
            f"across {stats['files']} file(s)  |  by type: {stats['by_type']}")
    except Exception as e:
        _log_rag.error(f"[ChromaDB] Init failed — RAG unavailable: {e}")

    # ── 3. Pre-warm the LLM agent ─────────────────────────────────────────────
    logger.info("[Agent] Pre-warming LLM…")
    t0 = time.time()
    get_agent()
    logger.info(f"[Agent] Ready in {time.time()-t0:.1f}s")
    logger.info("Startup complete ✓")
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
    response: str;         tools_used: list
    iterations: int;       phase: int
    status_updates: list;  elapsed_seconds: float

class IngestRequest(BaseModel):
    docs_dir: str;  force: bool = False

class IngestResponse(BaseModel):
    results: list;  total_files: int;  total_chunks: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    stats = get_doc_stats()
    return {
        "status":        "ok",
        "phase":         PHASE,
        "model":         os.getenv("LLM_MODEL_DIR") or LLM_MODEL,
        "model_source":  "local_dir" if os.getenv("LLM_MODEL_DIR") else "ollama",
        "embed_source":  "local_dir" if os.getenv("EMBED_DIR")      else "ollama",
        "num_gpu":       NUM_GPU,
        "chroma_dir":    CHROMA_DIR,
        "chroma_chunks": stats["total_chunks"],
        "k8s_tools":     len(K8S_TOOLS),
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
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except: temp = 0
            try:    pw   = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except: pw   = None
            gpus.append({
                "index":        i,
                "name":         name,
                "util_pct":     util.gpu,
                "mem_used_gb":  round(mem.used  / 1e9, 1),
                "mem_total_gb": round(mem.total / 1e9, 1),
                "mem_pct":      round(mem.used  / mem.total * 100, 1),
                "temp_c":       temp,
                "power_w":      pw,
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
        "cpu_per_core": [round(p, 1) for p in cpu_per],
        "cpu_count":    psutil.cpu_count(logical=True),
        "freq_mhz":     round(freq.current) if freq else 0,
        "load_avg":     [round(x, 2) for x in psutil.getloadavg()],
        "mem_total_gb": round(mem.total / 1e9, 1),
        "mem_used_gb":  round(mem.used  / 1e9, 1),
        "mem_pct":      mem.percent,
        "gpus":         _gpu_metrics(),
        "num_gpu":      NUM_GPU,
    }


@app.get("/namespaces")
async def namespaces():
    try:
        return {"namespaces": [ns.metadata.name
                               for ns in _core.list_namespace().items]}
    except Exception as e:
        return {"namespaces": ["default"], "error": str(e)}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_api(req: IngestRequest):
    if PHASE < 2:
        raise HTTPException(400, "Requires PHASE=2")
    results      = ingest_directory(req.docs_dir, force=req.force)
    total_chunks = sum(r.get("chunks", 0) for r in results)
    return IngestResponse(results=results,
                          total_files=len(results),
                          total_chunks=total_chunks)


@app.get("/docs/stats")
async def doc_stats():
    return {"stats": get_doc_stats()}


# ── Serve HTML UI + static assets ─────────────────────────────────────────────
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

    parser.add_argument("--host",      default="0.0.0.0")
    parser.add_argument("--port",      type=int, default=8000)
    parser.add_argument("--model-dir", metavar="PATH", default=None,
                        help="Path to a local LLM directory (HuggingFace safetensors).")
    parser.add_argument("--embed-dir", metavar="PATH", default=None,
                        help="Path to a local sentence-transformers model directory.")
    parser.add_argument("--ingest",    metavar="DOCS_DIR", default=None,
                        help="Ingest .md / .pdf / .txt files into ChromaDB before starting.")
    parser.add_argument("--force",     action="store_true",
                        help="Re-ingest all files even if unchanged (use with --ingest).")
    parser.add_argument("--reload",    action="store_true",
                        help="Enable uvicorn auto-reload (development mode).")
    args = parser.parse_args()

    if args.model_dir:
        os.environ["LLM_MODEL_DIR"] = args.model_dir
    if args.embed_dir:
        os.environ["EMBED_DIR"] = args.embed_dir

    # ── Optional: ingest before starting ─────────────────────────────────────
    if args.ingest:
        if PHASE < 2:
            print("ERROR: --ingest requires PHASE=2 (set in env file)")
            sys.exit(1)
        print(f"\n📂 Ingesting documents from: {args.ingest}  (force={args.force})")
        init_db()
        results = ingest_directory(args.ingest, force=args.force)
        total   = sum(r.get("chunks", 0) for r in results)
        print(f"\n✅  {len(results)} file(s)  |  {total} total chunks stored in ChromaDB\n")
        for r in results:
            icon = ("✓" if r["status"] == "ingested"
                    else "—" if r["status"] == "skipped" else "✗")
            print(f"  {icon}  {r['file']:<42} {r['status']:<10} ({r['chunks']} chunks)")
        print()

    # ── Print startup banner ──────────────────────────────────────────────────
    import uvicorn
    model_src  = args.model_dir or LLM_MODEL
    embed_src  = args.embed_dir or f"Ollama:{EMBED_MODEL}"
    gpu_str    = (f"{NUM_GPU} GPU(s) — GPU inference"
                  if NUM_GPU > 0 else "None — CPU inference")
    tool_count = len(K8S_TOOLS)

    print(f"""
╔════════════════════════════════════════════════════════════╗
║            Cloudera ECS AI Ops  v2.0                       ║
╠════════════════════════════════════════════════════════════╣
║  Phase    : {PHASE}                                              ║
║  LLM      : {model_src:<46} ║
║  Embed    : {embed_src:<46} ║
║  GPU      : {gpu_str:<46} ║
║  Tools    : {tool_count} kubectl tools registered{'':<26} ║
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
