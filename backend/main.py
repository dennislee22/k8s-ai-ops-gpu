"""
FastAPI Backend — K8s AI Ops Assistant
Serves the agent API and document ingestion endpoints.
"""

import os
import sys
import time
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(_BACKEND_DIR / "env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import psutil
from core.logger import get_logger, configure_root
configure_root()
logger = get_logger(__name__)

PHASE = int(os.getenv("PHASE", "2"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"K8s AI Ops Assistant starting — Phase {PHASE}")
    logger.info(f"  Model     : {os.getenv('LLM_MODEL', 'qwen2.5:7b')}")
    logger.info(f"  Ollama    : {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    logger.info(f"  Threads   : {os.getenv('NUM_THREAD', '16')}")
    logger.info(f"  Context   : {os.getenv('NUM_CTX', '4096')}")
    logger.info("=" * 60)

    if PHASE >= 2:
        from rag.rag_tool import init_db
        try:
            logger.info("[DB] Initialising pgvector database...")
            init_db()
            logger.info("[DB] pgvector ready")
        except Exception as e:
            logger.error(f"[DB] Init failed: {e}. RAG will be unavailable.")

    logger.info("[Agent] Pre-warming agent (loading LLM into memory)...")
    t0 = time.time()
    from agent.agent import get_agent
    get_agent()
    logger.info(f"[Agent] Ready in {time.time() - t0:.1f}s")
    logger.info("Startup complete — listening for requests")
    yield
    logger.info("Shutting down gracefully")


app = FastAPI(
    title="K8s AI Ops Assistant",
    description="Air-gapped Kubernetes operations assistant powered by local LLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = (time.time() - t0) * 1000
    logger.info(f"[HTTP] {request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")
    return response


# ── Models ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    # namespace removed — agent always scans all namespaces


class ChatResponse(BaseModel):
    response: str
    tools_used: list[str]
    iterations: int
    phase: int
    status_updates: list[str]
    elapsed_seconds: float


class IngestRequest(BaseModel):
    docs_dir: str
    force: bool = False


class IngestResponse(BaseModel):
    results: list[dict]
    total_files: int
    total_chunks: int


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "phase":     PHASE,
        "model":     os.getenv("LLM_MODEL", "qwen2.5:7b"),
        "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "num_gpu":   int(os.getenv("NUM_GPU", "0")),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(f"[Chat] Query: {req.message[:80]}...")
    t0 = time.time()
    try:
        from agent.agent import run_agent
        result = await run_agent(req.message)
        logger.info(
            f"[Chat] Completed in {time.time()-t0:.1f}s | "
            f"tools={result['tools_used']} | iterations={result['iterations']}"
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"[Chat] Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_docs(req: IngestRequest):
    if PHASE < 2:
        raise HTTPException(status_code=400, detail="Requires Phase 2")
    logger.info(f"[Ingest] Starting from {req.docs_dir} (force={req.force})")
    from rag.rag_tool import ingest_directory
    try:
        results = ingest_directory(req.docs_dir, force=req.force)
        total_chunks = sum(r.get("chunks", 0) for r in results)
        logger.info(f"[Ingest] Done — {len(results)} files, {total_chunks} chunks")
        return IngestResponse(results=results, total_files=len(results), total_chunks=total_chunks)
    except Exception as e:
        logger.error(f"[Ingest] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs/stats")
async def doc_stats():
    if PHASE < 2:
        return {"message": "Phase 1: RAG not enabled"}
    from rag.rag_tool import get_doc_stats
    return {"stats": get_doc_stats()}


@app.get("/metrics")
async def get_metrics():
    """Live CPU, memory and load metrics for the bpytop panel."""
    cpu_per   = psutil.cpu_percent(interval=0.2, percpu=True)
    cpu_total = psutil.cpu_percent(interval=None)
    mem       = psutil.virtual_memory()
    load      = psutil.getloadavg()          # (1m, 5m, 15m)
    freq      = psutil.cpu_freq()
    cpu_count = psutil.cpu_count(logical=True)

    return {
        "cpu_total":    round(cpu_total, 1),
        "cpu_per_core": [round(p, 1) for p in cpu_per],
        "cpu_count":    cpu_count,
        "freq_mhz":     round(freq.current) if freq else 0,
        "load_avg":     [round(x, 2) for x in load],
        "mem_total_gb": round(mem.total / 1e9, 1),
        "mem_used_gb":  round(mem.used  / 1e9, 1),
        "mem_pct":      mem.percent,
    }


@app.get("/namespaces")
async def list_namespaces():
    try:
        from kubernetes import client, config
        try:
            config.load_kube_config()
        except Exception:
            config.load_incluster_config()
        v1 = client.CoreV1Api()
        ns_list = [ns.metadata.name for ns in v1.list_namespace().items]
        logger.info(f"[K8s] Namespaces: {ns_list}")
        return {"namespaces": ns_list}
    except Exception as e:
        logger.warning(f"[K8s] Could not list namespaces: {e}")
        return {"namespaces": ["default"], "error": str(e)}
