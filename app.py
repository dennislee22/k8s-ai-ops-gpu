#!/usr/bin/env python3
"""
Cloudera ECS AI Ops — Single-file application (Transformers Edition)
====================================================================
Runs the complete stack from one Python file.
No external APIs or services required. Uses HuggingFace Transformers locally.

Usage:
    python3 app.py                                    # start on port 8000
    python3 app.py --port 9000                        # custom port
    python3 app.py --host 0.0.0.0                     # bind address
    python3 app.py --model-dir  Qwen/Qwen2.5-7B-Instruct # LLM from local dir or HF hub
    python3 app.py --embed-dir  nomic-ai/nomic-embed-text-v1.5 # embeddings
    python3 app.py --ingest ./docs                    # ingest docs then start
    python3 app.py --ingest ./docs --force            # re-ingest all docs
    python3 app.py --reload                           # dev auto-reload

Dependencies:
    pip install fastapi "uvicorn[standard]" langgraph langchain-core \
                kubernetes python-dotenv psutil chromadb sentence-transformers \
                pypdf markdown-it-py langchain-huggingface transformers torch accelerate

Optional (GPU monitoring):
    pip install nvidia-ml-py
"""

import os, sys, argparse, re, hashlib, time, json, logging, logging.handlers
from pathlib import Path
from typing import Annotated, TypedDict, Literal, Optional

import psutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--model-dir", default=None)
_pre.add_argument("--embed-dir", default=None)
_pre.add_argument("--ingest",    default=None)
_pre.add_argument("--force",     action="store_true")
_pre.add_argument("--port",      type=int, default=8000)
_pre.add_argument("--host",      default="0.0.0.0")
_pre.add_argument("--reload",    action="store_true")
_ARGS, _ = _pre.parse_known_args()

_env_file = _HERE / "env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

# ── ENV DEFAULTS ──────────────────────────────────────────────────────────────
os.environ.setdefault("LLM_MODEL",       "Qwen/Qwen2.5-7B-Instruct")
os.environ.setdefault("EMBED_MODEL",     "nomic-ai/nomic-embed-text-v1.5")
os.environ.setdefault("KUBECONFIG_PATH", "~/kubeconfig")
os.environ.setdefault("PHASE",           "2")
os.environ.setdefault("LOG_LEVEL",       "INFO")
os.environ.setdefault("CHROMA_DIR",      str(_HERE / "chromadb"))
os.environ.setdefault("CUSTOM_RULES",    "- Do NOT recommend migrating to cgroupv2. This environment uses cgroupv1.")

if _ARGS.model_dir:
    os.environ["LLM_MODEL"] = _ARGS.model_dir
if _ARGS.embed_dir:
    os.environ["EMBED_MODEL"] = _ARGS.embed_dir

PHASE           = int(os.getenv("PHASE",           "2"))
LLM_MODEL       = os.getenv("LLM_MODEL",           "Qwen/Qwen2.5-7B-Instruct").strip()
EMBED_MODEL     = os.getenv("EMBED_MODEL",         "nomic-ai/nomic-embed-text-v1.5").strip()
CHROMA_DIR      = os.getenv("CHROMA_DIR",          str(_HERE / "chromadb"))
CUSTOM_RULES    = os.getenv("CUSTOM_RULES",        "").strip()

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
            _LOG_DIR / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
        fh.setLevel(_LEVEL)
        fh.setFormatter(logging.Formatter(_FMT_FIL, datefmt=_DATE))
        log.addHandler(fh)
        log.propagate = False
    _cfg_set.add(name)
    return log

for _noisy in ["httpx", "httpcore", "urllib3", "kubernetes.client", "langchain", "langsmith", "watchfiles", "chromadb"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger   = get_logger("app")
_log_rag = get_logger("rag")
_log_ag  = get_logger("agent")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — KUBERNETES TOOLS
# ─────────────────────────────────────────────────────────────────────────────

from tools_k8s import K8S_TOOLS, _core

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RAG: ChromaDB + Transformers
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K         = 5

_chroma_client     = None
_chroma_collection = None
_embedder_fn       = None

def _get_embedder():
    global _embedder_fn
    if _embedder_fn is not None:
        return _embedder_fn

    _log_rag.info(f"[Embed] Loading SentenceTransformer: {EMBED_MODEL}")
    from sentence_transformers import SentenceTransformer

    device = "cpu"
    if NUM_GPU > 0:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

    _log_rag.info(f"[Embed] Inference running on device={device}")
    _st = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)

    def _local(text: str) -> list:
        return _st.encode(text, normalize_embeddings=True).tolist()

    _embedder_fn = _local
    return _embedder_fn

def embed_text(text: str) -> list:
    return _get_embedder()(text)

def _get_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_client, _chroma_collection

    import chromadb
    from chromadb.config import Settings

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    _log_rag.info(f"[ChromaDB] Opening persistent store: {CHROMA_DIR}")

    _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    _chroma_collection = _chroma_client.get_or_create_collection(name="k8s_docs", metadata={"hnsw:space": "cosine"})
    
    return _chroma_client, _chroma_collection

def init_db():
    _get_chroma()
    _get_embedder()

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
        if chunk: chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks

def _doc_type(filename: str) -> str:
    n = filename.lower()
    if any(k in n for k in ["known", "issue", "bug", "error"]):   return "known_issue"
    if any(k in n for k in ["runbook", "playbook", "procedure"]): return "runbook"
    if any(k in n for k in ["dos", "donts", "guidelines"]):       return "dos_donts"
    return "general"

def ingest_file(file_path: str, force: bool = False) -> dict:
    path  = Path(file_path)
    fhash = hashlib.md5(path.read_bytes()).hexdigest()
    _, col = _get_chroma()

    if not force:
        existing = col.get(where={"source": str(path)}, limit=1, include=["metadatas"])
        if existing["ids"] and existing["metadatas"]:
            if existing["metadatas"][0].get("file_hash", "") == fhash:
                _log_rag.info(f"[RAG] Skip (unchanged): {path.name}")
                return {"file": path.name, "status": "skipped", "chunks": 0}

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
        return {"file": path.name, "status": "error", "chunks": 0, "error": str(e)}

    if not text.strip(): return {"file": path.name, "status": "empty", "chunks": 0}

    chunks   = chunk_text(text)
    doc_type = _doc_type(path.name)
    _log_rag.info(f"[RAG] {path.name}: {len(chunks)} chunks  type={doc_type}")

    try: col.delete(where={"source": str(path)})
    except Exception: pass

    ids       = [f"{fhash}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": str(path), "doc_type": doc_type, "chunk_index": i, "file_hash": fhash} for i in range(len(chunks))]
    embeddings = [embed_text(ch) for ch in chunks]

    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return {"file": path.name, "status": "ingested", "chunks": len(chunks), "doc_type": doc_type}

def ingest_directory(docs_dir: str, force: bool = False) -> list:
    p = Path(docs_dir)
    files = sorted(p.glob("**/*.md")) + sorted(p.glob("**/*.pdf")) + sorted(p.glob("**/*.txt"))
    return [ingest_file(str(f), force=force) for f in files]

def rag_retrieve(query: str, top_k: int = TOP_K, doc_type: Optional[str] = None) -> str:
    _, col = _get_chroma()
    n = col.count()
    if n == 0: return "No documents ingested yet."

    qe = embed_text(query)
    where = {"doc_type": doc_type} if doc_type else None
    try:
        res = col.query(query_embeddings=[qe], n_results=min(top_k, n), where=where, include=["documents", "metadatas", "distances"])
    except Exception as e:
        return f"RAG query failed: {e}"

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not docs: return "No relevant documentation found."

    lines = [f"Retrieved {len(docs)} relevant chunks:\n"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        sim = round(1 - dist, 3)
        src = Path(meta.get("source", "?")).name
        lines.append(f"[{i}] {src} | relevance:{sim}\n{doc}\n")
    return "\n".join(lines)

def get_doc_stats() -> dict:
    _, col = _get_chroma()
    total = col.count()
    if total == 0: return {"total_chunks": 0, "files": 0, "by_type": {}}
    from collections import Counter
    all_meta = col.get(include=["metadatas"])["metadatas"]
    by_type = dict(Counter(m.get("doc_type", "general") for m in all_meta))
    by_src = Counter(m.get("source", "?") for m in all_meta)
    return {"total_chunks": total, "files": len(by_src), "by_type": by_type}

RAG_TOOLS = {
    "search_documentation": {
        "fn": rag_retrieve,
        "description": "Search the internal knowledge base for known issues, runbooks, and guidelines.",
        "parameters": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "doc_type": {"type": "string", "default": None},
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LANGGRAPH AGENT (Transformers Edition)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Kubernetes operations assistant running in an air-gapped environment.

CRITICAL RULES:
1. ALWAYS call tools first — never answer from memory alone.
2. NEVER fabricate data — only report what tools actually returned.
3. NEVER output raw bash commands, kubectl pipes (e.g., `| jq`), or mock terminal output. 
4. Be specific — name the exact pod, node, or deployment with the issue.
5. NEVER suggest write operations (restart, delete, scale) — diagnose only.
6. When you have tool results, answer IMMEDIATELY and CONCISELY. One answer only.
{rag_instruction}
SITE-SPECIFIC RULES:
{custom_rules}

RESPONSE FORMAT:
- Format your response as a direct, factual summary. 
- NO conversational filler (e.g., do NOT say "Here is the status...", "Based on the results...").
- Provide ONLY the direct answer, facts, or raw data summaries.
- DO NOT add a "Next Steps", "Investigation Steps", "Summary", or "Conclusion" section.
- DO NOT repeat or restate the user's question.
"""

RAG_INSTRUCTION = "\n7. ALWAYS search documentation before finalising a diagnosis."

def _make_tool(name: str, cfg: dict):
    fn, desc, params = cfg["fn"], cfg["description"], cfg.get("parameters", {})
    if not params:
        @tool(name, description=desc)
        def _t() -> str: return fn()
        return _t

    full_desc = desc + "\nParameters: " + ", ".join(f"{k}:{v.get('type','str')}" for k, v in params.items())

    @tool(name, description=full_desc)
    def _t(tool_input: str) -> str:
        try: kwargs = json.loads(tool_input) if tool_input.strip().startswith("{") else {}
        except: kwargs = {}
        for k, v in params.items():
            if k not in kwargs and "default" in v: kwargs[k] = v["default"]
        return fn(**kwargs)
    return _t

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_made: list
    iteration: int
    status_updates: list

def _build_llm():
    _log_ag.info(f"[LLM] Loading HuggingFace model: {LLM_MODEL}")
    try:
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        import transformers, torch

        device_map = "auto" if NUM_GPU > 0 else "cpu"
        dtype = torch.float16 if NUM_GPU > 0 else torch.float32

        pipe = transformers.pipeline(
            "text-generation",
            model=LLM_MODEL,
            tokenizer=LLM_MODEL,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        
        # Greedy decoding to stop repetition penalty artifacts and looping
        pipe.model.generation_config.max_new_tokens = 2048
        pipe.model.generation_config.temperature = 0.0
        pipe.model.generation_config.do_sample = False
        if hasattr(pipe.model.generation_config, "repetition_penalty"):
            pipe.model.generation_config.repetition_penalty = 1.0
            
        llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
        _log_ag.info("[LLM] ChatHuggingFace ready (supports tool calling)")
        return llm
    except Exception as e:
        _log_ag.error(f"[LLM] HuggingFace load failed: {e}")
        raise

def build_agent():
    all_tools = {**K8S_TOOLS}
    if PHASE >= 2: all_tools.update(RAG_TOOLS)

    lc_tools = [_make_tool(n, c) for n, c in all_tools.items()]
    tool_map = {t.name: t for t in lc_tools}
    prompt = SYSTEM_PROMPT.format(rag_instruction=RAG_INSTRUCTION if PHASE >= 2 else "", custom_rules=CUSTOM_RULES or "None.")
    llm = _build_llm().bind_tools(lc_tools)

    def _default_tools_for(user_msg: str):
        lm = user_msg.lower()
        ns = "all"
        
        # Smart namespace extraction from query
        m = re.search(r'(?:in|for|namespace|ns)\s+([a-z0-9-]+)', lm)
        if m and m.group(1) not in ("all", "namespace", "ns", "the", "this"):
            ns = m.group(1)
        elif "vault" in lm:
            ns = "vault-system"
        elif "longhorn" in lm:
            ns = "longhorn-system"

        # Map queries to tools - FIXED routing
        is_asking_for_count = any(k in lm for k in ["how many", "list"])
        
        if any(k in lm for k in ["namespace", "ns", "namespaces"]) and (not m or is_asking_for_count):
            return [("get_namespace_status", {})]
        if any(k in lm for k in ["node", "pressure"]):
            return [("get_node_health", {})]
        if any(k in lm for k in ["pvc", "volume", "storage"]):
            return [("get_pvc_status", {"namespace": ns})]
            
        return [("get_node_health", {}), ("get_pod_status", {"namespace": ns, "show_all": True})]

    def _prepare_messages_for_hf(msgs: list) -> list:
        """
        Convert tool results into plain text observations rather than relying on 
        brittle XML tags which can confuse local models and break token generation.
        """
        if not msgs: return msgs
        
        if not any(isinstance(m, ToolMessage) for m in msgs):
            return [m for m in msgs if isinstance(m, HumanMessage) or isinstance(m, SystemMessage)]

        original_question = next((m.content for m in msgs if isinstance(m, HumanMessage)), "")
        tool_results = [m for m in msgs if isinstance(m, ToolMessage)]
        
        parts = []
        for i, tr in enumerate(tool_results, 1):
            body = tr.content if len(tr.content) <= 4000 else tr.content[:4000] + "\n...[truncated]"
            parts.append(f"--- TOOL RESULT {i} ---\n{body}\n")
        
        synthesis_prompt = (
            f"Question: {original_question}\n\n"
            f"Tool Results:\n{''.join(parts)}\n"
            "Based STRICTLY on the tool results above, summarize the data directly and concisely."
        )
        return [HumanMessage(content=synthesis_prompt)]

    def llm_node(state: AgentState):
        itr = state.get("iteration", 0) + 1
        msgs = state["messages"]
        updates = list(state.get("status_updates", []))
        
        sys_msg = SystemMessage(content=prompt)
        invoke_msgs = _prepare_messages_for_hf(msgs)
        
        response = llm.invoke([sys_msg] + invoke_msgs)
        tcs = getattr(response, "tool_calls", None) or []
        
        # Synthetic fallback tools for zero-shot failures
        if not tcs and itr == 1:
            user_msg = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
            default_calls = _default_tools_for(user_msg)
            import uuid
            synthetic_tcs = []
            for tname, targs in default_calls:
                if tname in tool_map:
                    synthetic_tcs.append({
                        "name": tname, "args": targs,
                        "id": f"auto_{uuid.uuid4().hex[:8]}", "type": "tool_call",
                    })
            if synthetic_tcs:
                response.tool_calls = synthetic_tcs
                tcs = synthetic_tcs
                updates.append("⚙️  Auto-invoking fallback tools…")

        if tcs: updates.append(f"🔧 Calling: {', '.join(tc['name'] for tc in tcs)}")
        return {"messages": [response], "tool_calls_made": state.get("tool_calls_made", []), "iteration": itr, "status_updates": updates}

    def tool_node(state: AgentState):
        last = state["messages"][-1]
        results, tools_called = [], list(state.get("tool_calls_made", []))
        updates = list(state.get("status_updates", []))
        
        tcs = getattr(last, "tool_calls", []) or []
        for tc in tcs:
            name, args = tc["name"], tc.get("args", {})
            tools_called.append(name)
            if name == "kubectl_exec" and "command" in args:
                updates.append(f"$ {args['command']}")
            else:
                updates.append(f"⚙️ {name}")
                
            try:
                fn = tool_map.get(name)
                out = fn.invoke(json.dumps(args) if args else "{}") if fn else f"Tool '{name}' not found."
            except Exception as e: out = f"Tool '{name}' failed: {e}"
            results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
            
        return {"messages": results, "tool_calls_made": tools_called, "iteration": state.get("iteration", 0), "status_updates": updates}

    def router(state: AgentState) -> Literal["tools", "end"]:
        if state.get("iteration", 0) >= 8: return "end"
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "end"

    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", router, {"tools": "tools", "end": END})
    g.add_edge("tools", "llm")
    return g.compile()

_agent = None
def get_agent():
    global _agent
    if _agent is None: _agent = build_agent()
    return _agent

def _clean_response(text: str, user_question: str = "") -> str:
    text = re.sub(r'<\|im_start\|>\w+\s*\n?[\s\S]*?<\|im_end\|>\n?', '', text)
    if '<|im_start|>' in text: text = re.sub(r'^\w+\s*\n', '', text.split('<|im_start|>')[-1], count=1)
    for tok in ['<|im_end|>', '<s>', '</s>', '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>']: text = text.replace(tok, '')
    
    if user_question:
        q_stripped = user_question.strip()
        escaped    = re.escape(q_stripped)
        text = re.sub(r'(?i)(\s*' + escaped + r'[?!.]?\s*){2,}', ' ', text)
        text = re.sub(r'(?i)^\s*' + escaped + r'[?!.]?\s*\n', '', text)
    
    text = re.sub(r'Summarise the above tool results.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

async def run_agent(user_message: str) -> dict:
    agent = get_agent()
    t0 = time.time()
    final = await agent.ainvoke({
        "messages": [HumanMessage(content=user_message)],
        "tool_calls_made": [],
        "iteration": 0,
        "status_updates": [f"🤖 Model: {LLM_MODEL}"],
    })
    elapsed = time.time() - t0
    last = final["messages"][-1]
    raw = last.content if hasattr(last, "content") else str(last)
    updates = final.get("status_updates", [])
    updates.append(f"✅ Done in {elapsed:.0f}s")
    return {
        "response": _clean_response(raw, user_message),
        "tools_used": final.get("tool_calls_made", []),
        "iterations": final.get("iteration", 0),
        "phase": PHASE,
        "status_updates": updates,
        "elapsed_seconds": round(elapsed, 1),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

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

def _run_startup_checks():
    """Run kubectl tool smoke-tests and log results."""
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
            if result.startswith("K8s API error") or result.startswith("K8s error") or result.startswith("[ERROR]"):
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
    gpu_info = f"{NUM_GPU} GPU(s) — GPU inference" if NUM_GPU > 0 else "No GPU — CPU inference"
    logger.info(f"  LLM      : {LLM_MODEL}")
    logger.info(f"  Embed    : {EMBED_MODEL}")
    logger.info(f"  GPU      : {gpu_info}")
    logger.info(f"  ChromaDB : {CHROMA_DIR}")
    logger.info(f"  Tools    : {len(K8S_TOOLS)} kubectl tools registered")
    logger.info("=" * 60)

    # 1. kubectl self-test
    _run_startup_checks()

    # 2. ChromaDB + embedder
    try:
        _log_rag.info("[ChromaDB] Initialising persistent store…")
        init_db()
        stats = get_doc_stats()
        _log_rag.info(
            f"[ChromaDB] Ready — {stats['total_chunks']} chunks "
            f"across {stats['files']} file(s)  |  by type: {stats['by_type']}")
    except Exception as e:
        _log_rag.error(f"[ChromaDB] Init failed — RAG unavailable: {e}")

    # 3. Pre-warm LLM
    logger.info("[Agent] Pre-warming LLM…")
    t0 = time.time()
    get_agent()
    logger.info(f"[Agent] Ready in {time.time()-t0:.1f}s")
    logger.info("Startup complete ✓")
    yield
    logger.info("Shutting down")

app = FastAPI(title="Cloudera ECS AI Ops", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel): message: str
class ChatResponse(BaseModel): response: str; tools_used: list; iterations: int; phase: int; status_updates: list; elapsed_seconds: float
class IngestRequest(BaseModel): docs_dir: str; force: bool = False
class IngestResponse(BaseModel): results: list; total_files: int; total_chunks: int

@app.get("/health")
async def health():
    stats = get_doc_stats()
    return {"status": "ok", "phase": PHASE, "model": LLM_MODEL, "model_source": "huggingface", "embed_source": "huggingface", "num_gpu": NUM_GPU, "chroma_chunks": stats["total_chunks"], "k8s_tools": len(K8S_TOOLS)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip(): raise HTTPException(400, "Empty message")
    try: return ChatResponse(**await run_agent(req.message))
    except Exception as e:
        logger.error(f"[Chat] {e}", exc_info=True)
        raise HTTPException(500, f"Agent failed: {e}")

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

@app.post("/ingest", response_model=IngestResponse)
async def ingest_api(req: IngestRequest):
    results = ingest_directory(req.docs_dir, force=req.force)
    return IngestResponse(results=results, total_files=len(results), total_chunks=sum(r.get("chunks", 0) for r in results))

if _HERE.joinpath("static").exists(): app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

@app.get("/", response_class=FileResponse)
async def serve_ui():
    if _HERE.joinpath("index.html").exists(): return FileResponse(str(_HERE / "index.html"), media_type="text/html")
    return {"error": "index.html not found"}

if __name__ == "__main__":
    if _ARGS.ingest:
        if PHASE < 2:
            print("ERROR: --ingest requires PHASE=2 (set in env file)")
            sys.exit(1)
        print(f"\n📂 Ingesting documents from: {_ARGS.ingest}  (force={_ARGS.force})")
        init_db()
        results = ingest_directory(_ARGS.ingest, force=_ARGS.force)
        total   = sum(r.get("chunks", 0) for r in results)
        print(f"\n✅  {len(results)} file(s)  |  {total} total chunks stored in ChromaDB\n")
        for r in results:
            icon = ("✓" if r["status"] == "ingested" else "—" if r["status"] == "skipped" else "✗")
            print(f"  {icon}  {r['file']:<42} {r['status']:<10} ({r['chunks']} chunks)")
        print()

    import uvicorn
    gpu_str    = (f"{NUM_GPU} GPU(s) — GPU inference" if NUM_GPU > 0 else "None — CPU inference")
    tool_count = len(K8S_TOOLS)

    print(f"""
╔════════════════════════════════════════════════════════════╗
║            Cloudera ECS AI Ops  v2.0 (Transformers)        ║
╠════════════════════════════════════════════════════════════╣
║  Phase    : {PHASE:<46} ║
║  LLM      : {LLM_MODEL:<46} ║
║  Embed    : {EMBED_MODEL:<46} ║
║  GPU      : {gpu_str:<46} ║
║  Tools    : {tool_count} kubectl tools registered{'':<26} ║
║  ChromaDB : {CHROMA_DIR:<46} ║
║  Server   : http://{_ARGS.host}:{_ARGS.port:<38} ║
╚════════════════════════════════════════════════════════════╝
""")

    uvicorn.run("app:app", host=_ARGS.host, port=_ARGS.port, reload=_ARGS.reload, log_level="warning")
