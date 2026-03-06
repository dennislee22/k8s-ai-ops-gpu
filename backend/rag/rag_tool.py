"""
Phase 2 — RAG Tool
Ingests markdown/PDF documents into pgvector.
Retrieves relevant chunks at query time using nomic-embed-text via Ollama.
"""

import sys as _sys
from pathlib import Path as _Path
_BACKEND_DIR = _Path(__file__).resolve().parent
while not (_BACKEND_DIR / "main.py").exists() and _BACKEND_DIR != _BACKEND_DIR.parent:
    _BACKEND_DIR = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in _sys.path:
    _sys.path.insert(0, str(_BACKEND_DIR))


import os
import re
import time
import logging
from pathlib import Path
from typing import Optional
import psycopg2
from psycopg2.extras import execute_values
import httpx
from dotenv import load_dotenv

load_dotenv("env")

from core.logger import get_logger
logger = get_logger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K           = 5


def _get_db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "k8sops"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def init_db():
    logger.info("[RAG] Initialising pgvector schema...")
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id          SERIAL PRIMARY KEY,
                    source      TEXT NOT NULL,
                    doc_type    TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   vector(768),
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id          SERIAL PRIMARY KEY,
                    file_path   TEXT UNIQUE NOT NULL,
                    file_hash   TEXT NOT NULL,
                    ingested_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()
        logger.info("[RAG] Schema ready")
    except Exception as e:
        logger.error(f"[RAG] Schema init failed: {e}")
        raise
    finally:
        conn.close()


def embed_text(text: str) -> list[float]:
    t0 = time.time()
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30.0,
    )
    response.raise_for_status()
    elapsed = time.time() - t0
    logger.debug(f"[RAG] Embedded {len(text)} chars in {elapsed:.2f}s")
    return response.json()["embedding"]


def chunk_text(text: str) -> list[str]:
    chunks = []
    start  = 0
    text   = text.strip()
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


def _detect_doc_type(filename: str) -> str:
    name = filename.lower()
    if any(kw in name for kw in ["known","issue","bug","error"]):    return "known_issue"
    if any(kw in name for kw in ["runbook","playbook","procedure"]): return "runbook"
    if any(kw in name for kw in ["dos","donts","guidelines","policy"]): return "dos_donts"
    return "general"


def ingest_file(file_path: str, force: bool = False) -> dict:
    import hashlib
    path      = Path(file_path)
    file_hash = hashlib.md5(path.read_bytes()).hexdigest()

    logger.info(f"[RAG] Ingesting {path.name} (hash={file_hash[:8]}...)")

    conn = _get_db_conn()
    try:
        if not force:
            with conn.cursor() as cur:
                cur.execute("SELECT file_hash FROM ingestion_log WHERE file_path = %s", (str(path),))
                row = cur.fetchone()
                if row and row[0] == file_hash:
                    logger.info(f"[RAG] Skipping {path.name} — unchanged")
                    return {"file": path.name, "status": "skipped", "chunks": 0}

        # Read content
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text   = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            logger.info(f"[RAG] PDF extracted — {len(reader.pages)} pages")
        elif path.suffix.lower() == ".md":
            from markdown_it import MarkdownIt
            raw  = path.read_text(encoding="utf-8")
            html = MarkdownIt().render(raw)
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            logger.info(f"[RAG] Markdown parsed — {len(text)} chars")
        else:
            text = path.read_text(encoding="utf-8")

        if not text.strip():
            logger.warning(f"[RAG] {path.name} is empty — skipping")
            return {"file": path.name, "status": "empty", "chunks": 0}

        chunks   = chunk_text(text)
        doc_type = _detect_doc_type(path.name)
        logger.info(f"[RAG] {path.name} → {len(chunks)} chunks (type={doc_type})")

        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE source = %s", (str(path),))

        rows = []
        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk)
            rows.append((str(path), doc_type, i, chunk, embedding))
            if (i + 1) % 10 == 0:
                logger.info(f"[RAG] Embedded {i+1}/{len(chunks)} chunks...")

        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO documents (source, doc_type, chunk_index, content, embedding) VALUES %s",
                rows,
                template="(%s, %s, %s, %s, %s::vector)",
            )
            cur.execute(
                """INSERT INTO ingestion_log (file_path, file_hash)
                   VALUES (%s, %s)
                   ON CONFLICT (file_path) DO UPDATE
                   SET file_hash = EXCLUDED.file_hash, ingested_at = NOW()""",
                (str(path), file_hash),
            )
        conn.commit()
        logger.info(f"[RAG] ✓ {path.name} ingested — {len(chunks)} chunks")
        return {"file": path.name, "status": "ingested", "chunks": len(chunks), "doc_type": doc_type}

    except Exception as e:
        logger.error(f"[RAG] Failed to ingest {path.name}: {e}", exc_info=True)
        return {"file": path.name, "status": "error", "chunks": 0, "error": str(e)}
    finally:
        conn.close()


def ingest_directory(docs_dir: str, force: bool = False) -> list[dict]:
    docs_path = Path(docs_dir)
    files     = list(docs_path.glob("**/*.md")) + list(docs_path.glob("**/*.pdf"))
    if not files:
        logger.warning(f"[RAG] No .md or .pdf files found in {docs_dir}")
        return []
    logger.info(f"[RAG] Found {len(files)} files to ingest in {docs_dir}")
    results = []
    for fp in sorted(files):
        results.append(ingest_file(str(fp), force=force))
    total = sum(r.get("chunks", 0) for r in results)
    logger.info(f"[RAG] Ingestion complete — {len(results)} files, {total} total chunks")
    return results


def retrieve(query: str, top_k: int = TOP_K, doc_type: Optional[str] = None) -> str:
    logger.info(f"[RAG] Retrieving top {top_k} chunks for query: '{query[:60]}...'")
    t0             = time.time()
    query_embedding = embed_text(query)

    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            if doc_type:
                cur.execute("""
                    SELECT source, doc_type, content,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM documents WHERE doc_type = %s
                    ORDER BY embedding <=> %s::vector LIMIT %s
                """, (query_embedding, doc_type, query_embedding, top_k))
            else:
                cur.execute("""
                    SELECT source, doc_type, content,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM documents
                    ORDER BY embedding <=> %s::vector LIMIT %s
                """, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()

        elapsed = time.time() - t0
        logger.info(f"[RAG] Retrieved {len(rows)} chunks in {elapsed:.2f}s")

        if not rows:
            return "No relevant documentation found."

        lines = [f"Retrieved {len(rows)} relevant document chunks:\n"]
        for i, (source, dt, content, similarity) in enumerate(rows, 1):
            lines.append(
                f"[{i}] Source: {Path(source).name} | Type: {dt} | Relevance: {similarity:.2f}\n{content}\n"
            )
        return "\n".join(lines)
    finally:
        conn.close()


def get_doc_stats() -> str:
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_type, COUNT(*) as chunks, COUNT(DISTINCT source) as files FROM documents GROUP BY doc_type ORDER BY doc_type")
            rows = cur.fetchall()
        if not rows:
            return "No documents ingested yet."
        lines = ["Document store stats:"]
        for doc_type, chunks, files in rows:
            lines.append(f"  {doc_type}: {files} files, {chunks} chunks")
        return "\n".join(lines)
    finally:
        conn.close()


RAG_TOOLS = {
    "search_documentation": {
        "fn": retrieve,
        "description": (
            "Search the internal knowledge base for known issues, runbooks, and guidelines. "
            "Use when you observe a K8s anomaly to check if it matches a known issue. "
            "Always cross-reference live cluster data with documentation before diagnosing."
        ),
        "parameters": {
            "query":    {"type": "string",  "description": "Natural language search query"},
            "top_k":    {"type": "integer", "default": 5},
            "doc_type": {"type": "string",  "default": None, "description": "known_issue | runbook | dos_donts | None"},
        },
    },
}
