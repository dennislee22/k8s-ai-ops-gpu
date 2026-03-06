"""
Centralised logging configuration — K8s AI Ops Assistant
All components import get_logger() from here to ensure consistent
formatting, log rotation, and per-component log files.

Log files written to:  ../logs/<component>.log
Console output:        always enabled
Log rotation:          10MB per file, 5 backups kept
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# ── Log directory ─────────────────────────────────────────────────────────

# Resolve logs/ relative to the backend root (one level up from core/)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = _BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Log level from env ────────────────────────────────────────────────────

_LEVEL_MAP = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_LEVEL = _LEVEL_MAP.get(
    os.getenv("LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)

# ── Formatters ────────────────────────────────────────────────────────────

CONSOLE_FORMAT = (
    "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
)
FILE_FORMAT = (
    "%(asctime)s  %(levelname)-8s  [%(name)s]  %(filename)s:%(lineno)d  %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Component → log file mapping ──────────────────────────────────────────

COMPONENT_FILES = {
    "main":         "api.log",
    "agent.agent":  "agent.log",
    "tools.k8s_tool": "k8s.log",
    "rag.rag_tool":   "rag.log",
    "ingest":         "ingest.log",
    # catch-all written by root logger
    "root":           "app.log",
}

_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given component name.
    First call configures the root logger and per-component file handlers.
    Subsequent calls for the same name return the cached logger.

    Usage:
        from core.logger import get_logger
        logger = get_logger(__name__)
    """
    if name in _configured:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Avoid adding duplicate handlers if uvicorn --reload re-imports modules
    if logger.handlers:
        _configured.add(name)
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    )

    # Determine which log file this component writes to
    log_filename = None
    for prefix, filename in COMPONENT_FILES.items():
        if name == prefix or name.startswith(prefix):
            log_filename = filename
            break
    if not log_filename:
        log_filename = "app.log"

    file_handler = logging.handlers.RotatingFileHandler(
        filename=LOG_DIR / log_filename,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(
        logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prevent log records bubbling up to the root logger's default handler
    # (avoids duplicate console output when uvicorn also sets up a root handler)
    logger.propagate = False

    _configured.add(name)
    return logger


def configure_root():
    """
    Configure the root logger once at application startup (called from main.py).
    Also silences overly verbose third-party loggers.
    """
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    # Root console handler
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(LOG_LEVEL)
        ch.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(ch)

    # Root file handler — catches anything not handled by component loggers
    root_file = LOG_DIR / COMPONENT_FILES["root"]
    if not any(
        isinstance(h, logging.handlers.RotatingFileHandler) and
        str(root_file) in str(getattr(h, "baseFilename", ""))
        for h in root.handlers
    ):
        fh = logging.handlers.RotatingFileHandler(
            filename=root_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(fh)

    # Silence noisy third-party libraries
    for noisy in [
        "httpx", "httpcore", "urllib3", "kubernetes.client",
        "langchain", "langsmith", "openai", "watchfiles",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
