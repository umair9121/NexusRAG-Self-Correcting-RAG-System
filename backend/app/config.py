"""
Configuration for the Hybrid RAG System backend.
All Ollama and system-level settings are centralized here.
"""

import os
from pathlib import Path

# ── Supported document types (embeddings + ingestion) ────────────────────────
ALLOWED_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx",
    ".xlsx", ".xlsm", ".xls",
    ".json", ".html", ".htm",
    ".png", ".jpg", ".jpeg",
})

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_ROOT = BASE_DIR.parent                             # Custom_RAG_System/
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
PAGEINDEX_DIR = STORAGE_DIR / "pageindex"
GRAPH_DIR = STORAGE_DIR / "graph_data"
FIGURES_DIR = STORAGE_DIR / "figures"
DATABASE_PATH = STORAGE_DIR / "rag_system.db"

# Ensure dirs exist
for d in [DATA_DIR, STORAGE_DIR, CHROMA_DIR, PAGEINDEX_DIR, GRAPH_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama Server ───────────────────────────────────────────────────────────
# Use 127.0.0.1 by default (more reliable than localhost on Windows/Docker)
# Env: OLLAMA_BASE_URL must be a full URL (e.g. http://127.0.0.1:11434).
# 0.0.0.0 is for Ollama's bind address; clients should connect to 127.0.0.1 or host.
_raw_ollama = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
if not _raw_ollama.startswith(("http://", "https://")):
    _raw_ollama = "http://" + _raw_ollama
# Normalize 0.0.0.0 -> 127.0.0.1 so connecting from this app works (0.0.0.0 is bind-only)
OLLAMA_BASE_URL = _raw_ollama.replace("0.0.0.0", "127.0.0.1")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"

# ── Model Configuration ────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen3-vl:2b")

# ── Inference Parameters ────────────────────────────────────────────────────
REASONING_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": 8192,
}

GENERATION_OPTIONS = {
    "temperature": 0.2,
    "num_ctx": 8192,
}

# ── PageIndex Settings ──────────────────────────────────────────────────────
PAGEINDEX_H1_FONT_SIZE = 16.0
PAGEINDEX_H2_FONT_SIZE = 14.0
PAGEINDEX_BODY_FONT_SIZE = 12.0
PAGEINDEX_BEAM_WIDTH = 2
PAGEINDEX_SUMMARY_MAX_TOKENS = 50

# ── Vector Embeddings Contextual Process Settings ────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CONTEXT_WINDOW_TOKENS = 500
TOP_K_RESULTS = 5

# ── Hybrid Search (RRF) ────────────────────────────────────────────────────
RRF_K = 60

# ── Graph RAG Settings ──────────────────────────────────────────────────────
GRAPH_COMMUNITY_RESOLUTION = 1.0
GRAPH_MIN_COMMUNITY_SIZE = 3

# ── CAG Settings ────────────────────────────────────────────────────────────
CAG_MAX_DOCUMENT_TOKENS = 6000
CAG_ENABLED_RAM_THRESHOLD_GB = 8

# ── VLM Fallback Settings ───────────────────────────────────────────────────
# Chars-per-KB ratio: if Docling markdown / file size falls below this, trigger VLM
VLM_CHARS_PER_KB_THRESHOLD = 2.0
# Don't bother checking the ratio on files smaller than this (KB)
VLM_MIN_FILE_SIZE_KB = 50

# ── Ollama Client Settings ──────────────────────────────────────────────────
# None = no timeout (wait indefinitely for Ollama). Set to an int (seconds) to cap wait time.
OLLAMA_TIMEOUT: int | None = None
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_DELAY = 2
