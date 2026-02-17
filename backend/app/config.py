"""
Configuration for the Hybrid RAG System backend.
All Ollama and system-level settings are centralized here.
Focused on GraphRAG + Image Extraction + EDI Evaluation.
"""

import os
from pathlib import Path

# ── Supported document types (ingestion) ─────────────────────────────────────
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
GRAPH_DIR = STORAGE_DIR / "graph_data"
FIGURES_DIR = STORAGE_DIR / "figures"
EXTRACTION_DIR = STORAGE_DIR / "extractions"
DATABASE_PATH = STORAGE_DIR / "rag_system.db"

# Ensure dirs exist
for d in [DATA_DIR, STORAGE_DIR, GRAPH_DIR, FIGURES_DIR, EXTRACTION_DIR]:
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
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b-it-qat")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:3b")

# ── Inference Parameters ────────────────────────────────────────────────────
REASONING_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": 8192,
}

GENERATION_OPTIONS = {
    "temperature": 0.2,
    "num_ctx": 8192,
}

# ── Graph RAG Settings ──────────────────────────────────────────────────────
GRAPH_COMMUNITY_RESOLUTION = 1.0
GRAPH_MIN_COMMUNITY_SIZE = 3

# ── CAG Settings ────────────────────────────────────────────────────────────
CAG_MAX_DOCUMENT_TOKENS = 6000
CAG_ENABLED_RAM_THRESHOLD_GB = 8

# ── VLM Extraction Settings ─────────────────────────────────────────────────
VLM_MAX_NEW_TOKENS = 700

# ── Ollama Client Settings ──────────────────────────────────────────────────
# None = no timeout (wait indefinitely for Ollama). Set to an int (seconds) to cap wait time.
OLLAMA_TIMEOUT: int | None = None
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_DELAY = 2
