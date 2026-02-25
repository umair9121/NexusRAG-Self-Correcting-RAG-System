"""
Configuration for the Hybrid RAG System backend.
All Ollama and system-level settings are centralized here.
"""

import os
from pathlib import Path

# Load backend/.env so EMBEDDING_MODEL, VLM_MODEL, etc. are applied (Python does not load .env by default).
# override=True so that backend/.env wins over shell/system env (e.g. after changing LLM_MODEL in .env).
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# ── Supported document types (embeddings + ingestion) ────────────────────────
ALLOWED_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx",
    ".xlsx", ".xlsm", ".xls",
    ".json", ".html", ".htm",
    ".png", ".jpg", ".jpeg",
})

# Image-only (UI screenshots); PageIndex uses only VLM output for these.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})

# ── Regulatory knowledge base (EDI, HIPAA, WCAG) ─────────────────────────────
# Documents with these tags or filename patterns are forced into Graph RAG pipeline at ingestion.
REGULATORY_TAGS = [s.strip() for s in os.getenv("REGULATORY_TAGS", "regulatory,WCAG,HIPAA,accessibility,EDI").split(",") if s.strip()]
REGULATORY_FILENAME_PATTERNS = [s.strip() for s in os.getenv("REGULATORY_FILENAME_PATTERNS", "wcag,hipaa,accessibility,cfr,ocr").split(",") if s.strip()]
# When True, regulatory docs run only graph (and optionally contextual) to emphasize entity-relationship extraction.
REGULATORY_GRAPH_ONLY = os.getenv("REGULATORY_GRAPH_ONLY", "false").strip().lower() in ("true", "1", "yes")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_ROOT = BASE_DIR.parent                             # Custom_RAG_System/
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
PAGEINDEX_DIR = STORAGE_DIR / "pageindex"
PDF_IMAGES_DIR = STORAGE_DIR / "pdf_images"
GRAPH_DIR = STORAGE_DIR / "graph_data"
FIGURES_DIR = STORAGE_DIR / "figures"
CHROMA_GRAPH_DIR = STORAGE_DIR / "chroma_graph"
KNOWLEDGE_BASE_DIR = STORAGE_DIR / "knowledge_base"
UI_AUDIT_DIR = STORAGE_DIR / "ui_audits"
DATABASE_PATH = STORAGE_DIR / "rag_system.db"

# Ensure dirs exist
for d in [DATA_DIR, STORAGE_DIR, CHROMA_DIR, CHROMA_GRAPH_DIR, PAGEINDEX_DIR, PDF_IMAGES_DIR, GRAPH_DIR, FIGURES_DIR, KNOWLEDGE_BASE_DIR, UI_AUDIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Import folder path tags ──────────────────────────────────────────────────
# Where to START extracting tags: first path segment that matches one of these (e.g. Trader).
# From that segment onward, every slash-separated folder name is used as a tag; no filename.
# Comma-separated in env if you need to override.
_DEFAULT_ANCHOR_SEGMENTS = [
    "Trader", "Transporter", "Pusher", "Swift", "Reliant",
    "DukePoint", "SwartzBay", "Surrey", "Tilbury",
]
_anchor_env = os.getenv("IMPORT_TAG_ANCHOR_SEGMENTS", "").strip()
IMPORT_TAG_ANCHOR_SEGMENTS = (
    [s.strip() for s in _anchor_env.split(",") if s.strip()]
    if _anchor_env
    else _DEFAULT_ANCHOR_SEGMENTS
)

# ── Ollama Server ───────────────────────────────────────────────────────────
# Ollama runs on localhost on the backend machine. Override with OLLAMA_BASE_URL if needed.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"

# ── Model Configuration ────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:30b")
# Lighter model used when "Fast Response" is selected in the Chat pane (vector search + LLM only).
FAST_RESPONSE_MODEL = os.getenv("FAST_RESPONSE_MODEL", "qwen3:4b")
# Lightweight model for internal tasks (KB entity classification, disambiguation). Keeps heavy LLM for chat/generation.
INTERNAL_REASONING_AGENT = os.getenv("INTERNAL_REASONING_AGENT", "phi3:3.8b")
_DEFAULT_EMBEDDING_MODEL = "aksh-desai/qwen3-vl-embedding:8b-q4_k_m"
# If env set to a VLM (e.g. qwen3-vl:8b) that doesn't support /api/embed, use the real embedding model.
_raw_embed = (os.getenv("EMBEDDING_MODEL") or _DEFAULT_EMBEDDING_MODEL).strip()
_is_vlm_not_embedding = "qwen3-vl" in _raw_embed.lower() and "embedding" not in _raw_embed.lower()
EMBEDDING_MODEL = _DEFAULT_EMBEDDING_MODEL if _is_vlm_not_embedding else _raw_embed
# Dimension for embedding model; must match the model or ChromaDB will reject embeddings.
EMBEDDING_DIMENSION = 4096
VLM_MODEL = os.getenv("VLM_MODEL", "qwen3-vl:2b")

# ── Inference Parameters ────────────────────────────────────────────────────
# num_ctx 8192–16384 to avoid silent prompt truncation during extraction and generation.
# This is sent per request. On the Ollama server machine, set OLLAMA_CONTEXT_LENGTH (e.g. 16384)
# when starting "ollama serve" if the server default is lower and you see truncation errors.
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
REASONING_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": DEFAULT_NUM_CTX,
}

GENERATION_OPTIONS = {
    "temperature": 0.2,
    "num_ctx": DEFAULT_NUM_CTX,
}

# ── Graph Extraction (Deterministic) ─────────────────────────────────────────
# Optional smaller model for extraction to save VRAM (e.g. SciPhi Triplex 3.8B).
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "").strip()
# Model used for graph ingestion (extraction + community summaries). Priority: EXTRACTION_MODEL > FAST_RESPONSE_MODEL > LLM_MODEL.
GRAPH_INGESTION_MODEL = (EXTRACTION_MODEL or FAST_RESPONSE_MODEL or LLM_MODEL).strip() or LLM_MODEL
# Minimum context for extraction so large chunks are not truncated.
GRAPH_EXTRACTION_NUM_CTX = int(os.getenv("GRAPH_EXTRACTION_NUM_CTX", "16384"))

# ── PageIndex Settings ──────────────────────────────────────────────────────
# PageIndex uses official node format (node_id, start_index, end_index, summary, nodes)
# and Ollama (LLM_MODEL) for tree building and reasoning-based retrieval.
PAGEINDEX_H1_FONT_SIZE = 16.0
PAGEINDEX_H2_FONT_SIZE = 14.0
PAGEINDEX_BODY_FONT_SIZE = 12.0
PAGEINDEX_BEAM_WIDTH = 2
# Official PageIndex-style summaries: describe main points (allow ~200 tokens, not one short sentence).
PAGEINDEX_SUMMARY_MAX_TOKENS = int(os.getenv("PAGEINDEX_SUMMARY_MAX_TOKENS", "1250"))
# Use legacy PDF parser (font-based) for accurate start_index/end_index; else use pymupdf4llm markdown.
PAGEINDEX_USE_LEGACY_PDF = os.getenv("PAGEINDEX_USE_LEGACY_PDF", "false").lower() in ("true", "1", "yes")
# Delimiter for contextually flattened PageIndex blocks (one block per leaf with hierarchy).
PAGEINDEX_FLATTENED_BLOCK_DELIMITER = "\n\n--- PAGEINDEX BLOCK ---\n\n"

# ── Vector Embeddings Contextual Process Settings ────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CONTEXT_WINDOW_TOKENS = 500
TOP_K_RESULTS = 5
# Semantic chunking: use structure-aware (Markdown/paragraph) chunks when True.
USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "true").lower() in ("true", "1", "yes")
# Max chunk size in words for semantic chunks (merge until this size).
SEMANTIC_CHUNK_MAX_WORDS = int(os.getenv("SEMANTIC_CHUNK_MAX_WORDS", "800"))
SEMANTIC_CHUNK_MIN_WORDS = int(os.getenv("SEMANTIC_CHUNK_MIN_WORDS", "100"))

# ── Hybrid Search (RRF) ────────────────────────────────────────────────────
RRF_K = 60

# ── Reranker (second-pass relevance scoring) ────────────────────────────────
# Reranker re-scores candidates from vector+BM25 for better precision. On by default.
# Requires Ollama with /api/rerank (e.g. Qwen3-Reranker or bge-reranker).
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").strip().lower() in ("true", "1", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "dengcao/Qwen3-Reranker-8B:Q4_K_M").strip()
# How many RRF candidates to send to the reranker (e.g. 20); then return top_k from reranker.
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "20"))

# ── Graph RAG Settings ──────────────────────────────────────────────────────
GRAPH_COMMUNITY_RESOLUTION = 1.0
GRAPH_MIN_COMMUNITY_SIZE = 3
# When > 0, frontend can suggest skipping Graph for documents with page_count >= this (e.g. 500 for long textbooks).
GRAPH_SKIP_PAGE_THRESHOLD = int(os.getenv("GRAPH_SKIP_PAGE_THRESHOLD", "0"))

# ── CAG Settings ────────────────────────────────────────────────────────────
CAG_MAX_DOCUMENT_TOKENS = 6000
CAG_ENABLED_RAM_THRESHOLD_GB = 8

# ── RAG orchestrator context (router + generator) ───────────────────────────
# Max characters of retrieved context sent to the generator. Derived from model context
# if not set: ~4 chars/token. For 120B models with 32K+ context, set RAG_CONTEXT_MAX_CHARS
# (e.g. 100000) or use default. 6000 chars ≈ 1500 tokens; 100000 ≈ 25K tokens.
_chars_per_token = 4
RAG_CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", str(DEFAULT_NUM_CTX * _chars_per_token)))
# Share of that budget for each retrieval route (PageIndex gets the most).
RAG_CONTEXT_SHARE_PAGEINDEX = float(os.getenv("RAG_CONTEXT_SHARE_PAGEINDEX", "0.53"))
RAG_CONTEXT_SHARE_VECTOR = float(os.getenv("RAG_CONTEXT_SHARE_VECTOR", "0.33"))
RAG_CONTEXT_SHARE_GRAPH = float(os.getenv("RAG_CONTEXT_SHARE_GRAPH", "0.14"))

# ── Knowledge Base Settings ──────────────────────────────────────────────────
KB_ENTITY_CLASSIFICATION_BATCH_SIZE = 30
KB_AUTO_REFRESH_INTERVAL_SECONDS = int(os.getenv("KB_AUTO_REFRESH_INTERVAL", "3600"))

# ── VLM Fallback Settings ───────────────────────────────────────────────────
# Chars-per-KB ratio: if Docling markdown / file size falls below this, trigger VLM
VLM_CHARS_PER_KB_THRESHOLD = 2.0
# Don't bother checking the ratio on files smaller than this (KB)
VLM_MIN_FILE_SIZE_KB = 50
# For PDFs with a text layer: if average chars per page is below this, treat as image-heavy and run full VLM extraction so context is persisted (PageIndex, vector, graph)
VLM_MIN_CHARS_PER_PAGE = int(os.getenv("VLM_MIN_CHARS_PER_PAGE", "200"))

# ── PDF Triage (page-level routing) ─────────────────────────────────────────
# Per-page: above this char count = text_heavy (selectable). Below image_heavy_char_max = image_heavy.
PDF_TRIAGE_TEXT_HEAVY_CHARS = int(os.getenv("PDF_TRIAGE_TEXT_HEAVY_CHARS", "50"))
PDF_TRIAGE_IMAGE_HEAVY_CHAR_MAX = int(os.getenv("PDF_TRIAGE_IMAGE_HEAVY_CHAR_MAX", "50"))

# ── OCR (fast path for scanned text; not the heavy VLM) ──────────────────────
# Ollama model for OCR (e.g. qwen3-vl:2b). Use with Tesseract: Tesseract for plain scans, OCR_MODEL for complex columns/tables.
OCR_MODEL = os.getenv("OCR_MODEL", "maternion/LightOnOCR-2:1b").strip()
# Use triage + routing: fast_pageindex / ocr_scanned / mixed_layout / complex_visual. If False, keep previous behavior.
PDF_TRIAGE_ENABLED = os.getenv("PDF_TRIAGE_ENABLED", "true").strip().lower() in ("true", "1", "yes")

# When True, use reasoning-based PageIndex retrieval and VLM with retrieved page images for answer generation when available.
USE_VLM_VISUAL_ANSWER = os.getenv("USE_VLM_VISUAL_ANSWER", "true").strip().lower() in ("true", "1", "yes")

# ── Ollama Client Settings ──────────────────────────────────────────────────
# None = no timeout (wait indefinitely for Ollama). Set to an int (seconds) to cap wait time.
OLLAMA_TIMEOUT: int | None = None
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_DELAY = 2
