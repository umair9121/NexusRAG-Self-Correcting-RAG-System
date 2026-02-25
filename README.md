# Advanced Hybrid RAG System

A fully local, hardware-agnostic Retrieval-Augmented Generation system built as a proper frontend-backend application. Combines six retrieval paradigms orchestrated by an Adaptive RAG router, designed to manage thousands of documents on consumer hardware.

## Architecture

```
Frontend (Next.js :3000)            Backend (FastAPI :8001)
┌──────────────────────┐           ┌──────────────────────────┐
│  Document Dashboard  │──HTTP──>  │  REST API Routes         │
│  Chat Interface      │──HTTP──>  │  Background Ingestion    │
│  Live Log Viewer     │──WS────>  │  WebSocket Log Stream    │
│  Data Explorer       │──HTTP──>  │  Rich Terminal Logging   │
└──────────────────────┘           └────────────┬─────────────┘
                                                │
                                   ┌────────────▼─────────────┐
                                   │  SQLite  │ ChromaDB      │
                                   │  NetworkX│ Ollama :11434 │
                                   └──────────────────────────┘
```

### Retrieval Modules

| Module | Best For | Method |
|--------|----------|--------|
| **PageIndex** | Structural / navigational queries | LLM-guided tree traversal with beam search |
| **Contextual + Hybrid** | Specific fact lookups | Contextual enrichment + Vector + BM25 + RRF fusion |
| **Graph RAG** | Thematic / relational queries | Entity extraction + Louvain communities + dual-mode search |
| **CAG** | Conversational follow-ups | Full document preloaded into Ollama KV cache |

### Adaptive RAG Flow

1. **Router** classifies query intent and selects the retrieval engine
2. **Retriever** fetches relevant context (PageIndex / Vector / Graph / CAG)
3. **Grader** evaluates if the context answers the query
4. **Rewriter** rewrites the query and retries with a different engine (if needed)
5. **Generator** synthesizes the final answer

## Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Ollama** running locally (`ollama serve`)
- **LLM and embedding models** pulled (match `backend/.env` / `LLM_MODEL`, `EMBEDDING_MODEL`):
  ```bash
  ollama pull qwen3:30b
  ollama pull aksh-desai/qwen3-vl-embedding:8b-q4_k_m
  ```

## Quick Start

### Backend

**Important:** The backend must be run so that the `app` package is `backend/app`. Either:

**Option A – from project root (recommended):**
```bash
cd Custom_RAG_System
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r backend/requirements.txt
python backend/run.py
```

**Option B – from backend directory:**
```bash
cd Custom_RAG_System/backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Then open **http://localhost:8001/api** in a browser. You should see JSON listing endpoints (e.g. `GET /api/documents`, `GET /api/chat/sessions`). If you get 404 there, the wrong app is on port 8001 or the backend did not load correctly.

The backend terminal will show Rich-formatted logs for every ingestion step, LLM call, and routing decision.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## Deploy backend on another machine (e.g. Linux server)

You can run the backend on a separate machine and use the frontend from your PC.

### On the backend machine (Linux)

1. **Clone the repo** (use your branch if you use one):
   ```bash
   git clone https://github.com/YOUR_USER/Custom_RAG_System.git
   cd Custom_RAG_System
   ```

2. **Ollama** must be installed and running on this machine (default `127.0.0.1:11434`). Pull the models this backend will use, e.g.:
   ```bash
   ollama pull qwen3:30b
   ollama pull aksh-desai/qwen3-vl-embedding:8b-q4_k_m
   ```

3. **Python venv and dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

4. **Set model env vars** (optional; defaults are in `backend/app/config.py`). Either copy `backend/.env.example` to `backend/.env`, edit model names to match what you pulled, then before starting:
   ```bash
   set -a && source backend/.env && set +a
   ```
   Or export before running:
   ```bash
   export LLM_MODEL=qwen3:30b
   export EMBEDDING_MODEL=aksh-desai/qwen3-vl-embedding:8b-q4_k_m
   export VLM_MODEL=qwen3-vl:8b
   ```
   (If you use a process manager like systemd, set `Environment=LLM_MODEL=qwen3:30b` etc. in the unit file.) **Restart the backend after changing `.env`** so the new model is used; the startup log shows which LLM/EMBEDDING/VLM models are active.

5. **Start the backend** (binds to all interfaces so your PC can connect):
   ```bash
   python backend/run.py
   ```
   Backend listens on **port 8001**. Ensure the machine’s firewall allows inbound TCP on 8001.

6. **Note:** The frontend does **not** need to be on this machine. Chat and health use the backend’s `/api/system/health`; the list of Ollama models shown in the UI is the one from this backend (and its Ollama), not from your PC.

### On your PC (frontend)

1. Open the repo (same branch as the backend).
2. In the frontend directory, set the API base URL to the backend machine’s IP (or hostname). For example if the backend is at `100.117.26.120`:
   ```bash
   cd frontend
   # Create .env.local (create the file if it doesn't exist)
   echo "NEXT_PUBLIC_API_URL=http://100.117.26.120:8001" > .env.local
   npm install
   npm run dev
   ```
3. Open http://localhost:3000. All API and WebSocket requests will go to `http://100.117.26.120:8001`. The chat model dropdown and health info will reflect the backend machine’s Ollama and models.

## Project Structure

```
Custom_RAG_System/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, lifespan
│   │   ├── config.py            # All settings (Ollama, models, paths)
│   │   ├── database.py          # Async SQLite (SQLAlchemy)
│   │   ├── models.py            # ORM: Document, ChatSession
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   ├── logger.py            # Rich logging + WebSocket broadcaster
│   │   ├── routes/
│   │   │   ├── documents.py     # CRUD, upload, search, bulk ops
│   │   │   ├── ingestion.py     # Start/cancel/status ingestion jobs
│   │   │   ├── query.py         # Adaptive RAG query endpoint
│   │   │   ├── chat.py          # Chat sessions + CAG management
│   │   │   ├── system.py        # Health check, stats
│   │   │   └── websocket.py     # Live log streaming
│   │   ├── services/
│   │   │   ├── document_service.py
│   │   │   └── ingestion_worker.py  # Background async worker
│   │   └── modules/             # RAG engine modules
│   │       ├── ollama_client.py
│   │       ├── pageindex.py
│   │       ├── contextual.py
│   │       ├── graph.py
│   │       ├── cag.py
│   │       └── router.py        # LangGraph Adaptive RAG
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root layout with sidebar
│   │   │   ├── page.tsx         # Document dashboard
│   │   │   ├── chat/page.tsx    # Chat interface
│   │   │   ├── logs/page.tsx    # Live log viewer
│   │   │   └── explore/page.tsx # Data explorer
│   │   ├── components/
│   │   │   ├── documents/       # DocumentTable, UploadModal
│   │   │   ├── chat/            # ChatPanel, TraceViewer
│   │   │   ├── logs/            # LogStream
│   │   │   └── layout/          # Sidebar
│   │   └── lib/
│   │       ├── api.ts           # Backend API client
│   │       ├── websocket.ts     # WebSocket hook for logs
│   │       └── types.ts         # TypeScript interfaces
│   └── package.json
├── data/                        # Uploaded PDFs
└── storage/
    ├── chroma/                  # ChromaDB vector store
    ├── pageindex/               # JSON tree files
    └── graph_data/              # NetworkX graph files
```

## Key Features

### Document Management (scales to thousands)
- Upload, search, filter, tag, and bulk-manage PDFs
- Per-document ingestion status tracking (PageIndex / Contextual / Graph)
- Server-side pagination (50 docs/page)

### Background Ingestion with Live Monitoring
- Sequential processing queue respecting hardware constraints
- Each pipeline (PageIndex, Contextual, Graph) runs independently
- Cancel support for long-running jobs
- All progress streamed via WebSocket to the Live Log Viewer

### Terminal Logging
- Backend uses Rich for color-coded, module-labeled terminal output
- Every LLM call, routing decision, and ingestion step is logged
- Same logs streamed to the frontend Log Viewer

### Chat with Reasoning Traces
- Multiple chat sessions with persistent message history
- Collapsible colored reasoning trace showing the full retrieval path
- Force-route and temperature controls per query
- CAG integration for zero-latency document Q&A

## API Reference

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/documents` | List with pagination + search + filter |
| POST | `/api/documents/upload` | Upload PDFs |
| DELETE | `/api/documents/{id}` | Delete doc + indexed data |
| PATCH | `/api/documents/{id}` | Update tags, notes |
| POST | `/api/documents/bulk-delete` | Bulk delete |
| POST | `/api/ingestion/start` | Start ingestion (doc IDs + pipelines) |
| GET | `/api/ingestion/status` | Queue / active job status |
| POST | `/api/ingestion/cancel/{id}` | Cancel job |
| POST | `/api/query` | Adaptive RAG query |
| GET/POST | `/api/chat/sessions` | List / create sessions |
| POST | `/api/chat/sessions/{id}/message` | Send message |
| GET | `/api/system/health` | System health |
| WS | `/ws/logs` | Live log stream |

Full interactive docs available at http://localhost:8001/docs

## Graph RAG and ingestion optimizations

The system implements a structured, deterministic Graph RAG pipeline suitable for local LLMs:

- **Unified PageIndex parsing (PDF + OCR paths)**  
  Digital PDFs now use **pymupdf4llm** to convert native PDF layout into Markdown first, then reuse the same `build_tree_from_markdown` parser used by Docling/VLM OCR output. This reduces heading/paragraph fragmentation from raw span iteration and keeps PageIndex tree construction consistent across document types. If `pymupdf4llm` is unavailable (or returns empty output), PageIndex falls back to the legacy PyMuPDF span parser.

- **Structurally aware ingestion**  
  PDFs are converted to hierarchical Markdown via **Docling** (OCR + layout + tables) before chunking. **Semantic chunking** (structure-aware from Markdown or paragraph boundaries) replaces fixed-token splitting for both vector and graph pipelines, preserving conceptual continuity.

- **Deterministic extraction**  
  Entity/relationship extraction uses **Ollama’s JSON Schema `format`** so the model outputs valid triples at the logit level. Context length for extraction is set to **8192–16384** tokens (`OLLAMA_NUM_CTX` / `GRAPH_EXTRACTION_NUM_CTX`) to avoid silent truncation. An optional smaller **extraction model** (`EXTRACTION_MODEL`, e.g. SciPhi Triplex 3.8B) can be used to save VRAM.

- **LightRAG-style metadata**  
  The extraction prompt asks for **entity_description**, **relationship_description**, and **relationship_keywords**. These are stored on NetworkX nodes/edges and embedded into an **auxiliary Chroma collection** (`storage/chroma_graph`) for hybrid k-NN semantic search over the graph.

- **Hybrid vector–graph retrieval**  
  Graph search combines (1) k-NN over node/edge description embeddings, (2) graph traversal from matched nodes, and (3) **raw document chunks** from the vector pipeline appended to the prompt so answers are grounded even when the graph has no direct edge between query and answer.

## Configuration

All backend settings are in `backend/app/config.py`. Key parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL (localhost on backend machine) |
| `LLM_MODEL` | `qwen3:30b` | Ollama model for reasoning/generation (PageIndex, chat, graph) |
| `EMBEDDING_MODEL` | `aksh-desai/qwen3-vl-embedding:8b-q4_k_m` | Embedding model (VL: text + images) |
| `OLLAMA_NUM_CTX` | `16384` | Context length (tokens) this app sends in API requests; avoid truncation |
| *Ollama server* `OLLAMA_CONTEXT_LENGTH` | *(varies by VRAM)* | Set **on the machine running `ollama serve`** (e.g. `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`) if you hit context/truncation errors; see [Ollama context length](https://docs.ollama.com/context-length) |
| `EXTRACTION_MODEL` | *(empty)* | Optional smaller model for graph extraction |
| `GRAPH_EXTRACTION_NUM_CTX` | `16384` | Context length for extraction |
| `USE_SEMANTIC_CHUNKING` | `true` | Use structure-aware chunking |
| `SEMANTIC_CHUNK_MAX_WORDS` | `800` | Max words per semantic chunk |
| `PAGEINDEX_BEAM_WIDTH` | `2` | Branches explored per tree level |
| `CHUNK_SIZE` | `500` | Words per chunk (when semantic chunking off) |
| `RRF_K` | `60` | Reciprocal Rank Fusion smoothing |

Override via environment variables or `backend/.env` (`.env` overrides shell when the backend starts). Restart the backend after changing so the new model is used:
```bash
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set LLM_MODEL=qwen3:30b
```

## License

MIT
