# Advanced Hybrid RAG System

A fully local, hardware-agnostic Retrieval-Augmented Generation system built as a proper frontend-backend application. Combines six retrieval paradigms orchestrated by an Adaptive RAG router, designed to manage thousands of documents on consumer hardware.

## Architecture

```
Frontend (Next.js :3000)            Backend (FastAPI :8000)
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
- **Gemma 4B** and embedding model pulled:
  ```bash
  ollama pull gemma3:4b
  ollama pull nomic-embed-text
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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/api** in a browser. You should see JSON listing endpoints (e.g. `GET /api/documents`, `GET /api/chat/sessions`). If you get 404 there, the wrong app is on port 8000 or the backend did not load correctly.

The backend terminal will show Rich-formatted logs for every ingestion step, LLM call, and routing decision.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

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

Full interactive docs available at http://localhost:8000/docs

## Configuration

All backend settings are in `backend/app/config.py`. Key parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gemma3:4b` | Ollama model for reasoning/generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `PAGEINDEX_BEAM_WIDTH` | `2` | Branches explored per tree level |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `RRF_K` | `60` | Reciprocal Rank Fusion smoothing |

Override via environment variables:
```bash
set OLLAMA_BASE_URL=http://192.168.1.100:11434
set LLM_MODEL=gemma3:12b
```

## License

MIT
