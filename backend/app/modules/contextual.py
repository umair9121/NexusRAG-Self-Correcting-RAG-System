"""
Module II: Vector Embeddings Contextual Process, Multivector & Hybrid Search
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi

from app import config
from app.modules.ollama_client import OllamaClient, AsyncOllamaClient

logger = logging.getLogger(__name__)

_chroma_client = None
_chroma_collection = None


def _get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        import chromadb
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="rag_documents", metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def _reset_chroma_collection():
    """Delete and recreate the ChromaDB collection (dimension mismatch recovery)."""
    global _chroma_client, _chroma_collection
    import chromadb
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        _chroma_client.delete_collection("rag_documents")
        logger.warning("Deleted ChromaDB collection 'rag_documents' due to dimension mismatch.")
    except Exception as exc:
        logger.warning("Could not delete collection: %s", exc)
    _chroma_collection = _chroma_client.get_or_create_collection(
        name="rag_documents", metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


def _delete_chroma_by_doc_name(doc_name: str) -> int:
    """Remove all ChromaDB vectors for the given doc_name. Returns count removed."""
    collection = _get_chroma_collection()
    try:
        result = collection.get(where={"doc_name": doc_name}, include=[])
    except Exception as exc:
        logger.warning("Chroma get for delete doc_name=%s failed: %s", doc_name, exc)
        return 0
    ids = result.get("ids") or []
    if not ids:
        return 0
    try:
        collection.delete(ids=ids)
        return len(ids)
    except Exception as exc:
        logger.warning("Chroma delete for doc_name=%s failed: %s", doc_name, exc)
        return 0


def get_chroma_entries_for_document(doc_name: str) -> dict[str, Any] | None:
    """
    Return ChromaDB entries for the given document stem (filename without extension).
    Used by the metadata API so Storage Metadata can show vector embeddings per document.
    """
    collection = _get_chroma_collection()
    try:
        result = collection.get(
            where={"doc_name": doc_name},
            include=["embeddings", "documents", "metadatas"],
        )
    except Exception as exc:
        logger.warning("Chroma get for doc_name=%s failed: %s", doc_name, exc)
        return None
    if not result or not result["ids"]:
        return None
    ids = result["ids"]
    # Avoid "truth value of array is ambiguous" — ChromaDB may return numpy arrays
    raw_emb = result.get("embeddings")
    raw_docs = result.get("documents")
    raw_meta = result.get("metadatas")
    embeddings = [] if raw_emb is None else raw_emb
    documents = [] if raw_docs is None else raw_docs
    metadatas = [] if raw_meta is None else raw_meta
    n = len(ids)
    chunks = []
    for i, chunk_id in enumerate(ids):
        emb = embeddings[i] if i < len(embeddings) else []
        # Ensure JSON-serializable (ChromaDB can return numpy arrays)
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        elif not isinstance(emb, list):
            emb = list(emb) if emb is not None else []
        chunks.append({
            "id": chunk_id,
            "document": documents[i] if i < len(documents) else None,
            "metadata": metadatas[i] if i < len(metadatas) else None,
            "embedding": emb,
            "embedding_dimensions": len(emb),
        })
    return {"chunk_count": len(chunks), "chunks": chunks}


class DocStore:
    def __init__(self, path: Path | None = None):
        self.path = path or (config.STORAGE_DIR / "docstore.json")
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._store = json.load(f)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False)

    def put(self, doc_id: str, data: dict[str, Any]):
        self._store[doc_id] = data
        self._save()

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._store.get(doc_id)

    def put_batch(self, items: list[tuple[str, dict[str, Any]]]):
        for doc_id, data in items:
            self._store[doc_id] = data
        self._save()

    def count(self) -> int:
        return len(self._store)

    def all_ids(self) -> list[str]:
        return list(self._store.keys())

    def clear(self):
        self._store = {}
        self._save()

    def delete_by_doc_name(self, doc_name: str) -> int:
        """Remove all entries for the given doc_name. Returns count removed."""
        to_remove = [k for k, v in self._store.items() if v.get("doc_name") == doc_name]
        for k in to_remove:
            del self._store[k]
        if to_remove:
            self._save()
        return len(to_remove)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    doc = fitz.open(str(pdf_path))
    text_parts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(text_parts)


# ── Placeholder stripping (Docling/export_to_markdown emits <!-- image --> etc.) ──

_HTML_COMMENT_LINE = re.compile(r"^\s*<!--.*?-->\s*$", re.IGNORECASE | re.DOTALL)


def sanitize_extracted_text(text: str) -> str:
    """
    Remove lines that are only HTML comments (e.g. <!-- image -->, <!-- figure -->)
    from Docling/export_to_markdown so they never become chunks and trigger
    useless LLM context like "Please provide the image chunks...".
    """
    if not text or not text.strip():
        return text
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _HTML_COMMENT_LINE.match(stripped):
            continue
        kept.append(line)
    result = "\n".join(kept)
    # Collapse many blank lines into at most two
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    return result


def _is_placeholder_chunk(chunk_text: str) -> bool:
    """True if chunk has no real content (only HTML comment, whitespace, or trivial placeholder)."""
    s = chunk_text.strip()
    if not s:
        return True
    if _HTML_COMMENT_LINE.match(s):
        return True
    # Single line that is only a short HTML comment or placeholder
    if len(s) < 80 and ("<!--" in s and "-->" in s):
        return True
    return False


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        char_start = len(" ".join(words[:start]))
        char_end = char_start + len(chunk_text_str)
        chunks.append({
            "id": str(uuid.uuid4())[:12], "text": chunk_text_str,
            "index": len(chunks), "start_char": char_start, "end_char": char_end,
        })
        start += chunk_size - overlap
    return chunks


class ContextualRetriever:
    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self.async_llm = AsyncOllamaClient()
        self.docstore = DocStore()

    def ingest(self, pdf_path: str | Path, callback: Any | None = None) -> int:
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.stem
        if callback:
            callback(f"Extracting text from {pdf_path.name}...")
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            return 0
        return self.ingest_text(doc_name, full_text, callback)

    def ingest_text(self, doc_name: str, full_text: str, callback: Any | None = None) -> int:
        """Ingest pre-extracted text (e.g. from deepdoctection for image-based PDFs)."""
        if not full_text.strip():
            return 0
        full_text = sanitize_extracted_text(full_text)
        if not full_text.strip():
            return 0
        # Replace existing chunks for this doc so re-ingestion removes old placeholder garbage
        n_chroma = _delete_chroma_by_doc_name(doc_name)
        n_docstore = self.docstore.delete_by_doc_name(doc_name)
        if n_chroma or n_docstore:
            logger.info("Cleared %d Chroma and %d DocStore entries for %s", n_chroma, n_docstore, doc_name)
        if callback:
            callback("Chunking document...")
        chunks = chunk_text(full_text)
        if callback:
            callback(f"Created {len(chunks)} chunks. Vector embeddings contextual process...")
        loop = asyncio.new_event_loop()
        try:
            contextualized = loop.run_until_complete(
                self._contextualize_chunks(chunks, full_text, doc_name, callback)
            )
        finally:
            loop.close()
        if callback:
            callback("Generating embeddings and indexing...")
        self._index_chunks(contextualized, doc_name)
        if callback:
            callback(f"Indexed {len(contextualized)} chunks for {doc_name}")
        return len(contextualized)

    async def _contextualize_chunks(self, chunks, full_text, doc_name, callback=None):
        words = full_text.split()
        results = []
        for i, chunk in enumerate(chunks):
            # Skip placeholder-only chunks (e.g. <!-- image --> from Docling); do not index or call LLM
            if _is_placeholder_chunk(chunk["text"]):
                logger.debug("Skipping placeholder chunk %d (no real content)", i)
                continue
            chunk_start_word = chunk["index"] * (config.CHUNK_SIZE - config.CHUNK_OVERLAP)
            context_start = max(0, chunk_start_word - config.CONTEXT_WINDOW_TOKENS)
            context_end = min(len(words), chunk_start_word + config.CHUNK_SIZE + config.CONTEXT_WINDOW_TOKENS)
            surrounding = " ".join(words[context_start:context_end])
            prompt = (
                "Describe the context of the following text chunk. "
                "Who is speaking? What is the main topic? Answer in one sentence.\n\n"
                f"Surrounding context:\n{surrounding[:2000]}\n\nChunk:\n{chunk['text'][:1500]}"
            )
            try:
                context_desc = await self.async_llm.generate(prompt, options=config.REASONING_OPTIONS)
                context_desc = context_desc.strip()
            except Exception as exc:
                logger.warning("Failed to contextualize chunk %d: %s", i, exc)
                context_desc = ""
            contextualized_text = f"{context_desc} {chunk['text']}" if context_desc else chunk['text']
            parent_id = str(uuid.uuid4())[:12]
            results.append({
                "id": parent_id, "contextualized_text": contextualized_text,
                "original_text": chunk["text"], "context_description": context_desc,
                "doc_name": doc_name, "chunk_index": chunk["index"],
            })
            self.docstore.put(parent_id, {
                "text": chunk["text"], "doc_name": doc_name,
                "chunk_index": chunk["index"], "context": context_desc,
            })
            if callback and (i + 1) % 5 == 0:
                callback(f"Processed {i + 1}/{len(chunks)} chunks (vector embeddings contextual)")
        return results

    def _index_chunks(self, chunks, doc_name):
        collection = _get_chroma_collection()
        texts = [c["contextualized_text"] for c in chunks]
        embeddings = self._batch_embed(texts)
        if not embeddings:
            return
        ids = [c["id"] for c in chunks]
        metadatas = [
            {"doc_name": c["doc_name"], "chunk_index": c["chunk_index"], "context": c["context_description"][:500]}
            for c in chunks
        ]
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            be = min(i + batch_size, len(ids))
            try:
                collection.upsert(ids=ids[i:be], embeddings=embeddings[i:be], documents=texts[i:be], metadatas=metadatas[i:be])
            except ValueError as exc:
                if "dimension" in str(exc).lower():
                    logger.warning("Embedding dimension mismatch — resetting ChromaDB collection: %s", exc)
                    collection = _reset_chroma_collection()
                    # Retry the full upsert from the beginning on the fresh collection
                    for j in range(0, len(ids), batch_size):
                        je = min(j + batch_size, len(ids))
                        collection.upsert(ids=ids[j:je], embeddings=embeddings[j:je], documents=texts[j:je], metadatas=metadatas[j:je])
                    return
                raise

    def _batch_embed(self, texts):
        all_embs = []
        fallback_dim = None  # determined from first successful batch
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            try:
                embs = self.llm.embed(batch)
                all_embs.extend(embs)
                if fallback_dim is None and embs:
                    fallback_dim = len(embs[0])
            except Exception as exc:
                logger.error("Embedding batch failed: %s", exc)
                dim = fallback_dim or 2560  # default to qwen3-embedding:4b dim
                all_embs.extend([[0.0] * dim] * len(batch))
        return all_embs

    def vector_search(self, query: str, top_k: int = config.TOP_K_RESULTS):
        collection = _get_chroma_collection()
        query_embedding = self.llm.embed(query)
        if not query_embedding:
            return []
        try:
            results = collection.query(query_embeddings=query_embedding, n_results=top_k, include=["documents", "metadatas", "distances"])
        except ValueError as exc:
            if "dimension" in str(exc).lower():
                logger.warning("Embedding dimension mismatch during search — resetting collection: %s", exc)
                _reset_chroma_collection()
                return []
            raise
        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            parent = self.docstore.get(doc_id)
            parent_text = parent["text"] if parent else results["documents"][0][i]
            docs.append({
                "id": doc_id, "text": parent_text,
                "contextualized": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            })
        return docs


class BM25Index:
    def __init__(self, docstore: DocStore):
        self.docstore = docstore
        self._corpus_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._build()

    def _build(self):
        self._corpus_ids = self.docstore.all_ids()
        corpus = []
        for doc_id in self._corpus_ids:
            doc = self.docstore.get(doc_id)
            text = doc.get("text", "") if doc else ""
            corpus.append(text.lower().split())
        if corpus:
            self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = config.TOP_K_RESULTS):
        if self._bm25 is None or not self._corpus_ids:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            doc_id = self._corpus_ids[idx]
            doc = self.docstore.get(doc_id)
            results.append({"id": doc_id, "text": doc.get("text", "") if doc else "", "metadata": doc or {}, "score": float(scores[idx])})
        return results


class HybridSearcher:
    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self.contextual = ContextualRetriever(self.llm)
        self.bm25: BM25Index | None = None

    def _ensure_bm25(self):
        if self.bm25 is None:
            self.bm25 = BM25Index(self.contextual.docstore)

    def search(self, query: str, top_k: int = config.TOP_K_RESULTS, k: int = config.RRF_K):
        vector_results = self.contextual.vector_search(query, top_k=top_k * 2)
        self._ensure_bm25()
        assert self.bm25 is not None
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        rrf_scores: dict[str, float] = {}
        doc_texts: dict[str, str] = {}
        doc_sources: dict[str, list[str]] = {}
        for rank, doc in enumerate(vector_results, start=1):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0) + 1.0 / (k + rank)
            doc_texts[doc["id"]] = doc["text"]
            doc_sources.setdefault(doc["id"], []).append("vector")
        for rank, doc in enumerate(bm25_results, start=1):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0) + 1.0 / (k + rank)
            doc_texts[doc["id"]] = doc["text"]
            doc_sources.setdefault(doc["id"], []).append("bm25")
        sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
        fused = [{"id": did, "text": doc_texts.get(did, ""), "rrf_score": rrf_scores[did], "sources": doc_sources.get(did, [])} for did in sorted_ids[:top_k]]
        return {"results": fused, "vector_results": vector_results[:top_k], "bm25_results": bm25_results[:top_k]}

    def get_context(self, query: str, top_k: int = config.TOP_K_RESULTS) -> str:
        result = self.search(query, top_k=top_k)
        return "\n\n---\n\n".join(r["text"] for r in result["results"] if r["text"])
