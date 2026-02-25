"""
Module III: Graph RAG – Relational and Thematic Layer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from app import config
from app.modules.ollama_client import OllamaClient, AsyncOllamaClient
from app.modules.chunking import chunk_text_semantic, is_likely_markdown

logger = logging.getLogger(__name__)

_chroma_graph_client = None
_chroma_graph_collection = None


def _get_chroma_graph_collection():
    global _chroma_graph_client, _chroma_graph_collection
    if _chroma_graph_collection is None:
        import chromadb
        config.CHROMA_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_graph_client = chromadb.PersistentClient(path=str(config.CHROMA_GRAPH_DIR))
        _chroma_graph_collection = _chroma_graph_client.get_or_create_collection(
            name="graph_descriptions", metadata={"hnsw:space": "cosine"},
        )
    return _chroma_graph_collection

# JSON Schema for deterministic extraction (Ollama format parameter).
TRIPLES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity1": {"type": "string"},
                    "relationship": {"type": "string"},
                    "entity2": {"type": "string"},
                },
                "required": ["entity1", "relationship", "entity2"],
            },
        },
    },
    "required": ["triples"],
}

# LightRAG-style: expansive descriptions for semantic routing and hybrid k-NN.
TRIPLES_EXTENDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity1": {"type": "string"},
                    "entity2": {"type": "string"},
                    "relationship": {"type": "string"},
                    "entity1_description": {"type": "string"},
                    "entity2_description": {"type": "string"},
                    "relationship_description": {"type": "string"},
                    "relationship_keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity1", "relationship", "entity2"],
            },
        },
    },
    "required": ["triples"],
}


def _index_graph_descriptions(G: nx.Graph, doc_name: str, llm: OllamaClient) -> None:
    """Embed node and edge descriptions into auxiliary Chroma for hybrid k-NN semantic search."""
    collection = _get_chroma_graph_collection()
    # Remove existing entries for this doc
    try:
        existing = collection.get(where={"doc_name": doc_name}, include=[])
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception as e:
        logger.debug("Chroma graph delete by doc_name: %s", e)
    ids_list = []
    documents_list = []
    metadatas_list = []
    for node in G.nodes():
        desc = G.nodes[node].get("entity_description", "") or node
        if not desc.strip():
            desc = str(node)
        ids_list.append(f"node:{doc_name}:{node}")
        documents_list.append(desc[:4000])
        metadatas_list.append({"doc_name": doc_name, "type": "node", "node_id": node})
    for u, v in G.edges():
        rels = G[u][v].get("relationships", [])
        rel_desc = G[u][v].get("relationship_description", "") or " ".join(rels)
        kw = G[u][v].get("relationship_keywords", [])
        text = f"{rel_desc} " + " ".join(kw[:10])
        if not text.strip():
            text = " ".join(rels)
        ids_list.append(f"edge:{doc_name}:{u}|{v}")
        documents_list.append(text[:4000])
        metadatas_list.append({"doc_name": doc_name, "type": "edge", "u": u, "v": v})
    if not ids_list:
        return
    try:
        embeddings = llm.embed(documents_list)
        if embeddings and len(embeddings) == len(ids_list):
            collection.upsert(ids=ids_list, embeddings=embeddings, documents=documents_list, metadatas=metadatas_list)
            logger.info("Indexed %d graph descriptions for %s", len(ids_list), doc_name)
    except Exception as e:
        logger.warning("Failed to index graph descriptions: %s", e)


def _detect_communities(G: nx.Graph) -> dict[str, int]:
    try:
        import community as community_louvain
        return community_louvain.best_partition(G.to_undirected(), resolution=config.GRAPH_COMMUNITY_RESOLUTION)
    except ImportError:
        partition = {}
        for i, comp in enumerate(nx.connected_components(G.to_undirected())):
            for node in comp:
                partition[node] = i
        return partition


class GraphBuilder:
    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self.async_llm = AsyncOllamaClient()
        self.graph = nx.DiGraph()
        self.communities: dict[str, int] = {}
        self.community_summaries: dict[int, str] = {}

    def _normalize_triple(self, t: dict) -> dict[str, str] | None:
        """Accept entity1/relationship/entity2 or subject/predicate/object."""
        if not isinstance(t, dict):
            return None
        if "entity1" in t and "relationship" in t and "entity2" in t:
            return {"entity1": str(t["entity1"]), "relationship": str(t["relationship"]), "entity2": str(t["entity2"])}
        if "subject" in t and "predicate" in t and "object" in t:
            return {"entity1": str(t["subject"]), "relationship": str(t["predicate"]), "entity2": str(t["object"])}
        return None

    def _extraction_options(self) -> dict[str, Any]:
        """Options for extraction: large num_ctx to prevent truncation; optional smaller model."""
        opts = {
            "temperature": 0.0,
            "num_ctx": config.GRAPH_EXTRACTION_NUM_CTX,
        }
        return opts

    def _extract_triples(self, text: str) -> list[dict[str, Any]]:
        text_slice = (text or "").strip()[:4000]
        if not text_slice or len(text_slice) < 50:
            logger.warning("Graph entity extraction skipped: chunk too short (%d chars)", len(text_slice))
            return []
        prompt = (
            "Identify all key entities and their relationships in the following text. "
            "Return ONLY a valid JSON object (no explanation, no markdown, no thinking). "
            "The JSON must have a key \"triples\" containing an array of objects, "
            "each with keys: \"entity1\", \"relationship\", \"entity2\", "
            "\"entity1_description\", \"entity2_description\", "
            "\"relationship_description\" (one sentence each), and \"relationship_keywords\" (list of 2-5 keywords). "
            "Extract at least 2-5 triples if the text has meaningful content.\n\n"
            "Example output format:\n"
            "{\"triples\": [{\"entity1\": \"Engine\", \"relationship\": \"powers\", \"entity2\": \"Ship\", "
            "\"entity1_description\": \"Main propulsion unit\", \"entity2_description\": \"Maritime vessel\", "
            "\"relationship_description\": \"The engine provides propulsion power to the ship\", "
            "\"relationship_keywords\": [\"propulsion\", \"power\", \"engine\"]}]}\n\n"
            f"Text:\n{text_slice}"
        )
        model = config.GRAPH_INGESTION_MODEL or None
        options = self._extraction_options()
        result = None
        # Try schema-constrained generation first (not all models support this).
        try:
            if hasattr(self.llm, "generate_with_schema"):
                result = self.llm.generate_with_schema(
                    prompt,
                    TRIPLES_EXTENDED_JSON_SCHEMA,
                    model=model or self.llm.model,
                    system="You are a JSON extraction assistant. Output ONLY valid JSON. No thinking, no explanation, no markdown fences.",
                    options=options,
                )
        except Exception as e:
            logger.debug("Schema-constrained extraction failed, falling back to JSON prompt: %s", e)
            result = None
        # If strict/extended schema fails, retry with a smaller schema.
        if result is None or (isinstance(result, dict) and "raw" in result):
            try:
                result = self.llm.generate_with_schema(
                    (
                        "Extract entity-relation triples from the text. "
                        "Return ONLY JSON: {\"triples\": [{\"entity1\": \"...\", \"relationship\": \"...\", \"entity2\": \"...\"}]}\n\n"
                        f"Text:\n{text_slice}"
                    ),
                    TRIPLES_JSON_SCHEMA,
                    model=model or self.llm.model,
                    system="You are a JSON extraction assistant. Output ONLY valid JSON. No explanation.",
                    options=options,
                )
            except Exception as e:
                logger.debug("Basic schema extraction failed: %s", e)
                result = None
        # Fallback: check if result is missing, empty, or unparseable ("raw" key means parse failure).
        if result is None or (isinstance(result, dict) and "raw" in result):
            if result and "raw" in result:
                logger.debug("Schema extraction returned unusable result, falling back to generate_json")
            result = self.llm.generate_json(
                prompt,
                model=model,
                system="You are a JSON extraction assistant. Output ONLY valid JSON. No thinking, no explanation, no markdown fences.",
                options=options,
            )
        # Last-resort fallback: simpler prompt with minimal required fields only.
        if isinstance(result, dict) and "raw" in result:
            result = self.llm.generate_json(
                (
                    "Extract factual triples from the text. Return ONLY this JSON shape:\n"
                    "{\"triples\": [{\"entity1\": \"...\", \"relationship\": \"...\", \"entity2\": \"...\"}]}\n"
                    "If no triples are present, return {\"triples\": []}.\n\n"
                    f"Text:\n{text_slice}"
                ),
                model=model,
                system="Output only valid JSON.",
                options=options,
            )
        if isinstance(result, list):
            triples = result
        elif isinstance(result, dict):
            if "raw" in result:
                logger.warning(
                    "Graph entity extraction: LLM returned invalid JSON. Response preview: %.300s",
                    result.get("raw", ""),
                )
                return []
            triples = result.get("triples", [])
        else:
            triples = []
        if not isinstance(triples, list):
            triples = []
        normalized = []
        for t in triples:
            nt = self._normalize_triple(t)
            if nt:
                # Preserve LightRAG-style fields if present
                nt["entity1_description"] = (t.get("entity1_description") or "").strip() or None
                nt["entity2_description"] = (t.get("entity2_description") or "").strip() or None
                nt["relationship_description"] = (t.get("relationship_description") or "").strip() or None
                rk = t.get("relationship_keywords")
                nt["relationship_keywords"] = list(rk) if isinstance(rk, list) else []
                normalized.append(nt)
        return normalized

    def _add_triples_to_graph(self, triples, source_chunk):
        for triple in triples:
            e1 = triple["entity1"].strip().title()
            e2 = triple["entity2"].strip().title()
            rel = triple["relationship"].strip().lower()
            if not self.graph.has_node(e1):
                self.graph.add_node(e1, type="entity", mentions=0, entity_description="")
            self.graph.nodes[e1]["mentions"] = self.graph.nodes[e1].get("mentions", 0) + 1
            desc1 = triple.get("entity1_description") or ""
            if desc1 and (not self.graph.nodes[e1].get("entity_description")):
                self.graph.nodes[e1]["entity_description"] = desc1
            elif desc1:
                self.graph.nodes[e1]["entity_description"] = (
                    self.graph.nodes[e1].get("entity_description", "") + " " + desc1
                ).strip()[:2000]
            if not self.graph.has_node(e2):
                self.graph.add_node(e2, type="entity", mentions=0, entity_description="")
            self.graph.nodes[e2]["mentions"] = self.graph.nodes[e2].get("mentions", 0) + 1
            desc2 = triple.get("entity2_description") or ""
            if desc2 and (not self.graph.nodes[e2].get("entity_description")):
                self.graph.nodes[e2]["entity_description"] = desc2
            elif desc2:
                self.graph.nodes[e2]["entity_description"] = (
                    self.graph.nodes[e2].get("entity_description", "") + " " + desc2
                ).strip()[:2000]
            keywords = triple.get("relationship_keywords") or []
            rel_desc = (triple.get("relationship_description") or "").strip()
            if self.graph.has_edge(e1, e2):
                existing = self.graph[e1][e2].get("relationships", [])
                if rel not in existing:
                    existing.append(rel)
                self.graph[e1][e2]["relationships"] = existing
                self.graph[e1][e2]["weight"] = self.graph[e1][e2].get("weight", 1) + 1
                if rel_desc:
                    self.graph[e1][e2]["relationship_description"] = (
                        self.graph[e1][e2].get("relationship_description", "") + " " + rel_desc
                    ).strip()[:1500]
                self.graph[e1][e2]["relationship_keywords"] = list(
                    set(self.graph[e1][e2].get("relationship_keywords", []) + keywords)
                )[:30]
            else:
                self.graph.add_edge(
                    e1, e2,
                    relationships=[rel], weight=1, source_chunks=[source_chunk],
                    relationship_description=rel_desc[:1500] if rel_desc else "",
                    relationship_keywords=keywords[:30],
                )

    def _summarize_communities(self, callback=None):
        community_nodes: dict[int, list[str]] = {}
        for node, cid in self.communities.items():
            community_nodes.setdefault(cid, []).append(node)
        for cid, nodes in community_nodes.items():
            if len(nodes) < config.GRAPH_MIN_COMMUNITY_SIZE:
                continue
            edges_desc = []
            for n in nodes:
                for nb in self.graph.successors(n):
                    if nb in nodes:
                        rels = self.graph[n][nb].get("relationships", [])
                        edges_desc.append(f"{n} --[{', '.join(rels)}]--> {nb}")
                for nb in self.graph.predecessors(n):
                    if nb in nodes:
                        rels = self.graph[nb][n].get("relationships", [])
                        edges_desc.append(f"{nb} --[{', '.join(rels)}]--> {n}")
            prompt = (
                "Summarize the following cluster of entities and relationships "
                f"in 2-3 sentences.\n\nEntities: {', '.join(nodes)}\n\n"
                f"Relationships:\n{chr(10).join(set(edges_desc))[:2000]}"
            )
            try:
                summary = self.llm.generate(
                    prompt,
                    model=config.GRAPH_INGESTION_MODEL,
                    options=config.REASONING_OPTIONS,
                )
                self.community_summaries[cid] = summary.strip()
            except Exception as exc:
                self.community_summaries[cid] = f"Cluster: {', '.join(nodes)}"
            if callback:
                callback(f"Summarized community {cid} ({len(nodes)} entities)")

    def ingest(self, text: str, doc_name: str = "document", callback=None) -> dict[str, Any]:
        # Semantic chunking for conceptual continuity (structure-aware for Markdown).
        if config.USE_SEMANTIC_CHUNKING and text.strip():
            if is_likely_markdown(text):
                chunk_list = chunk_text_semantic(text)
            else:
                from app.modules.chunking import semantic_chunk_plain
                chunk_list = semantic_chunk_plain(
                    text,
                    max_words=config.SEMANTIC_CHUNK_MAX_WORDS,
                    min_words=config.SEMANTIC_CHUNK_MIN_WORDS,
                )
        else:
            words = text.split()
            chunk_list = [{"text": " ".join(words[i:i + config.SEMANTIC_CHUNK_MAX_WORDS]), "index": i} for i in range(0, len(words), config.SEMANTIC_CHUNK_MAX_WORDS)]
        chunks = [c["text"] if isinstance(c, dict) else c for c in chunk_list]
        if callback:
            callback(f"Extracting entities from {len(chunks)} chunks (structure-aware)...")
        total_triples = 0
        for i, chunk in enumerate(chunks):
            triples = self._extract_triples(chunk)
            self._add_triples_to_graph(triples, i)
            total_triples += len(triples)
            if callback and (i + 1) % 3 == 0:
                callback(f"Processed {i + 1}/{len(chunks)} chunks ({self.graph.number_of_nodes()} entities)")
        if callback:
            callback("Detecting communities...")
        if self.graph.number_of_nodes() > 0:
            self.communities = _detect_communities(self.graph)
            if callback:
                callback("Generating community summaries...")
            self._summarize_communities(callback)
        self._save(doc_name)
        if callback:
            callback("Indexing graph descriptions for hybrid search...")
        _index_graph_descriptions(self.graph, doc_name, self.llm)
        stats = {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges(), "communities": len(set(self.communities.values())), "triples_extracted": total_triples}
        if callback:
            callback(f"Graph built: {stats}")
        return stats

    def _save(self, doc_name):
        config.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        base = config.GRAPH_DIR / doc_name
        with open(f"{base}_graph.json", "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(self.graph), f, indent=2, ensure_ascii=False)
        with open(f"{base}_communities.json", "w", encoding="utf-8") as f:
            json.dump({"partition": self.communities, "summaries": {str(k): v for k, v in self.community_summaries.items()}}, f, indent=2, ensure_ascii=False)

    def load(self, doc_name: str) -> bool:
        base = config.GRAPH_DIR / doc_name
        gp = Path(f"{base}_graph.json")
        if not gp.exists():
            return False
        with open(gp, "r", encoding="utf-8") as f:
            self.graph = nx.node_link_graph(json.load(f))
        # Ensure LightRAG-style attributes exist for older graphs
        for n in self.graph.nodes():
            if "entity_description" not in self.graph.nodes[n]:
                self.graph.nodes[n]["entity_description"] = ""
        for u, v in self.graph.edges():
            if "relationship_description" not in self.graph[u][v]:
                self.graph[u][v]["relationship_description"] = ""
            if "relationship_keywords" not in self.graph[u][v]:
                self.graph[u][v]["relationship_keywords"] = []
        cp = Path(f"{base}_communities.json")
        if cp.exists():
            with open(cp, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.communities = data.get("partition", {})
                self.community_summaries = {int(k): v for k, v in data.get("summaries", {}).items()}
        return True


class GraphSearcher:
    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self.builder = GraphBuilder(self.llm)

    def list_graphs(self) -> list[str]:
        if not config.GRAPH_DIR.exists():
            return []
        return list(set(p.stem.replace("_graph", "").replace("_communities", "") for p in config.GRAPH_DIR.glob("*.json")))

    def load_graph(self, doc_name: str) -> bool:
        return self.builder.load(doc_name)

    def local_search(self, query: str, doc_name: str | None = None):
        if doc_name:
            self.load_graph(doc_name)
        elif not self.builder.graph.number_of_nodes():
            graphs = self.list_graphs()
            if graphs:
                self.load_graph(graphs[0])
        G = self.builder.graph
        if G.number_of_nodes() == 0:
            return {"context": "", "entities": [], "mode": "local"}
        # Hybrid: k-NN on graph descriptions (node/edge embeddings) then traverse.
        matched = []
        try:
            coll = _get_chroma_graph_collection()
            effective_doc = doc_name or (self.list_graphs()[:1] or [None])[0]
            if effective_doc:
                query_emb = self.llm.embed(query)
                if query_emb:
                    n_results = 15
                    if effective_doc:
                        try:
                            n_results = min(15, getattr(coll, "count", lambda: 999)())
                        except Exception:
                            pass
                    res = coll.query(
                        query_embeddings=query_emb,
                        n_results=n_results,
                        where={"doc_name": effective_doc} if effective_doc else None,
                        include=["metadatas", "documents"],
                    )
                    for meta in (res.get("metadatas") or [[]])[0]:
                        if not meta:
                            continue
                        if meta.get("type") == "node":
                            n = meta.get("node_id")
                            if n and n not in matched:
                                matched.append(n)
                        elif meta.get("type") == "edge":
                            u, v = meta.get("u"), meta.get("v")
                            if u and u not in matched:
                                matched.append(u)
                            if v and v not in matched:
                                matched.append(v)
        except Exception as e:
            logger.debug("Graph k-NN search failed, falling back to entity matching: %s", e)
        if not matched:
            all_entities = list(G.nodes())[:100]
            prompt = (
                f"Given the user question and entities from a knowledge graph, identify the most relevant.\n\n"
                f"Question: {query}\n\nEntities: {', '.join(all_entities)}\n\n"
                "Return JSON: {\"entities\": [...]}"
            )
            result = self.llm.generate_json(prompt, options=config.REASONING_OPTIONS)
            targets = result.get("entities", [])
            if not isinstance(targets, list):
                targets = [targets]
            el = {e.lower(): e for e in G.nodes()}
            for e in targets:
                e_low = str(e).lower().strip()
                if e_low in el:
                    matched.append(el[e_low])
            if not matched:
                for e in targets:
                    for ge in G.nodes():
                        if str(e).lower() in ge.lower():
                            matched.append(ge)
                            break
        ctx_parts = []
        for entity in matched[:5]:
            ego = nx.ego_graph(G, entity, radius=2, undirected=True)
            edges = []
            for u, v, d in ego.edges(data=True):
                rels = d.get("relationships", ["related_to"])
                desc = d.get("relationship_description", "")
                edges.append(f"  {u} --[{', '.join(rels)}]--> {v}" + (f" ({desc[:100]})" if desc else ""))
            node_desc = G.nodes[entity].get("entity_description", "")
            ctx_parts.append(
                f"Entity: {entity}" + (f"\nDescription: {node_desc[:300]}" if node_desc else "")
                + "\nConnections:\n" + "\n".join(edges)
            )
        return {"context": "\n\n".join(ctx_parts), "entities": matched, "mode": "local"}

    def global_search(self, query: str, doc_name: str | None = None):
        if doc_name:
            self.load_graph(doc_name)
        elif not self.builder.graph.number_of_nodes():
            graphs = self.list_graphs()
            if graphs:
                self.load_graph(graphs[0])
        summaries = self.builder.community_summaries
        if not summaries:
            return {"context": "", "communities": [], "mode": "global"}
        sl = "\n".join(f"  Community {cid}: {s}" for cid, s in summaries.items())
        prompt = f"Given the question and community summaries, identify relevant communities.\n\nQuestion: {query}\n\n{sl}\n\nReturn JSON: {{\"community_ids\": [...]}}"
        result = self.llm.generate_json(prompt, options=config.REASONING_OPTIONS)
        sel = result.get("community_ids", [])
        if not isinstance(sel, list):
            sel = list(summaries.keys())
        ctx = []
        for cid in sel:
            ci = int(cid) if not isinstance(cid, int) else cid
            if ci in summaries:
                ctx.append(f"[Community {ci}]: {summaries[ci]}")
        if not ctx:
            ctx = [f"[Community {cid}]: {s}" for cid, s in summaries.items()]
        return {"context": "\n\n".join(ctx), "communities": sel, "mode": "global"}

    def search(self, query: str, mode: str = "auto", doc_name: str | None = None):
        if mode == "auto":
            mode = self._classify_mode(query)
        return self.local_search(query, doc_name) if mode == "local" else self.global_search(query, doc_name)

    def _classify_mode(self, query: str) -> str:
        """Classify query as local (specific entities) or global (themes). On any failure, default to 'local' so retrieval never blocks."""
        try:
            prompt = (
                "Classify as 'local' or 'global'.\n"
                "- local: specific entities/relationships\n- global: themes/trends/analysis\n\n"
                f"Question: {query}\n\nReturn JSON: {{\"mode\": \"local\"}} or {{\"mode\": \"global\"}}"
            )
            result = self.llm.generate_json(prompt, options=config.REASONING_OPTIONS)
            mode = (result.get("mode") or "local").strip().lower()
            if mode in ("local", "global"):
                return mode
        except Exception as e:
            logger.warning("Graph _classify_mode failed, using local: %s", e)
        return "local"
