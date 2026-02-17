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

logger = logging.getLogger(__name__)


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

    def _extract_triples(self, text: str) -> list[dict[str, str]]:
        text_slice = (text or "").strip()[:3000]
        if not text_slice or len(text_slice) < 50:
            logger.warning("Graph entity extraction skipped: chunk too short (%d chars)", len(text_slice))
            return []
        prompt = (
            "Identify all key entities and their relationships in the following text. "
            "Return a JSON object with a key \"triples\" containing a list of objects, "
            "each with keys \"entity1\", \"relationship\", \"entity2\". "
            "Extract at least 2-5 triples if the text has meaningful content.\n\n"
            f"Text:\n{text_slice}"
        )
        result = self.llm.generate_json(prompt, options=config.REASONING_OPTIONS)
        if "raw" in result:
            logger.warning(
                "Graph entity extraction: LLM returned invalid JSON (parse failed). "
                "Response preview: %s",
                (result.get("raw", "") or "")[:300],
            )
            return []
        triples = result.get("triples", [])
        if not isinstance(triples, list):
            logger.warning(
                "Graph entity extraction: \"triples\" is not a list (type=%s). Keys: %s",
                type(triples).__name__,
                list(result.keys()),
            )
            return []
        normalized = []
        for t in triples:
            nt = self._normalize_triple(t)
            if nt:
                normalized.append(nt)
        if not normalized and triples:
            logger.warning(
                "Graph entity extraction: %d raw triples rejected (missing entity1/relationship/entity2 or subject/predicate/object). Keys in first: %s",
                len(triples),
                list(triples[0].keys()) if triples else [],
            )
        elif not normalized:
            logger.warning(
                "Graph entity extraction: LLM returned 0 triples for chunk (%d chars). Response keys: %s",
                len(text_slice),
                list(result.keys()),
            )
        return normalized

    def _add_triples_to_graph(self, triples, source_chunk):
        for triple in triples:
            e1 = triple["entity1"].strip().title()
            e2 = triple["entity2"].strip().title()
            rel = triple["relationship"].strip().lower()
            if not self.graph.has_node(e1):
                self.graph.add_node(e1, type="entity", mentions=0)
            self.graph.nodes[e1]["mentions"] = self.graph.nodes[e1].get("mentions", 0) + 1
            if not self.graph.has_node(e2):
                self.graph.add_node(e2, type="entity", mentions=0)
            self.graph.nodes[e2]["mentions"] = self.graph.nodes[e2].get("mentions", 0) + 1
            if self.graph.has_edge(e1, e2):
                existing = self.graph[e1][e2].get("relationships", [])
                if rel not in existing:
                    existing.append(rel)
                self.graph[e1][e2]["relationships"] = existing
                self.graph[e1][e2]["weight"] = self.graph[e1][e2].get("weight", 1) + 1
            else:
                self.graph.add_edge(e1, e2, relationships=[rel], weight=1, source_chunks=[source_chunk])

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
                summary = self.llm.generate(prompt, options=config.REASONING_OPTIONS)
                self.community_summaries[cid] = summary.strip()
            except Exception as exc:
                self.community_summaries[cid] = f"Cluster: {', '.join(nodes)}"
            if callback:
                callback(f"Summarized community {cid} ({len(nodes)} entities)")

    def ingest(self, text: str, doc_name: str = "document", callback=None) -> dict[str, Any]:
        words = text.split()
        chunks = [" ".join(words[i:i + 800]) for i in range(0, len(words), 800)]
        if callback:
            callback(f"Extracting entities from {len(chunks)} chunks...")
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
        matched = []
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
            edges = [f"  {u} --[{', '.join(d.get('relationships', ['related_to']))}]--> {v}" for u, v, d in ego.edges(data=True)]
            ctx_parts.append(f"Entity: {entity}\nConnections:\n" + "\n".join(edges))
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
        prompt = (
            "Classify as 'local' or 'global'.\n"
            "- local: specific entities/relationships\n- global: themes/trends/analysis\n\n"
            f"Question: {query}\n\nReturn JSON: {{\"mode\": \"local\"}} or {{\"mode\": \"global\"}}"
        )
        return self.llm.generate_json(prompt, options=config.REASONING_OPTIONS).get("mode", "local")
