"""
Knowledge Base: Global Entity Taxonomy & Relational Fact Registry

Maintains a cross-document, deduplicated registry of all entities and their
relationships extracted from the document corpus. This supplements the
per-document PageIndex and Graph stores with a global view that enables:

1. Taxonomy lookups  – instant "which manufacturers exist?" queries
2. Disambiguation    – detect ambiguous user queries before retrieval
3. Metadata filters  – constrain vector search to verified entities
4. Fact verification – check if a claimed entity actually exists
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from app import config
from app.modules.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

TAXONOMY_PATH = config.KNOWLEDGE_BASE_DIR / "taxonomy.json"
RELATIONS_PATH = config.KNOWLEDGE_BASE_DIR / "relations.json"

# Entity-type inference from relationship verbs.
# If entity1 --[relationship]--> entity2, these patterns classify entity1's role.
_SUBJECT_ROLE_PATTERNS: dict[str, list[str]] = {
    "manufacturer": ["manufactures", "produces", "makes", "built by", "made by", "supplied by", "fabricated by"],
    "vessel": ["carries", "transports", "sails", "operates", "navigates"],
    "power_supply": ["powers", "supplies power", "provides power", "energizes", "feeds"],
    "component": ["contains", "includes", "has", "comprises", "consists of", "equipped with"],
    "system": ["controls", "manages", "monitors", "regulates", "operates"],
    "specification": ["rated at", "rated for", "has capacity", "has rating"],
}
_OBJECT_ROLE_PATTERNS: dict[str, list[str]] = {
    "manufacturer": ["manufactured by", "produced by", "made by", "supplied by", "built by", "from"],
    "component": ["installed in", "part of", "mounted on", "connected to", "attached to"],
    "vessel": ["installed on", "aboard", "on board", "deployed on"],
    "specification": ["has specification", "has rating", "rated"],
}


def _infer_type_from_relationships(entity: str, rels: list[dict]) -> str | None:
    """Attempt to infer entity type from relationship verbs without LLM."""
    for rel in rels:
        rel_text = (rel.get("relationship") or "").lower()
        is_subject = rel.get("role") == "subject"
        patterns = _SUBJECT_ROLE_PATTERNS if is_subject else _OBJECT_ROLE_PATTERNS
        for entity_type, verbs in patterns.items():
            if any(v in rel_text for v in verbs):
                return entity_type
    return None


class KnowledgeBase:
    """
    Global entity taxonomy and relational fact registry.

    The taxonomy tracks every unique entity name across all documents, classified
    by type (manufacturer, vessel, component, etc.) with source provenance.

    The relations index stores flattened subject-relationship-object triples
    for fast cross-document lookups.
    """

    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm
        self._taxonomy: dict[str, dict[str, Any]] = {}
        self._by_type: dict[str, list[str]] = defaultdict(list)
        self._relations: list[dict[str, Any]] = []
        self._meta: dict[str, Any] = {"last_updated": None, "document_count": 0, "entity_count": 0}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        logger.debug("KB _load: starting (taxonomy_path=%s, relations_path=%s)", TAXONOMY_PATH, RELATIONS_PATH)
        if TAXONOMY_PATH.exists():
            try:
                with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._taxonomy = data.get("entities", {})
                self._by_type = defaultdict(list, data.get("by_type", {}))
                self._meta = data.get("meta", self._meta)
                logger.info(
                    "KB _load: taxonomy loaded — entities=%d, types=%s, last_updated=%s",
                    len(self._taxonomy),
                    list(self._by_type.keys()),
                    self._meta.get("last_updated"),
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("KB _load: failed to load taxonomy: %s", e)
        else:
            logger.debug("KB _load: no taxonomy file at %s", TAXONOMY_PATH)

        if RELATIONS_PATH.exists():
            try:
                with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._relations = data.get("facts", [])
                logger.info("KB _load: relations loaded — facts=%d", len(self._relations))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("KB _load: failed to load relations: %s", e)
        else:
            logger.debug("KB _load: no relations file at %s", RELATIONS_PATH)

    def _save(self) -> None:
        config.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
        self._meta["entity_count"] = len(self._taxonomy)
        self._meta["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.debug("KB _save: writing taxonomy (%d entities, %d relations) to %s", len(self._taxonomy), len(self._relations), TAXONOMY_PATH)

        with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "entities": self._taxonomy,
                "by_type": dict(self._by_type),
                "meta": self._meta,
            }, f, indent=2, ensure_ascii=False)

        with open(RELATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump({"facts": self._relations}, f, indent=2, ensure_ascii=False)
        logger.debug("KB _save: done")

    # ── Ingestion: update from a single document's graph ─────────────────

    def update_from_graph(self, doc_name: str, graph: nx.DiGraph, callback: Any = None) -> dict[str, int]:
        """
        Merge entities and relationships from a per-document graph into the
        global taxonomy. Called after graph ingestion completes for a document.
        Returns stats dict.
        """
        logger.info(
            "KB update_from_graph: START doc_name=%s graph_nodes=%d graph_edges=%d taxonomy_before=%d entities",
            doc_name, graph.number_of_nodes(), graph.number_of_edges(), len(self._taxonomy),
        )
        new_entities = 0
        new_relations = 0
        updated_entities = 0
        inferred_count = 0

        for node in graph.nodes():
            node_data = graph.nodes[node]
            entity_name = str(node).strip()
            if not entity_name:
                continue

            desc = (node_data.get("entity_description") or "").strip()
            mentions = node_data.get("mentions", 1)

            # Collect relationships for type inference
            rels_for_inference = []
            for successor in graph.successors(node):
                edge = graph[node][successor]
                for r in edge.get("relationships", []):
                    rels_for_inference.append({"relationship": r, "role": "subject", "target": str(successor)})
            for predecessor in graph.predecessors(node):
                edge = graph[predecessor][node]
                for r in edge.get("relationships", []):
                    rels_for_inference.append({"relationship": r, "role": "object", "target": str(predecessor)})

            inferred_type = _infer_type_from_relationships(entity_name, rels_for_inference)
            if inferred_type:
                inferred_count += 1

            if entity_name in self._taxonomy:
                entry = self._taxonomy[entity_name]
                if doc_name not in entry["documents"]:
                    entry["documents"].append(doc_name)
                entry["mentions"] += mentions
                if desc and len(desc) > len(entry.get("description") or ""):
                    entry["description"] = desc[:2000]
                if inferred_type and inferred_type not in entry.get("types", []):
                    entry.setdefault("types", []).append(inferred_type)
                    if entity_name not in self._by_type.get(inferred_type, []):
                        self._by_type[inferred_type].append(entity_name)
                updated_entities += 1
            else:
                types_list = [inferred_type] if inferred_type else []
                self._taxonomy[entity_name] = {
                    "types": types_list,
                    "description": desc[:2000] if desc else "",
                    "mentions": mentions,
                    "documents": [doc_name],
                }
                if inferred_type:
                    self._by_type[inferred_type].append(entity_name)
                new_entities += 1

        # Extract relational facts
        for u, v, edge_data in graph.edges(data=True):
            u_str, v_str = str(u).strip(), str(v).strip()
            rels = edge_data.get("relationships", [])
            rel_desc = (edge_data.get("relationship_description") or "").strip()

            for rel in rels:
                existing = any(
                    f["subject"] == u_str and f["object"] == v_str and f["relationship"] == rel
                    for f in self._relations
                )
                if not existing:
                    self._relations.append({
                        "subject": u_str,
                        "relationship": rel,
                        "object": v_str,
                        "description": rel_desc[:500] if rel_desc else "",
                        "documents": [doc_name],
                    })
                    new_relations += 1
                else:
                    for f in self._relations:
                        if f["subject"] == u_str and f["object"] == v_str and f["relationship"] == rel:
                            if doc_name not in f["documents"]:
                                f["documents"].append(doc_name)
                            break

        doc_names = set()
        for entry in self._taxonomy.values():
            doc_names.update(entry.get("documents", []))
        self._meta["document_count"] = len(doc_names)

        self._save()

        stats = {"new_entities": new_entities, "updated_entities": updated_entities, "new_relations": new_relations}
        logger.info(
            "KB update_from_graph: DONE doc_name=%s new_entities=%d updated_entities=%d new_relations=%d inferred_types=%d taxonomy_total=%d",
            doc_name, new_entities, updated_entities, new_relations, inferred_count, len(self._taxonomy),
        )
        if callback:
            callback(f"Knowledge base updated: {stats}")
        return stats

    # ── Full rebuild from all document graphs ────────────────────────────

    def refresh_all(self, callback: Any = None, model: str | None = None) -> dict[str, Any]:
        """
        Rebuild the entire knowledge base from all per-document graphs.
        Used for periodic refresh or on-demand from frontend.
        model: optional Ollama model name for entity classification; default is INTERNAL_REASONING_AGENT (e.g. phi3:3.8b).
        """
        effective_model = (model.strip() if model and model.strip() else getattr(self.llm, "model", None)) or "unknown"
        logger.info("KB refresh_all: START — clearing taxonomy and relations; classification model=%s", effective_model)
        self._taxonomy = {}
        self._by_type = defaultdict(list)
        self._relations = []
        self._meta = {"last_updated": None, "document_count": 0, "entity_count": 0}

        graph_files = list(config.GRAPH_DIR.glob("*_graph.json")) if config.GRAPH_DIR.exists() else []
        logger.info("KB refresh_all: found %d graph files in %s", len(graph_files), config.GRAPH_DIR)
        if not graph_files:
            self._save()
            logger.info("KB refresh_all: DONE (no graphs) documents_processed=0 total_entities=0 total_relations=0")
            return {"documents_processed": 0, "total_entities": 0, "total_relations": 0}

        total_new = 0
        total_updated = 0
        total_rels = 0

        for gf in graph_files:
            doc_name = gf.stem.replace("_graph", "")
            try:
                logger.debug("KB refresh_all: processing graph %s", gf.name)
                with open(gf, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)
                G = nx.node_link_graph(graph_data)
                if callback:
                    callback(f"Processing graph for {doc_name} ({G.number_of_nodes()} entities)...")
                stats = self.update_from_graph(doc_name, G, callback=None)
                total_new += stats["new_entities"]
                total_updated += stats["updated_entities"]
                total_rels += stats["new_relations"]
                logger.debug("KB refresh_all: processed %s -> %s", doc_name, stats)
            except Exception as e:
                logger.warning("KB refresh_all: failed to process graph %s: %s", gf.name, e)
                if callback:
                    callback(f"Warning: failed to process {doc_name}: {e}")

        # Classify untyped entities if we have an LLM (use override model or default KB client)
        untyped = [name for name, data in self._taxonomy.items() if not data.get("types")]
        classify_llm = (OllamaClient(model=model) if (model and model.strip()) else self.llm)
        classify_model_name = getattr(classify_llm, "model", "unknown") if classify_llm else "none"
        logger.info("KB refresh_all: untyped entities=%d; classifying with model=%s", len(untyped), classify_model_name)
        if untyped and classify_llm:
            if callback:
                callback(f"Classifying {len(untyped)} untyped entities with model {classify_model_name}...")
            self._classify_entities_batch(untyped, classify_llm=classify_llm)

        self._save()

        result = {
            "documents_processed": len(graph_files),
            "total_entities": len(self._taxonomy),
            "total_relations": len(self._relations),
            "newly_classified": len(untyped),
        }
        logger.info(
            "KB refresh_all: DONE documents_processed=%d total_entities=%d total_relations=%d newly_classified=%d",
            result["documents_processed"], result["total_entities"], result["total_relations"], result["newly_classified"],
        )
        if callback:
            callback(f"Knowledge base refresh complete: {result}")
        return result

    def _classify_entities_batch(self, entity_names: list[str], classify_llm: OllamaClient | None = None) -> None:
        """Use LLM to classify a batch of entities into types. classify_llm overrides self.llm when provided."""
        llm = classify_llm if classify_llm is not None else self.llm
        if not llm or not entity_names:
            logger.debug("KB _classify_entities_batch: skip (llm=%s, names_count=%d)", llm is not None, len(entity_names or []))
            return

        model_name = getattr(llm, "model", "unknown")
        batch_size = config.KB_ENTITY_CLASSIFICATION_BATCH_SIZE
        num_batches = (len(entity_names) + batch_size - 1) // batch_size
        logger.info("KB _classify_entities_batch: using model=%s; classifying %d entities in %d batches (batch_size=%d)", model_name, len(entity_names), num_batches, batch_size)
        for i in range(0, len(entity_names), batch_size):
            batch = entity_names[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.debug("KB _classify_entities_batch: batch %d/%d (%d entities): %s", batch_num, num_batches, len(batch), batch[:5] if len(batch) > 5 else batch)
            entities_with_context = []
            for name in batch:
                entry = self._taxonomy.get(name, {})
                desc = entry.get("description", "")
                context_rels = [
                    f for f in self._relations
                    if f["subject"] == name or f["object"] == name
                ][:3]
                rel_strs = [f"{r['subject']} --[{r['relationship']}]--> {r['object']}" for r in context_rels]
                entities_with_context.append(
                    f"- {name}: {desc[:200]}" + (f" (rels: {'; '.join(rel_strs)})" if rel_strs else "")
                )

            prompt = (
                "Classify each entity into ONE primary type from this list:\n"
                "manufacturer, vessel, component, power_supply, system, specification, location, person, organization, material, other\n\n"
                "Entities:\n" + "\n".join(entities_with_context) + "\n\n"
                "Return JSON: {\"classifications\": [{\"entity\": \"...\", \"type\": \"...\"}]}"
            )
            schema = {
                "type": "object",
                "properties": {
                    "classifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity": {"type": "string"},
                                "type": {"type": "string"},
                            },
                            "required": ["entity", "type"],
                        },
                    }
                },
                "required": ["classifications"],
            }
            try:
                result = llm.generate_with_schema(prompt, schema, options=config.REASONING_OPTIONS)
                classifications = result.get("classifications", [])
                for c in classifications:
                    ent_name = c.get("entity", "").strip()
                    ent_type = c.get("type", "other").strip().lower()
                    if ent_name in self._taxonomy:
                        if ent_type not in self._taxonomy[ent_name].get("types", []):
                            self._taxonomy[ent_name].setdefault("types", []).append(ent_type)
                    if ent_name not in self._by_type.get(ent_type, []):
                        self._by_type[ent_type].append(ent_name)
                logger.debug("KB _classify_entities_batch: batch %d applied %d classifications", batch_num, len(classifications))
            except Exception as e:
                logger.warning("KB _classify_entities_batch: batch %d failed: %s", batch_num, e)

    # ── Removal: clean up when a document is deleted ─────────────────────

    def remove_document(self, doc_name: str) -> None:
        """Remove all contributions from a specific document."""
        logger.info("KB remove_document: START doc_name=%s taxonomy_size=%d relations_count=%d", doc_name, len(self._taxonomy), len(self._relations))
        to_remove = []
        for entity_name, entry in self._taxonomy.items():
            if doc_name in entry.get("documents", []):
                entry["documents"].remove(doc_name)
                if not entry["documents"]:
                    to_remove.append(entity_name)

        relations_before = len(self._relations)
        for name in to_remove:
            removed_types = self._taxonomy[name].get("types", [])
            del self._taxonomy[name]
            for t in removed_types:
                if t in self._by_type and name in self._by_type[t]:
                    self._by_type[t].remove(name)

        self._relations = [
            f for f in self._relations
            if not (len(f.get("documents", [])) == 1 and f["documents"][0] == doc_name)
        ]
        for f in self._relations:
            if doc_name in f.get("documents", []):
                f["documents"].remove(doc_name)

        relations_removed = relations_before - len(self._relations)
        self._save()
        logger.info(
            "KB remove_document: DONE doc_name=%s entities_removed=%d relations_removed=%d taxonomy_size=%d",
            doc_name, len(to_remove), relations_removed, len(self._taxonomy),
        )

    # ── Query: lookups for the disambiguation layer ──────────────────────

    def lookup_entity(self, name: str) -> dict[str, Any] | None:
        """Exact or case-insensitive lookup of an entity."""
        if name in self._taxonomy:
            return {"name": name, **self._taxonomy[name]}
        name_lower = name.lower()
        for ent, data in self._taxonomy.items():
            if ent.lower() == name_lower:
                return {"name": ent, **data}
        return None

    def search_entities(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Fuzzy substring search across all entity names."""
        query_lower = query.lower()
        matches = []
        for ent, data in self._taxonomy.items():
            if query_lower in ent.lower():
                matches.append({"name": ent, **data})
        matches.sort(key=lambda x: x.get("mentions", 0), reverse=True)
        return matches[:limit]

    def get_entities_by_type(self, entity_type: str) -> list[str]:
        """Get all entity names of a given type."""
        return list(self._by_type.get(entity_type, []))

    def get_all_types(self) -> dict[str, int]:
        """Return all entity types with their counts."""
        return {t: len(entities) for t, entities in self._by_type.items() if entities}

    def get_related_entities(self, entity_name: str) -> list[dict[str, Any]]:
        """Get all entities related to a given entity."""
        related = []
        for f in self._relations:
            if f["subject"] == entity_name:
                related.append({"entity": f["object"], "relationship": f["relationship"], "direction": "outgoing"})
            elif f["object"] == entity_name:
                related.append({"entity": f["subject"], "relationship": f["relationship"], "direction": "incoming"})
        return related

    def get_facts_for_entity(self, entity_name: str) -> list[dict[str, Any]]:
        """Get all relational facts involving an entity."""
        return [
            f for f in self._relations
            if f["subject"] == entity_name or f["object"] == entity_name
        ]

    # ── Disambiguation: the core pre-retrieval check ─────────────────────

    def disambiguate(self, query: str) -> dict[str, Any] | None:
        """
        Analyze a query against the taxonomy to detect ambiguity.

        Returns None if the query is sufficiently specific.
        Returns a disambiguation dict if the query references an entity type
        or partial name that maps to multiple options.

        The disambiguation dict contains:
        - ambiguous_types: entity types where multiple options exist
        - options: for each ambiguous type, the list of valid entity names
        - suggested_constraint: text the user could add to disambiguate
        """
        logger.debug("KB disambiguate: query=%r taxonomy_size=%d llm=%s", query, len(self._taxonomy), self.llm is not None)
        if not self._taxonomy or not self.llm:
            logger.debug("KB disambiguate: skip (empty taxonomy or no LLM)")
            return None

        # Build a summary of available entity types and counts
        type_summary = {t: entities[:15] for t, entities in self._by_type.items() if len(entities) > 1}
        if not type_summary:
            logger.debug("KB disambiguate: no multi-option types in taxonomy")
            return None
        logger.debug("KB disambiguate: type_summary keys=%s", list(type_summary.keys()))

        type_listing = "\n".join(
            f"- {t} ({len(ents)} options): {', '.join(ents[:8])}"
            + (f" ... and {len(ents) - 8} more" if len(ents) > 8 else "")
            for t, ents in type_summary.items()
        )

        schema = {
            "type": "object",
            "properties": {
                "is_ambiguous": {"type": "boolean"},
                "ambiguous_types": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "mentioned_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reason": {"type": "string"},
            },
            "required": ["is_ambiguous"],
        }

        prompt = (
            "You are checking whether a user query is specific enough to retrieve a single answer, "
            "or whether it is ambiguous because the document corpus contains multiple items that could match.\n\n"
            f"User query: \"{query}\"\n\n"
            "Known entity types in the corpus (with examples):\n"
            f"{type_listing}\n\n"
            "Rules:\n"
            "- If the query mentions a generic category (e.g. 'the battery bank', 'the power supply', 'the motor') "
            "AND multiple specific entities of that type exist, mark as ambiguous.\n"
            "- If the query already names a specific entity that exists in the taxonomy, it is NOT ambiguous.\n"
            "- If the query is about a general topic (e.g. 'how does X work?') that doesn't require "
            "picking a specific entity, it is NOT ambiguous.\n"
            "- If only ONE entity of the referenced type exists, it is NOT ambiguous.\n\n"
            "Return JSON with is_ambiguous (bool), ambiguous_types (list of type names), "
            "mentioned_entities (entities from the query that exist in taxonomy), "
            "and reason (one sentence)."
        )

        try:
            result = self.llm.generate_with_schema(prompt, schema, options=config.REASONING_OPTIONS)
        except Exception as e:
            logger.warning("KB disambiguate: LLM call failed: %s", e)
            return None

        if isinstance(result, dict) and "raw" in result:
            logger.debug("KB disambiguate: LLM returned unparseable JSON")
            return None

        is_ambiguous = result.get("is_ambiguous", False)
        logger.info("KB disambiguate: query=%r is_ambiguous=%s ambiguous_types=%s", query, is_ambiguous, result.get("ambiguous_types", []))

        if not is_ambiguous:
            logger.debug("KB disambiguate: query deemed specific, returning None")
            return None

        ambiguous_types = result.get("ambiguous_types", [])
        options: dict[str, list[str]] = {}
        for t in ambiguous_types:
            t_lower = t.lower().strip()
            entities = self._by_type.get(t_lower, [])
            if len(entities) > 1:
                options[t_lower] = entities[:20]

        if not options:
            logger.debug("KB disambiguate: ambiguous_types had no multi-option match in by_type")
            return None

        logger.info("KB disambiguate: AMBIGUOUS options=%s reason=%s", {k: len(v) for k, v in options.items()}, (result.get("reason") or "")[:80])
        return {
            "is_ambiguous": True,
            "ambiguous_types": list(options.keys()),
            "options": options,
            "mentioned_entities": result.get("mentioned_entities", []),
            "reason": result.get("reason", ""),
        }

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        logger.debug("KB get_stats: entity_count=%d relation_count=%d", len(self._taxonomy), len(self._relations))
        return {
            "entity_count": len(self._taxonomy),
            "relation_count": len(self._relations),
            "document_count": self._meta.get("document_count", 0),
            "last_updated": self._meta.get("last_updated"),
            "types": self.get_all_types(),
            "top_entities": sorted(
                [{"name": k, "mentions": v.get("mentions", 0), "types": v.get("types", [])}
                 for k, v in self._taxonomy.items()],
                key=lambda x: x["mentions"], reverse=True,
            )[:20],
        }


# Singleton instance
_kb_instance: KnowledgeBase | None = None


def get_knowledge_base(llm: OllamaClient | None = None, model: str | None = None) -> KnowledgeBase:
    """
    Get the global KnowledgeBase. Uses INTERNAL_REASONING_AGENT (e.g. phi3:3.8b) by default
    so KB operations (classification, disambiguation) do not consume the heavy LLM.
    llm: optional pre-built client (overrides model if both given).
    model: optional model name; used only when llm is None (default: config.INTERNAL_REASONING_AGENT).
    """
    global _kb_instance
    if _kb_instance is None:
        if llm is not None:
            client = llm
        elif model and model.strip():
            client = OllamaClient(model=model.strip())
        else:
            client = OllamaClient(model=config.INTERNAL_REASONING_AGENT)
        _kb_instance = KnowledgeBase(llm=client)
    elif llm is not None and _kb_instance.llm is None:
        _kb_instance.llm = llm
    return _kb_instance
