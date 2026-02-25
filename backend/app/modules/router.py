"""
The Orchestrator: Adaptive RAG with LangGraph
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import StateGraph, END

from app import config
from app.modules.ollama_client import OllamaClient
from app.modules.pageindex import TreeSearcher, create_node_mapping
from app.modules.contextual import HybridSearcher
from app.modules.graph import GraphSearcher
from app.modules.cag import CAGManager
from app.modules.knowledge_base import get_knowledge_base
from app.modules.pdf_images import get_stored_page_images, get_page_images_for_nodes
from app.modules.vlm_extract import generate_answer_with_images

logger = logging.getLogger(__name__)

StateDict = dict[str, Any]


class AdaptiveRAG:
    VALID_ROUTES = {"PAGEINDEX", "VECTOR", "GRAPH", "CAG"}
    FALLBACK_ORDER = ["VECTOR", "PAGEINDEX", "GRAPH"]
    # Context budgets from config (token-aware via RAG_CONTEXT_MAX_CHARS); PageIndex gets largest share.

    def __init__(self, llm: OllamaClient | None = None, cag_manager: CAGManager | None = None):
        self.llm = llm or OllamaClient()
        self.tree_searcher = TreeSearcher(self.llm)
        self.hybrid_searcher = HybridSearcher(self.llm)
        self.graph_searcher = GraphSearcher(self.llm)
        self.cag_manager = cag_manager or CAGManager(self.llm)
        self.knowledge_base = get_knowledge_base()
        self._graph = self._build_graph()

    @staticmethod
    def _append_confusion_reason(meta: dict[str, Any], reason: str) -> None:
        reasons = meta.get("confusion_reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        if reason not in reasons:
            reasons.append(reason)
        meta["confusion_reasons"] = reasons

    def _build_refinement_suggestions(
        self,
        query: str,
        context_excerpt: str = "",
        model: str | None = None,
    ) -> tuple[list[str], list[str], str]:
        """Run a mini vector search, extract real entities from the corpus,
        identify what specific info is missing, and build grounded suggestion
        queries.

        Returns (entities, suggestions, snippets_block).
        """

        # ── Step 1: Quick vector retrieval for real entities ──
        retrieved_snippets: list[str] = []
        try:
            search_results = self.hybrid_searcher.search(query, top_k=6)
            for r in search_results.get("results", []):
                text = (r.get("text") or "").strip()
                if text:
                    retrieved_snippets.append(text[:400])
        except Exception:
            pass

        snippets_block = "\n---\n".join(retrieved_snippets[:5]) if retrieved_snippets else context_excerpt[:1500]

        # ── Step 2: Extract concrete entities from retrieved chunks ──
        entity_schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        }
        entity_prompt = (
            "Extract specific named entities from the following document excerpts "
            "that are directly relevant to the user query.\n"
            "Return concrete names the user would recognize: company/brand names, "
            "model numbers, product/equipment names, rated values with units, part numbers.\n"
            "Do NOT return section headings or section numbers like '4.3'.\n"
            "Do NOT return generic terms or categories.\n\n"
            f"User query: {query}\n\n"
            f"Document excerpts:\n{snippets_block[:2000]}"
        )
        entities: list[str] = []
        try:
            ent_out = self.llm.generate_with_schema(
                entity_prompt, entity_schema, model=model, options=config.REASONING_OPTIONS,
            )
            raw_ents = ent_out.get("entities", [])
            if isinstance(raw_ents, list):
                entities = [str(e).strip() for e in raw_ents if str(e).strip()][:10]
        except Exception:
            pass

        entities_str = ", ".join(entities) if entities else "(none found)"

        # ── Step 3: Build refined queries grounded in real entities ──
        suggestion_schema = {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 4,
                }
            },
            "required": ["suggestions"],
            "additionalProperties": False,
        }
        suggestion_prompt = (
            "You are rewriting a vague user query into 3-4 specific, ready-to-submit query alternatives.\n\n"
            "CRITICAL RULES:\n"
            "- Each suggestion MUST be a complete query the user can send directly.\n"
            "- Each suggestion MUST incorporate at least one REAL entity name from the entity list below.\n"
            "- Do NOT use generic placeholders like '[specific model]' or '(section referenced in documents)'.\n"
            "- Prefer company/brand + model/product names over section references.\n"
            "- Do NOT mention section numbers/headings unless absolutely required by the query.\n"
            "- Keep the original intent of the user query intact.\n\n"
            f"User query: {query}\n\n"
            f"Real entities found in the documents: {entities_str}\n\n"
            f"Document excerpts for context:\n{snippets_block[:1500]}"
        )
        suggestions: list[str] = []
        try:
            out = self.llm.generate_with_schema(
                suggestion_prompt, suggestion_schema, model=model, options=config.REASONING_OPTIONS,
            )
            raw_qs = out.get("suggestions", [])
            if isinstance(raw_qs, list):
                qs = [str(q).strip() for q in raw_qs if str(q).strip()]
                deduped: list[str] = []
                for q in qs:
                    if q not in deduped:
                        deduped.append(q)
                if len(deduped) >= 2:
                    suggestions = deduped[:4]
        except Exception:
            pass

        # ── Fallback: use extracted entities directly in query rewrites ──
        if not suggestions:
            if entities:
                for ent in entities[:3]:
                    suggestions.append(f"{query} specifically for {ent}")
            if not suggestions:
                suggestions = [
                    f"{query} (include specific equipment name and rated value with units)",
                ]
            suggestions = suggestions[:4]

        return entities, suggestions, snippets_block

    def _clarifier_node(self, state: StateDict) -> StateDict:
        """Grounded clarification: run mini retrieval, identify what specific
        information is missing from the user query, and present entity-grounded
        suggestions so the user can narrow down their question."""
        query = state.get("original_query", "") or state.get("query", "")
        ctx = state.get("context", "")
        trace = state.get("trace", [])
        model = state.get("model")
        meta = state.get("metadata", {})
        trace.append("Clarifier: building grounded clarification...")
        logger.info("router.clarifier: ENTER query=%r", (query or "")[:80])

        # ── Try taxonomy-enriched clarification first ──
        kb_stats = self.knowledge_base.get_stats()
        taxonomy_entities: list[str] = []
        if kb_stats["entity_count"] > 0:
            kb_search = self.knowledge_base.search_entities(query, limit=15)
            taxonomy_entities = [e["name"] for e in kb_search]
            logger.info("router.clarifier: taxonomy search returned %d entities: %s", len(taxonomy_entities), taxonomy_entities[:10])
            if taxonomy_entities:
                trace.append(f"Clarifier: found {len(taxonomy_entities)} taxonomy matches")

        # ── Reuse _build_refinement_suggestions for entities + suggestions ──
        entities, suggestions, snippets_block = self._build_refinement_suggestions(
            query=query,
            context_excerpt=ctx,
            model=model,
        )
        logger.info("router.clarifier: refinement_suggestions entities=%d suggestions=%d", len(entities), len(suggestions))

        # Merge taxonomy entities with vector-retrieved entities (taxonomy first, deduplicated)
        before_merge = len(entities)
        if taxonomy_entities:
            existing_lower = {e.lower() for e in entities}
            for te in taxonomy_entities:
                if te.lower() not in existing_lower:
                    entities.append(te)
                    existing_lower.add(te.lower())
            logger.info("router.clarifier: merged taxonomy entities: before=%d after=%d added=%d", before_merge, len(entities), len(entities) - before_merge)

        # ── LLM "missing info" classifier ──
        entities_str = ", ".join(entities) if entities else "(none found)"
        missing_info_schema = {
            "type": "object",
            "properties": {
                "missing_info": {
                    "type": "string",
                    "description": "One sentence describing what specific identifier or detail the user needs to provide.",
                }
            },
            "required": ["missing_info"],
            "additionalProperties": False,
        }
        missing_info_prompt = (
            "A user asked a question but the retrieved documents contain multiple matching items "
            "and the query is too vague to pick one.\n\n"
            f"User query: {query}\n\n"
            f"Entities found in the documents: {entities_str}\n\n"
            f"Document excerpts:\n{snippets_block[:1500]}\n\n"
            "In ONE concise sentence, describe what specific identifier or detail the user needs "
            "to provide so the system can give a precise answer. For example:\n"
            '- "The documents mention motors from Siemens, ABB, and WEG. Please specify which manufacturer."\n'
            '- "There are specs for Model A-100 and Model B-200. Please indicate which model you mean."\n'
            '- "Multiple rated voltages are listed (400V, 690V). Please clarify which voltage rating."\n\n'
            'Return JSON: {"missing_info": "..."}'
        )
        missing_info_desc = ""
        try:
            mi_out = self.llm.generate_with_schema(
                missing_info_prompt, missing_info_schema, model=model, options=config.REASONING_OPTIONS,
            )
            missing_info_desc = str(mi_out.get("missing_info", "")).strip()
        except Exception:
            pass

        if not missing_info_desc:
            if entities:
                missing_info_desc = (
                    f"Your documents mention several items ({', '.join(entities[:5])}). "
                    "Could you specify which one you are asking about?"
                )
            else:
                missing_info_desc = (
                    "Your question could match multiple items in the documents. "
                    "Could you add more specific details (e.g. manufacturer, model number, or part name)?"
                )

        # ── Set clarification metadata only; do NOT set answer. Generator will synthesize from
        # retrieved context first, then append "is this what you mean?" so results are always shown.
        meta["needs_clarification"] = True
        meta["clarification_questions"] = suggestions
        meta["missing_info_description"] = missing_info_desc
        trace.append(f"Clarifier: identified missing info — {missing_info_desc[:120]}")
        trace.append(f"Clarifier: offering {len(suggestions)} grounded suggestions (answer will be from context)")

        logger.info("router.clarifier: EXIT suggestions=%d missing_info=%s", len(suggestions), (missing_info_desc or "")[:60])
        return {**state, "trace": trace, "metadata": meta}

    def _taxonomy_disambiguator_node(self, state: StateDict) -> StateDict:
        """
        Pre-retrieval disambiguation using the global knowledge base taxonomy.

        Before any expensive retrieval (beam search, vector search, graph traversal),
        this node checks whether the query is ambiguous by consulting the entity
        taxonomy. If the taxonomy contains multiple entities of a type referenced
        in the query (e.g. multiple manufacturers when the user just says "battery
        bank"), it short-circuits to a taxonomy-grounded clarification—faster and
        more reliable than post-retrieval disambiguation.

        If the query is sufficiently specific, it passes through unchanged.
        """
        if not state.get("enable_taxonomy", True):
            trace = state.get("trace", [])
            trace.append("Taxonomy: disabled by options, skipping")
            logger.info("router.taxonomy_check: EXIT skip (disabled)")
            return {**state, "trace": trace}

        query = state.get("query", "")
        trace = state.get("trace", [])
        meta = state.get("metadata", {})
        route = state.get("route", "")

        logger.info("router.taxonomy_check: ENTER query=%r route=%s", query[:80] if query else "", route)

        if route == "CAG":
            logger.info("router.taxonomy_check: skip (CAG route)")
            return state

        kb_stats = self.knowledge_base.get_stats()
        logger.info(
            "router.taxonomy_check: KB stats entity_count=%d relation_count=%d document_count=%d",
            kb_stats["entity_count"], kb_stats["relation_count"], kb_stats.get("document_count", 0),
        )
        if kb_stats["entity_count"] == 0:
            trace.append("Taxonomy: empty knowledge base, skipping disambiguation")
            logger.info("router.taxonomy_check: EXIT proceed (empty KB)")
            return {**state, "trace": trace}

        trace.append(f"Taxonomy: checking query against {kb_stats['entity_count']} known entities...")

        try:
            disambiguation = self.knowledge_base.disambiguate(query)
        except Exception as e:
            logger.warning("router.taxonomy_check: disambiguate failed: %s", e)
            trace.append(f"Taxonomy: disambiguation error ({e}), proceeding to retrieval")
            return {**state, "trace": trace}

        if disambiguation is None:
            trace.append("Taxonomy: query is sufficiently specific")
            logger.info("router.taxonomy_check: EXIT proceed (query specific)")
            return {**state, "trace": trace}

        # The query IS ambiguous—store disambiguation in metadata and proceed to retrieval
        # so we always return a real answer first, then offer selectable options.
        ambiguous_types = disambiguation.get("ambiguous_types", [])
        options = disambiguation.get("options", {})
        reason = disambiguation.get("reason", "")

        trace.append(f"Taxonomy: AMBIGUOUS — {reason}, proceeding to retrieval then append clarification")
        for t, ents in options.items():
            trace.append(f"  {t}: {', '.join(ents[:10])}")

        # Build concrete query suggestions (refined queries) for "Show refined queries" only
        suggestions = []
        for t, ents in options.items():
            for ent in ents[:3]:
                suggestions.append(f"{query} — specifically for {ent}")
        suggestions = suggestions[:4]

        meta["needs_clarification"] = True
        meta["clarification_questions"] = suggestions
        meta["disambiguation_source"] = "taxonomy"
        meta["missing_info_description"] = reason
        meta["taxonomy_options"] = options

        logger.info(
            "router.taxonomy_check: EXIT proceed (ambiguous) ambiguous_types=%s options_count=%d",
            list(options.keys()), sum(len(e) for e in options.values()),
        )
        return {**state, "trace": trace, "metadata": meta}

    def _taxonomy_decision(self, state: StateDict) -> str:
        """Route after taxonomy check: when ambiguous we proceed to retrieval so we always answer first."""
        decision = "proceed"
        logger.info("router.taxonomy_decision: %s", decision)
        return decision

    def _build_graph(self):
        wf = StateGraph(state_schema=dict)
        wf.add_node("router", self._router_node)
        wf.add_node("taxonomy_check", self._taxonomy_disambiguator_node)
        wf.add_node("all_retrievers", self._all_retrievers_node)
        wf.add_node("cag_retriever", self._cag_node)
        wf.add_node("grader", self._grader_node)
        wf.add_node("rewriter", self._rewriter_node)
        wf.add_node("clarifier", self._clarifier_node)
        wf.add_node("generator", self._generator_node)
        wf.set_entry_point("router")

        # Router → taxonomy check (pre-retrieval disambiguation) or CAG
        wf.add_conditional_edges("router", self._router_next_node, {
            "cag_retriever": "cag_retriever",
            "all_retrievers": "taxonomy_check",
        })
        # Taxonomy check: when ambiguous we still proceed to retrievers so we answer first, then append clarification
        wf.add_conditional_edges("taxonomy_check", self._taxonomy_decision, {
            "generator": "generator",
            "proceed": "all_retrievers",
        })
        wf.add_edge("all_retrievers", "grader")
        wf.add_edge("cag_retriever", "generator")
        wf.add_conditional_edges("grader", self._grade_decision, {
            "relevant": "generator",
            "underspecified": "clarifier",
            "not_relevant": "rewriter",
            "fail": "clarifier",
        })
        wf.add_edge("rewriter", "router")
        wf.add_edge("clarifier", "generator")
        wf.add_edge("generator", END)
        return wf.compile()

    def _get_vector_grounding(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Run vector retrieval once and return reusable grounding context."""
        k = top_k or config.TOP_K_RESULTS
        try:
            result = self.hybrid_searcher.search(query, top_k=k)
            fused = result.get("results", []) or []
        except Exception:
            fused = []
        ctx = "\n\n---\n\n".join(r.get("text", "") for r in fused if r.get("text"))
        sources = [
            {
                "doc_name": r.get("doc_name"),
                "snippet": (r.get("text") or "")[:500],
                "chunk_id": r.get("id"),
                "page_number": r.get("page_number"),
            }
            for r in fused
            if r.get("doc_name")
        ]
        return {"context": ctx, "results": fused, "sources": sources}

    def _router_node(self, state: StateDict) -> StateDict:
        query = state.get("query", "")
        trace = state.get("trace", [])
        cag_active = self.cag_manager.is_active
        model = state.get("model")
        meta = state.get("metadata", {})

        # Always perform at least one vector retrieval pass for grounding.
        grounding = self._get_vector_grounding(query)
        grounding_ctx = grounding.get("context", "")
        grounding_sources = grounding.get("sources", [])
        grounding_count = len(grounding.get("results", []))
        trace.append(f"Vector grounding: found {grounding_count} results")
        if grounding_count > 0:
            meta["vector_results_count"] = grounding_count
            meta["vector_sources"] = grounding_sources
        state = {
            **state,
            "grounding_context": grounding_ctx,
            "grounding_vector_results": grounding.get("results", []),
            "metadata": meta,
        }

        # Compliance/accessibility queries: force GRAPH so regulatory knowledge (WCAG, HIPAA) is used.
        _compliance_keywords = ("compliance", "wcag", "accessibility", "hipaa", "violation", "edi", "regulatory", "contrast", "focus indicator")
        if any(kw in (query or "").lower() for kw in _compliance_keywords):
            trace.append("Router: compliance/accessibility query → forcing GRAPH (regulatory knowledge)")
            return {**state, "route": "GRAPH", "trace": trace, "metadata": meta}

        # Respect forced route or rewriter-chosen route: do NOT overwrite with LLM.
        existing_route = (state.get("route") or "").strip().upper()
        is_forced = state.get("forced_route") is True
        is_retry = (state.get("retries") or 0) > 0
        if existing_route and (is_forced or is_retry):
            if existing_route in self.VALID_ROUTES and (existing_route != "CAG" or cag_active):
                trace.append(f"Router: using chosen route {existing_route} (not overwriting)")
                return {**state, "route": existing_route, "trace": trace, "metadata": meta}
            # Invalid or CAG when disabled: fall through to classify

        prompt = (
            "Analyze the user query and classify it into exactly one category.\n\n"
            "Categories:\n- VECTOR: specific numbers, facts, definitions\n"
            "- PAGEINDEX: compare sections, document structure\n"
            "- GRAPH: high-level themes, relationships, trends\n"
            f"- CAG: {'follow-up to previous context' if cag_active else 'DISABLED'}\n\n"
            f"User query: \"{query}\"\n\nReturn JSON: {{\"route\": \"VECTOR\"}}"
        )
        schema = {
            "type": "object",
            "properties": {"route": {"type": "string", "enum": ["VECTOR", "PAGEINDEX", "GRAPH", "CAG"]}},
            "required": ["route"],
            "additionalProperties": False,
        }
        result = self.llm.generate_with_schema(prompt, schema, model=model, options=config.REASONING_OPTIONS)
        route = result.get("route", "VECTOR").upper().strip()
        if "raw" in result:
            self._append_confusion_reason(meta, "router_json_parse_failed")
        if route not in self.VALID_ROUTES:
            route = "VECTOR"
        if route == "CAG" and not cag_active:
            route = "VECTOR"
        trace.append(f"Router: classified as {route}")
        logger.info("router.router_node: query=%r -> route=%s", (query or "")[:60], route)
        return {**state, "route": route, "trace": trace, "metadata": meta}

    def _pageindex_node(self, state: StateDict) -> StateDict:
        query, trace = state.get("query", ""), state.get("trace", [])
        meta = state.get("metadata", {})
        # Official PageIndex-style: first list available documents with summaries, then search.
        available = self.tree_searcher.list_documents_with_summaries()
        meta["pageindex_available_docs"] = available
        if available:
            trace.append(f"PageIndex: {len(available)} document(s) available for analysis.")
            for d in available[:10]:
                name = d.get("doc_name", "")
                summary = (d.get("summary") or "")[:120]
                pages = d.get("page_count") or (d.get("end_index", 1) - d.get("start_index", 1) + 1)
                trace.append(f"  - {name} (pp. {d.get('start_index', 1)}–{d.get('end_index', 1)}): {summary}…")
        trace.append("PageIndex: starting tree traversal...")
        result = self.tree_searcher.search(query)
        for s in result.get("path", []):
            trace.append(f"  PageIndex: {s}")
        trace.append(f"PageIndex: found {len(result.get('leaf_nodes', []))} relevant sections")
        page_ctx = result.get("context", "")
        # Prepend short "available docs" block so the model can reference which doc(s) are in use.
        if available and page_ctx:
            avail_block = "Available documents: " + "; ".join(
                f"{d.get('doc_name', '')} (pp.{d.get('start_index', 1)}–{d.get('end_index', 1)})"
                for d in available[:8]
            )
            page_ctx = f"{avail_block}\n\n---\n\n{page_ctx}"
        grounding_ctx = state.get("grounding_context", "")
        if grounding_ctx.strip():
            page_ctx = (page_ctx + "\n\n--- Vector grounding ---\n\n" + grounding_ctx).strip()
        meta["pageindex_path"] = result.get("path", [])
        meta["pageindex_leaves"] = result.get("leaf_nodes", [])
        return {**state, "context": page_ctx, "trace": trace, "metadata": meta}

    def _vector_node(self, state: StateDict) -> StateDict:
        query, trace = state.get("query", ""), state.get("trace", [])
        trace.append("Vector: using hybrid grounding results...")
        fused = state.get("grounding_vector_results", []) or []
        ctx = state.get("grounding_context", "")
        if not fused:
            # Safety fallback if grounding was unavailable for this turn.
            trace.append("Vector: fallback hybrid search...")
            result = self.hybrid_searcher.search(query)
            fused = result.get("results", [])
            ctx = "\n\n---\n\n".join(r["text"] for r in fused if r["text"])
        trace.append(f"Vector: found {len(fused)} results")
        meta = state.get("metadata", {})
        meta["vector_results_count"] = len(fused)
        meta["vector_sources"] = [
            {
                "doc_name": r.get("doc_name"),
                "snippet": (r.get("text") or "")[:500],
                "chunk_id": r.get("id"),
                "page_number": r.get("page_number"),
            }
            for r in fused if r.get("doc_name")
        ]
        return {**state, "context": ctx, "trace": trace, "metadata": meta}

    def _graph_node(self, state: StateDict) -> StateDict:
        query, trace = state.get("query", ""), state.get("trace", [])
        trace.append("Graph: searching knowledge graph...")
        result = self.graph_searcher.search(query, mode="auto")
        trace.append(f"Graph: used {result.get('mode', 'unknown')} mode")
        graph_ctx = result.get("context", "")
        # Hybrid vector-graph: append vector grounding chunks.
        vector_ctx = state.get("grounding_context", "")
        if vector_ctx.strip():
            graph_ctx = (graph_ctx + "\n\n--- Document chunks (grounding) ---\n\n" + vector_ctx).strip()
        meta = state.get("metadata", {})
        meta["graph_mode"] = result.get("mode", "")
        return {**state, "context": graph_ctx, "trace": trace, "metadata": meta}

    def _all_retrievers_node(self, state: StateDict) -> StateDict:
        """Run VECTOR (grounding), PAGEINDEX, and GRAPH retrieval and merge with heavier weight to PageIndex. Respect enable_* flags."""
        query, trace = state.get("query", ""), state.get("trace", [])
        meta = state.get("metadata", {})
        enable_vector = state.get("enable_vector", True)
        enable_pageindex = state.get("enable_pageindex", True)
        enable_graph = state.get("enable_graph", True)

        vector_ctx = (state.get("grounding_context") or "").strip() if enable_vector else ""
        page_ctx = ""
        graph_ctx = ""

        logger.info("router.all_retrievers: ENTER query=%r vector=%s pageindex=%s graph=%s", (query or "")[:60], enable_vector, enable_pageindex, enable_graph)

        # 1. Vector (already in state from router)
        if enable_vector and vector_ctx:
            trace.append("AllRetrievers: using vector grounding")
        elif not enable_vector:
            trace.append("AllRetrievers: vector disabled")
        else:
            trace.append("AllRetrievers: no vector grounding")

        # 2. PageIndex (heavier weight: first in order, largest budget). List docs first, then search (official flow).
        if enable_pageindex:
            trace.append("AllRetrievers: running PageIndex...")
            try:
                available = self.tree_searcher.list_documents_with_summaries()
                meta["pageindex_available_docs"] = available
                if available:
                    trace.append(f"AllRetrievers: {len(available)} document(s) available; traversing for relevance.")
                pi_result = self.tree_searcher.search(query)
                for s in pi_result.get("path", []):
                    trace.append(f"  PageIndex: {s}")
                trace.append(f"AllRetrievers: PageIndex found {len(pi_result.get('leaf_nodes', []))} sections")
                page_ctx = (pi_result.get("context") or "").strip()
                if available and page_ctx:
                    avail_block = "Available documents: " + "; ".join(
                        f"{d.get('doc_name', '')} (pp.{d.get('start_index', 1)}–{d.get('end_index', 1)})"
                        for d in available[:8]
                    )
                    page_ctx = f"{avail_block}\n\n---\n\n{page_ctx}"
                meta["pageindex_path"] = pi_result.get("path", [])
                meta["pageindex_leaves"] = pi_result.get("leaf_nodes", [])
                # Reasoning-based retrieval (VLM) + page images for VLM visual answer (all docs with trees + stored images)
                if config.USE_VLM_VISUAL_ANSWER:
                    pageindex_retrieved_images: list[str] = []
                    pageindex_reasoning_results: list[dict[str, Any]] = []  # metadata: VLM reasoning per doc
                    available_docs = [d.get("doc_name") for d in (available or []) if d.get("doc_name")]
                    logger.info(
                        "router.all_retrievers: PageIndex reasoning (VLM) USE_VLM_VISUAL_ANSWER=True available_docs=%s",
                        available_docs,
                    )
                    meta["pageindex_reasoning_model"] = "VLM"
                    for doc_name in available_docs:
                        stored = get_stored_page_images(doc_name)
                        if not stored:
                            logger.debug("router.all_retrievers: doc_name=%s has no stored page images, skip reasoning", doc_name)
                            continue
                        total_pages = max(stored.keys()) if stored else 0
                        logger.info("router.all_retrievers: reasoning_retrieve (VLM) doc_name=%s total_pages=%d", doc_name, total_pages)
                        rr = self.tree_searcher.reasoning_retrieve(query, doc_name, total_pages=total_pages)
                        node_list = rr.get("node_list", [])
                        thinking = rr.get("thinking", "") or ""
                        pageindex_reasoning_results.append({
                            "doc_name": doc_name,
                            "node_list": node_list,
                            "thinking_excerpt": thinking[:500] if thinking else "",
                        })
                        if not node_list:
                            logger.debug("router.all_retrievers: doc_name=%s reasoning returned empty node_list", doc_name)
                            continue
                        tree_data = self.tree_searcher.load_tree_raw(doc_name)
                        if not tree_data:
                            continue
                        node_map = create_node_mapping(tree_data, max_page=total_pages)
                        paths = get_page_images_for_nodes(node_list, node_map, stored)
                        pageindex_retrieved_images.extend(paths)
                        logger.info(
                            "router.all_retrievers: doc_name=%s node_list=%s -> %d image path(s)",
                            doc_name, node_list, len(paths),
                        )
                        if thinking:
                            trace.append(f"PageIndex reasoning (VLM): {thinking[:200]}…")
                    meta["pageindex_reasoning_results"] = pageindex_reasoning_results
                    if pageindex_retrieved_images:
                        meta["pageindex_retrieved_images"] = pageindex_retrieved_images
                        meta["pageindex_vlm_visual_page_count"] = len(pageindex_retrieved_images)
                        trace.append(f"AllRetrievers: {len(pageindex_retrieved_images)} page image(s) for VLM answer")
                        logger.info(
                            "router.all_retrievers: pageindex_retrieved_images total=%d path_sample=%s",
                            len(pageindex_retrieved_images),
                            pageindex_retrieved_images[:2] if len(pageindex_retrieved_images) > 2 else pageindex_retrieved_images,
                        )
                    else:
                        logger.info("router.all_retrievers: no page images collected for VLM (no stored images or empty node_list)")
            except Exception as e:
                trace.append(f"AllRetrievers: PageIndex error — {e}")
                meta["pageindex_path"] = []
                meta["pageindex_leaves"] = []
                meta["pageindex_available_docs"] = []
                # Do not clear pageindex_retrieved_images — they may have been set before the exception
        else:
            trace.append("AllRetrievers: PageIndex disabled")
            meta["pageindex_path"] = []
            meta["pageindex_leaves"] = []

        # 3. Graph
        if enable_graph:
            trace.append("AllRetrievers: running Graph...")
            try:
                graph_result = self.graph_searcher.search(query, mode="auto")
                trace.append(f"AllRetrievers: Graph used {graph_result.get('mode', 'unknown')} mode")
                graph_ctx = (graph_result.get("context") or "").strip()
                meta["graph_mode"] = graph_result.get("mode", "")
            except Exception as e:
                trace.append(f"AllRetrievers: Graph error — {e}")
                meta["graph_mode"] = ""
        else:
            trace.append("AllRetrievers: Graph disabled")
            meta["graph_mode"] = ""

        # Merge with PageIndex first and largest share (budgets from config, scale with model context).
        total = config.RAG_CONTEXT_MAX_CHARS
        cap_pi = int(total * config.RAG_CONTEXT_SHARE_PAGEINDEX)
        cap_vec = int(total * config.RAG_CONTEXT_SHARE_VECTOR)
        cap_graph = int(total * config.RAG_CONTEXT_SHARE_GRAPH)
        parts = []
        if page_ctx:
            parts.append(f"--- PageIndex (primary) ---\n{page_ctx[:cap_pi]}")
        if vector_ctx:
            parts.append(f"--- Vector ---\n{vector_ctx[:cap_vec]}")
        if graph_ctx:
            parts.append(f"--- Graph ---\n{graph_ctx[:cap_graph]}")
        merged = "\n\n---\n\n".join(parts).strip() if parts else ""

        trace.append(f"AllRetrievers: merged with PageIndex primary (max={total} chars, PI={cap_pi}, V={cap_vec}, G={cap_graph})")
        logger.info("router.all_retrievers: EXIT merged_ctx_len=%d", len(merged))
        return {**state, "context": merged, "trace": trace, "metadata": meta}

    def _cag_node(self, state: StateDict) -> StateDict:
        query, trace = state.get("query", ""), state.get("trace", [])
        model = state.get("model")
        trace.append(f"CAG: using cached context for '{self.cag_manager.loaded_document}'")
        result = self.cag_manager.chat(query, model=model)
        trace.append(f"CAG: response generated (turn {result.get('conversation_turn', 0)})")
        return {**state, "answer": result.get("response", ""), "context": "[CAG]", "trace": trace}

    # Minimum context length (chars): if grader LLM fails or is inconclusive, we still proceed to generator.
    GRADER_FALLBACK_MIN_CTX = 2000

    def _grader_node(self, state: StateDict) -> StateDict:
        query, ctx, trace = state.get("query", ""), state.get("context", ""), state.get("trace", [])
        model = state.get("model")
        meta = state.get("metadata", {})
        trace.append("Grader: evaluating relevance...")
        if not ctx.strip():
            trace.append("Grader: NO context -> not_relevant")
            return {**state, "grader_verdict": "not_relevant", "trace": trace}

        prompt = (
            "You are evaluating whether retrieved context can answer a user query.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{ctx[:3000]}\n\n"
            "Classify into exactly ONE category:\n"
            '- "relevant": The context contains enough specific information to fully answer the question.\n'
            '- "underspecified": The context contains related information but the question is too vague '
            "to give a definitive answer. For example: the user asks about \"the motor\" but the context "
            "mentions multiple motors from different manufacturers; or the user asks for \"the rated voltage\" "
            "but the context has specs for several different models. The information IS present but the user "
            "needs to narrow down which specific item they mean.\n"
            '- "not_relevant": The context does not contain information related to the question at all.\n\n'
            'Return JSON: {"verdict": "relevant"} or {"verdict": "underspecified"} or {"verdict": "not_relevant"}'
        )
        schema = {
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["relevant", "underspecified", "not_relevant"]}},
            "required": ["verdict"],
            "additionalProperties": False,
        }

        verdict = "not_relevant"
        grader_ok = False
        try:
            result = self.llm.generate_with_schema(prompt, schema, model=model, options=config.REASONING_OPTIONS)
            if "raw" in result:
                self._append_confusion_reason(meta, "grader_json_parse_failed")
            else:
                v = result.get("verdict", "not_relevant")
                if isinstance(v, str):
                    v = v.strip().lower()
                if v in ("relevant", "underspecified", "not_relevant"):
                    verdict = v
                    grader_ok = True
        except Exception as e:
            logger.warning("router.grader: LLM call failed: %s", e)
            trace.append(f"Grader: LLM error — {e}")

        # If grader failed or returned inconclusive but we have substantial context, proceed to generator anyway.
        if not grader_ok and len(ctx) >= self.GRADER_FALLBACK_MIN_CTX:
            verdict = "relevant"
            trace.append(f"Grader: LLM inconclusive or failed but context has {len(ctx)} chars — proceeding to generator")
            logger.info("router.grader: fallback to generator (ctx_len=%d)", len(ctx))

        if grader_ok:
            trace.append(f"Grader: {verdict.upper()}")
        logger.info("router.grader: verdict=%s ctx_len=%d", verdict, len(ctx))
        return {**state, "grader_verdict": verdict, "trace": trace, "metadata": meta}

    def _rewriter_node(self, state: StateDict) -> StateDict:
        """Rewrite for wording/route mismatch only. Never invents identifiers."""
        query = state.get("query", "")
        original = state.get("original_query", query)
        retries = state.get("retries", 0)
        route = state.get("route", "VECTOR")
        trace = state.get("trace", [])
        model = state.get("model")
        meta = state.get("metadata", {})
        prompt = (
            "The retrieved context was NOT relevant to the user query. "
            "This is likely a wording or retrieval-route mismatch, NOT a case of missing user detail.\n\n"
            "Rewrite the query to improve retrieval. You may:\n"
            "- Use synonyms or alternative phrasings\n"
            "- Add technical terms that might appear in documents\n"
            "- Simplify overly complex phrasing\n\n"
            "You MUST NOT:\n"
            "- Add specific identifiers (manufacturer names, model numbers, part numbers) "
            "that the user did not mention\n"
            "- Guess at details the user did not provide\n"
            "- Change the fundamental intent of the query\n\n"
            f"Original query: {original}\n"
            f"Failed query: {query}\n\n"
            "Return JSON: {\"rewritten_query\": \"...\"}"
        )
        schema = {
            "type": "object",
            "properties": {"rewritten_query": {"type": "string"}},
            "required": ["rewritten_query"],
            "additionalProperties": False,
        }
        result = self.llm.generate_with_schema(prompt, schema, model=model, options=config.REASONING_OPTIONS)
        rewritten = result.get("rewritten_query", query)
        if "raw" in result:
            self._append_confusion_reason(meta, "rewriter_json_parse_failed")
        if not isinstance(rewritten, str) or not rewritten.strip() or rewritten.strip().lower() == query.strip().lower():
            rewritten = query
            self._append_confusion_reason(meta, "rewriter_no_improvement")
        alts = [r for r in self.FALLBACK_ORDER if r != route]
        new_route = alts[retries % len(alts)] if alts else "VECTOR"
        trace.append(f"Rewriter: '{query}' -> '{rewritten}' (trying {new_route})")
        return {**state, "query": rewritten, "route": new_route, "retries": retries + 1, "trace": trace, "metadata": meta}

    @staticmethod
    def _answer_indicates_context_gap(answer: str) -> bool:
        """True if the model's answer indicates the context does not provide the specific information requested."""
        if not answer or len(answer) < 20:
            return False
        lower = answer.lower()
        gap_phrases = (
            "does not provide",
            "does not state",
            "does not include",
            "does not specify",
            "does not contain",
            "not provided",
            "not specified",
            "not stated",
            "no information",
            "no specific",
            "context does not",
            "the context does not",
            "the text does not",
            "the documents do not",
            "while the text mentions",
            "although the",
            "however, the context",
        )
        return any(p in lower for p in gap_phrases)

    @staticmethod
    def _parse_vlm_visual_response(raw: str) -> tuple[str, str]:
        """
        Parse VLM response that was asked to show its work: extract 'Supporting context from the pages'
        and 'Answer' sections so we can save the supporting text in metadata.
        Returns (supporting_text, answer). Either may be empty if not found.
        """
        if not raw or not raw.strip():
            return "", ""
        text = raw.strip()
        supporting_text = ""
        answer = ""
        # Find "Answer:" (case-insensitive) and take everything after it as the answer
        answer_heading = re.search(r"\bAnswer\s*:\s*", text, re.IGNORECASE)
        if answer_heading:
            answer = text[answer_heading.end() :].strip()
        # Find "Supporting context from the pages:" and take everything after it until "Answer:" (or end) as supporting text
        support_heading = re.search(r"Supporting\s+context\s+from\s+the\s+pages\s*:\s*", text, re.IGNORECASE)
        if support_heading:
            if answer_heading and answer_heading.start() > support_heading.end():
                supporting_text = text[support_heading.end() : answer_heading.start()].strip()
            else:
                supporting_text = text[support_heading.end() :].strip()
        if not answer:
            answer = text
        return supporting_text, answer

    def _generate_context_gap_alternatives(
        self, query: str, context_excerpt: str, answer: str, model: str | None
    ) -> list[str]:
        """Suggest 2–4 alternative questions that the context might answer (for 'Did you mean?' UI)."""
        schema = {
            "type": "object",
            "properties": {
                "alternatives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 4,
                }
            },
            "required": ["alternatives"],
            "additionalProperties": False,
        }
        prompt = (
            "The user asked a question but the assistant replied that the retrieved context does not contain "
            "the specific information requested.\n\n"
            f"User question: {query}\n\n"
            f"Assistant reply (excerpt): {answer[:400]}\n\n"
            "Based on the context excerpt below, suggest 2 to 4 short, concrete alternative questions that "
            "the context DOES or might answer (e.g. weight of a related component, similar specs, mounting requirements). "
            "Each alternative must be a complete question the user can ask. Keep each under 15 words.\n\n"
            f"Context excerpt:\n{context_excerpt[:2500]}\n\n"
            "Return JSON: {\"alternatives\": [\"question 1\", \"question 2\", ...]}"
        )
        try:
            out = self.llm.generate_with_schema(
                prompt, schema, model=model, options=config.REASONING_OPTIONS
            )
            raw = out.get("alternatives", [])
            if isinstance(raw, list):
                return [str(q).strip() for q in raw if str(q).strip()][:4]
        except Exception as e:
            logger.warning("router.context_gap_alternatives: LLM failed: %s", e)
        return []

    def _generator_node(self, state: StateDict) -> StateDict:
        """Pure answer synthesizer. Clarification is handled by the clarifier node."""
        query = state.get("original_query", "") or state.get("query", "")
        ctx, trace = state.get("context", ""), state.get("trace", [])
        model = state.get("model")
        meta = state.get("metadata", {})
        # If an earlier node (CAG) already produced an answer, pass through. Clarifier no longer
        # sets answer; it only sets metadata so we synthesize from context then append clarification.
        if state.get("answer", ""):
            trace.append("Generator: using CAG answer")
            return {**state, "trace": trace}
        trace.append("Generator: synthesizing answer...")

        # VLM visual answer: use retrieved PDF page images when PageIndex reasoning retrieval provided them
        pageindex_images = meta.get("pageindex_retrieved_images") or []
        if config.USE_VLM_VISUAL_ANSWER and pageindex_images:
            logger.info(
                "router.generator: using VLM visual answer path image_count=%d path_sample=%s",
                len(pageindex_images),
                pageindex_images[:2] if len(pageindex_images) > 2 else pageindex_images,
            )
            # Ask VLM to show its work: output supporting text from the pages then the answer, so we can save the text for metadata/citations
            answer_prompt = (
                "Answer the question based on the images of the document pages as context.\n\n"
                f"Question: {query}\n\n"
                "You must show your work so the supporting text is saved:\n"
                "1. First, under the heading 'Supporting context from the pages:', write the relevant text or excerpts from the document pages that you used (transcribe or summarize what you read in the images that supports your answer).\n"
                "2. Then, under the heading 'Answer:', give a clear, concise answer based only on that context.\n\n"
                "Use exactly these two headings so the system can save both the supporting context and the answer."
            )
            raw_vlm_response = generate_answer_with_images(answer_prompt, pageindex_images)
            if raw_vlm_response and raw_vlm_response.strip():
                raw_vlm_response = raw_vlm_response.strip()
                supporting_text, answer = _parse_vlm_visual_response(raw_vlm_response)
                if supporting_text:
                    meta["pageindex_vlm_supporting_text"] = supporting_text
                    trace.append(f"Generator: saved VLM supporting text ({len(supporting_text)} chars)")
                if not answer:
                    answer = raw_vlm_response
                answer = answer.strip()
                trace.append(f"Generator: used VLM with {len(pageindex_images)} retrieved page image(s)")
                logger.info("router.generator: VLM visual answer produced answer_len=%d supporting_len=%d", len(answer), len(supporting_text))
                meta["vlm_visual_answer"] = True
                meta["vlm_visual_page_count"] = len(pageindex_images)
                meta["pageindex_answer_generated_with"] = "vlm_visual"
                meta["pageindex_answer_context"] = "Generated answer using VLM with retrieved PDF page images as visual context"
                # Context-gap and clarification logic (same as below)
                if meta.get("needs_clarification"):
                    if meta.get("taxonomy_options"):
                        answer += "\n\nIf you'd like a more specific answer, you can pick one of the options below."
                    elif meta.get("clarification_questions") or meta.get("missing_info_description"):
                        answer += "\n\nYour question could refer to more than one item in the documents. If you'd like a more specific answer, pick one of the refined queries below, edit one, or rephrase with more detail."
                trace.append("Generator: done")
                return {**state, "answer": answer, "trace": trace, "metadata": meta}
            else:
                logger.info("router.generator: VLM visual path returned empty answer, falling back to text context")
                meta["vlm_visual_answer"] = False
                meta["vlm_visual_page_count"] = 0
                meta["vlm_visual_reason"] = "VLM returned empty answer"

        # Text-context path: set metadata so UI can show "VLM visual not used"
        if "vlm_visual_answer" not in meta:
            meta["vlm_visual_answer"] = False
            meta["vlm_visual_page_count"] = 0
            meta["vlm_visual_reason"] = "no page images retrieved" if not pageindex_images else meta.get("vlm_visual_reason", "unknown")
        logger.debug("router.generator: using text context path ctx_len=%d", len(ctx))
        if ctx.strip():
            ctx_limit = config.RAG_CONTEXT_MAX_CHARS
            prompt = f"Answer using ONLY the context.\n\nQuestion: {query}\n\nContext:\n{ctx[:ctx_limit]}\n\nProvide a clear answer."
        else:
            prompt = f"No context found. Answer from knowledge.\n\nQuestion: {query}"
        answer = self.llm.generate(prompt, model=model, options=config.GENERATION_OPTIONS)
        answer = answer.strip()
        # When the answer indicates context doesn't have the specific info, suggest alternative questions (tap to re-search).
        if ctx.strip() and self._answer_indicates_context_gap(answer):
            alternatives = self._generate_context_gap_alternatives(
                query, ctx, answer, model
            )
            if alternatives:
                meta["context_gap_alternatives"] = alternatives
                meta["context_gap_heading"] = "Did you mean one of these?"
                trace.append(f"Generator: added {len(alternatives)} context-gap alternatives")
        # When disambiguation was set (taxonomy or grader underspecified), append a short note so the UI can show refined queries.
        if meta.get("needs_clarification"):
            if meta.get("taxonomy_options"):
                answer += "\n\nIf you'd like a more specific answer, you can pick one of the options below."
            elif meta.get("clarification_questions") or meta.get("missing_info_description"):
                answer += "\n\nYour question could refer to more than one item in the documents. If you'd like a more specific answer, pick one of the refined queries below, edit one, or rephrase with more detail."
        trace.append("Generator: done")
        return {**state, "answer": answer, "trace": trace, "metadata": meta}

    @staticmethod
    def _route_decision(state: StateDict) -> str:
        return state.get("route", "VECTOR")

    @staticmethod
    def _router_next_node(state: StateDict) -> str:
        """Send CAG to cag_retriever; all other routes go to all_retrievers (merged context)."""
        return "cag_retriever" if state.get("route") == "CAG" else "all_retrievers"

    @staticmethod
    def _grade_decision(state: StateDict) -> str:
        verdict = state.get("grader_verdict", "not_relevant")
        if verdict == "relevant":
            return "relevant"
        if verdict == "underspecified":
            return "underspecified"
        # not_relevant: try rewrite if retries remain, else clarify
        if state.get("retries", 0) < state.get("max_retries", 2):
            return "not_relevant"
        return "fail"

    def _fast_response(self, query: str, model: str) -> dict[str, Any]:
        """Vector search only + LLM. No taxonomy, PageIndex, Graph, grader, or clarifier."""
        trace: list[str] = [f"Query: {query}", "Fast Response: vector search only"]
        grounding = self._get_vector_grounding(query)
        ctx = grounding.get("context", "")
        sources = grounding.get("sources", [])
        meta: dict[str, Any] = {"vector_results_count": len(grounding.get("results", [])), "vector_sources": sources}
        trace.append(f"Fast Response: found {len(grounding.get('results', []))} chunks")
        if ctx.strip():
            ctx_limit = config.RAG_CONTEXT_MAX_CHARS
            prompt = f"Answer using ONLY the context below.\n\nQuestion: {query}\n\nContext:\n{ctx[:ctx_limit]}\n\nProvide a clear, concise answer."
        else:
            prompt = f"No context found from documents. Answer from general knowledge.\n\nQuestion: {query}"
        answer = self.llm.generate(prompt, model=model, options=config.GENERATION_OPTIONS)
        answer = answer.strip()
        if ctx.strip() and self._answer_indicates_context_gap(answer):
            alternatives = self._generate_context_gap_alternatives(query, ctx, answer, model)
            if alternatives:
                meta["context_gap_alternatives"] = alternatives
                meta["context_gap_heading"] = "Did you mean one of these?"
                trace.append(f"Fast Response: added {len(alternatives)} context-gap alternatives")
        trace.append("Fast Response: done")
        return {
            "answer": answer,
            "route": "VECTOR",
            "trace": trace,
            "context": ctx[:1000],
            "metadata": meta,
            "retries": 0,
            "error": "",
        }

    def query(
        self,
        query: str,
        force_route: str | None = None,
        model: str | None = None,
        fast_response: bool = False,
        enable_vector: bool = True,
        enable_taxonomy: bool = True,
        enable_pageindex: bool = True,
        enable_graph: bool = True,
    ) -> dict[str, Any]:
        effective_model = config.FAST_RESPONSE_MODEL if fast_response else (model or config.LLM_MODEL)
        logger.info(
            "router.query: ENTER query=%r force_route=%s model=%s fast_response=%s",
            (query or "")[:100], force_route, effective_model, fast_response,
        )
        # Fast Response: vector search only + LLM, no grader/taxonomy/pageindex/graph
        if fast_response:
            return self._fast_response(query, effective_model)

        initial: StateDict = {
            "query": query, "original_query": query,
            "route": force_route.upper() if force_route else "",
            "context": "", "answer": "", "grader_verdict": "not_relevant",
            "retries": 0, "max_retries": 2,
            "trace": [f"Query: {query}"], "metadata": {}, "error": "",
            "model": effective_model,
            "forced_route": bool(force_route),
            "enable_vector": enable_vector,
            "enable_taxonomy": enable_taxonomy,
            "enable_pageindex": enable_pageindex,
            "enable_graph": enable_graph,
        }
        if force_route:
            initial["trace"].append(f"Router: forced to {force_route.upper()}")
        try:
            final = self._graph.invoke(initial)
        except Exception as exc:
            logger.error("router.query: RAG pipeline failed: %s", exc)
            final = {**initial, "answer": f"Error: {exc}", "trace": initial["trace"] + [f"ERROR: {exc}"], "error": str(exc)}
        route = final.get("route", "")
        has_answer = bool(final.get("answer", ""))
        needs_clarification = bool((final.get("metadata") or {}).get("needs_clarification"))
        logger.info(
            "router.query: EXIT route=%s has_answer=%s needs_clarification=%s retries=%d error=%s",
            route, has_answer, needs_clarification, final.get("retries", 0), final.get("error") or "none",
        )
        return {
            "answer": final.get("answer", ""), "route": route,
            "trace": final.get("trace", []), "context": final.get("context", "")[:1000],
            "metadata": final.get("metadata", {}), "retries": final.get("retries", 0),
            "error": final.get("error", ""),
        }
