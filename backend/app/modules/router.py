"""
The Orchestrator: Adaptive RAG with LangGraph
Focused on Graph RAG + CAG routing (PageIndex and Vector removed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import StateGraph, END

from app import config
from app.modules.ollama_client import OllamaClient
from app.modules.graph import GraphSearcher
from app.modules.cag import CAGManager

logger = logging.getLogger(__name__)

StateDict = dict[str, Any]


class AdaptiveRAG:
    VALID_ROUTES = {"GRAPH", "CAG"}
    FALLBACK_ORDER = ["GRAPH"]

    def __init__(self, llm: OllamaClient | None = None, cag_manager: CAGManager | None = None):
        self.llm = llm or OllamaClient()
        self.graph_searcher = GraphSearcher(self.llm)
        self.cag_manager = cag_manager or CAGManager(self.llm)
        self._graph = self._build_graph()

    def _build_graph(self):
        wf = StateGraph(state_schema=dict)
        wf.add_node("router", self._router_node)
        wf.add_node("graph_retriever", self._graph_node)
        wf.add_node("cag_retriever", self._cag_node)
        wf.add_node("grader", self._grader_node)
        wf.add_node("rewriter", self._rewriter_node)
        wf.add_node("generator", self._generator_node)
        wf.set_entry_point("router")
        wf.add_conditional_edges("router", self._route_decision, {"GRAPH": "graph_retriever", "CAG": "cag_retriever"})
        wf.add_edge("graph_retriever", "grader")
        wf.add_edge("cag_retriever", "generator")
        wf.add_conditional_edges("grader", self._grade_decision, {"relevant": "generator", "rewrite": "rewriter", "fail": "generator"})
        wf.add_edge("rewriter", "router")
        wf.add_edge("generator", END)
        return wf.compile()

    def _router_node(self, state: StateDict) -> StateDict:
        query = state.get("query", "")
        trace = state.get("trace", [])
        cag_active = self.cag_manager.is_active
        model = state.get("model")
        prompt = (
            "Analyze the user query and classify it into exactly one category.\n\n"
            "Categories:\n"
            "- GRAPH: questions about entities, relationships, themes, trends, or any factual query\n"
            f"- CAG: {'follow-up to previous context or conversational query' if cag_active else 'DISABLED'}\n\n"
            f"User query: \"{query}\"\n\nReturn JSON: {{\"route\": \"GRAPH\"}}"
        )
        result = self.llm.generate_json(prompt, model=model, options=config.REASONING_OPTIONS)
        route = result.get("route", "GRAPH").upper().strip()
        if route not in self.VALID_ROUTES:
            route = "GRAPH"
        if route == "CAG" and not cag_active:
            route = "GRAPH"
        trace.append(f"Router: classified as {route}")
        return {**state, "route": route, "trace": trace}

    def _graph_node(self, state: StateDict) -> StateDict:
        query, trace = state.get("query", ""), state.get("trace", [])
        trace.append("Graph: searching knowledge graph...")
        result = self.graph_searcher.search(query, mode="auto")
        trace.append(f"Graph: used {result.get('mode', 'unknown')} mode")
        meta = state.get("metadata", {})
        meta["graph_mode"] = result.get("mode", "")
        return {**state, "context": result.get("context", ""), "trace": trace, "metadata": meta}

    def _cag_node(self, state: StateDict) -> StateDict:
        query, trace = state.get("query", ""), state.get("trace", [])
        model = state.get("model")
        trace.append(f"CAG: using cached context for '{self.cag_manager.loaded_document}'")
        result = self.cag_manager.chat(query, model=model)
        trace.append(f"CAG: response generated (turn {result.get('conversation_turn', 0)})")
        return {**state, "answer": result.get("response", ""), "context": "[CAG]", "trace": trace}

    def _grader_node(self, state: StateDict) -> StateDict:
        query, ctx, trace = state.get("query", ""), state.get("context", ""), state.get("trace", [])
        model = state.get("model")
        trace.append("Grader: evaluating relevance...")
        if not ctx.strip():
            trace.append("Grader: NO context")
            return {**state, "is_relevant": False, "trace": trace}
        prompt = f"Does this context answer the question?\n\nQuestion: {query}\n\nContext:\n{ctx[:2000]}\n\nReturn JSON: {{\"relevant\": true/false}}"
        result = self.llm.generate_json(prompt, model=model, options=config.REASONING_OPTIONS)
        rel = result.get("relevant", False)
        if isinstance(rel, str):
            rel = rel.lower() in ("true", "yes", "1")
        trace.append(f"Grader: {'RELEVANT' if rel else 'NOT RELEVANT'}")
        return {**state, "is_relevant": rel, "trace": trace}

    def _rewriter_node(self, state: StateDict) -> StateDict:
        query = state.get("query", "")
        original = state.get("original_query", query)
        retries = state.get("retries", 0)
        route = state.get("route", "GRAPH")
        trace = state.get("trace", [])
        model = state.get("model")
        prompt = f"Rewrite this query to be more specific.\n\nOriginal: {original}\nFailed: {query}\n\nReturn JSON: {{\"rewritten_query\": \"...\"}}"
        result = self.llm.generate_json(prompt, model=model, options=config.REASONING_OPTIONS)
        rewritten = result.get("rewritten_query", query)
        new_route = "GRAPH"
        trace.append(f"Rewriter: '{query}' -> '{rewritten}' (trying {new_route})")
        return {**state, "query": rewritten, "route": new_route, "retries": retries + 1, "trace": trace}

    def _generator_node(self, state: StateDict) -> StateDict:
        query = state.get("original_query", "") or state.get("query", "")
        ctx, trace = state.get("context", ""), state.get("trace", [])
        model = state.get("model")
        if state.get("answer", ""):
            trace.append("Generator: using CAG answer")
            return {**state, "trace": trace}
        trace.append("Generator: synthesizing answer...")
        if ctx.strip():
            prompt = f"Answer using ONLY the context.\n\nQuestion: {query}\n\nContext:\n{ctx[:6000]}\n\nProvide a clear answer."
        else:
            prompt = f"No context found. Answer from knowledge.\n\nQuestion: {query}"
        answer = self.llm.generate(prompt, model=model, options=config.GENERATION_OPTIONS)
        trace.append("Generator: done")
        return {**state, "answer": answer.strip(), "trace": trace}

    @staticmethod
    def _route_decision(state: StateDict) -> str:
        return state.get("route", "GRAPH")

    @staticmethod
    def _grade_decision(state: StateDict) -> str:
        if state.get("is_relevant", False):
            return "relevant"
        if state.get("retries", 0) < state.get("max_retries", 2):
            return "rewrite"
        return "fail"

    def query(self, query: str, force_route: str | None = None, model: str | None = None) -> dict[str, Any]:
        effective_model = model or config.LLM_MODEL
        initial: StateDict = {
            "query": query, "original_query": query,
            "route": force_route.upper() if force_route else "",
            "context": "", "answer": "", "is_relevant": False,
            "retries": 0, "max_retries": 2,
            "trace": [f"Query: {query}"], "metadata": {}, "error": "",
            "model": effective_model,
        }
        if force_route:
            initial["trace"].append(f"Router: forced to {force_route.upper()}")
        try:
            final = self._graph.invoke(initial)
        except Exception as exc:
            logger.error("RAG pipeline failed: %s", exc)
            final = {**initial, "answer": f"Error: {exc}", "trace": initial["trace"] + [f"ERROR: {exc}"], "error": str(exc)}
        return {
            "answer": final.get("answer", ""), "route": final.get("route", ""),
            "trace": final.get("trace", []), "context": final.get("context", "")[:1000],
            "metadata": final.get("metadata", {}), "retries": final.get("retries", 0),
            "error": final.get("error", ""),
        }
