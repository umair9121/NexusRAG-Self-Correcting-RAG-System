"""
Module I: PageIndex – Tree-Based Reasoning Retrieval Engine
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app import config
from app.modules.ollama_client import OllamaClient, AsyncOllamaClient

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    level: int = 0
    text: str = ""
    summary: str = ""
    children: list[TreeNode] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "level": self.level,
            "text": self.text, "summary": self.summary,
            "page_numbers": self.page_numbers,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            id=data["id"], title=data.get("title", ""),
            level=data.get("level", 0), text=data.get("text", ""),
            summary=data.get("summary", ""),
            page_numbers=data.get("page_numbers", []), children=children,
        )


class TreeBuilder:
    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self.async_llm = AsyncOllamaClient()

    @staticmethod
    def extract_blocks(pdf_path: str | Path) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        blocks.append({
                            "text": text,
                            "font_size": span.get("size", 10.0),
                            "font_name": span.get("font", ""),
                            "is_bold": "Bold" in span.get("font", "") or "bold" in span.get("font", ""),
                            "page": page_num,
                        })
        doc.close()
        return blocks

    @staticmethod
    def classify_level(font_size: float, is_bold: bool) -> int:
        if font_size >= config.PAGEINDEX_H1_FONT_SIZE:
            return 1
        elif font_size >= config.PAGEINDEX_H2_FONT_SIZE or (
            font_size >= config.PAGEINDEX_BODY_FONT_SIZE and is_bold
        ):
            return 2
        else:
            return 3

    def build_tree(self, pdf_path: str | Path) -> TreeNode:
        blocks = self.extract_blocks(pdf_path)
        if not blocks:
            return TreeNode(title="Empty Document", level=0)
        root = TreeNode(title=Path(pdf_path).stem, level=0)
        current_h1: TreeNode | None = None
        current_h2: TreeNode | None = None
        body_buffer: list[str] = []
        page_buffer: list[int] = []

        def _flush_body():
            nonlocal body_buffer, page_buffer
            if not body_buffer:
                return
            text = " ".join(body_buffer)
            parent = current_h2 or current_h1 or root
            parent.children.append(TreeNode(
                title=f"Paragraph ({parent.title})", level=3,
                text=text, page_numbers=list(set(page_buffer)),
            ))
            body_buffer = []
            page_buffer = []

        for blk in blocks:
            level = self.classify_level(blk["font_size"], blk["is_bold"])
            if level == 1:
                _flush_body()
                current_h1 = TreeNode(title=blk["text"], level=1, page_numbers=[blk["page"]])
                root.children.append(current_h1)
                current_h2 = None
            elif level == 2:
                _flush_body()
                current_h2 = TreeNode(title=blk["text"], level=2, page_numbers=[blk["page"]])
                parent = current_h1 or root
                parent.children.append(current_h2)
            else:
                body_buffer.append(blk["text"])
                page_buffer.append(blk["page"])
        _flush_body()
        if not root.children:
            all_text = " ".join(b["text"] for b in blocks)
            root.children.append(TreeNode(title="Document Content", level=1, text=all_text))
        return root

    @staticmethod
    def build_tree_from_markdown(md_content: str, doc_name: str) -> TreeNode:
        """
        Build a PageIndex tree from structured Markdown (e.g. from Docling or VLM).
        Parses # H1, ## H2, ### H3 and body text into the same TreeNode structure
        the reasoning engine expects.

        Page-number tracking:
        If the Markdown contains ``<!-- PAGE N -->`` markers (emitted by the VLM
        pipeline), the parser tracks the current page and attaches ``page_numbers``
        to every node created under that marker.  Without markers the page lists
        remain empty (backward-compatible with plain Docling output).
        """
        import re
        root = TreeNode(title=doc_name, level=0)
        current_h1: TreeNode | None = None
        current_h2: TreeNode | None = None
        current_h3: TreeNode | None = None
        body_buffer: list[str] = []
        page_buffer: list[int] = []      # pages seen while accumulating body text
        current_page: int | None = None   # set by <!-- PAGE N --> markers

        def _flush_body():
            nonlocal body_buffer, page_buffer
            if not body_buffer:
                return
            text = "\n".join(body_buffer).strip()
            if not text:
                body_buffer = []
                page_buffer = []
                return
            parent = current_h3 or current_h2 or current_h1 or root
            parent.children.append(
                TreeNode(
                    title=f"Paragraph ({parent.title})",
                    level=3,
                    text=text,
                    page_numbers=sorted(set(page_buffer)) if page_buffer else [],
                )
            )
            body_buffer = []
            page_buffer = []

        page_marker_re = re.compile(r"^<!--\s*PAGE\s+(\d+)\s*-->$")

        lines = md_content.splitlines()
        for line in lines:
            stripped = line.strip()

            # Track <!-- PAGE N --> markers (VLM output)
            pm = page_marker_re.match(stripped)
            if pm:
                current_page = int(pm.group(1))
                continue

            if not stripped:
                if body_buffer:
                    body_buffer.append("")
                continue

            s = stripped
            m1 = re.match(r"^#\s+(.+)$", s)
            m2 = re.match(r"^##\s+(.+)$", s)
            m3 = re.match(r"^###\s+(.+)$", s)
            if m1:
                _flush_body()
                current_h1 = TreeNode(
                    title=m1.group(1).strip(), level=1,
                    page_numbers=[current_page] if current_page else [],
                )
                root.children.append(current_h1)
                current_h2 = None
                current_h3 = None
            elif m2:
                _flush_body()
                current_h2 = TreeNode(
                    title=m2.group(1).strip(), level=2,
                    page_numbers=[current_page] if current_page else [],
                )
                (current_h1 or root).children.append(current_h2)
                current_h3 = None
            elif m3:
                _flush_body()
                current_h3 = TreeNode(
                    title=m3.group(1).strip(), level=3,
                    page_numbers=[current_page] if current_page else [],
                )
                parent = current_h2 or current_h1 or root
                parent.children.append(current_h3)
            else:
                body_buffer.append(s)
                if current_page is not None:
                    page_buffer.append(current_page)

        _flush_body()
        if not root.children:
            root.children.append(
                TreeNode(title="Document Content", level=1, text=md_content.strip(), page_numbers=[])
            )
        return root

    def ingest_markdown(
        self, md_content: str, doc_name: str, out_name: str | None = None, callback: Any | None = None
    ) -> TreeNode:
        """
        Build and summarize a tree from pre-structured Markdown, then save to PageIndex JSON.
        Use after Docling (or similar) conversion for empty documents.
        """
        if callback:
            callback(f"Building tree from Markdown for {doc_name}...")
        root = self.build_tree_from_markdown(md_content, doc_name)
        if callback:
            callback(f"Tree built with {self._count_nodes(root)} nodes. Summarizing...")
        root = self.summarize_tree(root, callback)
        name = out_name or doc_name
        out_path = config.PAGEINDEX_DIR / f"{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(root.to_dict(), f, indent=2, ensure_ascii=False)
        if callback:
            callback(f"PageIndex saved to {out_path.name}")
        return root

    def summarize_tree(self, root: TreeNode, callback: Any | None = None) -> TreeNode:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._async_summarize(root, callback))
        finally:
            loop.close()
        return root

    async def _async_summarize(self, node: TreeNode, callback: Any | None = None) -> None:
        if node.children:
            for child in node.children:
                await self._async_summarize(child, callback)
            child_summaries = "\n".join(
                f"- [{c.title}]: {c.summary}" for c in node.children if c.summary
            )
            if child_summaries:
                prompt = (
                    f"Summarize the following subsections in one concise sentence "
                    f"(max {config.PAGEINDEX_SUMMARY_MAX_TOKENS} tokens):\n\n{child_summaries}"
                )
                node.summary = await self.async_llm.generate(prompt, options=config.REASONING_OPTIONS)
                node.summary = node.summary.strip()
        elif node.text:
            prompt = (
                f"Summarize the following text in one concise sentence "
                f"(max {config.PAGEINDEX_SUMMARY_MAX_TOKENS} tokens):\n\n{node.text[:3000]}"
            )
            node.summary = await self.async_llm.generate(prompt, options=config.REASONING_OPTIONS)
            node.summary = node.summary.strip()
        if callback:
            callback(f"Summarized: {node.title}")

    def ingest(self, pdf_path: str | Path, callback: Any | None = None) -> TreeNode:
        pdf_path = Path(pdf_path)
        if callback:
            callback(f"Building tree from {pdf_path.name}...")
        root = self.build_tree(pdf_path)
        if callback:
            callback(f"Tree built with {self._count_nodes(root)} nodes. Summarizing...")
        root = self.summarize_tree(root, callback)
        out_path = config.PAGEINDEX_DIR / f"{pdf_path.stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(root.to_dict(), f, indent=2, ensure_ascii=False)
        if callback:
            callback(f"PageIndex saved to {out_path.name}")
        return root

    @staticmethod
    def _count_nodes(node: TreeNode) -> int:
        return 1 + sum(TreeBuilder._count_nodes(c) for c in node.children)


class TreeSearcher:
    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()

    def load_tree(self, doc_name: str) -> TreeNode | None:
        path = config.PAGEINDEX_DIR / f"{doc_name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return TreeNode.from_dict(json.load(f))

    def list_trees(self) -> list[str]:
        if not config.PAGEINDEX_DIR.exists():
            return []
        return [p.stem for p in config.PAGEINDEX_DIR.glob("*.json")]

    def search(
        self, query: str, tree: TreeNode | None = None,
        doc_name: str | None = None, beam_width: int = config.PAGEINDEX_BEAM_WIDTH,
    ) -> dict[str, Any]:
        if tree is None:
            if doc_name is None:
                return self._search_all(query, beam_width)
            tree = self.load_tree(doc_name)
            if tree is None:
                return {"context": "", "path": ["No tree found"], "leaf_nodes": []}
        trace: list[str] = [f"Root: {tree.title}"]
        frontier = [tree]
        visited_leaves: list[TreeNode] = []
        while frontier:
            next_frontier: list[TreeNode] = []
            for node in frontier:
                if node.is_leaf():
                    visited_leaves.append(node)
                    continue
                selected = self._select_children(query, node, beam_width)
                for child in selected:
                    trace.append(f"  -> {child.title}")
                    next_frontier.append(child)
            frontier = next_frontier
        context_parts = []
        leaf_dicts = []
        for leaf in visited_leaves:
            text = leaf.text or leaf.summary
            if text:
                context_parts.append(text)
                leaf_dicts.append({"id": leaf.id, "title": leaf.title, "pages": leaf.page_numbers})
                trace.append(f"  Leaf: {leaf.title}")
        return {"context": "\n\n---\n\n".join(context_parts), "path": trace, "leaf_nodes": leaf_dicts}

    def _search_all(self, query: str, beam_width: int) -> dict[str, Any]:
        all_ctx, all_paths, all_leaves = [], [], []
        for name in self.list_trees():
            r = self.search(query, doc_name=name, beam_width=beam_width)
            if r["context"]:
                all_ctx.append(r["context"])
                all_paths.extend(r["path"])
                all_leaves.extend(r["leaf_nodes"])
        return {"context": "\n\n===\n\n".join(all_ctx), "path": all_paths, "leaf_nodes": all_leaves}

    def _select_children(self, query: str, node: TreeNode, beam_width: int) -> list[TreeNode]:
        if not node.children:
            return []
        child_desc = "\n".join(
            f"  ID: {c.id} | Title: {c.title} | Summary: {c.summary}" for c in node.children
        )
        prompt = (
            f"You are a research assistant navigating a document tree.\n"
            f"The user asked: \"{query}\"\n"
            f"You are currently at the section: \"{node.title}\"\n\n"
            f"Here are the subsections:\n{child_desc}\n\n"
            f"Which subsection(s) are most likely to contain the answer?\n"
            f"Return a JSON object with a key \"node_ids\" containing a list "
            f"of up to {beam_width} Node IDs.\n"
            f"Example: {{\"node_ids\": [\"abc123\", \"def456\"]}}"
        )
        result = self.llm.generate_json(prompt, options=config.REASONING_OPTIONS)
        selected_ids = result.get("node_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = [selected_ids]
        id_to_child = {c.id: c for c in node.children}
        selected = [id_to_child[nid] for nid in selected_ids if nid in id_to_child]
        if not selected:
            selected = node.children[:beam_width]
        return selected
