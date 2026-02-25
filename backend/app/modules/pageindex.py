"""
Module I: PageIndex – Tree-Based Reasoning Retrieval Engine

Supports the official PageIndex node format (node_id, start_index, end_index,
summary, nodes) as per https://pageindex.ai/blog/pageindex-intro and
https://docs.pageindex.ai. Trees are stored in that format; legacy format
(children, id, page_numbers) is still supported when loading.
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
try:
    import pymupdf4llm as pdf4llm
except ImportError:  # optional dependency at runtime
    pdf4llm = None

from app import config
from app.modules.ollama_client import OllamaClient, AsyncOllamaClient

logger = logging.getLogger(__name__)

# Official PageIndex uses "nodes" for children and node_id/start_index/end_index (page range).
OFFICIAL_NODE_IDS = "node_id"
OFFICIAL_NODES_KEY = "nodes"


@dataclass
class TreeNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    level: int = 0
    text: str = ""
    summary: str = ""
    children: list[TreeNode] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    # Official format fields (when loaded from or saved as official format)
    start_index: int | None = None  # start page (1-based)
    end_index: int | None = None    # end page (1-based)

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
        sub = data.get("children") or data.get("nodes") or []
        children = [cls.from_dict(c) for c in sub]
        return cls(
            id=data.get("id", data.get("node_id", str(uuid.uuid4())[:8])),
            title=data.get("title", ""),
            level=data.get("level", 0),
            text=data.get("text", ""),
            summary=data.get("summary", ""),
            page_numbers=data.get("page_numbers", []),
            children=children,
            start_index=data.get("start_index"),
            end_index=data.get("end_index"),
        )


def _is_paragraph_child(node: TreeNode, parent_title: str) -> bool:
    """True if node is a leaf titled 'Paragraph (ParentTitle)' (our in-house pattern)."""
    if not node.is_leaf() or not node.title:
        return False
    return node.title.strip().startswith("Paragraph (") and (
        parent_title in node.title or node.title.strip() == f"Paragraph ({parent_title})"
    )


def _collect_descendant_pages(node: TreeNode) -> list[int]:
    """Collect all page numbers from this node and its descendants. Used to infer start_index/end_index when node has none."""
    pages: list[int] = []
    if node.page_numbers:
        pages.extend(node.page_numbers)
    for child in node.children:
        pages.extend(_collect_descendant_pages(child))
    return sorted(set(pages)) if pages else []


def _tree_to_official_node(node: TreeNode, node_id_counter: list[int]) -> dict[str, Any]:
    """
    Convert one TreeNode to official PageIndex node.
    Collapse section + single "Paragraph (Section)" child into one node (company-style output).
    """
    nid = node_id_counter[0]
    node_id_counter[0] += 1
    node_id_str = str(nid).zfill(4)

    # Merge section + single paragraph child into one node (like official PageIndex output).
    if (
        len(node.children) == 1
        and _is_paragraph_child(node.children[0], node.title or "")
    ):
        child = node.children[0]
        pages = child.page_numbers or node.page_numbers or []
        if not pages:
            pages = _collect_descendant_pages(node)
        if child.start_index is not None and child.end_index is not None:
            start, end = child.start_index, child.end_index
        elif node.start_index is not None and node.end_index is not None:
            start, end = node.start_index, node.end_index
        elif pages:
            start, end = min(pages), max(pages)
        else:
            start, end = 1, 1
        out = {
            "title": node.title or "",
            OFFICIAL_NODE_IDS: node_id_str,
            "start_index": start,
            "end_index": end,
            "summary": (child.summary or node.summary or "").strip(),
        }
        if (child.text or "").strip():
            out["text"] = child.text.strip()
        return out

    pages = node.page_numbers or []
    if not pages and node.children:
        pages = _collect_descendant_pages(node)
    if node.start_index is not None and node.end_index is not None:
        start, end = node.start_index, node.end_index
    elif pages:
        start, end = min(pages), max(pages)
    else:
        start, end = 1, 1
    out = {
        "title": node.title or "",
        OFFICIAL_NODE_IDS: node_id_str,
        "start_index": start,
        "end_index": end,
        "summary": (node.summary or "").strip(),
    }
    if node.children:
        out[OFFICIAL_NODES_KEY] = [
            _tree_to_official_node(c, node_id_counter) for c in node.children
        ]
    else:
        if (node.text or "").strip():
            out["text"] = node.text.strip()
    return out


def tree_to_official_format(root: TreeNode, doc_title: str | None = None) -> dict[str, Any]:
    """
    Convert internal TreeNode to official PageIndex JSON structure:
    { "title", "node_id", "start_index", "end_index", "summary", "nodes" }.
    """
    counter: list[int] = [1]
    title = doc_title or root.title or "Document"
    pages = root.page_numbers or []
    if not pages and root.children:
        pages = _collect_descendant_pages(root)
    if root.start_index is not None and root.end_index is not None:
        start, end = root.start_index, root.end_index
    elif pages:
        start, end = (min(pages), max(pages)) if pages else (1, 1)
    else:
        start, end = 1, 1
    root_node: dict[str, Any] = {
        "title": title,
        OFFICIAL_NODE_IDS: "0000",
        "start_index": start,
        "end_index": end,
        "summary": (root.summary or "").strip(),
    }
    if root.children:
        root_node[OFFICIAL_NODES_KEY] = [_tree_to_official_node(c, counter) for c in root.children]
    else:
        root_node[OFFICIAL_NODES_KEY] = []
    return root_node


def minimal_official_tree(doc_title: str, content_text: str) -> dict[str, Any]:
    """
    Build a minimal PageIndex tree in official format (node_id, start_index, end_index, nodes)
    for a single block of content (e.g. non-PDF or fallback after OCR).
    """
    text = (content_text or "")[:500000].strip()
    return {
        "title": doc_title or "Document",
        OFFICIAL_NODE_IDS: "0000",
        "start_index": 1,
        "end_index": 1,
        "summary": "",
        OFFICIAL_NODES_KEY: [
            {
                "title": "Content",
                OFFICIAL_NODE_IDS: "0001",
                "start_index": 1,
                "end_index": 1,
                "summary": "",
                "text": text,
                # Leaf: no "nodes" key
            }
        ],
    }


def official_format_to_tree(data: dict[str, Any]) -> TreeNode:
    """Load official PageIndex JSON (nodes, node_id, start_index, end_index) into TreeNode."""
    node_id = data.get(OFFICIAL_NODE_IDS, data.get("id", str(uuid.uuid4())[:8]))
    title = data.get("title", "")
    start_index = data.get("start_index")
    end_index = data.get("end_index")
    sub = data.get(OFFICIAL_NODES_KEY, data.get("children", []))
    children = [official_format_to_tree(n) for n in sub] if sub else []
    page_numbers = data.get("page_numbers")
    if page_numbers is None and start_index is not None and end_index is not None:
        page_numbers = list(range(start_index, end_index + 1))
    return TreeNode(
        id=str(node_id),
        title=title,
        level=data.get("level", 0),
        text=data.get("text", ""),
        summary=data.get("summary", ""),
        page_numbers=page_numbers or [],
        children=children,
        start_index=start_index,
        end_index=end_index,
    )


def create_node_mapping(
    tree_data: dict[str, Any],
    max_page: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build a flat mapping node_id -> {node, start_index, end_index} from official PageIndex tree.
    Used with get_page_images_for_nodes for reasoning-based retrieval.
    """
    node_map: dict[str, dict[str, Any]] = {}

    def walk(n: dict[str, Any]) -> None:
        nid = n.get(OFFICIAL_NODE_IDS, n.get("id", ""))
        if not nid:
            return
        start = n.get("start_index", 1)
        end = n.get("end_index", 1)
        if max_page is not None:
            end = min(end, max_page)
        node_map[str(nid)] = {"node": n, "start_index": start, "end_index": end}
        for child in n.get(OFFICIAL_NODES_KEY, n.get("children", [])):
            walk(child)

    # Include root if present
    root_id = tree_data.get(OFFICIAL_NODE_IDS, tree_data.get("id"))
    if root_id is not None:
        start = tree_data.get("start_index", 1)
        end = tree_data.get("end_index", 1)
        if max_page is not None:
            end = min(end, max_page)
        node_map[str(root_id)] = {"node": tree_data, "start_index": start, "end_index": end}
    for child in tree_data.get(OFFICIAL_NODES_KEY, tree_data.get("children", [])):
        walk(child)
    logger.debug("pageindex.create_node_mapping: built map with %d nodes (max_page=%s)", len(node_map), max_page)
    return node_map


def remove_fields_from_tree(tree_data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Return a deep copy of the tree with the given keys removed from every node."""
    import copy
    out = copy.deepcopy(tree_data)
    to_remove = set(fields)

    def strip(n: dict[str, Any]) -> None:
        for k in list(n.keys()):
            if k in to_remove:
                del n[k]
        for child in n.get(OFFICIAL_NODES_KEY, n.get("children", [])):
            strip(child)

    strip(out)
    return out


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

    @staticmethod
    def _custom_hdr_prefix(span: dict[str, Any]) -> str:
        """
        Preserve existing business rules when pdf4llm asks for heading prefixes.
        """
        size = float(span.get("size", 0.0))
        is_bold = "Bold" in span.get("font", "") or "bold" in span.get("font", "")
        if size >= config.PAGEINDEX_H1_FONT_SIZE:
            return "# "
        if size >= config.PAGEINDEX_H2_FONT_SIZE or (
            size >= config.PAGEINDEX_BODY_FONT_SIZE and is_bold
        ):
            return "## "
        return ""

    def _to_markdown_pdf4llm(self, pdf_path: str | Path) -> str | None:
        if pdf4llm is None:
            return None

        with fitz.open(str(pdf_path)) as doc:
            try:
                # page_chunks=True returns one dict per page with "metadata" (page_number) and "text"
                # so we can inject <!-- PAGE N --> for correct start_index/end_index in the tree.
                md = pdf4llm.to_markdown(
                    doc,
                    page_chunks=True,
                    hdr_info=lambda span, page=None: self._custom_hdr_prefix(span),
                )
            except TypeError:
                try:
                    md = pdf4llm.to_markdown(
                        doc,
                        hdr_info=lambda span, page=None: self._custom_hdr_prefix(span),
                    )
                except TypeError:
                    md = pdf4llm.to_markdown(doc)

        if isinstance(md, str):
            return md
        if isinstance(md, list) and md:
            first = md[0]
            # page_chunks format: list of dicts with "metadata" (page_number) and "text"
            if isinstance(first, dict) and "metadata" in first:
                parts: list[str] = []
                for chunk in md:
                    meta = chunk.get("metadata") or {}
                    page_num = meta.get("page_number")
                    text = (chunk.get("text") or "").strip()
                    if isinstance(page_num, int) and page_num >= 1:
                        parts.append(f"<!-- PAGE {page_num} -->\n{text}")
                    else:
                        parts.append(text)
                return "\n".join(parts).strip()
            # Legacy list format (no page info)
            parts = []
            for item in md:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return None

    def _build_tree_legacy(self, pdf_path: str | Path) -> TreeNode:
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

    def build_tree(self, pdf_path: str | Path) -> TreeNode:
        # Optional: use legacy PDF parser for accurate page numbers (start_index/end_index).
        if getattr(config, "PAGEINDEX_USE_LEGACY_PDF", False):
            return self._build_tree_legacy(pdf_path)
        total_pages: int | None = None
        try:
            with fitz.open(str(pdf_path)) as doc:
                total_pages = len(doc)
        except Exception:
            pass
        md_content = self._to_markdown_pdf4llm(pdf_path)
        if md_content and md_content.strip():
            return self.build_tree_from_markdown(md_content, Path(pdf_path).stem, total_pages=total_pages)

        if pdf4llm is None:
            logger.info("pymupdf4llm not installed; using legacy PageIndex parser")
        else:
            logger.warning("pymupdf4llm returned empty Markdown; using legacy PageIndex parser")
        return self._build_tree_legacy(pdf_path)

    @staticmethod
    def build_tree_from_markdown(md_content: str, doc_name: str, total_pages: int | None = None) -> TreeNode:
        """
        Build a PageIndex tree from structured Markdown (e.g. from Docling or VLM).
        Parses # H1, ## H2, ### H3 and body text into the same TreeNode structure
        the reasoning engine expects.

        Page-number tracking:
        If the Markdown contains ``<!-- PAGE N -->`` markers (emitted by the VLM
        pipeline), the parser tracks the current page and attaches ``page_numbers``
        to every node created under that marker. If there are no markers but
        total_pages is provided (e.g. from the source PDF), synthetic markers are
        injected by character offset so start_index/end_index reflect approximate
        page spans. Without markers or total_pages the page lists remain empty.
        """
        import re
        md_content = (md_content or "").strip()
        if not md_content:
            return TreeNode(title=doc_name, level=0)
        # Inject synthetic <!-- PAGE N --> when we have total_pages but no markers (e.g. Docling output).
        if total_pages is not None and total_pages >= 1 and "<!-- PAGE" not in md_content:
            lines = md_content.splitlines()
            total_chars = sum(len(line) + 1 for line in lines)
            if total_chars > 0:
                parts: list[str] = []
                cumulative = 0
                current_injected_page = 0
                for line in lines:
                    cumulative += len(line) + 1
                    # 1-based page by proportional position
                    p = min(1 + int(cumulative * total_pages / total_chars), total_pages)
                    if p != current_injected_page:
                        current_injected_page = p
                        parts.append(f"<!-- PAGE {p} -->")
                    parts.append(line)
                md_content = "\n".join(parts)
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

            # Track <!-- PAGE N --> markers (VLM/OCR output); flush on page change so each page gets its own node(s)
            pm = page_marker_re.match(stripped)
            if pm:
                new_page = int(pm.group(1))
                if current_page is not None and new_page != current_page and body_buffer:
                    _flush_body()
                current_page = new_page
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
        self,
        md_content: str,
        doc_name: str,
        out_name: str | None = None,
        callback: Any | None = None,
        pdf_path: str | Path | None = None,
    ) -> TreeNode:
        """
        Build and summarize a tree from pre-structured Markdown, then save to PageIndex JSON.
        Use after Docling (or similar) conversion for empty documents.
        If pdf_path is provided and is a PDF, total_pages is inferred so start_index/end_index reflect page spans.
        """
        total_pages: int | None = None
        if pdf_path is not None:
            pdf_path = Path(pdf_path)
            if pdf_path.suffix.lower() == ".pdf" and pdf_path.exists():
                try:
                    with fitz.open(str(pdf_path)) as doc:
                        total_pages = len(doc)
                except Exception:
                    pass
        if callback:
            callback(f"Building tree from Markdown for {doc_name}...")
        root = self.build_tree_from_markdown(md_content, doc_name, total_pages=total_pages)
        if callback:
            callback(f"Tree built with {self._count_nodes(root)} nodes. Summarizing...")
        root = self.summarize_tree(root, callback)
        name = out_name or doc_name
        out_path = config.PAGEINDEX_DIR / f"{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        official = tree_to_official_format(root, doc_title=name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(official, f, indent=2, ensure_ascii=False)
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
            # Official PageIndex-style prompt: describe main points (no short-sentence cap).
            # See VectifyAI/PageIndex pageindex/utils.py generate_node_summary()
            text_slice = node.text[:8000]
            prompt = (
                "You are given a part of a document. Your task is to generate a description "
                "of the partial document about what are the main points covered in the partial document.\n\n"
                f"Partial Document Text:\n{text_slice}\n\n"
                "Directly return the description, do not include any other text."
            )
            node.summary = await self.async_llm.generate(prompt, options=config.REASONING_OPTIONS)
            node.summary = (node.summary or "").strip()
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
        official = tree_to_official_format(root, doc_title=pdf_path.stem)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(official, f, indent=2, ensure_ascii=False)
        if callback:
            callback(f"PageIndex saved to {out_path.name}")
        return root

    @staticmethod
    def _count_nodes(node: TreeNode) -> int:
        return 1 + sum(TreeBuilder._count_nodes(c) for c in node.children)


class TreeSearcher:
    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()

    def load_tree_raw(self, doc_name: str) -> dict[str, Any] | None:
        """Load PageIndex tree as raw JSON (official format) for reasoning retrieval and node mapping."""
        path = config.PAGEINDEX_DIR / f"{doc_name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_tree(self, doc_name: str) -> TreeNode | None:
        path = config.PAGEINDEX_DIR / f"{doc_name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get(OFFICIAL_NODES_KEY) is not None or data.get(OFFICIAL_NODE_IDS) is not None:
            return official_format_to_tree(data)
        return TreeNode.from_dict(data)

    def list_trees(self) -> list[str]:
        if not config.PAGEINDEX_DIR.exists():
            return []
        return [p.stem for p in config.PAGEINDEX_DIR.glob("*.json")]

    def list_documents_with_summaries(self) -> list[dict[str, Any]]:
        """
        List available PageIndex documents with root summary and page range (official PageIndex-style).
        Returns list of {doc_name, summary, start_index, end_index, page_count} for each tree.
        """
        out: list[dict[str, Any]] = []
        for doc_name in self.list_trees():
            tree = self.load_tree(doc_name)
            if tree is None:
                continue
            pages = tree.page_numbers or []
            if not pages and tree.children:
                pages = _collect_descendant_pages(tree)
            start_index = end_index = 1
            if tree.start_index is not None and tree.end_index is not None:
                start_index, end_index = tree.start_index, tree.end_index
            elif pages:
                start_index, end_index = min(pages), max(pages)
            out.append({
                "doc_name": doc_name,
                "summary": (tree.summary or "").strip(),
                "start_index": start_index,
                "end_index": end_index,
                "page_count": end_index - start_index + 1 if end_index >= start_index else 0,
            })
        return out

    def get_document_structure(self, doc_name: str) -> dict[str, Any] | None:
        """
        Return document structure in official PageIndex format: flat list of nodes with
        title, node_id, start_index, end_index, summary (no full text). Use for
        list-docs → get_structure → get_page_content flow alignment.
        """
        path = config.PAGEINDEX_DIR / f"{doc_name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Already in official format; flatten nodes for structure view.
        structure: list[dict[str, Any]] = []

        def flatten_nodes(nodes: list[dict[str, Any]]) -> None:
            for n in nodes or []:
                structure.append({
                    "title": n.get("title", ""),
                    "node_id": n.get(OFFICIAL_NODE_IDS, n.get("id", "")),
                    "start_index": n.get("start_index", 1),
                    "end_index": n.get("end_index", 1),
                    "summary": (n.get("summary") or "").strip(),
                })
                flatten_nodes(n.get(OFFICIAL_NODES_KEY, n.get("children", [])))

        flatten_nodes(data.get(OFFICIAL_NODES_KEY, data.get("children", [])))
        return {
            "success": True,
            "doc_name": doc_name,
            "structure": structure,
            "next_steps": {
                "options": ["Use get_page_content() to extract specific content from pages"],
                "summary": "Document structure retrieved successfully.",
            },
        }

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
        # Frontier: (node, path_from_root) where path_from_root = list of ancestors (root..parent of node)
        frontier: list[tuple[TreeNode, list[TreeNode]]] = [(tree, [])]
        visited_leaves: list[tuple[TreeNode, list[TreeNode]]] = []  # (leaf, ancestors)
        while frontier:
            next_frontier: list[tuple[TreeNode, list[TreeNode]]] = []
            for node, ancestors in frontier:
                if node.is_leaf():
                    visited_leaves.append((node, ancestors))
                    continue
                selected = self._select_children(query, node, beam_width)
                path_here = ancestors + [node]
                for child in selected:
                    trace.append(f"  -> {child.title}")
                    next_frontier.append((child, path_here))
            frontier = next_frontier
        context_parts = []
        leaf_dicts = []
        for leaf, ancestors in visited_leaves:
            # Contextual flattening: Document / Section / Subsection + summaries, then content.
            # Include summaries so empty-parent text and markdown tables are backed by natural language.
            section_parts = [f"Document: {tree.title}."]
            for a in ancestors:
                if a.title:
                    section_parts.append(f"Section ({a.level}): {a.title}.")
                if a.summary and a.summary.strip():
                    section_parts.append(f"Summary: {a.summary.strip()}")
            if leaf.title:
                section_parts.append(f"Subsection: {leaf.title}.")
            # Use summary only for retrieval context (official PageIndex has no text in output).
            # We keep text in our tree/JSON but don't feed it to context to avoid formatting noise.
            leaf_summary = (leaf.summary or "").strip()
            leaf_content = leaf_summary
            if section_parts or leaf_content:
                block = " ".join(section_parts)
                if block and leaf_content:
                    block = block + " Content: " + leaf_content
                elif leaf_content:
                    block = leaf_content
                if block:
                    context_parts.append(block)
                pages = leaf.page_numbers
                if not pages and leaf.start_index is not None and leaf.end_index is not None:
                    pages = list(range(leaf.start_index, leaf.end_index + 1))
                leaf_dicts.append({
                    "id": leaf.id,
                    "node_id": leaf.id,
                    "title": leaf.title,
                    "pages": pages,
                    "doc_name": doc_name,
                })
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
            f"You are a research assistant navigating a document tree (PageIndex style).\n"
            f"The user asked: \"{query}\"\n"
            f"You are currently at the section: \"{node.title}\"\n\n"
            f"Here are the subsections:\n{child_desc}\n\n"
            f"Which subsection(s) are most likely to contain the answer?\n"
            f"Return a JSON object with a key \"node_ids\" containing a list "
            f"of up to {beam_width} Node IDs (use the ID values shown above).\n"
            f"Example: {{\"node_ids\": [\"0001\", \"0002\"]}}"
        )
        result = self.llm.generate_json(prompt, options=config.REASONING_OPTIONS)
        selected_ids = result.get("node_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = [selected_ids]
        selected_ids = [str(s).strip() for s in selected_ids]
        id_to_child = {c.id: c for c in node.children}
        selected = [id_to_child[nid] for nid in selected_ids if nid in id_to_child]
        if not selected:
            selected = node.children[:beam_width]
        return selected

    def reasoning_retrieve(
        self,
        query: str,
        doc_name: str,
        total_pages: int | None = None,
    ) -> dict[str, Any]:
        """
        Reasoning-based retrieval: use VLM with tree structure (no full text) to select
        relevant node IDs. Returns {"node_list": [...], "thinking": "..."} for use with
        create_node_mapping and get_page_images_for_nodes.
        """
        tree_data = self.load_tree_raw(doc_name)
        if not tree_data:
            logger.info("pageindex.reasoning_retrieve: no tree for doc_name=%s", doc_name)
            return {"node_list": [], "thinking": ""}
        logger.info("pageindex.reasoning_retrieve: doc_name=%s query=%r using VLM=%s", doc_name, (query or "")[:80], config.VLM_MODEL)
        tree_without_text = remove_fields_from_tree(tree_data, fields=["text"])
        search_prompt = (
            "You are given a question and a tree structure of a document.\n"
            "Each node contains a node id, node title, and a corresponding summary.\n"
            "Your task is to find all tree nodes that are likely to contain the answer to the question.\n\n"
            f"Question: {query}\n\n"
            f"Document tree structure:\n{json.dumps(tree_without_text, indent=2)}\n\n"
            'Reply with ONLY a JSON object: {"thinking": "<your reasoning>", "node_list": ["id1", "id2", ...]}'
        )
        try:
            raw = self.llm.generate(search_prompt, model=config.VLM_MODEL, options=config.REASONING_OPTIONS)
            if not (raw and raw.strip()):
                return {"node_list": [], "thinking": ""}
            text = raw.strip()
            if "```" in text:
                for part in text.split("```"):
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        text = part
                        break
            parsed = json.loads(text)
            node_list = parsed.get("node_list", [])
            if not isinstance(node_list, list):
                node_list = [node_list] if node_list else []
            node_list = [str(n).strip() for n in node_list if n]
            thinking = str(parsed.get("thinking", "")).strip()
            logger.info(
                "pageindex.reasoning_retrieve: doc_name=%s -> node_list=%s thinking_len=%d",
                doc_name, node_list, len(thinking),
            )
            if thinking:
                logger.debug("pageindex.reasoning_retrieve: thinking=%s", thinking[:300])
            return {"node_list": node_list, "thinking": thinking}
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("pageindex.reasoning_retrieve: JSON parse failed doc_name=%s: %s", doc_name, e)
            return {"node_list": [], "thinking": ""}
