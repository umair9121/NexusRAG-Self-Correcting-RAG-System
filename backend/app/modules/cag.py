"""
Module IV: Cache-Augmented Generation (CAG)
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app import config
from app.modules.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def get_available_ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024 ** 3)
        else:
            meminfo_path = Path("/proc/meminfo")
            if meminfo_path.exists():
                with open(meminfo_path) as f:
                    for line in f:
                        if line.startswith("MemAvailable"):
                            return int(line.split()[1]) / (1024 ** 2)
            result = os.popen("sysctl -n hw.memsize").read().strip()
            if result:
                return int(result) / (1024 ** 3) * 0.5
            return 8.0
    except Exception:
        return 8.0


class CAGManager:
    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm or OllamaClient()
        self._loaded_doc: str | None = None
        self._doc_text: str = ""
        self._doc_token_count: int = 0
        self._conversation_history: list[dict[str, str]] = []
        self._system_prompt: str = ""
        self._is_active: bool = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def loaded_document(self) -> str | None:
        return self._loaded_doc

    def can_enable(self, doc_token_count: int | None = None):
        available_ram = get_available_ram_gb()
        tc = doc_token_count or self._doc_token_count
        can = available_ram >= config.CAG_ENABLED_RAM_THRESHOLD_GB and tc <= config.CAG_MAX_DOCUMENT_TOKENS
        return {
            "enabled": can, "available_ram_gb": round(available_ram, 1),
            "estimated_ram_needed_gb": round(3.0 + (tc / 4000), 1),
            "document_tokens": tc, "max_tokens": config.CAG_MAX_DOCUMENT_TOKENS,
            "reason": "OK" if can else (
                f"Document too large ({tc} tokens)" if tc > config.CAG_MAX_DOCUMENT_TOKENS
                else f"Insufficient RAM ({available_ram:.1f}GB)"
            ),
        }

    def load_document(self, pdf_path=None, text=None, doc_name=None):
        if pdf_path:
            pdf_path = Path(pdf_path)
            doc_name = doc_name or pdf_path.stem
            doc = fitz.open(str(pdf_path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        elif text is None:
            return {"success": False, "error": "No document provided"}
        doc_name = doc_name or "document"
        self._doc_token_count = len(text) // 4
        check = self.can_enable(self._doc_token_count)
        if not check["enabled"]:
            return {"success": False, "error": check["reason"], **check}
        max_chars = config.CAG_MAX_DOCUMENT_TOKENS * 4
        if len(text) > max_chars:
            text = text[:max_chars]
            self._doc_token_count = config.CAG_MAX_DOCUMENT_TOKENS
        self._doc_text = text
        self._loaded_doc = doc_name
        self._is_active = True
        self._conversation_history = []
        self._system_prompt = (
            "You are a helpful assistant with deep knowledge of the following document. "
            "Answer questions based on the document content.\n\n"
            f"=== DOCUMENT START ===\n{text}\n=== DOCUMENT END ==="
        )
        try:
            self.llm.chat(
                messages=[{"role": "system", "content": self._system_prompt}, {"role": "user", "content": "Acknowledge the document."}],
                options={"temperature": 0.0, "num_ctx": 8192, "num_keep": self._doc_token_count + 200},
            )
        except Exception as exc:
            logger.warning("CAG priming failed (non-fatal): %s", exc)
        return {"success": True, "doc_name": doc_name, "tokens": self._doc_token_count, **check}

    def chat(self, query: str, model: str | None = None):
        if not self._is_active:
            return {"response": "No document loaded in CAG.", "conversation_turn": 0, "doc_name": None}
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": query})
        try:
            response = self.llm.chat(
                messages=messages,
                model=model,
                options={"temperature": 0.2, "num_ctx": 8192, "num_keep": self._doc_token_count + 200},
            )
        except Exception as exc:
            return {"response": f"Error: {exc}", "conversation_turn": len(self._conversation_history) // 2, "doc_name": self._loaded_doc}
        self._conversation_history.append({"role": "user", "content": query})
        self._conversation_history.append({"role": "assistant", "content": response})
        return {"response": response, "conversation_turn": len(self._conversation_history) // 2, "doc_name": self._loaded_doc}

    def unload(self):
        self._loaded_doc = None
        self._doc_text = ""
        self._doc_token_count = 0
        self._conversation_history = []
        self._system_prompt = ""
        self._is_active = False

    def get_status(self):
        return {"active": self._is_active, "document": self._loaded_doc, "tokens": self._doc_token_count, "conversation_turns": len(self._conversation_history) // 2, "available_ram_gb": round(get_available_ram_gb(), 1)}
