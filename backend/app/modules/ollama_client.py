"""
OllamaClient – Hardware-agnostic wrapper for the Ollama REST API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import aiohttp
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app import config

logger = logging.getLogger(__name__)

# Use this for /api/embed when config points at a VLM that doesn't support embeddings.
_EMBEDDING_FALLBACK_MODEL = "aksh-desai/qwen3-vl-embedding:8b-q4_k_m"


def _resolve_embedding_model(requested: str) -> str:
    """Use an actual embedding model; if requested looks like a VLM (e.g. qwen3-vl:8b), use fallback."""
    r = (requested or "").strip()
    if "qwen3-vl" in r.lower() and "embedding" not in r.lower():
        return _EMBEDDING_FALLBACK_MODEL
    return r or _EMBEDDING_FALLBACK_MODEL


def _is_local_ollama_base(base_url: str) -> bool:
    base = base_url.rstrip("/").lower()
    return base.startswith("http://127.0.0.1:11434") or base.startswith("http://localhost:11434")


class OllamaClient:
    """Synchronous Ollama REST API wrapper with retry logic."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.LLM_MODEL,
        timeout: int | None = config.OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout  # None = no timeout
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.embed_url = f"{self.base_url}/api/embed"
        self.rerank_url = f"{self.base_url}/api/rerank"

    def is_alive(self) -> bool:
        urls_to_try = [self.base_url]
        # If default was localhost, also try 127.0.0.1 (often fixes Windows)
        if "localhost" in self.base_url.lower():
            fallback = self.base_url.replace("localhost", "127.0.0.1")
            if fallback not in urls_to_try:
                urls_to_try.append(fallback)
        for base in urls_to_try:
            try:
                resp = requests.get(f"{base}/api/tags", timeout=5)
                if resp.status_code == 200:
                    self.base_url = base.rstrip("/")
                    self.generate_url = f"{self.base_url}/api/generate"
                    self.chat_url = f"{self.base_url}/api/chat"
                    self.embed_url = f"{self.base_url}/api/embed"
                    self.rerank_url = f"{self.base_url}/api/rerank"
                    return True
            except Exception as exc:
                logger.debug("Ollama check %s: %s", base, exc)
        return False

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.warning("Failed to list models: %s", exc)
            return []

    def get_model_num_ctx(self, model_name: str) -> int | None:
        """Return num_ctx from Ollama /api/show for the given model, or None if unavailable."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"model": model_name},
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            params = data.get("parameters") or ""
            # parameters is a string like "temperature 0.7\nnum_ctx 2048"
            for line in params.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == "num_ctx":
                    return int(parts[1])
            return None
        except Exception as exc:
            logger.debug("Could not get model num_ctx from /api/show for %s: %s", model_name, exc)
            return None

    @retry(
        stop=stop_after_attempt(config.OLLAMA_MAX_RETRIES),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        format_json: bool = False,
        stream: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
            "options": options or config.REASONING_OPTIONS,
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"
        effective_model = model or self.model
        resp = requests.post(self.generate_url, json=payload, timeout=self.timeout)
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for model {effective_model!r}. "
                f"The model is likely not pulled. Pull it with: ollama pull {effective_model}"
            ) from None
        resp.raise_for_status()
        text = self._extract_generate_text(resp.json())
        # Strip <think>...</think> blocks that reasoning models prepend
        if "<think>" in text.lower():
            text = self._strip_thinking_tags(text)
        return text

    def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        effective_system = system or "Respond with only a single JSON object. No explanation, no prefix, no other text."
        # Try with format:"json" first
        raw = self.generate(prompt, model=model, system=effective_system, options=options, format_json=True)
        if raw.strip():
            parsed = self._parse_json(raw)
            if "raw" not in parsed:
                return parsed
            logger.debug("generate_json (format:json) returned unparseable: %.200s", raw)
        else:
            logger.debug("generate_json (format:json) returned empty response; trying without format constraint")
        # Fallback: generate without format constraint and extract JSON from free-text output
        raw = self.generate(prompt, model=model, system=effective_system, options=options, format_json=False)
        if raw.strip():
            return self._parse_json(raw)
        logger.warning("generate_json: model returned empty response with and without format constraint")
        return {"raw": ""}

    def generate_with_schema(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate JSON constrained by a JSON Schema (Ollama format parameter).
        Ensures valid structure at the logit level when the server supports it.
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": options or config.REASONING_OPTIONS,
        }
        if system:
            payload["system"] = system
        resp = requests.post(self.generate_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        raw = self._extract_generate_text(resp.json())
        if raw.strip():
            parsed = self._parse_json(raw)
            if "raw" not in parsed:
                return parsed
            logger.debug("generate_with_schema: unparseable schema output, falling back to generate_json")
        else:
            logger.warning("generate_with_schema: model returned empty response")
        # Some models return empty output for strict schema constraints; fallback to normal JSON flow.
        return self.generate_json(prompt, model=model, system=system, options=options)

    @retry(
        stop=stop_after_attempt(config.OLLAMA_MAX_RETRIES),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        format_json: bool = False,
        stream: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": options or config.REASONING_OPTIONS,
        }
        if format_json:
            payload["format"] = "json"
        resp = requests.post(self.chat_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    @retry(
        stop=stop_after_attempt(config.OLLAMA_MAX_RETRIES),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def embed(
        self,
        text: str | list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        requested_model = _resolve_embedding_model(model or config.EMBEDDING_MODEL)
        payload = {
            "model": requested_model,
            "input": text if isinstance(text, list) else [text],
        }
        resp = requests.post(self.embed_url, json=payload, timeout=self.timeout)
        if resp.status_code == 404:
            # Some Ollama versions expose the older /api/embeddings endpoint.
            legacy_embed_url = f"{self.base_url}/api/embeddings"
            try:
                legacy_embeddings: list[list[float]] = []
                for item in payload["input"]:
                    legacy_resp = requests.post(
                        legacy_embed_url,
                        json={"model": requested_model, "prompt": item},
                        timeout=self.timeout,
                    )
                    legacy_resp.raise_for_status()
                    emb = legacy_resp.json().get("embedding")
                    if not emb:
                        raise RuntimeError("Legacy /api/embeddings response missing 'embedding'.")
                    legacy_embeddings.append(emb)
                self.embed_url = legacy_embed_url
                logger.warning(
                    "Embed endpoint %s returned 404; switched to legacy endpoint %s.",
                    resp.url,
                    self.embed_url,
                )
                return legacy_embeddings
            except Exception:
                pass
            # 404 can also mean the model is not pulled, or wrong Ollama instance.
            fallback_model = _EMBEDDING_FALLBACK_MODEL
            if requested_model != fallback_model:
                try:
                    fallback_resp = requests.post(
                        self.embed_url,
                        json={"model": fallback_model, "input": payload["input"]},
                        timeout=self.timeout,
                    )
                    fallback_resp.raise_for_status()
                    logger.warning(
                        "Embedding model %s returned 404 at %s; retried successfully with %s.",
                        requested_model,
                        self.embed_url,
                        fallback_model,
                    )
                    return fallback_resp.json().get("embeddings", [])
                except Exception:
                    pass
            raise RuntimeError(
                "Ollama returned 404 Not Found for /api/embed. "
                "This usually means the embedding model is not installed. "
                f"Pull it with: ollama pull {requested_model} "
                f"(or try: ollama pull {fallback_model}). "
                "Also ensure only one Ollama is running (not both Docker and local). "
                f"Request URL: {self.embed_url}."
            ) from None
        if resp.status_code == 501:
            # If a non-embedding model was configured by mistake, retry with the known embedding model.
            fallback_model = _EMBEDDING_FALLBACK_MODEL
            if requested_model != fallback_model:
                fallback_payload = {
                    "model": fallback_model,
                    "input": payload["input"],
                }
                try:
                    fallback_resp = requests.post(self.embed_url, json=fallback_payload, timeout=self.timeout)
                    fallback_resp.raise_for_status()
                    logger.warning(
                        "Embedding model %s returned 501 at %s; retried successfully with %s.",
                        requested_model,
                        self.embed_url,
                        fallback_model,
                    )
                    return fallback_resp.json().get("embeddings", [])
                except Exception:
                    pass
            # If a remote/base URL is misconfigured, prefer local Ollama on the backend host.
            if not _is_local_ollama_base(self.base_url):
                fallback_base = "http://127.0.0.1:11434"
                fallback_embed_url = f"{fallback_base}/api/embed"
                try:
                    fallback_resp = requests.post(fallback_embed_url, json=payload, timeout=self.timeout)
                    fallback_resp.raise_for_status()
                    data = fallback_resp.json()
                    self.base_url = fallback_base
                    self.generate_url = f"{self.base_url}/api/generate"
                    self.chat_url = f"{self.base_url}/api/chat"
                    self.embed_url = f"{self.base_url}/api/embed"
                    logger.warning(
                        "Embedding endpoint returned 501 at %s; switched Ollama base URL to %s.",
                        resp.url,
                        self.base_url,
                    )
                    return data.get("embeddings", [])
                except Exception:
                    pass
            raise RuntimeError(
                "Ollama returned 501 Not Implemented for /api/embed. "
                "The server that answered does not support this endpoint. "
                f"Embedding model: {requested_model}. "
                f"Request URL: {self.embed_url}. "
                f"Response preview: {(resp.text or '')[:200]}. "
                "Check: (1) OLLAMA_BASE_URL—if the backend runs in Docker or on another host, "
                "point it to the machine where Ollama and the embedding model run; "
                "(2) On that machine, upgrade Ollama (older versions may not implement /api/embed)."
            ) from None
        resp.raise_for_status()
        return resp.json().get("embeddings", [])

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        top_n: int | None = None,
    ) -> list[tuple[str, float]] | None:
        """
        Rerank documents by relevance to the query using Ollama's /api/rerank.
        Returns list of (document_text, relevance_score) in descending score order,
        or None if the endpoint is unavailable (404/501) or on error.
        """
        if not documents or not (query or "").strip():
            return []
        model = (model or getattr(config, "RERANKER_MODEL", "") or "").strip()
        if not model:
            logger.debug("rerank: no RERANKER_MODEL configured")
            return None
        top_n = top_n if top_n is not None else min(len(documents), getattr(config, "TOP_K_RESULTS", 5))
        documents = [str(d) for d in documents]
        payload = {
            "model": model,
            "query": query.strip(),
            "documents": [d[:32000] for d in documents],  # truncate per-doc for API limits
            "top_n": top_n,
        }
        try:
            resp = requests.post(
                self.rerank_url,
                json=payload,
                timeout=self.timeout or 120,
            )
        except Exception as e:
            logger.warning("rerank: request failed: %s", e)
            return None
        if resp.status_code == 404:
            logger.debug("rerank: /api/rerank not found (Ollama may not support rerank)")
            return None
        if resp.status_code == 501:
            logger.debug("rerank: model %s returned 501 (not a reranker model?)", model)
            return None
        if resp.status_code != 200:
            logger.warning("rerank: %s %s", resp.status_code, (resp.text or "")[:200])
            return None
        try:
            data = resp.json()
        except Exception as e:
            logger.warning("rerank: invalid JSON: %s", e)
            return None
        results = data.get("results")
        if not isinstance(results, list):
            return None
        out: list[tuple[str, float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            doc = item.get("document", "")
            score = item.get("relevance_score")
            if score is not None:
                try:
                    out.append((doc, float(score)))
                except (TypeError, ValueError):
                    pass
        return out if out else None

    @staticmethod
    def _extract_balanced(raw: str, open_char: str, close_char: str) -> str | None:
        """Extract the first balanced open_char..close_char span (e.g. {...})."""
        start = raw.find(open_char)
        if start == -1:
            return None
        depth = 0
        i = start
        while i < len(raw):
            if raw[i] == open_char:
                depth += 1
            elif raw[i] == close_char:
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
            i += 1
        return None

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove <think>...</think> blocks that reasoning models prepend."""
        return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    @staticmethod
    def _extract_generate_text(data: dict[str, Any]) -> str:
        """
        Normalize generated text across Ollama variants.
        Primary field is `response`; some server/model combos return message/content shaped payloads.
        """
        text = data.get("response", "")
        if isinstance(text, str) and text:
            return text
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str) and content:
                return content
        content = data.get("content", "")
        if isinstance(content, str) and content:
            return content
        output_text = data.get("output_text", "")
        if isinstance(output_text, str):
            return output_text
        return ""

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        # Strip <think>...</think> reasoning blocks before attempting parse
        raw = OllamaClient._strip_thinking_tags(raw)
        if not raw:
            logger.warning("Model returned empty response after stripping think tags")
            return {"raw": ""}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        if "```" in raw:
            for block in raw.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
        # Extract first balanced {...} so leading prose (e.g. "The user asks: ...") is skipped
        for open_c, close_c in [("{", "}"), ("[", "]")]:
            span = OllamaClient._extract_balanced(raw, open_c, close_c)
            if span:
                try:
                    return json.loads(span)
                except json.JSONDecodeError:
                    pass
        # Legacy: try first { to last }
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = raw.find(start_char)
            end = raw.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
        logger.warning("Failed to parse JSON from model output: %s", raw[:300])
        return {"raw": raw}


class AsyncOllamaClient:
    """Async Ollama wrapper for high-throughput ingestion pipelines."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.LLM_MODEL,
        timeout: int | None = config.OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)  # None = no timeout
        self.generate_url = f"{self.base_url}/api/generate"
        self.embed_url = f"{self.base_url}/api/embed"

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        format_json: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": options or config.REASONING_OPTIONS,
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"
        effective_model = model or self.model
        for attempt in range(config.OLLAMA_MAX_RETRIES):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(self.generate_url, json=payload) as resp:
                        if resp.status == 404:
                            raise RuntimeError(
                                f"Ollama returned 404 for model {effective_model!r}. "
                                f"The model is likely not pulled. Pull it with: ollama pull {effective_model}"
                            ) from None
                        resp.raise_for_status()
                        data = await resp.json()
                        text = OllamaClient._extract_generate_text(data)
                        if "<think>" in text.lower():
                            text = OllamaClient._strip_thinking_tags(text)
                        return text
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Async generate attempt %d failed: %s", attempt + 1, exc)
                if attempt < config.OLLAMA_MAX_RETRIES - 1:
                    await asyncio.sleep(config.OLLAMA_RETRY_DELAY * (attempt + 1))
                else:
                    raise
        return ""

    async def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        effective_system = system or "Respond with only a single JSON object. No explanation, no prefix, no other text."
        raw = await self.generate(prompt, model=model, system=effective_system, options=options, format_json=True)
        if raw.strip():
            parsed = OllamaClient._parse_json(raw)
            if "raw" not in parsed:
                return parsed
            logger.debug("async generate_json (format:json) returned unparseable: %.200s", raw)
        else:
            logger.debug("async generate_json (format:json) returned empty; trying without format constraint")
        raw = await self.generate(prompt, model=model, system=effective_system, options=options, format_json=False)
        if raw.strip():
            return OllamaClient._parse_json(raw)
        logger.warning("async generate_json: model returned empty with and without format constraint")
        return {"raw": ""}

    async def embed(
        self,
        text: str | list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        requested_model = _resolve_embedding_model(model or config.EMBEDDING_MODEL)
        payload = {
            "model": requested_model,
            "input": text if isinstance(text, list) else [text],
        }
        for attempt in range(config.OLLAMA_MAX_RETRIES):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    requested_embed_url = self.embed_url
                    async with session.post(requested_embed_url, json=payload) as resp:
                        if resp.status == 404:
                            # Some Ollama versions expose the older /api/embeddings endpoint.
                            legacy_embed_url = f"{self.base_url}/api/embeddings"
                            try:
                                legacy_embeddings: list[list[float]] = []
                                for item in payload["input"]:
                                    async with session.post(
                                        legacy_embed_url,
                                        json={"model": requested_model, "prompt": item},
                                    ) as legacy_resp:
                                        legacy_resp.raise_for_status()
                                        legacy_data = await legacy_resp.json()
                                        emb = legacy_data.get("embedding")
                                        if not emb:
                                            raise RuntimeError("Legacy /api/embeddings response missing 'embedding'.")
                                        legacy_embeddings.append(emb)
                                self.embed_url = legacy_embed_url
                                logger.warning(
                                    "Embed endpoint %s returned 404; switched to legacy endpoint %s.",
                                    requested_embed_url,
                                    self.embed_url,
                                )
                                return legacy_embeddings
                            except Exception:
                                pass
                            fallback_model = _EMBEDDING_FALLBACK_MODEL
                            if requested_model != fallback_model:
                                try:
                                    async with session.post(
                                        requested_embed_url,
                                        json={"model": fallback_model, "input": payload["input"]},
                                    ) as fallback_resp:
                                        fallback_resp.raise_for_status()
                                        data = await fallback_resp.json()
                                        logger.warning(
                                            "Embedding model %s returned 404 at %s; retried successfully with %s.",
                                            requested_model,
                                            requested_embed_url,
                                            fallback_model,
                                        )
                                        return data.get("embeddings", [])
                                except Exception:
                                    pass
                            resp_preview = (await resp.text())[:200]
                            raise RuntimeError(
                                "Ollama returned 404 Not Found for /api/embed. "
                                "This usually means the embedding model is not installed. "
                                f"Pull it with: ollama pull {requested_model} "
                                f"(or try: ollama pull {fallback_model}). "
                                "Also ensure only one Ollama is running (not both Docker and local). "
                                f"Request URL: {requested_embed_url}. Response: {resp_preview}"
                            ) from None
                        if resp.status == 501:
                            # If a non-embedding model was configured by mistake, retry with the known embedding model.
                            fallback_model = _EMBEDDING_FALLBACK_MODEL
                            if requested_model != fallback_model:
                                fallback_payload = {
                                    "model": fallback_model,
                                    "input": payload["input"],
                                }
                                try:
                                    async with session.post(requested_embed_url, json=fallback_payload) as fallback_resp:
                                        fallback_resp.raise_for_status()
                                        data = await fallback_resp.json()
                                        logger.warning(
                                            "Embedding model %s returned 501 at %s; retried successfully with %s.",
                                            requested_model,
                                            requested_embed_url,
                                            fallback_model,
                                        )
                                        return data.get("embeddings", [])
                                except Exception:
                                    pass
                            # If a remote/base URL is misconfigured, prefer local Ollama on the backend host.
                            if not _is_local_ollama_base(self.base_url):
                                fallback_base = "http://127.0.0.1:11434"
                                fallback_embed_url = f"{fallback_base}/api/embed"
                                try:
                                    async with session.post(fallback_embed_url, json=payload) as fallback_resp:
                                        fallback_resp.raise_for_status()
                                        data = await fallback_resp.json()
                                        self.base_url = fallback_base
                                        self.generate_url = f"{self.base_url}/api/generate"
                                        self.embed_url = f"{self.base_url}/api/embed"
                                        logger.warning(
                                            "Embedding endpoint returned 501 at %s; switched Ollama base URL to %s.",
                                            requested_embed_url,
                                            self.base_url,
                                        )
                                        return data.get("embeddings", [])
                                except Exception:
                                    pass
                            resp_preview = (await resp.text())[:200]
                            raise RuntimeError(
                                "Ollama returned 501 Not Implemented for /api/embed. "
                                "The server that answered does not support this endpoint. "
                                f"Embedding model: {requested_model}. "
                                f"Request URL: {self.embed_url}. "
                                f"Response preview: {resp_preview}. "
                                "Check: (1) OLLAMA_BASE_URL—if the backend runs in Docker or on another host, "
                                "point it to the machine where Ollama and the embedding model run; "
                                "(2) On that machine, upgrade Ollama (older versions may not implement /api/embed)."
                            )
                        resp.raise_for_status()
                        data = await resp.json()
                        return data.get("embeddings", [])
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Async embed attempt %d failed: %s", attempt + 1, exc)
                if attempt < config.OLLAMA_MAX_RETRIES - 1:
                    await asyncio.sleep(config.OLLAMA_RETRY_DELAY * (attempt + 1))
                else:
                    raise
        return []
