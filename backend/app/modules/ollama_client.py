"""
OllamaClient – Hardware-agnostic wrapper for the Ollama REST API.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
        resp = requests.post(self.generate_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = self.generate(prompt, model=model, system=system, options=options, format_json=True)
        return self._parse_json(raw)

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
        payload = {
            "model": model or config.EMBEDDING_MODEL,
            "input": text if isinstance(text, list) else [text],
        }
        resp = requests.post(self.embed_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("embeddings", [])

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        raw = raw.strip()
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
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = raw.find(start_char)
            end = raw.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
        logger.warning("Failed to parse JSON from model output: %s", raw[:200])
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
        for attempt in range(config.OLLAMA_MAX_RETRIES):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(self.generate_url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data.get("response", "")
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
        raw = await self.generate(prompt, model=model, system=system, options=options, format_json=True)
        return OllamaClient._parse_json(raw)

    async def embed(
        self,
        text: str | list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        payload = {
            "model": model or config.EMBEDDING_MODEL,
            "input": text if isinstance(text, list) else [text],
        }
        for attempt in range(config.OLLAMA_MAX_RETRIES):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(self.embed_url, json=payload) as resp:
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
