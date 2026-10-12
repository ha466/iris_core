"""
llm_client.py - Streaming client for llama-server OpenAI-compatible SSE chat completions.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Dict, List

import httpx

import config

log = logging.getLogger("llm_client")


class LlamaLLMClient:
    """Consumes SSE streaming tokens from local llama-server."""

    def __init__(self, base_url: str = config.LLAMA_BASE_URL):
        self.endpoint = f"{base_url}/chat/completions"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = getattr(config, "LLAMA_MAX_TOKENS", 512),
        temperature: float = getattr(config, "LLAMA_TEMPERATURE", 1.0),
        top_p: float = getattr(config, "LLAMA_TOP_P", 0.95),
        top_k: int = getattr(config, "LLAMA_TOP_K", 20),
    ) -> AsyncIterator[str]:
        """
        Stream tokens from llama-server.
        Yields text deltas as they arrive.
        """
        payload = {
            "model": getattr(config, "LLAMA_MODEL_NAME", "Ling-3.0-tiny"),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
            "chat_template_kwargs": {
                "enable_thinking": getattr(config, "LLAMA_THINKING_ENABLED", False),
            },
        }

        try:
            async with self.client.stream("POST", self.endpoint, json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    log.error(f"llama-server error {response.status_code}: {err_body.decode('utf-8', errors='ignore')}")
                    return

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
            log.error(f"Network error communicating with llama-server: {e}")

    async def close(self) -> None:
        await self.client.aclose()
