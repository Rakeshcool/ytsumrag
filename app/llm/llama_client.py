"""LLM generation client — communicates with the generation llama-server."""

from __future__ import annotations

import logging

import httpx

from app.config.settings import settings
from app.model_servers.manager import model_server_manager

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


class LlamaClient:
    """Async client for the local generation llama-server (OpenAI-compatible API)."""

    def __init__(self) -> None:
        self._base_url = settings.generation_base_url

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant content."""
        await model_server_manager.ensure_generation_server()

        payload = {
            "model": "generation",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        for attempt in range(_MAX_RETRIES):
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()

            result = resp.json()
            content = result["choices"][0]["message"]["content"]

            if content and content.strip():
                return content

            # Empty content — retry
            logger.warning(
                "LLM returned empty content (attempt %d/%d), retrying...",
                attempt + 1,
                _MAX_RETRIES,
            )

        # All retries exhausted
        logger.error("LLM returned empty content after %d attempts", _MAX_RETRIES)
        return ""

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Convenience method: build messages list and call chat_completion."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.chat_completion(messages, **kwargs)


llama_client = LlamaClient()
