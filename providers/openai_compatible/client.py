"""OpenAI-compatible generic provider implementation."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import OPENAI_COMPATIBLE_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class OpenAICompatibleProvider(OpenAIChatTransport):
    """Generic provider for any OpenAI-compatible chat completions API.

    Works with LiteLLM proxy, vLLM, Together AI, and custom OpenAI-compatible servers.
    Model names are passed upstream exactly as configured (e.g. 'gpt-4', 'anthropic/claude-3-opus').
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OPENAI_COMPATIBLE",
            base_url=config.base_url or OPENAI_COMPATIBLE_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build OpenAI-format request body from Anthropic request."""
        thinking = self._is_thinking_enabled(request, thinking_enabled)
        return build_request_body(request, thinking_enabled=thinking)
