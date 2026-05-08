"""OpenAI-compatible generic provider.

Supports any OpenAI Chat Completions API-compatible endpoint (LiteLLM, vLLM,
Together AI, custom deployments). Model names are passed upstream as-is.
"""

from providers.defaults import OPENAI_COMPATIBLE_DEFAULT_BASE

from .client import OpenAICompatibleProvider

__all__ = ["OPENAI_COMPATIBLE_DEFAULT_BASE", "OpenAICompatibleProvider"]
