"""OpenRouter provider - Anthropic-compatible native transport."""

from config.provider_catalog import OPENROUTER_DEFAULT_BASE

from .client import OpenRouterProvider

__all__ = ["OPENROUTER_DEFAULT_BASE", "OpenRouterProvider"]
