"""Request builder for openai_compatible provider."""

from __future__ import annotations

from typing import Any

from core.anthropic import ReasoningReplayMode, build_base_request_body
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request.

    Converts Anthropic message format to OpenAI chat format, handling
    the thinking/reasoning_content field based on the thinking_enabled flag.
    """
    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=(
                ReasoningReplayMode.REASONING_CONTENT
                if thinking_enabled
                else ReasoningReplayMode.DISABLED
            ),
        )
    except Exception as exc:
        raise InvalidRequestError(f"Request conversion failed: {exc}") from exc

    # Pass through any extra_body for provider-specific parameters (e.g. LiteLLM extras)
    extra_body = getattr(request_data, "extra_body", None)
    if extra_body:
        body["extra_body"] = extra_body

    return body
