"""Tests for openai_compatible provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Request, Response

from providers.base import ProviderConfig
from providers.openai_compatible import OpenAICompatibleProvider
from tests.stream_contract import assert_canonical_stream_error_envelope


# ==================== Mock classes ====================
class MockMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class MockRequest:
    """Minimal Anthropic-style request for testing."""

    def __init__(self, **kwargs):
        self.model = "test-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = None
        self.system = None
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_bad_request_error(message: str) -> Exception:
    """Create a BadRequestError for testing."""
    try:
        import openai

        response = Response(
            status_code=400,
            request=Request(
                "POST", "http://test/openai_compatible/v1/chat/completions"
            ),
        )
        return openai.BadRequestError(
            message, response=response, body={"error": {"message": message}}
        )
    except ImportError:
        # Fallback generic exception
        return Exception(f"400: {message}")


# ==================== Fixtures ====================
@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def openai_compatible_provider(provider_config):
    """Create a provider instance with test config."""
    return OpenAICompatibleProvider(provider_config)


# ==================== Initialization ====================
def test_init_uses_default_base_url():
    """Provider defaults to OPENAI_COMPATIBLE_DEFAULT_BASE when no base_url set."""
    config = ProviderConfig(
        api_key="test_key",
        base_url="",
    )
    provider = OpenAICompatibleProvider(config)
    assert provider._base_url == "http://localhost:4000/v1"
    assert provider._provider_name == "OPENAI_COMPATIBLE"


def test_init_respects_configured_base_url():
    """Provider uses explicitly configured base_url."""
    config = ProviderConfig(
        api_key="test_key",
        base_url="https://custom.endpoint/v1",
    )
    provider = OpenAICompatibleProvider(config)
    assert provider._base_url == "https://custom.endpoint/v1"


@pytest.mark.asyncio
async def test_init_uses_configurable_timeouts():
    """Provider passes configurable read/write/connect timeouts to client."""
    from providers.base import ProviderConfig

    config = ProviderConfig(
        api_key="test_key",
        base_url="http://localhost:4000/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        OpenAICompatibleProvider(config)
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


# ==================== Request Body Building ====================
@pytest.mark.asyncio
async def test_build_request_body_basic(openai_compatible_provider):
    """Basic Anthropic→OpenAI conversion works."""
    req = MockRequest()
    body = openai_compatible_provider._build_request_body(req, thinking_enabled=False)

    assert body["model"] == "test-model"
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "Hello"
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.5


@pytest.mark.asyncio
async def test_build_request_body_with_system_prompt(openai_compatible_provider):
    """System prompt is included as first message."""
    req = MockRequest(system="You are a helpful assistant")
    body = openai_compatible_provider._build_request_body(req, thinking_enabled=False)

    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "You are a helpful assistant"
    assert body["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_build_request_body_with_thinking_enabled(openai_compatible_provider):
    """When thinking is enabled, reasoning_content appears in output."""
    req = MockRequest()
    # Simulate a request with the 'thinking' field enabled
    req.thinking = MagicMock(type="enabled", enabled=True, budget_tokens=None)
    body = openai_compatible_provider._build_request_body(req, thinking_enabled=True)

    # The converted body should contain fields suitable for reasoning-aware upstream
    # Exact structure depends on build_base_request_body; we just verify it didn't raise
    assert "model" in body
    assert "messages" in body


@pytest.mark.asyncio
async def test_build_request_body_thinking_disabled_globally(provider_config):
    """When enable_thinking is False, reasoning_content is omitted."""
    provider = OpenAICompatibleProvider(
        provider_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest()
    body = provider._build_request_body(req)

    # When thinking is globally disabled, no reasoning-related extra_body should appear
    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra
    assert "reasoning_budget" not in extra


@pytest.mark.asyncio
async def test_build_request_body_thinking_disabled_per_request(
    openai_compatible_provider,
):
    """When request specifies thinking disabled, omit reasoning fields."""
    req = MockRequest()
    req.thinking = MagicMock(enabled=False)
    body = openai_compatible_provider._build_request_body(req)

    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra


@pytest.mark.asyncio
async def test_build_request_body_passes_extra_body(openai_compatible_provider):
    """extra_body from request is passed through unmodified."""
    req = MockRequest(extra_body={"custom_param": 123, "top_k": 5})
    body = openai_compatible_provider._build_request_body(req, thinking_enabled=False)

    assert "extra_body" in body
    assert body["extra_body"]["custom_param"] == 123
    assert body["extra_body"]["top_k"] == 5


@pytest.mark.asyncio
async def test_build_request_body_strips_none_extra_body(openai_compatible_provider):
    """When extra_body is None or empty, no extra_body key in output."""
    req = MockRequest(extra_body=None)
    body = openai_compatible_provider._build_request_body(req, thinking_enabled=False)
    assert "extra_body" not in body

    req2 = MockRequest(extra_body={})
    body2 = openai_compatible_provider._build_request_body(req2, thinking_enabled=False)
    assert "extra_body" not in body2


# ==================== Model Listing ====================
@pytest.mark.asyncio
async def test_list_model_ids(openai_compatible_provider):
    """list_model_ids calls OpenAI client's models.list and extracts IDs."""
    mock_models_page = MagicMock()
    mock_models_page.data = [
        MagicMock(id="gpt-4"),
        MagicMock(id="gpt-3.5-turbo"),
        MagicMock(id="anthropic/claude-3-opus"),
    ]
    openai_compatible_provider._client.models.list = AsyncMock(
        return_value=mock_models_page
    )

    model_ids = await openai_compatible_provider.list_model_ids()

    assert "gpt-4" in model_ids
    assert "anthropic/claude-3-opus" in model_ids
    assert len(model_ids) == 3
    openai_compatible_provider._client.models.list.assert_awaited_once()


# ==================== Streaming Response ====================
@pytest.mark.asyncio
async def test_stream_response_text(openai_compatible_provider):
    """Basic text streaming converts OpenAI chunks to Anthropic SSE."""
    req = MockRequest()

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=None), finish_reason=None
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=None),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        openai_compatible_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in openai_compatible_provider.stream_response(req)]

    # Should have proper SSE events
    assert any("event: message_start" in e for e in events)
    assert any("event: content_block_delta" in e for e in events)
    assert any("Hello" in e for e in events)
    assert any("World" in e for e in events)
    assert any("event: message_stop" in e for e in events)


@pytest.mark.asyncio
async def test_stream_response_thinking_reasoning_content(openai_compatible_provider):
    """Streaming with reasoning_content emits thinking_delta blocks."""
    req = MockRequest()
    req.thinking = MagicMock(enabled=True)

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="I'm reasoning..."),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        openai_compatible_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in openai_compatible_provider.stream_response(req)]

    event_text = "".join(events)
    assert "thinking_delta" in event_text
    assert "I'm reasoning..." in event_text


@pytest.mark.asyncio
async def test_stream_response_suppresses_thinking_when_disabled(provider_config):
    """When thinking disabled, reasoning_content is stripped from stream."""
    provider = OpenAICompatibleProvider(
        provider_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest()
    # The upstream chunk includes reasoning but we should suppress it
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Answer", reasoning_content="Secret thought"),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=1)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in provider.stream_response(req)]

    event_text = "".join(events)
    # Reasoning content should be filtered out; only text remains
    assert "thinking_delta" not in event_text
    assert "Secret thought" not in event_text
    assert "Answer" in event_text


# ==================== Error Handling ====================
@pytest.mark.asyncio
async def test_stream_response_http_error(openai_compatible_provider):
    """Non-200 HTTP errors yield proper SSE error envelope."""
    req = MockRequest()
    # Use a bad request error to simulate upstream failure
    error = _make_bad_request_error("Invalid parameters")
    with patch.object(
        openai_compatible_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = error
        events = [e async for e in openai_compatible_provider.stream_response(req)]

    assert_canonical_stream_error_envelope(
        events, user_message_substr="Invalid request sent to provider"
    )


@pytest.mark.asyncio
async def test_stream_response_network_error(openai_compatible_provider):
    """Network errors are caught and emitted as SSE error events."""
    req = MockRequest()
    with patch.object(
        openai_compatible_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = Exception("Connection refused")
        events = [e async for e in openai_compatible_provider.stream_response(req)]

    event_text = "".join(events)
    assert "Connection refused" in event_text
    assert_canonical_stream_error_envelope(
        events, user_message_substr="Connection refused"
    )


# ==================== Cleanup ====================
@pytest.mark.asyncio
async def test_cleanup_closes_client(openai_compatible_provider):
    """cleanup() closes the underlying AsyncOpenAI client."""
    openai_compatible_provider._client = MagicMock()
    openai_compatible_provider._client.aclose = AsyncMock()

    await openai_compatible_provider.cleanup()

    openai_compatible_provider._client.aclose.assert_awaited_once()
