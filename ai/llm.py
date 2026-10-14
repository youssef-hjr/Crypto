"""
The connection to Claude, behind a small interface so everything else can
be tested offline.

`AnthropicLLM` calls the Messages API with the official SDK. The API key is
read by the SDK from the ANTHROPIC_API_KEY environment variable; this code
never handles it. `ScriptedLLM` returns canned answers for tests and dry
runs, and never touches the network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol, Union

from config.settings import AISettings

# Claude Opus 5.5 list prices, USD per million tokens (cache reads are 0.1x
# input, cache writes 1.25x). Used only for the cost estimate in the log.
PRICES = {"claude-opus-5-5": {"input": 4.0, "output": 20.0}}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class LLMResponse:
    text: str  # The JSON answer, or "" if there is none
    stop_reason: Optional[str]
    model: str
    request_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    refusal_category: Optional[str] = None

    def cost_usd(self) -> Optional[float]:
        price = next((p for m, p in PRICES.items() if self.model.startswith(m)), None)
        if price is None:
            return None
        return round((self.input_tokens * price["input"]
                      + self.cache_read_tokens * price["input"] * 0.1
                      + self.cache_write_tokens * price["input"] * 1.25
                      + self.output_tokens * price["output"]) / 1e6, 6)


class LLMClient(Protocol):
    async def complete(self, system: str, user: str, schema: dict) -> LLMResponse:
        ...


class MissingAPIKey(RuntimeError):
    pass


class AnthropicLLM:

    def __init__(self, settings: AISettings):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise MissingAPIKey(
                "ANTHROPIC_API_KEY is not set. Put it in the .env file in the project folder "
                "(see CHANGES.md), or run without --ai.")
        import anthropic  # Only needed when Claude is actually used

        self._settings = settings
        # The SDK reads ANTHROPIC_API_KEY itself. Our own timeout wraps each
        # call, so the SDK's retries stay small.
        self._client = anthropic.AsyncAnthropic(max_retries=1, timeout=settings.timeout_seconds)

    async def complete(self, system: str, user: str, schema: dict) -> LLMResponse:
        s = self._settings
        extra = {}
        if s.server_fallback:
            # If a safety classifier declines, the API re-runs the request on
            # Anthropic's recommended fallback model inside the same call.
            extra = {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        msg = await self._client.beta.messages.create(
            model=s.model,
            max_tokens=s.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": s.effort,
                           "format": {"type": "json_schema", "schema": schema}},
            cache_control={"type": "ephemeral"},
            **extra,
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        u = msg.usage
        details = getattr(msg, "stop_details", None)
        return LLMResponse(
            text=text, stop_reason=msg.stop_reason, model=msg.model,
            request_id=getattr(msg, "_request_id", None),
            input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            refusal_category=getattr(details, "category", None) if details else None,
        )


@dataclass
class ScriptedLLM:
    """Offline stand-in. Each call takes the next scripted item: an
    LLMResponse, a JSON string (wrapped as a normal answer), or an
    exception to raise. Every prompt is kept in `calls` for inspection."""
    script: list[Union[LLMResponse, str, BaseException]] = field(default_factory=list)
    default: Optional[Union[LLMResponse, str]] = None
    calls: list[dict] = field(default_factory=list)

    async def complete(self, system: str, user: str, schema: dict) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "schema": schema})
        item = self.script.pop(0) if self.script else self.default
        if item is None:
            raise RuntimeError("ScriptedLLM has no answer left")
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return LLMResponse(text=item, stop_reason="end_turn", model="scripted",
                               input_tokens=1000, output_tokens=200)
        return item
