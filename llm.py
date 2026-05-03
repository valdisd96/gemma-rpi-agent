"""Async client for llama.cpp's OpenAI-compatible HTTP server.

Entry points:
  * `stream_chat(messages)` — async generator yielding token deltas.
  * `chat(messages)` — non-streaming one-shot.
  * `health()` — short status string for /status.
  * `modalities()` — "text+vision+audio" string from /props.
  * `bench()` — tiny prompt → "<chars> in <s>s (~<rate> tok/s)".

Messages may be plain text or OpenAI-style content arrays (built in media.py)
— the wire format is opaque to this module. SSE / JSON parsing is factored
into pure helpers for testability.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Awaitable, Callable

import httpx


LLAMA_BASE = "http://127.0.0.1:8080"
LLAMA_URL = f"{LLAMA_BASE}/v1/chat/completions"
HEALTH_URL = f"{LLAMA_BASE}/health"
PROPS_URL = f"{LLAMA_BASE}/props"

# Vision/audio prompt processing on a Pi can take many minutes, so the long
# tail covers a worst-case image+text turn under -c 2048.
DEFAULT_TIMEOUT = 1200.0  # 20 minutes

_DONE = "[DONE]"


def _parse_sse_delta(raw: str) -> str | None:
    """Return the content delta from one SSE line, or None to skip/terminate."""
    if not raw.startswith("data:"):
        return None
    payload = raw[5:].strip()
    if not payload or payload == _DONE:
        return None
    try:
        chunk = json.loads(payload)
        return chunk["choices"][0]["delta"].get("content") or None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def _parse_completion(payload: dict) -> str:
    return payload["choices"][0]["message"].get("content", "")


def _format_modalities(mods: dict | None) -> str:
    parts = ["text"]
    mods = mods or {}
    if mods.get("vision"):
        parts.append("vision")
    if mods.get("audio"):
        parts.append("audio")
    return "+".join(parts)


async def stream_chat(
    messages: list[dict],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncIterator[str]:
    """Stream token deltas as they arrive from the server."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            LLAMA_URL,
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                delta = _parse_sse_delta(raw)
                if delta:
                    yield delta


async def chat(
    messages: list[dict],
    *,
    max_tokens: int = 256,
    temperature: float = 0.8,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Return the full assistant reply as a single string."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            LLAMA_URL,
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        r.raise_for_status()
        return _parse_completion(r.json())


async def health() -> str:
    """Return the server's self-reported status, or an error string."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(HEALTH_URL)
            return r.json().get("status", "unknown")
    except Exception as e:  # noqa: BLE001 — surface verbatim to the user
        return f"unreachable ({e})"


async def modalities() -> str:
    """Return e.g. 'text+vision+audio' reflecting /props.modalities."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(PROPS_URL)
            return _format_modalities(r.json().get("modalities"))
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


_BENCH_PROMPT = [{"role": "user", "content": "Reply with one short sentence."}]


async def bench(
    *,
    chat_fn: Callable[..., Awaitable[str]] | None = None,
    timeout: float = 30.0,
    now: Callable[[], float] = time.monotonic,
) -> str:
    """Run a tiny prompt through the model and return a one-line perf summary."""
    if chat_fn is None:
        chat_fn = chat
    t0 = now()
    try:
        text = await asyncio.wait_for(
            chat_fn(_BENCH_PROMPT, max_tokens=32, temperature=0.2),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return "model not responding"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}"
    elapsed = now() - t0
    chars = len(text)
    tok_per_s = (chars / 4) / elapsed if elapsed > 0 else 0.0
    return f"{chars} chars in {elapsed:.1f}s (~{tok_per_s:.1f} tok/s)"
