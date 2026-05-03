"""Build multimodal LLM message content + transcode Telegram voice notes.

llama.cpp's /v1/chat/completions accepts OpenAI-style content arrays where each
item is `{"type": "text" | "image_url" | "input_audio", ...}`. The helpers here
return those arrays so handlers in bot.py stay thin.

Voice notes from Telegram arrive as OGG/Opus; the audio mmproj wants WAV, so
`transcode_to_wav` shells out to ffmpeg. ffmpeg is treated as a runtime
dependency — the bot reports a clear error if it's not on PATH.
"""

from __future__ import annotations

import base64
import shutil
import subprocess


class TranscodeError(RuntimeError):
    """ffmpeg missing or returned non-zero while decoding the voice note."""


def encode_data_url(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def user_with_image(text: str, image_bytes: bytes, *, mime: str = "image/jpeg") -> list[dict]:
    """Content array for a chat message carrying one image plus a text prompt."""
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": encode_data_url(image_bytes, mime)}},
    ]


def user_with_audio(text: str, audio_bytes: bytes, *, fmt: str = "wav") -> list[dict]:
    """Content array for a chat message carrying audio + an optional caption."""
    parts: list[dict] = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.append(
        {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "format": fmt,
            },
        }
    )
    return parts


def transcode_to_wav(
    input_bytes: bytes,
    *,
    ffmpeg: str = "ffmpeg",
    runner: object = subprocess,
) -> bytes:
    """Decode arbitrary audio bytes (OGG/Opus, etc.) to 16 kHz mono WAV.

    `runner` is injected for tests — must expose a `run(...)` matching subprocess.
    """
    if shutil.which(ffmpeg) is None:
        raise TranscodeError(
            f"{ffmpeg} not found in PATH; install with `apt install ffmpeg`"
        )
    cmd = [
        ffmpeg, "-loglevel", "error",
        "-i", "pipe:0",
        "-ac", "1", "-ar", "16000",
        "-f", "wav", "pipe:1",
    ]
    try:
        proc = runner.run(cmd, input=input_bytes, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise TranscodeError(str(e)) from e
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode("utf-8", errors="replace").strip() or "ffmpeg failed"
        raise TranscodeError(msg) from e
    return proc.stdout
