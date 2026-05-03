# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A multimodal Telegram bot that streams responses from a local **Gemma 4 E2B-It** model running via **llama.cpp**'s OpenAI-compatible HTTP server on `http://127.0.0.1:8080`. Users can send:

- **Text** — streamed back via placeholder message edits.
- **Photo** — `image_url` data-URL attached to the user turn; reply streams the same way.
- **Voice note** — Telegram OGG/Opus is transcoded to 16 kHz mono WAV via ffmpeg, then sent as `input_audio` to the model.

Conversation history is kept per-chat in memory (system + user + assistant turns) and rotated on `/clear`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For running tests: pip install -r requirements-dev.txt
cp .env.example .env   # then fill in TELEGRAM_TOKEN
```

System dependencies:

- **llama.cpp** running with the Gemma 4 main GGUF **and** `--mmproj <projector>.gguf` so `/props` reports `"vision": true, "audio": true`. Without `--mmproj` the bot still works for text but rejects photos and voice with a model-side error.
- **ffmpeg** on PATH for voice support (`apt install ffmpeg`). Without it, voice messages return a friendly error.

## Running

```bash
source .venv/bin/activate
python bot.py
```

```bash
source .venv/bin/activate && python -m pytest -q    # run tests
```

## Environment variables (`.env`)

| Variable | Required | Default |
|---|---|---|
| `TELEGRAM_TOKEN` | Yes | — |
| `SYSTEM_PROMPT` | No | `"You are a helpful assistant. Reply concisely in plain text."` |
| `ALLOWED_USER_IDS` | No | empty (allow all) — comma/whitespace-separated Telegram user IDs; if set, other users are silently ignored and logged |

## Bot commands

| Command | Purpose |
|---|---|
| `/start` | Show the welcome / help message. |
| `/help` | Same as `/start`. |
| `/clear` | Reset the chat history (LLM memory) for this chat and rotate the transcript file. |
| `/status` | Host diagnostics (hardware, OS, load, temp, disk free), llama.cpp endpoint/health/modalities, and a short bench. |

Plain text, photos (with optional caption), and voice notes are all routed to the model and answered with streamed replies.

## Architecture

Code is split into focused modules (entrypoint is `bot.py`):

- **`bot.py`** — python-telegram-bot wiring. Command/message handlers, history dict, transcript bookkeeping, streaming-edit core.
- **`llm.py`** — llama.cpp HTTP client. `stream_chat()` + `chat()` accept either plain text content or OpenAI-style content arrays. `health()`, `modalities()`, and `bench()` back `/status`. SSE/completion parsing is factored into pure helpers.
- **`media.py`** — pure helpers for building multimodal content arrays (`user_with_image`, `user_with_audio`, `encode_data_url`) plus `transcode_to_wav` (ffmpeg subprocess for OGG/Opus → 16 kHz mono WAV). `TranscodeError` is raised when ffmpeg is missing or fails.
- **`transcripts.py`** — `TranscriptStore` writes per-chat transcripts to `logs/convs/<chat_id>/NNN.txt`; `start()` rotates on `/clear` and on the first message of a chat.
- **`sysinfo.py`** — pure readers for host diagnostics used by `/status` (hardware, OS, load, temp, disk free). Each reader has an injectable dependency and a safe fallback so /status works off-Pi too.
- **`tests/`** — pytest suite covering SSE/completion parsing, modality formatting, bench paths, content-array shape, ffmpeg wrapper (mocked), transcript rotation/format, and sysinfo readers.

## Logs on disk

- `logs/bot.log` — rotating file log (5 MB × 5 backups), mirrors journald output. `httpx` / `telegram` loggers are pinned to WARNING so the bot token never appears in request URLs.
- `logs/convs/<chat_id>/NNN.txt` — human-readable transcript, one file per `/clear`-bounded conversation, zero-padded sequential numbering per chat. Photo / voice turns are logged with a short tag (e.g. `[photo 12345 bytes] caption…`).
- `logs/` is git-ignored.

## Key constants

- `bot.py`: `EDIT_INTERVAL = 2.0` (seconds between stream edits), `MAX_MSG_LEN = 4000` (per-message cap; long responses spill into additional messages).
- `llm.py`: `LLAMA_URL`, `DEFAULT_TIMEOUT = 1200.0` (vision/audio prompt processing on a Pi can take many minutes).

## Performance notes

On a Raspberry Pi 4 with the Q4_0 model and `mmproj-BF16`:

- **Text**: ~2 tok/s generation; first token within a couple of seconds.
- **Photo**: vision encoding alone is ~1.5 s per image token (≈256 tokens for one frame), so a single image takes ~6–7 minutes before the first reply token. Generation continues at ~2 tok/s.
- **Voice**: similar order of magnitude — audio prompt processing is the bottleneck.

The bot keeps the placeholder message visible during this wait; the user just sees the cursor until generation begins.
