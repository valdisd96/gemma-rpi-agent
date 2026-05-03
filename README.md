# gemma-rpi-agent

A multimodal Telegram bot that streams responses from a local **Gemma 4 E2B-It** model running via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s OpenAI-compatible HTTP server on a Raspberry Pi.

Send the bot **text**, a **photo**, or a **voice note** — replies stream live by editing a placeholder message.

## Requirements

- Raspberry Pi (tested on a Pi 4 with 64-bit kernel)
- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) server running Gemma 4 with **`--mmproj`** loaded so `/props` reports `vision: true, audio: true`
- `ffmpeg` on PATH for voice support (`apt install ffmpeg`)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in TELEGRAM_TOKEN
```

For tests: `pip install -r requirements-dev.txt && python -m pytest -q`.

## Running manually

```bash
source .venv/bin/activate
python bot.py
```

The llama.cpp server must be running before starting the bot.

## Running as a systemd service

```bash
sudo bash install-service.sh
```

This copies `gemma-rpi-agent.service` to `/etc/systemd/system/`, enables it on boot, and starts it.

```bash
systemctl status gemma-rpi-agent
journalctl -u gemma-rpi-agent -f   # live logs
systemctl restart gemma-rpi-agent
```

## Environment variables

| Variable | Required | Default |
|---|---|---|
| `TELEGRAM_TOKEN` | Yes | — |
| `SYSTEM_PROMPT` | No | `"You are a helpful assistant. Reply concisely in plain text."` |
| `ALLOWED_USER_IDS` | No | empty (allow all) |

## Bot commands

| Command | Description |
|---|---|
| `/start` | Show the welcome / help message. |
| `/help` | Same as `/start`. |
| `/clear` | Reset the chat history (LLM memory) and start a new transcript file. |
| `/status` | Show host diagnostics, llama.cpp endpoint/health/modalities, and a short bench. |

Plain text, photos (with optional caption), and voice notes are all answered with streamed replies. Image and voice replies can take several minutes on Pi-class hardware while the multimodal prompt is processed.

## Architecture

See [`CLAUDE.md`](CLAUDE.md) for the full module breakdown. The code is split into `bot.py` (Telegram wiring) plus dedicated modules for `llm` (llama.cpp client), `media` (multimodal content + ffmpeg), `transcripts` (per-chat conversation files), and `sysinfo` (host diagnostics).
