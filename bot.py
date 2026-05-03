#!/usr/bin/env python3
"""Gemma 4 Telegram bot — multimodal chat (text, photo, voice).

Wires python-telegram-bot to llama.cpp; the heavy lifting lives in:
  * llm.py         — llama.cpp client (stream + one-shot + health + bench)
  * media.py       — content-array builders + ffmpeg voice transcoding
  * transcripts.py — per-chat conversation files under logs/convs/
  * sysinfo.py     — host diagnostics for /status

Plain text streams via placeholder edits. Photos and voice notes use the same
streaming path; the first token can take minutes for image/voice on a Pi
because vision/audio prompt processing is the bottleneck — that's fine, the
user-side timeouts are generous.
"""

from __future__ import annotations

import asyncio
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from telegram import BotCommand, Update
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

import llm
import media
import sysinfo
import transcripts

load_dotenv()

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Reply concisely in plain text.",
)
ALLOWED_USER_IDS: set[int] = {
    int(x)
    for x in os.getenv("ALLOWED_USER_IDS", "").replace(",", " ").split()
    if x.strip()
}

CURSOR = "▌"
EDIT_INTERVAL = 2.0
MAX_MSG_LEN = 4000

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            LOGS_DIR / "bot.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)
# httpx/telegram INFO logs leak the bot token in request URLs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

histories: dict[int, list[dict]] = {}
transcript = transcripts.TranscriptStore(LOGS_DIR / "convs")


COMMANDS: list[tuple[str, str]] = [
    ("start", "Start chatting with Gemma"),
    ("help", "Show this help message"),
    ("clear", "Reset the chat history"),
    ("status", "Show host diagnostics and model bench"),
]

HELP_TEXT = (
    "🤖 Gemma 4 multimodal bot\n\n"
    "Send me a text message, a photo, or a voice note and I'll reply.\n\n"
    "Commands:\n"
    + "\n".join(f"/{name} — {desc}" for name, desc in COMMANDS)
    + "\n\nNote: image and voice replies can take several minutes on this "
    "hardware — please be patient."
)


def fresh_history() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def is_allowed(update: Update) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    user = update.effective_user
    if user is None or user.id not in ALLOWED_USER_IDS:
        log.warning(
            "Rejected message from unauthorized user id=%s username=%s",
            getattr(user, "id", None),
            getattr(user, "username", None),
        )
        return False
    return True


def split_point(text: str, max_len: int) -> int:
    """Return an index to split `text` so the head fits within `max_len`."""
    if len(text) <= max_len:
        return len(text)
    floor = max_len // 2
    for sep, off in (("\n\n", 2), ("\n", 1), (". ", 2), ("! ", 2), ("? ", 2), (" ", 1)):
        idx = text.rfind(sep, floor, max_len)
        if idx != -1:
            return idx + off
    return max_len


# --- telegram send helpers --------------------------------------------------


async def safe_edit(message, text: str) -> None:
    try:
        await message.edit_text(text)
    except RetryAfter as e:
        await asyncio.sleep(e.retry_after + 0.5)
        await message.edit_text(text)
    except BadRequest as e:
        if "message is not modified" not in str(e).lower():
            raise


async def safe_send(bot, chat_id: int, text: str):
    try:
        return await bot.send_message(chat_id=chat_id, text=text)
    except RetryAfter as e:
        await asyncio.sleep(e.retry_after + 0.5)
        return await bot.send_message(chat_id=chat_id, text=text)


# --- shared streaming reply core --------------------------------------------


async def stream_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_content: str | list[dict],
    transcript_label: str,
) -> None:
    """Append user content to history, stream Gemma's reply via placeholder edits."""
    chat_id = update.effective_chat.id
    if chat_id not in histories:
        histories[chat_id] = fresh_history()
        transcript.start(chat_id)
    histories[chat_id].append({"role": "user", "content": user_content})
    transcript.append(chat_id, "user", transcript_label)

    current_msg = await update.message.reply_text(CURSOR)
    current_page = ""
    accumulated = ""
    last_edit = asyncio.get_event_loop().time()

    try:
        async for token in llm.stream_chat(histories[chat_id]):
            accumulated += token
            current_page += token

            while len(current_page) > MAX_MSG_LEN:
                idx = split_point(current_page, MAX_MSG_LEN)
                head, current_page = current_page[:idx], current_page[idx:]
                await safe_edit(current_msg, head)
                current_msg = await safe_send(context.bot, chat_id, current_page + CURSOR)
                last_edit = asyncio.get_event_loop().time()

            now = asyncio.get_event_loop().time()
            if now - last_edit >= EDIT_INTERVAL:
                await safe_edit(current_msg, current_page + CURSOR)
                last_edit = now
    except Exception as e:  # noqa: BLE001 — surface to user instead of crashing
        log.error("Streaming error: %s", e)
        transcript.append(chat_id, "error", f"{type(e).__name__}: {e}")
        await safe_edit(current_msg, current_page or f"⚠️ Error: {e}")
        # Drop the trailing user turn so the broken state doesn't poison history.
        histories[chat_id].pop()
        return

    final_text = current_page or "⚠️ No response from model."
    await safe_edit(current_msg, final_text)
    histories[chat_id].append({"role": "assistant", "content": accumulated})
    transcript.append(chat_id, "assistant", accumulated)


# --- command handlers -------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    await update.message.reply_text(HELP_TEXT)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    await update.message.reply_text(HELP_TEXT)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    chat_id = update.effective_chat.id
    histories[chat_id] = fresh_history()
    transcript.start(chat_id)
    await update.message.reply_text("Conversation cleared.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    hardware = sysinfo.read_hardware()
    os_name = sysinfo.read_os_release()
    load1, load5, load15 = sysinfo.read_loadavg()
    temp = sysinfo.read_temperature()
    free, total = sysinfo.read_disk_free("/")
    server = await llm.health()
    mods = await llm.modalities()
    bench_line = await llm.bench()
    await update.message.reply_text(
        "System\n"
        f"  Hardware: {hardware}\n"
        f"  OS: {os_name}\n"
        f"  Load: {load1:.2f} {load5:.2f} {load15:.2f}\n"
        f"  Temp: {temp}\n"
        f"  Disk /: {sysinfo.format_bytes(free)} free / "
        f"{sysinfo.format_bytes(total)}\n"
        "\n"
        "Model\n"
        f"  Endpoint: {llm.LLAMA_URL}\n"
        f"  Server: {server}\n"
        f"  Modalities: {mods}\n"
        f"  Bench: {bench_line}"
    )


# --- message handlers -------------------------------------------------------


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    text = (update.message.text or "").strip()
    if not text:
        return
    await stream_reply(update, context, text, transcript_label=text)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    caption = (update.message.caption or "").strip() or "Describe this image."
    photo = update.message.photo[-1]  # largest size variant
    tg_file = await photo.get_file()
    image_bytes = bytes(await tg_file.download_as_bytearray())
    content = media.user_with_image(caption, image_bytes, mime="image/jpeg")
    label = f"[photo {len(image_bytes)} bytes] {caption}"
    await stream_reply(update, context, content, transcript_label=label)


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    voice = update.message.voice or update.message.audio
    if voice is None:
        return
    tg_file = await voice.get_file()
    raw = bytes(await tg_file.download_as_bytearray())
    try:
        wav = await asyncio.to_thread(media.transcode_to_wav, raw)
    except media.TranscodeError as e:
        await update.message.reply_text(f"⚠️ Couldn't decode voice note: {e}")
        return
    caption = (update.message.caption or "").strip()
    content = media.user_with_audio(caption, wav, fmt="wav")
    label = f"[voice {len(raw)}B → wav {len(wav)}B] {caption or '(no caption)'}"
    await stream_reply(update, context, content, transcript_label=label)


# --- bootstrap --------------------------------------------------------------


async def _post_init(application: Application) -> None:
    await application.bot.set_my_commands(
        [BotCommand(name, desc) for name, desc in COMMANDS]
    )
    log.info("Bot ready; commands registered.")


def main() -> None:
    request = HTTPXRequest(connect_timeout=30, read_timeout=60, write_timeout=60)
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .request(request)
        .post_init(_post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))

    log.info("Bot started. Polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
