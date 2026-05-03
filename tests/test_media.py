"""Tests for media.py — content-array builders + ffmpeg transcode wrapper."""

from __future__ import annotations

import base64
import subprocess
from types import SimpleNamespace

import pytest

import media


def test_encode_data_url_round_trip() -> None:
    raw = b"\xff\xd8\xff\xe0"
    url = media.encode_data_url(raw, "image/jpeg")
    assert url.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == raw


def test_user_with_image_shape() -> None:
    parts = media.user_with_image("describe", b"\x89PNG", mime="image/png")
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_user_with_audio_omits_text_when_empty() -> None:
    parts = media.user_with_audio("", b"RIFF...")
    # No text part when caption is empty — saves tokens on a tight context.
    assert len(parts) == 1
    assert parts[0]["type"] == "input_audio"
    assert parts[0]["input_audio"]["format"] == "wav"
    assert base64.b64decode(parts[0]["input_audio"]["data"]) == b"RIFF..."


def test_user_with_audio_includes_caption() -> None:
    parts = media.user_with_audio("translate this", b"WAV", fmt="wav")
    assert parts[0] == {"type": "text", "text": "translate this"}
    assert parts[1]["type"] == "input_audio"


def test_transcode_to_wav_passes_correct_ffmpeg_args(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input")
        return SimpleNamespace(stdout=b"WAVDATA", stderr=b"")

    monkeypatch.setattr(media.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    runner = SimpleNamespace(run=fake_run)
    out = media.transcode_to_wav(b"OGG-BYTES", runner=runner)
    assert out == b"WAVDATA"
    assert captured["input"] == b"OGG-BYTES"
    assert "-ac" in captured["cmd"] and "1" in captured["cmd"]
    assert "-ar" in captured["cmd"] and "16000" in captured["cmd"]
    assert captured["cmd"][-2:] == ["-f", "wav"] or "wav" in captured["cmd"]


def test_transcode_raises_when_ffmpeg_missing(monkeypatch) -> None:
    monkeypatch.setattr(media.shutil, "which", lambda _: None)
    with pytest.raises(media.TranscodeError, match="ffmpeg not found"):
        media.transcode_to_wav(b"x")


def test_transcode_raises_on_ffmpeg_failure(monkeypatch) -> None:
    monkeypatch.setattr(media.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, output=b"", stderr=b"bad input")

    runner = SimpleNamespace(run=boom)
    with pytest.raises(media.TranscodeError, match="bad input"):
        media.transcode_to_wav(b"x", runner=runner)
