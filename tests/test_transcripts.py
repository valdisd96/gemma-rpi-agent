"""Tests for transcripts.TranscriptStore — file rotation and append format."""

from __future__ import annotations

import datetime

from transcripts import TranscriptStore


def test_start_creates_first_file(tmp_path) -> None:
    store = TranscriptStore(tmp_path)
    p = store.start(123)
    assert p.parent == tmp_path / "123"
    assert p.name == "001.txt"
    assert p.exists()


def test_start_rotates_to_next_number(tmp_path) -> None:
    store = TranscriptStore(tmp_path)
    store.start(42)
    store.start(42)
    third = store.start(42)
    assert third.name == "003.txt"
    assert sorted(p.name for p in (tmp_path / "42").glob("*.txt")) == [
        "001.txt",
        "002.txt",
        "003.txt",
    ]


def test_start_ignores_non_numeric_files(tmp_path) -> None:
    chat_dir = tmp_path / "9"
    chat_dir.mkdir()
    (chat_dir / "notes.txt").write_text("ignored")
    store = TranscriptStore(tmp_path)
    p = store.start(9)
    assert p.name == "001.txt"


def test_append_writes_timestamped_turn(tmp_path) -> None:
    fixed = datetime.datetime(2026, 5, 3, 14, 5, 9)
    store = TranscriptStore(tmp_path, now=lambda: fixed)
    store.append(7, "user", "hello there")
    body = (tmp_path / "7" / "001.txt").read_text()
    assert body == "[2026-05-03 14:05:09] user: hello there\n\n"


def test_append_creates_file_on_demand(tmp_path) -> None:
    """First append for a chat must implicitly call start()."""
    store = TranscriptStore(tmp_path)
    store.append(1, "assistant", "hi")
    assert (tmp_path / "1" / "001.txt").exists()


def test_append_uses_active_file_for_chat(tmp_path) -> None:
    store = TranscriptStore(tmp_path)
    store.start(1)
    store.append(1, "user", "first")
    store.start(1)  # rotate to 002.txt
    store.append(1, "user", "second")
    assert (tmp_path / "1" / "001.txt").read_text().strip().endswith("first")
    assert (tmp_path / "1" / "002.txt").read_text().strip().endswith("second")
